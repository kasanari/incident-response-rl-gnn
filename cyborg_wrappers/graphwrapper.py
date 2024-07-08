import inspect
import re
from itertools import chain, combinations

import gymnasium
import gymnasium.spaces as spaces
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from CybORG import CybORG as CybORGEnv
from CybORG.Agents import B_lineAgent, RedMeanderAgent
from CybORG.Agents.Wrappers import BlueTableWrapper, ChallengeWrapper
from CybORG.Simulator import SimulationController
from CybORG.Simulator.Actions.AbstractActions import Impact
from cyborg_wrappers.actions import InvalidAction, Remove, Analyse
from gymnasium.spaces import Box, MultiDiscrete
from gymnasium.utils.env_checker import check_env
from tqdm import tqdm

from cyborg_wrappers.base_wrapper import cyborg_env, get_scenario_path
from cyborg_wrappers.dynamic_graph_wrapper import DynamicGraphWrapper
from cyborg_wrappers.merged_graph_wrapper import MergedGraphWrapper
from cyborg_wrappers.static import (
    ACTIONS,
    ACTIVITY_VOCAB,
    COMPROMISED_VOCAB,
    DECOYS,
    _map_edges,
    action_from_index,
    binary_encoded,
    binary_vec,
    compromise_and_activity_encoded,
    decoy_state,
    get_edges_entity,
    get_reverse_vocab,
    one_hot_encoded,
    possible_actions_encoded,
    possible_objects_encoded,
    scan_state,
)

encoding_length_funcs = {
    "binary": lambda _: (len(ACTIVITY_VOCAB) - 1).bit_length()
    + (len(COMPROMISED_VOCAB) - 1).bit_length(),
    "onehot": lambda x: x + len(ACTIVITY_VOCAB) + len(COMPROMISED_VOCAB),
}


class GraphWrapper(gymnasium.Env):
    scan_history: np.ndarray
    decoy_history: np.ndarray

    @property
    def environment_controller(self) -> SimulationController:
        return self.env.env.env.environment_controller

    @staticmethod
    def get_obs(
        obs,
        encoding_func,
        asset_vocab,
        edges,
        id2idx,
        scan_state,
        decoy_state,
        attacker_start,
        selected_host=None,
    ):
        rows = obs.observation._rows
        routers = filter(lambda x: re.match(".+_router", x), id2idx.keys())
        rows = rows + [[None, None, router, "None", "No"] for router in routers]

        # assets = asset_vec(obs._rows, asset_vocab, len(subnets))

        encoded_scan_history = np.array([encoding_func(x, 3) for x in scan_state], dtype=np.float32)
        id_vec = np.array(
            [encoding_func(i, len(id2idx)) for i, _ in enumerate(id2idx)],
            dtype=np.float32,
        )

        vector_obs = np.concatenate([compromise_and_activity_encoded(rows, encoding_func)])

        # success = obs.success
        # sucess_vec = np.array([map_success(TrinaryEnum.UNKNOWN)] * len(assets), dtype=np.int32)
        # if selected_host is not None:
        #     sucess_vec[selected_host] = map_success(success)

        nodes = np.concatenate([vector_obs], axis=1, dtype=np.float32)

        for action in list(obs.action_space["action"].keys()):
            if action not in ACTIONS:
                del obs.action_space["action"][action]

        possible_objects = possible_objects_encoded(obs.action_space)
        possible_objects[id2idx[attacker_start]] = 0
        possible_actions = possible_actions_encoded(obs.action_space)

        return {
            "nodes": nodes,
            "edges": edges.T,
            "possible_objects": possible_objects,
            "possible_actions": possible_actions,
        }

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 1}

    def dump_dot(self, filename):
        graph = nx.Graph()
        graph.add_edges_from(self.edges)
        nx.drawing.nx_agraph.write_dot(graph, filename)


    def __init__(
        self,
        encoding="binary",
        scenario=None,
        red_agent="meander",
        add_green=False,
        scenario_file_path=None,
        **kwargs,
    ) -> None:
        
        scenario_path = scenario_file_path or get_scenario_path(scenario)

        env = cyborg_env(scenario_path, red_agent, add_green)
        env = BlueTableWrapper(env, output_mode="table")
        self.env = env
        self.edges = None

        self.env.reset(agent="Blue")

        unscaled = not kwargs.get("scale_rewards", False)

        num_nodes = len(self.environment_controller.state.scenario.hosts)
        edges = get_edges_entity(self.environment_controller.state)
        self.attacker_start = self.environment_controller.state.scenario.starting_sessions[2][0].hostname

        num_bits = encoding_length_funcs[encoding](num_nodes)

        subnet_cidr_map = self.environment_controller.subnet_cidr_map
        hostname_ip_map = self.environment_controller.hostname_ip_map
        idx2id = list(hostname_ip_map.keys())
        id2idx = {v: k for k, v in enumerate(idx2id)}

        reward_mappings = (
            {
                "None": 0.0,
                "Low": 0.1,
                "Medium": 1.0,
                "High": 10.0,
            }
            if unscaled
            else {
                "None": 0.0,
                "Low": 0.1,
                "Medium": 1.0,
                "High": 10.0,
            }
        )

        ####
        self.observation_space = spaces.Dict(
            {
                "nodes": Box(low=0, high=1, shape=(num_nodes, num_bits), dtype=np.float32),
                "edges": Box(low=0, high=num_nodes, shape=(2, len(edges)), dtype=np.int32),
                "possible_objects": Box(0, 1, shape=(num_nodes,), dtype=np.int8),
                "possible_actions": Box(0, 1, shape=(len(ACTIONS),), dtype=np.int8),
            }
        )
        self.action_space = MultiDiscrete([len(ACTIONS), num_nodes])
        self.edges_vec = _map_edges(edges, id2idx)
        self.edges = edges
        self.idx2id = idx2id
        self.id2idx = id2idx
        self.asset_vocab = get_reverse_vocab(self.environment_controller.state)
        self.action_history = []
        self.encoding_func = binary_encoded if encoding == "binary" else one_hot_encoded
        self.reward_mappings = reward_mappings
        self.time = 0
        ####

    @property
    def reward_calculator(self):
        return self.environment_controller.state.scenario.team_calc["Blue"][
            "HybridAvailabilityConfidentiality"
        ]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.env.set_seed(seed)
        result = self.env.reset(agent="Blue", seed=seed)
        idx2id = self.idx2id
        id2idx = self.id2idx
        asset_vocab = self.asset_vocab

        calc = self.env.unwrapped.environment_controller.state.scenario.team_calc["Blue"][
            "HybridAvailabilityConfidentiality"
        ]
        calc.availability_calculator.disrupt_rc.mapping = self.reward_mappings
        calc.confidentiality_calculator.infiltrate_rc.mapping = self.reward_mappings

        decoy_indexes = [i for i, a in enumerate(ACTIONS) if a.__name__.startswith("Decoy")]
        scan_history = np.zeros(len(idx2id), dtype=np.int32)
        decoy_history = np.zeros((len(idx2id), len(DECOYS)), dtype=np.int32)
        edges_vec = self.edges_vec
        obs = self.get_obs(
            result,
            self.encoding_func,
            asset_vocab,
            edges_vec,
            id2idx,
            scan_history,
            decoy_history,
            self.attacker_start,
            None,
        )
        available_actions = list(result.action_space["action"].keys())
        info = {"action": available_actions, "hosts": idx2id}

        ####
        self.last_obs = obs
        self.scan_history = scan_history
        self.decoy_history = decoy_history
        self.decoy_indexes = decoy_indexes
        self.action_history = []
        self.time = 0
        ####

        return obs, result.info + info if result.info else info

    def step(self, action):
        action_action, action_hostname = action

        _scan_state = scan_state(self.scan_history, action_action, action_hostname)
        _decoy_state = decoy_state(
            self.decoy_history, self.decoy_indexes, action_action, action_hostname
        )
        action_obj = action_from_index(action_action, self.idx2id[action_hostname])

        result = self.env.step(action=action_obj, agent="Blue")

        if isinstance(result.action, InvalidAction):
            raise ValueError(
                f"Invalid action {ACTIONS[action_action]} on {self.idx2id[action_hostname]}, {result.action}"
            )

        obs = self.get_obs(
            result,
            self.encoding_func,
            None,
            self.edges_vec,
            self.id2idx,
            _scan_state,
            _decoy_state,
            self.attacker_start,
            action_hostname,
        )

        available_actions = list(result.action_space["action"].keys())
        info = {"action": available_actions, "hosts": self.idx2id}
        terminated = result.done
        truncated = False

        ####
        self.last_obs = obs
        self.scan_history = _scan_state
        self.decoy_history = _decoy_state
        self.action_history.append(action)
        self.time += 1
        ####

        return (
            obs,
            result.reward,
            terminated,
            truncated,
            result.info + info if result.info else info,
        )

    def render(self):
        obs = self.last_obs
        step = self.time
        labels = {i: "".join([f"{int(x)}" for x in v]) for i, v in enumerate(obs["nodes"])}

        graph = nx.DiGraph()
        for k, v in labels.items():
            graph.add_node(k, label=v)
        graph.add_edges_from([(f, t) for f, t in obs["edges"].T])
        nx.drawing.nx_agraph.write_dot(graph, f"cyborg_net_{step}.dot")
        pos = nx.nx_agraph.graphviz_layout(graph, prog="sfdp")

        # colors = ["green" if obs["nodes"][x] == 0 else "red" for x in list(graph)]
        fig, ax = plt.subplots()
        nx.draw_networkx(graph, pos=pos, labels=labels, node_size=800, ax=ax)
        plt.axis("off")
        plt.margins(None)
        plt.tight_layout()

        # return as a numpy array for rendering
        return fig, ax


def main():
    register_env()
    max_steps = 100
    env = gymnasium.make(
        f"CyborgEnv{max_steps}-v0",
        encoding="binary",
        scenario="2",
        red_agent="b_line",
        add_green=False,
    )
    # check_env(env, skip_render_check=True)
    obs, info = env.reset(seed=0)

    graph = nx.DiGraph()
    graph.add_edges_from([(f, t) for f, t in obs["edges"].T])

    # action = ACTIONS.index(Analyse)
    # host = 10
    # pass
    # obs, reward, terminated, truncated, info = env.step((action, host))
    # host = 6
    # pass
    # obs, reward, terminated, truncated, info = env.step((action, host))
    # pass
    operations_hacked = False
    _return = 0
    steps = 0
    for _ in tqdm(range(max_steps)):
        # env.render()
        possible_hosts = np.where(obs["possible_objects"] == 1)[0]
        random_host = np.random.choice(possible_hosts)
        random_action = np.random.choice(np.where(obs["possible_actions"] == 1)[0])
        obs, reward, _, _, info = env.step((0, 0))
        attacker_action = env.unwrapped.env.get_last_action("Red")
        if isinstance(attacker_action, Impact) and attacker_action.hostname == "Op_Server0":
            operations_hacked = True
        print(reward)
        _return += reward
        steps += 1

    print(f"Operations hacked: {operations_hacked}")
    print(f"Return: {_return}")
    print(f"Steps: {steps}")


def challenge():
    path = str(inspect.getfile(CybORGEnv))
    path = path[:-10] + "/Shared/Scenarios/Scenario2.yaml"
    env = CybORGEnv(path, "sim", agents={"Red": RedMeanderAgent})
    env = ChallengeWrapper("Blue", env)
    obs = env.reset(agent="Blue")
    env.step(agent="Blue")
    pass


def register_env():
    for l in [30, 50, 100]:
        env_name = f"CyborgEnv{l}-v0"
        gymnasium.register(env_name, entry_point=GraphWrapper, max_episode_steps=l)
        env_name = f"MergedGraphEnv{l}-v0"
        gymnasium.register(env_name, entry_point=MergedGraphWrapper, max_episode_steps=l)
        env_name = f"DynamicGraphEnv{l}-v0"
        gymnasium.register(env_name, entry_point=DynamicGraphWrapper, max_episode_steps=l)
    return env_name


if __name__ == "__main__":
    main()
    # challenge()
