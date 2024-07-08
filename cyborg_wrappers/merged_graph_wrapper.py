import inspect
import re
from itertools import chain

import gymnasium
import gymnasium.spaces as spaces
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from CybORG import CybORG as CybORGEnv
from CybORG.Agents import B_lineAgent, RedMeanderAgent
from CybORG.Agents.Wrappers import BlueTableWrapper

from cyborg_wrappers.actions import Analyse, InvalidAction, Remove
from gymnasium.spaces import Box, MultiDiscrete
from gymnasium.utils.env_checker import check_env
from tqdm import tqdm

from cyborg_wrappers.base_wrapper import cyborg_env
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


class MergedGraphWrapper(gymnasium.Env):
    def __init__(self) -> None:
        path = str(inspect.getfile(CybORGEnv))
        path = path[:-10] + "/Shared/Scenarios/Scenario2.yaml"

        env = CybORGEnv(path, "sim", agents={"Red": RedMeanderAgent})
        env = BlueTableWrapper(env, output_mode="table")

        num_actions = len(ACTIONS)

        env.reset(agent="Blue")
        raw_obs = env.get_agent_state("Blue")
        raw_obs = {alias(k): v for k, v in raw_obs.items()}
        edges = get_edges(raw_obs)
        num_edges = len(edges)
        num_nodes = 16
        decoy_indexes = [i for i, a in enumerate(ACTIONS) if a.__name__.startswith("Decoy")]
        num_decoys = len(decoy_indexes)

        self.observation_space = spaces.Dict(
            {
                "nodes": spaces.Box(low=0, high=1, shape=(num_nodes, 28), dtype=np.float32),
                "edges": spaces.Box(low=0, high=num_nodes, shape=(num_edges, 2), dtype=np.int32),
                "possible_actions": spaces.Box(0, 1, shape=(num_actions,), dtype=np.int8),
                "possible_objects": spaces.Box(0, 1, shape=(num_nodes,), dtype=np.int8),
            }
        )
        self.env = env
        self.action_space = spaces.MultiDiscrete([num_actions, num_nodes])
        self.edges = edges
        self.num_subnets = 3
        self.last_obs = None
        self.edges_vec = None
        self.decoy_indexes = decoy_indexes
        self.scan_history = np.zeros(num_nodes * 2, dtype=np.int32)

    @classmethod
    def get_obs(
        cls,
        result,
        asset_vocab,
        edges_vec,
        num_subnets,
        scan_history,
        decoy_history,
        id2idx,
        selected_host=None,
    ):
        rows = result.observation._rows
        assets = asset_vec(rows, asset_vocab, num_subnets)
        num_bits_per_asset = 7
        success = result.observation.success
        success_vec = np.array(
            np.repeat(one_hot_encoded(map_success(success), 3), len(assets)).reshape(16, 3),
            dtype=np.int32,
        )
        if selected_host is not None:
            success_vec[selected_host] = one_hot_encoded(map_success(success), 3)
        encoded_scan_history = np.array(
            [one_hot_encoded(x, 3) for x in scan_history], dtype=np.float32
        )
        vector_obs = np.concatenate(
            [
                compromise_and_activity_encoded(rows, one_hot_encoded),
                np.zeros((num_subnets, num_bits_per_asset)),
            ]
        )
        per_node_obs = np.array(
            [
                np.concatenate([o, a, s, succ, dec])
                for o, a, s, succ, dec in zip(
                    vector_obs, assets, encoded_scan_history, success_vec, decoy_history
                )
            ],
            dtype=np.float32,
        )
        possible_objects = possible_objects_encoded(result.action_space)
        possible_objects[id2idx["Attacker"]] = 0
        possible_actions = possible_actions_encoded(result.action_space)
        for i in range(len(possible_actions)):
            if i in cls.decoy_indexes:
                possible_actions[i] = 0
        obs = {
            "nodes": per_node_obs,
            "edges": edges_vec,
            "possible_actions": possible_actions,
            "possible_objects": np.concatenate(
                [possible_objects, np.zeros(num_subnets, dtype=np.int8)], dtype=np.int8
            ),
        }
        return obs

    def reset(self, seed=None, options=None):
        result = self.env.reset(agent="Blue")
        raw_obs = self.env.env.get_agent_state("Blue")
        raw_obs = {alias(k): v for k, v in raw_obs.items()}
        asset_vocab = get_reverse_vocab(raw_obs)
        edges = get_edges(raw_obs)
        idx2id = id_mappings(result.observation._rows)
        id2idx = {v: k for k, v in enumerate(idx2id)}
        edges_vec = _map_edges(edges, id2idx)
        scan_history = np.zeros(len(idx2id), dtype=np.int32)
        decoy_history = np.zeros((len(idx2id), len(DECOYS)), dtype=np.int32)

        obs = self.get_obs(
            result,
            asset_vocab,
            edges_vec,
            self.num_subnets,
            scan_history,
            decoy_history,
            id2idx,
            None,
        )

        self.idx2id = idx2id
        self.id2idx = id2idx
        self.edges = edges
        self.edges_vec = edges_vec
        self.asset_vocab = asset_vocab
        self.last_obs = obs
        self.scan_history = scan_history
        self.decoy_history = decoy_history

        return obs, result.info if result.info else {}

    def step(self, action):
        action_action, action_hostname = action
        action_obj = create_action(action_action, self.idx2id[action_hostname])
        decoy_history = self.decoy_history.copy()

        scan_history = scan_state(self.scan_history, action_action, action_hostname)

        decoy_history = decoy_state(
            decoy_history, self.decoy_indexes, action_action, action_hostname
        )

        result = self.env.step(action=action_obj, agent="Blue")

        if isinstance(result.action, InvalidAction):
            raise ValueError("Invalid action")

        obs = self.get_obs(
            result,
            self.asset_vocab,
            self.edges_vec,
            self.num_subnets,
            scan_history,
            decoy_history,
            self.id2idx,
            action_hostname,
        )

        terminated = result.done
        truncated = False
        self.last_obs = obs
        self.scan_history = scan_history
        self.decoy_history = decoy_history

        return obs, result.reward, terminated, truncated, result.info if result.info else {}