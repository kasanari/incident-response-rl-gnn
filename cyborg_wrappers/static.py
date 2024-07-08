import enum
import inspect
import json
import re
from functools import cache
from ipaddress import IPv4Address, IPv4Network
from itertools import chain, combinations
from typing import Optional

import gymnasium
import gymnasium.spaces as spaces
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from CybORG import CybORG as CybORGEnv
from CybORG.Agents import B_lineAgent, RedMeanderAgent
from CybORG.Agents.Wrappers import BlueTableWrapper, ChallengeWrapper
from CybORG.Shared.Enums import (
    OperatingSystemDistribution,
    OperatingSystemType,
    OperatingSystemVersion,
    SessionType,
    TrinaryEnum,
)
from CybORG.Simulator import SimulationController
from cyborg_wrappers.actions import (
    Analyse,
    DecoyApache,
    DecoyFemitter,
    DecoyHarakaSMPT,
    DecoySmss,
    DecoySSHD,
    DecoySvchost,
    DecoyTomcat,
    DecoyVsftpd,
    InvalidAction,
    Misinform,
    Monitor,
    Remove,
    Restore,
    Sleep,
)
from CybORG.Simulator.State import State
from gymnasium.spaces import GraphInstance
from gymnasium.utils.env_checker import check_env
from tqdm import tqdm

ACTIONS = [
    Sleep,
    Monitor,
    Analyse,
    Remove,
    Restore,
    # Misinform,
    # DecoyApache,
    # DecoyFemitter,
    # DecoyHarakaSMPT,
    # DecoySmss,
    # DecoySSHD,
    # DecoySvchost,
    # DecoyTomcat,
]

DECOYS = [
    DecoyApache,
    DecoyFemitter,
    DecoyHarakaSMPT,
    DecoySmss,
    DecoySSHD,
    DecoySvchost,
    DecoyTomcat,
    DecoyVsftpd,
]

ACTIVITY_VOCAB = {
    "None": 0,
    "Exploit": 1,
    "Scan": 2,
}

COMPROMISED_VOCAB = {
    "No": 0,
    "Unknown": 1,
    "User": 2,
    "Privileged": 3,
}

SUBNET_EDGES = [
    ("User", "Enterprise"),
    ("Enterprise", "User"),
    ("Enterprise", "Operational"),
    ("Operational", "Enterprise"),
]

ALIASES = {
    "User0": "Attacker",
}


class Encoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, TrinaryEnum):
            return str(obj)
        if isinstance(obj, IPv4Address):
            return str(obj)
        if isinstance(obj, IPv4Network):
            return str(obj)
        if isinstance(obj, SessionType):
            return str(obj)
        if isinstance(obj, SessionType):
            return str(obj)
        if isinstance(obj, OperatingSystemType):
            return str(obj)
        if isinstance(obj, OperatingSystemDistribution):
            return str(obj)
        if isinstance(obj, OperatingSystemVersion):
            return str(obj)
        if isinstance(obj, enum.Enum):
            return str(obj)

        return json.JSONEncoder.default(self, obj)


scan_index = ACTIONS.index(Analyse)
restore_index = ACTIONS.index(Restore)
pattern1 = re.compile(r"([aA-zZ]+)(\d*)")


def action_from_index(action_idx: int, host: str):
    action = ACTIONS[action_idx]
    if Sleep in ACTIONS and action_idx == ACTIONS.index(Sleep):
        return action()
    elif Monitor in ACTIONS and action_idx == ACTIONS.index(Monitor):
        return action(session=0, agent="Blue")
    else:
        return action(session=0, agent="Blue", hostname=host)


def map_activity(activity) -> Optional[int]:
    return ACTIVITY_VOCAB[activity]


def map_compromised(compromised) -> Optional[int]:
    return COMPROMISED_VOCAB[compromised]


def scan_state(scan_history, action, hostname):
    def new_value(action):
        if action == scan_index:
            return 2
        elif action == restore_index:
            return 0
        else:
            return 1

    scan_state = [1 if x == 2 or x == 1 else 0 for x in scan_history]
    scan_state[hostname] = new_value(action)
    return scan_state


def decoy_state(decoy_history, decoy_indexes, action, hostname):
    decoy_state = decoy_history.copy()
    if action in decoy_indexes:
        decoy_state[hostname][DECOYS.index(ACTIONS[action])] = 1

    if action == restore_index:
        decoy_state[hostname] = np.zeros(len(DECOYS), dtype=np.int8)

    return decoy_state


def plot_graph(graph: GraphInstance):
    g = nx.DiGraph()
    for edge in graph.edge_links:
        g.add_edge(edge[0], edge[1])

    labels = {i: f"{n}" for i, n in enumerate(graph.nodes)}

    pos = nx.nx_agraph.graphviz_layout(g, prog="sfdp")
    nx.draw_networkx(g, pos=pos, labels=labels, with_labels=True)
    plt.show()


def split_id(host):
    re_match = pattern1.match(host)
    return re_match.group(1), re_match.group(2)


def asset(host):
    return split_id(host)[0]


# subnet_names = {
#     "User": '10.0.23.48/28',
#     "Enterprise": '10.0.7.64/28',
#     "Operations": '10.0.1.192/28',
# }

# subnet_edges = [
#     (subnet_names["User"], subnet_names["Enterprise"]),
#     (subnet_names["Enterprise"], subnet_names["User"]),
#     (subnet_names["Enterprise"], subnet_names["Operations"]),
#     (subnet_names["Operations"], subnet_names["Enterprise"]),
# ]


def are_subnets_connected(subnet1, subnet2):
    return (subnet1, subnet2) in SUBNET_EDGES


def get_edges_network(state: State):
    """Connect hosts with direct edges"""

    scenario = state.scenario

    edges = []

    host_to_subnet = {}

    def are_subnets_connected(subnet1, subnet2):
        return (subnet1, subnet2) in SUBNET_EDGES

    def are_hosts_connected(host1, host2):
        return are_subnets_connected(host_to_subnet[host1], host_to_subnet[host2])

    subnets = scenario._scenario["Subnets"]
    for subnet in subnets:
        hosts = subnets[subnet]["Hosts"]
        for host in combinations(hosts, 2):
            # all hosts in a subnet are connected
            edges.extend([(host[0], host[1]), (host[1], host[0])])
            host_to_subnet[host[0]] = subnet

    for host in combinations(host_to_subnet, 2):
        if are_hosts_connected(host[0], host[1]):
            edges.extend([(host[0], host[1]), (host[1], host[0])])

    pass

    graph = nx.DiGraph()

    for edge in edges:
        graph.add_edge(edge[0], edge[1])

    # pos = nx.nx_agraph.graphviz_layout(graph, prog='sfdp')
    # nx.draw_networkx(graph, pos=pos)
    # plt.show()

    # for node in nodes:
    #     graph.add_node(node["id"], **node)

    # for edge in edges:
    #     graph.add_edge(edge["from"], edge["to"])

    # pos = nx.nx_agraph.graphviz_layout(graph, prog='sfdp')
    # nx.draw_networkx(graph, pos=pos)
    # plt.show()
    return edges


def get_edges_entity(state: State):
    """Connect hosts with subnet nodes as intermediaries"""

    graph = state.link_diagram
    digraph = graph.to_directed()

    edges = list(digraph.edges)

    edges.remove(('Operational_router', 'User_router'))
    edges.remove(('User_router', 'Operational_router'))

    return np.array(edges)


def asset_vocab(obs: State):
    components = obs.connected_components[0]

    hosts = list(filter(lambda x: not re.match(".+_router", x), components))
    assets, _ = zip(*(split_id(host) for host in hosts))
    
    vocab = set(assets)
    
    routers = set(filter(lambda x: re.match(".+_router", x), components))
    vocab = vocab.union(routers)
    return sorted(vocab)


def get_reverse_vocab(obs: dict):
    vocab = asset_vocab(obs)
    return {asset: i for i, asset in enumerate(vocab)}


def alias(x):
    return ALIASES.get(x, x)


def id_mappings(rows, add_subnets=False) -> list:
    hosts = list(map(lambda x: alias(x[2]), rows))
    subnets = sorted(set(map(lambda x: alias(x[0]), rows)))
    id_mappings = hosts + subnets if add_subnets else hosts
    return id_mappings


def asset_vec(rows, asset_vocab, num_subnets):
    return np.array(
        list(
            map(
                lambda x: one_hot_encoded(asset_vocab[x], len(asset_vocab)),
                chain(
                    map(lambda x: asset(alias(x[2])), rows),
                    ["Subnet"] * num_subnets,
                ),
            ),
        ),
        dtype=np.float32,
    )


def compromise_and_activity_encoded(rows, encoding_func):
    vector = [
        np.concatenate(
            [
                encoding_func(ACTIVITY_VOCAB[row[3]], len(ACTIVITY_VOCAB)),
                encoding_func(COMPROMISED_VOCAB[row[4]], len(COMPROMISED_VOCAB)),
            ]
        )
        for row in rows
    ]
    return np.array(vector)


def map_success(success: TrinaryEnum):
    vocab = {"TRUE": 1, "FALSE": 0, "UNKNOWN": 2}
    return vocab[success.name]


def _map_edges(edges, id2idx):
    return np.stack(
        [np.array([id2idx[edge[0]], id2idx[edge[1]]], dtype=np.int32) for edge in edges]
    )


def possible_actions_encoded(action_space_dict: dict):
    valid_actions = [valid for _, valid in action_space_dict["action"].items()]
    return np.array(valid_actions, dtype=np.int8)


def possible_objects_encoded(action_space_dict: dict):
    valid_actions = [valid for _, valid in action_space_dict["hostname"].items()]
    return np.array(valid_actions, dtype=np.int8)


@cache
def binary_encoded(val, num_vals):
    num_bits = np.ceil(np.log2(num_vals)).astype(np.int32)
    return np.array([float(x) for x in np.binary_repr(val, num_bits)], dtype=np.float32)


def binary_vec(vals, num_vals):
    return np.array([binary_encoded(x, num_vals) for x in vals], dtype=np.float32)


@cache
def one_hot_encoded(val, num_vals):
    vec1 = np.zeros(num_vals)
    vec1[val] = 1
    return vec1
