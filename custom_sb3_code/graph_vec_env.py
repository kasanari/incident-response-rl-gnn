import warnings
from collections import OrderedDict
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type

import gymnasium as gym
import gymnasium.spaces as spaces
from gymnasium.spaces import GraphInstance
import numpy as np

from stable_baselines3.common.vec_env.base_vec_env import (
    VecEnv,
    VecEnvIndices,
    VecEnvObs,
    VecEnvStepReturn,
)
from stable_baselines3.common.vec_env.patch_gym import _patch_env
from stable_baselines3.common.vec_env.util import dict_to_obs
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from torch_geometric.data import Data

def obs_space_info(obs_space: spaces.Space) -> Tuple[List[str], Dict[Any, Tuple[int, ...]], Dict[Any, np.dtype]]:
    if isinstance(obs_space, spaces.Dict):
        assert isinstance(obs_space.spaces, OrderedDict), "Dict space must have ordered subspaces"
        subspaces = obs_space.spaces
        return {k: obs_space_info(v) for k, v in subspaces.items()}
    elif isinstance(obs_space, spaces.Tuple):
        subspaces = {i: space for i, space in enumerate(obs_space.spaces)}  # type: ignore[assignment]
    elif isinstance(obs_space, spaces.Graph):
        subspaces = {'nodes': obs_space.node_space, 'edge_attrs': obs_space.edge_space}
    elif isinstance(obs_space, spaces.Sequence):
        subspaces = {None: obs_space.feature_space}
    else:
        assert not hasattr(obs_space, "spaces"), f"Unsupported structured space '{type(obs_space)}'"
        subspaces = {None: obs_space}  # type: ignore[assignment]
    
    keys = []
    shapes = {}
    dtypes = {}
    for key, box in subspaces.items():
        keys.append(key)
        shapes[key] = box.shape
        dtypes[key] = box.dtype
    return {None: (keys, shapes, dtypes)}


class GraphVecEnv(DummyVecEnv):
    def __init__(self, env_fns: List[Callable[[], gym.Env]]):
        self.envs = [_patch_env(fn()) for fn in env_fns]
        if len(set([id(env.unwrapped) for env in self.envs])) != len(self.envs):
            raise ValueError(
                "You tried to create multiple environments, but the function to create them returned the same instance "
                "instead of creating different objects. "
                "You are probably using `make_vec_env(lambda: env)` or `DummyVecEnv([lambda: env] * n_envs)`. "
                "You should replace `lambda: env` by a `make_env` function that "
                "creates a new instance of the environment at every call "
                "(using `gym.make()` for instance). You can take a look at the documentation for an example. "
                "Please read https://github.com/DLR-RM/stable-baselines3/issues/1151 for more information."
            )
        env = self.envs[0]
        super(DummyVecEnv, self).__init__(len(env_fns), env.observation_space, env.action_space)
        # obs_space = env.observation_space
        # node_space = obs_space.node_space
        # edge_space = obs_space.edge_space
        # self.keys = ["nodes", "edge_indices", "edge_attrs"]
        # dtypes = {"nodes": node_space.dtype, "edge_indices": edge_space.dtype, "edge_attrs": edge_space.dtype}
        # shapes = {"nodes": node_space.shape, "edge_indices": 2, "edge_attrs": edge_space.shape}

        self.keys = [
            "graph",
            "possible_objects",
            "possible_actions",
        ]

        self.buf_obs = {k:[None] * self.num_envs for k in self.keys}
        self.buf_dones = np.zeros((self.num_envs,), dtype=bool)
        self.buf_rews = np.zeros((self.num_envs,), dtype=np.float32)
        self.buf_infos: List[Dict[str, Any]] = [{} for _ in range(self.num_envs)]
        self.metadata = env.metadata

    def _save_obs(self, env_idx: int, obs: GraphInstance) -> None:
        for k, v in obs.items():
            self.buf_obs[k][env_idx] = v
        #self.buf_obs[env_idx] = obs

    def _obs_from_buf(self) -> VecEnvObs:
        return copy_obs_dict(self.buf_obs)


def copy_graph(g):
    return GraphInstance(
        nodes=g.nodes.copy(),
        edges=g.edges.copy() if g.edges is not None else None,
        edge_links=g.edge_links.copy(),
    )


def copy_obs_dict(obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    if isinstance(obs, dict):
        return {
            "graph": [copy_graph(g) for g in obs["graph"]],
            "possible_objects": [o.copy() for o in obs["possible_objects"]],
            "possible_actions": np.stack(obs["possible_actions"]).copy(),
        }
    else:
        return [copy_graph(g) for g in obs]
