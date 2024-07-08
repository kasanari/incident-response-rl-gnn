from typing import Union, Tuple, Dict
import gymnasium.spaces as spaces
import torch as th
import torch_geometric as thg

# preprocessing.py
def get_obs_shape(
    observation_space: spaces.Space,
) -> Union[Tuple[int, ...], Dict[str, Tuple[int, ...]]]:
    """
    Get the shape of the observation (useful for the buffers).

    :param observation_space:
    :return:
    """
    if isinstance(observation_space, spaces.Box):
        return observation_space.shape
    elif isinstance(observation_space, spaces.Discrete):
        # Observation is an int
        return (1,)
    elif isinstance(observation_space, spaces.MultiDiscrete):
        # Number of discrete features
        return (int(len(observation_space.nvec)),)
    elif isinstance(observation_space, spaces.MultiBinary):
        # Number of binary features
        return observation_space.shape
    elif isinstance(observation_space, spaces.Dict):
        return {key: get_obs_shape(subspace) for (key, subspace) in observation_space.spaces.items()}  # type: ignore[misc]
    elif isinstance(observation_space, spaces.Graph):
        return (get_obs_shape(observation_space.node_space), 2, get_obs_shape(observation_space.edge_space))
    elif isinstance(observation_space, spaces.Sequence):
        return get_obs_shape(observation_space.feature_space)
    else:
        raise NotImplementedError(f"{observation_space} observation space is not supported")


#utils.py
def obs_to_tensor(self, observation: spaces.GraphInstance):
    if isinstance(observation, list):
        vectorized_env = True
    else:
        vectorized_env = False
    if vectorized_env:
        torch_obs = list()
        for obs in observation:
            x = th.tensor(obs.nodes).float()
            edge_index = th.tensor(obs.edge_links, dtype=th.long).t().contiguous().view(2, -1)
            torch_obs.append(thg.data.Data(x=x, edge_index=edge_index))
        if len(torch_obs) == 1:
            torch_obs = torch_obs[0]
    else:
        x = th.tensor(observation.nodes).float()
        edge_index = th.tensor(observation.edge_links, dtype=th.long).t().contiguous().view(2, -1)
        torch_obs = thg.data.Data(x=x, edge_index=edge_index)
    return torch_obs, vectorized_env


def graphinstancetodata(g: spaces.GraphInstance):
        x = th.tensor(g.nodes).float()
        edge_index = th.tensor(g.edge_links, dtype=th.long).t().contiguous().view(2, -1)
        return thg.data.Data(x=x, edge_index=edge_index)