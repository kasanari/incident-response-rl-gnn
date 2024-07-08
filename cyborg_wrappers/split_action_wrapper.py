import gymnasium
import gymnasium.spaces as spaces
import numpy as np
from CybORG.Agents.Wrappers import BlueTableWrapper
from CybORG.Simulator.Actions import InvalidAction, Monitor, Sleep

from cyborg_wrappers.base_wrapper import cyborg_env
from cyborg_wrappers.static import ACTIONS, action_from_index


class SplitActionWrapper(gymnasium.Env):
    def __init__(self) -> None:
        env = cyborg_env(2, "b_line")
        env = BlueTableWrapper(env, output_mode="vector")
        self.env = env

        num_actions = len(ACTIONS)
        num_objects = 13

        self.observation_space = spaces.Box(
            0, 1.0, shape=self.env.reset("Blue").observation.shape, dtype=np.float32
        )

        self.action_space = spaces.MultiDiscrete([num_actions, num_objects])
        self.last_obs = None
        self.idx2id = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.env.set_seed(seed)
        result = self.env.reset(agent="Blue")
        self.idx2id = list(result.action_space["hostname"].keys())
        return result.observation.astype(np.float32), result.info if result.info else {}

    def step(self, action):
        action_action, action_hostname = action

        action_obj = action_from_index(action_action, self.idx2id[action_hostname])

        result = self.env.step(action=action_obj, agent="Blue")

        if isinstance(result.action, InvalidAction):
            raise ValueError("Invalid action")

        terminated = result.done
        truncated = False
        self.last_obs = result

        return (
            result.observation.astype(np.float32),
            result.reward,
            terminated,
            truncated,
            result.info if result.info else {},
        )


if __name__ == "__main__":
    env = SplitActionWrapper()
    obs = env.reset()
    print(obs)
    for _ in range(10):
        action = env.action_space.sample()
        print(action)
        obs, reward, terminated, truncated, info = env.step(action)
        print(obs, reward, terminated, info)
        if terminated:
            break
    env.close()
    print("Done")
