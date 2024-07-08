from cyborg_mlp_agent import MLPPolicy
from cyborg_wrappers.base_wrapper import cyborg_env
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

import gymnasium

import torch
try:
    import wandb
    from wandb.integration.sb3 import WandbCallback
    wandb_available = True
except ModuleNotFoundError:
    print("wandb not found, install it to use wandb logging.")
    wandb_available = False
from stable_baselines3.common.evaluation import evaluate_policy
from CybORG.Agents.Wrappers import ChallengeWrapper
import gymnasium.wrappers

from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv


def make_env(agent, scenario, max_steps, add_green):
    env = cyborg_env(scenario, agent, add_green)
    env = ChallengeWrapper("Blue", env)
    env = gymnasium.wrappers.EnvCompatibility(env)
    env = gymnasium.wrappers.TimeLimit(env, max_episode_steps=max_steps)
    return env


def main(scenario):
    policy_kwargs = {
        "activation_fn": torch.nn.Tanh,
        "share_features_extractor": False,
        "net_arch": dict(pi=[256, 256], vf=[256, 256]),
    }
    n_envs = 16
    max_steps = 50
    n_multiples = 4
    seed = 1
    add_green = True

    algorithm_kwargs = {
        "seed": seed,
        "clip_range": 0.3,
        "vf_coef": 0.0001,
        "n_epochs": 30,
        "batch_size": 30 * n_envs * n_multiples,
        "n_steps": 30 * n_multiples,
        "clip_range_vf": None,
    }

    config = {
        "agent": "meander",
        "max_steps": max_steps,
        "algorithm_kwargs": algorithm_kwargs,
        "policy_kwargs": policy_kwargs,
        "n_envs": n_envs,
    }

    agent = config["agent"]
    max_steps = config["max_steps"]
    env_kwargs = {
        "scenario": scenario,
        "add_green": add_green,
        "agent": agent,
        "max_steps": max_steps,
    }
    env = make_vec_env(
        make_env,
        n_envs=n_envs,
        seed=seed,
        env_kwargs=env_kwargs,
        vec_env_cls=SubprocVecEnv,
    )

    config.update(env_kwargs)

    use_wandb = False and wandb_available
    if use_wandb:
        run = wandb.init(
            project="sb3_cyborg",
            tags=["MLP"],
            config=config,
            sync_tensorboard=True,
        )
        callback = WandbCallback(
            model_save_path=f"models/{run.id}",
            model_save_freq=100,
            gradient_save_freq=50,
            verbose=2,
        )
    else:
        callback = None

    model = PPO(
        MLPPolicy,
        env,
        verbose=1,
        tensorboard_log="./logs/",
        **algorithm_kwargs,
        policy_kwargs=policy_kwargs,
    )

    model.learn(
        total_timesteps=800_000, log_interval=5, progress_bar=True, callback=callback
    )

    env = make_vec_env(
        make_env,
        n_envs=1,
        seed=seed,
        env_kwargs=env_kwargs,
    )

    reward, std = evaluate_policy(model, env, n_eval_episodes=10, render=False)

    if use_wandb:
        wandb.log({"eval/reward_avg": reward, "eval/reward_std": std})

if __name__ == "__main__":
    from pathlib import Path

    scenario_folder = Path("scenarios")
    for s in scenario_folder.iterdir():
        print(s)
        main(s)
        if wandb_available:
            wandb.finish()
