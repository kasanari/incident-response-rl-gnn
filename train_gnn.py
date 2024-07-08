from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import cyborg_wrappers.graphwrapper as graphwrapper
from gnn.graph_net import MultiMessagePassing
from gnn.graph_net_local import LocalMultiMessagePassing
from gnn.graph_policy import GNNPolicy
import torch
import numpy as np
import random
import gymnasium

try:
    import wandb
    from wandb.integration.sb3 import WandbCallback

    wandb_available = True
except ModuleNotFoundError:
    print("wandb not found, install it to use wandb logging.")
    wandb_available = False

from stable_baselines3.common.evaluation import evaluate_policy


graphwrapper.register_env()

gnn_to_class = {
    "MMP": MultiMessagePassing,
    "LMMP": LocalMultiMessagePassing,
}

algorithm = {
    "ppo": PPO,
}


def train(
    algo,
    gnn_type,
    scenario_file,
    gnn_steps=3,
    max_steps=30,
    embedding_size=32,
    use_wandb=True,
):
    scenario = "2b"
    env_name = f"CyborgEnv{max_steps}-v0"
    n_envs = 16
    encoding = "binary"
    n_multiples = 2
    add_green = False
    config = {
        "n_envs": n_envs,
        "seed": 1,
        "env_name": env_name,
        "load_model": False,
        "model_name": f"{algo}_{gnn_type}",
        "scenario": scenario,
        "encoding": encoding,
    }

    if not wandb_available:
        use_wandb = False

    seed = config["seed"]
    env_kwargs = {
        "scenario_file_path": scenario_file,
        "encoding": encoding,
        "add_green": add_green,
    }
    config.update(env_kwargs)
    env = make_vec_env(
        config["env_name"],
        n_envs=config["n_envs"],
        seed=seed,
        env_kwargs=env_kwargs,
        wrapper_class=gymnasium.wrappers.TimeLimit,
        wrapper_kwargs={"max_episode_steps": max_steps},
    )
    action_mode = "node_then_action"
    algorithm_kwargs = dict(
        n_steps=30 * n_multiples,
        clip_range=0.3,
        learning_rate=0.0003,
        vf_coef=0.0001,
        gae_lambda=0.95,
        n_epochs=30,
        batch_size=30 * n_envs * n_multiples,
        policy_kwargs={
            "activation_fn": torch.nn.Tanh,
            "action_mode": action_mode,
            "gnn_steps": gnn_steps,
            "gnn_class": gnn_to_class[gnn_type],
            "features_extractor_kwargs": {
                "embedding_size": embedding_size,
                "activation_fn": torch.nn.Tanh,
            },
        },
    )

    config.update(algorithm_kwargs)

    if use_wandb:
        run = wandb.init(
            project="sb3_cyborg",
            tags=["remove_penalty", "GNN", "MMP", "No IDs"],
            config=config,
            sync_tensorboard=True,
        )
        callback = WandbCallback(
            model_save_path=f"models/{run.id}",
            model_save_freq=10_000,
            gradient_save_freq=500,
            verbose=2,
        )
    else:
        callback = None

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    trainer_class: PPO = algorithm[algo]
    load = config["load_model"]
    name = config["model_name"]
    if load:
        model = trainer_class.load(name)
        model.env = env
    else:
        model = trainer_class(
            GNNPolicy,
            env,
            verbose=1,
            tensorboard_log="./logs/",
            seed=seed,
            **algorithm_kwargs,
        )

    untrained_path = Path("untrained")
    untrained_path.mkdir(exist_ok=True)
    model.save(untrained_path / f"{gnn_type}{gnn_steps}_untrained")

    model.learn(
        total_timesteps=800_000,
        log_interval=5,
        tb_log_name=name,
        progress_bar=True,
        callback=callback,
    )
    model.save(f"{name}_{action_mode}_cyborg")

    # include model as artifact
    # run.log_artifact(f"models/{run.id}", type="model")

    # evaluate model

    env = make_vec_env(
        config["env_name"],
        n_envs=n_envs,
        env_kwargs=env_kwargs,
        seed=seed,
        wrapper_class=gymnasium.wrappers.TimeLimit,
        wrapper_kwargs={"max_episode_steps": max_steps},
    )
    reward, std = evaluate_policy(
        model,
        env,
        n_eval_episodes=10,
        deterministic=True,
    )

    if use_wandb:
        wandb.log({"eval/reward_avg": reward, "eval/reward_std": std})


if __name__ == "__main__":
    scenario_file_dir = Path("scenarios")
    for model in ["MMP", "LMMP"]:
        for gnn_steps in [2, 3, 4]:
            for scenario_file in scenario_file_dir.glob("*.yaml"):
                train(
                    "ppo",
                    model,
                    scenario_file,
                    gnn_steps=gnn_steps,
                    max_steps=50,
                    embedding_size=128,
                    use_wandb=True,
                )
                if wandb_available:
                    wandb.finish()
