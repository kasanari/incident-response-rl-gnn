# Description

## Dependencies.

Project dependencies are managed by [Poetry](https://python-poetry.org/)
Dependencies can be installed by running `poetry install` in an virtual environment of choice. Poetry will create a venv if not already in one.


PyTorch is not included in the dependency list to allow users to install GPU or CPU variants as they please. 

## How to use

The scripts `train_mlp` and `train_gnn` will respectively train the MLP and GNN models on all the scenarios in the `scenarios` folder. This may take a bit of time. If you want to change settings you will have to go into the scripts, as I have not made a CLI interface.

If you want to evaluate trained models, run `policy_eval` which by default will run two evaluations, one with different GNN layers and one with all models on all scenarios. **This takes a lot of time**, so if you want to run a shorter evaluation, then go into the script and change the values.

The scenario files which define the network structures/graphs are in `scenarios/`. 

## Code Sources

[SR-DRL](https://github.com/jaromiru/sr-drl), GNN RL policy.

[Oracle SAGE](https://github.com/AndrewPaulChester/oracle-sage), SR-DRL code adapted for SB3.

[Stable Baselines 3](https://github.com/DLR-RM/stable-baselines3), PPO implementation.

[CybORG](https://github.com/cage-challenge/CybORG), CybORG environment.