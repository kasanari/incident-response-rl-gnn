import sys
import gnn
from stable_baselines3 import PPO
from pathlib import Path
sys.modules['oracle_sage'] = gnn
sys.modules['oracle_sage.sage.agent'] = gnn

model_dir = Path("trained_agent")
for model_file in model_dir.glob("**/*.zip"):
	model = PPO.load(model_file)
	model.save(model_file.stem)
