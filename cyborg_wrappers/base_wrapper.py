import inspect
from pathlib import Path

from CybORG import CybORG as CybORGEnv
from CybORG.Agents import B_lineAgent, RedMeanderAgent
from CybORG.Agents import GreenAgent


RED_AGENTS = {
    "meander": RedMeanderAgent,
    "b_line": B_lineAgent,
}

def get_scenario_path(scenario_name):
    path = str(inspect.getfile(CybORGEnv))
    path = path[:-7] + f"/Simulator/Scenarios/scenario_files/Scenario{scenario_name}.yaml"
    return Path(path)

def cyborg_env_v2(scenario, agent, add_green=False):
    path = str(inspect.getfile(CybORGEnv))
    path = path[:-10] + f"/Shared/Scenarios/Scenario{scenario}.yaml"
    agents = {"Red": RED_AGENTS[agent]}
    agents = agents | {"Green": GreenAgent} if add_green else agents
    return CybORGEnv(path, "sim", agents=agents)


def cyborg_env(scenario_file_path, agent, add_green=False):
    from CybORG.Simulator.Scenarios.FileReaderScenarioGenerator import FileReaderScenarioGenerator
    scenario_generator = FileReaderScenarioGenerator(scenario_file_path)
    agents = {"Red": RED_AGENTS[agent]}
    agents = agents | {"Green": GreenAgent} if add_green else agents
    return CybORGEnv(scenario_generator, "sim", agents=agents)
