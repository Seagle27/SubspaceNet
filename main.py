from run_simulation import SimulationRunner
from src.config.load_config import load_simulation_config

if __name__ == "__main__":
    config = load_simulation_config("src/config/sparseNet.yaml")
    SimulationRunner(config).run()
