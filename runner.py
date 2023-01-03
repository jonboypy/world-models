# Imports
from abc import ABC
import argparse
from pathlib import Path
from utils import MasterConfig
from environments import GymEnvironment
from plugins import DataRecorder
from agents import RandomGymAgent


class Runner(ABC):

    def __init__(self, config: MasterConfig) -> None:
        super().__init__()
        self.config = config


class DataCollector(Runner):

    def __init__(self, config: MasterConfig, steps: int) -> None:
        super().__init__(config)
        data_dir = Path('./data')
        data_recorder = DataRecorder(data_dir)
        plugins = [data_recorder]
        self.env = GymEnvironment(config, plugins)
        self.agent = RandomGymAgent(self.env, plugins)
        self.steps = steps

    def execute(self) -> None:
        for _ in range(self.steps):
            self.agent.act()


def main() -> None:
    config = MasterConfig.from_yaml(args.config)
    if config.PROCEDURE == 'data-collection':
        runner = DataCollector(config, 2000)
    runner.execute()


# CLI
parser = argparse.ArgumentParser(description='Main interface to project.')
parser.add_argument('--config', help='Path to .yaml configuration file.', default='./config.yml')
args = parser.parse_args()

if __name__ == '__main__':
    main()


