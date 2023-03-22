# Imports
from abc import ABC
import argparse
from pathlib import Path
from tqdm import tqdm
from utils import MasterConfig
from environments import GymEnvironment
from plugins import DataRecorder
from agents import RandomGymAgent


class Runner(ABC):

    def __init__(self, config: MasterConfig) -> None:
        super().__init__()
        self.config = config


class DataCollector(Runner):

    def __init__(self, config: MasterConfig, steps: int, 
                 data_dir: str, name: str) -> None:
        super().__init__(config)
        data_dir = Path(data_dir)
        data_recorder = DataRecorder(data_dir, name, as_hdf5=True)
        plugins = [data_recorder]
        self.env = GymEnvironment(config, plugins)
        self.agent = RandomGymAgent(self.env, plugins)
        self.steps = steps

    def execute(self) -> None:
        for _ in tqdm(range(self.steps),
                      desc='Collecting data', unit='step'):
            self.agent.act()


def main() -> None:
    config = MasterConfig.from_yaml(args.config)
    if args.collect_data:
        runner = DataCollector(config, args.collection_steps,
                               args.collection_dir,
                               args.collection_name)
    if 'runner' in locals():
        runner.execute()

# CLI
parser = argparse.ArgumentParser(description='Main interface to project.')
parser.add_argument('--config', help='Path to .yaml configuration file.',
                    default='./config.yml')
parser.add_argument('--collect-data', action='store_true',
                    help='Flag to run data collection procedure.')
parser.add_argument('--collection-steps',
                    help='Specify # of steps to run data collection for.',
                    default=2000)
parser.add_argument('--collection-name',
                    help='Specify name for data collection.',
                    default='data')
parser.add_argument('--collection-dir',
                    help='Specify directory for data collection.',
                    default='data')
args = parser.parse_args()

if __name__ == '__main__':
    main()


