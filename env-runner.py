# Imports
from abc import ABC
import argparse
from pathlib import Path
from tqdm import tqdm
import torch
from utils import MasterConfig
from environments import GymEnvironment
from plugins import DataRecorder
from networks import Cnet
from training_modules import VnetTrainingModule, MnetTrainingModule
from agents import RandomGymAgent, WorldModelGymAgent

# Runners
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
        if config.DATA_COLLECTOR.AGENT == 'random':
            self.agent = RandomGymAgent(self.env, plugins)
        else:
            vnet_encoder = VnetTrainingModule.load_from_checkpoint(
                config.DATA_COLLECTOR.AGENT.VNET_CKPT,
                config=config).net.encoder.cpu().eval()
            mnet = MnetTrainingModule.load_from_checkpoint(
                config.DATA_COLLECTOR.AGENT.MNET_CKPT,
                config=config).net.cpu().eval()
            cnet = Cnet(config)
            st_dict = torch.load(
                config.DATA_COLLECTOR.AGENT.CNET_CKPT)['model_state_dict']
            cnet.load_state_dict(st_dict)
            cnet.eval()
            self.agent = WorldModelGymAgent(
                self.env, vnet_encoder, mnet, cnet)
        self.steps = steps

    def execute(self) -> None:
        for _ in tqdm(range(self.steps),
                      desc='Collecting data', unit='step'):
            self.agent.act()

class AgentTester(Runner):

    def __init__(self, config) -> None:
        super().__init__(config)
    #TODO

def main() -> None:
    config = MasterConfig.from_yaml(args.config)
    if args.collect_data:
        runner = DataCollector(config, args.collection_steps,
                               args.collection_dir,
                               args.collection_name)
    if 'runner' in locals():
        runner.execute()

# CLI
parser = argparse.ArgumentParser(description='Runs environment for collecting data and testing agents.')
parser.add_argument('--config', help='Path to .yaml configuration file.',
                    default='./master-config.yml')
parser.add_argument('--collect-data', action='store_true',
                    help='Flag to run data collection procedure.')
parser.add_argument('--collection-steps',
                    help='Specify # of steps to run data collection for.',
                    type=int,
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


