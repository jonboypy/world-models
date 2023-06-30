# Imports
from abc import ABC, abstractmethod
from typing import List
import argparse
from pathlib import Path
import shutil
from tqdm import tqdm
import h5py
import glob
import torch
import ray
import wandb
from utils import MasterConfig
from environments import GymEnvironment, Environment
from plugins import DataRecorder, Plugin
from networks import Cnet
from training_modules import VnetTrainingModule, MnetTrainingModule
from agents import RandomGymAgent, WorldModelGymAgent, Agent
from trainers import LitTrainer, EvolutionTrainer

#####################################################################
#   Runners
#####################################################################

class Runner(ABC):

    def __init__(self, cfg: MasterConfig) -> None:
        super().__init__()
        self.cfg = cfg

    @abstractmethod
    def execute(self) -> None:
        raise NotImplementedError

class NetworkTrainer(Runner):

    def __init__(self, cfg: MasterConfig) -> None:
        super().__init__(cfg)
        if cfg.RUN_TYPE in ['V-NET', 'M-NET']:
            self.trainer = LitTrainer(cfg)
        elif cfg.RUN_TYPE == 'C-NET':
            self.trainer = EvolutionTrainer(cfg)
        else:
            raise SyntaxError(f'Run type {cfg.RUN_TYPE} not supported.')
        self.trainer.setup()

    def execute(self) -> None:
        self.trainer.train()
        
class DataCollector(Runner):

    def __init__(self, cfg: MasterConfig) -> None:
        super().__init__(cfg)
        if (Path(self.cfg.SAVE_DIR)/
            f'{self.cfg.DATA_NAME}.hdf5').exists():
            (Path(self.cfg.SAVE_DIR)/
             f'{self.cfg.DATA_NAME}.hdf5').unlink()

    def execute(self) -> None:
        # Create tmp directory to store episode results
        if (Path(self.cfg.SAVE_DIR)/'tmp').exists():
            shutil.rmtree(self.cfg.SAVE_DIR+'/tmp/')
        (Path(self.cfg.SAVE_DIR)/'tmp').mkdir(parents=True)
        # Launch episodes in parallel via Ray
        eps_ids = []
        datacollector = ray.put(self)
        for eps_idx in range(self.cfg.N_EPISODES):
            eps_ids += [
                self._execute_signal_agent.remote(
                datacollector, eps_idx)]
        ray.get(eps_ids)
        # Combine episode hdf5's into 1 file.
        self._combine_hdf5()
        # Remove tmp directory.
        shutil.rmtree(self.cfg.SAVE_DIR+'/tmp/')

    def _combine_hdf5(self) -> None:
        with h5py.File(Path(self.cfg.SAVE_DIR)/
                       f'{self.cfg.DATA_NAME}.hdf5',mode='w') as h5fw:
            for h5name in tqdm(
                    glob.glob(f'{self.cfg.SAVE_DIR}/tmp/*.hdf5'),
                    desc='Combining episode files', unit='episode'):
                h5fr = h5py.File(h5name,'r') 
                for obj in h5fr.keys():        
                    h5fr.copy(obj, h5fw)
                h5fr.close()
                Path(h5name).unlink()
        
    @ray.remote
    def _execute_signal_agent(self, eps_idx: int) -> None:
        plugins = [DataRecorder(
                    Path(self.cfg.SAVE_DIR)/'tmp',
                    self.cfg.DATA_NAME, eps_idx)]
        env = GymEnvironment(self.cfg, plugins)
        agent = self._load_agent(env, plugins)
        done = False
        while not done:
            done = agent.act()

    def _load_agent(self, env: Environment,
                    plugins: List[Plugin]) -> Agent:
        if self.cfg.AGENT == 'random':
            agent = RandomGymAgent(env, plugins)
        else:
            vnet_encoder = VnetTrainingModule.load_from_checkpoint(
                self.cfg.AGENT.VNET_CKPT,
                cfg=self.cfg, map_location='cpu').net.encoder.eval()
            mnet = MnetTrainingModule.load_from_checkpoint(
                self.cfg.AGENT.MNET_CKPT,
                cfg=self.cfg, map_location='cpu').net.eval()
            cnet = Cnet(self.cfg)
            st_dict = torch.load(
                self.cfg.AGENT.CNET_CKPT)['model_state_dict']
            cnet.load_state_dict(st_dict)
            cnet = cnet.cpu().eval()
            agent = WorldModelGymAgent(
                env, vnet_encoder, mnet, cnet, plugins)
        return agent

class AgentTester(Runner):

    def __init__(self, config) -> None:
        super().__init__(config)
    #TODO

# Setup functions
def setup_data_collector() -> DataCollector:
    cfg = MasterConfig.from_yaml(
        args.config, 'DATA_COLLECTION')
    cfg.DEBUG = args.debug
    ray.init(local_mode=cfg.DEBUG)
    runner = DataCollector(cfg)
    return runner

def setup_vnet_trainer() -> NetworkTrainer:
    cfg = MasterConfig.from_yaml(
        args.config, 'V-NET')
    cfg.DEBUG = args.debug
    # Create experiment directory
    if (not cfg.DEBUG and not Path(
            cfg.EXPERIMENT_DIR).exists()):
        Path(cfg.EXPERIMENT_DIR).mkdir(parents=True)
    # login to WandB
    if not cfg.DEBUG:
        wandb.login(key=cfg.WANDB_KEY)
        del cfg.WANDB_KEY
    runner = NetworkTrainer(cfg)
    return runner

def setup_mnet_trainer() -> NetworkTrainer:
    cfg = MasterConfig.from_yaml(
        args.config, 'M-NET')
    cfg.DEBUG = args.debug
    # Create experiment directory
    if (not cfg.DEBUG and not Path(
            cfg.EXPERIMENT_DIR).exists()):
        Path(cfg.EXPERIMENT_DIR).mkdir(parents=True)
    # login to WandB
    if not cfg.DEBUG:
        wandb.login(key=cfg.WANDB_KEY)
        del cfg.WANDB_KEY
    runner = NetworkTrainer(cfg)
    return runner

def setup_cnet_trainer() -> NetworkTrainer:
    cfg = MasterConfig.from_yaml(
        args.config, 'C-NET')
    cfg.DEBUG = args.debug
    ray.init(local_mode=cfg.DEBUG)
    # Create experiment directory
    if (not cfg.DEBUG and not Path(
            cfg.EXPERIMENT_DIR).exists()):
        Path(cfg.EXPERIMENT_DIR).mkdir(parents=True)
    # login to WandB
    if not cfg.DEBUG:
        wandb.login(key=cfg.WANDB_KEY)
        del cfg.WANDB_KEY
    runner = NetworkTrainer(cfg)
    return runner

def setup_agent_tester() -> AgentTester:
    cfg = MasterConfig.from_yaml(
        args.config, 'TEST_AGENT')
    cfg.DEBUG = args.debug
    ray.init(local_mode=cfg.DEBUG)
    runner = AgentTester(cfg)
    return runner

# Main function
def main() -> None:
    if args.collect_data:
        runner = setup_data_collector()
    elif args.train_vnet:
        runner = setup_vnet_trainer()
    elif args.train_mnet:
        runner = setup_mnet_trainer()
    elif args.train_cnet:
        runner = setup_cnet_trainer()
    elif args.test_agent:
        runner = setup_agent_tester()
    else:
        parser.print_help()
        return
    runner.execute()

# CLI
parser = argparse.ArgumentParser(description='Runs environment for collecting data and testing agents.')
parser.add_argument('--config', help='Path to .yaml configuration file.',
                    default='./master-config.yml')
parser.add_argument('--collect-data', action='store_true',
                    help='Flag to execute data collection procedure.')
parser.add_argument('--train-vnet', action='store_true',
                    help='Flag to execute training of vision-network.')
parser.add_argument('--train-mnet', action='store_true',
                    help='Flag to execute training of memory-network.')
parser.add_argument('--train-cnet', action='store_true',
                    help='Flag to execute training of controller-network.')
parser.add_argument('--test-agent', action='store_true',
                    help='Flag to execute testing of a trained agent.')
parser.add_argument('--debug', action='store_true',
                    help='Flag to run task in debug mode.')
args = parser.parse_args()

if __name__ == '__main__':
    main()
