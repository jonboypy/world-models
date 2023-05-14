# Imports
import pathlib
import argparse
from abc import ABC, abstractmethod
from typing import Any, List
import wandb
import pytorch_lightning as pl
from utils import MasterConfig
from training_modules import (
    VnetTrainingModule, MnetTrainingModule,
    CnetTrainingModule, MetricLoggerCallback)
from datasets import DataModule

# Trainers
class Trainer(ABC):

    """
        Base class which all other trainers derive from.

        Args:
            config: MasterConfig object.
    """

    def __init__(self, config: MasterConfig) -> None:
        super().__init__()
        self.config = config

    @abstractmethod
    def setup(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def train(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def test(self) -> None:
        raise NotImplementedError

class LitTrainer(Trainer):

    """
        Trainer for PyTorch-Lightning
        training-modules (i.e. V-Net/M-Net).
    """

    def __init__(self, config: MasterConfig) -> None:
        super().__init__(config)
        self._is_setup = False

    def setup(self) -> None:
        self._loggers = self._get_loggers()
        self._callbacks = self._get_callbacks()
        self._profiler = self._get_profiler()
        self._data_module = self._get_data_module()
        self._training_module = self._get_training_module()
        self._lit_trainer = self._get_lit_trainer()
        self._is_setup = True

    def train(self) -> None:
        assert self._is_setup, (
            f'Trainer isnt setup! Call .setup()')
        self._lit_trainer.fit(self._training_module,
                              self._data_module)

    def test(self) -> None:
        assert self._is_setup
        return super().test()

    def _get_loggers(self):
        loggers = []
        if not self.config.DEBUG:
            loggers += [pl.loggers.WandbLogger(
                        save_dir=self.config.EXPERIMENT_DIR,
                        name=self.config.EXPERIMENT_NAME,
                        project='World-Models')]
            self.exp_dir = (pathlib.Path(loggers[0].save_dir) /
                            loggers[0].name /
                            f'version_{loggers[0].version}')
        return loggers


    def _get_callbacks(self) -> List[pl.Callback]:
        callbacks = []
        if not self.config.DEBUG:
            #callbacks += [pl.callbacks.ModelCheckpoint(
            #        monitor='validation_loss',
            #        dirpath=(self.exp_dir / 'checkpoints'),
            #        filename='{epoch:03d}-{validation_loss:.3e}',
            #        mode='min',
            #        save_last=True,
            #        save_top_k=3)]
            callbacks += [MetricLoggerCallback()]
        return callbacks

    def _get_profiler(self) -> Any:
        return None

    def _get_data_module(self) -> pl.LightningDataModule:
        datamodule = DataModule(self.config)
        return datamodule

    def _get_training_module(self) -> pl.LightningModule:
        exp_type = self.config.EXPERIMENT_TYPE
        if exp_type == 'V-Net':
            return VnetTrainingModule(self.config)
        elif exp_type == 'M-Net':
            return MnetTrainingModule(self.config)
        else:
            raise SyntaxError(
                f'Experiment type {exp_type} not supported.')
    
    def _get_lit_trainer(self) -> pl.Trainer:
        return pl.Trainer(max_epochs=self.config.EPOCHS,
                          precision=16,
                          logger=self._loggers,
                          callbacks=self._callbacks,
                          profiler=self._profiler,
                          fast_dev_run=self.config.DEBUG)

class EvolutionTrainer(Trainer):

    def __init__(self, config: MasterConfig) -> None:
        super().__init__(config)
        self._is_setup = False
        raise NotImplementedError()

    def setup(self) -> None:
        self._is_setup = True

# CLI
parser = argparse.ArgumentParser(description='Runs training of networks.')
parser.add_argument('--config', help='Path to .yaml configuration file.',
                    default='./master-config.yml')
parser.add_argument('--debug', action='store_true',
                    help='Run trainer in debug-mode')
args = parser.parse_args()

def main() -> None:
    # Create config object
    config = MasterConfig.from_yaml(args.config)

    # Set debugging options
    if args.debug:
        config.DEBUG = True
    else:
        config.DEBUG = False
    # Create experiment directory
    if (not config.DEBUG and not pathlib.Path(
            config.EXPERIMENT_DIR).exists()):
        pathlib.Path(config.EXPERIMENT_DIR).mkdir(parents=True)
    # Create trainer
    if any(t == config.EXPERIMENT_TYPE for t in ('V-Net', 'M-Net')):
        if not config.DEBUG:
            wandb.login(key=config.WANDB_KEY)
        trainer = LitTrainer(config)
    elif config.EXPERIMENT_TYPE == 'C-Net':
        trainer = EvolutionTrainer(config)
    else:
        raise SyntaxError('Experiment type '
                          f'{config.EXPERIMENT_TYPE} not supported.')
    # Setup
    trainer.setup()
    # Train!
    trainer.train()

if __name__ == '__main__':
    main()
