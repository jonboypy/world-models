# Imports
import pathlib
from abc import ABC, abstractmethod
from typing import Any, List
import torch
import pytorch_lightning as pl
import ray
from utils import MasterConfig
from plugins import Plugin, CNetModelCheckpoint, CNetWandbLogger
from training_modules import (
    VnetTrainingModule, MnetTrainingModule,
    CnetTrainingModule, VnetMetricLoggerCallback,
    MnetMetricLoggerCallback)
from datasets import DataModule

#####################################################################
#   Trainers
#####################################################################

class Trainer(ABC):

    """
        Base class which all other trainers derive from.

        Args:
            cfg: MasterConfig object.
    """

    def __init__(self, cfg: MasterConfig) -> None:
        super().__init__()
        self.cfg = cfg

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
    
        Args:
            cfg: Master configuration object.
    """

    def __init__(self, cfg: MasterConfig) -> None:
        super().__init__(cfg)
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
        if not self.cfg.RUN_TYPE == 'M-NET' and not self.cfg.DEBUG: #BUG: M-Net won't compile
            self._training_module = torch.compile(self._training_module)
        self._lit_trainer.fit(self._training_module,
                              self._data_module)

    def test(self) -> None:
        assert self._is_setup
        return super().test()

    def _get_loggers(self):
        loggers = []
        if not self.cfg.DEBUG:
            loggers += [pl.loggers.WandbLogger(
                        project='World-Models',
                        save_dir=self.cfg.EXPERIMENT_DIR,
                        name=(f'{self.cfg.RUN_TYPE}/'
                              f'{self.cfg.EXPERIMENT_NAME}'),
                        log_model='all')]
            self.exp_dir = (pathlib.Path(loggers[0].save_dir) /
                            loggers[0].name /
                            f'version_{loggers[0].version}')
        return loggers

    def _get_callbacks(self) -> List[pl.Callback]:
        callbacks = []
        if not self.cfg.DEBUG:
            if self.cfg.RUN_TYPE == 'V-NET':
                callbacks += [pl.callbacks.ModelCheckpoint(
                        monitor='validation_combined_loss',
                        filename='{epoch:03d}-{validation_combined_loss:.3e}',
                        mode='min',
                        save_last=True,
                        save_top_k=3)]
                callbacks += [VnetMetricLoggerCallback()]
            elif self.cfg.RUN_TYPE == 'M-NET':
                callbacks += [pl.callbacks.ModelCheckpoint(
                        monitor='validation_loss',
                        filename='{epoch:03d}-{validation_loss:.3e}',
                        mode='min',
                        save_last=True,
                        save_top_k=3)]
                if hasattr(self.cfg, 'VNET_CKPT'):
                    vnet_decoder = \
                        VnetTrainingModule.load_from_checkpoint(
                        self.cfg.VNET_CKPT, cfg=self.cfg).net.decoder
                    vnet_decoder.eval()
                    callbacks += [MnetMetricLoggerCallback(vnet_decoder)]
                else:
                    callbacks += [MnetMetricLoggerCallback()]
        return callbacks

    def _get_profiler(self) -> Any:
        return None

    def _get_data_module(self) -> pl.LightningDataModule:
        datamodule = DataModule(self.cfg)
        return datamodule

    def _get_training_module(self) -> pl.LightningModule:
        if self.cfg.RUN_TYPE == 'V-NET':
            return VnetTrainingModule(self.cfg)
        elif self.cfg.RUN_TYPE == 'M-NET':
            return MnetTrainingModule(self.cfg)
        else:
            raise SyntaxError(
                ('Network type '
                 f'{self.cfg.RUN_TYPE} '
                 'not supported.'))
    
    def _get_lit_trainer(self) -> pl.Trainer:
        if self.cfg.RUN_TYPE == 'V-NET':
            return pl.Trainer(max_epochs=self.cfg.EPOCHS,
                              precision=16,
                              logger=self._loggers,
                              callbacks=self._callbacks,
                              profiler=self._profiler,
                              fast_dev_run=self.cfg.DEBUG,
                              benchmark=True,
                              num_sanity_val_steps=0,
                              gradient_clip_algorithm=(
                                'value' if self.cfg.GRADIENT_CLIP else None),
                              gradient_clip_val=self.cfg.GRADIENT_CLIP,
                              val_check_interval=(0.1))
        elif self.cfg.RUN_TYPE == 'M-NET':
            return pl.Trainer(max_epochs=self.cfg.EPOCHS,
                              precision=16,
                              logger=self._loggers,
                              callbacks=self._callbacks,
                              profiler=self._profiler,
                              fast_dev_run=self.cfg.DEBUG,
                              benchmark=True,
                              gradient_clip_algorithm=(
                                'value' if self.cfg.GRADIENT_CLIP else None),
                              gradient_clip_val=self.cfg.GRADIENT_CLIP,
                              num_sanity_val_steps=0)

class EvolutionTrainer(Trainer):

    def __init__(self, cfg: MasterConfig) -> None:
        super().__init__(cfg)
        self._is_setup = False

    def setup(self) -> None:
        self._plugins = self._get_plugins()
        self._training_module = self._get_training_module()
        self._is_setup = True

    def train(self) -> None:
        assert self._is_setup, (
            f'Trainer isnt setup! Call .setup()')
        self._training_module.optimize()

    def test(self) -> None:
        return super().test()

    def _get_training_module(self) -> CnetTrainingModule:
        return CnetTrainingModule(self.cfg, self._plugins)

    def _get_plugins(self) -> List[Plugin]:
        plugins = []
        if not self.cfg.DEBUG:
            # Logger
            plugins += [CNetWandbLogger(self.cfg)]
            # Checkpointing
            ckpt_dir = (pathlib.Path(self.cfg.EXPERIMENT_DIR)/
                        'World-Models'/plugins[0].run.id/'checkpoints')
            self.cfg.EXPERIMENT_DIR = str(pathlib.Path(self.cfg.EXPERIMENT_DIR)/
                                            'World-Models'/plugins[0].run.id)
            plugins += [CNetModelCheckpoint(ckpt_dir)]
        return plugins
