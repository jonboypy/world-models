# Imports
from abc import ABC, abstractmethod
import pytorch_lightning as pl
from utils import MasterConfig

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
        self._callbacks = self._get_callbacks()
        self._loggers = self._get_loggers()
        self._strategy = self._get_strategy()
        self._profilers = self._get_profilers()
        self._data_module = self._get_data_module()
        self._training_module = self._get_training_module()
        self._lit_trainer = self._get_lit_trainer()
        self._is_setup = True

    def train(self) -> None:
        assert self._is_setup
        return super().train()

    def test(self) -> None:
        assert self._is_setup
        return super().test()

    def _get_callbacks(self):
        ...

    def _get_loggers(self):
        ...

    def _get_stategy(self):
        ...

    def _get_profilers(self):
        ...

    def _get_data_module(self) -> pl.LightningDataModule:
        ...

    def _get_training_module(self) -> pl.LightningModule:
        ...

    def _get_lit_trainer(self) -> pl.Trainer:
        ...


class EvolutionTrainer(Trainer):

    def __init__(self, config: MasterConfig) -> None:
        super().__init__(config)
        self._is_setup = False

    def setup(self) -> None:
        self._is_setup = True

