# Imports
from abc import abstractmethod
from typing import Any
import torch
import pytorch_lightning as pl
from utils import MasterConfig
from networks import Vnet


# PTL modules
class TrainingModule(pl.LightningModule):
    """
    Base class which all other training modules derive from.

    Args:
        config: Master config object.
    """

    def __init__(self, config: MasterConfig) -> None:
        super().__init__()
        self.config = config

    @abstractmethod
    def training_step(self) -> Any:
        raise NotImplementedError()

    @abstractmethod
    def validation_step(self) -> Any:
        raise NotImplementedError()


class VnetLitModule(TrainingModule):

    def __init__(self, config: MasterConfig) -> None:
        super().__init__(config)
        self.net = Vnet(config)
