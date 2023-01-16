# Imports
from abc import abstractmethod
import torch
import pytorch_lightning as pl
from utils import MasterConfig
from networks import Vnet, Mnet, Cnet


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
    def training_step(self) -> torch.Tensor:
        raise NotImplementedError()

    @abstractmethod
    def validation_step(self) -> torch.Tensor:
        raise NotImplementedError()


class VnetTrainingModule(TrainingModule):

    def __init__(self, config: MasterConfig) -> None:
        super().__init__(config)
        self.net = Vnet(config)

    def training_step(self, image: torch.Tensor) -> torch.Tensor:
        recon_image, _ = self.net(image)
        loss = self.loss_function(image, recon_image)
        return loss

    def validation_step(self, image: torch.Tensor) -> torch.Tensor:
        reconstructed, _ = self.net(image)
        loss = self.loss_function(image, reconstructed)
        return loss

    def loss_function(self, original: torch.Tensor,
                      reconstructed: torch.Tensor) -> torch.Tensor:
        ...


class MnetTrainingModule(TrainingModule):

    def __init__(self, config: MasterConfig) -> None:
        super().__init__(config)
        self.net = Mnet(config)


class CnetTrainingModule(TrainingModule):

    def __init__(self, config: MasterConfig) -> None:
        super().__init__(config)
        self.net = Cnet(config)
