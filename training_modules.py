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
    """
    Training module for training the 'V' vision network.
    """

    def __init__(self, config: MasterConfig) -> None:
        super().__init__(config)
        self.net = Vnet(config)

    def loss_function(self, original: torch.Tensor,
                      reconstructed: torch.Tensor,
                      latent: torch.Tensor) -> torch.Tensor:
        generative_loss = torch.nn.MSELoss()(reconstructed, original)
        latent_loss = torch.nn.KLDivLoss()(latent, torch.randn_like(latent))
        loss = generative_loss + latent_loss
        return loss

    def training_step(self, image: torch.Tensor) -> torch.Tensor:
        reconstructed, latent = self.net(image)
        loss = self.loss_function(image, reconstructed, latent)
        return loss

    def validation_step(self, image: torch.Tensor) -> torch.Tensor:
        reconstructed, latent = self.net(image)
        loss = self.loss_function(image, reconstructed, latent)
        return loss


class MnetTrainingModule(TrainingModule):
    """
    Training module for training the 'M' memory network.
    """
    
    def __init__(self, config: MasterConfig) -> None:
        super().__init__(config)
        self.net = Mnet(config)

    def loss_function(self,
            z_next_est_dist: torch.distributions.Distribution,
            z_next: torch.Tensor) -> torch.Tensor:
        # negative log probability
        #TODO: might need more sophisticated
        #       form of liklihood calculation
        return -z_next_est_dist.log_prob(z_next)

    def training_step(self, batch: torch.Tensor) -> torch.Tensor:
        # unpack batch
        # (B,S,Z), (B,S,A)
        z_prev, a_prev, z_next = batch
        z_next_est_dist = self.net(z_prev, a_prev)
        loss = self.loss_function(z_next_est_dist, z_next)
        return loss

    def validation_step(self, batch: torch.Tensor) -> torch.Tensor:
        # unpack batch
        # (B,S,Z), (B,S,A)
        z_prev, a_prev, z_next = batch
        z_next_est_dist = self.net(z_prev, a_prev)
        loss = self.loss_function(z_next_est_dist, z_next)
        return loss


class CnetTrainingModule(TrainingModule):
    """
    Module designed for 'Covariance-Matrix
    Adaptation Evolution Strategy' to train the
    'C' controller network.
    """

    def __init__(self, config: MasterConfig) -> None:
        super().__init__(config)
        self.net = Cnet(config)

