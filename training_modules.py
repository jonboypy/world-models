# Imports
from typing import Tuple
import copy
from abc import abstractmethod
import torch
import pytorch_lightning as pl
import ray
from tqdm import tqdm
from utils import MasterConfig
from networks import Vnet, Mnet, Cnet
from environments import GymEnvironment
from agents import WorldModelGymAgent


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


class CnetTrainingModule:
    """
    Module designed for 'Covariance-Matrix
    Adaptation Evolution Strategy' to train the
    'C' controller network.
    """

    def __init__(self, config: MasterConfig) -> None:
        self.config = config
        self.pop_size = config.POP_SIZE
        self.top_percentage = config.TOP_PERCENTAGE
        self.epochs = config.EPOCHS
        self._load_modules()
        self._initialize()

    def _load_modules(self) -> None:
        self.cnet = Cnet(self.config)
        if hasattr(self, 'TEST'):
            self.vnet_encoder = Vnet(self.config).encoder
            self.mnet = Mnet(self.config)
        else:
            self.vnet_encoder = VnetTrainingModule.\
                load_from_checkpoint(
                self.config.VNET_CKPT,
                config = self.config).net.encoder
            self.mnet = MnetTrainingModule.\
                load_from_checkpoint(
                self.config.MNET_CKPT,
                config = self.config).net

    def _initialize(self) -> Tuple[torch.Tensor]:
        # Calculate the number of parameters in C-Net
        self.event_space = self.cnet.l1.weight.numel() + \
                            self.cnet.l1.bias.numel()
        # Initialize random location & covariance matricies
        mu = torch.rand(self.event_space)
        sigma = torch.eye(self.event_space)
        return mu, sigma

    def optimize(self) -> None:
        # Get initial mu & sigma
        mu, sigma = self._initialize()
        # For every epoch...
        for e in tqdm(range(self.epochs), description='Optimizing'):
            # Create a sample space based on the current mu & sigma
            space = torch.distributions.MultivariateNormal(mu, sigma)
            # Sample a population of parameters from the space
            population = space.sample((self.pop_size,))
            # Evaluate each set of parameters in population
            scores = self._evaluate_population(population)
            # Get the top % of the population
            best = torch.topk(scores, int(
                self.pop_size * self.top_percentage))
            # Calculate new covariance matrix
            var = ((population[best.indices] - mu).pow(2)).mean(0)
            cov = ((population[best.indices] - mu).prod(1)).mean(0)
            sigma = torch.diag(cov.view(1,),1) + \
                        torch.diag(cov.view(1,),-1) + \
                            torch.diag(var)
            # Enforce sigma to be greater than 0
            sigma = sigma.clamp(1e-32)
            # Calculate new mu
            mu = population[best.indices].mean(0)
            # Perform evaluation of top performer to log algorithm progress
            if e % self.config.TEST_EVERY_N_EPOCHS == 0:
                score = self._evaluate_individual(
                    population[best.indices][0],
                    self.config.TEST_N_ROLLOUTS)
                self.log(score)
        # TODO: Save best individual

    def _evaluate_population(self,
            population: torch.Tensor) -> torch.Tensor:
        # Launch parallel evaluations of each individual in population
        cfg = copy.deepcopy(self.config)
        scores = ray.get([self._evaluate_individual.remote(
                          cfg,
                          population[i],
                          self.vnet_encoder, self.mnet,
                          self.config.EVAL_N_ROLLOUTS)
                          for i in range(self.pop_size)]) 
        return scores
    
    @staticmethod
    @ray.remote
    def _evaluate_individual(config: MasterConfig,
                             individual: torch.Tensor,
                             vnet_encoder: torch.nn.Module,
                             mnet: torch.nn.Module,
                             n_rollouts: int) -> torch.Tensor:
        # Load parameters into C-Net
        cnet = Cnet(config)
        # Reshape params to size of weight matrix (+1 for bias vector)
        individual = individual.reshape(cnet.l1.weight.size(0),
                                        cnet.l1.weight.size(1)+1)
        # Load params into net
        cnet.l1.weight = torch.nn.Parameter(individual[:,:-1], False)
        cnet.l1.bias = torch.nn.Parameter(individual[:,-1], False)
        # Make environment
        env = GymEnvironment(config)
        env.reset()
        # Make agent
        agent = WorldModelGymAgent(env, vnet_encoder,
                                   mnet, cnet)
        # Run rollouts
        n_eps = 0
        while n_eps < n_rollouts: #TODO could parallelize the rollouts too?
            n_eps += agent.act()
        # Get fitness score of individual (i.e. mean return)
        score = agent.avg_cum_reward
        return score
