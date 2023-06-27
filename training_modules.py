# ----------------------------- Imports ------------------------------
from typing import Any, Tuple, List
from abc import abstractmethod
import torch
import torchvision
import pytorch_lightning as pl
import ray
import numpy as np
import cma
from tqdm import tqdm
from utils import MasterConfig
from networks import Vnet, Mnet, Cnet
from environments import GymEnvironment
from plugins import Plugin
from agents import WorldModelGymAgent

# -------------------------------------------------------------------
#   Training Modules
# -------------------------------------------------------------------

class TrainingModule(pl.LightningModule):
    """
    Base class which all other *Lightning* training modules derive from.

    Args:
        config: Master config object.
    """

    def __init__(self, config: MasterConfig) -> None:
        super().__init__()
        self.config = config
        self.save_hyperparameters(self.config)

    @abstractmethod
    def configure_optimizers(self) -> torch.optim.Optimizer:
        raise NotImplementedError()

    @abstractmethod
    def loss_function(self) -> torch.Tensor:
        raise NotImplementedError()

    @abstractmethod
    def training_step(self) -> torch.Tensor:
        raise NotImplementedError()

    @abstractmethod
    def validation_step(self) -> torch.Tensor:
        raise NotImplementedError()

class VnetTrainingModule(TrainingModule):
    """
    Training module for training the 'V' vision network.

    Args:
        config: Master config object.
    """

    def __init__(self, config: MasterConfig) -> None:
        super().__init__(config)
        self.net = Vnet(config)
        self.net.initialize_parameters('kaiming')

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optim = torch.optim.Adam(self.net.parameters(),
                                 lr=self.config.LR)
        return optim

    def loss_function(self, original: torch.Tensor,
                      reconstructed: torch.Tensor,
                      mu: torch.Tensor, sigma: torch.Tensor
                      ) -> torch.Tensor:
        # Image-Reconstruction loss
        reconstruction_loss = torch.nn.functional.mse_loss(
                    reconstructed, original, reduction='none')
        reconstruction_loss = reconstruction_loss.sum(dim=(1,2,3)).mean()
        # KL-Divergence loss (sigma == log-variance)
        kld_loss = -0.5 * torch.sum(
            1 + sigma - mu.pow(2) - sigma.exp(),
            dim = 1)
        kld_loss = kld_loss.mean()
        combined_loss = reconstruction_loss + kld_loss
        return combined_loss, reconstruction_loss, kld_loss

    def training_step(self, batch: torch.Tensor,
                      batch_idx: int) -> torch.Tensor:
        reconstructed, _, mu, sigma = self.net(batch)
        (loss, reconstrution_loss,
         kld_loss) = self.loss_function(
            batch, reconstructed, mu, sigma)
        result = {'loss': loss,
                  'reconstruction_loss': reconstrution_loss,
                  'kld_loss': kld_loss}
        return result
    
    def validation_step(self, batch: torch.Tensor,
                        batch_idx: int) -> torch.Tensor:
        reconstructed, _, mu, sigma = self.net(batch)
        (loss, reconstruction_loss,
         kld_loss) = self.loss_function(
            batch, reconstructed, mu, sigma)
        result = {'reconstructed': reconstructed,
                  'loss': loss,
                  'reconstruction_loss': reconstruction_loss,
                  'kld_loss': kld_loss}
        return result

class MnetTrainingModule(TrainingModule):
    """
    Training module for training the 'M' memory network.

    Args:
        config: Master config object.
    """
    
    def __init__(self, config: MasterConfig) -> None:
        super().__init__(config)
        self.net = Mnet(config)
        self.net.initialize_parameters('kaiming')

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optim = torch.optim.Adam(self.net.parameters(),
                                 lr=self.config.LR)
        return optim

    def loss_function(self,
            z_next_est_dist: torch.distributions.Distribution,
            z_next: torch.Tensor) -> torch.Tensor:
        return (-z_next_est_dist.log_prob(z_next)).mean()

    def training_step(self, batch: torch.Tensor,
                      batch_idx: int) -> torch.Tensor:
        # unpack batch
        z_prev, a_prev, z_next = batch
        z_next_est_dist, _ = self.net(z_prev, a_prev)
        loss = self.loss_function(z_next_est_dist, z_next)
        result = {'loss': loss}
        return result

    def validation_step(self, batch: torch.Tensor,
                        batch_idx: int) -> torch.Tensor:
        # unpack batch
        z_prev, a_prev, z_next = batch
        z_next_est_dist, _ = self.net(z_prev, a_prev)
        loss = self.loss_function(z_next_est_dist, z_next)
        result = {'loss': loss,
                  'z_next_est_dist': z_next_est_dist}
        return result

class CnetTrainingModule:

    """
    Module designed for 'Covariance-Matrix
    Adaptation Evolution Strategy' to train the
    'C' controller network.

    Args:
        config: Master config object.
        plugins: Plugins to extend module's functions.
    """

    def __init__(self, config: MasterConfig,
                 plugins: List=None) -> None:
        self.config = config
        self.pop_size = config.POP_SIZE
        self.generations = config.EPOCHS
        self.plugins = plugins

    @Plugin.hookable    
    def log(self, name: str, metric: Any,
            plot_type: str, step: int) -> None:
        # NOTE: For plugin to extend
        return

    @Plugin.hookable    
    def save(self, model: torch.nn.Module,
             filename: str, metric: float) -> None:
        # NOTE: For plugin to extend
        return

    def optimize(self) -> None:
        # Execute initialization
        self._initialize()
        # For every epoch...
        for g in tqdm(range(self.generations),
                       desc='Optimizing', unit='Generation'):
            # Get new population from algorithm to evaluate
            population = self.alg.ask()
            # Evaluate each individual in population
            scores = self._evaluate_population(population)
            # Report results to algorithm
            self.alg.tell(population, -scores)
            # log progress
            self.log('Population Average Return', scores.mean(), 'line', g)
            self.log('Best Average Return', scores.max(), 'line', g)
            self.log('Worst Average Return', scores.min(), 'line', g)
            self.log('µ', self.alg.mean, 'histogram', g)
            self.log('σ', self.alg.C, 'histogram', g)
            # checkpoint best model
            self.save(self._params2network(self.alg.result.xbest),
                f'generation={g}-return={scores.max():.3e}.ckpt', scores.max())
            # Perform evaluation of top performer to log algorithm progress
            if (g+1) % self.config.TEST_EVERY_N_GENERATIONS == 0:
                training_module = ray.put(self)
                video_name = f'Generation={g}-Return={scores.max():.3e}'
                score = self._evaluate_individual.remote(
                    training_module,
                    self.alg.result.xbest,
                    self.config.TEST_N_ROLLOUTS,
                    (True if not self.config.DEBUG else False),
                    video_name)
                score = ray.get(score)
                for i in range(self.config.TEST_N_ROLLOUTS):
                    self.log(video_name + f'-episode-{i}', 
                             self.config.EXPERIMENT_DIR + '/' + \
                                video_name + f'-episode-{i}.mp4',
                             'video', g)
                self.log('Test Average Return', score, 'line', g)

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
        self.vnet_encoder = self.vnet_encoder.eval().cpu()
        self.mnet = self.mnet.eval().cpu()
        for p in self.vnet_encoder.parameters():
            p.requires_grad = False
        for p in self.mnet.parameters():
            p.requires_grad = False

    def _initialize(self) -> Tuple[torch.Tensor]:
        # Load models
        self._load_modules()
        # Calculate the number of parameters in C-Net
        self.event_space = self.cnet.l1.weight.numel() + \
                            self.cnet.l1.bias.numel()
        # Initialize algorithm
        self.alg = cma.CMAEvolutionStrategy(
            [0]*self.event_space,
            self.config.INIT_SIGMA,
            {'popsize': self.config.POP_SIZE}
        )

    def _evaluate_population(
            self, population: np.ndarray) -> np.ndarray:
        # Launch parallel evaluations of each individual in population
        training_module = ray.put(self)
        scores = []
        for idx in range(self.pop_size):
            scores += [
                self._evaluate_individual.remote(
                    training_module, population[idx],
                    self.config.EVAL_N_ROLLOUTS)]
        scores = ray.get(scores)
        return np.array(scores)

    def _params2network(
            self, params: np.ndarray) -> torch.nn.Module:
        # Load C-Net instance
        cnet = Cnet(self.config)
        # Convert params to Torch tensor
        params = torch.from_numpy(params.copy()).float()
        # Reshape params to size of weight matrix (+1 for bias vector)
        params = params.reshape(cnet.l1.weight.size(0),
                                cnet.l1.weight.size(1)+1)
        # Load params into net
        cnet.l1.weight = torch.nn.Parameter(params[:,:-1], False)
        cnet.l1.bias = torch.nn.Parameter(params[:,-1], False)
        return cnet
    
    @ray.remote
    def _evaluate_individual(
            self, individual: np.ndarray,
            n_rollouts: int,
            record_video: bool = False,
            video_name: str = None) -> float:
        # convert individual into a network
        cnet = self._params2network(individual)
        # Make environment
        if record_video:
            env = GymEnvironment(
                self.config, record_video=True,
                video_name=video_name)
        else:
            env = GymEnvironment(self.config)
        env.reset()
        # Make agent
        agent = WorldModelGymAgent(env, self.vnet_encoder,
                                   self.mnet, cnet)
        # Run rollouts
        n_eps = 0
        while n_eps < n_rollouts:
            done = agent.act() 
            n_eps += done
        # Get fitness score of individual (i.e. mean return)
        score = agent.avg_cum_reward
        return score
        
# -------------------------------------------------------------------
#   Callbacks
# -------------------------------------------------------------------

class VnetMetricLoggerCallback(pl.Callback):

    """
    Simple callback to log training & validation losses for W&B.
    """

    def __init__(self) -> None:
        super().__init__()

    def on_fit_start(self, trainer: pl.Trainer,
                     pl_module: pl.LightningModule) -> None:
        pl_module.logger.watch(pl_module, log="all")

    def on_train_batch_end(self, trainer: pl.Trainer,
                           pl_module: pl.LightningModule,
                           outputs: torch.Tensor, batch: Any,
                           batch_idx: int) -> None:
        pl_module.log('training_combined_loss', outputs['loss'])
        pl_module.log('training_reconstruction_loss',
                      outputs['reconstruction_loss'])
        pl_module.log('training_kld_loss', outputs['kld_loss'])


    def on_validation_batch_end(
            self, trainer: pl.Trainer,
            pl_module: pl.LightningModule,
            outputs: torch.Tensor, batch: Any,
            batch_idx: int, dataloader_idx: int = 0) -> None:
        # Log loss
        pl_module.log('validation_combined_loss', outputs['loss'])
        pl_module.log('validation_reconstruction_loss',
                      outputs['reconstruction_loss'])
        pl_module.log('validation_kld_loss', outputs['kld_loss'])
        # Convert tensors to images
        original = (batch[0] * 255.).to(
            dtype=torch.uint8, device='cpu')
        original = torchvision.transforms.functional.to_pil_image(
                                                            original)
        reconstructed = (outputs['reconstructed'][0] * 255.).to(
                                dtype=torch.uint8, device='cpu')
        reconstructed = torchvision.transforms.functional.to_pil_image(
                                                            reconstructed)
        # Log images
        pl_module.logger.log_image(
            key=(f"Original vs. Reconstructed Image Comparison"),
            images=[original, reconstructed],
            caption=["Original", "Reconstructed"])

class MnetMetricLoggerCallback(pl.Callback):

    """
    Simple callback to log training & validation metrics for W&B.

    Args:
        vnet_decoder: vision network's decoder for
            visualization of latent vectors.
    """

    def __init__(self, vnet_decoder: torch.nn.Module = None) -> None:
        super().__init__()
        self.vnet_decoder = vnet_decoder

    def on_fit_start(self, trainer: pl.Trainer,
                     pl_module: pl.LightningModule) -> None:
        pl_module.logger.watch(pl_module, log="all")
        if self.vnet_decoder is not None:
            self.vnet_decoder = self.vnet_decoder.to(pl_module.device)

    def on_train_batch_end(self, trainer: pl.Trainer,
                           pl_module: pl.LightningModule,
                           outputs: torch.Tensor, batch: Any,
                           batch_idx: int) -> None:
        pl_module.log('training_loss', outputs['loss'])

    def on_validation_batch_end(self, trainer: pl.Trainer,
                                pl_module: pl.LightningModule,
                                outputs: torch.Tensor, batch: Any,
                                batch_idx: int, dataloader_idx: int = 0) -> None:
        pl_module.log('validation_loss', outputs['loss'])
        if batch_idx == 0 and self.vnet_decoder is not None:
            _, _, z_next = batch
            pred_z_next = outputs['z_next_est_dist'].sample().float()
            true_decoded = self.vnet_decoder(z_next[0,-1].unsqueeze(0))[0]
            pred_decoded = self.vnet_decoder(pred_z_next[0,-1].unsqueeze(0))[0]
            # Convert tensors to images
            true_decoded = (true_decoded * 255.).to(
                dtype=torch.uint8, device='cpu')
            true_decoded = torchvision.transforms.functional.to_pil_image(
                                                                true_decoded)
            pred_decoded = (pred_decoded * 255.).to(
                            dtype=torch.uint8, device='cpu')
            pred_decoded = torchvision.transforms.functional.to_pil_image(
                                                                pred_decoded)
            # Log images
            pl_module.logger.log_image(
                key=(f"True Z_next vs. Predicted Z_next Image Comparison"),
                images=[true_decoded, pred_decoded],
                caption=["True", "Predicted"])
