#Imports
from abc import abstractmethod
from pathlib import Path
from typing import Tuple, Optional, Union
import shutil
from tqdm import tqdm
import torch
import numpy as np
import pytorch_lightning as pl
import h5py
from utils import MasterConfig
from training_modules import VnetTrainingModule

# Datasets
class Dataset(torch.utils.data.Dataset):
    """
    Base class which all other training
    datasets are derived from.

    Args:
        config: A MasterConfig object.
    """

    def __init__(self, config: MasterConfig) -> None:
        super().__init__()
        self.config = config

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError() 

    @abstractmethod
    def __getitem__(self) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
        raise NotImplementedError()

class VnetDataset(Dataset):

    def __init__(self, config: MasterConfig) -> None:
        super().__init__(config)
        self._setup()
    
    def _setup(self) -> None:
        # load HDF5
        self._hf = h5py.File(self.config.DATA, 'r')
        # calculate length of data and create LUT for indicies
        self._len = 0
        self._idx_dict = {} # {(start_idx, end_idx): episode #}
        for eps, ds in self._hf.items():
            st = self._len
            self._len += len(ds['actions'])
            end = self._len
            self._idx_dict[(st, end)] = int(eps)

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int) -> torch.Tensor:
        # get episode number and step number from index
        eps, step = self._idx_2_eps_step(idx)
        # get image from dataset
        img = self._hf[str(eps)]['observations'][step]
        # preprocess image
        img = self._preprocess(img)
        return img

    @classmethod
    def preprocess(cls, img: np.ndarray) -> torch.Tensor:
        # convert numpy uint8 arr to torch 32-bit float tensor
        img = torch.tensor(img, dtype=torch.float32)
        # scale all pixels to the range [0,1]
        img = img / 255.
        # permute dims to Torch standard (C,H,W)
        img = img.permute(2,0,1)
        return img

    def _idx_2_eps_step(self, idx: int) -> Tuple[int]:
        for idx_range, eps in self._idx_dict.items():
            st, end = idx_range
            if idx >= st and idx < end:
                step = idx - st
                return eps, step

class MnetDataset(Dataset):
    
    def __init__(self, config: MasterConfig) -> None:
        super().__init__(config)
        self._setup()

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int) -> torch.Tensor:
        # get episode number and step number from index
        eps, step = self._idx_2_eps_step(idx)
        # get data from dataset
        mu = self._hf[str(eps)]['latent_means'][step]
        sigma = self._hf[str(eps)]['latent_stds'][step]
        action = self._hf[str(eps)]['actions'][step]
        # get distribution of *next* latent vector
        next_mu = self._hf[str(eps)]['latent_means'][step + 1]
        next_sigma = self._hf[str(eps)]['latent_stds'][step + 1]
        # preprocess data
        (mu, sigma, action,
         next_mu, next_sigma) = self.preprocess(mu, sigma, action,
                                                 next_mu, next_sigma)
        # sample latent vectors from distribution parameters
        #   *this helps prevent overfitting*
        z = mu + sigma * self._gaussian.sample(sigma.size())
        next_z = next_mu + next_sigma * self._gaussian.sample(sigma.size())
        return z, action, next_z

    @classmethod
    def preprocess(cls, mu: np.ndarray, sigma: np.ndarray,
                   action: np.ndarray, next_mu: np.ndarray,
                   next_sigma: np.ndarray) -> torch.Tensor:
        # convert all to torch tensors
        mu = torch.tensor(mu, dtype=torch.float32)
        sigma = torch.tensor(sigma, dtype=torch.float32)
        action = torch.tensor(action, dtype=torch.float32)
        next_mu = torch.tensor(next_mu, dtype=torch.float32)
        next_sigma = torch.tensor(next_sigma, dtype=torch.float32)
        return mu, sigma, action, next_mu, next_sigma

    def _idx_2_eps_step(self, idx: int) -> Tuple[int]:
        for idx_range, eps in self._idx_dict.items():
            st, end = idx_range
            if idx >= st and idx < end:
                step = idx - st
                return eps, step

    def __del__(self) -> None:
        self._hf.close()
        shutil.rmtree(self._hf_dir)

    def _setup(self) -> None:
        # create Gaussian distribution for
        #   sampling latent vectors
        self._gaussian = torch.distributions.Normal(0, 1)
        # Load encoder
        #   if in 'test' mode, don't try to load a checkpoint
        if hasattr(self, 'TEST'):
            vnet_encoder = VnetTrainingModule(
                            self.config).net.encoder
        # else load encoder checkpoint
        else:
            vnet_encoder = \
                VnetTrainingModule.load_from_checkpoint(
                self.config.VNET_CKPT, config=self.config).net.encoder
        vnet_encoder.eval()
        # load HDF5
        self._hf = h5py.File(self.config.DATA, 'r')
        # generate tmp encoded dataset
        self._generate_encoded_dataset(vnet_encoder)

    @torch.no_grad()
    def _generate_encoded_dataset(self, vnet_encoder: torch.nn.Module) -> None:
        # Create tmp directory for Z distributions
        z_hf_path = Path('/tmp') / \
            (str(Path(self.config.DATA).stem)
                + '_Z_distributions')
        if z_hf_path.exists():
            shutil.rmtree(z_hf_path)
        z_hf_path.mkdir()
        # Create tmp dataset for Z distributions
        z_hf = h5py.File(z_hf_path / \
            (z_hf_path.stem + '.hdf5'), 'a')
        self._len = 0
        self._idx_dict = {} # {(start_idx, end_idx): episode #}
        # for each episode...
        for eps, ds in tqdm(self._hf.items(),
                desc='Generating observation encodings'):
            eps_st = self._len
            group = z_hf.create_group(eps)
            means, sigmas = [], []
            # for each timestep..
            for obs in ds['observations']:
                # preprocess frame for encoder
                obs = VnetDataset.preprocess(obs)
                obs = obs.unsqueeze(0)
                # calculate mu & sigma from V-Net encoder
                _, mu, sigma = vnet_encoder(obs)
                mu, sigma = mu.squeeze(0), sigma.squeeze(0)
                # add to lists
                means.append(mu.cpu().numpy())
                sigmas.append(sigma.cpu().numpy())
            # convert lists to arrays
            means = np.array(means, dtype=np.float32)
            sigmas = np.array(sigmas, dtype=np.float32)
            # add to HDF5
            _ = group.create_dataset('latent_means', data=means)
            _ = group.create_dataset('latent_stds', data=sigmas)
            _ = group.create_dataset('actions', data=ds['actions'])
            # update length of dataset.
            #   (len - 1) b/c last Z has no next Z
            self._len += len(ds['observations']) - 1
            # record index of episode end
            eps_end = self._len
            # record mapping from indicies to episode number
            self._idx_dict[(eps_st, eps_end)] = int(eps)
        # close dataset with images
        self._hf.close()
        # update dataset to Z distribution dataset
        self._hf = z_hf
        self._hf_dir = z_hf_path

# Data Module
class DataModule(pl.LightningDataModule):

    """
    handles loading appropriate dataset
        for current training task.
    """

    def __init__(self, config: MasterConfig):
        super().__init__()
        self.config = config

    def prepare_data(self) -> None:
        pass
        
    def setup(self, stage: Optional[str] = None) -> None:
        # Get correct dataset
        exp_type = self.config.EXPERIMENT_TYPE
        if exp_type == 'V-Net':
            dataset = VnetDataset(self.config)
        elif exp_type == 'M-Net':
            dataset = MnetDataset(self.config)
        else:
            raise SyntaxError
        # Split dataset
        (self.train_ds,
         self.val_ds) = self._train_val_split(dataset)

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.train_ds, batch_size=self.config.BATCH_SIZE,
            shuffle=True, num_workers=self.config.N_WORKERS)

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.val_ds, batch_size=self.config.BATCH_SIZE,
            shuffle=True, num_workers=self.config.N_WORKERS)
    
    def _train_val_split(self, dataset: torch.utils.data.Dataset
                         ) -> Tuple[torch.utils.data.Dataset]:
        generator = torch.Generator().manual_seed(
                                self.config.RNG_SEED)
        p = self.config.TRAIN_VAL_SPLIT
        train_ds, val_ds = torch.utils.data.random_split(
                                dataset, [p, 1-p], generator)
        return train_ds, val_ds
