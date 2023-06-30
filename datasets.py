#Imports
from abc import abstractmethod
from pathlib import Path
from typing import Tuple, Optional, Union
from tqdm import tqdm
import torch
import numpy as np
import pytorch_lightning as pl
import h5py
from utils import MasterConfig
from training_modules import VnetTrainingModule

#####################################################################
#   Datasets
#####################################################################

class Dataset(torch.utils.data.Dataset):
    """
    Base class which all other training
    datasets are derived from.

    Args:
        cfg: A MasterConfig object.
    """

    def __init__(self, cfg: MasterConfig) -> None:
        super().__init__()
        self.cfg = cfg

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError() 

    @abstractmethod
    def __getitem__(self) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
        raise NotImplementedError()

class VnetDataset(Dataset):

    def __init__(self, cfg: MasterConfig) -> None:
        super().__init__(cfg)
        self._setup()
    
    def _setup(self) -> None:
        # load HDF5
        self._hf = h5py.File(self.cfg.DATA, 'r')
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
        img = self.preprocess(img)
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
    
    def __init__(self, cfg: MasterConfig) -> None:
        super().__init__(cfg)
        self._setup()

    def __len__(self) -> int:
        return len(self._hf)

    def __getitem__(self, idx: int) -> torch.Tensor:
        # Index corresponds to episode number
        eps = idx
        # get data from dataset
        means = np.array(self._hf[str(eps)]['latent_means'])
        logvars = np.array(self._hf[str(eps)]['latent_logvars'])
        actions = np.array(self._hf[str(eps)]['actions'])
        # preprocess data
        (means, logvars, actions) = self.preprocess(means, logvars, actions)
        # sample latent vectors from distribution parameters
        #   *this helps prevent overfitting*
        z_vecs = means + torch.exp(logvars * 0.5) * torch.randn_like(logvars)
        z_prev = z_vecs[:-1]
        a_prev = actions[:-1]
        z_next = z_vecs[1:]
        return z_prev, a_prev, z_next

    @classmethod
    def preprocess(cls, means: np.ndarray, logvars: np.ndarray,
                   actions: np.ndarray) -> Tuple[torch.Tensor]:
        # convert all to torch tensors
        means = torch.tensor(means, dtype=torch.float32)
        logvars = torch.tensor(logvars, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.float32)
        return means, logvars, actions

    def __del__(self) -> None:
        self._hf.close()

    def _setup(self) -> None:
        # load HDF5
        self._hf = h5py.File(self.cfg.DATA, 'r')
        if self.cfg.CREATE_LATENT_DATASET:
            # Load encoder
            #   if in 'test' mode, don't try to load a checkpoint
            if hasattr(self, 'UNITTEST'):
                vnet_encoder = VnetTrainingModule(
                                self.cfg).net.encoder
            # else load encoder checkpoint
            else:
                vnet_encoder = \
                    VnetTrainingModule.load_from_checkpoint(
                    self.cfg.VNET_CKPT, cfg=self.cfg).net.encoder
            vnet_encoder.eval()
            # generate encoded dataset
            self._generate_encoded_dataset(vnet_encoder)

    @torch.no_grad()
    def _generate_encoded_dataset(self, vnet_encoder: torch.nn.Module) -> None:
        # Get encoder network device
        device = next(vnet_encoder.parameters()).device
        # Create path & filename for dataset
        z_hf_path = Path(self.cfg.DATA).parent / \
            (str(Path(self.cfg.DATA).stem)
                + '_Z-distributions' + '.hdf5')
        if z_hf_path.exists():
            z_hf_path.unlink()
        # Create dataset for Z distributions
        z_hf = h5py.File(z_hf_path, 'a')
        # for each episode...
        for eps, ds in tqdm(self._hf.items(),
                desc='Generating observation encodings'):
            group = z_hf.create_group(eps)
            obs = np.array(ds['observations'])
            # preprocess frame for encoder
            obs = torch.tensor(obs, dtype=torch.float32,
                                device=device)
            # scale all pixels to the range [0,1]
            obs = obs / 255.
            # permute dimensions to (N,C,H,W)
            obs = obs.permute(0,3,1,2)
            # calculate mean & log-variance from V-Net encoder
            _, means, logvars = vnet_encoder(obs)
            # convert to numpy arrays
            means = np.array(means.cpu(), dtype=np.float32)
            logvars = np.array(logvars.cpu(), dtype=np.float32)
            # add to HDF5
            group.create_dataset('latent_means', data=means)
            group.create_dataset('latent_logvars', data=logvars)
            group.create_dataset('actions', data=ds['actions'])
        # close dataset with images
        self._hf.close()
        # re-assign _hf to latent encodings dataset
        self._hf = z_hf

#####################################################################
#   Data-Module
#####################################################################

class DataModule(pl.LightningDataModule):

    """
    handles loading appropriate dataset
        for current training task.
    """

    def __init__(self, cfg: MasterConfig):
        super().__init__()
        self.cfg = cfg

    def prepare_data(self) -> None:
        pass
        
    def setup(self, stage: Optional[str] = None) -> None:
        # Get correct dataset
        exp_type = self.cfg.RUN_TYPE
        if exp_type == 'V-NET':
            dataset = VnetDataset(self.cfg)
        elif exp_type == 'M-NET':
            dataset = MnetDataset(self.cfg)
        else:
            raise SyntaxError
        # Split dataset
        (self.train_ds,
         self.val_ds) = self._train_val_split(dataset)

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.train_ds, batch_size=self.cfg.BATCH_SIZE,
            shuffle=True, num_workers=self.cfg.N_WORKERS)

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.val_ds, batch_size=self.cfg.BATCH_SIZE,
            shuffle=False, num_workers=self.cfg.N_WORKERS)
    
    def _train_val_split(self, dataset: torch.utils.data.Dataset
                         ) -> Tuple[torch.utils.data.Dataset]:
        generator = torch.Generator().manual_seed(
                                self.cfg.RNG_SEED)
        p = self.cfg.TRAIN_VAL_SPLIT
        train_ds, val_ds = torch.utils.data.random_split(
                                dataset, [p, 1-p], generator)
        return train_ds, val_ds
