#Imports
from abc import abstractmethod
from typing import Tuple, Optional, Union
import torch
import numpy as np
import pytorch_lightning as pl
import h5py
from utils import MasterConfig

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

    def _preprocess(self, img: np.ndarray) -> torch.Tensor:
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
    #TODO: on init, compute mu, sigma for each frame.
    # getitem samples Z from mu,sigma and returns it along with action and next Z
    #NOTE: sample next Z too? i think so.

class CnetDataset(Dataset):

    def __init__(self, config: MasterConfig) -> None:
        super().__init__(config)

    #TODO: getitem returns z_prev, h_prev

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
        #TODO: optionally run data collector here
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        # TODO: instantiate appropriate dataset here
        #       split, etc.
        pass

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        pass

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        pass
