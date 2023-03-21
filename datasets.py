#Imports
from abc import abstractmethod
from typing import Tuple, Optional
import torch
import pytorch_lightning as pl
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
    def __getitem__(self) -> Tuple[torch.Tensor]:
        raise NotImplementedError()

class VnetDataset(Dataset):

    def __init__(self, config: MasterConfig) -> None:
        super().__init__(config)
    #TODO: get item returns just image

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
