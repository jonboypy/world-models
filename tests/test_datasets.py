# Imports
import unittest
import torch
from datasets import VnetDataset, MnetDataset, CnetDataset
from utils import MasterConfig


# Tests
class TestVnetDataset(unittest.TestCase):

    def setUp(self) -> None:
        self.config = MasterConfig.from_yaml('./config.yml')
        self.dataset = VnetDataset(self.config)

    def test_getitem(self) -> None:
        img = self.dataset[0]
        self.assertIsInstance(img, torch.Tensor)


class TestMnetDataset(unittest.TestCase):

    def setUp(self) -> None:
        self.config = MasterConfig.from_yaml('./config.yml')
        self.dataset = VnetDataset(self.config)
    
    #TODO


class TestCnetDataset(unittest.TestCase):

    def setUp(self) -> None:
        self.config = MasterConfig.from_yaml('./config.yml')
        self.dataset = VnetDataset(self.config)
    
    #TODO
