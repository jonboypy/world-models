# Imports
import unittest
import torch
from datasets import VnetDataset, MnetDataset
from utils import MasterConfig


# Tests
class TestVnetDataset(unittest.TestCase):

    def setUp(self) -> None:
        self.config = MasterConfig.from_yaml(
            './tests/master-config.yml')
        self.dataset = VnetDataset(self.config)

    def test_getitem(self) -> None:
        img = self.dataset[0]
        self.assertIsInstance(img, torch.Tensor)


class TestMnetDataset(unittest.TestCase):

    def setUp(self) -> None:
        self.config = MasterConfig.from_yaml(
            './tests/master-config.yml')           
        MnetDataset.UNITTEST = True
        self.dataset = MnetDataset(self.config)
 