# Imports
import unittest
import torch
from training_modules import VnetTrainingModule
from utils import MasterConfig


# Tests
class TestVnetTrainingModule(unittest.TestCase):

    def setUp(self) -> None:
        self.config = MasterConfig.from_yaml('./config.yml')
        self.train_module = VnetTrainingModule(self.config)

    def test_loss_function(self) -> None:
        func = self.train_module.loss_function
        img = torch.rand(1, 3, 64, 64)
        latent = torch.randn_like(1, self.config.Z_SIZE)
        loss = func(img, img, latent)
        self.assertEquals(loss, 0.0)
