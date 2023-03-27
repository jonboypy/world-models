# Imports
import unittest
import torch
import ray
from training_modules import (VnetTrainingModule,
            MnetTrainingModule, CnetTrainingModule)
from utils import MasterConfig

# Tests
class TestVnetTrainingModule(unittest.TestCase):

    def setUp(self) -> None:
        self.config = MasterConfig.from_yaml(
            './tests/training-configurations/Vnet-config.yml')
        self.train_module = VnetTrainingModule(self.config)

    def test_loss_function(self) -> None:
        func = self.train_module.loss_function
        img = torch.rand(1, 3, 64, 64)
        latent = torch.randn(1, self.config.Z_SIZE)
        loss = func(img, img, latent)
        self.assertAlmostEquals(loss.item(), 0.0, 1)

class TestMnetTrainingModule(unittest.TestCase):

    def setUp(self) -> None:
        self.config = MasterConfig.from_yaml(
            './tests/training-configurations/Mnet-config.yml')
        self.train_module = MnetTrainingModule(self.config)
    
    #TODO

class TestCnetTrainingModule(unittest.TestCase):

    def setUp(self) -> None:
        self.config = MasterConfig.from_yaml(
            './tests/training-configurations/Cnet-config.yml')
        CnetTrainingModule.TEST = True
        self.train_module = CnetTrainingModule(self.config)

    def test_evaluate_individual(self) -> None:
        params = torch.rand(self.train_module.event_space)
        avg_return = self.train_module._evaluate_individual(params, 2)

    def test_evaluate_population(self) -> None:
        population = torch.rand(self.train_module.pop_size,
                                self.train_module.event_space)
        scores = self.train_module._evaluate_population(population)


