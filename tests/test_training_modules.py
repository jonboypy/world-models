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
            './tests/master-config.yml')
        self.training_module = VnetTrainingModule(self.config)

    def test_loss_function(self) -> None:
        func = self.training_module.loss_function
        img = torch.rand(1, 3, 64, 64)
        mu = torch.zeros(1, self.config.Z_SIZE)
        sigma = torch.ones(1, self.config.Z_SIZE)
        loss = func(img, img, mu, sigma)
        self.assertAlmostEquals(loss.item(), 0.0, 1)
    
    def test_training_step(self) -> None:
        image = torch.rand(1,3,64,64)
        loss = self.training_module.training_step(image)

    def test_validation_step(self) -> None:
        image = torch.rand(1,3,64,64)
        loss = self.training_module.validation_step(image)

class TestMnetTrainingModule(unittest.TestCase):

    def setUp(self) -> None:
        self.config = MasterConfig.from_yaml(
            './tests/master-config.yml')
        self.training_module = MnetTrainingModule(self.config)

    def test_loss_function(self) -> None:
        pi = torch.distributions.Categorical(
            torch.softmax(torch.rand(1,5), -1))
        mu = torch.rand(1,1,5,16)
        sigma = torch.rand_like(mu)
        gaussians = torch.distributions.Independent(
                torch.distributions.Normal(mu, sigma), 1)
        z_next_est_dist = torch.distributions.MixtureSameFamily(
                                                    pi, gaussians)
        loss = self.training_module.loss_function(
                    z_next_est_dist, torch.rand(1,16))
        
    def test_training_step(self) -> None:
        z_prev = torch.rand(1,1,self.config.Z_SIZE)
        a_prev = torch.rand(1,1,3)
        z_next = torch.rand_like(z_prev)
        batch = (z_prev, a_prev, z_next)
        loss = self.training_module.training_step(batch)

    def test_validation_step(self) -> None:
        z_prev = torch.rand(1,1,self.config.Z_SIZE)
        a_prev = torch.rand(1,1,3)
        z_next = torch.rand_like(z_prev)
        batch = (z_prev, a_prev, z_next)
        loss = self.training_module.validation_step(batch)

class TestCnetTrainingModule(unittest.TestCase):

    def setUp(self) -> None:
        self.config = MasterConfig.from_yaml(
            './tests/master-config.yml')
        CnetTrainingModule.TEST = True
        self.train_module = CnetTrainingModule(self.config)
        #ray.init(local_mode=True)

    def test_evaluate_individual(self) -> None:
        params = torch.rand(self.train_module.event_space)
        avg_return = self.train_module._evaluate_individual.remote(
                                            self.train_module, params)
        avg_return = ray.get(avg_return)

    def test_evaluate_population(self) -> None:
        population = torch.rand(self.train_module.pop_size,
                                self.train_module.event_space)
        scores = self.train_module._evaluate_population(population)


