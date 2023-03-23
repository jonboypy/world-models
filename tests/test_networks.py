# Imports
import unittest
import torch
from networks import Vnet, Mnet, Cnet
from utils import MasterConfig


# Tests
class TestVnet(unittest.TestCase):

    def setUp(self) -> None:
        self.config = MasterConfig.from_yaml('./config.yml')
        self.net = Vnet(self.config)

    def test_encoder_network(self) -> None:
        N_z = self.config.Z_SIZE
        x = torch.rand(1, 3, 64, 64)
        z, _, _ = self.net.encoder(x)
        self.assertEqual(
            z.size(),
            torch.Size([1, N_z]))

    def test_decoder_network(self) -> None:
        N_z = self.config.Z_SIZE
        z = torch.rand(1, N_z)
        y = self.net.decoder(z)
        self.assertEqual(
            y.size(),
            torch.Size([1, 3, 64, 64]))
        self.assertTrue((y >= 0).all())
        self.assertTrue((y <= 1).all())

    def test_full_network(self) -> None:
        N_z = self.config.Z_SIZE
        x = torch.rand(1, 3, 64, 64)
        y, z = self.net(x)
        self.assertEqual(
            z.size(),
            torch.Size([1, N_z]))
        self.assertEqual(y.size(), x.size())
        self.assertTrue((y >= 0).all())
        self.assertTrue((y <= 1).all())


class TestMnet(unittest.TestCase):

    def setUp(self) -> None:
        config = MasterConfig.from_yaml('./config.yml')
        self.N_z = config.Z_SIZE
        self.N_h = config.HX_SIZE
        self.N_a = config.ACTION_SPACE_SIZE
        self.net = Mnet(config)

    def test_LSTM(self) -> None:
        z_prev = torch.rand(1, 1, self.N_z)
        a_prev = torch.rand(1, 1, self.N_a)
        za = torch.cat([z_prev, a_prev], -1)
        h_next, _ = self.net.lstm(za)
        self.assertEqual(h_next.size(-1), self.N_h)

    def test_MDN(self) -> None:
        z = torch.rand(1, 1, self.N_z)
        h = torch.rand(1, 1, self.N_h)
        mix = self.net.mdn(h)
        self.assertIsInstance(
            mix,
            torch.distributions.MixtureSameFamily)
        z_next = mix.sample()
        self.assertEqual(z.size(), z_next.size())

    def test_full_network(self) -> None:
        # dims: (S,B,Z), (S,B,A)
        z_prev = torch.rand(1, 1, self.N_z)
        a_prev = torch.rand(1, 1, self.N_a)
        z_next_est_dist, h_next = self.net(z_prev, a_prev)
        self.assertIsInstance(
            z_next_est_dist,
            torch.distributions.MixtureSameFamily)
        z_next = z_next_est_dist.sample()
        self.assertEqual(z_prev.size(), z_next.size())
        self.assertEqual(h_next.size(-1), self.N_h)


class TestCnet(unittest.TestCase):

    def setUp(self) -> None:
        self.config = MasterConfig.from_yaml('./config.yml')
        self.net = Cnet(self.config)
        self.N_z = self.config.Z_SIZE
        self.N_h = self.config.HX_SIZE

    def test_forward(self) -> None:
        x = torch.rand(1, self.N_z + self.N_h)
        y = self.net(x)
        self.assertEqual(y.size(-1), 3)
        self.assertTrue((y >= -1.).all())
        self.assertTrue((y <= 1.).all())
