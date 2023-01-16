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
        z = self.net.encoder(x)
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
        config.LSTM_CELL_ST = True
        self.N_z = config.Z_SIZE
        self.N_h = config.HX_SIZE
        self.N_a = config.ACTION_SPACE_SIZE
        self.net = Mnet(config)

    def test_LSTM(self) -> None:
        z0 = torch.rand(1, self.N_z)
        a0 = torch.rand(1, self.N_a)
        h0 = torch.rand(1, self.N_h)
        c0 = torch.rand(1, self.N_h)
        za = torch.cat([z0, a0], -1)
        h1, c1 = self.net.lstm(za, h0, c0)
        self.assertEqual(h0.size(), h1.size())
        self.assertEqual(c0.size(), c1.size())

    def test_MDN(self) -> None:
        z = torch.rand(1, self.N_z)
        h = torch.rand(1, self.N_h)
        c = torch.rand(1, self.N_h)
        zh = torch.cat([z, h, c], -1)
        mix = self.net.mdn(zh)
        self.assertIsInstance(
            mix,
            torch.distributions.MixtureSameFamily)
        z_next = mix.sample(z.size()).flatten(1)
        self.assertEqual(z.size(), z_next.size())

    def test_full_network(self) -> None:
        z0 = torch.rand(1, self.N_z)
        a0 = torch.rand(1, self.N_a)
        h0 = torch.rand(1, self.N_h)
        c0 = torch.rand(1, self.N_h)
        mix, h1, c1 = self.net(z0, h0, c0, a0)
        self.assertEqual(h0.size(), h1.size())
        self.assertEqual(c0.size(), c1.size())
        self.assertIsInstance(
            mix,
            torch.distributions.MixtureSameFamily)
        z_next = mix.sample(z0.size()).flatten(1)
        self.assertEqual(z0.size(), z_next.size())


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
