# Imports
import unittest
import torch
from networks import Vnet, Mnet, Cnet
from master_config import MasterConfig


# Tests
class TestVnet(unittest.TestCase):

    def setUp(self) -> None:
        self.net = Vnet(MasterConfig)

    def test_encoder_network(self) -> None:
        N_z = MasterConfig.Z_SIZE
        x = torch.rand(1,3,64,64)
        z = self.net.encoder(x)
        self.assertEqual(z.size(),
            torch.Size([1,N_z]))

    def test_decoder_network(self) -> None:
        N_z = MasterConfig.Z_SIZE
        z = torch.rand(1,N_z)
        y = self.net.decoder(z)
        self.assertEqual(y.size(),
            torch.Size([1,3,64,64]))
        self.assertTrue((y >= 0).all())
        self.assertTrue((y <= 1).all())

    def test_full_network(self) -> None:
        N_z = MasterConfig.Z_SIZE
        x = torch.rand(1,3,64,64)
        y, z = self.net(x)
        self.assertEqual(z.size(),
            torch.Size([1,N_z]))
        self.assertEqual(y.size(), x.size())
        self.assertTrue((y >= 0).all())
        self.assertTrue((y <= 1).all())


class TestMnet(unittest.TestCase):

    def setUp(self) -> None:
        self.net = Mnet(MasterConfig)

class TestCnet(unittest.TestCase):

    def setUp(self) -> None:
        self.net = Cnet(MasterConfig)

