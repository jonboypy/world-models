# Imports
import unittest
import torch
from networks import Vnet, Mnet, Cnet
from master_config import MasterConfig


# Tests
class TestVnet(unittest.TestCase):

    def setUp(self) -> None:
        self.net = Vnet(MasterConfig)

class TestMnet(unittest.TestCase):

    def setUp(self) -> None:
        self.net = Mnet(MasterConfig)

class TestCnet(unittest.TestCase):

    def setUp(self) -> None:
        self.net = Cnet(MasterConfig)

