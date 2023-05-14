# Imports
import unittest
from agents import RandomGymAgent
from environments import GymEnvironment
from utils import MasterConfig


class TestRandomGymAgent(unittest.TestCase):

    def setUp(self) -> None:
        self.config = MasterConfig.from_yaml(
            './tests/master-config.yml')
        env = GymEnvironment(self.config)
        self.agent = RandomGymAgent(env)

    def test_act(self) -> None:
        prev_st = self.agent.state
        self.agent.act()
        cur_st = self.agent.state
        self.assertGreater(abs((prev_st - cur_st)).sum(), 1.)
