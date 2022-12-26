# Imports
import unittest
from agents import RandomGymAgent
from environments import GymEnvironment
from master_config import MasterConfig



class TestRandomGymAgent(unittest.TestCase):

    def setUp(self) -> None:
        env = GymEnvironment(MasterConfig)
        self.agent = RandomGymAgent(env)

    def test_act(self) -> None:
        prev_st = self.agent.state
        self.agent.act()
        cur_st = self.agent.state
        self.assertGreater(abs((prev_st - cur_st)).sum(), 1.)