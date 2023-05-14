# Imports
import unittest
from environments import GymEnvironment
from utils import MasterConfig


# Tests
class TestGymEnv(unittest.TestCase):

    def setUp(self) -> None:
        self.config = MasterConfig.from_yaml(
            './tests/master-config.yml')
        self.env = GymEnvironment(self.config)

    def test_setup(self) -> None:
        self.assertTrue(hasattr(self.env, 'gym'))

    def test_reset(self) -> None:
        obs1, _ = self.env.reset()
        obs2, _ = self.env.reset()
        self.assertFalse((obs1 == obs2).all())

    def test_step(self) -> None:
        obs1, _ = self.env.reset()
        obs2, _, _, _, _ = self.env.step(
            self.env.gym.action_space.sample())
        self.assertFalse((obs1 == obs2).all())
