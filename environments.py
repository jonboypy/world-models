# Imports
from abc import ABC, abstractmethod
from typing import List
import gymnasium as gym
import numpy as np
from plugins import Plugin
from utils import MasterConfig

# Environments
class Environment(ABC):
    """
    Abstract base class which all other environments derive from.

    Args:
        cfg: a master cfg object.
        plugins: A list of plugins
    """

    def __init__(self, cfg: MasterConfig = None,
                 plugins: List[Plugin] = None) -> None:
        self.cfg = cfg
        self.plugins = plugins

    @abstractmethod
    def reset(self, *args, **kwargs) -> None:
        raise NotImplementedError()

    @abstractmethod
    def step(self, *args, **kwargs) -> None:
        raise NotImplementedError()

class GymEnvironment(Environment):
    """
    Wrapper around OpenAI's Gym environments.
    """

    def __init__(self, cfg: MasterConfig = None,
                 plugins: List[Plugin] = None,
                 record_video: bool = False,
                 video_name: str = None) -> None:
        super().__init__(cfg, plugins)
        if record_video:
            self.gym = gym.make(cfg.ENV_NAME,
                                render_mode="rgb_array")
            self.gym = gym.wrappers.RecordVideo(
                self.gym, self.cfg.EXPERIMENT_DIR,
                episode_trigger=\
                    lambda e: e < self.cfg.TEST_N_ROLLOUTS,
                name_prefix=video_name,
                disable_logger=True)
        else:
            self.gym = gym.make(cfg.ENV_NAME)

    def __del__(self) -> None:
        if hasattr(self, 'gym'):
            self.gym.close()

    @Plugin.hookable
    def reset(self) -> np.ndarray:
        """
        Resets environment to start a new episode.

        returns:
            The initial observation.
        """
        return self.gym.reset()

    @Plugin.hookable
    def step(self, action: np.ndarray) -> np.ndarray:
        """
        Takes a step in the environment.

        returns:
            The observation, reward, and
            done-boolean after taking step.
        """
        return self.gym.step(action)
