# Imports
import functools
from abc import ABC, abstractmethod
from typing import List, NewType
import gym
import numpy as np
from plugins import PluginBase
from master_config import MasterConfig


# Type
Environment = NewType('Environment', object)

# Environments
class EnvironmentBase(ABC):

    def __init__(self, config: MasterConfig=None,
                plugins: List[PluginBase]=None) -> None:
        self.config = config
        self.plugins = plugins
    
    @abstractmethod
    def reset(self, *args, **kwargs) -> None:
        raise NotImplementedError()

    @abstractmethod
    def step(self, *args, **kwargs) -> None:
        raise NotImplementedError()


class GymEnvironment(EnvironmentBase):

    def __init__(self, config: MasterConfig=None,
            plugins: List[PluginBase]=None) -> None:
        super().__init__(config, plugins)
        self.gym = gym.make(config.ENV_NAME)
        self.reset()

    @PluginBase.hookable
    def reset(self) -> np.ndarray:
        """
        Resets environment to start a new episode.

        returns:
            The initial observation.
        """
        return self.gym.reset()


    @PluginBase.hookable
    def step(self, action: np.ndarray) -> np.ndarray:
        """
        Takes a step in the environment.
        
        returns:
            The observation, reward, and
            done-boolean after taking step.
        """
        return self.gym.step(action)


