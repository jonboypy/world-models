# Imports
import functools
from abc import ABC, abstractmethod
from typing import List
import gym
import numpy as np
from plugins import PluginBase
from master_config import MasterConfig


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

    @staticmethod 
    def hookable(func):
        @functools.wraps(func)
        def hooked(self, *args, **kwargs):
            func_name = func.__name__
            if self.plugins:
                for plugin in self.plugins:
                    if hasattr(plugin,
                    f'pre_{func_name}'):
                        hook = getattr(
                            plugin, f'pre_{func_name}')
                        prehook_return = hook(*args, **kwargs)
                        if prehook_return:
                            kwargs = prehook_return
                output = func(
                    self, *args, **kwargs)
                for plugin in self.plugins:
                    if hasattr(
                        plugin, f'post_{func_name}'):
                        hook = getattr(plugin, f'post_{func_name}')
                        output = hook(output)
            else: output = func(self, *args, **kwargs)
            return output
        return hooked


class GymEnvironment(EnvironmentBase):

    def __init__(self, config: MasterConfig=None,
            plugins: List[PluginBase]=None) -> None:
        super().__init__(config, plugins)
        self.gym = gym.make(config.ENV_NAME)
        self.reset()

    @EnvironmentBase.hookable
    def reset(self) -> np.ndarray:
        """
        Resets environment to start a new episode.

        returns:
            The initial observation.
        """
        return self.gym.reset()


    @EnvironmentBase.hookable
    def step(self, action: np.ndarray) -> np.ndarray:
        """
        Takes a step in the environment.
        
        returns:
            The observation, reward, and
            done-boolean after taking step.
        """
        return self.gym.step(action)


