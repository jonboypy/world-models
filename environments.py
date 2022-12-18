# Imports
import functools
from abc import ABC, abstractmethod
from typing import List
from plugins import PluginBase
from master_config import MasterConfig


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
            for plugin in self.plugins:
                if hasattr(plugin, f'pre_{func_name}'):
                    hook = getattr(plugin, f'pre_{func_name}')
                    prehook_return = hook(*args, **kwargs)
                    if prehook_return: kwargs = prehook_return
            output = func(self, *args, **kwargs)
            for plugin in self.plugins:
                if hasattr(plugin, f'post_{func_name}'):
                    hook = getattr(plugin, f'post_{func_name}')
                    output = hook(output)
            return output
        return hooked



class GymEnvironment(EnvironmentBase):

    def __init__(self, config: MasterConfig,
            plugins: List[PluginBase]) -> None:
        super().__init__(config, plugins)

    @EnvironmentBase.hookable
    def reset(self, *args, **kwargs) -> None:
        print('resetting')
    
    @EnvironmentBase.hookable
    def step(self, *args, **kwargs) -> None:
        print('stepping')

