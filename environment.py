# Imports
from abc import ABC, abstractmethod



class EnvironmentBase(ABC):

    def __init__(self, config: MasterConfig,
                plugins: List[PluginBase]) -> None:
        self.config = config
        self.plugins = plugins
    
    def hookable(func):
        @functools.wraps(func)
        def hooked(self, *args, **kwargs):
            func_name = func.__name__
            for plugin in self.plugins:
                if hasattr(plugin, f'pre_{func_name}'):
                    hook = getattr(plugin, f'pre_{func_name}')
                    prehook_return = hook(*args, **kwargs)
            value = func(self, *args, **kwargs)
            for plugin in self.plugins:
                if hasattr(plugin, f'post_{func_name}'):
                    hook = getattr(plugin, f'post_{func_name}')
                    hook(value)
            return value
        return hooked

    @hookable
    @abstractmethod
    def reset(self, *args, **kwargs) -> None:
        raise NotImplementedError()

    @hookable
    @abstractmethod
    def step(self, *args, **kwargs) -> None:
        raise NotImplementedError()

