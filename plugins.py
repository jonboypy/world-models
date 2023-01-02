# Imports
import functools
from abc import ABC
from typing import Union, Dict, Any


class Plugin(ABC):
    """
    Abstract base class which all plugins derive from.

    """

    def __init__(self):
        super().__init__()

    def pre_(self, *args, **kwargs) -> Union[
                        Dict[str, Any], None]:
        """
        pre_<function-name>:
            modify input to a function or perform a 
            task before function is called.
        
        returns:
            Either a Dict with updated kwargs or None.
        """
        raise NotImplementedError()

    def post_(self, output: Any) -> Any:
        """
        post_<function-name>:
            modify output of a function or perform a 
            task after function is called.
        
        returns:
            Any.
        """
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


class EnvDataRecorder(Plugin):

    """
    Saves environment data to files.
    """

    def __init__(self) -> None:
        super().__init__()
        self.eps = 0
        self.step = 0
        self.eps_data = {}

    def post_step(self, output: Any) -> Any:
        self.eps_data[self.step] = output
        self.step += 1
        return output

    def post_reset(self, output: Any) -> Any:
        self._save_episode_data()
        self.eps += 1
        self.eps_data = {}
        self.step = 0
        return output
