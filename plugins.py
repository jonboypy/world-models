# Imports
from pathlib import Path
from typing import Union, Dict, Any
import shutil
import functools
from abc import ABC
from PIL import Image
import numpy as np


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
                    if hasattr(plugin, f'pre_{func_name}'):
                        hook = getattr(
                            plugin, f'pre_{func_name}')
                        prehook_return = hook(*args, **kwargs)
                        if prehook_return:
                            kwargs = prehook_return
                output = func(
                    self, *args, **kwargs)
                for plugin in self.plugins:
                    if hasattr(plugin, f'post_{func_name}'):
                        hook = getattr(plugin, f'post_{func_name}')
                        og_output = output
                        output = hook(output)
                        if output is None:
                            output = og_output
            else:
                output = func(self, *args, **kwargs)
            return output
        return hooked


class DataRecorder(Plugin):

    """
    Saves agent experiences to storage for training data.
    Plugin must be given to both the environment and agent.

    Args:
        save_dir: Path to directory to save data to.
    """

    def __init__(self, save_dir: Path) -> None:
        super().__init__()
        self.save_dir = save_dir
        if save_dir.exists():
            shutil.rmtree(save_dir)
        self.eps = -1
        self.step = 0
        self.eps_data = {}

    def __del__(self) -> None:
        if self.eps_data:
            self._save_episode_data()

    def pre_policy(self, state: np.ndarray):
        self.eps_data[self.step] = state

    def post_policy(self, output: Any) -> Any:
        self.eps_data[self.step] = (
            self.eps_data[self.step],
            output)

    def post_step(self, output: Any) -> Any:
        self.step += 1

    def post_reset(self, output: Any) -> Any:
        if not self.eps < 0:
            self._save_episode_data()
        self.eps += 1
        self.eps_data = {}
        self.step = 0
        return output

    def _save_episode_data(self) -> None:
        (self.save_dir / f'episode-{self.eps}').mkdir(
                            parents=True, exist_ok=True)
        eps_dir = self.save_dir / f'episode-{self.eps}'
        actions = []
        for step, data in self.eps_data.items():
            obs, action = data
            actions.append(action)
            img = self._preprocess(obs)
            step = str(step).zfill(len(str(self.step)))
            img.save(eps_dir / f'step-{step}.png')
        actions = np.array(actions)
        np.save(eps_dir / 'actions', actions)

    def _preprocess(self, obs: np.ndarray) -> Image.Image:
        img = Image.fromarray(obs)
        img = img.resize((64, 64))
        return img
