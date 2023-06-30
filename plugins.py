# Imports
from pathlib import Path
from typing import Union, Dict, Any
import functools
from abc import ABC
from PIL import Image
import numpy as np
import torch
import h5py
import wandb
from utils import MasterConfig

#####################################################################
#   Plugins
#####################################################################

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
            while func_name[0] == '_':
                func_name = func_name[1:]
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
    Saves single episode of an agent's experiences for training data.
    Plugin must be given to both the environment and agent.

    Args:
        save_dir: Path to directory to save data to.
    """

    def __init__(self, save_dir: Path, name: str, eps_idx: int) -> None:
        super().__init__()
        self.save_dir = save_dir
        self.eps_idx = eps_idx
        self.eps = -1
        self.step = 0
        self.eps_data = {}
        self.name = name

    def __del__(self) -> None:
        if self.eps_data:
            self._save_episode_data()
        self.hf.close()

    def pre_policy(self, state: np.ndarray) -> None:
        self.eps_data[self.step] = state

    def post_policy(self, action: np.ndarray) -> None:
        self.eps_data[self.step] = (
            self.eps_data[self.step],
            action)

    def post_step(self, *args, **kwargs) -> None:
        self.step += 1

    def post_reset(self, *args, **kwargs) -> Any:
        if not self.eps < 0:
            self._save_episode_data()
        self.eps += 1
        self.eps_data = {}
        self.step = 0

    def _save_episode_data(self) -> None:
        self.hf = h5py.File(
            self.save_dir/f'{self.name}-eps={self.eps_idx}.hdf5', 'a') 
        group = self.hf.create_group(str(self.eps_idx))
        imgs = []
        actions = []
        for data in self.eps_data.values():
            obs, action = data
            img = self._preprocess_observation(obs)
            img = np.array(img, dtype=np.uint8)
            imgs.append(img)
            actions.append(action)
        imgs = np.array(imgs)
        actions = np.array(actions)
        group.create_dataset('observations', data=imgs)
        group.create_dataset('actions', data=actions)
        self.hf.close()

    def _preprocess_observation(self, obs: np.ndarray) -> Image.Image:
        img = Image.fromarray(obs)
        img = img.crop((0,0,96,83))
        img = img.resize((64, 64))
        return img

class CNetWandbLogger(Plugin):

    """
    Plugin to log C-Net training metrics to WandB.
    """

    def __init__(self, cfg: MasterConfig) -> None:
        super().__init__()
        self.run = wandb.init(
            project="World-Models",
            config=cfg,
            dir=cfg.EXPERIMENT_DIR,
            name=(f'{cfg.RUN_TYPE}/'
                  f'{cfg.EXPERIMENT_NAME}')
            )

    def pre_log(self, name: str, metric: Any,
                plot_type: str, step: int) -> None:
        if plot_type == 'line':
            wandb.log({name: metric}, step)
        elif plot_type == 'histogram':
            wandb.log({name: wandb.Histogram(metric)}, step)
        elif plot_type == 'video':
            wandb.log(
                {name: wandb.Video(
                    metric, fps=30, format='mp4')}, step)
        else:
            raise NotImplementedError(
                f'Plot type {plot_type} not supported.')

class CNetModelCheckpoint(Plugin):

    """
    Plugin to save C-Net models.
    """

    def __init__(self, save_dir: Path) -> None:
        super().__init__()
        if not save_dir.exists():
            save_dir.mkdir(parents=True)
        self.save_dir = str(save_dir)
        self.previous_best = None
        self.previous_ckpt = None

    def pre_save(self, model: torch.nn.Module,
                 filename: str, metric: float) -> None:
        if self.previous_best is None:
            self.previous_best = metric
            self.previous_ckpt = self.save_dir+'/'+filename
            torch.save({'model_state_dict': model.state_dict()},
                       self.previous_ckpt)
        elif self.previous_best < metric:
            Path(self.previous_ckpt).unlink()
            self.previous_best = metric
            self.previous_ckpt = self.save_dir+'/'+filename
            torch.save({'model_state_dict': model.state_dict()},
                       self.previous_ckpt)
