# Imports
from pathlib import Path
from typing import Union, Dict, Any
import shutil
import functools
from abc import ABC
from PIL import Image
import numpy as np
import torch
import h5py
import wandb
from utils import MasterConfig


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
    Saves agent experiences to storage for training data.
    Plugin must be given to both the environment and agent.

    Args:
        save_dir: Path to directory to save data to.
        as_hdf5: If true, save dataset as HDF5.
    """

    def __init__(self, save_dir: Path, name: str,
                 as_hdf5: bool = False) -> None:
        super().__init__()
        self.save_dir = save_dir
        if as_hdf5:
            if (save_dir/f'{name}.hdf5').exists():
                (save_dir/f'{name}.hdf5').unlink()
        else:
            if save_dir.exists():
                shutil.rmtree(save_dir)
                save_dir.mkdir(parents=True)
        self.eps = -1
        self.step = 0
        self.eps_data = {}
        self.as_hdf5 = as_hdf5
        if as_hdf5:
            self.hf = h5py.File(save_dir / 
                        f'{name}.hdf5', 'a') 

    def __del__(self) -> None:
        if self.eps_data:
            self._save_episode_data()
        self.hf.close()

    def pre_policy(self, state: np.ndarray) -> None:
        self.eps_data[self.step] = state

    def post_policy(self, output: Any) -> None:
        self.eps_data[self.step] = (
            self.eps_data[self.step],
            output)

    def post_step(self, output: Any) -> None:
        self.step += 1

    def post_reset(self, output: Any) -> Any:
        if not self.eps < 0:
            self._save_episode_data()
        self.eps += 1
        self.eps_data = {}
        self.step = 0
        return output

    def _save_episode_data(self) -> None:
        if self.as_hdf5:
            self._save_episode_data_hdf5()
        else:
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

    def _save_episode_data_hdf5(self) -> None:
        group = self.hf.create_group(str(self.eps))
        imgs = []
        actions = []
        for _, data in self.eps_data.items():
            obs, action = data
            img = self._preprocess(obs)
            img = np.array(img, dtype=np.uint8)
            imgs.append(img)
            actions.append(action)
        imgs = np.array(imgs)
        actions = np.array(actions)
        _ = group.create_dataset('observations', data=imgs)
        _ = group.create_dataset('actions', data=actions)

    def _preprocess(self, obs: np.ndarray) -> Image.Image:
        img = Image.fromarray(obs)
        img = img.crop((0,0,96,83))
        img = img.resize((64, 64))
        return img

class CNetWandbLogger(Plugin):

    """
    Plugin to log C-Net training metrics to WandB.
    """

    def __init__(self, config: MasterConfig) -> None:
        super().__init__()
        self.run = wandb.init(
            project="World-Models",
            config=config,
            dir=config.EXPERIMENT_DIR,
            name=(f'{config.EXPERIMENT_TYPE}/'
                  f'{config.EXPERIMENT_NAME}')
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
