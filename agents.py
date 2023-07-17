# Imports
from typing import List
from abc import ABC, abstractmethod
import numpy as np
from PIL import Image
import torch
from environments import Environment
from plugins import Plugin

# Agents
class Agent(ABC):
    """
    Abstract base class which all other agents derive from.

    Args:
        env: An instance of an environment.
        plugins: A list of plugins. Defaults to None.
    """

    def __init__(self, env: Environment,
                 plugins: List[Plugin] = None,
                 force_full_episode: bool = False) -> None:
        super().__init__()
        self.env = env
        self.plugins = plugins
        self.eps_cum_reward = 0.
        self.avg_cum_reward = 0.
        self.eps_steps = 0
        self.state, _ = self.env.reset()
        self.force_full_episode = force_full_episode

    def act(self) -> np.ndarray:
        action = self.policy(self.state)
        obs, reward, term, trunc, _ = self.env.step(action)
        self.eps_steps += 1
        self.state = obs
        self.eps_cum_reward += reward
        if self.force_full_episode:
            done = (self.eps_steps == 1000)
        else:
            done = max(term, trunc)
        if done:
            self.state, _ = self.env.reset()
            self.avg_cum_reward = (self.avg_cum_reward +
                                   self.eps_cum_reward) / 2
            self.eps_cum_reward = 0.
            self.eps_steps = 0
        return done

    @abstractmethod
    def policy(self, state: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

class RandomGymAgent(Agent):
    """
    An agent for Gym environments that
    follows a random policy.
    """

    def __init__(self, env: Environment,
                 plugins: List[Plugin] = None,
                 force_full_episode: bool = False) -> None:
        super().__init__(env, plugins, force_full_episode)

    @Plugin.hookable
    def policy(self, state: np.ndarray) -> np.ndarray:
        action = self.env.gym.action_space.sample()
        return action

class WorldModelGymAgent(Agent):
    """
    An agent for Gym environments that
    utilizes the world models.
    
    Args:
        vnet_encoder: The encoder of the vision network.
        mnet: The memory network.
        cnet: The controller network.
    """

    def __init__(self, env: Environment,
                 vnet_encoder: torch.nn.Module,
                 mnet: torch.nn.Module,
                 cnet: torch.nn.Module,
                 plugins: List[Plugin] = None,
                 force_full_episode: bool = False) -> None:
        super().__init__(env, plugins, force_full_episode)
        self.vnet_encoder = vnet_encoder
        self.mnet = mnet
        self.cnet = cnet
        if env.gym.unwrapped.spec.id == "CarRacing-v2":
            self._preprocess_obs = self._preprocess_car_racing_obs
        else: raise NotImplementedError(
                'WorldModelGymAgent does not currently support '
                f'{env.gym.unwrapped.spec.id}.')
        self.cnet_device = next(cnet.parameters()).device
        self.world_model_device = next(vnet_encoder.parameters()).device
        assert(self.world_model_device ==
            next(mnet.parameters()).device)
        # Create initial hidden vector
        self.hx = (torch.zeros(1, 1, self.mnet.cfg.HX_SIZE,
                               device=self.world_model_device),
                    torch.zeros(1, 1, self.mnet.cfg.HX_SIZE,
                                device=self.world_model_device))
        # Create initial previous action
        self.a_prev = torch.zeros(
            1, 1, self.mnet.cfg.ACTION_SPACE_SIZE,
            device=self.world_model_device)
    
    @torch.no_grad()
    @Plugin.hookable
    def policy(self, obs: np.ndarray) -> np.ndarray:
        # preprocess observaion
        obs = self._preprocess_obs(obs)
        # Encoder observation to latent vector
        z, _, _= self.vnet_encoder(obs.unsqueeze(0))
        # Integrate latent vector with memory
        _, h, self.hx = self.mnet.forward_with_hx(
                        z.view(1, 1, -1),
                        self.hx, self.a_prev)
        # Concatenate z & h
        zh = torch.cat([z.view(1, 1, -1), h], -1).view(1, -1)
        # Move to C-Net device
        zh = zh.to(self.cnet_device)
        # Feed the latent vector and hidden vector
        #   to the controller network
        action = self.cnet(zh)
        # update previous action
        self.a_prev = action.view(
            1, 1, -1).to(self.world_model_device)
        return action.view(-1).cpu().numpy()
    
    def _preprocess_car_racing_obs(self, obs: np.ndarray) -> torch.Tensor:
        # Crop observation
        obs = Image.fromarray(obs)
        obs = obs.crop((0,0,96,83))
        # Resize observaion
        obs = obs.resize((64, 64))
        obs = np.array(obs)
        # convert numpy uint8 arr to torch 32-bit float tensor
        obs = torch.tensor(obs, dtype=torch.float32,
                           device=self.world_model_device)
        # scale all pixels to the range [0,1]
        obs = obs / 255.
        # permute dims to Torch standard (C,H,W)
        obs = obs.permute(2, 0, 1)
        return obs
