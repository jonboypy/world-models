# Imports
from typing import Tuple
from abc import ABC, abstractmethod
import numpy as np
from environments import Environment


class Agent(ABC):

    def __init__(self, env: Environment) -> None:
        super().__init__()
        self.env = env
        self.eps_cum_reward = 0.
        self.avg_cum_reward = 0.
        self.state = self.env.reset()
 
    def act(self) -> Tuple[np.ndarray]:
        action = self.policy(self.state)
        obs, reward, done, _ = self.env.step(action)
        self.state = obs
        self.eps_cum_reward += reward
        if done:
            self.state = self.env.reset()
            self.avg_cum_reward = (self.avg_cum_reward +
                                    self.eps_cum_reward) / 2
            self.eps_cum_reward = 0.
      
    @abstractmethod
    def policy(self, state: np.ndarray) -> np.ndarray:
        raise NotImplementedError()


class RandomGymAgent(Agent):

    def __init__(self, env: Environment) -> None:
        super().__init__(env)

    def policy(self, state: np.ndarray) -> np.ndarray:
        action = self.env.gym.action_space.sample()
        return action