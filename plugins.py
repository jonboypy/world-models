# Imports
from abc import ABC, abstractmethod


class PluginBase(ABC):

    def __init__(self):
        ...



class EnvironmentPlugin(PluginBase):


    def __init__(self) -> None:
        super().__init__()

    
    def pre_reset(self):
        print('calling from plugin pre hook!')

    def post_reset(self):
        print('calling from plugin pre hook!')


