# Imports
from pathlib import Path
import yaml


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

class MasterConfig(dotdict):

    @staticmethod
    def from_yaml(config_path: Path):
        with open(config_path) as f:
            config_dict = yaml.load(f, Loader=yaml.FullLoader)
        return MasterConfig(config_dict)