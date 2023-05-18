# Imports
from pathlib import Path
import yaml


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError as e:
            raise AttributeError from e
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class MasterConfig(dotdict):

    REQUIRED_ATTR = [
        "ENV_NAME",
        "Z_SIZE",
    ]

    @staticmethod
    def from_yaml(config_path: Path):
        with open(config_path) as f:
            config_dict = yaml.load(f, Loader=yaml.FullLoader)
        recursive_dotdictify(config_dict)
        cfg = MasterConfig(config_dict)
        cfg.check_config()
        return cfg

    def check_config(self) -> None:
        for attr in self.REQUIRED_ATTR:
            if not hasattr(self, attr):
                raise ConfigurationError(
                    f"configuration must contain {attr}")

def recursive_dotdictify(d):
    if isinstance(d, dict):
        for k,v in d.items():
            v = recursive_dotdictify(v)
            d[k] = v
        return dotdict(d)
    else:
        return d

class ConfigurationError(Exception):
    pass
