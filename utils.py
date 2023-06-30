# Imports
from pathlib import Path
import yaml

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError as e:
            raise AttributeError(f'{item} does not exist.')
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

class MasterConfig(dotdict):

    REQUIRED_ATTR = [
        "COMMON",
        "DATA_COLLECTION",
        "V-NET",
        "M-NET",
        "C-NET"
    ]

    @staticmethod
    def from_yaml(config_path: Path, section: str=None):
        with open(config_path) as f:
            config_dict = yaml.load(f, Loader=yaml.FullLoader)
        _recursive_dotdictify(config_dict)
        cfg = MasterConfig(config_dict)
        cfg.check_config()
        if section:
            cfg = _get_section_cfg(cfg, section)
        return cfg

    def check_config(self) -> None:
        for attr in self.REQUIRED_ATTR:
            if not hasattr(self, attr):
                raise ConfigurationError(
                    f"configuration must contain {attr}")

def _recursive_dotdictify(d):
    if isinstance(d, dict):
        for k,v in d.items():
            v = _recursive_dotdictify(v)
            d[k] = v
        return dotdict(d)
    else:
        return d
    
def _get_section_cfg(cfg: MasterConfig, section: str) -> MasterConfig:
    common = cfg.COMMON
    setattr(common, 'RUN_TYPE', section)
    section = getattr(cfg, section)
    for attr in section.keys():
        setattr(common, attr, getattr(section, attr))
    return common

class ConfigurationError(Exception):
    pass
