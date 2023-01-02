# Imports
from abc import ABC
from utils import MasterConfig
from environments import GymEnvironment
from plugins import EnvDataRecorder


class Runner(ABC):

    def __init__(self, config: MasterConfig) -> None:
        super().__init__()
        self.config = config


class DataCollector(Runner):

    def __init__(self, config: MasterConfig) -> None:
        super().__init__(config)
        data_recorder = EnvDataRecorder()
        plugins = [data_recorder]
        self.env = GymEnvironment(config, plugins)

    def collect(self) -> None:
        ...





if __name__ == '__main__':
    runner = DataCollector()

