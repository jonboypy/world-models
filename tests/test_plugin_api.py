# Imports
import unittest
from typing import List, Union, Dict, Any
from master_config import MasterConfig
from environments import EnvironmentBase
from plugins import PluginBase


# Helpers
class FakeEnvPlugin(PluginBase):

    def __init__(self):
        super().__init__()

    def pre_reset(self, *args, **kwargs) -> \
                Union[Dict[str, Any], None]:
        kwargs['test'] += 1
        return kwargs

    def post_step(self, output: Any) -> Any:
        output += 1
        return output

class FakeEnv(EnvironmentBase):

    def __init__(self, config: MasterConfig = None, 
            plugins: List[PluginBase] = None) -> None:
        super().__init__(config, plugins)
    
    @EnvironmentBase.hookable
    def reset(self, *args, **kwargs) -> None:
        return kwargs['test']

    @EnvironmentBase.hookable
    def step(self, *args, **kwargs) -> None:
        return kwargs['test']


# Tests
class TestPluginAPI(unittest.TestCase):

    def setUp(self) -> None:
        self.plugins = [FakeEnvPlugin()]
        self.env = FakeEnv(plugins=self.plugins)

    def test_pre_hook(self) -> None:
        self.assertEqual(self.env.reset(test=-1), 0)

    def test_post_hook(self) -> None:
        self.assertEqual(self.env.step(), 0)


class TestMultiPluginAPI(unittest.TestCase):

    def setUp(self) -> None:
        self.plugins = [FakeEnvPlugin(), FakeEnvPlugin()]
        self.env = FakeEnv(plugins=self.plugins)

    def test_pre_hook(self) -> None:
        self.assertEqual(self.env.reset(test=-2), 0)

    def test_post_hook(self) -> None:
        self.assertEqual(self.env.step(test=-2), 0)

