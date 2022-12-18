# Imports
from abc import ABC, abstractmethod
from typing import Union, Dict, Any


class PluginBase(ABC):

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




class EnvironmentPlugin(PluginBase):


    def __init__(self) -> None:
        super().__init__()

    
    def pre_reset(self):
        print('calling from plugin pre hook!')

    def post_reset(self):
        print('calling from plugin pre hook!')


