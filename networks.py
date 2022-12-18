# Imports
import torch
from master_config import MasterConfig




# Networks
class NetworkBase(torch.nn.Module):

    """
    Base class which all other networks derived from.

    Args:
        config: A MasterConfig object.
    """

    def __init__(self, config: MasterConfig=None) -> None:
        super().__init__()
        self.config = config


class Vnet(NetworkBase):

    """
    V 'vision' network as described in https://arxiv.org/pdf/1809.01999.pdf
    Is a VAE architecture mapping: image -> Z_t -> reconstructed image
    """

    def __init__(self, config: MasterConfig = None) -> None:
        super().__init__(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...


class Mnet(NetworkBase):
    """
    M 'memory' network as described in https://arxiv.org/pdf/1809.01999.pdf
    Is a mixed-density RNN architecture mapping: Z_t, h_t, a_t -> P(Z_{t+1})
    """

    def __init__(self, config: MasterConfig = None) -> None:
        super().__init__(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...


class Cnet(NetworkBase):
    """
    C 'controller' network as described in https://arxiv.org/pdf/1809.01999.pdf
    Is a simple fully-connected network mapping: (Z_t, P(Z_{t+1})) -> a_{t+1}
    """

    def __init__(self, config: MasterConfig = None) -> None:
        super().__init__(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...

