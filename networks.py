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
    
    Args:
        config: MasterConfig object.
    """

    def __init__(self, config: MasterConfig) -> None:
        super().__init__(config)
        self.encoder = self.Encoder(config)
        self.decoder = self.Decoder(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        y = self.decoder(z.clone())
        return y, z


    class Encoder(NetworkBase):
        """
        Encoder for V network. Maps: Image -> Z_t

        Args:
            config: MasterConfig object.
        """
        def __init__(self, config: MasterConfig = None) -> None:
            super().__init__(config)
            N_z = config.Z_SIZE
            self.conv1 = torch.nn.Conv2d(3,32,4,2)
            self.relu1 = torch.nn.ReLU(True)
            self.conv2 = torch.nn.Conv2d(32,64,4,2)
            self.relu2 = torch.nn.ReLU(True)
            self.conv3 = torch.nn.Conv2d(64,128,4,2)
            self.relu3 = torch.nn.ReLU(True)
            self.conv4 = torch.nn.Conv2d(128,256,4,2)
            self.relu4 = torch.nn.ReLU(True)
            self.mu_flatten = torch.nn.Flatten()
            self.fv2mu = torch.nn.Linear(1024, N_z) # fv = 'feature vector'
            self.sigma_flatten = torch.nn.Flatten()
            self.fv2sigma = torch.nn.Linear(1024, N_z)
            self.gaussian = torch.distributions.Normal(0,1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.conv1(x)
            x = self.relu1(x)
            x = self.conv2(x)
            x = self.relu2(x)
            x = self.conv3(x)
            x = self.relu3(x)
            x = self.conv4(x)
            x = self.relu4(x)
            fv = self.mu_flatten(x)
            mu = self.fv2mu(fv)
            fv = self.sigma_flatten(x)
            sigma = self.fv2sigma(fv)
            z = mu + sigma * self.gaussian.sample(sigma.size())
            return z


    class Decoder(NetworkBase):
        """
        Decoder for V network. Maps: Z_t -> Image

        Args:
            config: MasterConfig object.
        """
        def __init__(self, config: MasterConfig = None) -> None:
            super().__init__(config)
            N_z = config.Z_SIZE
            self.linear1 = torch.nn.Linear(N_z, 1024)
            self.unflatten = torch.nn.Unflatten(-1,(1024,1,1))
            self.deconv1 = torch.nn.ConvTranspose2d(1024,128,5,2)
            self.relu1 = torch.nn.ReLU(True)
            self.deconv2 = torch.nn.ConvTranspose2d(128,64,5,2)
            self.relu2 = torch.nn.ReLU(True)
            self.deconv3 = torch.nn.ConvTranspose2d(64,32,6,2)
            self.relu3 = torch.nn.ReLU(True)
            self.deconv4 = torch.nn.ConvTranspose2d(32,3,6,2)
            self.sigmoid1 = torch.nn.Sigmoid()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.linear1(x)
            x = self.unflatten(x)
            x = self.deconv1(x)
            x = self.relu1(x)
            x = self.deconv2(x)
            x = self.relu2(x)
            x = self.deconv3(x)
            x = self.relu3(x)
            x = self.deconv4(x)
            y = self.sigmoid1(x)
            return y


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

