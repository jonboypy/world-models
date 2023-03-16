# Imports
from typing import Tuple
import torch
from utils import MasterConfig


# Networks
class Network(torch.nn.Module):

    """
    Base class which all other networks derived from.

    Args:
        config: A MasterConfig object.
    """

    def __init__(self, config: MasterConfig = None) -> None:
        super().__init__()
        self.config = config


class Vnet(Network):

    """
    V 'vision' network as described in https://arxiv.org/pdf/1809.01999.pdf
    Is a VAE architecture mapping: image -> Z_t -> reconstructed image
    """

    def __init__(self, config: MasterConfig) -> None:
        super().__init__(config)
        # Create encoder instance
        self.encoder = self.Encoder(config)
        # Create decoder instance
        self.decoder = self.Decoder(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        y = self.decoder(z.clone())
        return y, z

    class Encoder(Network):
        """
        Encoder for V network. Maps: Image -> Z_t
        """
        def __init__(self, config: MasterConfig = None) -> None:
            super().__init__(config)
            # Latent vector size
            N_z = config.Z_SIZE
            # instantiate encoder layers
            self.conv1 = torch.nn.Conv2d(3, 32, 4, 2)
            self.relu1 = torch.nn.ReLU(True)
            self.conv2 = torch.nn.Conv2d(32, 64, 4, 2)
            self.relu2 = torch.nn.ReLU(True)
            self.conv3 = torch.nn.Conv2d(64, 128, 4, 2)
            self.relu3 = torch.nn.ReLU(True)
            self.conv4 = torch.nn.Conv2d(128, 256, 4, 2)
            self.relu4 = torch.nn.ReLU(True)
            self.mu_flatten = torch.nn.Flatten()
            self.fv2mu = torch.nn.Linear(1024, N_z)
            self.sigma_flatten = torch.nn.Flatten()
            self.fv2sigma = torch.nn.Linear(1024, N_z)
            self.gaussian = torch.distributions.Normal(0, 1)

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

    class Decoder(Network):
        """
        Decoder for V network. Maps: Z_t -> Image
        """
        def __init__(self, config: MasterConfig = None) -> None:
            super().__init__(config)
            # Latent vector size
            N_z = config.Z_SIZE
            # instantiate Decoder layers
            self.linear1 = torch.nn.Linear(N_z, 1024)
            self.unflatten = torch.nn.Unflatten(-1, (1024, 1, 1))
            self.deconv1 = torch.nn.ConvTranspose2d(1024, 128, 5, 2)
            self.relu1 = torch.nn.ReLU(True)
            self.deconv2 = torch.nn.ConvTranspose2d(128, 64, 5, 2)
            self.relu2 = torch.nn.ReLU(True)
            self.deconv3 = torch.nn.ConvTranspose2d(64, 32, 6, 2)
            self.relu3 = torch.nn.ReLU(True)
            self.deconv4 = torch.nn.ConvTranspose2d(32, 3, 6, 2)
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


class Mnet(Network):
    """
    M 'memory' network as described in https://arxiv.org/pdf/1809.01999.pdf
    Is a mixed-density RNN architecture mapping: Z_t, h_t, a_t -> P(Z_{t+1})
    """

    def __init__(self, config: MasterConfig = None) -> None:
        super().__init__(config)
        # boolean determining if LSTM uses a cell state
        self.LSTM_CELL_ST = self.config.LSTM_CELL_ST
        # Instantiate LSTM
        self.lstm = self.LSTM(config)
        # Instantiate MDN
        self.mdn = self.MDN(config)

    def forward(self, z0: torch.Tensor, h0: torch.Tensor,
                c0: torch.Tensor, a0: torch.Tensor) -> Tuple[
            torch.distributions.MixtureSameFamily, torch.Tensor]:
        # Concatenate the latent vector with the previous action
        za = torch.cat([z0, a0], -1)
        # Compute LSTM
        h1, c1 = self.lstm(za, h0, c0)
        # if LSTM uses a cell state, concatenate latent vector,
        #   hidden vector, and cell state vector
        if self.LSTM_CELL_ST:
            h = torch.cat([z0, h1, c1], -1)
        # else concatente the hidden vector
        #   with the latent vector
        else:
            h = torch.cat([z0, h1], -1)
        # Compute MDN
        z_mixture = self.mdn(h)
        # return mixture distribution of the next latent vector,
        #   the new hidden vector, and the new cell state vector
        return z_mixture, h1, c1

    class LSTM(Network):
        """
        Long-Short-Term-Memory Network.
        """

        def __init__(self, config: MasterConfig = None) -> None:
            super().__init__(config)
            # Latent vector size
            self.N_z = config.Z_SIZE
            # Hidden vector size
            self.N_h = config.HX_SIZE
            # Action-space size
            self.N_a = config.ACTION_SPACE_SIZE
            # Create LSTM cell
            self.cell = torch.nn.LSTMCell(
                self.N_z+self.N_a, self.N_h)

        def forward(self, x: torch.Tensor,
                    hx: torch.Tensor,
                    cx: torch.Tensor) -> Tuple[torch.Tensor]:
            # Compute LSTM & return output
            return self.cell(x, (hx, cx))

    class MDN(Network):
        """
        Mixture Density Network. Outputs a mixture of
        multivariate Gaussian distributions.
        Note: Does not model covariances.
        """

        def __init__(self, config: MasterConfig = None) -> None:
            super().__init__(config)
            # Latent vector size
            N_z = config.Z_SIZE
            # Hidden vector size
            N_h = config.HX_SIZE
            # Number of Gaussians in mixture
            N_GAUSSIANS = config.N_GAUSSIANS
            # if LSTM is using hidden state AND cell state
            C = 2 if config.LSTM_CELL_ST else 1
            # Mixing coefficient for each Gaussian
            self.pi = torch.nn.Linear(N_z + N_h * C, N_GAUSSIANS)
            # Softmax for pi output so they add to 1
            self.pi_softmax = torch.nn.Softmax(-1)
            # Mean vectors for each Gaussian
            self.mu = torch.nn.Linear(N_z + N_h * C, N_GAUSSIANS * N_z)
            # Covariance matrix diagonal for each Gaussian
            self.sigma = torch.nn.Linear(N_z + N_h * C, N_GAUSSIANS * N_z)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # Calculate mixing coefficients for each Gaussian
            pi = self.pi(x)
            # Apply softmax (so pi sums to 1)
            pi = self.pi_softmax(pi)
            # Calculate mean vectors for each Gaussian
            mu = self.mu(x)
            # Calculate variance vectors for each Gaussian
            sigma = torch.exp(self.sigma(x)) # exp to keep it positive
            # Reshape to (Batch, N Gaussians, N_z)
            mu = mu.view(pi.size(0), pi.size(-1), -1)
            sigma = sigma.view(pi.size(0), pi.size(-1), -1)
            # Create a categorical distribution with pi
            pi = torch.distributions.Categorical(pi)
            # Create independent multivariate Gaussian distribution
            gaussians = torch.distributions.Independent(
                torch.distributions.Normal(mu, sigma), 1)
            # Create Gaussian mixture distribution
            mixture = torch.distributions.MixtureSameFamily(pi, gaussians)
            return mixture

class Cnet(Network):
    """
    C 'controller' network as described in https://arxiv.org/pdf/1809.01999.pdf
    Is a simple fully-connected network mapping: (Z_t, H_t) -> a_{t+1}
    """

    def __init__(self, config: MasterConfig = None) -> None:
        super().__init__(config)
        # Latent vector size
        self.N_z = config.Z_SIZE
        # Hidden vector size
        self.N_h = config.HX_SIZE
        # Create Wx+b
        self.l1 = torch.nn.Linear(self.N_z + self.N_h, 3)
        # Sigmoid activation function for output
        self.sigmoid1 = torch.nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.l1(x)
        y = self.sigmoid1(x)
        return y
