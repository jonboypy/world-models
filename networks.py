# Imports
from typing import Tuple
import torch
from utils import MasterConfig

#####################################################################
#   Networks
#####################################################################

class Network(torch.nn.Module):

    """
    Base class which all other networks derived from.

    Args:
        cfg: A MasterConfig object.
    """

    def __init__(self, cfg: MasterConfig = None) -> None:
        super().__init__()
        self.cfg = cfg

    def initialize_parameters(self, init_type: str) -> None:
        """
        Initializes model parameters.
        """
        # Init funcs.
        def normal_init(m):
            if isinstance(m, (
                torch.nn.Conv2d,
                torch.nn.ConvTranspose2d,
                torch.nn.Linear)):
                torch.nn.init.normal_(m.weight.data, 0., 0.02)
                if m.bias is not None:
                    m.bias.data.zero_()

        def kaiming_init(m):
            if isinstance(m, (
                torch.nn.Conv2d,
                torch.nn.ConvTranspose2d,
                torch.nn.Linear)):
                torch.nn.init.kaiming_normal_(
                    m.weight.data, nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
        # Apply
        if init_type == 'normal':
            init_fx = normal_init
        elif init_type == 'kaiming':
            init_fx = kaiming_init
        else:
            raise SyntaxError(
                f'init type {init_type} not supported.')
        self.apply(init_fx)

class Vnet(Network):

    """
    V 'vision' network as described in https://arxiv.org/pdf/1809.01999.pdf
    Is a VAE architecture mapping: image -> Z_t -> reconstructed image
    """

    def __init__(self, cfg: MasterConfig) -> None:
        super().__init__(cfg)
        # Create encoder instance
        self.encoder = self.Encoder(cfg)
        # Create decoder instance
        self.decoder = self.Decoder(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z, mu, sigma = self.encoder(x)
        y = self.decoder(z.clone())
        return y, z, mu, sigma

    class Encoder(Network):
        """
        Encoder for V network. Maps: Image -> Z_t
        """
        def __init__(self, cfg: MasterConfig = None) -> None:
            super().__init__(cfg)
            # Latent vector size
            N_z = cfg.Z_SIZE
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
            # reparameterization trick
            std = torch.exp(sigma * 0.5)
            z = mu + std * torch.randn_like(std)
            return z, mu, sigma # sigma -> log-variance

    class Decoder(Network):
        """
        Decoder for V network. Maps: Z_t -> Image
        """
        def __init__(self, cfg: MasterConfig = None) -> None:
            super().__init__(cfg)
            # Latent vector size
            N_z = cfg.Z_SIZE
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

    def __init__(self, cfg: MasterConfig = None) -> None:
        super().__init__(cfg)
        # Instantiate LSTM
        self.lstm = self.LSTM(cfg)
        # Instantiate MDN
        self.mdn = self.MDN(cfg)

    def forward(self, z: torch.Tensor,
                a_prev: torch.Tensor) -> Tuple[
                torch.distributions.MixtureSameFamily, torch.Tensor]:
        # Dims: (Batch, Sequence, *)
        # Concatenate the latent vector & previous action
        za = torch.cat([z, a_prev], -1)
        # Compute LSTM
        h, _ = self.lstm(za)
        # Compute MDN
        z_next_est_dist = self.mdn(h)
        # return mixture distribution of the
        #   next latent vector and the new hidden vector
        return z_next_est_dist, h

    def forward_with_hx(self, z: torch.Tensor,
                h_prev: Tuple[torch.Tensor],
                a_prev: torch.Tensor) -> Tuple[
                torch.distributions.MixtureSameFamily, torch.Tensor]:
        # Dims: (Batch, Sequence, *)
        # Concatenate the previous latent vector & previous action
        za = torch.cat([z, a_prev], -1)
        # Compute LSTM
        out, h_new = self.lstm.forward_with_hx(za, h_prev)
        # Compute MDN
        z_next_est_dist = self.mdn(out)
        # return mixture distribution of the
        #   next latent vector and the new hidden vector
        return z_next_est_dist, out, h_new


    class LSTM(Network):
        """
        Long-Short-Term-Memory Network.
        """

        def __init__(self, cfg: MasterConfig = None) -> None:
            super().__init__(cfg)
            # Latent vector size
            N_z = cfg.Z_SIZE
            # Hidden vector size
            N_h = cfg.HX_SIZE
            # Action-space size
            N_a = cfg.ACTION_SPACE_SIZE
            # Create LSTM
            self.net = torch.nn.LSTM(N_z + N_a, N_h, batch_first=True)

        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
            # Compute LSTM & return output
            #   Uses h0, c0 == 0
            return self.net(x)

        def forward_with_hx(self, x: torch.Tensor,
                            hx: Tuple[torch.Tensor]
                            ) -> Tuple[torch.Tensor]:
            # Compute LSTM & return output
            return self.net(x, hx)


    class MDN(Network):
        """
        Mixture Density Network. Outputs a mixture of
        multivariate Gaussian distributions.
        Note: Does not model covariances.
        """

        def __init__(self, cfg: MasterConfig = None) -> None:
            super().__init__(cfg)
            # Latent vector size
            N_z = cfg.Z_SIZE
            # Hidden vector size
            N_h = cfg.HX_SIZE
            # Number of Gaussians in mixture
            N_g = cfg.N_GAUSSIANS
            # Mixing coefficient for each Gaussian
            self.pi = torch.nn.Linear(N_h, N_g)
            # Softmax for pi output so they add to 1
            self.pi_softmax = torch.nn.Softmax(-1)
            # Mean vectors for each Gaussian
            self.mu = torch.nn.Linear(N_h, N_g * N_z)
            # Covariance matrix diagonal for each Gaussian
            self.sigma = torch.nn.Linear(N_h, N_g * N_z)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # Calculate mixing coefficients for each Gaussian
            pi = self.pi(x)
            # Apply softmax (so pi sums to 1)
            pi = self.pi_softmax(pi)
            #BUG: gumbel softmax goes to nan sometimes.
            #pi = torch.nn.functional.gumbel_softmax(
            #    pi, tau=self.cfg.TEMP, dim=-1) #TODO: is this right for temperature?
            # Calculate mean vectors for each Gaussian
            mu = self.mu(x)
            # Calculate variance vectors for each Gaussian
            sigma = torch.exp(self.sigma(x)) # exp to keep it positive
            # Reshape to (Seq, Batch, N Gaussians, N_z)
            mu = mu.view(*pi.size(), -1)
            sigma = sigma.view(*pi.size(), -1)
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

    def __init__(self, cfg: MasterConfig = None) -> None:
        super().__init__(cfg)
        # Latent vector size
        N_z = cfg.Z_SIZE
        # Hidden vector size
        N_h = cfg.HX_SIZE
        # Action space size
        N_a = cfg.ACTION_SPACE_SIZE
        # Create Wx+b
        self.l1 = torch.nn.Linear(N_z + N_h, N_a)
        # turn off grads since we are doing
        #   gradient free optimzation :-)
        self.l1.weight.requires_grad = False
        self.l1.bias.requires_grad = False
        # Create action-space scaling func. for output
        if cfg.ENV_NAME == "CarRacing-v2":
            self.scale_2_action_space = \
                self._2_car_racing_action_space
        else: raise NotImplementedError(
                'Cnet does not currently support '
                f'{cfg.ENV_NAME}')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.l1(x)
        y = self.scale_2_action_space(x)
        return y
    
    def _2_car_racing_action_space(
            self, action: torch.Tensor) -> torch.Tensor:
        action[0] = torch.nn.functional.tanh(action[0])
        action[1:] = torch.nn.functional.sigmoid(action[1:])
        return action
