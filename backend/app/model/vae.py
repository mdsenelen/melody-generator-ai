import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):
    """Efficient convolution for lightweight encoding"""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size,
            stride=stride, padding=padding, groups=in_channels
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class WebVAE(nn.Module):
    def __init__(
        self,
        input_shape=(1, 128, 256),
        latent_dim=258,
        batch_size=32,
        num_epochs=100,
        init_lr: float = 3e-4  # type: ignore
    ):
        super().__init__()
        self.input_shape = input_shape
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.Sequential(
            DepthwiseSeparableConv(1, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),

            DepthwiseSeparableConv(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            DepthwiseSeparableConv(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )

        # Output size after conv layers
        self._conv_shape = self._get_conv_output_shape()
        conv_flat_dim = int(torch.prod(torch.tensor(self._conv_shape)))

        # Latent space
        self.fc_mu = nn.Linear(conv_flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(conv_flat_dim, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, conv_flat_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2,
                               padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(64, 32, 3, stride=2,
                               padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(32, 1, 3, stride=2,
                               padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def _get_conv_output_shape(self):
        with torch.no_grad():
            x = torch.zeros(1, *self.input_shape)
            x = self.encoder(x)
            return x.shape[1:]  # (C, H, W)

    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.fc_decode(z)
        x = x.view(-1, *self._conv_shape)
        return self.decoder(x)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar
