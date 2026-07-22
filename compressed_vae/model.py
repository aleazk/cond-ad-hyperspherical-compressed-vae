"""Residual VAE architecture used by the reference CIFAR experiment."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from .coordinates import normalize_to_hypersphere


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, groups: int = 1, scale: float = 1.0):
        super().__init__()
        mid_channels = int(out_channels * scale)
        self.conv_expand = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            if in_channels != out_channels
            else None
        )
        self.conv1 = nn.Conv2d(
            in_channels, mid_channels, kernel_size=3, padding=1, groups=groups, bias=False
        )
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(
            mid_channels, out_channels, kernel_size=3, padding=1, groups=groups, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.conv_expand(x) if self.conv_expand is not None else x
        output = self.relu1(self.bn1(self.conv1(x)))
        output = self.bn2(self.conv2(output))
        return self.relu2(output + identity)


class Encoder(nn.Module):
    def __init__(
        self,
        image_channels: int = 3,
        latent_dim: int = 128,
        channels: Sequence[int] = (16, 32, 64),
        image_size: int = 32,
    ):
        super().__init__()
        self.zdim = latent_dim
        self.cdim = image_channels
        self.image_size = image_size

        current = channels[0]
        self.main = nn.Sequential(
            nn.Conv2d(image_channels, current, 5, 1, 2, bias=False),
            nn.BatchNorm2d(current),
            nn.LeakyReLU(0.2),
            nn.AvgPool2d(2),
        )
        size = image_size // 2
        for next_channels in channels[1:]:
            self.main.add_module(f"res_in_{size}", ResidualBlock(current, next_channels))
            self.main.add_module(f"down_to_{size // 2}", nn.AvgPool2d(2))
            current, size = next_channels, size // 2
        self.main.add_module(f"res_in_{size}", ResidualBlock(current, current))

        self.conv_output_size = self._conv_output_size()
        n_features = int(torch.tensor(self.conv_output_size).prod().item())
        # Preserve the historical name so the supplied checkpoint loads directly.
        self.F2L = nn.Sequential(nn.Linear(n_features, 2 * latent_dim))

    def _conv_output_size(self) -> torch.Size:
        was_training = self.main.training
        self.main.eval()
        with torch.no_grad():
            out = self.main(torch.zeros(1, self.cdim, self.image_size, self.image_size))
        self.main.train(was_training)
        return out[0].shape

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.main(x).view(x.shape[0], -1)
        mu, logvar = self.F2L(features).chunk(2, dim=1)
        return mu, logvar, features


class Decoder(nn.Module):
    def __init__(
        self,
        image_channels: int,
        latent_dim: int,
        channels: Sequence[int],
        image_size: int,
        conv_input_size: torch.Size,
    ):
        super().__init__()
        self.cdim = image_channels
        self.image_size = image_size
        self.conv_input_size = conv_input_size
        n_features = int(torch.tensor(conv_input_size).prod().item())
        self.L2F = nn.Sequential(
            nn.Linear(latent_dim, n_features),
            nn.BatchNorm1d(n_features),
            nn.LeakyReLU(0.2),
        )

        current = channels[-1]
        size = 4
        self.main = nn.Sequential()
        for next_channels in reversed(channels):
            self.main.add_module(f"res_in_{size}", ResidualBlock(current, next_channels))
            self.main.add_module(f"up_to_{size * 2}", nn.Upsample(scale_factor=2, mode="nearest"))
            current, size = next_channels, size * 2
        self.main.add_module(f"res_in_{size}", ResidualBlock(current, current))
        self.main.add_module("predict", nn.Conv2d(current, image_channels, 5, 1, 2))
        self.main.add_module("Sigmoid", nn.Sigmoid())

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.L2F(z.view(z.shape[0], -1))
        image = self.main(features.view(features.shape[0], *self.conv_input_size))
        return image, features


class CompressedVAE(nn.Module):
    """Residual VAE with optional projection of samples to radius ``sqrt(d)``."""

    def __init__(
        self,
        image_channels: int = 3,
        latent_dim: int = 128,
        channels: Sequence[int] = (16, 32, 64),
        image_size: int = 32,
        normalize_samples: bool = True,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.normalize_samples = normalize_samples
        self.encoder = Encoder(image_channels, latent_dim, channels, image_size)
        self.decoder = Decoder(
            image_channels,
            latent_dim,
            channels,
            image_size,
            self.encoder.conv_output_size,
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.encoder(x)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        z = mu + torch.randn_like(std) * std
        return normalize_to_hypersphere(z) if self.normalize_samples else z

    def decode(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.decoder(z)

    def forward(self, x: torch.Tensor):
        mu, logvar, enc_features = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon, dec_features = self.decode(z)
        return mu, logvar, z, recon, enc_features, dec_features
