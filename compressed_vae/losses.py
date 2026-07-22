"""Hyperspherical volume-compression loss used in the reference experiments."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .coordinates import cartesian_to_cosine_hyperspherical, hyperspherical_radius


def _per_class_rotated(values: torch.Tensor, labels: torch.Tensor, n_classes: int, stride: int):
    groups = []
    for class_id in range(n_classes):
        group = values[labels == class_id]
        if group.numel() == 0:
            raise ValueError(
                f"Mini-batch has no samples for class {class_id}. "
                "Use a larger batch or a class-balanced sampler."
            )
        groups.append(torch.roll(group, shifts=-class_id * stride, dims=1))
    return groups


def angular_mu_loss(
    mu: torch.Tensor,
    labels: torch.Tensor,
    end_compress: int,
    *,
    n_classes: int = 10,
    class_axis_stride: int = 10,
    epoch: int = 0,
    training: bool = True,
) -> torch.Tensor:
    """Angular loss on encoder means, matching the historical implementation."""
    d = mu.shape[1]
    if not 0 <= end_compress <= d - 2:
        raise ValueError(f"end_compress must be in [0, {d-2}], got {end_compress}")
    if (n_classes - 1) * class_axis_stride >= d:
        raise ValueError("latent_dim is too small for the requested class-axis stride")

    rotated = _per_class_rotated(mu, labels, n_classes, class_axis_stride)
    cosines = [cartesian_to_cosine_hyperspherical(group) for group in rotated]
    cos_std = torch.stack([torch.sqrt(c.var(dim=0, unbiased=True) + 1e-4) for c in cosines])

    ones = torch.ones((1, d), device=mu.device, dtype=mu.dtype)
    std_prior = cartesian_to_cosine_hyperspherical(ones).squeeze(0)
    beta = 50.0 * math.sqrt(d) / torch.sqrt(
        torch.arange(1, d, device=mu.device, dtype=mu.dtype)
    )
    shifted = cos_std + 1.0 - std_prior
    var_term = ((shifted.square() - torch.log(shifted.square()) - 1.0) * beta).sum()

    cos_mean = torch.stack([c.mean(dim=0) for c in cosines])
    mean_prior = F.one_hot(
        torch.arange(0, end_compress + 1, device=mu.device), num_classes=d - 1
    ).to(mu.dtype).sum(dim=0)
    sq_error = (mean_prior - cos_mean).square() * beta

    # The final hyperspherical angle is special; the historical code excludes it.
    first_weight = 150.0 if training and epoch > 100 else 100.0
    remaining_weight = math.sqrt(epoch - 101) + 1.0 if training and epoch > 100 else 1.0
    mean_term = first_weight * sq_error[:, 0].sum()
    if d > 3:
        mean_term = mean_term + remaining_weight * sq_error[:, 1 : d - 2].sum()

    return var_term + 0.05 * mean_term


def radial_mu_loss(mu: torch.Tensor) -> torch.Tensor:
    d = mu.shape[1]
    radii = hyperspherical_radius(mu)
    r_std = torch.sqrt(radii.var(unbiased=True) + 1e-4)
    var_term = (r_std + 1.0).square() - torch.log((r_std + 1.0).square()) - 1.0
    mean_target = torch.sqrt(torch.tensor(d - 0.5, device=mu.device, dtype=mu.dtype))
    mean_term = (radii.mean() - mean_target).square()
    return 50.0 * var_term + 50.0 * mean_term


def angular_sigma_loss(logvar: torch.Tensor) -> torch.Tensor:
    d = logvar.shape[1]
    std = torch.exp(0.5 * logvar)
    cosines = cartesian_to_cosine_hyperspherical(std)
    cos_std = torch.sqrt(cosines.var(dim=0, unbiased=True) + 1e-4)
    beta = 50.0 * math.sqrt(d) / torch.sqrt(
        torch.arange(1, d, device=logvar.device, dtype=logvar.dtype)
    )
    shifted = cos_std + 1.0
    var_term = ((shifted.square() - torch.log(shifted.square()) - 1.0) * beta).sum()

    target = cartesian_to_cosine_hyperspherical(
        torch.ones((1, d), device=logvar.device, dtype=logvar.dtype)
    ).squeeze(0)
    mean_term = ((target - cosines.mean(dim=0)).square() * beta).sum()
    return var_term + 0.05 * mean_term


def radial_sigma_loss(logvar: torch.Tensor) -> torch.Tensor:
    d = logvar.shape[1]
    radii = hyperspherical_radius(torch.exp(0.5 * logvar))
    r_std = torch.sqrt(radii.var(unbiased=True) + 1e-4)
    target_std = torch.sqrt(torch.tensor(0.5, device=logvar.device, dtype=logvar.dtype))
    ratio_sq = (r_std / target_std).square()
    var_term = ratio_sq - torch.log(ratio_sq) - 1.0
    target_mean = 0.001 * torch.sqrt(
        torch.tensor(d - 0.5, device=logvar.device, dtype=logvar.dtype)
    )
    mean_term = (radii.mean() - target_mean).square()
    return 50.0 * var_term + 50.0 * mean_term


def hyperspherical_compression_loss(
    mu: torch.Tensor,
    logvar: torch.Tensor,
    labels: torch.Tensor,
    end_compress: int,
    *,
    n_classes: int = 10,
    class_axis_stride: int = 10,
    epoch: int = 0,
    training: bool = True,
) -> torch.Tensor:
    """Full KLD-like compression objective used for CIFAR-10."""
    phi_mu = angular_mu_loss(
        mu,
        labels,
        end_compress,
        n_classes=n_classes,
        class_axis_stride=class_axis_stride,
        epoch=epoch,
        training=training,
    )
    r_mu = radial_mu_loss(mu)
    phi_sigma = angular_sigma_loss(logvar)
    r_sigma = radial_sigma_loss(logvar)
    return 1000.0 * phi_mu + 300.0 * r_mu + 500000.0 * phi_sigma + 500.0 * r_sigma


def standard_kl_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return (-0.5 * (1.0 + logvar - logvar.exp() - mu.square()).mean(dim=1)).mean()


def reconstruction_loss(x: torch.Tensor, recon: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(recon.flatten(1), x.flatten(1), reduction="none").sum(dim=1).mean()


def legacy_annealing_factor(epoch: int) -> float:
    """Reproduce the piecewise schedule in the supplied research script."""
    if epoch <= 100:
        return math.sqrt(epoch) + 1.0
    if epoch <= 300:
        return 1.0
    if epoch <= 500:
        return math.sqrt(epoch - 301) + 1.0
    if epoch <= 700:
        return 1.0
    if epoch <= 900:
        return math.sqrt(epoch - 701) + 1.0
    return 1.0
