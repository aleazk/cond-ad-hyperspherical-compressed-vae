"""Differentiable Cartesian/hyperspherical coordinate utilities."""

from __future__ import annotations

import torch


def hyperspherical_radius(x: torch.Tensor) -> torch.Tensor:
    """Return the Euclidean radius for each row of ``x``."""
    if x.ndim != 2:
        raise ValueError(f"Expected a 2D tensor, got shape {tuple(x.shape)}")
    return torch.linalg.vector_norm(x, dim=1)


def cartesian_to_cosine_hyperspherical(
    x: torch.Tensor, eps: float = 1e-3
) -> torch.Tensor:
    """Convert Cartesian vectors to cosines of their hyperspherical angles.

    For a vector ``x`` of dimension ``d``, this returns ``d-1`` values,
    with coordinate ``k`` equal to

    ``x[k] / sqrt(sum_{j=k}^{d-1} x[j]^2 + eps)``.

    The vectorized implementation is differentiable and reproduces the
    transformation used in the research code and manuscript.
    """
    if x.ndim != 2:
        raise ValueError(f"Expected a 2D tensor, got shape {tuple(x.shape)}")
    if x.shape[1] < 2:
        raise ValueError("At least two Cartesian dimensions are required")
    suffix_sq = torch.flip(torch.cumsum(torch.flip(x.square(), dims=[1]), dim=1), dims=[1])
    denom = torch.sqrt(suffix_sq + eps)
    return (x / denom)[:, :-1]


def cartesian_to_hyperspherical_angles(
    x: torch.Tensor, eps: float = 1e-3
) -> torch.Tensor:
    """Return hyperspherical angles, including the signed final angle."""
    cosines = torch.clamp(cartesian_to_cosine_hyperspherical(x, eps), -1.0, 1.0)
    angles = torch.arccos(cosines)
    final = angles[:, -1]
    final = torch.where(x[:, -1] >= 0, final, 2 * torch.pi - final)
    angles = angles.clone()
    angles[:, -1] = final
    return angles


def hyperspherical_to_cartesian(radius: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    """Convert hyperspherical coordinates back to Cartesian coordinates."""
    if angles.ndim != 2:
        raise ValueError("angles must have shape (batch, d-1)")
    if radius.ndim == 0:
        radius = radius.expand(angles.shape[0])
    if radius.ndim != 1 or radius.shape[0] != angles.shape[0]:
        raise ValueError("radius must be scalar or have shape (batch,)")

    batch, n_angles = angles.shape
    d = n_angles + 1
    out = torch.empty((batch, d), device=angles.device, dtype=angles.dtype)
    sine_prefix = torch.ones((batch,), device=angles.device, dtype=angles.dtype)
    for i in range(n_angles):
        out[:, i] = radius * sine_prefix * torch.cos(angles[:, i])
        sine_prefix = sine_prefix * torch.sin(angles[:, i])
    out[:, -1] = radius * sine_prefix
    return out


def normalize_to_hypersphere(x: torch.Tensor, radius: float | torch.Tensor | None = None) -> torch.Tensor:
    """Project each row to a hypersphere, defaulting to radius ``sqrt(d)``."""
    if x.ndim != 2:
        raise ValueError("x must have shape (batch, d)")
    if radius is None:
        radius_t = torch.sqrt(torch.tensor(float(x.shape[1]), device=x.device, dtype=x.dtype))
    else:
        radius_t = torch.as_tensor(radius, device=x.device, dtype=x.dtype)
    norm = hyperspherical_radius(x).clamp_min(1e-12).unsqueeze(1)
    return radius_t * x / norm
