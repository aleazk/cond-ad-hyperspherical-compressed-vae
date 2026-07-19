import torch
from compressed_vae.coordinates import (
    cartesian_to_cosine_hyperspherical,
    cartesian_to_hyperspherical_angles,
    hyperspherical_to_cartesian,
    normalize_to_hypersphere,
)


def test_cosine_coordinates_known_vector():
    x = torch.tensor([[1.0, 1.0, 1.0]])
    result = cartesian_to_cosine_hyperspherical(x, eps=0.0)
    expected = torch.tensor([[1 / 3**0.5, 1 / 2**0.5]])
    assert torch.allclose(result, expected, atol=1e-6)


def test_round_trip():
    x = torch.tensor([[1.0, 2.0, 3.0], [-1.0, 0.5, -2.0]])
    r = torch.linalg.vector_norm(x, dim=1)
    angles = cartesian_to_hyperspherical_angles(x, eps=0.0)
    reconstructed = hyperspherical_to_cartesian(r, angles)
    assert torch.allclose(x, reconstructed, atol=1e-5)


def test_normalize_radius():
    x = torch.randn(8, 16)
    z = normalize_to_hypersphere(x)
    assert torch.allclose(torch.linalg.vector_norm(z, dim=1), torch.full((8,), 4.0), atol=1e-5)
