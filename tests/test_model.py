import torch
from compressed_vae.model import CompressedVAE


def test_model_shapes():
    model = CompressedVAE(latent_dim=128, channels=(16, 32, 64), image_size=32)
    x = torch.rand(4, 3, 32, 32)
    mu, logvar, z, recon, _, _ = model(x)
    assert mu.shape == (4, 128)
    assert logvar.shape == (4, 128)
    assert z.shape == (4, 128)
    assert recon.shape == x.shape
