import torch
from compressed_vae.losses import hyperspherical_compression_loss


def test_compression_loss_is_finite():
    # Twenty samples per class ensures the class-conditional batch statistics exist.
    labels = torch.arange(10).repeat_interleave(20)
    mu = torch.randn(200, 128, requires_grad=True)
    logvar = torch.randn(200, 128, requires_grad=True) * 0.1
    loss = hyperspherical_compression_loss(mu, logvar, labels, end_compress=126)
    assert torch.isfinite(loss)
    loss.backward()
    assert mu.grad is not None
