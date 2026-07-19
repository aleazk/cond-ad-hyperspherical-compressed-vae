from pathlib import Path
import torch
from compressed_vae.model import CompressedVAE
from compressed_vae.utils import load_checkpoint


def test_reference_checkpoint_loads():
    root = Path(__file__).resolve().parents[1]
    model = CompressedVAE(latent_dim=128, channels=(16, 32, 64), image_size=32)
    load_checkpoint(root / "checkpoints/cifar10_full_compression_z128_ec126.pt", model, torch.device("cpu"))
