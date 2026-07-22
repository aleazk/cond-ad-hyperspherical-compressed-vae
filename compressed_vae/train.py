"""Train the reference compressed VAE on CIFAR-10."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm

from .data import cifar10_cifar100_loaders
from .losses import (
    hyperspherical_compression_loss,
    legacy_annealing_factor,
    reconstruction_loss,
    standard_kl_loss,
)
from .model import CompressedVAE
from .utils import load_checkpoint, load_yaml, resolve_device, save_json, seed_everything


def _set_trainable(module: torch.nn.Module, flag: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = flag


def run(config_path: str):
    cfg = load_yaml(config_path)
    seed_everything(int(cfg.get("seed", 100)))
    device = resolve_device(cfg.get("device", "auto"))
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    compression = cfg["compression"]
    output = Path(cfg.get("output_dir", "runs/cifar10_training"))
    output.mkdir(parents=True, exist_ok=True)

    model = CompressedVAE(
        latent_dim=int(model_cfg["latent_dim"]),
        channels=tuple(model_cfg["channels"]),
        image_size=32,
        normalize_samples=bool(model_cfg.get("normalize_samples", True)),
    ).to(device)
    start_epoch = 0
    if cfg.get("pretrained"):
        checkpoint = load_checkpoint(cfg["pretrained"], model, device)
        start_epoch = int(checkpoint.get("epoch", -1)) + 1 if train_cfg.get("resume", False) else 0

    loaders = cifar10_cifar100_loaders(
        cfg.get("data_dir", "data"),
        batch_size=int(train_cfg["batch_size"]),
        eval_batch_size=int(cfg["evaluation"].get("batch_size", 512)),
        num_workers=int(cfg.get("num_workers", 2)),
        seed=int(cfg.get("seed", 100)),
        download=bool(cfg.get("download", True)),
        train_augmentation=True,
    )
    opt_e = Adam(model.encoder.parameters(), lr=float(train_cfg["lr_encoder"]))
    opt_d = Adam(model.decoder.parameters(), lr=float(train_cfg["lr_decoder"]))
    sched_e = MultiStepLR(opt_e, milestones=list(train_cfg.get("milestones", [280, 450])), gamma=0.5)
    sched_d = MultiStepLR(opt_d, milestones=list(train_cfg.get("milestones", [280, 450])), gamma=0.5)

    beta_target = float(train_cfg.get("beta_target", 2500.0))
    beta_effective = float(train_cfg.get("initial_beta", 1.0))
    history = []
    for epoch in range(start_epoch, int(train_cfg["epochs"])):
        model.train()
        rec_values, kl_values = [], []
        progress = tqdm(loaders.train, desc=f"epoch {epoch + 1}", unit="batch")
        for images, labels in progress:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # Encoder update, preserving the separate historical optimization steps.
            _set_trainable(model.encoder, True)
            _set_trainable(model.decoder, False)
            mu, logvar, _ = model.encode(images)
            z = model.reparameterize(mu, logvar)
            recon, _ = model.decode(z)
            rec = reconstruction_loss(images, recon)
            if compression["mode"] == "standard":
                kl = standard_kl_loss(mu, logvar)
                factor = 1.0
            else:
                kl = hyperspherical_compression_loss(
                    mu,
                    logvar,
                    labels,
                    int(compression["end_compress"]),
                    n_classes=10,
                    class_axis_stride=int(compression.get("class_axis_stride", 10)),
                    epoch=epoch,
                    training=True,
                )
                factor = legacy_annealing_factor(epoch)
            encoder_loss = 0.001 * (rec + beta_effective * kl * factor)
            opt_e.zero_grad(set_to_none=True)
            encoder_loss.backward()
            opt_e.step()

            # Decoder update.
            _set_trainable(model.encoder, False)
            _set_trainable(model.decoder, True)
            recon, _ = model.decode(z.detach())
            dec_rec = reconstruction_loss(images, recon)
            decoder_loss = 0.001 * dec_rec
            opt_d.zero_grad(set_to_none=True)
            decoder_loss.backward()
            opt_d.step()

            rec_values.append(float(rec.detach()))
            kl_values.append(float(kl.detach()))
            progress.set_postfix(rec=f"{rec_values[-1]:.2f}", kl=f"{kl_values[-1]:.2e}")

        sched_e.step()
        sched_d.step()
        if epoch == 0 and kl_values:
            beta_effective = beta_target * (sum(rec_values) / len(rec_values)) / (
                sum(kl_values) / len(kl_values) + 1e-3
            )
        record = {
            "epoch": epoch,
            "reconstruction": sum(rec_values) / len(rec_values),
            "compression_loss": sum(kl_values) / len(kl_values),
            "beta_effective": beta_effective,
        }
        history.append(record)
        print(record)

        save_every = int(train_cfg.get("save_every", 50))
        if (epoch + 1) % save_every == 0 or epoch + 1 == int(train_cfg["epochs"]):
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "config": cfg,
                    "history": history,
                },
                output / f"checkpoint_epoch_{epoch + 1:04d}.pt",
            )
            save_json({"history": history}, output / "history.json")
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cifar10_cifar100_full.yaml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
