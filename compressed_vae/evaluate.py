"""Evaluate a trained compressed VAE on CIFAR-10 vs CIFAR-100."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from .data import cifar10_cifar100_loaders
from .metrics import evaluate_scores, knn_anomaly_scores
from .model import CompressedVAE
from .utils import load_checkpoint, load_yaml, resolve_device, save_json, seed_everything
from .visualization import plot_latent_projection, plot_roc_and_scores


@torch.no_grad()
def encode_loader(model, loader, device, *, batchnorm_mode: str = "eval"):
    """Encode a loader using either standard inference or the historical protocol.

    The original experiment never switched the model to ``eval()`` after training.
    Consequently, BatchNorm layers used statistics from each evaluation batch.
    ``batchnorm_mode='train'`` intentionally reproduces that behaviour without
    computing gradients. ``batchnorm_mode='eval'`` is the standard protocol.
    """

    if batchnorm_mode == "train":
        model.train()
    elif batchnorm_mode == "eval":
        model.eval()
    else:
        raise ValueError("batchnorm_mode must be 'train' or 'eval'")

    embeddings, labels = [], []
    for images, target in tqdm(loader, desc="encoding", unit="batch"):
        mu, _, _ = model.encode(images.to(device, non_blocking=True))
        embeddings.append(mu.cpu().numpy())
        labels.append(target.numpy())
    return np.concatenate(embeddings), np.concatenate(labels)


def _protocol_settings(cfg: dict, protocol: str) -> dict:
    eval_cfg = cfg["evaluation"]
    if protocol == "historical":
        # Exact structural choices in the original training_loop_v12.py evaluation:
        # model left in train mode; CIFAR-10 train/test batches of 200; CIFAR-100
        # batches of 4000; all loaders shuffled; train bank horizontally augmented.
        return {
            "batchnorm_mode": "train",
            "train_batch_size": 200,
            "id_batch_size": 200,
            "ood_batch_size": 4000,
            "num_workers": 0,
            "train_augmentation": True,
            "shuffle_train": True,
            "shuffle_id": True,
            "shuffle_ood": True,
            "historical_rng": True,
            "plot_protocol": "historical",
        }
    if protocol == "standard":
        return {
            "batchnorm_mode": "eval",
            "train_batch_size": int(eval_cfg.get("train_batch_size", cfg["training"]["batch_size"])),
            "id_batch_size": int(eval_cfg.get("batch_size", 512)),
            "ood_batch_size": int(eval_cfg.get("batch_size", 512)),
            "num_workers": int(cfg.get("num_workers", 2)),
            "train_augmentation": False,
            "shuffle_train": False,
            "shuffle_id": False,
            "shuffle_ood": False,
            "historical_rng": False,
            "plot_protocol": "common",
        }
    raise ValueError("protocol must be 'historical' or 'standard'")


def run(
    config_path: str,
    checkpoint_path: str | None = None,
    output_dir: str | None = None,
    *,
    protocol: str | None = None,
):
    cfg = load_yaml(config_path)
    seed = int(cfg.get("seed", 100))
    seed_everything(seed)
    device = resolve_device(cfg.get("device", "auto"))

    model_cfg = cfg["model"]
    model = CompressedVAE(
        image_channels=3,
        latent_dim=int(model_cfg["latent_dim"]),
        channels=tuple(model_cfg["channels"]),
        image_size=32,
        normalize_samples=bool(model_cfg.get("normalize_samples", True)),
    ).to(device)
    checkpoint_path = checkpoint_path or cfg["checkpoint"]
    load_checkpoint(checkpoint_path, model, device)

    eval_cfg = cfg["evaluation"]
    protocol = protocol or str(eval_cfg.get("protocol", "standard"))
    settings = _protocol_settings(cfg, protocol)

    loaders = cifar10_cifar100_loaders(
        cfg.get("data_dir", "data"),
        train_batch_size=settings["train_batch_size"],
        id_batch_size=settings["id_batch_size"],
        ood_batch_size=settings["ood_batch_size"],
        num_workers=settings["num_workers"],
        seed=seed,
        download=bool(cfg.get("download", True)),
        train_augmentation=settings["train_augmentation"],
        shuffle_train=settings["shuffle_train"],
        shuffle_id=settings["shuffle_id"],
        shuffle_ood=settings["shuffle_ood"],
        historical_rng=settings["historical_rng"],
    )

    # Preserve the historical sequence: test ID, test OOD, then training bank.
    id_mu, id_labels = encode_loader(
        model, loaders.id_test, device, batchnorm_mode=settings["batchnorm_mode"]
    )
    ood_mu, ood_labels = encode_loader(
        model, loaders.ood_test, device, batchnorm_mode=settings["batchnorm_mode"]
    )
    train_mu, _ = encode_loader(
        model, loaders.train, device, batchnorm_mode=settings["batchnorm_mode"]
    )

    query = np.concatenate([id_mu, ood_mu])
    labels = np.concatenate([np.zeros(len(id_mu)), np.ones(len(ood_mu))]).astype(np.int64)
    scores = knn_anomaly_scores(train_mu, query, k=int(eval_cfg.get("knn_k", 3)))
    metrics, roc = evaluate_scores(labels, scores)

    default_output = f"runs/cifar10_vs_cifar100_{protocol}"
    output = Path(output_dir or cfg.get("output_dir", default_output))
    output.mkdir(parents=True, exist_ok=True)
    result = metrics.to_dict()
    result["protocol"] = protocol
    save_json(result, output / "metrics.json")
    np.savez_compressed(
        output / "scores.npz",
        scores=scores,
        labels=labels,
        id_embeddings=id_mu,
        id_class_labels=id_labels,
        ood_embeddings=ood_mu,
        ood_class_labels=ood_labels,
        protocol=np.array(protocol),
    )
    plot_roc_and_scores(labels, scores, roc, metrics, output / "roc_and_scores.png")
    # plot_latent_projection(
    #     id_mu,
    #     id_labels,
    #     ood_mu,
    #     ood_labels,
    #     output / "latent_projection.png",
    #     protocol=settings["plot_protocol"],
    # )

    print(f"Protocol: {protocol}")
    print(f"AUROC: {metrics.auroc:.6f}")
    print(f"FPR95: {metrics.fpr95:.6f}")
    print(f"Outputs: {output}")
    if protocol == "historical":
        print(
            "Note: historical mode deliberately uses BatchNorm training-mode batch statistics "
            "and the original stochastic training-bank augmentation."
        )
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cifar10_cifar100_full.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--protocol", choices=["historical", "standard"], default=None)
    args = parser.parse_args()
    run(args.config, args.checkpoint, args.output_dir, protocol=args.protocol)


if __name__ == "__main__":
    main()
