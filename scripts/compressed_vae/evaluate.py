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
def encode_loader(model, loader, device):
    embeddings, labels = [], []
    model.eval()
    for images, target in tqdm(loader, desc="encoding", unit="batch"):
        mu, _, _ = model.encode(images.to(device, non_blocking=True))
        embeddings.append(mu.cpu().numpy())
        labels.append(target.numpy())
    return np.concatenate(embeddings), np.concatenate(labels)


def run(config_path: str, checkpoint_path: str | None = None, output_dir: str | None = None):
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
    loaders = cifar10_cifar100_loaders(
        cfg.get("data_dir", "data"),
        batch_size=int(cfg["training"]["batch_size"]),
        eval_batch_size=int(eval_cfg.get("batch_size", 512)),
        num_workers=int(cfg.get("num_workers", 2)),
        seed=seed,
        download=bool(cfg.get("download", True)),
        train_augmentation=bool(eval_cfg.get("legacy_train_augmentation", False)),
    )
    train_mu, _ = encode_loader(model, loaders.train, device)
    id_mu, id_labels = encode_loader(model, loaders.id_test, device)
    ood_mu, ood_labels = encode_loader(model, loaders.ood_test, device)

    query = np.concatenate([id_mu, ood_mu])
    labels = np.concatenate([np.zeros(len(id_mu)), np.ones(len(ood_mu))]).astype(np.int64)
    scores = knn_anomaly_scores(train_mu, query, k=int(eval_cfg.get("knn_k", 3)))
    metrics, roc = evaluate_scores(labels, scores)

    output = Path(output_dir or cfg.get("output_dir", "runs/cifar10_vs_cifar100"))
    output.mkdir(parents=True, exist_ok=True)
    save_json(metrics.to_dict(), output / "metrics.json")
    np.savez_compressed(
        output / "scores.npz",
        scores=scores,
        labels=labels,
        id_embeddings=id_mu,
        id_class_labels=id_labels,
        ood_embeddings=ood_mu,
        ood_class_labels=ood_labels,
    )
    plot_roc_and_scores(labels, scores, roc, metrics, output / "roc_and_scores.png")
    plot_latent_projection(id_mu, id_labels, ood_mu, ood_labels, output / "latent_projection.png")
    print(f"AUROC: {metrics.auroc:.6f}")
    print(f"FPR95: {metrics.fpr95:.6f}")
    print(f"Outputs: {output}")
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cifar10_cifar100_full.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    run(args.config, args.checkpoint, args.output_dir)


if __name__ == "__main__":
    main()
