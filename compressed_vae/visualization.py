"""Publication- and README-ready plots for the CIFAR reproduction."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_roc_and_scores(labels, scores, roc, metrics, output: str | Path) -> None:
    fpr, tpr, _ = roc
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    fig = plt.figure(figsize=(11, 4.2))

    ax = fig.add_subplot(1, 2, 1)
    ax.plot(fpr, tpr, label=f"3-NN: AUROC={metrics.auroc:.3f}, FPR95={metrics.fpr95:.3f}")
    ax.plot([0, 1], [0, 1], "--", linewidth=1, label="random")
    ax.set(xlabel="False-positive rate", ylabel="True-positive rate", title="CIFAR-10 ID vs CIFAR-100 OOD")
    ax.legend(loc="lower right")

    ax = fig.add_subplot(1, 2, 2)
    ax.hist(scores[labels == 0], bins=50, alpha=0.65, density=True, label="CIFAR-10 (ID)")
    ax.hist(scores[labels == 1], bins=50, alpha=0.65, density=True, label="CIFAR-100 (OOD)")
    ax.set(xlabel="3-NN latent anomaly score", ylabel="Density", title="Score distributions")
    ax.legend()

    fig.tight_layout()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def coordinate_average_projection(embeddings: np.ndarray) -> np.ndarray:
    """Project a high-dimensional vector to 3D by averaging three coordinate blocks."""
    embeddings = np.asarray(embeddings)
    chunks = np.array_split(embeddings, 3, axis=1)
    projected = np.stack([chunk.mean(axis=1) for chunk in chunks], axis=1)
    mean_norm = np.linalg.norm(projected, axis=1).mean()
    if mean_norm > 0:
        projected = np.sqrt(3.0) * projected / mean_norm
    return projected


def plot_latent_projection(id_embeddings, id_labels, ood_embeddings, ood_labels, output: str | Path) -> None:
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    id_proj = coordinate_average_projection(id_embeddings)
    ood_proj = coordinate_average_projection(ood_embeddings)
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    for class_id in range(3):
        sample = id_proj[id_labels == class_id]
        ax.scatter(sample[:, 0], sample[:, 1], sample[:, 2], s=2, alpha=0.65, label=f"CIFAR-10 class {class_id}")
    sample = ood_proj[ood_labels == 0][:400]
    ax.scatter(sample[:, 0], sample[:, 1], sample[:, 2], s=5, marker="x", label="CIFAR-100 class 0")

    u, v = np.mgrid[0 : 2 * np.pi : 24j, 0 : np.pi : 12j]
    r = np.sqrt(3.0)
    ax.plot_wireframe(r * np.cos(u) * np.sin(v), r * np.sin(u) * np.sin(v), r * np.cos(v), linewidth=0.35, alpha=0.18)
    ax.set(xlabel="x", ylabel="y", zlabel="z", title="Coordinate-averaged latent projection")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    #fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
