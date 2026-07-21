"""Hyperspherical latent-volume compression for VAE anomaly detection."""

from .model import CompressedVAE
from .coordinates import cartesian_to_cosine_hyperspherical, hyperspherical_radius
from .metrics import evaluate_scores, knn_anomaly_scores

__all__ = [
    "CompressedVAE",
    "cartesian_to_cosine_hyperspherical",
    "hyperspherical_radius",
    "evaluate_scores",
    "knn_anomaly_scores",
]

__version__ = "0.1.0"
