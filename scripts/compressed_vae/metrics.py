"""Anomaly scoring and evaluation metrics."""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.neighbors import NearestNeighbors


@dataclass(frozen=True)
class DetectionMetrics:
    auroc: float
    fpr95: float
    n_id: int
    n_ood: int

    def to_dict(self) -> dict:
        return asdict(self)


def knn_anomaly_scores(
    train_embeddings: np.ndarray,
    query_embeddings: np.ndarray,
    k: int = 3,
    *,
    batch_size: int | None = None,
) -> np.ndarray:
    """Mean Euclidean distance to the ``k`` nearest training embeddings."""
    train = np.asarray(train_embeddings, dtype=np.float32)
    query = np.asarray(query_embeddings, dtype=np.float32)
    if train.ndim != 2 or query.ndim != 2 or train.shape[1] != query.shape[1]:
        raise ValueError("train and query embeddings must be 2D with equal feature dimensions")
    if not 1 <= k <= len(train):
        raise ValueError("k must be between 1 and the number of training embeddings")
    model = NearestNeighbors(n_neighbors=k, metric="euclidean")
    model.fit(train)
    distances = model.kneighbors(query, return_distance=True)[0]
    return distances.mean(axis=1)


def evaluate_scores(labels: np.ndarray, scores: np.ndarray) -> tuple[DetectionMetrics, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    labels = np.asarray(labels).astype(np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.shape != scores.shape:
        raise ValueError("labels and scores must have the same shape")
    fpr, tpr, thresholds = roc_curve(labels, scores)
    metrics = DetectionMetrics(
        auroc=float(roc_auc_score(labels, scores)),
        fpr95=float(np.interp(0.95, tpr, fpr)),
        n_id=int((labels == 0).sum()),
        n_ood=int((labels == 1).sum()),
    )
    return metrics, (fpr, tpr, thresholds)
