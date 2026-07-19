#!/usr/bin/env python3
"""Verify the exact metrics stored in the attached historical run."""
from pathlib import Path
import json
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

root = Path(__file__).resolve().parents[1]
data = np.load(root / "results/reference/cifar10_vs_cifar100/scores.npz")
scores = data["scores"]
labels = data["labels"]
fpr, tpr, _ = roc_curve(labels, scores)
metrics = {
    "auroc": float(roc_auc_score(labels, scores)),
    "fpr95": float(np.interp(0.95, tpr, fpr)),
    "n_id": int((labels == 0).sum()),
    "n_ood": int((labels == 1).sum()),
}
print(json.dumps(metrics, indent=2))
expected = json.loads((root / "results/reference/cifar10_vs_cifar100/metrics.json").read_text())
for key in ("auroc", "fpr95"):
    if abs(metrics[key] - expected[key]) > 1e-12:
        raise SystemExit(f"Archived metric mismatch for {key}")
print("Archived reference verified.")
