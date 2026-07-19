import numpy as np
from compressed_vae.metrics import evaluate_scores, knn_anomaly_scores


def test_knn_and_metrics():
    train = np.array([[0.0], [0.1], [-0.1]], dtype=np.float32)
    query = np.array([[0.0], [10.0]], dtype=np.float32)
    scores = knn_anomaly_scores(train, query, k=2)
    metrics, _ = evaluate_scores(np.array([0, 1]), scores)
    assert metrics.auroc == 1.0
    assert scores[1] > scores[0]
