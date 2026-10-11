"""Unit tests for CTR evaluation metrics and calibration calculations."""

import numpy as np
from ctr_prediction.evaluate import (
    compute_calibration_curve,
    compute_classification_metrics,
    compute_ranking_ndcg_at_k,
)


def test_compute_classification_metrics():
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7])

    metrics = compute_classification_metrics(y_true, y_prob)

    assert "log_loss" in metrics
    assert "roc_auc" in metrics
    assert "pr_auc" in metrics
    assert "brier_score" in metrics
    assert "ece" in metrics
    assert metrics["roc_auc"] == 1.0  # Perfect separation
    assert 0.0 <= metrics["ece"] <= 1.0


def test_compute_calibration_curve():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.1, 0.9, 0.9])

    ece, bin_edges, bin_confs, bin_accs, bin_counts = compute_calibration_curve(y_true, y_prob, n_bins=5)
    assert len(bin_edges) == 6
    assert len(bin_confs) == 5
    assert ece >= 0.0


def test_compute_ranking_ndcg_at_k():
    y_true = np.array([1, 0, 1, 0, 1, 0, 0, 1, 0, 1])
    y_prob = np.array([0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.2, 0.85, 0.15, 0.95])

    ndcg = compute_ranking_ndcg_at_k(y_true, y_prob, k=3, list_size=5, n_lists=10, random_state=42)
    assert 0.0 <= ndcg <= 1.0
