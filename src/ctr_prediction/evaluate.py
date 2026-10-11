"""Comprehensive evaluation metrics, calibration analysis, and ranking diagnostics."""

import logging
from typing import Any, Dict, List, Tuple
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Compute all primary and secondary evaluation metrics for CTR prediction.

    Primary metrics:
    - Log Loss (Binary Cross Entropy): Measures probabilistic fidelity and log-likelihood.
    - ROC-AUC: Measures discrimination across all thresholds (independent of prior class balance).
    - PR-AUC (Average Precision): Evaluates precision-recall trade-off under heavy class imbalance.
    - Brier Score: Mean squared error of probabilistic predictions.

    Secondary metrics:
    - Precision, Recall, F1 at specified decision threshold.
    - Confusion Matrix counts.
    - Expected Calibration Error (ECE) across 10 probability bins.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=np.float64)

    # Clip probabilities to avoid infinite log loss
    eps = 1e-15
    y_prob_clipped = np.clip(y_prob, eps, 1.0 - eps)

    # Probabilistic metrics
    loss = float(log_loss(y_true, y_prob_clipped))
    roc_auc = float(roc_auc_score(y_true, y_prob))
    pr_auc = float(average_precision_score(y_true, y_prob))
    brier = float(brier_score_loss(y_true, y_prob))

    # Threshold-dependent metrics
    y_pred = (y_prob >= threshold).astype(int)
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    acc = float(np.mean(y_true == y_pred))

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Expected Calibration Error (ECE) calculation with 10 uniform bins
    ece, bin_edges, bin_confs, bin_accs, bin_counts = compute_calibration_curve(y_true, y_prob, n_bins=10)

    return {
        "log_loss": loss,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "brier_score": brier,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "threshold": threshold,
        "confusion_matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
        "ece": ece,
    }


def compute_calibration_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute reliability diagram data and Expected Calibration Error (ECE)."""
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_confs = []
    bin_accs = []
    bin_counts = []
    ece = 0.0
    total_samples = len(y_true)

    for i in range(n_bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        mask = (y_prob >= low) & (y_prob <= high if i == n_bins - 1 else y_prob < high)
        count = np.sum(mask)
        bin_counts.append(int(count))
        if count > 0:
            mean_conf = float(np.mean(y_prob[mask]))
            mean_acc = float(np.mean(y_true[mask]))
            bin_confs.append(mean_conf)
            bin_accs.append(mean_acc)
            ece += (count / total_samples) * abs(mean_acc - mean_conf)
        else:
            bin_confs.append((low + high) / 2.0)
            bin_accs.append(0.0)

    return float(ece), bin_edges, np.array(bin_confs), np.array(bin_accs), np.array(bin_counts)


def compute_ranking_ndcg_at_k(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    k: int = 5,
    list_size: int = 10,
    n_lists: int = 1000,
    random_state: int = 42,
) -> float:
    """Evaluate ranking quality using simulated ad auction candidate slates (NDCG@K).

    In ad systems, models score a candidate set of competing ad impressions per user request.
    This function groups instances into slates of candidate size `list_size` and computes NDCG@K.
    """
    rng = np.random.default_rng(random_state)
    n = len(y_true)
    ndcg_scores = []

    for _ in range(n_lists):
        idx = rng.choice(n, size=list_size, replace=False)
        slate_true = y_true[idx]
        slate_prob = y_prob[idx]

        # Sort by predicted score
        pred_order = np.argsort(-slate_prob)[:k]
        dcg = np.sum((2 ** slate_true[pred_order] - 1) / np.log2(np.arange(2, k + 2)))

        # Ideal ranking
        ideal_order = np.argsort(-slate_true)[:k]
        idcg = np.sum((2 ** slate_true[ideal_order] - 1) / np.log2(np.arange(2, k + 2)))

        if idcg > 0:
            ndcg_scores.append(dcg / idcg)

    return float(np.mean(ndcg_scores)) if ndcg_scores else 0.0
