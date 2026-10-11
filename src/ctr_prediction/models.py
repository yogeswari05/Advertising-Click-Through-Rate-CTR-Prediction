"""CTR Prediction Models: Baseline, Regularized Logistic Regression, and Gradient Boosted Trees."""

import logging
from typing import Any, Dict, Optional, Tuple
import lightgbm as lgb
import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)


class ConstantProbabilityBaseline(BaseEstimator, ClassifierMixin):
    """Majority-class/Empirical prior baseline model.

    Always predicts the constant marginal click probability observed in the training split:
        P(click = 1) = y_train.mean()

    Serving as the fundamental benchmark in ad click modeling, any ML model must strictly
    outperform this baseline in Log Loss and Brier Score.
    """

    def __init__(self):
        self.prior_probability_: float = 0.5
        self.classes_ = np.array([0, 1])

    def fit(self, X, y):
        y_arr = np.asarray(y)
        self.prior_probability_ = float(np.mean(y_arr))
        logger.info(f"Fitted ConstantProbabilityBaseline. Empirical train prior CTR: {self.prior_probability_:.5f}")
        return self

    def predict_proba(self, X) -> np.ndarray:
        n_samples = X.shape[0] if hasattr(X, "shape") else len(X)
        p1 = np.full((n_samples, 1), self.prior_probability_, dtype=np.float64)
        p0 = 1.0 - p1
        return np.hstack([p0, p1])

    def predict(self, X, threshold: float = 0.5) -> np.ndarray:
        probas = self.predict_proba(X)[:, 1]
        return (probas >= threshold).astype(int)


def build_logistic_regression(
    C: float = 0.1,
    solver: str = "lbfgs",
    max_iter: int = 1000,
    random_state: int = 42,
) -> LogisticRegression:
    """Build regularized Logistic Regression for CTR prediction.

    Parameters
    ----------
    C : float, default=0.1
        Inverse of regularization strength (1 / lambda).
    solver : str, default='lbfgs'
        Optimization algorithm.
    max_iter : int, default=1000
        Maximum iterations to reach gradient tolerance.
    random_state : int, default=42
        Reproducibility seed.

    Returns
    -------
    LogisticRegression
        Scikit-learn estimator instance.
    """
    return LogisticRegression(
        penalty="l2",
        C=C,
        solver=solver,
        max_iter=max_iter,
        random_state=random_state,
        tol=1e-4,
    )


def build_lightgbm_model(
    n_estimators: int = 300,
    learning_rate: float = 0.05,
    num_leaves: int = 31,
    min_child_samples: int = 50,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    random_state: int = 42,
    n_jobs: int = -1,
) -> lgb.LGBMClassifier:
    """Build LightGBM classifier for CTR prediction.

    LightGBM natively handles sparse CSR matrices efficiently and constructs tree
    ensembles using histogram-based bucketed splits with leaf-wise expansion.
    """
    return lgb.LGBMClassifier(
        objective="binary",
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        min_child_samples=min_child_samples,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=random_state,
        n_jobs=n_jobs,
        importance_type="gain",
        verbose=-1,
    )
