"""Post-hoc probability calibration routines (Platt scaling and Isotonic regression)."""

import logging
from typing import Literal
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)


class Calibrator:
    """Post-hoc probability calibrator trained strictly on validation split."""

    def __init__(self, method: Literal["sigmoid", "isotonic"] = "sigmoid"):
        self.method = method
        self.calibrator_ = None

    def fit(self, val_probs: np.ndarray, y_val: np.ndarray):
        """Fit calibration function on validation probabilities without leaking test data."""
        val_probs = np.asarray(val_probs, dtype=np.float64).reshape(-1, 1)
        y_val = np.asarray(y_val, dtype=int)

        if self.method == "sigmoid":
            # Platt Scaling: Logistic regression on validation scores
            self.calibrator_ = LogisticRegression(solver="lbfgs", C=1.0)
            self.calibrator_.fit(val_probs, y_val)
        elif self.method == "isotonic":
            # Non-parametric isotonic regression
            self.calibrator_ = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            self.calibrator_.fit(val_probs.ravel(), y_val)
        else:
            raise ValueError(f"Unknown calibration method: {self.method}")

        logger.info(f"Fitted {self.method} calibrator on validation split.")
        return self

    def predict_proba(self, probs: np.ndarray) -> np.ndarray:
        """Calibrate probabilities."""
        probs = np.asarray(probs, dtype=np.float64)
        if self.method == "sigmoid":
            return self.calibrator_.predict_proba(probs.reshape(-1, 1))[:, 1]
        elif self.method == "isotonic":
            return self.calibrator_.predict(probs.ravel())
        return probs
