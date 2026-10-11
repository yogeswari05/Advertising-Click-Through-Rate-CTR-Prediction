"""Inference and batch prediction module with strict schema preservation."""

import logging
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from ctr_prediction.config import get_project_root
from ctr_prediction.train import load_pipeline

logger = logging.getLogger(__name__)


class CTRPredictor:
    """Production CTR Predictor serving predictions using serialized pipeline artifacts."""

    def __init__(self, pipeline_path: Optional[Path | str] = None):
        if pipeline_path is None:
            pipeline_path = get_project_root() / "models" / "best_model.joblib"
        self.model, self.preprocessor = load_pipeline(pipeline_path)
        logger.info(f"Loaded CTRPredictor from {pipeline_path}")

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Generate predicted click probabilities P(click = 1 | X).

        Parameters
        ----------
        df : pd.DataFrame
            Unseen impression feature records.

        Returns
        -------
        np.ndarray
            1D array of predicted click probabilities bounded in [0, 1].
        """
        # Ensure all expected columns exist (even if missing/NaN)
        for col in self.preprocessor.numerical_cols + self.preprocessor.categorical_cols:
            if col not in df.columns:
                df[col] = np.nan

        X_transformed = self.preprocessor.transform(df)
        probas = self.model.predict_proba(X_transformed)[:, 1]
        return probas

    def predict(self, df: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Predict binary click labels {0, 1}."""
        probas = self.predict_proba(df)
        return (probas >= threshold).astype(int)
