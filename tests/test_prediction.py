"""Unit tests for trained pipeline serialization, loading, and batch prediction."""

from pathlib import Path
import numpy as np
import pandas as pd
from ctr_prediction.models import ConstantProbabilityBaseline
from ctr_prediction.predict import CTRPredictor
from ctr_prediction.preprocessing import CTRPreprocessor
from ctr_prediction.train import save_pipeline


def test_save_load_predict_end_to_end(synthetic_criteo_df, tmp_path: Path):
    num_cols = [f"I{i}" for i in range(1, 14)]
    cat_cols = [f"C{i}" for i in range(1, 27)]

    preprocessor = CTRPreprocessor(
        numerical_cols=num_cols,
        categorical_cols=cat_cols,
        min_category_frequency=2,
    )
    preprocessor.fit(synthetic_criteo_df)

    model = ConstantProbabilityBaseline()
    model.fit(preprocessor.transform(synthetic_criteo_df), synthetic_criteo_df["click"])

    artifact_path = save_pipeline(model, preprocessor, tmp_path, "test_pipeline.joblib")
    assert artifact_path.exists()

    # Predict using saved artifact
    predictor = CTRPredictor(pipeline_path=artifact_path)
    test_samples = synthetic_criteo_df.head(5).copy()

    probs = predictor.predict_proba(test_samples)
    labels = predictor.predict(test_samples, threshold=0.5)

    assert len(probs) == 5
    assert len(labels) == 5
    assert ((probs >= 0.0) & (probs <= 1.0)).all()
    assert set(labels).issubset({0, 1})
