"""Unit tests for dataset loading, schema validation, and chronological splitting."""

import pandas as pd
import pytest
from ctr_prediction.validation import chronological_split, validate_schema


def test_validate_schema_success(synthetic_criteo_df):
    report = validate_schema(synthetic_criteo_df)
    assert report["num_rows"] == 200
    assert report["num_cols"] == 40
    assert "class_counts" in report
    assert 0 <= report["positive_rate"] <= 1.0


def test_validate_schema_missing_column(synthetic_criteo_df):
    corrupted_df = synthetic_criteo_df.drop(columns=["I1"])
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_schema(corrupted_df)


def test_validate_schema_invalid_label(synthetic_criteo_df):
    corrupted_df = synthetic_criteo_df.copy()
    corrupted_df.loc[0, "click"] = 2  # Invalid label
    with pytest.raises(ValueError, match="Invalid target values"):
        validate_schema(corrupted_df)


def test_chronological_split(synthetic_criteo_df):
    train_df, val_df, test_df = chronological_split(
        synthetic_criteo_df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15
    )
    assert len(train_df) == 140
    assert len(val_df) == 30
    assert len(test_df) == 30
    assert len(train_df) + len(val_df) + len(test_df) == len(synthetic_criteo_df)

    # Verify order preservation (no random shuffling)
    pd.testing.assert_series_equal(train_df.iloc[0], synthetic_criteo_df.iloc[0], check_names=False)
    pd.testing.assert_series_equal(test_df.iloc[-1], synthetic_criteo_df.iloc[-1], check_names=False)
