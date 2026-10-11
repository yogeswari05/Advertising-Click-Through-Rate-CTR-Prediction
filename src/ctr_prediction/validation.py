"""Dataset validation, schema enforcement, duplicate detection, and chronological splitting."""

import logging
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

EXPECTED_NUMERICAL_COLS = [f"I{i}" for i in range(1, 14)]
EXPECTED_CATEGORICAL_COLS = [f"C{i}" for i in range(1, 27)]
EXPECTED_TARGET_COL = "click"
ALL_EXPECTED_COLS = [EXPECTED_TARGET_COL] + EXPECTED_NUMERICAL_COLS + EXPECTED_CATEGORICAL_COLS


def validate_schema(
    df: pd.DataFrame,
    target_col: str = EXPECTED_TARGET_COL,
    numerical_cols: List[str] = EXPECTED_NUMERICAL_COLS,
    categorical_cols: List[str] = EXPECTED_CATEGORICAL_COLS,
) -> Dict[str, any]:
    """Validate DataFrame against expected Criteo schema.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    target_col : str
        Target label column name.
    numerical_cols : list of str
        Expected numerical column names.
    categorical_cols : list of str
        Expected categorical column names.

    Returns
    -------
    dict
        Validation report containing status and summary statistics.
    """
    missing_cols = [c for c in [target_col] + numerical_cols + categorical_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Schema validation failed. Missing required columns: {missing_cols}")

    # Check target label values
    unique_targets = set(df[target_col].dropna().unique())
    if not unique_targets.issubset({0, 1}):
        raise ValueError(f"Invalid target values detected: {unique_targets}. Expected subset of {{0, 1}}.")

    # Calculate missing rates
    missing_rates = df.isna().mean().to_dict()

    # Calculate duplicate rows
    duplicate_count = int(df.duplicated().sum())

    # Label distribution
    class_counts = df[target_col].value_counts().to_dict()
    pos_rate = float(df[target_col].mean())

    report = {
        "num_rows": len(df),
        "num_cols": len(df.columns),
        "missing_columns": missing_cols,
        "duplicate_rows": duplicate_count,
        "class_counts": class_counts,
        "positive_rate": pos_rate,
        "missing_rates": missing_rates,
    }
    logger.info(
        f"Schema validation passed: {report['num_rows']} rows, "
        f"{report['num_cols']} cols, positive rate: {pos_rate:.4f}, "
        f"duplicates: {duplicate_count}"
    )
    return report


def chronological_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split dataset chronologically preserving row order from impression logs.

    In production ad-serving systems, impression logs arrive sequentially. Randomly
    shuffling rows across time causes future ad impression behavior to leak into past
    feature distributions, invalidating temporal offline evaluation. Hence, an order-preserving
    split is the scientifically sound standard.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    train_ratio : float, default=0.70
        Fraction of data for training.
    val_ratio : float, default=0.15
        Fraction of data for validation.
    test_ratio : float, default=0.15
        Fraction of data for testing.

    Returns
    -------
    train_df, val_df, test_df : tuple of pd.DataFrame
        Chronologically partitioned splits.
    """
    total_ratio = train_ratio + val_ratio + test_ratio
    if not np.isclose(total_ratio, 1.0):
        raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio}")

    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end].copy().reset_index(drop=True)
    val_df = df.iloc[train_end:val_end].copy().reset_index(drop=True)
    test_df = df.iloc[val_end:].copy().reset_index(drop=True)

    logger.info(
        f"Chronological split complete: Train={len(train_df)} ({len(train_df)/n:.1%}), "
        f"Val={len(val_df)} ({len(val_df)/n:.1%}), "
        f"Test={len(test_df)} ({len(test_df)/n:.1%})"
    )

    return train_df, val_df, test_df
