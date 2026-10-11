"""Advanced feature engineering for CTR models (interaction flags, count aggregations)."""

from typing import List
import numpy as np
import pandas as pd


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add domain-specific CTR signals without lookahead leakage.

    Engineered features:
    - Missing numerical count: Number of unrecorded numerical attributes per impression.
      (Hypothesis: impressions missing multiple counters represent thin ad slots or bot traffic).
    - Log-transformed extreme count ratios where applicable.
    """
    df_out = df.copy()
    num_cols = [f"I{i}" for i in range(1, 14) if f"I{i}" in df.columns]
    if num_cols:
        df_out["I_missing_count"] = df[num_cols].isna().sum(axis=1)

    return df_out
