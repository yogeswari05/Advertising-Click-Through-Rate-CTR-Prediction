"""Shared fixtures for testing CTR prediction suite."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_criteo_df() -> pd.DataFrame:
    """Provide small, completely synthetic DataFrame matching Criteo schema for unit tests."""
    np.random.seed(42)
    n = 200
    num_cols = {f"I{i}": np.random.choice([np.nan, 0, 1, 5, 20, 100], size=n) for i in range(1, 14)}
    cat_cols = {f"C{i}": np.random.choice([None, "cat_a", "cat_b", "cat_c", "rare_x"], size=n) for i in range(1, 27)}
    targets = {"click": np.random.choice([0, 1], p=[0.75, 0.25], size=n)}

    data = {**targets, **num_cols, **cat_cols}
    return pd.DataFrame(data)
