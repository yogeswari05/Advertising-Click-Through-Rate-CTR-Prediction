"""Unit tests for feature preprocessing and leakage prevention."""

import numpy as np
import pandas as pd
from scipy import sparse
from ctr_prediction.preprocessing import CTRPreprocessor


def test_preprocessor_fit_transform(synthetic_criteo_df):
    num_cols = [f"I{i}" for i in range(1, 14)]
    cat_cols = [f"C{i}" for i in range(1, 27)]

    preprocessor = CTRPreprocessor(
        numerical_cols=num_cols,
        categorical_cols=cat_cols,
        min_category_frequency=2,
        sparse_output=True,
    )
    X_train = preprocessor.fit_transform(synthetic_criteo_df)

    assert sparse.issparse(X_train)
    assert X_train.shape[0] == len(synthetic_criteo_df)
    assert X_train.shape[1] == len(preprocessor.feature_names_)


def test_preprocessor_unseen_categories_and_missing(synthetic_criteo_df):
    num_cols = [f"I{i}" for i in range(1, 14)]
    cat_cols = [f"C{i}" for i in range(1, 27)]

    preprocessor = CTRPreprocessor(
        numerical_cols=num_cols,
        categorical_cols=cat_cols,
        min_category_frequency=2,
    )
    preprocessor.fit(synthetic_criteo_df)

    # Create new unseen test dataframe with novel categories and all-missing numericals
    unseen_df = pd.DataFrame({
        **{f"I{i}": [np.nan] for i in range(1, 14)},
        **{f"C{i}": [f"totally_unseen_cat_{i}"] for i in range(1, 27)},
    })

    X_unseen = preprocessor.transform(unseen_df)
    assert sparse.issparse(X_unseen)
    assert X_unseen.shape[0] == 1
    assert X_unseen.shape[1] == len(preprocessor.feature_names_)
    assert not np.isnan(X_unseen.data).any()
