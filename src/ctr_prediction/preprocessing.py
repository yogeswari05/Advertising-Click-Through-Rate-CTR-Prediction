"""Feature preprocessing pipelines with strict leakage prevention and sparse encoding."""

import logging
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)


class Log1pClipper(BaseEstimator, TransformerMixin):
    """Applies log1p transformation to non-negative numerical features with clipping.

    CTR count features (e.g., impression counts, query frequencies) are heavily right-skewed.
    Applying log1p: y = log(1 + max(0, x)) dampens heavy-tailed extremes, stabilizing
    linear gradients and split search.
    """

    def __init__(self, clip_min: float = 0.0):
        self.clip_min = clip_min

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_arr = np.asarray(X, dtype=np.float64)
        X_clipped = np.clip(X_arr, self.clip_min, None)
        return np.log1p(X_clipped)


class CategoricalCleaner(BaseEstimator, TransformerMixin):
    """Converts missing values and unseen types in categorical columns to a standard string token."""

    def __init__(self, missing_token: str = "__MISSING__"):
        self.missing_token = missing_token

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return X.fillna(self.missing_token).astype(str).to_numpy()
        X_arr = np.where(pd.isna(X), self.missing_token, X).astype(str)
        return X_arr


class CTRPreprocessor:
    """End-to-end CTR Preprocessor that fits only on training data to prevent leakage.

    Transforms:
    - Numerical features (I1-I13): Imputation -> optional log1p -> optional StandardScaler
    - Categorical features (C1-C26): Missing token filling -> Sparse One-Hot Encoding
      with minimum category frequency threshold to prevent explosion of sparse dimensions.
    """

    def __init__(
        self,
        numerical_cols: List[str],
        categorical_cols: List[str],
        imputer_strategy: str = "median",
        apply_log1p: bool = True,
        scale_numerical: bool = True,
        min_category_frequency: int = 10,
        sparse_output: bool = True,
    ):
        self.numerical_cols = numerical_cols
        self.categorical_cols = categorical_cols
        self.imputer_strategy = imputer_strategy
        self.apply_log1p = apply_log1p
        self.scale_numerical = scale_numerical
        self.min_category_frequency = min_category_frequency
        self.sparse_output = sparse_output

        # Sub-pipelines
        num_steps = [("imputer", SimpleImputer(strategy=self.imputer_strategy))]
        if self.apply_log1p:
            num_steps.append(("log1p", Log1pClipper()))
        if self.scale_numerical:
            num_steps.append(("scaler", StandardScaler()))
        self.num_pipeline = Pipeline(num_steps)

        self.cat_cleaner = CategoricalCleaner(missing_token="__MISSING__")
        self.cat_encoder = OneHotEncoder(
            handle_unknown="ignore",
            min_frequency=self.min_category_frequency,
            sparse_output=True,  # Always use sparse internally to prevent 8GB RAM exhaustion
        )

        self.is_fitted = False
        self.feature_names_: List[str] = []

    def fit(self, df: pd.DataFrame, y=None):
        """Fit preprocessor strictly on training data."""
        logger.info("Fitting CTRPreprocessor on training data...")
        X_num = df[self.numerical_cols]
        X_cat = df[self.categorical_cols]

        self.num_pipeline.fit(X_num)
        X_cat_cleaned = self.cat_cleaner.fit_transform(X_cat)
        self.cat_encoder.fit(X_cat_cleaned)

        # Store feature names
        num_feat_names = list(self.numerical_cols)
        cat_feat_names = list(self.cat_encoder.get_feature_names_out(self.categorical_cols))
        self.feature_names_ = num_feat_names + cat_feat_names

        self.is_fitted = True
        logger.info(
            f"Preprocessor fitted. Transformed feature dimensionality: {len(self.feature_names_)} "
            f"({len(num_feat_names)} numerical, {len(cat_feat_names)} categorical one-hot)"
        )
        return self

    def transform(self, df: pd.DataFrame) -> sparse.csr_matrix | np.ndarray:
        """Transform dataset into sparse feature matrix."""
        if not self.is_fitted:
            raise RuntimeError("CTRPreprocessor must be fitted before calling transform().")

        X_num = df[self.numerical_cols]
        X_cat = df[self.categorical_cols]

        X_num_trans = self.num_pipeline.transform(X_num)
        X_cat_cleaned = self.cat_cleaner.transform(X_cat)
        X_cat_trans = self.cat_encoder.transform(X_cat_cleaned)

        # Combine numerical dense array and categorical sparse matrix into CSR matrix
        X_combined = sparse.hstack([sparse.csr_matrix(X_num_trans), X_cat_trans], format="csr")

        if not self.sparse_output:
            return X_combined.toarray()

        return X_combined

    def fit_transform(self, df: pd.DataFrame, y=None):
        return self.fit(df, y).transform(df)
