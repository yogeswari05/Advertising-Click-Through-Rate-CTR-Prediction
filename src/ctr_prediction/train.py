"""Model training and artifact serialization coordinator."""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from ctr_prediction.config import get_project_root, load_config
from ctr_prediction.models import (
    ConstantProbabilityBaseline,
    build_lightgbm_model,
    build_logistic_regression,
)
from ctr_prediction.preprocessing import CTRPreprocessor

logger = logging.getLogger(__name__)


def train_ctr_model(
    model_name: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    preprocessor: Optional[CTRPreprocessor] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[Any, CTRPreprocessor]:
    """Train a CTR model end-to-end on chronological training and validation splits.

    Parameters
    ----------
    model_name : str
        'baseline', 'logistic_regression', or 'lightgbm'
    train_df : pd.DataFrame
        Training data split.
    val_df : pd.DataFrame
        Validation data split.
    preprocessor : CTRPreprocessor, optional
        Pre-fitted preprocessor. If None, fitted on train_df.
    config : dict, optional
        Configuration parameters.

    Returns
    -------
    model, preprocessor : tuple
        Trained model instance and fitted preprocessor.
    """
    if config is None:
        config = load_config()

    target_col = config["data"]["target_column"]
    num_cols = config["data"]["numerical_features"]
    cat_cols = config["data"]["categorical_features"]

    y_train = train_df[target_col].to_numpy()
    y_val = val_df[target_col].to_numpy()

    if preprocessor is None:
        preprocessor = CTRPreprocessor(
            numerical_cols=num_cols,
            categorical_cols=cat_cols,
            imputer_strategy=config["preprocessing"]["numerical_imputer_strategy"],
            apply_log1p=config["preprocessing"]["apply_log1p"],
            scale_numerical=config["preprocessing"]["scale_numerical"],
            min_category_frequency=config["preprocessing"]["min_category_frequency"],
            sparse_output=config["preprocessing"]["sparse_output"],
        )
        preprocessor.fit(train_df)

    X_train = preprocessor.transform(train_df)
    X_val = preprocessor.transform(val_df)

    if model_name == "baseline":
        model = ConstantProbabilityBaseline()
        model.fit(X_train, y_train)

    elif model_name == "logistic_regression":
        lr_cfg = config["models"]["logistic_regression"]
        model = build_logistic_regression(
            C=lr_cfg["C"],
            solver=lr_cfg["solver"],
            max_iter=lr_cfg["max_iter"],
            random_state=config["models"]["random_state"],
        )
        logger.info("Training Logistic Regression...")
        model.fit(X_train, y_train)

    elif model_name == "lightgbm":
        lgb_cfg = config["models"]["lightgbm"]
        model = build_lightgbm_model(
            n_estimators=lgb_cfg["n_estimators"],
            learning_rate=lgb_cfg["learning_rate"],
            num_leaves=lgb_cfg["num_leaves"],
            min_child_samples=lgb_cfg["min_child_samples"],
            subsample=lgb_cfg["subsample"],
            colsample_bytree=lgb_cfg["colsample_bytree"],
            random_state=config["models"]["random_state"],
            n_jobs=lgb_cfg["n_jobs"],
        )
        logger.info("Training LightGBM with validation early stopping...")
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=lgb_cfg["early_stopping_rounds"], verbose=False)],
        )
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    return model, preprocessor


def save_pipeline(model: Any, preprocessor: CTRPreprocessor, save_dir: Path | str, filename: str = "pipeline.joblib"):
    """Serialize model and preprocessing pipeline together."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / filename
    joblib.dump({"model": model, "preprocessor": preprocessor}, out_path)
    logger.info(f"Saved pipeline artifact to {out_path}")
    return out_path


def load_pipeline(pipeline_path: Path | str) -> Tuple[Any, CTRPreprocessor]:
    """Deserialize model and preprocessor artifact."""
    artifact = joblib.load(pipeline_path)
    return artifact["model"], artifact["preprocessor"]
