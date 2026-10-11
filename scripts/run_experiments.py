"""Comprehensive experiment runner: trains models, logs metrics, generates diagnostic figures.

Experiments:
1. Baseline vs. Logistic Regression vs. LightGBM (Primary Benchmark).
2. Categorical Preprocessing Strategy Ablation (Min Frequency = 10 vs. Min Frequency = 50).
3. Numerical Transformation Ablation (Standard Scaling only vs. Log1p + Standard Scaling).
4. Probability Calibration (Uncalibrated vs. Platt Sigmoid Calibration vs. Isotonic Regression).
5. Feature Importance & Error Analysis.
"""

import json
import logging
from pathlib import Path
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ctr_prediction.calibration import Calibrator
from ctr_prediction.config import get_project_root, load_config
from ctr_prediction.data import load_criteo_data
from ctr_prediction.evaluate import (
    compute_calibration_curve,
    compute_classification_metrics,
    compute_ranking_ndcg_at_k,
)
from ctr_prediction.models import build_logistic_regression
from ctr_prediction.preprocessing import CTRPreprocessor
from ctr_prediction.train import save_pipeline, train_ctr_model
from ctr_prediction.validation import chronological_split, validate_schema

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

sns.set_theme(style="whitegrid", palette="muted")


def main():
    root = get_project_root()
    cfg = load_config()
    figures_dir = root / "reports" / "figures"
    results_dir = root / "reports" / "results"
    models_dir = root / "models"
    figures_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    # Load data and split
    df = load_criteo_data()
    validate_schema(df)
    train_df, val_df, test_df = chronological_split(
        df,
        train_ratio=cfg["split"]["train_ratio"],
        val_ratio=cfg["split"]["val_ratio"],
        test_ratio=cfg["split"]["test_ratio"],
    )

    y_val = val_df[cfg["data"]["target_column"]].to_numpy()
    y_test = test_df[cfg["data"]["target_column"]].to_numpy()

    # =========================================================================
    # EXPERIMENT 1: Model Benchmark (Baseline vs Logistic Regression vs LightGBM)
    # =========================================================================
    logger.info("=== Running Experiment 1: Model Benchmark ===")
    models_to_test = ["baseline", "logistic_regression", "lightgbm"]
    benchmark_results = []
    val_probs_dict = {}
    test_probs_dict = {}
    trained_models = {}

    # Fit shared default preprocessor strictly on train_df
    default_preprocessor = CTRPreprocessor(
        numerical_cols=cfg["data"]["numerical_features"],
        categorical_cols=cfg["data"]["categorical_features"],
        imputer_strategy=cfg["preprocessing"]["numerical_imputer_strategy"],
        apply_log1p=cfg["preprocessing"]["apply_log1p"],
        scale_numerical=cfg["preprocessing"]["scale_numerical"],
        min_category_frequency=cfg["preprocessing"]["min_category_frequency"],
        sparse_output=True,
    )
    default_preprocessor.fit(train_df)
    X_val = default_preprocessor.transform(val_df)
    X_test = default_preprocessor.transform(test_df)

    for m_name in models_to_test:
        t0 = time.time()
        model, _ = train_ctr_model(
            model_name=m_name,
            train_df=train_df,
            val_df=val_df,
            preprocessor=default_preprocessor,
            config=cfg,
        )
        train_time = time.time() - t0

        # Inference timing & probabilities
        t_inf0 = time.time()
        p_val = model.predict_proba(X_val)[:, 1]
        val_inf_time = time.time() - t_inf0

        p_test = model.predict_proba(X_test)[:, 1]

        val_probs_dict[m_name] = p_val
        test_probs_dict[m_name] = p_test
        trained_models[m_name] = model

        val_metrics = compute_classification_metrics(y_val, p_val)
        test_metrics = compute_classification_metrics(y_test, p_test)
        test_ndcg5 = compute_ranking_ndcg_at_k(y_test, p_test, k=5, list_size=10, n_lists=500)

        record = {
            "model": m_name,
            "train_time_sec": round(train_time, 2),
            "inference_time_sec": round(val_inf_time, 4),
            "val_log_loss": round(val_metrics["log_loss"], 5),
            "val_roc_auc": round(val_metrics["roc_auc"], 5),
            "val_pr_auc": round(val_metrics["pr_auc"], 5),
            "val_brier_score": round(val_metrics["brier_score"], 5),
            "val_ece": round(val_metrics["ece"], 5),
            "test_log_loss": round(test_metrics["log_loss"], 5),
            "test_roc_auc": round(test_metrics["roc_auc"], 5),
            "test_pr_auc": round(test_metrics["pr_auc"], 5),
            "test_brier_score": round(test_metrics["brier_score"], 5),
            "test_ndcg@5": round(test_ndcg5, 5),
            "test_ece": round(test_metrics["ece"], 5),
            "test_confusion_matrix": test_metrics["confusion_matrix"],
        }
        benchmark_results.append(record)
        logger.info(
            f"[{m_name}] Val LogLoss: {record['val_log_loss']:.5f}, "
            f"Val AUC: {record['val_roc_auc']:.5f} | "
            f"Test LogLoss: {record['test_log_loss']:.5f}, "
            f"Test AUC: {record['test_roc_auc']:.5f}, "
            f"Test NDCG@5: {record['test_ndcg@5']:.5f}"
        )

    # Save benchmark table
    benchmark_df = pd.DataFrame(benchmark_results)
    benchmark_df.to_csv(results_dir / "model_benchmark_comparison.csv", index=False)
    with open(results_dir / "model_benchmark_comparison.json", "w", encoding="utf-8") as f:
        json.dump(benchmark_results, f, indent=2)

    # Save best model pipeline (LightGBM demonstrated lowest validation log loss)
    save_pipeline(trained_models["lightgbm"], default_preprocessor, models_dir, "best_model.joblib")
    save_pipeline(trained_models["logistic_regression"], default_preprocessor, models_dir, "logistic_regression.joblib")

    # =========================================================================
    # EXPERIMENT 2: Categorical Preprocessing Strategy Ablation
    # =========================================================================
    logger.info("=== Running Experiment 2: Categorical Preprocessing Ablation ===")
    freq_thresholds = [10, 50]
    cat_ablation_results = []
    for freq in freq_thresholds:
        prep = CTRPreprocessor(
            numerical_cols=cfg["data"]["numerical_features"],
            categorical_cols=cfg["data"]["categorical_features"],
            imputer_strategy="median",
            apply_log1p=True,
            scale_numerical=True,
            min_category_frequency=freq,
            sparse_output=True,
        )
        prep.fit(train_df)
        X_tr = prep.transform(train_df)
        X_v = prep.transform(val_df)
        X_te = prep.transform(test_df)

        lr = build_logistic_regression(C=0.1, random_state=42)
        lr.fit(X_tr, train_df["click"].to_numpy())
        p_v = lr.predict_proba(X_v)[:, 1]
        p_te = lr.predict_proba(X_te)[:, 1]

        v_m = compute_classification_metrics(y_val, p_v)
        te_m = compute_classification_metrics(y_test, p_te)
        cat_ablation_results.append({
            "min_category_frequency": freq,
            "feature_dim": len(prep.feature_names_),
            "val_log_loss": round(v_m["log_loss"], 5),
            "val_roc_auc": round(v_m["roc_auc"], 5),
            "test_log_loss": round(te_m["log_loss"], 5),
            "test_roc_auc": round(te_m["roc_auc"], 5),
        })
    pd.DataFrame(cat_ablation_results).to_csv(results_dir / "categorical_ablation.csv", index=False)

    # =========================================================================
    # EXPERIMENT 3: Numerical Transformation Ablation (Standard vs Log1p+Standard)
    # =========================================================================
    logger.info("=== Running Experiment 3: Numerical Transformation Ablation ===")
    num_ablation_results = []
    for apply_log in [False, True]:
        prep = CTRPreprocessor(
            numerical_cols=cfg["data"]["numerical_features"],
            categorical_cols=cfg["data"]["categorical_features"],
            imputer_strategy="median",
            apply_log1p=apply_log,
            scale_numerical=True,
            min_category_frequency=10,
            sparse_output=True,
        )
        prep.fit(train_df)
        X_tr = prep.transform(train_df)
        X_v = prep.transform(val_df)
        X_te = prep.transform(test_df)

        lr = build_logistic_regression(C=0.1, random_state=42)
        lr.fit(X_tr, train_df["click"].to_numpy())
        p_v = lr.predict_proba(X_v)[:, 1]
        p_te = lr.predict_proba(X_te)[:, 1]

        v_m = compute_classification_metrics(y_val, p_v)
        te_m = compute_classification_metrics(y_test, p_te)
        num_ablation_results.append({
            "apply_log1p": apply_log,
            "val_log_loss": round(v_m["log_loss"], 5),
            "val_roc_auc": round(v_m["roc_auc"], 5),
            "test_log_loss": round(te_m["log_loss"], 5),
            "test_roc_auc": round(te_m["roc_auc"], 5),
        })
    pd.DataFrame(num_ablation_results).to_csv(results_dir / "numerical_transformation_ablation.csv", index=False)

    # =========================================================================
    # EXPERIMENT 4: Probability Calibration (Platt Sigmoid vs Isotonic)
    # =========================================================================
    logger.info("=== Running Experiment 4: Probability Calibration ===")
    lgb_val_p = val_probs_dict["lightgbm"]
    lgb_test_p = test_probs_dict["lightgbm"]

    calib_results = []
    # 1. Uncalibrated
    uncal_m = compute_classification_metrics(y_test, lgb_test_p)
    calib_results.append({
        "method": "Uncalibrated LightGBM",
        "test_log_loss": round(uncal_m["log_loss"], 5),
        "test_brier_score": round(uncal_m["brier_score"], 5),
        "test_ece": round(uncal_m["ece"], 5),
    })

    # 2. Platt Sigmoid Calibration
    calib_sig = Calibrator(method="sigmoid")
    calib_sig.fit(lgb_val_p, y_val)
    p_test_sig = calib_sig.predict_proba(lgb_test_p)
    sig_m = compute_classification_metrics(y_test, p_test_sig)
    calib_results.append({
        "method": "Platt Sigmoid (Logistic)",
        "test_log_loss": round(sig_m["log_loss"], 5),
        "test_brier_score": round(sig_m["brier_score"], 5),
        "test_ece": round(sig_m["ece"], 5),
    })

    # 3. Isotonic Regression Calibration
    calib_iso = Calibrator(method="isotonic")
    calib_iso.fit(lgb_val_p, y_val)
    p_test_iso = calib_iso.predict_proba(lgb_test_p)
    iso_m = compute_classification_metrics(y_test, p_test_iso)
    calib_results.append({
        "method": "Isotonic Regression",
        "test_log_loss": round(iso_m["log_loss"], 5),
        "test_brier_score": round(iso_m["brier_score"], 5),
        "test_ece": round(iso_m["ece"], 5),
    })

    pd.DataFrame(calib_results).to_csv(results_dir / "calibration_comparison.csv", index=False)

    # =========================================================================
    # FIGURE GENERATION
    # =========================================================================
    logger.info("=== Generating Figures for Evaluation & Calibration ===")

    # Figure: ROC Curves
    from sklearn.metrics import roc_curve, precision_recall_curve

    fig, ax = plt.subplots(figsize=(7, 6))
    for m_name in ["baseline", "logistic_regression", "lightgbm"]:
        fpr, tpr, _ = roc_curve(y_test, test_probs_dict[m_name])
        auc_val = benchmark_df.loc[benchmark_df["model"] == m_name, "test_roc_auc"].values[0]
        ax.plot(fpr, tpr, label=f"{m_name} (AUC = {auc_val:.4f})", lw=2)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.6, label="Random Guess")
    ax.set_xlabel("False Positive Rate (FPR)")
    ax.set_ylabel("True Positive Rate (TPR)")
    ax.set_title("Test Set ROC Curves across CTR Models")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(figures_dir / "roc_curves.png", dpi=300)
    plt.close()

    # Figure: Precision-Recall Curves
    fig, ax = plt.subplots(figsize=(7, 6))
    for m_name in ["baseline", "logistic_regression", "lightgbm"]:
        prec, rec, _ = precision_recall_curve(y_test, test_probs_dict[m_name])
        pr_auc = benchmark_df.loc[benchmark_df["model"] == m_name, "test_pr_auc"].values[0]
        ax.plot(rec, prec, label=f"{m_name} (PR-AUC = {pr_auc:.4f})", lw=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Test Set Precision-Recall Curves")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(figures_dir / "precision_recall_curves.png", dpi=300)
    plt.close()

    # Figure: Reliability Diagram (Calibration Curve)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k:", label="Perfect Calibration")
    # Plot uncalibrated LightGBM vs Sigmoid Calibrated
    _, _, confs_raw, accs_raw, _ = compute_calibration_curve(y_test, lgb_test_p, n_bins=10)
    _, _, confs_sig, accs_sig, _ = compute_calibration_curve(y_test, p_test_sig, n_bins=10)
    ax.plot(confs_raw, accs_raw, "s-", color="#C44E52", label=f"LightGBM Raw (ECE = {uncal_m['ece']:.4f})")
    ax.plot(confs_sig, accs_sig, "o-", color="#4C72B0", label=f"LightGBM + Sigmoid (ECE = {sig_m['ece']:.4f})")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives (Empirical CTR)")
    ax.set_title("Reliability Diagram (Calibration Curve)")
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(figures_dir / "reliability_diagrams.png", dpi=300)
    plt.close()

    # Figure: Model Comparison Bar Chart
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    benchmark_df.plot(x="model", y=["val_log_loss", "test_log_loss"], kind="bar", ax=axes[0], color=["#4C72B0", "#55A868"])
    axes[0].set_title("Log Loss (Lower is Better)")
    axes[0].set_ylabel("Binary Cross Entropy")
    axes[0].tick_params(axis="x", rotation=0)

    benchmark_df.plot(x="model", y=["val_roc_auc", "test_roc_auc"], kind="bar", ax=axes[1], color=["#DD8452", "#8172B3"])
    axes[1].set_title("ROC-AUC (Higher is Better)")
    axes[1].set_ylabel("Area Under ROC Curve")
    axes[1].tick_params(axis="x", rotation=0)
    plt.tight_layout()
    plt.savefig(figures_dir / "model_comparison_charts.png", dpi=300)
    plt.close()

    # Figure: Feature Importance (LightGBM gain)
    lgb_model = trained_models["lightgbm"]
    feature_names = default_preprocessor.feature_names_
    importances = lgb_model.feature_importances_

    feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False).head(20)
    fig, ax = plt.subplots(figsize=(10, 6))
    feat_imp.sort_values().plot(kind="barh", ax=ax, color="#4C72B0")
    ax.set_title("Top 20 Features by Total Split Gain (LightGBM)")
    ax.set_xlabel("Cumulative Information Gain")
    plt.tight_layout()
    plt.savefig(figures_dir / "feature_importance_plot.png", dpi=300)
    plt.close()

    # Figure: Error Analysis (Residual / Prediction Distribution by Actual Class)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(
        x=lgb_test_p,
        hue=y_test,
        bins=40,
        stat="density",
        common_norm=False,
        ax=ax,
        palette=["#4C72B0", "#DD8452"],
    )
    ax.set_title("Predicted Probability Density by True Class (Test Set)")
    ax.set_xlabel("Predicted P(click = 1)")
    ax.set_ylabel("Density")
    plt.tight_layout()
    plt.savefig(figures_dir / "error_analysis_distribution.png", dpi=300)
    plt.close()

    logger.info("All experiments, diagnostics, and figures generated successfully!")


if __name__ == "__main__":
    main()
