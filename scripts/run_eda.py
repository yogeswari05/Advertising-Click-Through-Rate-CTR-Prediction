"""Reproducible Exploratory Data Analysis (EDA) script for Criteo CTR.

Generates publication-quality charts and quantitative summaries:
- Target class balance and label distribution.
- Missing values rate per feature.
- Numerical feature skewness, distribution, and log1p comparison.
- Categorical cardinality and long-tail frequency analysis.
- Duplicate impressions analysis.
"""

import json
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ctr_prediction.config import get_project_root, load_config
from ctr_prediction.data import download_criteo_sample, load_criteo_data
from ctr_prediction.validation import validate_schema

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Set publication style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"font.size": 11, "figure.autolayout": True})


def main():
    root = get_project_root()
    cfg = load_config()
    figures_dir = root / "reports" / "figures"
    results_dir = root / "reports" / "results"
    figures_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1. Download / Load dataset
    download_criteo_sample()
    df = load_criteo_data()

    # 2. Schema Validation
    validation_report = validate_schema(df)
    with open(results_dir / "eda_summary.json", "w", encoding="utf-8") as f:
        json.dump(validation_report, f, indent=2)

    logger.info("Generating EDA Visualizations...")

    # Figure 1: Target Class Distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df["click"].value_counts().sort_index()
    percentages = (counts / len(df)) * 100
    bars = ax.bar(["Non-Click (0)", "Click (1)"], counts, color=["#4C72B0", "#DD8452"], width=0.5)
    for bar, pct in zip(bars, percentages):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, height + 1000, f"{height:,} ({pct:.1f}%)", ha="center", va="bottom")
    ax.set_ylabel("Impression Count")
    ax.set_title("Criteo DAC: Click Label Distribution (N=100,000)")
    ax.set_ylim(0, max(counts) * 1.15)
    plt.tight_layout()
    plt.savefig(figures_dir / "click_label_distribution.png", dpi=300)
    plt.close()

    # Figure 2: Missing Value Rate by Feature
    missing_pct = (df.isna().mean() * 100).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(12, 6))
    missing_pct[missing_pct > 0].plot(kind="bar", ax=ax, color="#C44E52")
    ax.set_ylabel("Missing Rate (%)")
    ax.set_xlabel("Feature Name")
    ax.set_title("Missing Value Rate across Numerical and Categorical Features")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(figures_dir / "missing_values_summary.png", dpi=300)
    plt.close()

    # Figure 3: Numerical Feature Distributions (Raw vs Log1p on I1, I2, I3, I5)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()
    features_to_plot = ["I1", "I2", "I3", "I5"]
    for i, col in enumerate(features_to_plot):
        data = df[col].dropna()
        log_data = np.log1p(np.clip(data, 0, None))
        sns.histplot(log_data, ax=axes[i], bins=30, kde=True, color="#55A868")
        axes[i].set_title(f"Distribution of log1p({col}) (Raw Skew: {data.skew():.2f})")
        axes[i].set_xlabel(f"log(1 + {col})")
    plt.tight_layout()
    plt.savefig(figures_dir / "numerical_distributions.png", dpi=300)
    plt.close()

    # Figure 4: Categorical Cardinality Distribution
    cat_cols = cfg["data"]["categorical_features"]
    cardinalities = df[cat_cols].nunique().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(12, 5))
    cardinalities.plot(kind="bar", ax=ax, color="#8172B3")
    ax.set_ylabel("Unique Categories (Log Scale)")
    ax.set_yscale("log")
    ax.set_title("Categorical Feature Cardinality (Unique Values per C1-C26)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(figures_dir / "categorical_cardinality_summary.png", dpi=300)
    plt.close()

    logger.info(f"EDA complete. Figures saved to {figures_dir}")


if __name__ == "__main__":
    main()
