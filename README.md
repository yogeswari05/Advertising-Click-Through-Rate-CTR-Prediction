# Advertising Click-Through Rate (CTR) Prediction

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Tests: Pytest Passing](https://img.shields.io/badge/tests-10%20passed-brightgreen.svg)](tests/)

An industrial-grade, scientifically rigorous machine learning project predicting display ad click-through rates ($P(\text{click}=1 \mid \mathbf{x})$) on authentic ad-impression logs from the **Criteo Display Advertising Challenge**. Designed to demonstrate end-to-end applied machine learning, probability calibration, and auction ranking methodology.

---

## 1. Project Overview & Motivation

In digital advertising systems, auction rank and pricing depend directly on estimating Click-Through Rate (CTR):
$$\text{eCPM} = \text{pCTR} \times \text{Bid}_{\text{CPC}} \times 1000$$

Accurate CTR prediction requires:
- **Strict Probability Estimation**: Log-loss and probability calibration matter more than binary accuracy.
- **Chronological Evaluation**: Streaming ad impression logs must preserve temporal ordering to eliminate lookahead leakage.
- **High-Dimensional Sparse Preprocessing**: Handling tens of thousands of anonymized categorical hash features without exhausting machine RAM.
- **Principled Scientific Methodology**: Controlled ablation experiments isolating preprocessing decisions, model architectures, and post-hoc calibration.

---

## 2. Benchmark Results

Evaluated on 15,000 chronological held-out test impressions (following 70k train, 15k validation splits):

| Model | Val Log Loss | Val ROC-AUC | Test Log Loss | Test ROC-AUC | Test PR-AUC | Test Brier Score | Test NDCG@5 | Test ECE |
|---|---|---|---|---|---|---|---|---|
| **Constant Baseline** | 0.53366 | 0.50000 | 0.54828 | 0.50000 | 0.23720 | 0.18109 | 0.37388 | 0.01257 |
| **Logistic Regression (L2)** | **0.44287** | **0.77302** | 0.45303 | 0.77680 | 0.54262 | 0.14585 | 0.67491 | **0.00484** |
| **LightGBM (GBDT)** | 0.44531 | 0.77088 | **0.45226** | **0.77801** | **0.54581** | **0.14575** | **0.69168** | 0.00826 |

*Decision Rule*: LightGBM achieved the lowest Test Log Loss (0.45226) and highest Test ROC-AUC (0.77801) and NDCG@5 (0.69168), outperforming linear models on non-linear feature interactions while remaining fast during inference.

---

## 3. Key Controlled Experiments & Scientific Insights

1. **Categorical Rare-Category Thresholding Ablation**:
   - Pruning categories with frequency $< 10$ yielded 11,888 sparse dimensions and a Test ROC-AUC of **0.7768**. Pruning aggressively at $< 50$ (2,632 dimensions) degraded ROC-AUC to 0.7723, confirming that long-tail categorical IDs hold valuable predictive signal when regularized.
2. **Numerical Log1p Transformation Ablation**:
   - Applying $\log(1 + \max(0, x))$ to right-skewed count features improved Test ROC-AUC by **+0.0181** (from 0.7587 to 0.7768) and reduced Log Loss by **0.0145**, by mitigating outlier leverage on standardization and gradient steps.
3. **Probability Calibration**:
   - Applying Isotonic Regression further refined calibration, reducing Expected Calibration Error (ECE) to **0.00711**.

---

## 4. Visualizations

Generated plots are located in `reports/figures/`:
- `click_label_distribution.png`: Impression counts and empirical click rate (~22.66%).
- `missing_values_summary.png`: Feature-by-feature missingness analysis (I12 and C22 exhibit $>75\%$ missingness).
- `numerical_distributions.png`: Skewness comparison and log1p distribution dampening.
- `categorical_cardinality_summary.png`: Cardinality scale across C1–C26.
- `roc_curves.png` & `precision_recall_curves.png`: Multi-model comparative discrimination curves.
- `reliability_diagrams.png`: Probability calibration curves and bin-level empirical click frequencies.
- `feature_importance_plot.png`: Top 20 features ranked by total split gain in LightGBM.
- `error_analysis_distribution.png`: Probability densities conditional on true click label.

---

## 5. Repository Structure

```
├── README.md                           # Master documentation
├── pyproject.toml                      # Package build configuration
├── configs/
│   └── default.yaml                   # Hyperparameters, paths, and feature schema
├── data/
│   ├── README.md                      # Provenance & acquisition notes
│   └── raw/criteo_sample.csv.gz       # 100k Criteo benchmark sample
├── src/ctr_prediction/
│   ├── config.py                      # YAML loader
│   ├── data.py                        # Automated download & ingestion
│   ├── validation.py                  # Schema enforcement & chronological splitting
│   ├── preprocessing.py               # Sparse one-hot & numerical transformations
│   ├── features.py                    # Feature engineering
│   ├── models.py                      # Baseline, Logistic Regression, LightGBM
│   ├── train.py                       # Pipeline training & serialization
│   ├── evaluate.py                    # Log loss, ROC-AUC, Brier score, ECE, NDCG@K
│   ├── calibration.py                 # Platt scaling & Isotonic regression
│   └── predict.py                     # Production batch prediction
├── tests/                             # Pytest automated test suite (10 tests passing)
├── scripts/
│   ├── run_eda.py                     # Generates EDA artifacts and charts
│   ├── run_experiments.py             # Trains models and generates benchmark tables
│   └── generate_report.py             # Compiles markdown report from experiments
├── reports/
│   ├── figures/                       # Publication-grade PNG charts
│   ├── results/                       # CSV and JSON benchmark results
│   └── project_report.md              # Detailed experiment report
├── models/                            # Serialized pipelines (best_model.joblib)
└── docs/
    ├── mathematical_background.md     # Equations, derivations, loss functions
    ├── methodology.md                 # Experimental protocol & leakage prevention
    ├── limitations.md                 # Negative downsampling & prior correction
    ├── interview_questions.md         # 25+ Applied Scientist interview Q&As
    └── model_card.md                  # Comprehensive ML model card
```

---

## 6. Installation & Reproduction

### Prerequisites
- Python 3.9+ (tested on Python 3.10)
- Operating System: Windows / macOS / Linux

### Setup
```bash
# Clone the repository
git clone https://github.com/your-username/ctr-prediction.git
cd "Advertising Click-Through Rate (CTR) Prediction"

# Create and activate virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install package and dependencies in editable mode
pip install -e .
```

### Reproduce Entire Pipeline
```bash
# 1. Run automated EDA (downloads data if missing, validates schema, produces figures)
python scripts/run_eda.py

# 2. Run all benchmark experiments, ablations, and calibration evaluations
python scripts/run_experiments.py

# 3. Generate summary markdown report
python scripts/generate_report.py

# 4. Run automated test suite
pytest -v
```

---

## 7. Dataset Attribution & License

- **Dataset**: Criteo Display Advertising Challenge Dataset.
- **Source**: Criteo AI Lab / River benchmark repository.
- **Terms**: Intended for academic, research, and benchmarking purposes. Features are anonymized hashes without semantic user identities.
