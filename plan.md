# Implementation Plan: Advertising Click-Through Rate (CTR) Prediction

## 1. System & Resource Environment
- **OS**: Windows 11
- **Python**: 3.10.18
- **Hardware Profile**: 8 GB Visible RAM (~800 MB - 1.5 GB free physical RAM under concurrent apps), 17.4 GB free disk space on C:\.
- **Decision on Scale**: To guarantee stability and avoid Out-Of-Memory (OOM) crashes, we will use a genuine 100,000-row sample of the official Criteo Display Advertising Challenge (`criteo_sample.csv.gz`, ~8.75 MB compressed, ~30 MB uncompressed in memory).
- **Installed Key Libraries**: `numpy 2.0.2`, `scipy 1.15.3`, `pandas 2.2.3`, `scikit-learn 1.5.2`, `lightgbm 4.7.0`, `xgboost 3.2.0`, `matplotlib 3.10.8`, `seaborn 0.13.2`, `pytest 9.0.3`, `pyyaml 6.0.3`.

## 2. Authentic Dataset Provenance & Schema
- **Source**: Criteo Display Advertising Challenge (Kaggle DAC benchmark subset preserved via River / Max Halford benchmark).
- **Target**: `click` (binary integer {0, 1}).
- **Features**: 13 numerical count/integer features (`I1`-`I13`), 26 categorical hash-string features (`C1`-`C26`).
- **Data handling**: Tabular CSV with missing values represented as empty/NaN. Anonymized categorical hash values.
- **Split Strategy**: 70% Train, 15% Validation, 15% Held-out Test (chronological preservation: order-preserving row indices without random shuffle, reflecting impression streaming logs).

## 3. Architecture & Modular Structure
```
Advertising Click-Through Rate (CTR) Prediction/
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
├── pyproject.toml
├── configs/
│   └── default.yaml
├── data/
│   ├── README.md
│   └── raw/
├── notebooks/
│   └── 01_exploratory_data_analysis.ipynb
├── src/
│   └── ctr_prediction/
│       ├── __init__.py
│       ├── config.py
│       ├── data.py
│       ├── validation.py
│       ├── preprocessing.py
│       ├── features.py
│       ├── models.py
│       ├── train.py
│       ├── evaluate.py
│       ├── calibration.py
│       └── predict.py
├── tests/
│   ├── conftest.py
│   ├── test_data.py
│   ├── test_preprocessing.py
│   ├── test_metrics.py
│   └── test_prediction.py
├── scripts/
│   ├── run_eda.py
│   ├── run_experiments.py
│   └── generate_report.py
├── reports/
│   ├── figures/
│   ├── results/
│   └── project_report.md
├── models/
│   └── .gitkeep
└── docs/
    ├── methodology.md
    ├── experiments.md
    ├── limitations.md
    ├── mathematical_background.md
    ├── interview_questions.md
    └── model_card.md
```

## 4. Phase-by-Phase Roadmap

### Phase 1: Environment & Project Foundation (Complete)
- Inspect specs and test package dependencies.
- Verify online data provenance and verify 100k row Criteo sample schema.

### Phase 2: Core Engineering, Data Ingestion, EDA & Preprocessing
- Implement `configs/default.yaml` and `src/ctr_prediction/config.py`.
- Implement `data.py`: automated downloader, verification, chunked/safe loader.
- Implement `validation.py`: schema validator, missingness profile, duplicate detection.
- Implement `preprocessing.py`:
  - Numerical: Median imputation + optional log1p transform + StandardScaler.
  - Categorical: Missing token filling + infrequent category thresholding (`min_frequency=10`) + Sparse One-Hot Encoding / Hashing Encoder to prevent dense memory explosions.
  - Leakage protection: Preprocessing pipeline fit strictly on Train split only.
- Implement `scripts/run_eda.py`: generate publication-grade figures (class distribution, missingness heatmap, numerical skew, categorical cardinality distributions).

### Phase 3: Model Suite & Experimental Implementations
- **Model A: Naive/Empirical Baseline**:
  - Constant prior probability prediction ($\hat{p} = \bar{y}_{train}$).
- **Model B: Regularized Logistic Regression**:
  - `SGDClassifier(loss='log_loss')` or `LogisticRegression(solver='saga'/'lbfgs')` with L2 penalty, sparse matrix inputs.
- **Model C: Gradient Boosted Decision Trees**:
  - LightGBM (`LGBMClassifier`) with early stopping on validation log-loss, feature importance (split and gain).
  - Alternative XGBoost configuration for comparison.
- **Experimentation Engine (`scripts/run_experiments.py`)**:
  - Experiment 1: Baseline vs. Logistic Regression vs. LightGBM.
  - Experiment 2: Categorical Preprocessing Strategy Ablation (One-Hot Sparse vs. Categorical Ordinal/LightGBM native categorical).
  - Experiment 3: Numerical Transformation Ablation (Standard scaling vs. Log1p + Standard scaling).
  - Experiment 4: Probability Calibration (Uncalibrated vs. Platt Scaling / Sigmoid Calibration vs. Isotonic Regression).

### Phase 4: Rigorous Scientific Evaluation & Diagnostics
- Metrics: Log Loss, ROC-AUC, PR-AUC (Average Precision), Brier Score, Accuracy, Precision, Recall, F1 at threshold, Confusion Matrix.
- Calibration analysis: Reliability diagram with 10 bins, Expected Calibration Error (ECE).
- Ranking evaluation: Mean NDCG@K across simulated impression candidate lists.
- Model serialization: Save trained pipelines + estimators via `joblib`.
- Inference pipeline (`predict.py`): Test on new unseen samples, ensuring schema conformity.

### Phase 5: Automated Testing & Verification
- Unit test suite in `tests/`:
  - `test_data.py`: Synthetic test fixture, column schema validation, splitting logic.
  - `test_preprocessing.py`: Unseen categories handling, missing value imputation, sparse matrix output.
  - `test_metrics.py`: Correct computation of Log Loss, ROC-AUC, Brier score, ECE.
  - `test_prediction.py`: End-to-end test of model save, load, and batch prediction.
- Run `pytest` and ensure 100% test pass.

### Phase 6: Documentation & Scientific Report
- Generate comprehensive documentation:
  - `README.md` with complete architecture, instructions, benchmark results.
  - `docs/mathematical_background.md`: Derivations of logistic sigmoid, binary log-loss, tree split gain, gradient boosting gradients/hessians, AUC formulas, Platt calibration.
  - `docs/methodology.md`: Chronological splitting, feature pipelines, leakage prevention.
  - `docs/limitations.md`: In-depth analysis of Criteo negative downsampling, prior correction formula $p = \frac{q}{q + (1-q)/w}$, ranking vs probability calibration, lack of temporal timestamps.
  - `docs/model_card.md`: Production model card.
  - `docs/interview_questions.md`: 25+ Applied Scientist interview questions with exhaustive answers and follow-ups.
  - `reports/project_report.md`: Formal experimental writeup with saved figures and benchmark tables.
