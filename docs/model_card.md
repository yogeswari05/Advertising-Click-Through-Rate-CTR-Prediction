# Model Card: LightGBM CTR Predictor (Production Artifact)

## Model Details
- **Model Name**: LightGBM Gradient Boosted Decision Tree CTR Predictor
- **Version**: 1.0.0
- **Type**: Binary Classification / Probability Estimation Ensemble
- **Library**: `lightgbm 4.7.0`, `scikit-learn 1.5.2`
- **Trained By**: Yedu Krishnan
- **License**: Apache 2.0 / Academic & Research Benchmark Use

---

## Intended Use
- **Primary Intended Use**: Predicting the probability of a user clicking an advertisement impression ($P(\text{click} = 1 \mid \mathbf{x})$) to inform candidate ranking and expected CPM ranking in display ad auctions.
- **Intended Users**: Applied Scientists and ML Engineers evaluating display advertising, recommendation feeds, and RTB bidding systems.
- **Out-of-Scope Use Cases**:
  - High-stakes credit or lending scoring.
  - User profiling based on inferred sensitive demographic attributes.
  - Direct real-world bidding without calibrated negative sampling correction.

---

## Factors & Data
- **Dataset**: Criteo Display Advertising Challenge (100,000 real impression logs).
- **Features**:
  - 13 continuous numerical counts (I1–I13) transformed via median imputation, log1p dampening, and standard scaling.
  - 26 anonymized categorical attributes (C1–C26) encoded using sparse one-hot encoding with rare-category thresholding ($\tau = 10$).
- **Partitioning**: Chronological split (70% Train, 15% Validation, 15% Test).

---

## Performance & Metrics (Held-Out Test Set)
Evaluated on 15,000 chronological held-out impressions:

| Metric | Score | Interpretation |
|---|---|---|
| **Log Loss** | **0.45226** | Strong probabilistic calibration (vs 0.54828 baseline) |
| **ROC-AUC** | **0.77801** | High pairwise ranking discrimination |
| **PR-AUC** | **0.54581** | Robust precision-recall performance under imbalance |
| **Brier Score**| **0.14575** | Low quadratic mean squared error |
| **NDCG@5** | **0.69168** | Effective top-candidate selection in simulated slates |
| **ECE** | **0.00826** | Expected calibration error < 1% across 10 probability bins |

---

## Caveats & Limitations
1. **Negative Downsampling**: The benchmark dataset contains an artificial click prevalence of ~22.66% due to negative downsampling. Production deployment requires applying the prior correction $p = \frac{q}{q + (1-q)/w}$ using the true platform sampling rate $w$.
2. **Anonymized Semantics**: Features are anonymized hashes. No semantic claims regarding user gender, age, or demographics should be made.
3. **Concept Drift**: Performance degrades over time as campaign budgets rotate and user interests evolve. Continuous online model updates or daily retraining schedules are required.
