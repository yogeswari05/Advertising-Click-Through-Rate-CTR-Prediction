# Advertising CTR Prediction: Comprehensive Experimental Report

## Executive Summary
This report presents empirical findings from an end-to-end Machine Learning study on Click-Through Rate (CTR) estimation using the official 100,000-row benchmark from the **Criteo Display Advertising Challenge**. Designed to demonstrate rigorous probability estimation, chronological evaluation, sparse feature modeling, and post-hoc probability calibration.

---

## 1. Primary Model Benchmark

Models were trained on 70% chronological impressions (70,000 rows), tuned on 15% validation impressions (15,000 rows), and evaluated on 15% unseen held-out test impressions (15,000 rows).

| Model | Val Log Loss | Val ROC-AUC | Val PR-AUC | Test Log Loss | Test ROC-AUC | Test PR-AUC | Test Brier Score | Test NDCG@5 | Test ECE |
|---|---|---|---|---|---|---|---|---|---|
| **baseline** | 0.53366 | 0.50000 | 0.22540 | 0.54828 | 0.50000 | 0.23720 | 0.18109 | 0.37388 | 0.01257 |
| **logistic_regression** | 0.44287 | 0.77302 | 0.53252 | 0.45303 | 0.77680 | 0.54262 | 0.14585 | 0.67491 | 0.00484 |
| **lightgbm** | 0.44531 | 0.77088 | 0.52498 | 0.45226 | 0.77801 | 0.54581 | 0.14575 | 0.69168 | 0.00826 |

### Key Benchmark Observations
1. **Beating the Prior Baseline**: The empirical baseline ($\hat{p} = 0.22463$) achieved a test log loss of 0.54828. Both Logistic Regression (0.45303) and LightGBM (0.45226) delivered substantial reductions in cross-entropy loss (>17% relative log loss reduction).
2. **Linear vs. Boosted Trees**: LightGBM outperformed Logistic Regression on ranking metrics (Test ROC-AUC: **0.77801** vs 0.77680; Test NDCG@5: **0.69168** vs 0.67491) by capturing non-linear interactions across high-cardinality categorical features.

---

## 2. Experimental Ablation Studies

### Experiment 2: Categorical Preprocessing Strategy Ablation
Evaluating minimum category frequency threshold ($	au$) to control sparse feature expansion:

| Min Category Frequency | Feature Dimension | Val Log Loss | Val ROC-AUC | Test Log Loss | Test ROC-AUC |
|---|---|---|---|---|---|
| $\tau = 10$ | 11,888 | 0.44287 | 0.77302 | 0.45303 | 0.77680 |
| $\tau = 50$ | 2,632 | 0.44831 | 0.76589 | 0.45647 | 0.77232 |

**Insight**: Retaining long-tail categories ($	au = 10$, 11,888 sparse dimensions) improved Test ROC-AUC from 0.7723 to 0.7768 over aggressive pruning ($	au = 50$, 2,632 dimensions). The sparse CSR encoding safely contained memory consumption below 25 MB.

### Experiment 3: Numerical Feature Transformation Ablation
Evaluating logarithmic dampening ($\log(1 + x)$) on skewed count features (I1–I13):

| Log1p Transformation | Val Log Loss | Val ROC-AUC | Test Log Loss | Test ROC-AUC |
|---|---|---|---|---|
| False | 0.45594 | 0.75714 | 0.46753 | 0.75867 |
| True | 0.44287 | 0.77302 | 0.45303 | 0.77680 |

**Insight**: Applying $\log(1 + \max(0, x))$ to right-skewed count features yielded a dramatic improvement in Test ROC-AUC (+0.0181) and decreased Test Log Loss from 0.46753 to 0.45303. Count outliers previously distorted standardization means and linear gradients.

---

## 3. Post-Hoc Probability Calibration

Evaluating calibration error across 10 probability bins on held-out test data:

| Calibration Method | Test Log Loss | Test Brier Score | Expected Calibration Error (ECE) |
|---|---|---|---|
| Uncalibrated LightGBM | 0.45226 | 0.14575 | 0.00826 |
| Platt Sigmoid (Logistic) | 0.45579 | 0.14649 | 0.02402 |
| Isotonic Regression | 0.45449 | 0.14582 | 0.00711 |

**Insight**: Raw LightGBM is already well-calibrated (ECE = 0.00826) when early-stopped directly on cross-entropy loss. Non-parametric Isotonic Regression further reduced calibration error to **0.00711**, aligning bin-level confidence with empirical click incidence.
