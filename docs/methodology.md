# Methodology & Experimental Protocol

This document details the scientific and engineering methodology employed in this project.

---

## 1. Problem Formulation
In real-world programmatic display advertising (RTB, supply-side platforms, and ad exchanges), every ad impression opportunity generates an auction request.
The core prediction problem is binary classification:
$$P(\text{click} = 1 \mid \mathbf{x}) \in [0, 1]$$
where $\mathbf{x} = [I_1, \dots, I_{13}, C_1, \dots, C_{26}]$.

### Distinction of Objectives:
1. **Label Classification**: Predict $\hat{y} \in \{0, 1\}$ at a fixed decision threshold.
2. **Impression Ranking**: Order competing ad creatives by predicted probability to pick the top-$k$ winning ads (measured via ROC-AUC and NDCG@K).
3. **Calibrated Probability Estimation**: Ensure that when a model outputs $\hat{p} = 0.05$, exactly 5% of those impressions result in clicks. This is vital because expected revenue in second-price auctions relies directly on estimated CTR:
$$\text{eCPM} = \text{pCTR} \times \text{Bid}_{\text{CPC}} \times 1000$$
If pCTR is uncalibrated, advertisers overpay or publishers under-monetize.
4. **Revenue Optimization**: Involves auction dynamics, pacing algorithms, and advertiser budgets, which require counterfactual and causal evaluation beyond pure supervised learning.

---

## 2. Chronological Splitting & Leakage Prevention
In online advertising, impressions are recorded in continuous temporal streams. Standard random k-fold cross-validation is **invalid** for CTR prediction because:
- **Temporal Lookahead Leakage**: Future user click behavior and dynamic publisher campaign distributions leak into the training partition.
- **Inflated Cardinality Overlap**: Novel user devices, publisher IDs, and advertiser campaigns emerge dynamically in production. Random shuffling artificially exposes these IDs in the train set.

### Experimental Splitting Protocol:
- **Dataset Size**: 100,000 real Criteo impression logs.
- **Training Set**: First 70,000 impressions (70%) — used solely to fit feature transformers, imputers, and model parameters.
- **Validation Set**: Next 15,000 impressions (15%) — used strictly for early stopping, hyperparameter tuning, and post-hoc calibration fitting.
- **Held-Out Test Set**: Final 15,000 impressions (15%) — completely untouched until final evaluation.

---

## 3. Preprocessing Architecture
Feature pipelines must handle mixed numerical counts and high-cardinality categorical hashes without causing Out-of-Memory (OOM) failures:

1. **Numerical Pipeline (I1–I13)**:
   - Median Imputation (`SimpleImputer(strategy='median')`).
   - Log1p transformation ($\log(1 + \max(0, x))$) to dampen extreme right-skewed count outliers.
   - Standard scaling (`StandardScaler`) to normalize feature variances for gradient-based models.

2. **Categorical Pipeline (C1–C26)**:
   - Missing token replacement (`__MISSING__`).
   - Sparse One-Hot Encoding (`OneHotEncoder(sparse_output=True)`).
   - Frequency filtering (`min_frequency=10`): Infrequent categories appearing fewer than 10 times are mapped to an unknown bin, compressing the dimensionality from tens of thousands down to 11,888 while preserving meaningful categorical signals.
   - Representation: Matrices are maintained in Compressed Sparse Row (`scipy.sparse.csr_matrix`) format, ensuring memory footprints remain below 30 MB on an 8 GB development machine.

---

## 4. Models Implemented
1. **Model A: Constant Prior Baseline**
   - Empirical click probability $\hat{p} = \bar{y}_{\text{train}} = 0.22463$.
   - Proves whether ML models extract signal beyond trivial marginal frequency.
2. **Model B: Regularized Logistic Regression**
   - $L_2$-penalized logistic regression optimized via L-BFGS.
   - Provides linear interpretability and fast sub-millisecond inference.
3. **Model C: Gradient Boosted Decision Trees (LightGBM)**
   - Tree ensemble with histogram-based binning and leaf-wise tree growth.
   - Early stopping enabled on validation log loss to eliminate overfitting.
   - Captures complex non-linear feature interactions between count attributes and advertiser IDs.
