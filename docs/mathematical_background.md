# Mathematical Background for Click-Through Rate (CTR) Prediction

This document provides rigorous mathematical formulations, loss derivations, gradient boosting mechanics, evaluation properties, and probability calibration theory tailored for Applied Scientist roles in digital advertising and machine learning systems.

---

## 1. Problem Formulation and Logistic Regression

Let impression instance $i$ be characterized by feature vector $\mathbf{x}_i \in \mathbb{R}^d$ and binary click label $y_i \in \{0, 1\}$.
In CTR estimation, we model the conditional probability:
$$p_i = P(y_i = 1 \mid \mathbf{x}_i) = \sigma(z_i)$$
where $z_i$ is the linear score:
$$z_i = \mathbf{w}^T \mathbf{x}_i + b = \sum_{j=1}^d w_j x_{ij} + b$$
and $\sigma(z)$ denotes the standard logistic sigmoid function:
$$\sigma(z) = \frac{1}{1 + e^{-z}} = \frac{e^z}{1 + e^z}$$

### Key Properties of the Sigmoid Function
1. **Range**: $\sigma(z) \in (0, 1)$ for all $z \in \mathbb{R}$.
2. **Derivative**:
$$\frac{d\sigma(z)}{dz} = \sigma(z)(1 - \sigma(z))$$
*Proof*:
$$\frac{d}{dz}(1 + e^{-z})^{-1} = - (1 + e^{-z})^{-2}(-e^{-z}) = \frac{1}{1 + e^{-z}} \frac{e^{-z}}{1 + e^{-z}} = \sigma(z)(1 - \sigma(z))$$
3. **Log-Odds (Logit)**:
$$\log \left( \frac{p_i}{1 - p_i} \right) = \log \left( \frac{\sigma(z_i)}{1 - \sigma(z_i)} \right) = z_i = \mathbf{w}^T \mathbf{x}_i + b$$
Thus, weights $w_j$ represent the marginal change in log-odds of a click per unit increase in feature $x_j$, holding all other features fixed.

---

## 2. Binary Cross-Entropy (Log Loss) and Gradients

Assuming impression outcomes $y_i \sim \text{Bernoulli}(p_i)$ are conditionally independent, the likelihood of parameters $\mathbf{w}$ across dataset $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^N$ is:
$$\mathcal{L}(\mathbf{w}) = \prod_{i=1}^N p_i^{y_i} (1 - p_i)^{1 - y_i}$$

Maximizing likelihood is equivalent to minimizing the negative log-likelihood (empirical Binary Cross-Entropy loss):
$$\mathcal{J}(\mathbf{w}) = - \frac{1}{N} \sum_{i=1}^N \left[ y_i \log p_i + (1 - y_i) \log(1 - p_i) \right]$$

### Gradient Derivation
Using the chain rule:
$$\frac{\partial \mathcal{J}}{\partial w_j} = \frac{1}{N} \sum_{i=1}^N \frac{\partial \ell_i}{\partial p_i} \frac{\partial p_i}{\partial z_i} \frac{\partial z_i}{\partial w_j}$$
where $\ell_i = - [y_i \log p_i + (1 - y_i)\log(1 - p_i)]$.
1. $\frac{\partial \ell_i}{\partial p_i} = - \frac{y_i}{p_i} + \frac{1 - y_i}{1 - p_i} = \frac{p_i - y_i}{p_i (1 - p_i)}$
2. $\frac{\partial p_i}{\partial z_i} = p_i (1 - p_i)$
3. $\frac{\partial z_i}{\partial w_j} = x_{ij}$

Multiplying terms:
$$\frac{\partial \mathcal{J}}{\partial w_j} = \frac{1}{N} \sum_{i=1}^N \left( \frac{p_i - y_i}{p_i (1 - p_i)} \right) [p_i (1 - p_i)] x_{ij} = \frac{1}{N} \sum_{i=1}^N (p_i - y_i) x_{ij}$$
In vector form:
$$\nabla_{\mathbf{w}} \mathcal{J} = \frac{1}{N} \mathbf{X}^T (\mathbf{p} - \mathbf{y})$$
The gradient vector is directly proportional to the residual error vector $(\mathbf{p} - \mathbf{y})$ projected onto the feature space.

### Regularization ($L_2$ Ridge Penalty)
To prevent overfitting on high-cardinality sparse categories:
$$\mathcal{J}_{\text{reg}}(\mathbf{w}) = \mathcal{J}(\mathbf{w}) + \frac{\lambda}{2} \|\mathbf{w}\|_2^2$$
$$\nabla_{\mathbf{w}} \mathcal{J}_{\text{reg}} = \frac{1}{N} \mathbf{X}^T (\mathbf{p} - \mathbf{y}) + \lambda \mathbf{w}$$
This shrinks noisy sparse category coefficients toward zero.

---

## 3. Decision Trees & Gradient Boosted Trees (GBDT)

### Decision Tree Split Criteria
For classification, a decision tree node partition $S$ into $S_L$ and $S_R$ is evaluated by impurity reduction:
$$\Delta I = I(S) - \left( \frac{|S_L|}{|S|} I(S_L) + \frac{|S_R|}{|S|} I(S_R) \right)$$
Common impurity measures:
- **Entropy**: $H(S) = - \sum_{k \in \{0, 1\}} p_k \log_2 p_k$
- **Gini Impurity**: $G(S) = 1 - \sum_{k \in \{0, 1\}} p_k^2 = 2 p_1 (1 - p_1)$

### Gradient Boosted Decision Trees (Friedman / XGBoost / LightGBM)
GBDT builds an ensemble of $M$ additive regression trees:
$$\hat{y}_i^{(M)} = \sum_{m=1}^M f_m(\mathbf{x}_i)$$
At iteration $m$, objective function:
$$\mathcal{L}^{(m)} = \sum_{i=1}^N \ell\left(y_i, \hat{y}_i^{(m-1)} + f_m(\mathbf{x}_i)\right) + \Omega(f_m)$$

Using a 2nd-order Taylor expansion around $\hat{y}_i^{(m-1)}$:
$$\mathcal{L}^{(m)} \approx \sum_{i=1}^N \left[ \ell(y_i, \hat{y}_i^{(m-1)}) + g_i f_m(\mathbf{x}_i) + \frac{1}{2} h_i f_m^2(\mathbf{x}_i) \right] + \Omega(f_m)$$
where:
$$g_i = \left. \frac{\partial \ell(y_i, \hat{y})}{\partial \hat{y}} \right|_{\hat{y} = \hat{y}_i^{(m-1)}} = p_i - y_i \quad (\text{first-order gradient})$$
$$h_i = \left. \frac{\partial^2 \ell(y_i, \hat{y})}{\partial \hat{y}^2} \right|_{\hat{y} = \hat{y}_i^{(m-1)}} = p_i (1 - p_i) \quad (\text{second-order Hessian})$$

For leaf $j$ with sample index set $I_j$, the optimal leaf weight is:
$$w_j^* = - \frac{\sum_{i \in I_j} g_i}{\sum_{i \in I_j} h_i + \lambda}$$
and the optimal split gain is:
$$\text{Gain} = \frac{1}{2} \left[ \frac{\left(\sum_{i \in I_L} g_i\right)^2}{\sum_{i \in I_L} h_i + \lambda} + \frac{\left(\sum_{i \in I_R} g_i\right)^2}{\sum_{i \in I_R} h_i + \lambda} - \frac{\left(\sum_{i \in I} g_i\right)^2}{\sum_{i \in I} h_i + \lambda} \right] - \gamma$$

---

## 4. Evaluation Metric Mathematics

### 1. ROC-AUC (Area Under the Receiver Operating Characteristic)
ROC-AUC measures the probability that a randomly chosen positive impression ranks higher than a randomly chosen negative impression:
$$\text{ROC-AUC} = P\left(\hat{p}_i > \hat{p}_j \mid y_i = 1, y_j = 0\right) = \frac{1}{N_+ N_-} \sum_{i: y_i=1} \sum_{j: y_j=0} \mathbb{I}(\hat{p}_i > \hat{p}_j)$$
ROC-AUC is insensitive to class prevalence shifts because True Positive Rate ($\frac{\text{TP}}{\text{P}}$) and False Positive Rate ($\frac{\text{FP}}{\text{N}}$) normalize within each ground truth class independently.

### 2. PR-AUC (Precision-Recall Area / Average Precision)
$$\text{PR-AUC} = \sum_{k} (R_k - R_{k-1}) P_k$$
Unlike ROC-AUC, precision $P = \frac{\text{TP}}{\text{TP} + \text{FP}}$ depends directly on negative class size $\text{FP}$. Under severe CTR class imbalance (e.g., 0.1% CTR), PR-AUC heavily penalizes false positives that would barely move the False Positive Rate.

### 3. Brier Score
The mean squared probabilistic error:
$$\text{BS} = \frac{1}{N} \sum_{i=1}^N (p_i - y_i)^2$$
Brier score is a strictly proper scoring rule decomposing into reliability (calibration error), resolution, and uncertainty.

---

## 5. Post-Hoc Probability Calibration

A model is calibrated if:
$$P(y = 1 \mid \hat{p} = p) = p, \quad \forall p \in [0, 1]$$

### Expected Calibration Error (ECE)
Partition predictions into $M$ bins $B_1, \dots, B_M$.
$$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$
where:
$$\text{conf}(B_m) = \frac{1}{|B_m|} \sum_{i \in B_m} \hat{p}_i, \quad \text{acc}(B_m) = \frac{1}{|B_m|} \sum_{i \in B_m} y_i$$

### Platt Scaling
Fits a univariate logistic model over raw model scores or logits $z_i$:
$$P(y_i = 1 \mid z_i) = \frac{1}{1 + e^{-(A z_i + B)}}$$
Optimized via MLE strictly on the validation set to prevent label leakage.
