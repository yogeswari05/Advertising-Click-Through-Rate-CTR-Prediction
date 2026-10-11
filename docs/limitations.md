# CTR Sampling & Probability Limitations (Critical Research Analysis)

This document investigates the documented negative downsampling protocol in advertising challenge datasets (specifically Criteo DAC) and evaluates its implications for probability calibration, bidding, and production deployment.

---

## 1. Why Sampled Click Prevalence Differs from True Production CTR

In real-world display ad networks and supply-side platforms (SSPs / DSPs), the vast majority of ad impressions do not receive clicks. True real-world production CTRs typically range between:
$$\text{CTR}_{\text{production}} \approx 0.05\% - 0.5\% \quad (0.0005 - 0.005)$$

However, in the Criteo DAC dataset, the observed empirical click rate is:
$$\text{CTR}_{\text{dataset}} \approx 22.66\% \quad (0.2266)$$

### The Negative Downsampling Mechanism
To keep training data sizes manageable and reduce storage/bandwidth costs, ad platforms downsample non-clicked impressions (negative instances) while retaining all clicked impressions (positive instances):
- Let $w \in (0, 1]$ be the negative sampling rate (e.g., $w = 0.01$, keeping 1 out of every 100 non-clicked impressions).
- Let positive impressions be retained with probability 1.0 ($w_+ = 1$).

This artificial enrichment shifts the sampled prior odds:
$$\text{Odds}_{\text{sampled}} = \frac{P(y=1 \mid \text{sampled})}{P(y=0 \mid \text{sampled})} = \frac{P(y=1)}{w \cdot P(y=0)} = \frac{1}{w} \text{Odds}_{\text{true}}$$

---

## 2. Impact on Model Predictions: Ranking vs. Calibration

### Impact on Impression Ranking (ROC-AUC & NDCG@K)
Under monotonic transformations of probability, the ordering of impressions is **preserved**:
- Since the log-odds shift $\log(1/w)$ is an additive constant across all samples, the predicted ranking of ads within an auction is theoretically invariant to uniform negative downsampling.
- Hence, models evaluated on downsampled data can achieve strong ROC-AUC and NDCG scores that translate well to ad candidate selection.

### Impact on Probability Calibration
Directly using the model's raw probability output $\hat{q} = P(y=1 \mid \mathbf{x}, \text{sampled})$ in live ad auctions would be disastrous:
- In cost-per-click (CPC) bidding, expected CPM is computed as:
  $$\text{eCPM} = \hat{p} \times \text{Bid}_{\text{CPC}} \times 1000$$
- If the model predicts $\hat{q} \approx 0.22$ instead of true $\hat{p} \approx 0.002$, the ad exchange would overestimate the click probability by a factor of ~100x.
- Advertisers would blow through budgets instantaneously or bidding pacing algorithms would fail completely.

---

## 3. Mathematical Prior-Probability Correction

If the negative downsampling rate $w$ is known or estimated, we can apply the Bayes rule correction (McMahan et al., 2013; He et al., 2014):

Let $q = P(y=1 \mid \mathbf{x}, \text{sampled})$ be the model's predicted probability on the sampled dataset.
The calibrated production probability $p = P(y=1 \mid \mathbf{x})$ is:
$$p = \frac{q}{q + \frac{1 - q}{w}}$$

### Proof Derivation:
1. True odds:
   $$\frac{p}{1 - p} = w \cdot \frac{q}{1 - q}$$
2. Solving for $p$:
   $$p = w \cdot \frac{q}{1 - q} (1 - p)$$
   $$p \left( 1 + \frac{w \cdot q}{1 - q} \right) = \frac{w \cdot q}{1 - q}$$
   $$p \left( \frac{1 - q + w q}{1 - q} \right) = \frac{w q}{1 - q}$$
   $$p = \frac{w q}{1 - q + w q} = \frac{q}{q + \frac{1 - q}{w}}$$ $\quad \blacksquare$

---

## 4. Limitation: Unknown Sampling Parameters in Criteo DAC

While Criteo AI Lab explicitly documents that non-clicked impressions were subsampled to balance dataset volume, the **exact numerical sampling fraction $w$ is not disclosed** in the benchmark release metadata.

### Scientific Stance:
- We **do not** invent an arbitrary negative sampling rate $w$ or fabricate a correction without empirical justification.
- We report all probabilities, log loss, and ECE relative to the benchmark distribution.
- In a production ad serving architecture, logging pipelines explicitly attach the sampling weight $w$ to each impression payload, allowing the online serving layer to apply the exact prior correction before passing pCTR to the ad auction mechanism.
