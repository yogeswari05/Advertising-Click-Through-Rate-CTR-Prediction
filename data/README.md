# Criteo Display Advertising Challenge Dataset: Provenance & Notes

## 1. Dataset Overview
- **Name**: Criteo Display Advertising Challenge (DAC) Dataset Sample
- **Format**: Gzipped CSV (`criteo_sample.csv.gz`)
- **Row Count**: 100,000 real impression records
- **Feature Schema**:
  - `click`: Target binary integer label (0 = impression not clicked, 1 = impression clicked).
  - `I1`–`I13`: 13 integer count features. Anonymized integer counts representing various frequency / historical engagement counters.
  - `C1`–`C26`: 26 categorical features. Hashed to 32-bit hexadecimal strings to preserve confidentiality of advertisers and publishers.
- **Empirical Click Rate**: ~22.66% ($N = 22,663$ clicks out of $100,000$ impressions).

## 2. Acquisition Method & Verification
The dataset is downloaded directly via `download_criteo_sample()` from the official research sample preserved by the River benchmark repository:
`https://maxhalford.github.io/files/datasets/criteo_sample.csv.gz`

## 3. Negative Subsampling Disclaimer
As documented by Criteo AI Lab, non-clicked impressions were downsampled to reduce data volume. Because the exact downsampling rate is not disclosed in the benchmark release, raw model probabilities reflect $P(\text{click} \mid \text{sampled})$. See `docs/limitations.md` for the prior correction formulation.
