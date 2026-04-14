# DreamZero Duration-Effect Analysis

## Sources
- DROID: `evaluation_results_dualsystem/droid_3way_compare.json` joined with `data/final_data1/DRO_L1_150`, `DRO_L2_100`, `DRO_L3_150` metadata.
- AgiBot: `evaluation_results_dualsystem/agibot_3way_compare.json` joined with `data/final_data1/Agi_L1_150` and `Agi_L3_150` manifests.

## Core Read
- DROID old DreamZero episodes span only 3.1s to 37.7s; the requested absolute 0-30/30-60/60-120/120+ bucket analysis therefore leaves only 3 episodes outside the first bucket.
- AgiBot spans 14.3s to 264.2s, so the requested absolute buckets are meaningful there.
- The analysis below uses the `original/task_token_only` mode for the main claim, because the target hypothesis is that the original single-prompt method degrades with longer tasks.

## Descriptive Result
### DROID
- 0-30s: n=397, SR=0.330, 95% CI=[0.285, 0.375], mean progress=0.487, mean L2=0.1291
- 30-60s: n=3, SR=0.000, 95% CI=[0.000, 0.000], mean progress=0.267, mean L2=0.1567

### AgiBot
- 0-30s: n=133, SR=0.398, 95% CI=[0.316, 0.481], mean progress=0.513, mean L2=0.0858
- 30-60s: n=56, SR=0.429, 95% CI=[0.304, 0.554], mean progress=0.507, mean L2=0.0818
- 60-120s: n=50, SR=0.320, 95% CI=[0.200, 0.460], mean progress=0.448, mean L2=0.0764
- 120s+: n=61, SR=0.262, 95% CI=[0.164, 0.377], mean progress=0.459, mean L2=0.0671

## Pairwise Bucket Tests
- Fisher exact tests compare adjacent duration buckets on success rate only.
- Benjamini-Hochberg is applied across all available adjacent-bucket tests.

- AgiBot 0-30s vs 30-60s: p=0.747, BH-adjusted p=0.747, SR 0.398 -> 0.429
- AgiBot 30-60s vs 60-120s: p=0.3164, BH-adjusted p=0.7386, SR 0.429 -> 0.320
- AgiBot 60-120s vs 120s+: p=0.5336, BH-adjusted p=0.7386, SR 0.320 -> 0.262
- DROID 0-30s vs 30-60s: p=0.554, BH-adjusted p=0.7386, SR 0.330 -> 0.000

## Pooled Across Datasets
- This combines DROID and AgiBot into one unseen-test-set pool.
- Important caveat: the pooled 0-30s bucket is DROID-heavy, while the long-duration buckets are effectively AgiBot-only.

- 0-30s: n=530, SR=0.347, 95% CI=[0.306, 0.389], mean progress=0.493, mean L2=0.1183
- 30-60s: n=59, SR=0.407, 95% CI=[0.288, 0.525], mean progress=0.495, mean L2=0.0856
- 60-120s: n=50, SR=0.320, 95% CI=[0.200, 0.460], mean progress=0.448, mean L2=0.0764
- 120s+: n=61, SR=0.262, 95% CI=[0.164, 0.377], mean progress=0.459, mean L2=0.0671

- pooled 0-30s vs 30-60s: p=0.3901, BH-adjusted p=0.5336, SR 0.347 -> 0.407
- pooled 30-60s vs 60-120s: p=0.4262, BH-adjusted p=0.5336, SR 0.407 -> 0.320
- pooled 60-120s vs 120s+: p=0.5336, BH-adjusted p=0.5336, SR 0.320 -> 0.262

## Regression Read
- Combined model: `logit(success) ~ duration + level + dataset + duration:dataset` on original mode.
- A pooled no-interaction model is also reported because a single mixed statistic can be useful as a compact appendix summary.
- Additional per-dataset models are reported for interpretability.

### Combined Original-Mode Model
- Intercept: beta=0.0792, OR=1.082, p=0.7904
- duration_per_30s: beta=-0.9716, OR=0.378, p=0.2051
- level_L2: beta=-1.1942, OR=0.303, p=0.000191
- level_L3: beta=-0.3743, OR=0.688, p=0.1025
- dataset_AgiBot: beta=-0.3707, OR=0.690, p=0.2906
- duration_per_30s:dataset_AgiBot: beta=0.9300, OR=2.534, p=0.2143
- duration_per_30s | DROID slope: beta=-0.9716, OR=0.378, p=0.2051
- duration_per_30s | AgiBot net slope: beta=-0.0416, OR=0.959, p=0.6054

### Pooled No-Interaction Model
- Intercept: beta=-0.2303, OR=0.794, p=0.1645
- duration_per_30s: beta=-0.0243, OR=0.976, p=0.7602
- level_L2: beta=-1.1217, OR=0.326, p=0.0003689
- level_L3: beta=-0.5039, OR=0.604, p=0.01447
- dataset_AgiBot: beta=-0.0370, OR=0.964, p=0.8714

### DROID-Only Model
- Intercept: beta=0.0843, OR=1.088, p=0.7782
- duration_per_10s: beta=-0.3875, OR=0.679, p=0.1691
- level_L2: beta=-1.1519, OR=0.316, p=0.0004699
- level_L3: beta=-0.2586, OR=0.772, p=0.4041

### AgiBot-Only Model
- Intercept: beta=-0.2828, OR=0.754, p=0.1262
- duration_per_30s: beta=-0.0143, OR=0.986, p=0.8796
- level_L3: beta=-0.5150, OR=0.598, p=0.1337

## Verdict
- AgiBot shows a descriptive drop in original success once duration exceeds 60s, but the effect is not monotonic at the short end and the adjacent-bucket tests are not significant after correction.
- After controlling for level, the original-mode AgiBot duration coefficient is not significantly negative.
- DROID does not provide meaningful long-horizon coverage in the archived old DreamZero run, so it cannot strongly support a 'longer tasks cause collapse' claim on an absolute-seconds axis.
- The pooled curve is also not convincingly monotonic: 0-30s -> 30-60s rises before later buckets fall, and the pooled no-interaction duration coefficient is near zero and not significant.
- Overall, the current archived old DreamZero data provide at most a suggestive AgiBot trend, not a strong standalone causal argument.

