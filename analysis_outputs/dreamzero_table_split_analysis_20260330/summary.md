# DreamZero Appendix-Table Split Analysis

## Source Convention
- Inputs follow the user-verified appendix table values verbatim.
- DROID is analyzed with the archived table splits `L1/L2/L3`.
- AgiBot is analyzed with the appendix levels `L1/L2/L3`, whose labels carry approximate average durations `22s / 42s / 126s` in the table.
- Pooling is done only after computing split-wise statistics separately for each dataset.

## Separate Results
### DROID
- original: L1: SR=0.300, 95% CI=[0.180, 0.420], progress=0.492, L2=0.1291 | L2: SR=0.180, 95% CI=[0.080, 0.300], progress=0.420, L2=0.1255 | L3: SR=0.220, 95% CI=[0.120, 0.340], progress=0.434, L2=0.1389
- llm_dual: L1: SR=0.440, 95% CI=[0.300, 0.580], progress=0.520, L2=0.1095 | L2: SR=0.320, 95% CI=[0.200, 0.440], progress=0.480, L2=0.1089 | L3: SR=0.340, 95% CI=[0.220, 0.480], progress=0.464, L2=0.1125
- llm_val: L1: SR=0.500, 95% CI=[0.360, 0.640], progress=0.574, L2=0.1075 | L2: SR=0.380, 95% CI=[0.240, 0.520], progress=0.524, L2=0.1032 | L3: SR=0.480, 95% CI=[0.340, 0.620], progress=0.538, L2=0.1071

### AgiBot
- original: L1: SR=0.300, 95% CI=[0.180, 0.420], progress=0.460, L2=0.0845 | L2: SR=0.220, 95% CI=[0.120, 0.340], progress=0.416, L2=0.0942 | L3: SR=0.160, 95% CI=[0.060, 0.280], progress=0.406, L2=0.0748
- llm_dual: L1: SR=0.520, 95% CI=[0.380, 0.660], progress=0.574, L2=0.0685 | L2: SR=0.420, 95% CI=[0.280, 0.560], progress=0.502, L2=0.0745 | L3: SR=0.480, 95% CI=[0.340, 0.620], progress=0.536, L2=0.0614
- llm_val: L1: SR=0.620, 95% CI=[0.480, 0.760], progress=0.624, L2=0.0672 | L2: SR=0.480, 95% CI=[0.340, 0.620], progress=0.542, L2=0.0704 | L3: SR=0.580, 95% CI=[0.440, 0.720], progress=0.584, L2=0.0593

## Pooled Results
- original: L1: SR=0.300, 95% CI=[0.210, 0.390], progress=0.476, L2=0.1068 | L2: SR=0.200, 95% CI=[0.120, 0.280], progress=0.418, L2=0.1098 | L3: SR=0.190, 95% CI=[0.120, 0.270], progress=0.420, L2=0.1068
- llm_dual: L1: SR=0.480, 95% CI=[0.390, 0.580], progress=0.547, L2=0.0890 | L2: SR=0.370, 95% CI=[0.280, 0.460], progress=0.491, L2=0.0917 | L3: SR=0.410, 95% CI=[0.320, 0.510], progress=0.500, L2=0.0869
- llm_val: L1: SR=0.560, 95% CI=[0.460, 0.660], progress=0.599, L2=0.0873 | L2: SR=0.430, 95% CI=[0.330, 0.530], progress=0.533, L2=0.0868 | L3: SR=0.530, 95% CI=[0.440, 0.630], progress=0.561, L2=0.0832

## Adjacent-Split Fisher Tests
- AgiBot llm_dual L1 vs L2: p=0.423, BH p=0.7252, SR 0.520 -> 0.420
- AgiBot llm_dual L2 vs L3: p=0.6879, BH p=0.8255, SR 0.420 -> 0.480
- AgiBot llm_val L1 vs L2: p=0.2276, BH p=0.7252, SR 0.620 -> 0.480
- AgiBot llm_val L2 vs L3: p=0.423, BH p=0.7252, SR 0.480 -> 0.580
- AgiBot original L1 vs L2: p=0.4945, BH p=0.7418, SR 0.300 -> 0.220
- AgiBot original L2 vs L3: p=0.6111, BH p=0.8147, SR 0.220 -> 0.160
- DROID llm_dual L1 vs L2: p=0.303, BH p=0.7252, SR 0.440 -> 0.320
- DROID llm_dual L2 vs L3: p=1, BH p=1, SR 0.320 -> 0.340
- DROID llm_val L1 vs L2: p=0.3138, BH p=0.7252, SR 0.500 -> 0.380
- DROID llm_val L2 vs L3: p=0.4193, BH p=0.7252, SR 0.380 -> 0.480
- DROID original L1 vs L2: p=0.2414, BH p=0.7252, SR 0.300 -> 0.180
- DROID original L2 vs L3: p=0.8031, BH p=0.8761, SR 0.180 -> 0.220

- pooled llm_dual L1 vs L2: p=0.1524, BH p=0.3039, SR 0.480 -> 0.370
- pooled llm_dual L2 vs L3: p=0.6638, BH p=0.7965, SR 0.370 -> 0.410
- pooled llm_val L1 vs L2: p=0.08942, BH p=0.3039, SR 0.560 -> 0.430
- pooled llm_val L2 vs L3: p=0.2026, BH p=0.3039, SR 0.430 -> 0.530
- pooled original L1 vs L2: p=0.1412, BH p=0.3039, SR 0.300 -> 0.200
- pooled original L2 vs L3: p=1, BH p=1, SR 0.200 -> 0.190

## Ordered-Split Logistic Trend
### original
- pooled_interaction | Intercept: beta=-0.9737, OR=0.378, p=0.0008743
- pooled_interaction | split_ord: beta=-0.2248, OR=0.799, p=0.3456
- pooled_interaction | dataset_AgiBot: beta=0.1230, OR=1.131, p=0.7642
- pooled_interaction | split_ord:dataset_AgiBot: beta=-0.1816, OR=0.834, p=0.5954
- pooled_interaction | split_ord | DROID slope: beta=-0.2248, OR=0.799, p=0.3456
- pooled_interaction | split_ord | AgiBot slope: beta=-0.4064, OR=0.666, p=0.09749
- pooled_no_interaction | Intercept: beta=-0.8932, OR=0.409, p=0.0002895
- pooled_no_interaction | split_ord: beta=-0.3139, OR=0.731, p=0.06597
- pooled_no_interaction | dataset_AgiBot: beta=-0.0381, OR=0.963, p=0.8903
- DROID_only | Intercept: beta=-0.9737, OR=0.378, p=0.0008743
- DROID_only | split_ord: beta=-0.2248, OR=0.799, p=0.3456
- AgiBot_only | Intercept: beta=-0.8508, OR=0.427, p=0.003063
- AgiBot_only | split_ord: beta=-0.4064, OR=0.666, p=0.09749

### llm_dual
- pooled_interaction | Intercept: beta=-0.3345, OR=0.716, p=0.2036
- pooled_interaction | split_ord: beta=-0.2162, OR=0.806, p=0.3005
- pooled_interaction | dataset_AgiBot: beta=0.3079, OR=1.361, p=0.4037
- pooled_interaction | split_ord:dataset_AgiBot: beta=0.1359, OR=1.146, p=0.6386
- pooled_interaction | split_ord | DROID slope: beta=-0.2162, OR=0.806, p=0.3005
- pooled_interaction | split_ord | AgiBot slope: beta=-0.0803, OR=0.923, p=0.6888
- pooled_no_interaction | Intercept: beta=-0.4028, OR=0.668, p=0.06774
- pooled_no_interaction | split_ord: beta=-0.1456, OR=0.864, p=0.3135
- pooled_no_interaction | dataset_AgiBot: beta=0.4413, OR=1.555, p=0.06139
- DROID_only | Intercept: beta=-0.3345, OR=0.716, p=0.2036
- DROID_only | split_ord: beta=-0.2162, OR=0.806, p=0.3005
- AgiBot_only | Intercept: beta=-0.0266, OR=0.974, p=0.9179
- AgiBot_only | split_ord: beta=-0.0803, OR=0.923, p=0.6888

### llm_val
- pooled_interaction | Intercept: beta=-0.1469, OR=0.863, p=0.5706
- pooled_interaction | split_ord: beta=-0.0404, OR=0.960, p=0.8408
- pooled_interaction | dataset_AgiBot: beta=0.4695, OR=1.599, p=0.2018
- pooled_interaction | split_ord:dataset_AgiBot: beta=-0.0409, OR=0.960, p=0.8859
- pooled_interaction | split_ord | DROID slope: beta=-0.0404, OR=0.960, p=0.8408
- pooled_interaction | split_ord | AgiBot slope: beta=-0.0812, OR=0.922, p=0.6871
- pooled_no_interaction | Intercept: beta=-0.1266, OR=0.881, p=0.5593
- pooled_no_interaction | split_ord: beta=-0.0607, OR=0.941, p=0.6696
- pooled_no_interaction | dataset_AgiBot: beta=0.4286, OR=1.535, p=0.06508
- DROID_only | Intercept: beta=-0.1469, OR=0.863, p=0.5706
- DROID_only | split_ord: beta=-0.0404, OR=0.960, p=0.8408
- AgiBot_only | Intercept: beta=0.3226, OR=1.381, p=0.2169
- AgiBot_only | split_ord: beta=-0.0812, OR=0.922, p=0.6871

## Readout
- On the table-aligned bins, the `original` pooled SR drops from `30%` at `L1` to `20%` at `L2` and `19%` at `L3`, so the visual trend is cleaner than the earlier absolute-seconds analysis.
- AgiBot alone also shows a clean `30% -> 22% -> 16%` drop for `original`.
- DROID alone is weaker: `30% -> 18% -> 22%`, so it still does not give a perfectly monotone decline.
- The stronger modes do not show the same clean deterioration pattern under pooling, which is consistent with the story that the original method is the one that is most sensitive to longer/harder splits.
- Statistically, the split-aligned evidence is still suggestive rather than definitive: adjacent Fisher tests are not significant after correction, and the pooled `original` ordered-split trend is only marginal (`p` around the `0.05-0.10` band).

