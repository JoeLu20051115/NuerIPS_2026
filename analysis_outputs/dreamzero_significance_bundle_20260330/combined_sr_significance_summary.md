| Dataset   | Split   | Comparison           |   n_pairs |   Original SR (%) |   Compared SR (%) |   Delta (pp) |   Gain Pairs |   Loss Pairs |   BH p | Sig   |
|:----------|:--------|:---------------------|----------:|------------------:|------------------:|-------------:|-------------:|-------------:|-------:|:------|
| DROID     | L1      | original -> llm_dual |        50 |              30   |              44   |         14   |            9 |            2 | 0.0826 |       |
| DROID     | L1      | original -> llm_val  |        50 |              30   |              50   |         20   |           10 |            0 | 0.0031 | **    |
| DROID     | L2      | original -> llm_dual |        50 |              18   |              32   |         14   |            9 |            2 | 0.0826 |       |
| DROID     | L2      | original -> llm_val  |        50 |              18   |              38   |         20   |           10 |            0 | 0.0031 | **    |
| DROID     | L3      | original -> llm_dual |        50 |              20   |              34   |         14   |           11 |            4 | 0.1354 |       |
| DROID     | L3      | original -> llm_val  |        50 |              20   |              48   |         28   |           14 |            0 | 0.0005 | ***   |
| AgiBot    | L1      | original -> llm_dual |        50 |              40   |              60   |         20   |           10 |            0 | 0.0021 | **    |
| AgiBot    | L1      | original -> llm_val  |        48 |              41.7 |              64.6 |         22.9 |           12 |            1 | 0.0034 | **    |
| AgiBot    | L2      | original -> llm_dual |        50 |              44   |              70   |         26   |           13 |            0 | 0.0004 | ***   |
| AgiBot    | L2      | original -> llm_val  |        49 |              44.9 |              67.4 |         22.5 |           12 |            1 | 0.0034 | **    |
| AgiBot    | L3      | original -> llm_dual |        50 |              24   |              76   |         52   |           26 |            0 | 0      | ***   |
| AgiBot    | L3      | original -> llm_val  |        50 |              24   |              58   |         34   |           19 |            2 | 0.0004 | ***   |
