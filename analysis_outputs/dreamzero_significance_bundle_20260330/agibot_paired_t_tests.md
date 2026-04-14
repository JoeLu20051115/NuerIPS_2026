| Split   | Comparison           | Metric           |   n_pairs |   Baseline |   Compared |   Effect (better dir) |       t |      p |   BH p | Sig   |
|:--------|:---------------------|:-----------------|----------:|-----------:|-----------:|----------------------:|--------:|-------:|-------:|:------|
| L1      | original -> llm_dual | Task Progress    |        50 |     0.506  |     0.632  |                0.126  |  7.3695 | 0      | 0      | ***   |
| L1      | original -> llm_dual | Mean L2          |        50 |     0.0848 |     0.066  |                0.0188 | -7.8711 | 0      | 0      | ***   |
| L1      | original -> llm_dual | L2<0.1 Step Rate |        50 |     0.7267 |     0.8733 |                0.1467 |  4.6099 | 0      | 0      | ***   |
| L1      | original -> llm_val  | Task Progress    |        48 |     0.5229 |     0.6229 |                0.1    |  3.6865 | 0.0006 | 0.0006 | ***   |
| L1      | original -> llm_val  | Mean L2          |        48 |     0.0855 |     0.0688 |                0.0167 | -6.8423 | 0      | 0      | ***   |
| L1      | original -> llm_val  | L2<0.1 Step Rate |        48 |     0.7222 |     0.8889 |                0.1667 |  5.0632 | 0      | 0      | ***   |
| L2      | original -> llm_dual | Task Progress    |        50 |     0.45   |     0.658  |                0.208  |  7.5555 | 0      | 0      | ***   |
| L2      | original -> llm_dual | Mean L2          |        50 |     0.0895 |     0.0685 |                0.021  | -7.7486 | 0      | 0      | ***   |
| L2      | original -> llm_dual | L2<0.1 Step Rate |        50 |     0.6333 |     0.82   |                0.1867 |  4.4777 | 0      | 0.0001 | ***   |
| L2      | original -> llm_val  | Task Progress    |        49 |     0.4592 |     0.6204 |                0.1612 |  5.77   | 0      | 0      | ***   |
| L2      | original -> llm_val  | Mean L2          |        49 |     0.0897 |     0.0679 |                0.0217 | -7.8771 | 0      | 0      | ***   |
| L2      | original -> llm_val  | L2<0.1 Step Rate |        49 |     0.6327 |     0.8299 |                0.1973 |  4.2395 | 0.0001 | 0.0001 | ***   |
| L3      | original -> llm_dual | Task Progress    |        50 |     0.422  |     0.65   |                0.228  |  9.0917 | 0      | 0      | ***   |
| L3      | original -> llm_dual | Mean L2          |        50 |     0.0715 |     0.0553 |                0.0162 | -8.3406 | 0      | 0      | ***   |
| L3      | original -> llm_dual | L2<0.1 Step Rate |        50 |     0.82   |     0.9467 |                0.1267 |  4.2292 | 0.0001 | 0.0001 | ***   |
| L3      | original -> llm_val  | Task Progress    |        50 |     0.422  |     0.574  |                0.152  |  5.2873 | 0      | 0      | ***   |
| L3      | original -> llm_val  | Mean L2          |        50 |     0.0715 |     0.0563 |                0.0153 | -6.7564 | 0      | 0      | ***   |
| L3      | original -> llm_val  | L2<0.1 Step Rate |        50 |     0.82   |     0.9667 |                0.1467 |  4.4163 | 0.0001 | 0.0001 | ***   |
