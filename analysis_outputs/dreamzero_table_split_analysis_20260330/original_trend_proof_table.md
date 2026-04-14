# Original-Mode Trend Table

## Main Trend Table

| Dataset | L1 SR | L2 SR | L3 SR | Delta (L1->L3) | Monotone Decline | OR / split | p-value | Readout |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- |
| DROID | 30% | 18% | 22% | -8 pp | No | 0.799 | 0.346 | mixed trend |
| AgiBot | 30% | 22% | 16% | -14 pp | Yes | 0.666 | 0.097 | supports claim |
| Pooled | 30% | 20% | 19% | -11 pp | Yes | 0.731 | 0.066 | supports claim |

## Supporting Task-Progress Table

| Dataset | L1 Progress | L2 Progress | L3 Progress | Delta (L1->L3) |
| --- | ---: | ---: | ---: | ---: |
| DROID | 0.492 | 0.420 | 0.434 | -0.058 |
| AgiBot | 0.460 | 0.416 | 0.406 | -0.054 |
| Pooled | 0.476 | 0.418 | 0.420 | -0.056 |

## Suggested Claim

- In the original setting, AgiBot and the pooled unseen-test-set analysis both show a cleaner decline from `L1` to `L3` than the earlier absolute-duration bucket analysis.
- The pooled original trend is `30% -> 20% -> 19%`, with ordered-split `OR=0.731` and `p=0.066`, so it is best described as supportive or marginal rather than fully conclusive.
- DROID alone trends in the same general direction from `L1` to `L2` but is not perfectly monotone because `L3` rebounds slightly.

