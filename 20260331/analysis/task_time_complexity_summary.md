# Task-Time Complexity Summary

Longer episode duration is used here as a task-complexity proxy.

| Dataset | Split | N | Mean (s) | Median (s) | Range (s) | Delta vs prev (s) | x prev | x L1 |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| DreamDojo-AgiBot | L1 | 50 | 22.08 | 23.57 | 14.37-27.72 | - | - | 1.00 |
| DreamDojo-AgiBot | L2 | 50 | 39.22 | 32.36 | 27.75-67.53 | 17.14 | 1.78 | 1.78 |
| DreamDojo-AgiBot | L3 | 50 | 135.10 | 135.58 | 68.47-264.22 | 95.88 | 3.44 | 6.12 |
| DreamDojo-EgoDex | L1 | 133 | 2.92 | 2.95 | 1.40-3.75 | - | - | 1.00 |
| DreamDojo-EgoDex | L2 | 133 | 4.87 | 4.75 | 3.75-6.40 | 1.96 | 1.67 | 1.67 |
| DreamDojo-EgoDex | L3 | 134 | 12.90 | 11.03 | 6.45-39.75 | 8.03 | 2.65 | 4.42 |
| DreamZero-AgiBot | L1 | 50 | 21.40 | 21.66 | 14.30-27.15 | - | - | 1.00 |
| DreamZero-AgiBot | L2 | 50 | 44.66 | 43.88 | 27.20-69.38 | 23.25 | 2.09 | 2.09 |
| DreamZero-AgiBot | L3 | 50 | 125.25 | 118.43 | 72.77-237.57 | 80.59 | 2.80 | 5.85 |
| DreamZero-DROID | L1 | 50 | 6.60 | 6.87 | 3.20-8.20 | - | - | 1.00 |
| DreamZero-DROID | L2 | 50 | 9.99 | 10.00 | 8.27-11.73 | 3.39 | 1.51 | 1.51 |
| DreamZero-DROID | L3 | 50 | 15.52 | 15.13 | 11.80-28.53 | 5.53 | 1.55 | 2.35 |

## Notes

- `DreamZero-DROID`: archived table split durations from `droid_3way_selected150.json`, using `ep_len / 15`.
- `DreamZero-AgiBot`: archived `Agi_DualBetter_L1/L2/L3_50` manifests.
- `DreamDojo-AgiBot`: the current `20260331` submission manifests.
- `DreamDojo-EgoDex`: exact archived selected-50 split files are missing in the workspace, so durations are recovered from the readable `egodex_dreamdojo_easy400` source manifest by 20 FPS duration terciles.
