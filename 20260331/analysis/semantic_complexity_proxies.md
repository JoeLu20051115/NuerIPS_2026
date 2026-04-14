# Semantic Complexity Proxies

These are non-time proxies for semantic complexity. Higher values suggest more semantic stages or richer interaction structure.

| Dataset | Split | N | Mean step proxy | Mean unique ops | >=3-step rate | Stateful-op rate | Bimanual rate | Unique task types |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DreamZero-DROID | L1 | 50 | 1.10 | 1.10 | 0.00 | 0.30 | - | 50 |
| DreamZero-DROID | L2 | 50 | 1.32 | 1.32 | 0.08 | 0.34 | - | 49 |
| DreamZero-DROID | L3 | 50 | 1.34 | 1.34 | 0.06 | 0.32 | - | 49 |
| DreamZero-AgiBot | L1 | 50 | 2.62 | 2.06 | 0.32 | 0.02 | 0.30 | 6 |
| DreamZero-AgiBot | L2 | 50 | 2.66 | 1.82 | 0.34 | 0.02 | 0.26 | 6 |
| DreamZero-AgiBot | L3 | 50 | 3.26 | 3.26 | 0.60 | 0.60 | 0.22 | 4 |
| DreamDojo-AgiBot | L1 | 50 | 2.76 | 2.06 | 0.38 | 0.00 | 0.32 | 6 |
| DreamDojo-AgiBot | L2 | 50 | 3.02 | 2.18 | 0.54 | 0.06 | 0.18 | 6 |
| DreamDojo-AgiBot | L3 | 50 | 4.36 | 4.36 | 0.74 | 0.74 | 0.54 | 4 |
| DreamDojo-EgoDex | L1 | 133 | 2.02 | 2.02 | 0.01 | 0.26 | - | 6 |
| DreamDojo-EgoDex | L2 | 133 | 2.14 | 2.14 | 0.07 | 0.52 | - | 8 |
| DreamDojo-EgoDex | L3 | 134 | 2.37 | 2.37 | 0.19 | 0.72 | - | 8 |

## Notes

- `DreamZero-DROID`: semantic proxy is extracted from the natural-language `task` string, so it is a weak but usable heuristic.
- `DreamZero-AgiBot` and `DreamDojo-AgiBot`: semantic proxy uses manifest `action_plan`, which is the strongest signal currently available.
- `DreamDojo-EgoDex`: semantic proxy uses `task_group` verb composition because the source manifest does not expose per-episode action plans.
- The safest presentation is to call these `semantic complexity proxies`, not a perfectly calibrated universal complexity score.
