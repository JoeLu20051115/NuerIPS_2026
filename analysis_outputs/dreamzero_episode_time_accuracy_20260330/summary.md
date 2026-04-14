# Episode-Level Time-Accuracy Recovery

- Total recovered rows: 897
- DROID rows: 450
- AgiBot rows: 447

## Per-Dataset / Mode Counts
- AgiBot llm_dual L1: n=50
- AgiBot llm_dual L2: n=50
- AgiBot llm_dual L3: n=50
- AgiBot llm_val L1: n=48
- AgiBot llm_val L2: n=49
- AgiBot llm_val L3: n=50
- AgiBot original L1: n=50
- AgiBot original L2: n=50
- AgiBot original L3: n=50
- DROID llm_dual L1: n=50
- DROID llm_dual L2: n=50
- DROID llm_dual L3: n=50
- DROID llm_val L1: n=50
- DROID llm_val L2: n=50
- DROID llm_val L3: n=50
- DROID original L1: n=50
- DROID original L2: n=50
- DROID original L3: n=50

## Notes
- DROID points come directly from `droid_3way_selected150.json`, so the recovery is exact for all 450 episode-mode rows.
- AgiBot `original` and `llm_dual` come directly from the archived split result JSONs (`Agi_DualBetter_L1/L2/L3_50_results.json`).
- AgiBot `llm_val` is recovered from the archived `agibot_3way_compare.json` on the same split episode ids when present.

