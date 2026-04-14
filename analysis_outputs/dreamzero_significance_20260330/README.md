# DreamZero Significance Tables

Files:
- Exact paired DROID significance: `analysis_outputs/dreamzero_significance_20260330/droid_exact_paired_significance.csv` / `analysis_outputs/dreamzero_significance_20260330/droid_exact_paired_significance.md`
- Recoverable paired AgiBot significance: `analysis_outputs/dreamzero_significance_20260330/agibot_recoverable_paired_significance.csv` / `analysis_outputs/dreamzero_significance_20260330/agibot_recoverable_paired_significance.md`
- Screenshot-level SR significance (count-based for both DROID and AgiBot): `analysis_outputs/dreamzero_significance_20260330/screenshot_sr_count_significance.csv` / `analysis_outputs/dreamzero_significance_20260330/screenshot_sr_count_significance.md`

Notes:
- `droid_exact_paired_significance.*` uses exact episode-level raw data from `evaluation_results_dualsystem/droid_3way_selected150.json`.
- `agibot_recoverable_paired_significance.*` uses exact split raw for `original`/`llm_dual` from `Agi_DualBetter_L1/L2/L3_50_results.json`, plus recoverable overlapping `llm_val` rows from `agibot_3way_compare.json` on the same episode ids.
- The DROID raw file gives `L3/original` success rate `20%`; the screenshot shows `22%`. The exact-paired table uses the raw file as source of truth.
- The recoverable AgiBot table is useful for paired testing, but it is not guaranteed to be identical to the screenshot table because the exact screenshot-era 3-mode AgiBot split file is not currently recoverable from the repo/archive.
- `screenshot_sr_count_significance.*` uses the 50-episode success-rate cells shown in the screenshot tables and applies Fisher exact tests on success counts.
- The screenshot-level SR table is conservative and does not use episode pairing, because the exact paired 3-mode Dreamzero-AgiBot split file is not currently recoverable from the repo/archive.
