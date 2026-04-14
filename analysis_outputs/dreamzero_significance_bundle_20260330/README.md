# DreamZero Significance Bundle

This folder collects the DROID and AgiBot significance tables in one place.

Primary files:
- `combined_sr_significance_summary.md`: combined DROID + AgiBot SR significance summary.
- `paper_ready_sr_table.md`: compact appendix-style table.
- `droid_exact_paired_significance.md`: exact paired raw table from `droid_3way_selected150.json`.
- `agibot_recoverable_paired_significance.md`: recoverable paired raw table from `Agi_DualBetter_*_50_results.json` plus overlapping `agibot_3way_compare.json` rows.
- `screenshot_sr_count_significance.md`: conservative screenshot-cell Fisher tests for both datasets.

Interpretation:
- DROID uses exact paired raw episode data.
- AgiBot uses the best recoverable paired raw data currently available in the repo/archive; it is suitable for significance analysis, but it is not guaranteed to be identical to the screenshot-era 3-mode split file.
- The screenshot count-based table is preserved here as a conservative cross-check.
