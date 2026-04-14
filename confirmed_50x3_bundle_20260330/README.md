# Confirmed 50x3 Bundle

This directory is a clean bundle of the confirmed `50/50/50` split artifacts that are still available in the repo.

## Included

- `Dreamzero_DROID/source/droid_3way_selected150.json`
  Exact confirmed source for the `Dreamzero-DROID` 50x3 appendix table.
  This file already contains `summary` and `selection`, with `selection.method = top150_by_val_gap`.

- `Dreamzero_DROID/derived/`
  Cleanly split-out files derived from `droid_3way_selected150.json`:
  `DRO_L1_50_results.json`, `DRO_L2_50_results.json`, `DRO_L3_50_results.json`,
  `DRO_L1_50_episodes.tsv`, `DRO_L2_50_episodes.tsv`, `DRO_L3_50_episodes.tsv`,
  and `droid_50x3_summary.tsv`.

- `Dreamzero_AgiBot/data/Agi_DualBetter_L1_50/meta/manifest.json`
- `Dreamzero_AgiBot/data/Agi_DualBetter_L2_50/meta/manifest.json`
- `Dreamzero_AgiBot/data/Agi_DualBetter_L3_50/meta/manifest.json`
  These are the explicit saved `50/50/50` AgiBot split manifests still present in the repo.

- `Dreamzero_AgiBot/results/`
  Saved AgiBot 50x3 split results and episode lists:
  `Agi_DualBetter_L1_50_results.json`, `Agi_DualBetter_L2_50_results.json`,
  `Agi_DualBetter_L3_50_results.json`,
  `Agi_DualBetter_L1_50_episodes.tsv`, `Agi_DualBetter_L2_50_episodes.tsv`,
  `Agi_DualBetter_L3_50_episodes.tsv`, and `Agi_DualBetter_50x3_summary.tsv`.

- `Dreamzero_AgiBot/results/*.log`
  The `Agi_DualBetter` log files were recovered from `git` into this bundle only, so the current working tree stays untouched.

## Notes

- `Dreamzero-DROID`: exact confirmed 50x3 source is bundled here.
- `Dreamzero-AgiBot`: the explicit saved `50/50/50` traces left in the repo are the `Agi_DualBetter_*_50` manifests, result JSONs, TSVs, and logs.
- This bundle is intentionally separated from the earlier appendix整理, so you can inspect the 50x3 artifacts directly without mixing them with other logs/results.
