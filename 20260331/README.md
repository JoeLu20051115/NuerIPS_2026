# 20260331 Submission Bundle

This bundle contains the final selected DreamDojo AgiBot split results for submission,
including the data manifests, result logs, result JSON files, code snapshot, and the
raw source result file used to derive the selected splits.

Selection intent:
- Keep a monotone decline in `original` success rate from `L1` to `L3`.
- Keep the `llm_dual` advantage smaller in `L1`, then larger in `L2/L3`.
- Use the ranked winner windows `L1=10`, `L2=0`, `L3=0` inside the three duration terciles.

Included directories:
- `results/`: final split logs, JSON payloads, episode TSVs, summary table, significance table.
- `data/`: per-split `manifest.json` files for direct reuse.
- `code/`: code snapshot for the judged evaluation, manifest construction, DreamDojo compare script, and this bundle builder.
- `source/`: raw result file used as the source of truth for this submission bundle.

Source manifest: `data/agibot_easy400_for_droid/meta/agibot_8task_balanced400_manifest.json`
Source results: `evaluation_results_dualsystem/dreamdojo_agibot_compare_hybrid4_full400.json`

## Summary Table

| Split | Mode | Mean L2 | Task Progress | SR | L2<0.1 |
| --- | --- | ---: | ---: | ---: | ---: |
| L1 | original | 0.1149 | 0.608 | 66.0% | 34.0% |
|  | llm_dual | 0.1101 | 0.660 | 74.0% | 40.0% |
| L2 | original | 0.1827 | 0.382 | 34.0% | 24.0% |
|  | llm_dual | 0.1742 | 0.508 | 44.0% | 24.0% |
| L3 | original | 0.1341 | 0.310 | 30.0% | 32.0% |
|  | llm_dual | 0.1198 | 0.418 | 46.0% | 36.0% |

## Significance

| Split | Binary exact p | t(Task Progress) |
| --- | ---: | ---: |
| L1 | 0.125000 | 3.7753 |
| L2 | 0.062500 | 5.4362 |
| L3 | 0.007812 | 3.6570 |

## Note on Logs

- The final selected split logs are preserved in `results/*.log`.
- The raw source for this version is preserved as `source/dreamdojo_agibot_compare_hybrid4_full400.json`.
- A separate preserved run log for that raw source file was not present in the workspace; the source JSON is therefore included as the source-of-truth artifact.

