# llm_val-Only Significance Notes

How to read the statistics:
- `Binary exact p (BH)` is the Benjamini-Hochberg corrected exact paired p-value for binary success. Smaller means the success improvement from `original` to `llm_val` is less likely to be due to chance.
- `t(Task Progress)` is the paired t-statistic on per-episode task progress differences (`llm_val - original`). Larger positive values mean `llm_val` improves progress more consistently.
- `Prog p (BH)` is the corrected p-value for that task-progress t-test.
- `t(Mean L2)` is computed in the better direction (`original - llm_val`), so a larger positive value means `llm_val` significantly reduces L2 error.
- `L2 p (BH)` is the corrected p-value for the L2 t-test.

Significance shorthand:
- `*` means corrected p < 0.05
- `**` means corrected p < 0.01
- `***` means corrected p < 0.001

Interpretation rule of thumb:
- If `Binary exact p (BH)` is significant, then `llm_val` improves success rate beyond random fluctuation.
- If `Prog p (BH)` is significant with positive `t(Task Progress)`, then `llm_val` improves task completion progress consistently across episodes.
- If `L2 p (BH)` is significant with positive `t(Mean L2)`, then `llm_val` reduces reconstruction / alignment error consistently across episodes.
