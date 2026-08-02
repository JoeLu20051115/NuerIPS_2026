# Base terminal-settling audit

This development audit checks whether the direct Base arm and LOGIV use the
same physical terminal barrier. Before this correction, Base returned as soon
as LIBERO's step-level `done` signal became true, whereas LOGIV waited for
`STOPPED`, settling, fresh facts, and the external evaluator.

The corrected Base path preserves direct full-task execution but applies the
same configured settling interval before calling `check_success()`. The first
`done` signal and the post-settling evaluator result are logged separately.

## Reproduction

- Run: `runs/logiv-base-settled-task5-e0-v3/`
- Backbone: full `pi05_libero`
- Task: 5
- Episode index: 0
- Policy steps before first `done`: 160
- Settling steps: 10
- First `done`: true
- Post-settling `check_success()`: false
- Terminal receipt evaluator status: `EPISODE_FAIL`
- Video frames: 170 (160 policy + 10 settling)
- Final result: `EPISODE_FAIL`

This demonstrates a concrete transient-success case. Historical Base results
that did not use the settling barrier cannot be mixed with corrected paired
results. It does not by itself estimate how many other episodes change; all
paired Base arms must be rerun under the corrected contract.
