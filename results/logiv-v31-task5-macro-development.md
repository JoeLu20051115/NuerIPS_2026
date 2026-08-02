# LOGIV v31 Task 5 macro development result

This is failure-driven development evidence on LIBERO-10 Task 5, not holdout
evidence. Base and Full used the same full `pi05_libero` checkpoint, the same
GPU/server process, episode seeds 0--9, a 520-step total budget, and the unified
10-step post-terminal settling evaluator.

## Result

- Base: 4/10
- Full LOGIV v31: 5/10
- Paired positive flips: episode 9
- Paired negative flips: none
- Development point difference: +10 percentage points

The v31 nominal plan keeps the official pick-and-place instruction as one
certified `place-in` macro occurrence. After a fresh-fact effect failure, Repair
may use the finer `pick`, `put-down`, and `place-held-in` schemas. This removes
the v28 split-plan regression while retaining fact-gated recovery.

Task 5 has one nominal macro occurrence, so its action layer width is necessarily
one. This does not change the compiler contract: multi-object Task 8 retains a
width-two causal DAG with no artificial edge between the independent placement
occurrences. LOGIV therefore does not globally force plans into linked lists.

The sample is too small for a significance claim. v31 is saved as a rollback
point while the video-diagnosed v32 upright-insertion prompt is evaluated on the
same seeds.

## Artifacts

- Base: `runs/logiv-v30-base-task5-dev10/`
- Full: `runs/logiv-v31-full-task5-macro-dev10/`
- Proposal: `configs/logiv/libero10-scripted-proposals-v31-task5-macro.json`
- Coverage: `configs/logiv/libero10-coverage-v31-task5-macro.json`
