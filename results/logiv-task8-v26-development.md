# LOGIV task-8 v26 development result

This is development/tuning evidence on LIBERO-Long task 8 (`put both moka
pots on the stove`), not an untouched holdout and not a ten-task result. The
50 episode indices were inspected during prompt and controller development.
The raw v26 run omitted `--development-only`; this note is the authoritative
provenance correction and the run must not be presented as holdout evidence.
The paired Base arm also predates the Base terminal-settling correction added
after the v28 all-task smoke. It stopped at LIBERO's first `done` signal while
Full LOGIV used `STOPPED -> settling -> fresh evaluation`. The `+10` point
estimate below is therefore historical development evidence, not the final
paired estimator; Base must be rerun under the unified terminal contract.

## Frozen checkpoint

- Git commit: `14e7246d`
- Git tag: `logiv-task8-v26-72pct-dev`
- Full run: `runs/logiv-task8-v26-budget180-full50/`
- Paired Base: `runs/logiv-task8-base-v26-paired-full50/`
- Report: `runs/logiv-task8-v26-paired-report.{json,md}`
- Prompt: `configs/logiv/prompts/pi05-subtasks-v26.json`
- Backbone: full `pi05_libero` checkpoint (not the 2,000-step early checkpoint)

## Result

| Arm | Success | Rate | Wilson 95% interval |
| --- | ---: | ---: | ---: |
| Base | 31/50 | 62% | [48.2%, 74.1%] |
| Full LOGIV v26 | 36/50 | 72% | [58.3%, 82.5%] |

The paired estimate is `+10` percentage points. The 10,000-sample paired
bootstrap percentile interval is `[-4,+24]` percentage points. There are nine
positive flips (`0,4,7,8,10,13,39,42,49`) and four negative flips
(`9,17,27,31`), for a net gain of five episodes. This is a useful optimization
signal, but its interval includes zero and it must be validated across tasks.

All 50 Base/Full pairs have identical initial-state and first-frame hashes,
checkpoint, seed, prompt manifest, proposal manifest, coverage manifest, and
PDDL Domain hash. All 100 allocated episodes remain in the denominator.

## Non-chain graph audit

Every Full episode has initial action-layer width two. The two task-8 action
occurrences have zero action-to-action edges in every initial graph. The
improvement therefore does not come from adding a synthetic `place-both`
action or turning the causal DAG into a linear adjacency list.

The fixed typed STRIPS Domain was intentionally not changed: the two
`place-on` occurrences were already independent and VAL-valid. Failure traces
localized the regression to policy prompt distribution and recovery execution
boundaries. v26 changes bounded recovery-frontier prompt selection and sibling
completion only after verified primary effects, while keeping the episode
budget at 520 low-level actions.

## Next gate

`pi05-subtasks-v27.json` merges v26 task-8 behavior with the v12 prompt coverage
for the other nine tasks. It is a development candidate until an interactive
all-ten-task smoke run excludes obvious regressions. No ten-task improvement
claim is valid until matched task-wise episodes and macro-average paired
statistics are complete.
