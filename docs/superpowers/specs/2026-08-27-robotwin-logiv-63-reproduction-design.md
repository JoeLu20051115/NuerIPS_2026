# RoboTwin LOGIV 63/100 Reproduction Cleanup Design

**Status:** Approach A selected by the user on 2026-08-27. Implementation is
pending review of this written specification.

## Goal

Publish one compact, runnable RoboTwin 2.0 experiment for the frozen ten-task,
ten-seed-per-task protocol that produced LOGIV 63/100 and direct pi0.5 56/100
in the honest same-seed rerun. The package must make the live protocol easy for
another researcher to rerun while removing the post-hoc path that assembled
72/100 from historical successes.

The clean branch is `agent/robotwin-logiv-63-repro`. Commit `166d7755` is the
GitHub backup of the pre-cleanup experiment state.

## Accuracy Contract

The reproducible object is the complete live protocol: task set, 100 fixed
seeds, matching instructions, policy checkpoint interface, LOGIV parameters,
pi0.5 baseline parameters, runtime patches, and aggregation rules. A completed
run is checked against the observed totals LOGIV 63 and baseline 56.

The code must not claim bitwise or mathematical determinism. GPU kernels,
RoboTwin physics, and the live GPT-4o service can vary. If a fresh run differs,
the command exits nonzero after writing its compact summary; it never searches
for replacement seeds, retries a failed cell as a selection strategy, reuses a
historical success, or rewrites the observed outcome to 63/56.

The concise result note may state that the earlier post-hoc 72/100 collection
lost nine successes in the frozen rerun. It must not retain the selected
historical records or identify a new oracle result.

## Durable Files

The final branch keeps only the following experiment-specific surface:

- `configs/robotwin/logiv-gpt4o-63-vs-pi05-56.json`: the 100 immutable
  `(task, seed, instruction)` inputs and the final task-level control settings.
  It contains no `seed_selection` field and no machine-specific checkpoint
  path.
- `scripts/run_robotwin_logiv_10x10.py` and
  `scripts/run_robotwin_baseline_10x10.py`: worker launchers. Checkpoint and
  runtime paths are explicit CLI inputs.
- `scripts/reproduce_robotwin_logiv_63.py`: the single public entry point. It
  performs preflight, runs the three LOGIV workers, then the three baseline
  workers on the same GPU mapping, aggregates the run, checks 100/100
  completeness and expected 63/56 totals, and compacts generated artifacts.
- `scripts/report_robotwin_logiv.py`: a minimal outcome aggregator. It checks
  exact seed/instruction membership, duplicates, completion, success totals,
  per-task totals, and paired flips. It does not select records, revalidate
  plans with VAL, validate GPT-4o provenance, or require retained images.
- `patches/robotwin/`: only the runtime patches needed for baseline execution
  and live LOGIV control, plus an authoritative ordered README.
- `results/robotwin-logiv-63-vs-pi05-56.{json,md}`: only the compact observed
  63/56 aggregate and per-task table. No episode records, logs, prompts,
  images, certificates, or API responses are committed.
- A short reproduction README and focused tests for the frozen manifest,
  launch commands, aggregation, expectation checks, and safe compaction.

VAL remains an online dependency of `RobotwinPddlPlanner`; that is part of the
method. Only the separate post-hoc VAL revalidation/reporting path is removed.

## Public Command and Execution Topology

The public interface is one command of this form:

```bash
uv run python scripts/reproduce_robotwin_logiv_63.py \
  --taco /path/to/TACO \
  --checkpoint /path/to/pi05_TACO_robotwin2_finetuned \
  --tokenizer /path/to/paligemma_tokenizer.model \
  --val-binary /path/to/Validate \
  --python /path/to/robotwin-python \
  --output /path/to/run-output \
  --gpus 0 1 2
```

`OPENAI_API_KEY` is read only from the environment. The command refuses to
print or serialize it. Preflight requires the TACO/RoboTwin evaluator,
checkpoint, tokenizer, executable VAL binary, three distinct requested GPUs,
the complete frozen manifest, an unused output directory, and a valid key.

The LOGIV phase runs three worker subprocesses concurrently using the existing
task allocation. If all three finish successfully, the baseline phase runs the
same three allocations concurrently. A worker runs each of its tasks in order
and stops on the first nonzero task process. There are no automatic episode
retries or replacement seeds.

The run tag is unique and derived from the output directory name. The driver
passes the tag to TACO and resolves that tag's exact `eval_result` directory;
it never scans unrelated historical worktrees or result roots.

## Output Lifecycle

During execution, the driver writes worker logs and TACO writes native results
and LOGIV event JSONL under the unique tag. Aggregation happens only after both
phases succeed. The compact JSON and Markdown summaries are written under the
requested output directory.

After aggregation, default compaction removes the run's worker logs, native
episode directories, videos, VLM audit images, and event records, leaving only
the frozen manifest copy, resolved non-secret run metadata, and compact
summaries. Cleanup is confined to paths created for the unique run tag. A
failed or incomplete run is retained for diagnosis and clearly labeled
incomplete; the user may remove it explicitly after inspection.

## Removed Historical Surface

The final Git tree deletes:

- `scripts/freeze_robotwin_logiv_seeds.py` and its tests;
- `scripts/report_robotwin_logiv_oracle.py` and its tests;
- seed-scan, oracle-gap, and ablation configs/tests/docs;
- runtime patches for seed-pool discovery, unreachable-seed audit, persistent
  camera evidence, and GPT-4o provenance recording;
- the full `results/robotwin-logiv-gpt4o-63-development-20260817/` directory;
- the four older strict-baseline/oracle result files;
- all committed candidate pools, selected historical records, raw logs,
  symlinked logs, smoke logs, and PNG images.

After the final GitHub push, local cleanup removes only these seven explicit
run directories from the dedicated TACO experiment worktree:

- `logiv-gpt4o-audit-smoke-20260817`
- `logiv-gpt4o-seed-scan-v1-20260817`
- `logiv-gpt4o-seed-selected-final-v1-20260817`
- `pi05-same-seed-baseline-smoke-v1-20260817`
- `pi05-same-seed-baseline-final-v1-20260817`
- `pi05-same-seed-baseline-final-v2-20260817`
- `pi05-same-seed-baseline-final-v3-20260817`

This recovers roughly 1.5 GB. The TACO checkout's tracked modification and its
other untracked runtime files are not touched. The old implementation worktree
is also not reset or force-cleaned.

## Verification and Git Delivery

Implementation follows test-driven development. Focused tests first cover
manifest invariants, CLI command construction, exact event/log aggregation,
63/56 expectation failure, no secret serialization, and cleanup path
confinement. Then the complete repository suite, script `--help`/dry-run
preflight, `git diff --check`, secret scan, and final tree audit must pass.

The second GitHub push contains the design/plan and implementation commits on
the same branch. The pre-cleanup commit remains in branch history, while the
branch tip exposes only the cleaned reproduction package.
