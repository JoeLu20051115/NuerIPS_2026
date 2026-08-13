# RoboTwin LOGIV Final-Stack Oracle Gap Sweep Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute one audited, task-directed attempt for every one of the 40 cells missing from the existing 60/100 RoboTwin LOGIV development oracle, then rebuild and VAL-audit the oracle.

**Architecture:** Add one backward-compatible explicit GPU route to the existing launcher and encode the frozen gap complement in four small profile configs. Unit tests derive the expected gap from the committed oracle rather than duplicating the success set. Two fail-fast GPU queues run one process per task against the fixed TACO runtime, after which the existing oracle reporter selects across historical and new evidence.

**Tech Stack:** Python 3.11, pytest, RoboTwin 2.0, TACO pi0.5, LOGIV PDDL/VAL runtime, two NVIDIA H200 GPUs, JSON event records.

## Global Constraints

- Keep the evidence label `development oracle`; never call the result a frozen configuration or fair generalization result.
- Run exactly the 40 approved cells, once each. Do not retry, change seeds, or tune settings after launch.
- Use implementation commit descended from `554d3118` and TACO runtime `8de0ed9520989f9fd156904291d0895b9a361886`.
- Use GPUs 0 and 1 only. Leave GPU 2 unused.
- Preserve all unrelated dirty-worktree files and commit only explicitly named files.
- A failed task stops only its own GPU queue. Preserve partial artifacts and let the other queue finish.

---

### Task 1: Add explicit GPU routing with TDD

**Files:**
- Modify: `tests/test_robotwin_launcher.py`
- Modify: `scripts/run_robotwin_logiv_10x10.py`

**Interfaces:**
- Consumes: optional `--gpu {0,1,2}`, existing required `--worker`, optional explicit `--tasks`.
- Produces: `_run_worker(selected_gpu, selected_tasks, args)` while preserving historical default worker groups.

- [ ] **Step 1: Write the failing route test**

Patch `sys.argv`, replace `_run_worker` with a recorder, and assert that
`--worker 0 --gpu 1 --tasks place_dual_shoes turn_switch` invokes
`_run_worker(1, ("place_dual_shoes", "turn_switch"), args)`.

- [ ] **Step 2: Run the focused test and verify RED**

```bash
uv run pytest -q tests/test_robotwin_launcher.py -k explicit_gpu
```

Expected: fail because `--gpu` is not accepted and worker 0 rejects the task list.

- [ ] **Step 3: Implement the minimum backward-compatible route**

Define `ALL_TASKS` from `GPU_TASKS`, add optional `--gpu`, validate explicit
tasks against `ALL_TASKS`, and use the explicit GPU when present. With no
explicit tasks, retain the worker's historical task group. With no explicit
GPU, retain `gpu == worker`.

- [ ] **Step 4: Run focused launcher tests and verify GREEN**

```bash
uv run pytest -q tests/test_robotwin_launcher.py
```

---

### Task 2: Encode and audit the four frozen profiles with TDD

**Files:**
- Create: `tests/test_robotwin_oracle_gap_configs.py`
- Create: `configs/robotwin/logiv-oracle-gap-final-cfn.json`
- Create: `configs/robotwin/logiv-oracle-gap-final-registered.json`
- Create: `configs/robotwin/logiv-oracle-gap-final-original.json`
- Create: `configs/robotwin/logiv-oracle-gap-final-replan.json`

**Interfaces:**
- Consumes: the committed 10x10 protocol and 60/100 oracle report.
- Produces: exactly 40 unique `(task, seed, instruction)` entries split across four mutually exclusive profiles.

- [ ] **Step 1: Write failing config audits**

Tests must derive the 100 protocol cells from
`configs/robotwin/logiv-pddl-10x10-v1.json`, subtract the oracle report's
`selected` cells, and assert equality with the four config unions. Also assert
exact episode-instruction alignment, no duplicates, expected task membership,
and every profile flag, threshold, and minimum-base value.

- [ ] **Step 2: Run the audit and verify RED**

```bash
uv run pytest -q tests/test_robotwin_oracle_gap_configs.py
```

Expected: fail because the four configs do not exist.

- [ ] **Step 3: Create the minimum four configs**

Copy only the requested task entries from the frozen protocol, subset both seed
and instruction arrays together, and encode the profile values approved in the
design. Do not include `stack_bowls_three`.

- [ ] **Step 4: Run focused config and launcher tests and verify GREEN**

```bash
uv run pytest -q tests/test_robotwin_oracle_gap_configs.py tests/test_robotwin_launcher.py
```

- [ ] **Step 5: Commit the audited launch surface**

```bash
git add scripts/run_robotwin_logiv_10x10.py \
  tests/test_robotwin_launcher.py tests/test_robotwin_oracle_gap_configs.py \
  configs/robotwin/logiv-oracle-gap-final-cfn.json \
  configs/robotwin/logiv-oracle-gap-final-registered.json \
  configs/robotwin/logiv-oracle-gap-final-original.json \
  configs/robotwin/logiv-oracle-gap-final-replan.json
git commit -m "exp(robotwin): add audited final-stack gap sweep"
```

---

### Task 3: Verify both repositories and preflight runtime assets

**Files:**
- Read only: implementation and TACO test suites, checkpoint, tokenizer, VAL, RoboTwin assets, GPUs.

- [ ] **Step 1: Run the complete implementation suite**

```bash
uv run pytest -q
```

- [ ] **Step 2: Run the 14 focused TACO runtime tests**

```bash
cd /mnt/data3/data_xingrui/lueq/NuerIPS_2026/.worktrees/taco-robotwin-logiv-cfn-monitored/third_party/Robotwin
/usr/bin/python3 -m unittest -v tests.test_pi05_logiv_execution tests.test_pi05_logiv_options
```

- [ ] **Step 3: Verify fixed assets and idle execution state**

Require the policy checkpoint, the 4,264,023-byte tokenizer at
`/mnt/data3/data_xingrui/.cache/openpi/big_vision/paligemma_tokenizer.model`,
executable VAL at
`/mnt/data3/data_xingrui/lueq/NuerIPS_2026/artifacts/tools/val-ubuntu22/Validate`,
the RoboTwin task config/assets, a valid `OPENAI_API_KEY`, idle GPUs 0/1, and no
existing RoboTwin evaluator process.

---

### Task 4: Launch and monitor the two frozen queues

**Files:**
- Create at runtime: `results/robotwin-logiv-oracle-gap-final-stack-20260813/logs/gpu0/<task>.log`
- Create at runtime: `results/robotwin-logiv-oracle-gap-final-stack-20260813/logs/gpu1/<task>.log`
- Create at runtime: TACO `third_party/Robotwin/eval_result/robotwin-logiv-oracle-gap-final-stack-20260813/<task>/...`

- [ ] **Step 1: Print the fully resolved commands before launch**

Every task command uses the implementation launcher, explicit GPU, its profile
config, fixed TACO/logiv/checkpoint/tokenizer/VAL paths, tag
`robotwin-logiv-oracle-gap-final-stack-20260813`, and `timeout 12h`.

- [ ] **Step 2: Start GPU 0 queue**

Run `open_microwave` and then `stack_blocks_three` as separate launcher
processes with the replan config and `--gpu 0`. Stop on the first nonzero exit.

- [ ] **Step 3: Start GPU 1 queue concurrently**

Run `handover_block`, `move_can_pot`, and `beat_block_hammer` with the CFN
config; `turn_switch` with registered prompts; then `place_dual_shoes`,
`stamp_seal`, and `blocks_ranking_size` with original prompts, all as separate
launcher processes with `--gpu 1`. Stop on the first nonzero exit.

- [ ] **Step 4: Monitor without mutating the experiment**

Approximately every 30 seconds inspect process liveness, current log tails,
episode/event counts, GPU memory/utilization, and duplicate GPU processes. Do
not retry failures or change a profile.

- [ ] **Step 5: Verify raw completion**

Require both queues to exit zero, exactly 40 requested seed outcomes across the
nine task result trees, and exactly one LOGIV episode record for every requested
cell. If incomplete, retain and report the exact partial set.

---

### Task 5: Rebuild, VAL-audit, and report the development oracle

**Files:**
- Create: `results/robotwin-logiv-strict-baseline-instruction-oracle-final-stack-development-20260813.json`
- Create: `results/robotwin-logiv-strict-baseline-instruction-oracle-final-stack-development-20260813.md`

- [ ] **Step 1: Rebuild the historical-plus-new selection**

```bash
uv run python scripts/report_robotwin_logiv_oracle.py \
  --config configs/robotwin/logiv-pddl-10x10-v1.json \
  --events-root /mnt/data3/data_xingrui/lueq/NuerIPS_2026/.worktrees \
  --embed-records --require-baseline-instruction \
  --revalidate-val-binary /mnt/data3/data_xingrui/lueq/NuerIPS_2026/artifacts/tools/val-ubuntu22/Validate \
  --checkpoint-file /mnt/data3/data_xingrui/lueq/NuerIPS_2026/artifacts/checkpoints/pi05_TACO_robotwin2_finetuned/model.safetensors \
  --runtime-commit 8de0ed9520989f9fd156904291d0895b9a361886 \
  --output results/robotwin-logiv-strict-baseline-instruction-oracle-final-stack-development-20260813.json
```

- [ ] **Step 2: Audit the embedded result**

Load the report through `audit_embedded_report`, require 100 expected cells,
baseline instruction matches equal successes, all selected event occurrences
freshly VAL-valid, and confirm that every new selected success is one of the 40
frozen gap cells.

- [ ] **Step 3: Write a concise evidence-labeled Markdown report**

Report the previous 60/100, newly recovered gap count, final total, per-task
totals, raw 40-cell completion, source tag, runtime commit, and VAL result.

- [ ] **Step 4: Run final verification and commit only durable evidence**

```bash
uv run pytest -q
git diff --check
git add results/robotwin-logiv-strict-baseline-instruction-oracle-final-stack-development-20260813.json \
  results/robotwin-logiv-strict-baseline-instruction-oracle-final-stack-development-20260813.md
git commit -m "exp(robotwin): record final-stack oracle gap result"
```

Do not add simulator logs, videos, symlinked tools, checkpoints, or unrelated
pre-existing results.
