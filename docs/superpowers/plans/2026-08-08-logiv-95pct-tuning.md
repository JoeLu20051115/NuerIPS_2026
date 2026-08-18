# LOGIV 95% Development-Tuning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Raise LOGIV's paired LIBERO-10 development result from 930/1000 to at least 950/1000 without increasing the 520-action budget or introducing any negative flip.

**Architecture:** Keep BASE, scripted proposals, oracle grounding, online deviation confirmation, and native-success absorption unchanged. Change only task-8 prompt overlays and task-8 stall timing, select candidates through paired failure/safety screens, and reuse the unchanged records for other tasks in the final audited report.

**Tech Stack:** Python 3.11, JSON prompt overlays, pytest, jq, LIBERO/OpenPI Docker evaluator, three H200 policy servers on ports 8010/8020/8030.

## Global Constraints

- BASE keeps the original pi0.5 protocol until a strictly verified deviation.
- Both arms retain a maximum of 520 low-level actions.
- Native LIBERO `done=True` is absorbing success; no action may execute afterward.
- Negative flips must equal zero.
- No-trigger outcome, step, and policy-request parity must be exact.
- Evidence remains labelled development/tuning, not independent holdout.

---

### Task 1: Add two task-8-only prompt candidates

**Files:**
- Create: `configs/logiv/prompts/pi05-subtasks-online-v6-direct.json`
- Create: `configs/logiv/prompts/pi05-subtasks-online-v7-contextual.json`
- Modify: `tests/logiv/test_evaluator.py`

**Interfaces:**
- Consumes: `SubtaskPromptRenderer.render`, `render_phase`, `render_recovery_frontier`, and the task-8 actions from `libero10-scripted-proposals-online-v1.json`.
- Produces: two resolved prompt configurations that leave all non-task-8 prompt entries inherited from `pi05-subtasks-online-v5.json`.

- [ ] **Step 1: Write failing renderer tests**

Add tests that assert v6 has no `finish` phase for all four task-8 sourced placement actions and renders direct left/right placement prompts. Add a v7 test that asserts both task-8 initial actions share `put both moka pots on the stove` through `render_recovery_frontier`, while its single-action second-pot prompt contains `remaining moka pot` and `without moving the other moka pot`.

- [ ] **Step 2: Verify the tests fail because the files do not exist**

Run:

```bash
uv run pytest -q tests/logiv/test_evaluator.py -k 'online_v6 or online_v7'
```

Expected: both new tests fail with `FileNotFoundError`.

- [ ] **Step 3: Create the v6 direct overlay**

Create a JSON overlay extending `pi05-subtasks-online-v5.json`. Override the four task-8 initial/recovery `place-on` actions with direct left/right placement text and assign `{}` to each matching `action_phase_overrides` entry so inherited acquire/finish phases are removed.

- [ ] **Step 4: Create the v7 contextual overlay**

Create a JSON overlay extending `pi05-subtasks-online-v5.json`. Port only the task-8 labels, action overrides, held-action overrides, recovery frontier overrides, and frontier completion overrides from `pi05-subtasks-v26.json`; do not copy any non-task-8 prompt entry.

- [ ] **Step 5: Run focused and full tests**

Run:

```bash
uv run pytest -q tests/logiv/test_evaluator.py -k 'online_v'
uv run pytest -q
git diff --check
```

Expected: all tests pass and `git diff --check` is silent.

---

### Task 2: Screen prompt candidates on failures and safety sentinels

**Files:**
- Create through evaluator: `runs/logiv-online-tuning-95-20260808/v6-*`
- Create through evaluator: `runs/logiv-online-tuning-95-20260808/v7-*`
- Create through reporter: `results/logiv-online-tuning-95-prompt-screen-20260808.{json,md}`

**Interfaces:**
- Consumes: task-8 seed-7 failure indices `2,4,6,7,12,13,17,20,21,26,30,33,35,39,41,42,43,49`, seed-17 failure indices `2,4,7,12,15,16,17,20,21,22,23,26,33,35,37,38,39,41,42,44,47`, and triggered-success safety indices seed 7 `10,15,19,38,40`, seed 17 `1,9,10,32,43`.
- Produces: paired outcomes for both prompt candidates under the unchanged 180-step stall threshold.

- [ ] **Step 1: Launch v6 on three GPUs**

Use `scripts/run_logiv_eval.sh LOGIV_ONLINE` with GPU/port pairs `0/8010`, `1/8020`, and `2/8030`. Use the same options as `scripts/run_logiv_online_shard.sh`, the v6 prompt, task id 8, the exact episode lists above, seeds 7/17, `--online-stall-steps 180`, and `--online-stall-requires-handempty-task-ids 8`. GPU 2 runs both safety subsets sequentially.

- [ ] **Step 2: Launch v7 on the same three-way partition**

Repeat Step 1 with the v7 prompt and separate empty output directories.

- [ ] **Step 3: Audit the prompt screen**

Pair each record against the existing task-8 BASE record with the same seed and episode. Require all 10 safety sentinels to remain successful. Rank v6/v7 by recovered failures, then mean repair actions, then grounding-error count.

- [ ] **Step 4: Freeze the winning prompt name and resolved hash**

Record the winning prompt path, prompt version, and `resolved_json_sha256` in the prompt-screen JSON/Markdown report. Do not edit the shared launcher until the full task-8 gate passes.

---

### Task 3: Screen task-8 stall timing on all 100 pairs

**Files:**
- Create through evaluator: `runs/logiv-online-tuning-95-20260808/task8-stall120-*`
- Create through evaluator: `runs/logiv-online-tuning-95-20260808/task8-stall80-*`
- Create through reporter: `results/logiv-online-tuning-95-task8-20260808.{json,md}`

**Interfaces:**
- Consumes: the Task-2 winning prompt and existing task-8 BASE records.
- Produces: complete paired task-8 records for thresholds 120 and 80.

- [ ] **Step 1: Run threshold 120 across three GPUs**

Partition seed 7 episodes `0:50` across GPUs 0 and 1 and seed 17 episodes `0:50` on GPU 2, or use an equivalent balanced disjoint partition. Keep minimum intervention 120, confirmations 3, monitor interval 5, strict handempty gate, and total budget 520.

- [ ] **Step 2: Build the 100-pair threshold-120 report**

Run `scripts/report_logiv_online.py` against the existing task-8 BASE directories and the threshold-120 online root. Require 100 paired records, zero errors, zero negative flips, and exact no-trigger parity.

- [ ] **Step 3: Run threshold 80 and report it**

Repeat Steps 1–2 with only `--online-stall-steps 80` changed. This is a one-variable timing test.

- [ ] **Step 4: Select the safe timing candidate**

Reject any candidate with a negative flip or parity error. Among remaining candidates, select the one with most successes. The direct target is task 8 at least 81/100.

---

### Task 4: Conditional fallback on other triggered failures

**Files:**
- Reuse: existing task-scoped prompt overlays under `configs/logiv/prompts/`
- Create through evaluator: `runs/logiv-online-tuning-95-20260808/fallback-*`

**Interfaces:**
- Consumes: only triggered failures from tasks 0/2/3/6/7/9 and their triggered-success safety sentinels.
- Produces: additional zero-negative positive flips if task 8 ends below 81/100.

- [ ] **Step 1: Compute the exact remaining success deficit**

Set `deficit = 950 - (869 + winning_task8_successes)`. Skip this task when `deficit <= 0`.

- [ ] **Step 2: Screen existing task-scoped overlays one task at a time**

Use the already-created v54/v63/v71/v73/v74/v77/v78/v82/v93/v94 prompt overlays only on each task's currently triggered failures and triggered successes. Do not combine overlays until each individual task has a zero-negative winner.

- [ ] **Step 3: Run the complete 100-pair task gate for each accepted overlay**

Require the task's positive flips to cover the remaining deficit cumulatively, negative flips zero, errors empty, and no-trigger parity exact.

---

### Task 5: Freeze and verify the final 1000-pair development result

**Files:**
- Modify: `scripts/run_logiv_online_shard.sh`
- Create: `results/logiv-online-final-95-paired-20260808.json`
- Create: `results/logiv-online-final-95-paired-20260808.md`

**Interfaces:**
- Consumes: winning affected-task online records and unchanged paired records for unaffected tasks.
- Produces: the final audited 10-task development comparison.

- [ ] **Step 1: Update the launcher only with validated task-scoped settings**

Change the prompt path/version to the frozen winner and, if selected, the validated task-8 stall threshold. Preserve every other protocol flag.

- [ ] **Step 2: Build the merged 1000-pair report**

Use `scripts/report_logiv_online.py` with BASE and LOGIV roots containing exactly one record for every `(seed, task, episode)` key. Reuse an old record only for a task whose resolved behavior and task-scoped settings are unchanged.

- [ ] **Step 3: Assert the hard gates**

Use `jq -e` to require `paired_episodes == 1000`, `base_successes == 924`, `online_successes >= 950`, `flips.negative == 0`, `flips.net >= 26`, exact no-trigger parity, and an empty `errors` array. Independently assert every record has `combined_actions <= 520` and every native success has no later action artifact.

- [ ] **Step 4: Run final verification**

Run:

```bash
uv run pytest -q
git diff --check
```

Expected: the full suite passes and the diff check is silent.
