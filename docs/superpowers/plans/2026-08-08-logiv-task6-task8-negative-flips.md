# LOGIV Task 6/8 Negative Flips Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove all observed task-6/task-8 negative flips while retaining task-6 recovery and strict no-trigger BASE parity.

**Architecture:** Add two opt-in eligibility flags to the existing online detector: task 6 recovery requires prior goal progress, and task 8 frontier stall requires an empty hand. Reuse the already-tested v5 task-8 phase prompt; do not introduce a new controller or repair path.

**Tech Stack:** Python 3.11, pytest, JSON prompt overlays, LIBERO/OpenPI Docker evaluator.

## Global Constraints

- BASE execution and policy RNG protocol remain unchanged.
- Native LIBERO `done=True` remains absorbing success.
- No intervention may execute after success.
- Same-GPU prefix hashes must match before attributing a result to repair.

---

### Task 1: Task-6 recovery eligibility

**Files:**
- Modify: `tests/logiv/test_online_repair.py`
- Modify: `src/pi05_libero_repro/logiv/online_repair.py`
- Modify: `scripts/eval_logiv_libero.py`
- Modify: `tests/logiv/test_evaluator.py`
- Modify: `scripts/run_logiv_online_shard.sh`
- Modify: `tests/test_launchers.py`

**Interfaces:**
- Consumes: `OnlineGraphDeviationDetector.observe(...)` and evaluator task id.
- Produces: `recovery_requires_achieved_goal: bool`, `stall_requires_handempty: bool`, and audited CLI task-id routing.

- [ ] Write a detector test that presents confirmed recovery evidence before any goal and expects no request, then verifies a request after the goal is observed.
- [ ] Run `pytest tests/logiv/test_online_repair.py -q` and verify the new test fails because the constructor does not accept the flag.
- [ ] Add the boolean constructor field and a single guard in `_recovery_surface`.
- [ ] Run `pytest tests/logiv/test_online_repair.py -q` and verify it passes.
- [ ] Add `--online-recovery-requires-goal-task-ids 6`, pass the task-scoped boolean into detector settings, and serialize the list in `run.json`.
- [ ] Add evaluator and launcher assertions, then run `pytest tests/logiv/test_evaluator.py tests/test_launchers.py -q`.

- [ ] Write a detector test showing that a held object suppresses frontier stall when the flag is enabled, and verify it fails before implementation.
- [ ] Add the handempty stall guard, task-8 CLI routing, run-config field, and launcher assertions.
- [ ] Re-run the four original task-8 negative-flip episodes on GPU 2 with prompt v5.

### Task 2: Focused paired validation

**Files:**
- Create through evaluator: `runs/logiv-online-tuning-t68-20260808/`
- Create through reporter: `results/logiv-online-tuning-t68-20260808.json`
- Create through reporter: `results/logiv-online-tuning-t68-20260808.md`

**Interfaces:**
- Consumes: original BASE records plus focused task-6/task-8 LOGIV records.
- Produces: audited positive/negative/net flip report.

- [ ] Re-run task 6 seed 7 episode 33 and seed 17 episodes 12, 23, and 30 with the eligibility gate.
- [ ] Build a focused paired report for the eight original flip cases.
- [ ] Require positive flips at least 1, negative flips 0, no-trigger parity exact, and audit errors empty.
- [ ] If focused validation passes, run all 200 task-6/task-8 pairs and regenerate the paired report.
