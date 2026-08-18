# RoboTwin π₀.₅ Baseline 10×10 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Parameterize the existing RoboTwin π₀.₅ baseline evaluator, verify one end-to-end episode, then run ten tasks over ten distinct accepted simulator seeds per task.

**Architecture:** Keep RoboTwin's existing expert-seed filtering and policy loop intact. A tiny pure option resolver supplies the local checkpoint, trial count, local tokenizer, and video switch; the existing LeRobot wrapper constructs an equivalent local Gemma tokenizer through the processor pipeline's supported override. Three long-lived GPU workers execute fixed task lists sequentially so no GPU hosts more than one simulator/model process.

**Tech Stack:** Python 3.10, PyTorch, LeRobot π₀.₅, modified Hugging Face Transformers 4.53.2, SAPIEN/RoboTwin 2.0, `unittest`, Bash, three NVIDIA H200 GPUs.

## Global Constraints

- Use checkpoint `rhodes-team-teleai/pi05_TACO_robotwin2_finetuned`, local SHA-256 `5af5866f0e5f2ca446ee28d935b0dfc07c72031b455a2318e79249b3278ab87a`.
- Use task configuration `demo_clean`, instruction type `unseen`, action horizon 50, seed group `0`, and ten accepted simulator scenes per task.
- Evaluate exactly the ten tasks listed in the approved design and no others.
- Do not enable TACO CFN, LOGIV, PDDL, GPT-4o, or any intervention around the base π₀.₅ policy.
- Disable episode videos for the preliminary run; retain stdout logs, per-episode outcomes, and simulator seeds.
- Never automatically retry a crashed task. Preserve its log and report its exit code.
- Enforce a 12-hour timeout per task and poll liveness/GPU activity about every 30 seconds.
- Do not modify the root repository's existing LOGIV working-tree changes.

---

### Task 1: Parameterize the Existing Baseline Evaluator

**Files:**
- Create: `artifacts/tools/TACO/third_party/Robotwin/script/pi05_baseline_options.py`
- Create: `artifacts/tools/TACO/third_party/Robotwin/tests/test_pi05_baseline_options.py`
- Modify: `artifacts/tools/TACO/third_party/Robotwin/script/eval_lerobot_torch_pi05.py:20,60-180`
- Modify: `artifacts/tools/TACO/third_party/Robotwin/policy/pi05/pi05_model_torch.py:6-40`
- Modify: `artifacts/tools/TACO/third_party/Robotwin/policy/pi05/deploy_policy.yml:1-17`

**Interfaces:**
- Produces: `resolve_baseline_options(usr_args: dict) -> tuple[str, int, str | None, bool]`.
- Produces: `_tokenizer_processor_kwargs(tokenizer_path: str | None) -> dict`, returning an empty dictionary or `{"preprocessor_overrides": {"tokenizer_processor": {"tokenizer_name": None, "tokenizer": GemmaTokenizer(...)}}}`.
- Consumes: the existing `Lerobot_torch_PI05(task_name, pretrained_checkpoint_path)` constructor, extended with optional `tokenizer_path=None`.

- [ ] **Step 1: Write the failing option-resolution tests**

```python
import unittest

from script.pi05_baseline_options import resolve_baseline_options


class BaselineOptionsTest(unittest.TestCase):
    def test_resolves_explicit_baseline_options(self):
        self.assertEqual(
            resolve_baseline_options(
                {
                    "policy_path": "/checkpoints/pi05",
                    "test_num": "10",
                    "tokenizer_path": "/models/paligemma_tokenizer.model",
                    "record_videos": False,
                }
            ),
            ("/checkpoints/pi05", 10, "/models/paligemma_tokenizer.model", False),
        )

    def test_rejects_missing_checkpoint_and_nonpositive_trial_count(self):
        with self.assertRaisesRegex(ValueError, "policy_path"):
            resolve_baseline_options({"test_num": 10})
        with self.assertRaisesRegex(ValueError, "test_num"):
            resolve_baseline_options({"policy_path": "/checkpoints/pi05", "test_num": 0})
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
cd artifacts/tools/TACO/third_party/Robotwin
PYTHONPATH=.:../lerobot/src:/mnt/data3/data_xingrui/lueq/NuerIPS_2026/artifacts/tools/TACO \
  /mnt/data3/data_xingrui/lueq/NuerIPS_2026/artifacts/tools/robotwin-eval-venv/bin/python \
  -m unittest tests.test_pi05_baseline_options -v
```

Expected: FAIL because `script.pi05_baseline_options` does not exist.

- [ ] **Step 3: Implement the minimal pure resolver**

```python
def resolve_baseline_options(usr_args):
    policy_path = usr_args.get("policy_path")
    if not policy_path:
        raise ValueError("policy_path is required")
    test_num = int(usr_args.get("test_num", 100))
    if test_num < 1:
        raise ValueError("test_num must be positive")
    return (
        str(policy_path),
        test_num,
        usr_args.get("tokenizer_path"),
        bool(usr_args.get("record_videos", True)),
    )
```

- [ ] **Step 4: Add a failing local-tokenizer override test**

Append a test that patches `policy.pi05.pi05_model_torch.GemmaTokenizer`, calls `_tokenizer_processor_kwargs("/models/paligemma_tokenizer.model")`, and asserts that the tokenizer was constructed with `vocab_file`, `add_bos_token=True`, and `add_eos_token=False`, and passed under the existing `tokenizer_processor` override.

Run the same `unittest` command. Expected: FAIL because `_tokenizer_processor_kwargs` does not exist.

- [ ] **Step 5: Implement the local tokenizer override and wire all options**

In `pi05_model_torch.py`, import `GemmaTokenizer`, implement `_tokenizer_processor_kwargs`, extend `Lerobot_torch_PI05.__init__` with `tokenizer_path=None`, and pass `**_tokenizer_processor_kwargs(tokenizer_path)` into `make_pre_post_processors`.

In `eval_lerobot_torch_pi05.py`, resolve the four values immediately after loading `demo_clean`, overwrite only `args["eval_video_log"]`, construct `Lerobot_torch_PI05(task_name, policy_path, tokenizer_path)`, and pass the resolved `test_num` into the existing `eval_policy` loop. Do not alter expert seed filtering, instruction selection, action execution, or success checks.

Add these default keys to `deploy_policy.yml`:

```yaml
policy_path: null
test_num: 100
tokenizer_path: null
record_videos: true
```

- [ ] **Step 6: Verify GREEN and syntax**

Run the `unittest` command from Step 2, followed by:

```bash
/mnt/data3/data_xingrui/lueq/NuerIPS_2026/artifacts/tools/robotwin-eval-venv/bin/python \
  -m py_compile script/pi05_baseline_options.py script/eval_lerobot_torch_pi05.py \
  policy/pi05/pi05_model_torch.py
```

Expected: all tests PASS and `py_compile` exits 0.

- [ ] **Step 7: Commit only the adapter files in the clean TACO repository**

```bash
git add third_party/Robotwin/script/pi05_baseline_options.py \
  third_party/Robotwin/tests/test_pi05_baseline_options.py \
  third_party/Robotwin/script/eval_lerobot_torch_pi05.py \
  third_party/Robotwin/policy/pi05/pi05_model_torch.py \
  third_party/Robotwin/policy/pi05/deploy_policy.yml
git commit -m "fix(robotwin): parameterize pi05 baseline evaluation"
```

### Task 2: Prepare Shared RoboTwin Assets and Verify Preflight State

**Files:**
- Create symlink: `artifacts/tools/TACO/third_party/Robotwin/task_config` → `artifacts/tools/GuidedVLA/third_party/RoboTwin/task_config`
- Create symlink: `artifacts/tools/TACO/third_party/Robotwin/assets/objects` → `artifacts/tools/GuidedVLA/third_party/RoboTwin/assets/objects`
- Create symlink: `artifacts/tools/TACO/third_party/Robotwin/assets/embodiments` → `artifacts/tools/GuidedVLA/third_party/RoboTwin/assets/embodiments`

**Interfaces:**
- Consumes: the unmodified RoboTwin relative paths `./task_config` and `./assets/...`.
- Produces: a runnable evaluator checkout without duplicating 5.3 GB of extracted assets.

- [ ] **Step 1: Create only the three missing symlinks**

Use explicit absolute source and target paths, first asserting that each source exists and each target is absent. Do not replace existing paths.

- [ ] **Step 2: Validate tasks, configuration, assets, checkpoint, tokenizer, and GPUs**

Check that all ten `envs/<task>.py` files exist, `task_config/demo_clean.yml` resolves, the ALOHA embodiment config and representative object directories resolve, the checkpoint SHA-256 matches the global constraint, the tokenizer file is 4,264,023 bytes, GPUs 0-2 are idle, and no prior evaluation processes exist.

Expected: every check succeeds before any simulator episode starts.

### Task 3: Confirm and Execute the One-Episode Smoke Run

**Files:**
- Create at runtime: `artifacts/tools/TACO/third_party/Robotwin/eval_result/robotwin-pi05-baseline-smoke-20260812/...`
- Create at runtime: `results/robotwin-pi05-baseline-10x10-20260812/logs/smoke-handover_block.log`

**Interfaces:**
- Consumes: the parameterized evaluator from Task 1 and shared assets from Task 2.
- Produces: one completed diagnostic episode and `_result.txt`; it is excluded from the 100-episode aggregate.

- [ ] **Step 1: Present the exact fully resolved smoke command to the user and wait for confirmation**

The command must use GPU 0, `handover_block`, `demo_clean`, `seed=0`, `test_num=1`, the absolute local checkpoint and tokenizer paths, `record_videos=False`, and tag `robotwin-pi05-baseline-smoke-20260812`.

- [ ] **Step 2: Execute once with a 12-hour timeout and no automatic retry**

Capture stdout/stderr to `results/robotwin-pi05-baseline-10x10-20260812/logs/smoke-handover_block.log`. Poll liveness and GPU use approximately every 30 seconds.

- [ ] **Step 3: Verify smoke artifacts**

Assert exit code 0, exactly one success/failure episode line, one evaluated simulator seed, and an `_result.txt` containing either `0.0` or `1.0`. If any assertion fails, stop and report the command, exit code, and log tail.

### Task 4: Confirm and Execute the Three-GPU 10×10 Run

**Files:**
- Create at runtime: `results/robotwin-pi05-baseline-10x10-20260812/logs/<task>.log`
- Create at runtime: `artifacts/tools/TACO/third_party/Robotwin/eval_result/robotwin-pi05-baseline-10x10-20260812/<task>/.../_result.txt`

**Interfaces:**
- GPU 0 task list: `handover_block`, `stamp_seal`, `turn_switch`, `beat_block_hammer`.
- GPU 1 task list: `open_microwave`, `blocks_ranking_size`, `stack_blocks_three`.
- GPU 2 task list: `place_dual_shoes`, `move_can_pot`, `stack_bowls_three`.
- Every task consumes identical policy/config/tokenizer/seed settings and produces ten evaluated episodes.

- [ ] **Step 1: Present the exact three-worker Bash command to the user and wait for confirmation**

Each worker must set one `CUDA_VISIBLE_DEVICES` value and sequentially invoke the evaluator once per assigned task. Each task command must use `timeout 12h`, stop its worker on nonzero exit, write a separate log, and pass the exact shared settings from the global constraints.

- [ ] **Step 2: Start the three workers and monitor**

Record worker PIDs. Approximately every 30 seconds, verify each live process, its current task log, GPU memory/utilization, and the absence of duplicate processes on a GPU. Do not retry failures.

- [ ] **Step 3: Validate completion**

Require ten successful task process exits, ten `_result.txt` files, ten episode outcome lines and ten distinct evaluated simulator seeds per task, and 100 episode outcomes overall. Preserve and report any partial data if a worker stops.

### Task 5: Aggregate and Report the Baseline

**Files:**
- Create: `results/robotwin-pi05-baseline-10x10-20260812/summary.json`
- Create: `results/robotwin-pi05-baseline-10x10-20260812/summary.md`

**Interfaces:**
- Consumes: ten task logs and ten `_result.txt` files.
- Produces: per-task successes/10, macro average, total successes/100, seeds, checkpoint/config metadata, GPU assignment, and completion state.

- [ ] **Step 1: Parse logs and cross-check result files**

Use a read-only one-off Python command to parse each task's ten `Success!`/`Fail!` records and corresponding `current seed` values. Cross-check computed success fractions against `_result.txt`; stop and report any mismatch.

- [ ] **Step 2: Write the two summaries**

Write `summary.json` as machine-readable evidence and `summary.md` as a concise table. Include the checkpoint repository and SHA-256, `demo_clean`, `unseen`, action horizon 50, seed group 0, exact evaluated seeds, per-task GPU, success counts, macro average, total success rate, and any incomplete task.

- [ ] **Step 3: Final verification**

Verify both summaries parse/render, their aggregate counts equal the raw logs, and no evaluation process remains. Report the result as a preliminary 10-seed estimate, not a replacement for the official 100-seed-per-task protocol.

