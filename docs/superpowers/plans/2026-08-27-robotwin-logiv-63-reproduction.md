# RoboTwin LOGIV 63/100 Reproduction Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the historical seed-selection/oracle artifact stack with one portable live 10x10 rerun command whose frozen protocol produced LOGIV 63/100 and pi0.5 56/100.

**Architecture:** A machine-independent JSON manifest freezes the 100 task/seed/instruction cells and LOGIV controls. Two small worker launchers execute LOGIV and baseline phases, a minimal reporter pairs their outcomes, and one public driver performs preflight, concurrent three-GPU phase execution, expected-count checking, and path-confined compaction.

**Tech Stack:** Python 3.11+, standard library, pytest 8, RoboTwin 2.0, TACO pi0.5, GPT-4o, VAL.

## Global Constraints

- Preserve commit `166d7755` as the already-pushed pre-cleanup backup.
- Keep exactly ten tasks and ten fixed seed/instruction pairs per task.
- Never search, replace, rank, retry-select, or recover a historical seed.
- Do not retain post-hoc VAL revalidation, GPT-4o provenance, camera audit, episode logs, or historical success records.
- VAL remains required only for the online LOGIV planner.
- Never print or serialize `OPENAI_API_KEY`.
- Default successful-run compaction may delete only the current output's `raw/` directory and the two exact TACO tag directories created by that run.
- A live count mismatch must be reported honestly and return nonzero; it must never be rewritten to 63/56.
- Use no new third-party dependencies.

---

### Task 1: Freeze the portable 100-cell manifest and compact expected result

**Files:**
- Create: `configs/robotwin/logiv-gpt4o-63-vs-pi05-56.json`
- Create: `results/robotwin-logiv-63-vs-pi05-56.json`
- Create: `results/robotwin-logiv-63-vs-pi05-56.md`
- Create: `tests/test_robotwin_63_manifest.py`

**Interfaces:**
- Consumes: the exact `tasks`, `instructions`, and task-level control values from `results/robotwin-logiv-gpt4o-63-development-20260817/frozen-10x10-v1.json`.
- Produces: a manifest with `expected_successes == {"logiv": 63, "baseline": 56}`, no `checkpoint` key, and no `seed_selection` key.

- [ ] **Step 1: Write the failing manifest test**

```python
def test_manifest_is_exactly_ten_by_ten_and_portable() -> None:
    manifest = json.loads(MANIFEST.read_text())
    assert len(manifest["tasks"]) == 10
    assert all(len(seeds) == 10 for seeds in manifest["tasks"].values())
    assert set(manifest["tasks"]) == set(manifest["instructions"])
    assert all(
        len(manifest["instructions"][task]) == len(seeds)
        for task, seeds in manifest["tasks"].items()
    )
    assert manifest["expected_successes"] == {"logiv": 63, "baseline": 56}
    assert "checkpoint" not in manifest
    assert "seed_selection" not in manifest
```

- [ ] **Step 2: Run the test and verify RED**

Run: `uv run pytest -q tests/test_robotwin_63_manifest.py`

Expected: FAIL because the portable manifest and compact results do not exist.

- [ ] **Step 3: Create the minimal manifest and result files**

Copy the frozen task/seed/instruction arrays and these control keys verbatim:
`task_config`, `instruction_type`, `action_chunk_steps`,
`repair_action_chunk_steps`, `vlm_image_detail`, `dag_from_start`,
`repair_cfn_tasks`, `use_registered_dag_prompts_by_task`,
`preserve_original_repair_prompt_by_task`, `policy_replan_steps_by_task`,
`min_base_steps_by_task`, `base_stall_observations_by_task`, and
`stage_stall_observations_by_task`. Set:

```json
{
  "evidence_label": "development/frozen-rerun",
  "expected_successes": {"logiv": 63, "baseline": 56}
}
```

The compact JSON retains only totals, flips, and the ten per-task rows already
present in `paired-summary-final.json`. The Markdown labels them as the observed
2026-08-17 frozen rerun and notes that live reruns can vary.

- [ ] **Step 4: Run the focused test and verify GREEN**

Run: `uv run pytest -q tests/test_robotwin_63_manifest.py`

Expected: PASS.

---

### Task 2: Make both worker launchers portable

**Files:**
- Modify: `scripts/run_robotwin_logiv_10x10.py`
- Modify: `scripts/run_robotwin_baseline_10x10.py`
- Modify: `tests/test_robotwin_launcher.py`
- Create: `tests/test_robotwin_baseline_launcher.py`

**Interfaces:**
- Consumes: `--protocol PATH`, `--checkpoint PATH`, `--taco PATH`, `--python PATH`, `--tokenizer PATH`, and worker/GPU/output/tag arguments.
- Produces: task evaluator commands whose `--policy_path` is always the explicit CLI checkpoint, never a path embedded in the manifest.

- [ ] **Step 1: Add failing CLI/command tests**

```python
def test_logiv_worker_uses_explicit_checkpoint(tmp_path, monkeypatch) -> None:
    args = launcher_args(tmp_path, checkpoint=tmp_path / "checkpoint")
    command = LAUNCHER._task_command(load_manifest(), "turn_switch", args)
    assert command[command.index("--policy_path") + 1] == str(args.checkpoint)

def test_baseline_worker_uses_explicit_checkpoint(tmp_path) -> None:
    args = launcher_args(tmp_path, checkpoint=tmp_path / "checkpoint")
    command = BASELINE._task_command(load_manifest(), "turn_switch", args)
    assert command[command.index("--policy_path") + 1] == str(args.checkpoint)
```

- [ ] **Step 2: Run the launcher tests and verify RED**

Run: `uv run pytest -q tests/test_robotwin_launcher.py tests/test_robotwin_baseline_launcher.py`

Expected: FAIL because `_task_command` and explicit `--checkpoint` do not yet
exist.

- [ ] **Step 3: Extract command builders and add `--checkpoint`**

Both launchers define:

```python
def _task_command(config: dict, task: str, args: argparse.Namespace) -> list[str]:
    return [
        str(args.python),
        "script/eval_lerobot_torch_pi05.py",
        "--config", "policy/pi05/deploy_policy.yml",
        "--overrides",
        "--policy_name", "pi05",
        "--task_name", task,
        "--policy_path", str(args.checkpoint),
        # existing frozen options follow
    ]
```

Add `parser.add_argument("--checkpoint", type=Path, required=True)` and replace
the inline command lists with `_task_command`. Keep task allocation and LOGIV
control values unchanged.

- [ ] **Step 4: Run focused launcher tests and verify GREEN**

Run: `uv run pytest -q tests/test_robotwin_launcher.py tests/test_robotwin_baseline_launcher.py`

Expected: PASS.

---

### Task 3: Replace the audit/oracle reporter with a minimal paired outcome reporter

**Files:**
- Modify: `scripts/report_robotwin_logiv.py`
- Replace: `tests/test_report_robotwin_logiv.py`

**Interfaces:**
- Consumes: `build_report(config: dict[str, Any], events_root: Path, baseline_logs: Path | None) -> dict[str, Any]`.
- Produces: exact completeness, success totals, paired flips, per-task counts, and structural errors only.

- [ ] **Step 1: Write failing tests that reject selection and accept minimal records**

```python
def test_minimal_complete_report_needs_no_val_provenance_or_images(tmp_path) -> None:
    write_event(tmp_path, {"task": "turn_switch", "seed": 7,
                           "original_instruction": "press", "success": True})
    write_baseline_log(tmp_path / "baseline" / "turn_switch.log", seed=7,
                       success=False)
    report = REPORTER.build_report(one_cell_config(), tmp_path, tmp_path / "baseline")
    assert report["strict_protocol_complete"] is True
    assert report["successes"] == 1
    assert report["baseline_successes"] == 0
    assert report["errors"] == []

def test_report_rejects_duplicate_or_wrong_instruction(tmp_path) -> None:
    # two records for one key and one mismatched frozen instruction
    report = REPORTER.build_report(one_cell_config(), tmp_path, None)
    assert any("duplicate" in error for error in report["errors"])
    assert any("instruction mismatch" in error for error in report["errors"])
```

- [ ] **Step 2: Run the reporter tests and verify RED**

Run: `uv run pytest -q tests/test_report_robotwin_logiv.py`

Expected: FAIL because the old reporter requires events, VAL flags, GPT-4o
provenance, ternary facts, and image hashes.

- [ ] **Step 3: Remove all post-hoc evidence checks**

Keep `parse_baseline_log`, event JSONL loading, frozen key/instruction checks,
duplicate/unexpected/missing checks, paired totals, flips, per-task aggregation,
JSON CLI output, and Markdown rendering. Delete `_audit_file_matches`, hashes,
camera constants, ternary fact checks, `val_valid` checks, and GPT-4o call
checks. Parse malformed records into explicit errors instead of crashing.

- [ ] **Step 4: Run reporter and manifest tests and verify GREEN**

Run: `uv run pytest -q tests/test_report_robotwin_logiv.py tests/test_robotwin_63_manifest.py`

Expected: PASS.

---

### Task 4: Add the single-command driver with safe compaction

**Files:**
- Create: `scripts/reproduce_robotwin_logiv_63.py`
- Create: `tests/test_reproduce_robotwin_logiv_63.py`
- Create: `docs/robotwin-logiv-63-reproduction.md`

**Interfaces:**
- Consumes: the approved public CLI, environment `OPENAI_API_KEY`, portable manifest, two worker launchers, and minimal reporter.
- Produces: `_worker_commands(args, phase, tag) -> list[list[str]]`, `_expectations(report, manifest) -> list[str]`, and `_compact(output_root, eval_root, tags) -> None`.

- [ ] **Step 1: Write failing driver tests**

```python
def test_worker_commands_cover_three_distinct_gpus_and_both_phases(args) -> None:
    logiv = DRIVER._worker_commands(args, "logiv", "run-logiv")
    baseline = DRIVER._worker_commands(args, "baseline", "run-baseline")
    assert [command[command.index("--gpu") + 1] for command in logiv] == ["0", "1", "2"]
    assert all("run_robotwin_logiv_10x10.py" in command[1] for command in logiv)
    assert all("run_robotwin_baseline_10x10.py" in command[1] for command in baseline)

def test_expectations_report_live_mismatch() -> None:
    errors = DRIVER._expectations(
        {"strict_protocol_complete": True, "successes": 62,
         "baseline_successes": 56},
        {"expected_successes": {"logiv": 63, "baseline": 56}},
    )
    assert errors == ["expected LOGIV 63, observed 62"]

def test_compaction_refuses_paths_outside_current_tag(tmp_path) -> None:
    with pytest.raises(ValueError, match="outside eval root"):
        DRIVER._compact(tmp_path / "output", tmp_path / "eval", ["../other"])
```

- [ ] **Step 2: Run the driver tests and verify RED**

Run: `uv run pytest -q tests/test_reproduce_robotwin_logiv_63.py`

Expected: FAIL because the driver does not exist.

- [ ] **Step 3: Implement preflight and deterministic command construction**

Parse the approved CLI, default `--protocol` to the portable manifest, require
three distinct GPU integers, require every input path, validate an unused
output path, validate a non-secret API key, and derive safe tags
`<output-name>-logiv` and `<output-name>-baseline`. `--dry-run` prints JSON
commands with no environment values and creates nothing.

- [ ] **Step 4: Implement phase execution, reporting, expectation checking, and compaction**

Run three `subprocess.Popen` workers concurrently per phase and wait for all.
Run LOGIV before baseline. Invoke the reporter against exactly
`TACO/third_party/Robotwin/eval_result/<logiv-tag>` and the shared baseline log
directory. Write `manifest.json`, `run.json`, `summary.json`, and `summary.md`.
If reporting is structurally complete, remove `output/raw` and the two exact
tag directories with `shutil.rmtree`; then return nonzero on an expectation
mismatch. Keep raw data on process or structural failure.

- [ ] **Step 5: Document setup and the one public command**

The README lists Python/UV, TACO/RoboTwin, checkpoint, tokenizer, VAL,
`OPENAI_API_KEY`, the ordered runtime patches, the exact command, output files,
and the live-nondeterminism boundary. It contains no historical seed-selection
procedure.

- [ ] **Step 6: Run focused tests and dry-run verification**

Run:

```bash
uv run pytest -q tests/test_reproduce_robotwin_logiv_63.py \
  tests/test_robotwin_launcher.py tests/test_robotwin_baseline_launcher.py \
  tests/test_report_robotwin_logiv.py tests/test_robotwin_63_manifest.py
uv run python scripts/reproduce_robotwin_logiv_63.py --help
```

Expected: all tests PASS and help exits zero.

---

### Task 5: Delete the historical surface, verify, push, and remove explicit local runs

**Files:**
- Delete: `scripts/freeze_robotwin_logiv_seeds.py`
- Delete: `scripts/report_robotwin_logiv_oracle.py`
- Delete: obsolete seed/oracle tests and configs under `tests/` and `configs/robotwin/`
- Delete: old five imported RoboTwin design/plan documents
- Delete: audit/selection runtime patches `0006`, `0007`, `0008`, and `0009`
- Modify: `patches/robotwin/README.md`
- Delete: `results/robotwin-logiv-gpt4o-63-development-20260817/`
- Delete: the four older root-level RoboTwin oracle result files
- Restore to `origin/main`: three unrelated one-line files under `src/` and `tests/logiv/`

**Interfaces:**
- Consumes: the clean durable files from Tasks 1-4 and the explicit deletion list in the approved design.
- Produces: a branch tip containing no seed-selection/oracle/log/image artifacts and a second GitHub push.

- [ ] **Step 1: Mechanically delete only the approved tracked paths**

Use `git rm` with explicit file/directory arguments. Update the patch README to
list only the first 15 control/runtime patches and explain that patches 16-19
were evidence/selection-only and are intentionally excluded.

- [ ] **Step 2: Run final tree audits**

Run:

```bash
rg -n 'historical-success|seed_selection|report_robotwin_logiv_oracle|candidate-pool|vlm-camera-smoke' \
  configs scripts tests patches results docs/robotwin-logiv-63-reproduction.md
find results/robotwin-logiv-63-vs-pi05-56.* -type f
find results -type f \( -name '*.log' -o -name '*.png' \)
```

Expected: the first and third commands print nothing; the second prints only
the compact JSON and Markdown files.

- [ ] **Step 3: Run complete verification**

Run:

```bash
uv run pytest -q
git diff --check
rg -n --hidden 'sk-[A-Za-z0-9_-]{20,}' configs scripts tests patches results docs || true
git status --short
```

Expected: all tests PASS, no whitespace error, no real secret, and only the
approved cleanup/reproduction changes are present.

- [ ] **Step 4: Commit and push the completed clean version**

```bash
git add configs docs patches results scripts src tests
git commit -m "feat(robotwin): publish clean LOGIV 63 reproduction"
git push origin agent/robotwin-logiv-63-repro
```

- [ ] **Step 5: Delete only the seven approved local runtime directories**

Resolve each directory under
`.worktrees/taco-robotwin-logiv-63-gpt4o/third_party/Robotwin/eval_result`, prove
its parent is that exact `eval_result` directory, and remove the seven names
listed in the design. Re-run `du` and `git status` on the TACO checkout to prove
the run directories are gone and its pre-existing tracked/untracked code state
is unchanged.
