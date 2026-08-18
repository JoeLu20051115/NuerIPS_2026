# LOGIV_Experiment Oracle Compatibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore historical `UNKNOWN` oracle-record compatibility without changing current `UNRESOLVED` serialization, then merge `LOGIV_Experiment` into `main`.

**Architecture:** Normalize the one legacy string at the historical oracle ingestion boundary and continue to let `TruthValue` validate every normalized value. Keep the existing historical-record test as the regression test, then merge the already-open PR after all relevant tests pass.

**Tech Stack:** Python 3.12, pytest, Git, GitHub pull requests

## Global Constraints

- Change only the historical oracle ingestion behavior.
- Do not change experiment artifacts, accuracy values, seed selections, or current truth-value serialization.
- Add no dependencies or abstractions.
- Use `LOGIV_Experiment` as the pull-request name.

---

### Task 1: Accept legacy oracle truth values

**Files:**
- Modify: `scripts/report_robotwin_logiv_oracle.py:232-240`
- Test: `tests/test_report_robotwin_logiv_oracle.py:137-188`

**Interfaces:**
- Consumes: embedded event `facts` mappings whose values include historical `UNKNOWN` or current `TRUE`, `FALSE`, and `UNRESOLVED` strings.
- Produces: the existing `dict[str, str]` passed to `TruthValue`, with only `UNKNOWN` normalized to `UNRESOLVED`.

- [ ] **Step 1: Verify the existing regression test is red**

Run:

```bash
uv run pytest -q tests/test_report_robotwin_logiv_oracle.py::test_embedded_events_can_be_replanned_and_revalidated_with_real_val
```

Expected: FAIL with `ValueError: 'UNKNOWN' is not a valid TruthValue`.

- [ ] **Step 2: Implement the minimum boundary normalization**

Replace the `facts` comprehension with:

```python
facts = {
    str(name): "UNRESOLVED" if str(value) == "UNKNOWN" else str(value)
    for name, value in raw_facts.items()
}
```

- [ ] **Step 3: Verify the oracle regression file is green**

Run:

```bash
uv run pytest -q tests/test_report_robotwin_logiv_oracle.py
```

Expected: `7 passed`.

- [ ] **Step 4: Verify all relevant RoboTwin reporting tests**

Run:

```bash
uv run pytest -q tests/test_freeze_robotwin_logiv_seeds.py tests/test_report_robotwin_logiv.py tests/test_report_robotwin_logiv_oracle.py tests/test_robotwin_seed_scan_config.py tests/test_robotwin_launcher.py
```

Expected: `30 passed`.

- [ ] **Step 5: Commit the compatibility fix**

```bash
git add scripts/report_robotwin_logiv_oracle.py
git commit -m "fix(robotwin): accept legacy oracle truth values"
```

### Task 2: Publish and merge LOGIV_Experiment

**Files:**
- No repository file changes.

**Interfaces:**
- Consumes: branch `agent/robotwin-logiv-63-gpt4o` and pull request `#2`.
- Produces: remote `main` containing the result commit and compatibility fix.

- [ ] **Step 1: Push the compatibility commit**

```bash
git push origin agent/robotwin-logiv-63-gpt4o
```

Expected: the remote branch advances to the local `HEAD`.

- [ ] **Step 2: Rename and merge the pull request**

Update PR #2 title to `LOGIV_Experiment`, mark it ready if required, and merge it through GitHub after checks pass.

- [ ] **Step 3: Verify remote main contains the merged commits**

```bash
git fetch origin main
git merge-base --is-ancestor 65bd3401e8d3ee488fdb05067cd9629f54884598 origin/main
git merge-base --is-ancestor HEAD origin/main
```

Expected: both ancestry checks exit successfully.
