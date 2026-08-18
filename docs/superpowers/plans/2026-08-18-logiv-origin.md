# LOGIV Origin Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish a deployable LOGIV Origin tree with the full runtime method and no historical results, logs, or seed-selection machinery.

**Architecture:** Preserve the existing typed-fact, PDDL/VAL, DAG, monitoring, and bounded-repair runtime. Replace inherited experiment profiles with one resolved Origin profile, expose only deployment launchers, and enforce the release boundary with tests and ignore rules.

**Tech Stack:** Python 3.11+, pytest, shell, JSON/PDDL, OpenPI, LIBERO, RoboTwin/TACO, Docker, VAL, GitHub.

## Global Constraints

- The release name is **LOGIV Origin**.
- Git history remains intact; do not rewrite or force-push it.
- The current tree contains no historical experiments, benchmark claims, result archives, run logs, seed pools, or seed-selection tooling.
- A runtime `seed` may reproduce one simulator run, but it may not choose, rank, reject, scan, freeze, or substitute episodes.
- Machine-local checkpoints, validators, credentials, and benchmark checkouts are runtime inputs and are never committed.

---

### Task 1: Lock the Origin release contract

**Files:**
- Create: `tests/test_origin_release.py`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: the tracked Git tree and `configs/logiv/origin/`.
- Produces: release-hygiene tests that all later tasks must satisfy.

- [ ] **Step 1: Write failing release-hygiene tests**

```python
from pathlib import Path
import json
import subprocess

ROOT = Path(__file__).resolve().parents[1]

def tracked_files() -> tuple[str, ...]:
    output = subprocess.check_output(
        ["git", "ls-files"], cwd=ROOT, text=True
    )
    return tuple(output.splitlines())

def test_origin_tree_has_no_generated_records_or_seed_selection() -> None:
    files = tracked_files()
    forbidden_roots = ("results/", "evaluation_results/", "evaluation_results_dualsystem/")
    assert not any(path.startswith(forbidden_roots) for path in files)
    forbidden_names = ("accepted_seeds", "seed_scan", "seed_pool", "freeze_robotwin_logiv_seeds")
    deployment = "\n".join(
        (ROOT / path).read_text(encoding="utf-8", errors="ignore")
        for path in files
        if path.startswith(("scripts/", "configs/logiv/", "patches/robotwin/"))
    )
    assert not any(name in deployment for name in forbidden_names)

def test_origin_configs_are_resolved() -> None:
    paths = sorted((ROOT / "configs/logiv/origin").glob("*.json"))
    assert paths
    for path in paths:
        assert "extends" not in json.loads(path.read_text(encoding="utf-8"))
```

- [ ] **Step 2: Verify the tests fail against the research tree**

Run: `uv run pytest tests/test_origin_release.py -q`

Expected: failures identify tracked results, seed-selection code, and the missing resolved Origin profile.

- [ ] **Step 3: Expand ignore rules for runtime output**

Add exact rules for `results/`, `evaluation_results*/`, `*.log`, `*.jsonl`, videos, screenshots, checkpoints, datasets, tools, credentials, caches, and local benchmark checkouts. Do not ignore `configs/logiv/origin/*.json`.

- [ ] **Step 4: Commit the contract**

```bash
git add .gitignore tests/test_origin_release.py
git commit -m "test: define LOGIV Origin release contract"
```

### Task 2: Create one stable Origin configuration

**Files:**
- Create: `configs/logiv/origin/coverage.json`
- Create: `configs/logiv/origin/proposals.json`
- Create: `configs/logiv/origin/prompts.json`
- Create: `configs/logiv/origin/monitor-evidence.json`
- Create: `configs/logiv/origin/terminal-recovery.json`
- Create: `configs/logiv/origin/val-build.json`
- Create: `configs/logiv/origin/domain.pddl`
- Modify: `src/pi05_libero_repro/logiv/prompts.py`
- Modify: `src/pi05_libero_repro/logiv/proposal.py`
- Modify: `src/pi05_libero_repro/logiv/libero_adapter.py`
- Modify: `scripts/eval_logiv_libero.py`

**Interfaces:**
- Consumes: `load_extended_json(path: Path) -> dict` and the current online-v5 deployment profile.
- Produces: standalone JSON files with no `extends` key and stable runtime defaults under `configs/logiv/origin/`.

- [ ] **Step 1: Add default-path assertions to the release test**

```python
def test_origin_runtime_defaults_only_reference_origin_configs() -> None:
    sources = [
        ROOT / "src/pi05_libero_repro/logiv/prompts.py",
        ROOT / "src/pi05_libero_repro/logiv/proposal.py",
        ROOT / "src/pi05_libero_repro/logiv/libero_adapter.py",
        ROOT / "scripts/eval_logiv_libero.py",
    ]
    text = "\n".join(path.read_text(encoding="utf-8") for path in sources)
    assert "configs/logiv/origin/" in text
    assert "online-v" not in text
```

- [ ] **Step 2: Run the focused test and confirm failure**

Run: `uv run pytest tests/test_origin_release.py::test_origin_runtime_defaults_only_reference_origin_configs -q`

Expected: FAIL because the current defaults reference versioned experiment configs.

- [ ] **Step 3: Materialize resolved configuration**

Resolve the current deployment coverage, proposals, and prompt inheritance with `load_extended_json`; remove the top-level `extends` key; preserve values exactly. Copy the PDDL domain, monitor evidence, terminal recovery, and VAL build manifests under stable Origin names.

- [ ] **Step 4: Point runtime defaults to Origin paths**

Replace versioned paths with:

```python
Path("configs/logiv/origin/coverage.json")
Path("configs/logiv/origin/proposals.json")
Path("configs/logiv/origin/prompts.json")
Path("configs/logiv/origin/domain.pddl")
```

- [ ] **Step 5: Verify configuration and relevant LOGIV tests**

Run: `uv run pytest tests/test_origin_release.py tests/logiv/test_proposal.py tests/logiv/test_libero_adapter.py -q`

Expected: PASS.

- [ ] **Step 6: Commit stable configuration**

```bash
git add configs/logiv/origin src/pi05_libero_repro/logiv scripts/eval_logiv_libero.py tests/test_origin_release.py
git commit -m "feat: add stable LOGIV Origin configuration"
```

### Task 3: Reduce the repository to the deployable LOGIV surface

**Files:**
- Delete: `results/`, `evaluation_results/`, `evaluation_results_dualsystem/`
- Delete: unrelated top-level research assets and historical documents
- Delete: historical `configs/` entries outside `configs/logiv/origin/`
- Delete: experiment/report/seed-selection scripts outside the retained deployment scripts
- Delete: report, seed-scan, recovery-split, and historical-config tests
- Delete: `src/pi05_libero_repro/logiv/recovery_splits.py`
- Keep: `src/pi05_libero_repro/logiv/`, `src/pi05_libero_repro/protocol.py`, the retained tests, `docker/Dockerfile.libero`, `external_repos/openpi`, and package metadata

**Interfaces:**
- Consumes: the release contract from Task 1 and resolved paths from Task 2.
- Produces: a small tracked tree containing only LOGIV runtime, deployment assets, docs, and tests.

- [ ] **Step 1: Remove generated and unrelated tracked trees with explicit targets**

Use `git rm -r -- <exact target list>` after checking each target with `git ls-files`. Never delete local untracked checkpoints, tools, or virtual environments.

- [ ] **Step 2: Remove experiment-only source and tests**

Delete report builders, artifact/report libraries, recovery dataset splitting, seed freezer/scanner/selected-run launchers, ablation launchers, and their tests. Retain runtime recovery records because they are imported by the deployed monitor and adapter.

- [ ] **Step 3: Confirm import closure**

Run:

```bash
uv run python -c 'import pi05_libero_repro.logiv; import pi05_libero_repro.logiv.robotwin'
uv run pytest tests/logiv tests/test_protocol.py tests/test_origin_release.py -q
```

Expected: imports succeed and all retained tests pass.

- [ ] **Step 4: Commit the tree reduction**

```bash
git add -A
git commit -m "refactor: reduce repository to LOGIV Origin"
```

### Task 4: Publish deployment entry points without result logging or seed selection

**Files:**
- Rename: `scripts/serve_episode_seeded_policy.py` to `scripts/serve_policy.py`
- Modify: `scripts/run_policy_server.sh`
- Modify: `scripts/run_logiv_eval.sh`
- Create: `scripts/run_robotwin_logiv.py`
- Create: `patches/robotwin/logiv-origin.patch`
- Create: `patches/robotwin/README.md`
- Test: `tests/test_deployment_entrypoints.py`

**Interfaces:**
- Consumes: runtime checkpoints, ports, benchmark checkouts, VAL path, GPT-4o credentials, task names, episode count, and optional simulator RNG seed.
- Produces: stdout/stderr-only deployment commands with no output-directory requirement and no accepted-seed interface.

- [ ] **Step 1: Write launcher tests**

```python
def test_deployment_sources_have_no_seed_selector_or_log_sink() -> None:
    paths = [ROOT / "scripts/run_robotwin_logiv.py", ROOT / "scripts/run_policy_server.sh"]
    text = "\n".join(path.read_text(encoding="utf-8") for path in paths)
    for forbidden in ("accepted_seeds", "candidate_seed", "seed_pool", ".log", "log_dir"):
        assert forbidden not in text
```

- [ ] **Step 2: Confirm the new launcher test fails**

Run: `uv run pytest tests/test_deployment_entrypoints.py -q`

Expected: FAIL because the Origin launcher files do not exist.

- [ ] **Step 3: Simplify LIBERO launchers**

Make `run_policy_server.sh` accept `GPU PORT CHECKPOINT_DIR`, validate required checkpoint files, and stream server output. Restrict `run_logiv_eval.sh` to the deployable LOGIV mode, accept a caller-provided runtime output directory only as an external mount, and point to Origin config defaults.

- [ ] **Step 4: Build the generic RoboTwin launcher**

Expose `--task`, `--episodes`, `--seed`, `--taco`, `--checkpoint`, `--tokenizer`, and `--val-binary`. Pass the simulator seed directly to TACO and increment it naturally for requested episodes. Do not accept arrays, pools, allowlists, rankings, or reachability filters.

- [ ] **Step 5: Consolidate the RoboTwin integration**

Regenerate a single clean patch from the TACO base containing LOGIV PDDL/DAG execution, monitoring, CFN verification, local repair, GPT-4o grounding, and multi-camera gate evidence. Remove accepted/frozen-seed replay, candidate pool discovery, unreachable-seed auditing, selected-run provenance, and baseline experiment parameterization. Verify the patch applies with `git apply --check` to a clean TACO base.

- [ ] **Step 6: Verify launchers**

Run:

```bash
bash -n scripts/run_policy_server.sh scripts/run_logiv_eval.sh
uv run python scripts/eval_logiv_libero.py --help >/dev/null
uv run python scripts/run_robotwin_logiv.py --help >/dev/null
uv run pytest tests/test_deployment_entrypoints.py tests/test_origin_release.py -q
```

Expected: all commands exit zero.

- [ ] **Step 7: Commit deployment surface**

```bash
git add scripts patches/robotwin tests
git commit -m "feat: publish LOGIV Origin deployment entrypoints"
```

### Task 5: Rewrite package metadata and public documentation

**Files:**
- Modify: `pyproject.toml`
- Modify: `README.md`
- Create: `docs/METHOD.md`
- Create: `docs/DEPLOYMENT.md`
- Delete: `docs/superpowers/`

**Interfaces:**
- Consumes: the final Origin tree and deployment commands.
- Produces: the public LOGIV Origin package identity and reproducible deployment instructions without benchmark results.

- [ ] **Step 1: Rename the distribution**

Set `project.name = "logiv-origin"`, preserve the Python import package, and keep only required runtime plus pytest development dependencies.

- [ ] **Step 2: Rewrite README and docs**

Document the method loop, tree layout, setup, LIBERO deployment, RoboTwin patch/application, environment variables, validation, runtime seed semantics, and output hygiene. Include no result table, accuracy claim, selected seed, or historical experiment narrative.

- [ ] **Step 3: Remove implementation records from the release tree**

Delete `docs/superpowers/` after the implementation is complete so the current public tree contains no development log or plan archive.

- [ ] **Step 4: Verify documentation and package identity**

Run:

```bash
uv lock --check
uv build
uv run pytest tests/test_origin_release.py -q
```

Expected: lock/build succeed and the release contract passes.

- [ ] **Step 5: Commit documentation**

```bash
git add -A
git commit -m "docs: publish LOGIV Origin"
```

### Task 6: Final verification and GitHub integration

**Files:**
- Verify: all tracked release files

**Interfaces:**
- Consumes: completed Origin branch.
- Produces: a merged remote `main` and verified commit identity.

- [ ] **Step 1: Run the complete retained suite**

Run:

```bash
git diff --check origin/main...HEAD
uv run pytest -q
uv build
bash -n scripts/*.sh
git status --short
```

Expected: no whitespace errors, all tests pass, package builds, shell syntax passes, and only intentionally ignored local assets remain.

- [ ] **Step 2: Inspect tracked residue**

Run:

```bash
git ls-files | rg '(^|/)(results?|logs?|evaluation_results)(/|$)|seed_(scan|pool)|accepted_seeds|freeze_robotwin'
```

Expected: no output.

- [ ] **Step 3: Commit any final corrections and push**

```bash
git push -u origin agent/logiv-origin
```

- [ ] **Step 4: Open and merge the PR**

Create a PR titled `origin`, ensure it targets `main`, inspect checks, merge it, and delete the remote feature branch if supported.

- [ ] **Step 5: Verify remote main**

Fetch the remote and confirm `origin/main` is the merged commit and its tree passes the residue scan.
