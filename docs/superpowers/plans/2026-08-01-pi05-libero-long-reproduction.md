# π₀.₅ LIBERO-Long Reproduction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and execute an auditable OpenPI/LIBERO pipeline that evaluates the full and 2,000-step π₀.₅ checkpoints for 50 trials on each of the ten LIBERO-10 tasks and reproduces the public aggregate results within the approved tolerances.

**Architecture:** Keep the official OpenPI repository and pinned LIBERO gitlink unchanged. Add a small repository-local Python package for artifact manifests, faithful evaluator instrumentation, episode records, reports, and architecture snapshots; run two uninterrupted official-order model/evaluator chains concurrently on GPU 1 and GPU 2, with GPU 0 reserved for later audits.

**Tech Stack:** Python 3.11 for repository tooling and tests, Python 3.8 in the official LIBERO Docker evaluator, JAX/OpenPI, MuJoCo 3.2.3, robosuite 1.4.1, LIBERO/BDDL, pytest, Docker with NVIDIA Container Toolkit, JSON/JSONL, and Markdown.

## Global Constraints

- OpenPI is locked to `650c5b0283a49c42784fb5055a0507da2c6d347d` and must remain clean.
- LIBERO is locked to `f78abd68ee283de9f9be3c8f7e2a9ad60246e95c`.
- Full checkpoint source is `gs://openpi-assets/checkpoints/pi05_libero/`; early revision is `aaeeabc72f8a50a8fa2d04544332c8ec1cd0142e`.
- Dataset revision is `f13aa24a3da8c43c7225569f28c562979fa0e35a`; download only `libero_10/*.hdf5`.
- Each primary chain runs tasks and episodes sequentially with seed 7 and an uninterrupted JAX RNG stream.
- Preserve official image rotation, image sizes, state conversion, dummy steps, five-step replanning, 520-step horizon, and native `done` semantics.
- Infrastructure exceptions abort as invalid; they are never counted as policy failures.
- Never terminate another user's process, alter BDDL predicates, select favorable seeds, skip valid failures, or extend the horizon.
- Acceptance requires exactly 500 valid records per checkpoint, full success in `[89.4%, 95.4%]`, and early success in `[38%, 48%]`.
- Use red-green-refactor and commit after each independently testable task.

---

## Planned File Structure

- `configs/artifacts.json`: immutable revisions, file lists, include rules, and byte totals.
- `src/pi05_libero_repro/artifacts.py`: SHA-256 manifests.
- `src/pi05_libero_repro/records.py`: durable episode JSONL and Wilson intervals.
- `src/pi05_libero_repro/protocol.py`: faithful official observation/control loop.
- `src/pi05_libero_repro/report.py`: mapping, aggregation, acceptance, JSON/Markdown.
- `scripts/eval_libero.py`: real LIBERO suite integration.
- `scripts/verify_artifacts.py`: manifest CLI.
- `scripts/snapshot_architecture.py`: effective config and parameter-tree snapshot.
- `scripts/report_results.py`: independent audit CLI.
- `scripts/download_public_assets.sh`: pinned inference-only downloads.
- `scripts/run_policy_server.sh`, `scripts/run_libero_eval.sh`: reproducible launchers.
- `tests/test_artifacts.py`, `tests/test_records.py`, `tests/test_protocol.py`, `tests/test_report.py`: unit and invariant tests.
- `artifacts/manifests/`: committed small provenance; large payloads live in ignored directories.

---

### Task 1: Clean Legacy Workspace and Establish the Focused Project

**Files:**
- Create: `docs/cleanup/2026-08-01-pre-cleanup.txt`
- Replace: `.gitignore`, `.gitmodules`, `README.md`, `pyproject.toml`
- Create: `src/pi05_libero_repro/__init__.py`
- Delete: the exact authorized legacy paths below

**Interfaces:**
- Consumes: approved design and current workspace inventory.
- Produces: importable `pi05_libero_repro` skeleton and clean root.

- [ ] **Step 1: Record cleanup evidence and active-process check**

Run `git status --short --untracked-files=all`, `du -sh` for every target, `df -h .`, and `ps -eo user,pid,etimes,cmd` filtered for this workspace; save output with ISO timestamp and `pwd` to `docs/cleanup/2026-08-01-pre-cleanup.txt`.

- [ ] **Step 2: Delete only resolved authorized targets**

Targets are `.claude`, `.venv`, `.venv_agibot_eval`, `.vscode`, `__pycache__`, `check_act_aloha_baseline.py`, `groot`, old `scripts`, `socket_test_optimized_AR.py`, `table30v2_multitask_baseline_aloha_remote_files.json`, `table30v2_multitask_baseline_aloha_repo`, `vector_graph_white.pdf`, and `vector_graph_white.svg`. For each, require `realpath --no-symlinks` to start with the repository root; delete directories with `find "$target" -depth -delete` and files with `unlink -- "$target"`. Preserve `.git`, `docs`, and `external_repos/openpi`.

- [ ] **Step 3: Write the minimal project skeleton**

Use this `pyproject.toml`:

```toml
[project]
name = "pi05-libero-repro"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = []

[dependency-groups]
dev = ["pytest>=8.3,<9"]

[tool.pytest.ini_options]
addopts = "-ra"
pythonpath = ["src"]
testpaths = ["tests"]
```

Set `src/pi05_libero_repro/__init__.py` to a module docstring. Ignore `.venv/`, Python caches, `artifacts/checkpoints/`, `artifacts/datasets/`, `runs/`, and the OpenPI venv; retain `artifacts/manifests/`.

Replace `.gitmodules` with:

```ini
[submodule "external_repos/openpi"]
    path = external_repos/openpi
    url = https://github.com/Physical-Intelligence/openpi.git
```

Stage the existing clean nested repository as the corresponding gitlink so a recursive clone recovers the model architecture source.

- [ ] **Step 4: Verify and commit the clean baseline**

Run `uv sync --dev`, import the package with `uv run python`, then `git add -A && git commit -m "chore: replace legacy workspace with pi05 LIBERO repro"`. Verify large payloads and `external_repos/openpi/.venv` are not staged.

---

### Task 2: Add Deterministic Artifact Manifests

**Files:**
- Create: `src/pi05_libero_repro/artifacts.py`
- Create: `scripts/verify_artifacts.py`
- Create: `tests/test_artifacts.py`
- Create: `configs/artifacts.json`

**Interfaces:**
- Produces: `build_manifest(root: Path) -> list[ManifestEntry]`, `write_manifest(root: Path, output: Path) -> None`, and `verify_manifest(root: Path, manifest_path: Path) -> list[str]`.
- Entry fields are exactly `path`, `size`, and `sha256`, sorted by POSIX path.

- [ ] **Step 1: Write the failing tests**

```python
def test_manifest_detects_tamper(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    (root / "b").write_bytes(b"bb")
    (root / "a").write_bytes(b"a")
    output = tmp_path / "manifest.json"
    write_manifest(root, output)
    assert [x.path for x in build_manifest(root)] == ["a", "b"]
    assert verify_manifest(root, output) == []
    (root / "a").write_bytes(b"changed")
    assert verify_manifest(root, output) == ["hash/size mismatch: a"]

def test_manifest_detects_missing_and_extra(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    (root / "expected").write_bytes(b"x")
    output = tmp_path / "manifest.json"
    write_manifest(root, output)
    (root / "expected").unlink()
    (root / "extra").write_bytes(b"y")
    assert verify_manifest(root, output) == ["missing: expected", "unexpected: extra"]
```

- [ ] **Step 2: Confirm red, then implement minimal manifest logic**

Run `uv run pytest tests/test_artifacts.py -v` and require import failure. Implement a frozen dataclass, 8 MiB streaming SHA-256, stable `{"version": 1, "files": [...]}` JSON, and sorted missing/mismatch/unexpected errors. The CLI accepts `create ROOT OUTPUT` and `verify ROOT MANIFEST`, printing errors to stderr and exiting 1 on failure.

- [ ] **Step 3: Add exact artifact configuration**

Record all Global Constraint revisions, full cache path, early include patterns `_CHECKPOINT_METADATA`, `assets/**`, `params/**`, expected early bytes `12440616902`, dataset bytes `13730608904`, and these exact HDF5 payloads (path, bytes, SHA-256 LFS object ID):

```text
libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo.hdf5 1319613988 6b30906a52a5741e98ef447d27e7066d6c0be4a5f7acd7ecaf1cb7468aca4aa9
libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it_demo.hdf5 1235470452 703950f48a3c49dfde61be489ade91527f16e1449b4f29a85f2e51153cef3638
libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it_demo.hdf5 1511189856 1cef1974f76176214dfd1d983a1961507c2b23d54be9dbf9abb2f243f3efacef
libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove_demo.hdf5 2061429892 e329bb21a8ded3457854faf6a23513c90cb4b34f0e40f3f4e9e70451fc9ba504
libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket_demo.hdf5 1341166952 09fdc7cc0f546ab4b5232907c424670d08791476c69c1ee453b9ea4b70111cff
libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket_demo.hdf5 1467540242 3173ebc709b064683956b757ae8737909457d730aa305b23c5735292e59066af
libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket_demo.hdf5 1300335262 ce03c7be607cdc3d55f634998eb15beee770006b053fbcd9cbfb207c8b72847a
libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate_demo.hdf5 1285266594 a2a75bc6c301a914d0998ccac9dcf1d70d0f501897fbb6baf86aa61315b57b31
libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate_demo.hdf5 1268066750 75988131f8f443d68108bbfaacc22dab229ece93f141362524e69a2c49a23516
libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy_demo.hdf5 940528916 2e0127d0cf73afceacc2c0854e505d5743b6d3d5aaf5572c842f84c6a1c8173b
```

- [ ] **Step 4: Verify green and commit**

Run the focused test, create/verify a config manifest, then commit with `feat: add reproducible artifact manifests`.

---

### Task 3: Pin Sources and Download Public Artifacts

**Files:**
- Create: `scripts/download_public_assets.sh`
- Create: `artifacts/manifests/openpi-source.json`, `full-checkpoint.json`, `early-checkpoint.json`, `libero10-dataset.json`

**Interfaces:**
- Produces: verified `artifacts/checkpoints/pi05_libero_2000`, `artifacts/datasets/libero_10`, and manifests required by launchers.

- [ ] **Step 1: Initialize the locked LIBERO gitlink**

Assert the OpenPI HEAD, run `git -C external_repos/openpi submodule update --init --depth 1 third_party/libero`, assert the LIBERO HEAD, and require a clean nested repository.

- [ ] **Step 2: Write and run the pinned download script**

Use `set -euo pipefail` and these commands:

```bash
uvx --from huggingface-hub hf download brandonyang/openpi-libero-2000 \
  --revision aaeeabc72f8a50a8fa2d04544332c8ec1cd0142e \
  --include _CHECKPOINT_METADATA 'assets/**' 'params/**' \
  --local-dir artifacts/checkpoints/pi05_libero_2000
uvx --from huggingface-hub hf download yifengzhu-hf/LIBERO-datasets \
  --repo-type dataset --revision f13aa24a3da8c43c7225569f28c562979fa0e35a \
  --include 'libero_10/*.hdf5' --local-dir artifacts/datasets
```

Remove only Hugging Face's generated local-dir metadata directory `.cache/huggingface` after the CLI exits. Refuse success unless inference payloads total `12440616902` bytes, dataset payloads total `13730608904` bytes, no `train_state/` exists, and exactly the configured ten HDF5 files exist.

- [ ] **Step 3: Generate, verify, and commit manifests**

Manifest OpenPI tracked blobs and the three payload trees; verify byte totals and require full/early norm-stat SHA-256 values to differ. Commit only the script and small manifests with `chore: pin public pi05 and LIBERO artifacts`.

---

### Task 4: Add Durable Episode Records and Statistics

**Files:**
- Create: `src/pi05_libero_repro/records.py`
- Create: `tests/test_records.py`

**Interfaces:**
- Produces: `EpisodeRecord`, `append_record`, `load_records`, `validate_records`, and `wilson_interval`.
- Unique key is `(checkpoint, task_id, episode_idx)`.

- [ ] **Step 1: Write failing record tests**

```python
def make_record(episode_idx=0, success=True, checkpoint="full", task_id=0):
    return EpisodeRecord(
        checkpoint=checkpoint, task_id=task_id, task_name="task", episode_idx=episode_idx,
        init_state_sha256="0" * 64, seed=7, success=success, valid=True,
        steps=12, inference_requests=3, wall_seconds=1.5, exception=None,
        first_frame_sha256="1" * 64, action_min=-0.5, action_max=0.5,
        action_mean=0.0, done=success, check_success=success, video_path="video.mp4",
    )

def test_round_trip_and_duplicate(tmp_path):
    path = tmp_path / "episodes.jsonl"
    append_record(path, make_record())
    assert load_records(path) == [make_record()]
    with pytest.raises(ValueError, match="duplicate episode"):
        append_record(path, make_record())

def test_validation_and_wilson():
    records = [make_record(i, i < 46) for i in range(50)]
    assert validate_records(records, expected_trials=50) == []
    low, high = wilson_interval(46, 50)
    assert 0.81 < low < 0.82 and 0.96 < high < 0.97
```

- [ ] **Step 2: Confirm red and implement durable JSONL**

Require import failure first. Implement the frozen dataclass with exactly the fields above; serialize sorted compact JSON; load existing keys before append; write one line, flush, and `os.fsync`. Validation rejects duplicates, invalid records, predicate disagreement, exceptions, malformed hashes, non-finite actions, and incorrect counts. Implement Wilson score with `z=1.959963984540054` and no statistics dependency.

- [ ] **Step 3: Verify green and commit**

Run `uv run pytest tests/test_records.py -v` and commit with `feat: add durable LIBERO episode records`.

---

### Task 5: Implement and Test the Official Control Protocol

**Files:**
- Create: `src/pi05_libero_repro/protocol.py`
- Create: `tests/test_protocol.py`

**Interfaces:**
- Produces: `quat2axisangle`, `prepare_observation(obs, prompt, image_tools) -> tuple[dict, np.ndarray]`, and `run_episode(env, client, initial_state, prompt, image_tools, max_steps=520, wait_steps=10, replan_steps=5)`; `run_episode` performs `reset()` then `set_init_state()`.
- Returns `EpisodeOutcome(success, done, check_success, steps, inference_requests, first_frame, replay_frames, actions)`; raises `EpisodeInvalid` on infrastructure/invariant errors.

- [ ] **Step 1: Write failing preprocessing tests**

```python
class FakeImageTools:
    seen = []
    @classmethod
    def resize_with_pad(cls, image, height, width):
        cls.seen.append(image.copy())
        return image
    @staticmethod
    def convert_to_uint8(image):
        return image.astype(np.uint8)

def test_prepare_observation_rotation_and_state_order():
    image = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
    obs = {
        "agentview_image": image,
        "robot0_eye_in_hand_image": image + 20,
        "robot0_eef_pos": np.array([1.0, 2.0, 3.0]),
        "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0]),
        "robot0_gripper_qpos": np.array([4.0, 5.0]),
    }
    element, main_image = prepare_observation(obs, "prompt", FakeImageTools)
    np.testing.assert_array_equal(FakeImageTools.seen[0], image[::-1, ::-1])
    np.testing.assert_array_equal(main_image, image[::-1, ::-1])
    np.testing.assert_allclose(element["observation/state"], [1, 2, 3, 0, 0, 0, 4, 5])
    assert element["observation/state"].shape == (8,)
```

- [ ] **Step 2: Write failing control-loop tests**

```python
class FakeClient:
    def __init__(self): self.calls = 0
    def infer(self, element):
        self.calls += 1
        return {"actions": np.full((10, 7), self.calls, dtype=np.float32)}

def test_wait_replan_and_success(fake_env, initial_state):
    fake_env.succeed_on_policy_step = 7
    client = FakeClient()
    out = run_episode(fake_env, client, initial_state, "prompt", FakeImageTools)
    assert fake_env.actions[:10] == [[0.0] * 6 + [-1.0]] * 10
    assert client.calls == out.inference_requests == 2
    assert [a[0] for a in out.actions] == [1, 1, 1, 1, 1, 2, 2]
    assert out.success and out.done and out.check_success

def test_inference_error_is_invalid(fake_env, initial_state):
    class BrokenClient:
        def infer(self, element): raise ConnectionError("closed")
    with pytest.raises(EpisodeInvalid, match="closed"):
        run_episode(fake_env, BrokenClient(), initial_state, "prompt", FakeImageTools)

def test_timeout_is_valid_failure(fake_env, initial_state):
    out = run_episode(fake_env, FakeClient(), initial_state, "prompt", FakeImageTools, max_steps=520)
    assert not out.success and not out.done and not out.check_success
    assert out.steps == 520
```

- [ ] **Step 3: Confirm red and implement the faithful loop**

Require module import failure. Copy the pinned official quaternion and loop ordering. Use `np.ascontiguousarray(image[::-1, ::-1])`, pad/resize to 224, construct the exact input keys, execute `action[:7].tolist()`, and require finite chunks shaped at least `(5, 7)`. Preserve the first frame, every policy action, request count, and final predicate comparison.

- [ ] **Step 4: Verify semantics and commit**

Run `uv run pytest tests/test_protocol.py -v` and review a diff against `external_repos/openpi/examples/libero/main.py`; every semantic difference must be instrumentation, injection, validation, or exception handling from the design. Commit with `feat: preserve official OpenPI LIBERO protocol`.

---

### Task 6: Integrate the Real LIBERO Evaluator

**Files:**
- Create: `scripts/eval_libero.py`
- Modify: `tests/test_protocol.py`

**Interfaces:**
- Consumes: protocol functions, `EpisodeRecord`, LIBERO, and openpi-client.
- Produces: one unique video and durable record per valid episode; exits nonzero immediately on invalid episodes.

- [ ] **Step 1: Add failing iteration and filename tests**

```python
def test_pending_indices_and_rng_safe_prefix():
    assert pending_episode_indices([], "full", 0, 50) == list(range(50))
    complete = [make_record(i) for i in range(50)]
    assert pending_episode_indices(complete, "full", 0, 50) == []
    with pytest.raises(ValueError, match="resume would change policy RNG sequence"):
        pending_episode_indices([make_record(1)], "full", 0, 50)

def test_unique_video_name():
    assert video_name(0, 0, True) == "task_00_episode_00_success.mp4"
    assert video_name(0, 0, False) == "task_00_episode_00_failure.mp4"
```

- [ ] **Step 2: Confirm red and implement the entry point**

Use Python 3.8-compatible `argparse` with required `--checkpoint-name`, `--port`, `--output-dir`; defaults are host `127.0.0.1`, seed 7, trials 50, suite `libero_10`. Create `OffScreenRenderEnv` at 256×256, seed once per task, call `reset()` before `set_init_state`, and process official task/episode order. Write videos to temporary paths and atomically rename before appending records. On `EpisodeInvalid`, write `invalid.json` with traceback and version/request metadata, then exit 2. Refuse partial primary resume unless `--diagnostic-resume` is explicit.

- [ ] **Step 3: Verify tests and Python 3.8 syntax**

Run:

```bash
uv run pytest tests/test_protocol.py tests/test_records.py -v
docker run --rm -v "$PWD:/repro:ro" python:3.8-slim \
  python -m py_compile /repro/scripts/eval_libero.py \
  /repro/src/pi05_libero_repro/protocol.py /repro/src/pi05_libero_repro/records.py
```

Expected: tests pass and all evaluator-side files compile on Python 3.8.

- [ ] **Step 4: Commit**

Commit with `feat: add auditable LIBERO evaluator`.

---

### Task 7: Add Independent Reporting and Acceptance Audit

**Files:**
- Create: `src/pi05_libero_repro/report.py`
- Create: `scripts/report_results.py`
- Create: `tests/test_report.py`

**Interfaces:**
- Produces: `build_report(records) -> dict`, `render_markdown(report) -> str`, and CLI `summary.json`/`summary.md`.

- [ ] **Step 1: Write failing report tests**

```python
def records_for(checkpoint, successes):
    records = []
    for task_id in range(10):
        for episode_idx in range(50):
            index = task_id * 50 + episode_idx
            records.append(make_record(
                checkpoint=checkpoint, task_id=task_id, episode_idx=episode_idx,
                success=index < successes,
            ))
    return records

def test_public_rates_and_display_order():
    report = build_report(records_for("full", 462) + records_for("early", 215))
    assert report["checkpoints"]["full"]["rate"] == 0.924
    assert report["checkpoints"]["early"]["rate"] == 0.43
    assert report["accepted"] is True
    assert all(row["trials"] == 50 for row in report["tasks"])
    markdown = render_markdown(report)
    assert markdown.index("Cream cheese + Butter") < markdown.index("Black bowl")
    assert markdown.rindex("Chocolate pudding") > markdown.index("Book")

@pytest.mark.parametrize("checkpoint,successes", [("full", 440), ("early", 245)])
def test_out_of_range_is_rejected(checkpoint, successes):
    other = records_for("early" if checkpoint == "full" else "full", 215 if checkpoint == "full" else 462)
    report = build_report(records_for(checkpoint, successes) + other)
    assert not report["accepted"]
    assert any(error["code"] == "rate_out_of_range" for error in report["errors"])
```

```python
def test_cardinality_duplicate_invalid_and_predicate_errors():
    base = records_for("full", 462) + records_for("early", 215)
    cases = [
        (base[:-1], "count_mismatch"),
        (base + [base[0]], "duplicate_episode"),
        ([dataclasses.replace(base[0], valid=False)] + base[1:], "invalid_episode"),
        ([dataclasses.replace(base[0], check_success=False)] + base[1:], "predicate_mismatch"),
    ]
    for records, expected_code in cases:
        report = build_report(records)
        assert not report["accepted"]
        assert any(error["code"] == expected_code for error in report["errors"])
```

- [ ] **Step 2: Confirm red and implement reporting**

Hard-code the ten official task identifiers and a separate user display order. Compute successes, trials, rates, and Wilson intervals. Overall acceptance requires exact cardinality, uniqueness, validity, predicate agreement, and the approved checkpoint interval. JSON stores `[0, 1]` rates; Markdown displays successes and one-decimal percentages.

- [ ] **Step 3: Verify green and commit**

Run `uv run pytest tests/test_report.py -v` then the full suite. Commit with `feat: audit pi05 LIBERO results`.

---

### Task 8: Snapshot Architecture and Add Reproducible Launchers

**Files:**
- Create: `scripts/snapshot_architecture.py`
- Create: `scripts/run_policy_server.sh`, `scripts/run_libero_eval.sh`
- Create: `artifacts/manifests/pi05-libero-architecture.json`
- Modify: `README.md`

**Interfaces:**
- Server launcher: `run_policy_server.sh CHECKPOINT_NAME GPU PORT CHECKPOINT_DIR LOG_DIR`.
- Evaluator launcher: `run_libero_eval.sh CHECKPOINT_NAME GPU PORT OUTPUT_DIR`.
- Architecture JSON contains config, revisions, checkpoint-manifest hash, parameter leaf count/elements, and every path/shape/dtype.

- [ ] **Step 1: Establish failing launcher checks**

Run `bash -n scripts/run_policy_server.sh scripts/run_libero_eval.sh`; require failure because files do not exist.

- [ ] **Step 2: Implement the policy launcher**

Use `set -euo pipefail`, require five arguments, verify `params/`, checkpoint-local norm stats, and OpenPI HEAD, then `exec` from OpenPI with:

```bash
CUDA_VISIBLE_DEVICES="$gpu" XLA_PYTHON_CLIENT_MEM_FRACTION=0.70 \
  uv run scripts/serve_policy.py --port "$port" \
  policy:checkpoint --policy.config pi05_libero --policy.dir "$checkpoint_dir"
```

- [ ] **Step 3: Implement the evaluator launcher**

Require Docker image `pi05-libero-eval:650c5b0`, verified manifests, and a new/empty output directory. Use host networking, the selected NVIDIA device, OpenPI at `/app`, repository at `/repro`, `MUJOCO_GL=egl`, and `PYTHONPATH=/repro/src:/app:/app/packages/openpi-client/src:/app/third_party/libero`. Activate `/.venv` and invoke the evaluator with suite `libero_10`, seed 7, and 50 trials.

- [ ] **Step 4: Implement the architecture snapshot**

Assert effective config values `pi05=True`, `action_horizon=10`, `action_dim=32`, `discrete_state_input=False`, `paligemma_variant="gemma_2b"`, and `action_expert_variant="gemma_300m"`. Restore bfloat16 params, flatten with `jax.tree_util.tree_flatten_with_path`, and write stable paths/shapes/dtypes/totals plus source and checkpoint identities. Verify both checkpoints have the same architecture and different manifest hashes.

- [ ] **Step 5: Build and test the official image**

Run:

```bash
docker build -t pi05-libero-eval:650c5b0 \
  -f external_repos/openpi/examples/libero/Dockerfile external_repos/openpi
bash -n scripts/run_policy_server.sh scripts/run_libero_eval.sh
docker run --rm --gpus 'device=1' pi05-libero-eval:650c5b0 \
  nvidia-smi --query-gpu=name --format=csv,noheader
```

Expected: build succeeds and container reports NVIDIA H200 NVL.

- [ ] **Step 6: Document and commit**

README documents source init, downloads, manifest verification, build, both servers/evaluators, gates, no-resume rule, reports, and debug order. Commit scripts, README, and small snapshot with `feat: add reproducible pi05 LIBERO launchers`.

---

### Task 9: Execute Environment and Policy Gates

**Files:**
- Create ignored outputs under `runs/smoke/`, `runs/two-episode-{full,early}/`, and `runs/pilot-{full,early}/`
- Create committed `artifacts/manifests/smoke-gates.json`

**Interfaces:**
- Produces: verified gate evidence required before primary evaluation.

- [ ] **Step 1: Run the ten-task environment smoke test**

In the official container, create all ten tasks, set initial state 0, render both 256×256 cameras, execute one dummy action, and record task language, BDDL/init hashes, observation keys/shapes/dtypes, versions, `done`, and `check_success`. Require ten successes, finite state, both RGB images, and no predicate disagreement.

- [ ] **Step 2: Run one inference for each checkpoint**

Start each server fresh and send the same saved observation. Record input hashes, timing, action shape/dtype/min/max/mean, and norm-stat hash. Require finite 10×7 actions and checkpoint-specific norm stats.

- [ ] **Step 3: Run two episodes per checkpoint**

Use fresh key-0 server processes; run task 0 episodes 0–1. Require two unique valid records, two `ffprobe`-readable videos, correct first-frame orientation, finite actions, and predicate agreement per checkpoint.

- [ ] **Step 4: Run the 10×5 pilots**

Use fresh server processes and official order for trials 0–4 of all tasks. Never resume a pilot with a restarted server. Require 50 valid unique records, zero invalid files/disagreements, finite actions, and independently recomputed reports per checkpoint.

- [ ] **Step 5: Debug every failed gate before advancing**

Preserve logs and outputs. Follow provenance → rendering → state → action → reset/RNG → task semantics. State one hypothesis, run one A/B comparison, record it under `runs/debug/<timestamp>-<hypothesis>/notes.md`, fix only the evidenced cause, and rerun the entire gate.

- [ ] **Step 6: Commit durable gate metadata**

Commit only versions, hashes, and small gate summaries as `test: verify pi05 LIBERO smoke gates`.

---

### Task 10: Run and Audit Both 500-Episode Primary Evaluations

**Files:**
- Create ignored `runs/primary-full/` and `runs/primary-early/`
- Create: `results/pi05-libero-long-summary.json`
- Create: `results/pi05-libero-long-summary.md`
- Create: `results/pi05-libero-long-verification.txt`

**Interfaces:**
- Produces: exactly 1,000 requested episode outcomes and final audited reproduction report.

- [ ] **Step 1: Capture pre-run system state**

Record timestamps, `nvidia-smi`, disk, Docker digest, Git revisions/status, all artifact hashes, and exact launch commands. Require enough free memory on GPU 1/2 and never disturb GPU 0's unrelated process.

- [ ] **Step 2: Start fresh model servers**

Launch full on GPU 1/port 8001 and early on GPU 2/port 8002 with separate logs. Poll `/healthz`; verify checkpoint and norm-stat paths. No other client may query either server.

- [ ] **Step 3: Start and monitor both evaluators**

Launch both from task 0 episode 0 into empty output directories. Monitor logs, GPU memory/temperature, disk, record count, invalid diagnostics, and health at intervals under 60 seconds while actively working. Counts must increase monotonically and no invalid episode may appear.

- [ ] **Step 4: Preserve RNG integrity on interruption**

If a server restarts, an unrelated client connects, an invalid episode occurs, or RNG continuity is uncertain, preserve the run as diagnostic, fix the cause, and restart that checkpoint from episode zero in a new directory. Never splice RNG streams.

- [ ] **Step 5: Generate independent reports**

Run:

```bash
uv run python scripts/report_results.py \
  --full runs/primary-full/episodes.jsonl \
  --early runs/primary-early/episodes.jsonl \
  --json results/pi05-libero-long-summary.json \
  --markdown results/pi05-libero-long-summary.md
```

Require 500 unique valid records and 50/task per checkpoint, no predicate disagreements, full rate `[89.4%, 95.4%]`, and early rate `[38%, 48%]`.

- [ ] **Step 6: Systematically debug score discrepancies**

If outside either interval, do not accept. Compare failed videos, first-frame hashes, actions, prompt/state payloads, norm hashes, and official protocol line by line. Run isolated A/B diagnostics; for early, only after validating OpenPI protocol, run COAST 15/30. Repeat the affected uninterrupted 500-episode run after each evidence-backed fix until accepted.

- [ ] **Step 7: Run completion audit and commit summary**

Independently prove: two checkpoints; ten tasks/checkpoint; 50 unique valid episodes/task; 500/checkpoint; 1,000 total; zero invalids, duplicates, or predicate disagreements; valid manifests; exact report recomputation. Save commands, output, exit codes, revisions, image digest, and report hashes to the verification file. Commit the three small result files with `results: reproduce pi05 LIBERO-Long checkpoints`.

- [ ] **Step 8: Final verification**

Run `uv run pytest -v`, require clean pinned OpenPI status, and inspect root status for unintended changes. The goal remains incomplete until every Task 10 audit item passes.

---

## Execution Choice

The user granted continuous approval and requested no further consent prompts. Execute inline in the current session with `superpowers:executing-plans`; do not dispatch subagents under current collaboration constraints.
