# π₀.₅ LIBERO-Long Reproduction Design

Date: 2026-08-01

## Objective

Replace the legacy DreamZero, DreamDojo, WAM, and earlier NeurIPS 2026 workspace contents with a focused, auditable reproduction of two public π₀.₅ checkpoints on LIBERO-10:

- Full `pi05_libero` checkpoint from `gs://openpi-assets/checkpoints/pi05_libero/`.
- Early 2,000-step checkpoint from `brandonyang/openpi-libero-2000`.
- Ten LIBERO-10 tasks, 50 valid trials per task, 500 trials per checkpoint.
- MuJoCo physics, robosuite, BDDL task definitions, pinned LIBERO initial states, native environment success predicates, and RGB plus robot-state model inputs.

The primary reproduction is successful when the full checkpoint reaches 92.4% within 3 percentage points and the early checkpoint reaches 43% within 5 percentage points. Per-task early-checkpoint percentages are reported with Wilson 95% intervals rather than treated as exact 50-trial targets because the COAST paper values are integer multiples of 1/15 and are not the same protocol as OpenPI's 50-state evaluation.

## Source and Version Locks

The run must record and validate these immutable inputs before evaluation:

| Component | Locked source |
| --- | --- |
| OpenPI | `Physical-Intelligence/openpi` commit `650c5b0283a49c42784fb5055a0507da2c6d347d` |
| LIBERO | OpenPI gitlink commit `f78abd68ee283de9f9be3c8f7e2a9ad60246e95c` |
| robosuite | `1.4.1` from OpenPI's LIBERO lock file |
| MuJoCo | `3.2.3` from OpenPI's LIBERO lock file |
| Evaluator Python | Python 3.8 inside the official OpenPI LIBERO Docker image |
| Full checkpoint | `gs://openpi-assets/checkpoints/pi05_libero/` |
| Early checkpoint | Hugging Face revision `aaeeabc72f8a50a8fa2d04544332c8ec1cd0142e` |
| Dataset | `yifengzhu-hf/LIBERO-datasets`, revision `f13aa24a3da8c43c7225569f28c562979fa0e35a`, `libero_10` only |

Every downloaded artifact receives a path, byte-size, and SHA-256 manifest. The full checkpoint is already present in the local OpenPI cache and has an exact path-and-size match to all 16 objects in the public GCS prefix. The early checkpoint download includes only `_CHECKPOINT_METADATA`, `assets/**`, and `params/**` (12,440,616,902 bytes); its 32,287,153,562-byte optimizer/train state is not required for inference. The ten requested HDF5 files total about 13.73 GB and are retained for provenance and data checks, not used as evaluation initial states.

The two checkpoints have different normalization-statistics hashes. Each policy server must load the `assets/physical-intelligence/libero/norm_stats.json` shipped with its own checkpoint. Sharing or silently substituting normalization statistics is a hard failure.

## Workspace Cleanup

Retain only:

- `.git/` and the new focused repository history.
- `external_repos/openpi/` at the locked commit.
- The rewritten project README and new reproduction source, tests, configuration, documentation, manifests, logs, videos, and reports.

Delete the explicitly authorized legacy contents, including:

- `.venv/` and `.venv_agibot_eval/`.
- `table30v2_multitask_baseline_aloha_repo/` and its remote-file manifest.
- `groot/`, old ACT/socket scripts, old vector figures, editor-only configuration, and legacy caches.
- The already-missing 12,025 tracked DreamZero, DreamDojo, WAM, and earlier NeurIPS files.

Before deletion, resolve exact paths, check that no active process uses them, and record the cleanup manifest. Do not touch files outside the current workspace or terminate other users' processes. The cleanup is expected to release roughly 56 GB in addition to the large directories that were already removed externally before implementation began.

## Runtime Architecture

The authoritative run uses two independent, sequential OpenPI server/evaluator chains:

```text
GPU 1: full π₀.₅ server  <-> official LIBERO evaluator -> 500 episodes
GPU 2: early π₀.₅ server <-> official LIBERO evaluator -> 500 episodes
GPU 0: COAST audit, targeted reruns, and repeatability checks once available
```

The two 500-episode chains may run concurrently, but each chain processes the official task order and episode order sequentially. OpenPI's JAX policy starts at `jax.random.key(0)` and splits its key on every inference request, so task-sharding one checkpoint across multiple independent servers would reset the policy RNG and would not reproduce the official request sequence. Three-server task sharding is therefore permitted only as an explicitly labeled diagnostic and cannot supply the primary reported score.

GPU 0 is currently occupied by another user's training process. It is never preempted or terminated; it is used only after capacity becomes available. MuJoCo EGL rendering runs in the official Docker evaluator while the OpenPI model server runs in the repository's Python 3.11 environment.

## Evaluation Protocol

The evaluator preserves the official OpenPI data path and control semantics:

1. Select `libero_10`, set NumPy and environment seed to 7, and process the official LIBERO task order.
2. For task `t`, load the pinned BDDL file and `task_suite.get_task_init_states(t)`.
3. For episode indices 0 through 49, call `env.reset()` and then `env.set_init_state(initial_states[episode_idx])`.
4. Render `agentview_image` and `robot0_eye_in_hand_image` at 256×256, rotate both 180 degrees, and resize with padding to 224×224.
5. Build the 8-dimensional state from end-effector position, quaternion converted to axis-angle, and gripper joint positions.
6. Execute ten dummy actions `[0, 0, 0, 0, 0, 0, -1]` to settle objects.
7. Send the exact task-language prompt to π₀.₅. The model predicts a 10×7 action chunk; execute five actions before replanning.
8. Stop on the native robosuite/LIBERO `done` success signal or after 520 policy steps plus the ten settling steps.
9. Record `env.check_success()` alongside `done` as an audit. `done` remains the authoritative OpenPI success value; disagreement is an evaluation error requiring investigation.

The evaluator extension may add task selection, resumability, unique video names, structured records, and validation. It must not alter image geometry, prompt text, state conversion, normalization, action semantics, control rate, horizon, reset order, initial states, or success predicates.

An infrastructure exception, websocket disconnect, invalid tensor, or rendering failure marks an episode `invalid`, stops that shard, and is repaired before resuming the same episode. It is never silently counted as a policy failure. A valid final result contains exactly 50 valid outcomes for every task and checkpoint.

## Task Reporting

Internally retain LIBERO's official task IDs and order. The final report also renders the user's requested order:

1. Cream cheese and butter to basket.
2. Black bowl to bottom drawer and close.
3. Turn on stove and put moka pot on it.
4. White mug to left plate and yellow-white mug to right plate.
5. Both moka pots to stove.
6. Alphabet soup and cream cheese to basket.
7. Yellow-white mug to microwave and close.
8. Alphabet soup and tomato sauce to basket.
9. Book to back compartment of caddy.
10. White mug to plate and chocolate pudding right of plate.

Each episode record contains checkpoint identity, task ID and name, episode index, initial-state hash, seed, success, number of steps, wall time, exception state, first-frame hash, action statistics, video path, and environment revisions. Aggregate output contains successes out of 50, rate, Wilson 95% interval, overall successes out of 500, and comparison with the appropriate public reference.

## Validation Ladder

Evaluation advances only when the preceding gate passes:

1. **Static provenance:** revisions, object inventories, byte sizes, SHA-256 manifests, checkpoint-specific normalization stats, and ten dataset files.
2. **Environment smoke test:** import pinned LIBERO/robosuite/MuJoCo, create every task, load initial state 0, render both cameras, step a dummy action, and verify native success calls.
3. **Policy smoke test:** load each checkpoint, verify model/config compatibility, input masks and shapes, finite 10×7 actions, and plausible unnormalized action ranges.
4. **Two-episode test:** two episodes per checkpoint with complete structured records and playable, correctly oriented videos.
5. **Small pilot:** ten tasks times five trials per checkpoint. Require no invalid episodes and inspect per-task videos, action distributions, reset hashes, and predicate consistency.
6. **Primary run:** 500 sequential episodes for each checkpoint, with atomic progress records and exact resume.
7. **Result audit:** independently recompute aggregates from episode records and verify cardinality, uniqueness, task mapping, and confidence intervals.

## Systematic Discrepancy Debugging

If a pilot or primary score is outside the target interval, preserve the failing run and debug one layer at a time with an explicit baseline and expected observation:

1. **Provenance:** wrong OpenPI/LIBERO commit, incomplete checkpoint, wrong normalization assets, or mismatched task/init files.
2. **Rendering:** EGL device, camera names, 180-degree rotation, channel order, dtype, 256 render size, or 224 padding.
3. **State:** quaternion convention, axis-angle conversion, state ordering, gripper dimension, dtype, or normalization.
4. **Action:** 7-dimensional truncation, unnormalization, gripper sign, action chunk length, replanning interval, or control frequency.
5. **Reset/RNG:** environment seed timing, reset-before-set-state order, episode index, policy request sequence, or resume-induced RNG discontinuity.
6. **Task semantics:** exact prompt, BDDL file, horizon, settling actions, `done`, `_check_success`, and container dependency drift.
7. **Checkpoint protocol:** only after the OpenPI protocol is validated, run the COAST 15/30-rollout protocol to explain remaining early-checkpoint differences.

Never improve reported success by selecting favorable seeds, skipping failures, extending horizons, modifying BDDL predicates, using a visual judge in place of the environment, or tuning separately on the final 50 states. Any justified compatibility change must be isolated in an A/B run and documented with evidence.

## Acceptance and Deliverables

The reproduction is accepted only when all of the following are present and verified:

- Focused clean repository without the authorized legacy projects.
- Reproducible environment build and launch commands.
- Pinned source/checkpoint/dataset manifests.
- Full and early checkpoint policy-load evidence.
- Exactly 1,000 valid primary episode records: 10 tasks × 50 trials × 2 checkpoints.
- Per-episode videos or a documented storage-reduced video policy that still retains all failures and audit samples.
- Machine-readable per-task and aggregate reports plus a human-readable comparison table.
- Full checkpoint overall success in `[89.4%, 95.4%]`.
- Early checkpoint overall success in `[38%, 48%]`, or a root-cause-backed protocol analysis plus COAST-compatible audit if the official 50-state result is statistically different.
- Fresh verification commands proving counts, hashes, environment versions, result recomputation, and absence of invalid episodes.

The active goal is not complete until the two 500-episode runs and their audits are finished. Passing smoke tests or pilots alone is not completion.
