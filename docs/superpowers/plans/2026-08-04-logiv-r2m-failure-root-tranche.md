# LOGIV-R2M Phase 0 Failure-Root Tranche Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run a bounded paired Task 5/8 development tranche that preserves the exact Base rollout while obtaining at least one real, validated `CONFIRMED_DEVIATION` recovery root per task or returning an auditable bounded no-go result.

**Architecture:** Reuse the existing episode-seeded policy server, `run_logiv_eval.sh`, Phase 0 Shadow runtime, JSON parity artifacts, and recovery dataset validator. Run one Base/Shadow pair at a time under a single server process, apply the parity gate before interpreting roots, stop each task independently after its first valid confirmed root, and permit no detector code change without a real missed-deviation artifact and a separate evidence-specific TDD amendment.

**Tech Stack:** Bash, Docker, Python 3.11, OpenPI pi0.5 policy server, LIBERO simulator, jq, pytest, immutable JSON/NPZ recovery artifacts.

## Global Constraints

- Execute only in the clean linked worktree at `/mnt/data3/data_xingrui/lueq/NuerIPS_2026/.worktrees/logiv-r2m-phase0` on branch `logiv-r2m-phase0`.
- Do not modify or copy dirty experimental files from `pi05-libero-reproduction`.
- Keep master seed `7`, checkpoint `full`, `episode-seeded-v1` policy/simulator RNG, `METADATA_ASSISTED`, `NOMINAL`, and the 520-action Base budget fixed.
- Use one uninterrupted episode-seeded policy server process for every valid pair. If it restarts, rerun both arms of the in-progress pair.
- Run `BASE` before `SHADOW_LOGIV` for every candidate and require exact equality of the ten Base execution fields before interpreting Shadow evidence.
- Shadow collection is development-only, oracle-grounded, `DEV` split, read-only, and records zero Shadow VLM and recovery-policy requests.
- Do not train on or allocate roles to any raw root in this tranche.
- A terminal Base failure is not a confirmed deviation. Only frozen-contract oracle/event evidence can establish a detector miss.
- Permit at most one evidence-backed detector fix. A discovered fix requires a separate root-cause-specific TDD plan amendment before production code changes.
- Stop each task after its first mechanically valid confirmed root or after six frozen candidates.
- Do not claim success-rate improvement from this tranche. The later paired gate remains at least `+10/100` over Base across Task 5/8, at most two negative flips per task, and an enabled capability contract for every intervention.

---

## Runtime File Map

- Read: `scripts/run_policy_server.sh` — launch the pinned episode-seeded policy server.
- Read: `scripts/run_logiv_eval.sh` — launch one Base or Shadow LIBERO evaluator in the pinned container.
- Read: `scripts/validate_recovery_dataset.py` — verified root loading and split/leakage validation.
- Read: `configs/logiv/r2m-monitor-evidence-v1.json` — frozen Task 5/8 evidence contracts.
- Generate: `runs/r2m-phase0-root-tranche-policy-server/` — ignored server log directory.
- Generate: `runs/r2m-phase0-root-tranche/task-XX-episode-YYY/{base,shadow}/` — ignored per-pair artifacts.
- Create after evidence exists: `docs/experiments/2026-08-04-logiv-r2m-failure-root-tranche.md` — committed final report.
- Modify only if a proven miss occurs: one evidence-specific monitor, adapter, or contract file plus its focused test, as defined by a separate plan amendment.

## Pair Audit Contract

For a candidate with task ID `T` and episode index `E`, define:

```bash
task_tag=$(printf 'task-%02d-episode-%03d' "$T" "$E")
pair_dir="runs/r2m-phase0-root-tranche/$task_tag"
base_dir="$pair_dir/base"
shadow_dir="$pair_dir/shadow"
artifact_rel=$(printf 'artifacts/task_%02d/episode_%03d' "$T" "$E")
```

The audit must execute these checks after both evaluators exit zero:

```bash
test "$(wc -l < "$base_dir/episodes.jsonl")" -eq 1
test "$(wc -l < "$shadow_dir/episodes.jsonl")" -eq 1
jq -e '.valid == true and .task_id == $task and .episode_idx == $episode' \
  --argjson task "$T" --argjson episode "$E" "$base_dir/episodes.jsonl"
jq -e '.valid == true and .task_id == $task and .episode_idx == $episode' \
  --argjson task "$T" --argjson episode "$E" "$shadow_dir/episodes.jsonl"

diff -u \
  <(jq -S '{steps,base_policy_requests,done_signal,post_settling_success,initial_state_sha256,base_prompt_sha256,base_checkpoint_sha256,policy_client_config_sha256,request_envelope_log_sha256,actions_sha256}' "$base_dir/$artifact_rel/base_execution.json") \
  <(jq -S '{steps,base_policy_requests,done_signal,post_settling_success,initial_state_sha256,base_prompt_sha256,base_checkpoint_sha256,policy_client_config_sha256,request_envelope_log_sha256,actions_sha256}' "$shadow_dir/$artifact_rel/base_execution.json")

diff -u \
  <(jq -S . "$base_dir/$artifact_rel/policy_rng.json") \
  <(jq -S . "$shadow_dir/$artifact_rel/policy_rng.json")

jq -e '.status == "ACCEPTED" and .request_count == 1' \
  "$shadow_dir/$artifact_rel/initial_proposal.json"
jq -e '.initial_proposal_requests == 1 and .shadow_vlm_requests == 0 and .recovery_policy_requests == 0' \
  "$shadow_dir/$artifact_rel/compute_accounting.json"
jq -e '
  .shadow_parity_valid == true and
  ([.aggregate_errors,
    .callback_errors,
    .event_tracker_errors,
    .evidence_overflows,
    .proposal_callback_errors,
    .provenance_errors,
    .root_write_errors,
    .snapshot_errors,
    .trigger_callback_errors] | all(. == 0))
' "$shadow_dir/$artifact_rel/shadow_monitor.json"
```

Then count and validate roots:

```bash
root_count=$(find "$shadow_dir" -name recovery_root.json -type f | wc -l)
printf 'task=%d episode=%d root_count=%d\n' "$T" "$E" "$root_count"
if test "$root_count" -gt 0; then
  root_summary=$(uv run python scripts/validate_recovery_dataset.py \
    "$shadow_dir/artifacts")
  printf '%s\n' "$root_summary"
  jq -e '
    .DEV == .total and
    .TRAIN == 0 and
    .HELDOUT == 0
  ' <<<"$root_summary"
  if jq -e '.CONFIRMED_DEVIATION >= 1' <<<"$root_summary" >/dev/null; then
    echo 'pair classification: CONFIRMED_ROOT'
  else
    echo 'pair classification: CANDIDATE_ONLY'
  fi
else
  echo 'pair classification: NO_ROOT'
fi
```

The classification is `CONFIRMED_ROOT` only for a valid confirmed-root pair. A
nonzero `root_count` containing candidates only is diagnostic and does not stop
that task. Any command other than the explicit classification branch failing is
a hard gate, not a zero-yield candidate.

---

### Task 1: Verify the clean baseline and start one frozen policy server

**Files:**
- Read: `artifacts/manifests/full-checkpoint.json`
- Generate: `runs/r2m-phase0-root-tranche-policy-server/server.log`

**Interfaces:**
- Consumes: design commit `4d56a5de`, full checkpoint directory, idle GPU 0, free TCP port 8010.
- Produces: a clean tested branch and one uninterrupted episode-seeded policy server session used by Tasks 2–5.

- [ ] **Step 1: Verify worktree identity and cleanliness**

Run:

```bash
pwd -P
git branch --show-current
git status --porcelain=v1
git rev-parse HEAD
```

Expected: the path is `/mnt/data3/data_xingrui/lueq/NuerIPS_2026/.worktrees/logiv-r2m-phase0`, the branch is `logiv-r2m-phase0`, status prints nothing, and HEAD contains the approved design commit.

- [ ] **Step 2: Verify static checks and the complete baseline suite**

Run:

```bash
git diff --check
bash -n scripts/run_logiv_eval.sh scripts/run_policy_server.sh
uv run python -m compileall -q src scripts
uv run pytest -q
```

Expected: every command exits 0 and pytest reports `379 passed` before any detector change.

- [ ] **Step 3: Verify the selected device and port are free**

Run:

```bash
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader
if ss -ltn | rg -q ':8010[[:space:]]'; then
  echo 'port 8010 is already in use' >&2
  exit 1
fi
```

Expected: GPU 0 has enough free memory for the full checkpoint and port 8010 is absent. If GPU 0 is no longer free, stop and amend the execution record with one different idle GPU before starting; do not migrate after the first valid pair.

- [ ] **Step 4: Launch the policy server in a dedicated long-lived terminal session**

Run:

```bash
scripts/run_policy_server.sh full 0 8010 \
  /mnt/data3/data_xingrui/.cache/openpi/openpi-assets/checkpoints/pi05_libero \
  runs/r2m-phase0-root-tranche-policy-server
```

Expected: the process remains running, logs the full checkpoint and verified norm-statistics hash, loads the policy on GPU 0, and listens on 8010. Preserve the returned terminal/session identifier; do not send unrelated inference requests to this port.

- [ ] **Step 5: Verify readiness without sending a policy request**

Run from a second terminal after the server reports ready:

```bash
ss -ltn | rg ':8010[[:space:]]'
rg -n 'checkpoint=full|Creating episode-seeded server' \
  runs/r2m-phase0-root-tranche-policy-server/server.log
```

Expected: port 8010 is listening and the log identifies the episode-seeded full policy.

---

### Task 2: Run and audit the first Task 5 pair

**Files:**
- Generate: `runs/r2m-phase0-root-tranche/task-05-episode-000/base/`
- Generate: `runs/r2m-phase0-root-tranche/task-05-episode-000/shadow/`

**Interfaces:**
- Consumes: the uninterrupted Task 1 server and the Pair Audit Contract.
- Produces: the first Task 5 paired evidence and either a valid confirmed root, a clean zero-yield result, or a hard-gate failure.

- [ ] **Step 1: Prove the candidate directories do not exist**

Run:

```bash
test ! -e runs/r2m-phase0-root-tranche/task-05-episode-000/base
test ! -e runs/r2m-phase0-root-tranche/task-05-episode-000/shadow
```

Expected: both tests exit 0.

- [ ] **Step 2: Run Base**

Run:

```bash
scripts/run_logiv_eval.sh BASE 0 8010 \
  runs/r2m-phase0-root-tranche/task-05-episode-000/base \
  --run-id r2m-phase0-root-task05-episode000-base \
  --checkpoint-name full \
  --goal-mode METADATA_ASSISTED \
  --deviation-mode NOMINAL \
  --development-only \
  --task-ids 5 \
  --episode-indices 0 \
  --seed 7
```

Expected: exit 0 and exactly one valid episode record.

- [ ] **Step 3: Run Shadow with root collection**

Run:

```bash
scripts/run_logiv_eval.sh SHADOW_LOGIV 0 8010 \
  runs/r2m-phase0-root-tranche/task-05-episode-000/shadow \
  --run-id r2m-phase0-root-task05-episode000-shadow \
  --checkpoint-name full \
  --goal-mode METADATA_ASSISTED \
  --deviation-mode NOMINAL \
  --oracle-grounding \
  --development-only \
  --task-ids 5 \
  --episode-indices 0 \
  --seed 7 \
  --collect-recovery-roots \
  --recovery-root-split DEV
```

Expected: exit 0 and exactly one valid episode record.

- [ ] **Step 4: Execute the Pair Audit Contract with `T=5` and `E=0`**

Expected: all parity and error-gate commands pass. Record terminal outcomes, steps, Base requests, derived policy seed, proposal status, certificate state, anomaly count, confirmed count, root count, and validator JSON.

- [ ] **Step 5: Classify Task 5**

If the validator summary contains `CONFIRMED_DEVIATION >= 1`, mark Task 5 complete and skip its remaining candidates. If not, retain the artifacts and advance Task 5 to episode 3 only after Task 8 attempt 1.

---

### Task 3: Run and audit the first Task 8 pair

**Files:**
- Generate: `runs/r2m-phase0-root-tranche/task-08-episode-004/base/`
- Generate: `runs/r2m-phase0-root-tranche/task-08-episode-004/shadow/`

**Interfaces:**
- Consumes: the same uninterrupted Task 1 server and the Pair Audit Contract.
- Produces: the first Task 8 paired evidence and either a valid confirmed root, a clean zero-yield result, or a hard-gate failure.

- [ ] **Step 1: Prove the candidate directories do not exist**

Run:

```bash
test ! -e runs/r2m-phase0-root-tranche/task-08-episode-004/base
test ! -e runs/r2m-phase0-root-tranche/task-08-episode-004/shadow
```

Expected: both tests exit 0.

- [ ] **Step 2: Run Base**

Run:

```bash
scripts/run_logiv_eval.sh BASE 0 8010 \
  runs/r2m-phase0-root-tranche/task-08-episode-004/base \
  --run-id r2m-phase0-root-task08-episode004-base \
  --checkpoint-name full \
  --goal-mode METADATA_ASSISTED \
  --deviation-mode NOMINAL \
  --development-only \
  --task-ids 8 \
  --episode-indices 4 \
  --seed 7
```

Expected: exit 0 and exactly one valid episode record.

- [ ] **Step 3: Run Shadow with root collection**

Run:

```bash
scripts/run_logiv_eval.sh SHADOW_LOGIV 0 8010 \
  runs/r2m-phase0-root-tranche/task-08-episode-004/shadow \
  --run-id r2m-phase0-root-task08-episode004-shadow \
  --checkpoint-name full \
  --goal-mode METADATA_ASSISTED \
  --deviation-mode NOMINAL \
  --oracle-grounding \
  --development-only \
  --task-ids 8 \
  --episode-indices 4 \
  --seed 7 \
  --collect-recovery-roots \
  --recovery-root-split DEV
```

Expected: exit 0 and exactly one valid episode record.

- [ ] **Step 4: Execute the Pair Audit Contract with `T=8` and `E=4`**

Expected: all parity and error-gate commands pass. Record the same fields as Task 2.

- [ ] **Step 5: Classify Task 8**

If the validator summary contains `CONFIRMED_DEVIATION >= 1`, mark Task 8 complete and skip its remaining candidates. If not, retain the artifacts and advance Task 8 to episode 0 after the next eligible Task 5 candidate.

---

### Task 4: Continue the frozen alternating candidate schedule

**Files:**
- Generate conditionally: the remaining per-pair directories under `runs/r2m-phase0-root-tranche/`.

**Interfaces:**
- Consumes: Task 2/3 classifications, the same server process, and the Pair Audit Contract.
- Produces: a bounded final classification for both tasks.

- [ ] **Step 1: Freeze the remaining order before reading new outcomes**

Use exactly this order, skipping only a task already complete:

```text
Task 5 episode 3
Task 8 episode 0
Task 5 episode 38
Task 8 episode 2
Task 5 episode 45
Task 8 episode 3
Task 5 episode 2
Task 8 episode 6
Task 5 episode 6
Task 8 episode 8
```

Expected: no candidate is substituted, reordered, or removed based on observed outcomes.

- [ ] **Step 2: Run each eligible Base arm with the fixed command shape**

For each exact `(T, E)` above, compute `TT=$(printf '%02d' "$T")` and `EEE=$(printf '%03d' "$E")`, prove both output directories absent, then run:

```bash
scripts/run_logiv_eval.sh BASE 0 8010 \
  "runs/r2m-phase0-root-tranche/task-$TT-episode-$EEE/base" \
  --run-id "r2m-phase0-root-task$TT-episode${EEE}-base" \
  --checkpoint-name full \
  --goal-mode METADATA_ASSISTED \
  --deviation-mode NOMINAL \
  --development-only \
  --task-ids "$T" \
  --episode-indices "$E" \
  --seed 7
```

Expected: exit 0 and exactly one valid episode record.

- [ ] **Step 3: Run the matching Shadow arm before advancing to another candidate**

Run:

```bash
scripts/run_logiv_eval.sh SHADOW_LOGIV 0 8010 \
  "runs/r2m-phase0-root-tranche/task-$TT-episode-$EEE/shadow" \
  --run-id "r2m-phase0-root-task$TT-episode${EEE}-shadow" \
  --checkpoint-name full \
  --goal-mode METADATA_ASSISTED \
  --deviation-mode NOMINAL \
  --oracle-grounding \
  --development-only \
  --task-ids "$T" \
  --episode-indices "$E" \
  --seed 7 \
  --collect-recovery-roots \
  --recovery-root-split DEV
```

Expected: exit 0 and exactly one valid episode record.

- [ ] **Step 4: Execute the Pair Audit Contract immediately**

Expected: parity and all error gates pass before another candidate starts. Record the same report fields as Tasks 2 and 3.

- [ ] **Step 5: Apply the per-task stop rule**

After a validator summary reports at least one confirmed DEV root, mark that task complete and skip only its later candidates. Continue the other task until it also completes or reaches its sixth candidate.

- [ ] **Step 6: Apply the detector-miss gate when necessary**

If no confirmed root is emitted, inspect only that completed development pair's oracle facts, event provenance, state artifacts, and video. If they do not prove a frozen-contract deviation, classify the pair as clean zero-yield and continue. If they do prove a frozen-contract deviation, stop all rollout execution and proceed to Task 5; do not edit production code yet.

---

### Task 5: Diagnose one proven detector miss and hand off an evidence-specific fix

**Files:**
- Read: the missed pair's complete artifact directory and video.
- Read: `configs/logiv/r2m-monitor-evidence-v1.json`
- Read: `src/pi05_libero_repro/logiv/shadow_monitor.py`
- Read: `src/pi05_libero_repro/logiv/libero_adapter.py`
- Test candidate: `tests/logiv/test_shadow_monitor.py` or `tests/logiv/test_libero_adapter.py`

**Interfaces:**
- Consumes: one artifact-backed, contract-supported miss from Task 4.
- Produces: either a precise no-code explanation or a paused tranche plus an evidence-specific design/plan handoff consuming the tranche's single detector-fix budget.

This task is conditional. Skip it when no proven detector miss exists.

- [ ] **Step 1: Use `superpowers:systematic-debugging` to trace the evidence path**

Trace the real state from adapter observation, oracle facts, event tracker, evidence ledger, certificate reconciler, trigger creation, and root writer. Identify the first boundary at which expected evidence is lost or rejected.

Expected: one falsifiable root-cause statement tied to artifact values and exact code/config lines. “The episode failed” or “thresholds are too strict” is not sufficient.

- [ ] **Step 2: Decide whether the miss is inside the frozen supported contract**

If the state is OOD, visually ambiguous, lacks the required manipulation event, or never remains stable for the frozen confirmations, record a correct fail-closed decision and return to Task 4 without consuming the fix. If the evidence is supported and fresh, stop and write a plan amendment containing the exact failing test, expected RED failure, minimal code/config diff, focused GREEN command, complete verification command, and separate commit message.

Expected: no production edit occurs in this step.

- [ ] **Step 3: Pause this plan and create the evidence-specific handoff before editing**

Write and obtain approval for
`docs/superpowers/specs/2026-08-04-logiv-r2m-detector-miss-fix-design.md`, then create
`docs/superpowers/plans/2026-08-04-logiv-r2m-detector-miss-fix.md`. The design and
plan must name exactly one parser, contract, or state-machine defect and include
the artifact-backed failing test, expected RED failure, minimal implementation,
focused GREEN command, full verification, separate commit, and new-directory
rerun of both affected arms. They must not change prompts, checkpoint, action
budget, confirmation thresholds, or unrelated task classes.

Expected: this tranche plan remains paused with no production edit until the
evidence-specific design is approved and its plan is executed. After that plan
finishes, resume Task 4 using the same uninterrupted server process. If the
server did not remain uninterrupted, rerun both affected arms after starting a
new recorded tranche process. A second distinct proven detector defect ends the
tranche as `NO_GO`.

---

### Task 6: Verify all eligible roots and write the tranche report

**Files:**
- Create: `docs/experiments/2026-08-04-logiv-r2m-failure-root-tranche.md`
- Read: every attempted pair under `runs/r2m-phase0-root-tranche/`

**Interfaces:**
- Consumes: all valid, invalid, zero-yield, candidate-only, and confirmed-root pair results.
- Produces: one auditable `GO`, `PARTIAL`, or `NO_GO` report and a clean documentation commit.

- [ ] **Step 1: Revalidate the union of confirmed-root pairs**

Run:

```bash
eligible_shadow_dirs=()
while IFS= read -r shadow_dir; do
  monitor_path=$(find "$shadow_dir/artifacts" -name shadow_monitor.json -type f -print -quit)
  if test -n "$monitor_path" && jq -e '
    .shadow_parity_valid == true and
    .confirmed_deviations >= 1 and
    .root_count >= 1 and
    ([.aggregate_errors,
      .callback_errors,
      .event_tracker_errors,
      .evidence_overflows,
      .proposal_callback_errors,
      .provenance_errors,
      .root_write_errors,
      .snapshot_errors,
      .trigger_callback_errors] | all(. == 0))
  ' "$monitor_path" >/dev/null; then
    eligible_shadow_dirs+=("$shadow_dir/artifacts")
  fi
done < <(find runs/r2m-phase0-root-tranche -mindepth 2 -maxdepth 2 -type d -name shadow | sort)

printf 'eligible confirmed-root directories: %d\n' "${#eligible_shadow_dirs[@]}"
if test "${#eligible_shadow_dirs[@]}" -gt 0; then
  uv run python scripts/validate_recovery_dataset.py "${eligible_shadow_dirs[@]}"
fi
```

Expected: exit 0, only DEV roots, no leakage across eligible confirmed-root
pairs, and counts that agree with their per-pair validator results. Invalid,
candidate-only, and superseded rerun directories remain auditable but are not
included in this training-eligibility union.

- [ ] **Step 2: Rerun static and complete verification**

Run:

```bash
git diff --check
bash -n scripts/run_logiv_eval.sh scripts/run_policy_server.sh
uv run python -m compileall -q src scripts
uv run pytest -q
```

Expected: every command exits 0. Without a detector fix the count remains 379; with an approved fix it is greater than 379.

- [ ] **Step 3: Write the evidence report**

The report must include:

- design and implementation commits;
- server GPU, port, checkpoint/norm-stat hashes, and whether the process restarted;
- the complete frozen candidate order and explicit skipped-after-success entries;
- one row per attempted pair with task, episode, derived policy seed, Base/Shadow terminal status, steps, requests, ten-field parity, proposal status, certificate state, anomalies, confirmed deviations, roots, error buckets, and validator result;
- artifact paths for hard failures, candidate-only outcomes, confirmed roots, and any reruns;
- any detector-miss evidence, amendment, RED/GREEN proof, and fix commit;
- union validator JSON;
- `GO`, `PARTIAL`, or `NO_GO` under the approved definitions;
- the statements `Raw recovery roots were not passed to a trainer.` and `Recovery capability and takeover are not enabled.`;
- the limitation that this tranche does not establish detector recall, recovery competence, training sufficiency, or success-rate improvement over Base;
- the downstream fixed gate: at least `+10/100` across paired Task 5/8, at most two negative flips per task, and an enabled capability contract for every intervention.

- [ ] **Step 4: Stop the policy server after all evidence is durable**

Send an interrupt to the exact Task 1 server session and wait for it to exit. Do not use a broad process-kill command.

Expected: port 8010 is no longer listening and `server.log` remains intact.

- [ ] **Step 5: Commit only the report and any separately verified detector fix**

Run:

```bash
git status --short
git add docs/experiments/2026-08-04-logiv-r2m-failure-root-tranche.md
git commit -m "docs: record R2M failure-root tranche"
git status --short --branch
```

Expected: generated `runs/` artifacts remain ignored, the report commit succeeds, and the tracked worktree is clean. If a detector fix occurred, it already has its own earlier commit and is not folded into this documentation commit.
