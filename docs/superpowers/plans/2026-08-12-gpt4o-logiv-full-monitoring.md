# GPT-4o LOGIV Full-Monitoring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run LOGIV with GPT-4o initial PDDL planning, full-episode visual State Gate observation, and failure-localized plan repair, then reproduce the seed-7/seed-17 1000-pair development protocol on two GPUs with a target of at least 950 successes and zero negative flips.

**Architecture:** Add one dependency-free GPT-4o transport and three adapters at the existing proposal, grounder, and repair interfaces. Keep the current PDDL/VAL/DAG/controller/pi0.5 path intact, select the backend explicitly, and forward `OPENAI_API_KEY` into the existing evaluator container without persisting it. Run BASE then GPT-4o LOGIV on the same policy-server process for each two-GPU shard.

**Tech Stack:** Python 3.11, NumPy, Python standard library (`urllib`, `json`, `base64`, `zlib`), OpenAI Chat Completions API with `gpt-4o`, pytest, Bash, Docker, LIBERO/OpenPI, VAL.

## Global Constraints

- Use `gpt-4o`; do not substitute another model silently.
- Read `OPENAI_API_KEY` only from the environment and never log or persist its value.
- LOGIV visual monitoring starts at policy step 0 and remains active throughout execution.
- GPT-4o may propose PDDL actions and fact labels only; it may not create schemas, objects, goals, DAG edges, or continuous controls.
- Keep the fixed PDDL Domain, official BDDL goal, VAL, deterministic DAG compiler, four Gates, pi0.5 checkpoint, pi0.5 prompts, monitor interval 5, intervention/confirmation thresholds, retry limits, VAL limits, and 520-action budget unchanged.
- `UNKNOWN` never authorizes completion, failure repair, or dispatch.
- BASE remains action/request-identical before verified handoff; handoff is one-way `BASE -> LOGIV` high-level control.
- Native LIBERO success is absorbing.
- Unit tests use fake transports and incur no API charges.
- Final evaluation uses tasks 0..9, episode indices 0..49, and master seeds 7 and 17, producing 1000 paired episodes.
- Work in the current feature branch because the uncommitted LOGIV_ONLINE implementation is required input; preserve all unrelated/user changes.

---

### Task 1: Add a secret-safe GPT-4o structured-output transport

**Files:**
- Create: `src/pi05_libero_repro/logiv/gpt4o.py`
- Create: `tests/logiv/test_gpt4o.py`

**Interfaces:**
- Produces: `Gpt4oClient.from_env(...)`, `Gpt4oClient.complete_json(...)`, `Gpt4oRequestError`, `encode_png_data_url(...)`, and per-purpose request metrics.
- Consumes: `OPENAI_API_KEY`, two NumPy RGB arrays, strict JSON schemas, and a replaceable `urlopen` callable for tests.

- [ ] **Step 1: Write failing transport and PNG tests**

Add tests that require:

```python
def test_client_requires_environment_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(Gpt4oRequestError, match="OPENAI_API_KEY is not set"):
        Gpt4oClient.from_env()


def test_png_data_url_round_trips_signature():
    url = encode_png_data_url(np.zeros((2, 3, 3), dtype=np.uint8))
    assert base64.b64decode(url.removeprefix("data:image/png;base64,"))[:8] == b"\x89PNG\r\n\x1a\n"


def test_complete_json_sends_gpt4o_images_and_strict_schema(monkeypatch):
    # Fake urlopen captures the request and returns one Chat Completions JSON body.
    # Assert model == "gpt-4o", image_url inputs exist, json_schema.strict is true,
    # parsed content is returned, counters increment, and Authorization is redacted
    # from exceptions and provenance.
```

Also cover transient 429 retry, timeout exhaustion, refusal, malformed content,
model mismatch, HTTP error redaction, non-RGB/non-uint8 input, and stable request/
response hashes.

- [ ] **Step 2: Run RED tests**

Run:

```bash
uv run pytest -q tests/logiv/test_gpt4o.py
```

Expected: collection fails because `pi05_libero_repro.logiv.gpt4o` does not exist.

- [ ] **Step 3: Implement the minimum standard-library transport**

Implement:

```python
@dataclass(frozen=True)
class Gpt4oCallRecord:
    purpose: str
    response_id: str
    model: str
    input_tokens: int
    output_tokens: int
    elapsed_seconds: float
    retries: int
    request_sha256: str
    response_sha256: str


class Gpt4oClient:
    @classmethod
    def from_env(cls, *, model="gpt-4o", timeout_seconds=30.0,
                 max_retries=2, urlopen=urllib.request.urlopen): ...

    def complete_json(self, *, purpose: str, system: str, text: str,
                      images: Sequence[np.ndarray], schema_name: str,
                      schema: Mapping[str, Any]) -> Mapping[str, Any]: ...
```

Use `POST https://api.openai.com/v1/chat/completions`, multimodal user content,
and:

```json
{"type":"json_schema","json_schema":{"name":"...","strict":true,"schema":{}}}
```

Build PNG bytes using only `struct`, `zlib`, and `binascii.crc32`. Never retain
the key outside the client, render request headers, or include response bodies
in raised error messages.

- [ ] **Step 4: Run GREEN tests and commit**

Run:

```bash
uv run pytest -q tests/logiv/test_gpt4o.py
git diff --check
```

Expected: all focused tests pass and diff check is silent.

Commit only Task 1 files:

```bash
git add src/pi05_libero_repro/logiv/gpt4o.py tests/logiv/test_gpt4o.py
git commit -m "feat(logiv): add GPT-4o structured transport"
```

---

### Task 2: Add GPT-4o three-valued visual grounding

**Files:**
- Create: `src/pi05_libero_repro/logiv/gpt4o_grounding.py`
- Create: `tests/logiv/test_gpt4o_grounding.py`
- Modify: `src/pi05_libero_repro/logiv/shadow_runtime.py`
- Modify: `tests/logiv/test_shadow_runtime.py`

**Interfaces:**
- Produces: `Gpt4oGrounder`, implementing `acquire_epoch`, `ground`,
  `peek_snapshot`, `peek_advisory_partial_snapshot`, `observe_action_progress`,
  and `observe_action_progress_details` with the same observable contract as
  `LiberoOracleGrounder`.
- Consumes: `Gpt4oClient`, `LiberoObservationStore`, `TaskProblem`, the finite
  `monitored_fact_universe`, task instruction, and current graph/context IDs.

- [ ] **Step 1: Write failing grounder tests**

Create a fake GPT client returning canonical records such as:

```json
{
  "epoch_id": 0,
  "graph_version": null,
  "facts": [
    {"fact": "(at moka_pot_1 left_region)", "truth": "TRUE", "evidence": "main"},
    {"fact": "(holding moka_pot_1)", "truth": "FALSE", "evidence": "both"},
    {"fact": "(handempty)", "truth": "TRUE", "evidence": "wrist"}
  ]
}
```

Assert complete fact partitioning, `UNKNOWN` preservation, exact observation
hash binding, stale epoch/graph rejection, extra/missing/duplicate fact
rejection, exactly-one conflict rejection, cache reuse for the same observation,
fresh strict reads, request counting, and action-progress behavior.

Add a shadow-runtime test proving the initial observation is processed at policy
step 0 and later GPT observations occur on the unchanged five-step cadence
rather than monitoring beginning at failure time.

- [ ] **Step 2: Run RED tests**

Run:

```bash
uv run pytest -q tests/logiv/test_gpt4o_grounding.py \
  tests/logiv/test_shadow_runtime.py -k 'gpt4o or step_zero'
```

Expected: import/API failures for the missing grounder and cadence hook.

- [ ] **Step 3: Implement audited VLM snapshots**

Orient both LIBERO images exactly as `prepare_observation` does. Ask GPT-4o one
closed batch question for the full required fact universe and construct the
existing canonical `FactSnapshot` evidence payload:

```python
{
    "epoch_id": epoch_id,
    "observation_hash": observation_sha256(observation),
    "values": [[fact.pddl(), truth.value], ...],
    "dominance_overrides": [],
}
```

Validate identifiers and exactly-one groups in program code. A required
`UNKNOWN` returns the existing grounding failure status. `force_refresh=True`
must bypass the observation cache for strict confirmation.

Extend `ShadowValidatedProposal.strict_terminal_snapshot_reader` to use that
fresh read. For the GPT backend, sample at the existing
`online_monitor_interval_steps=5`; initialization remains at step 0. Do not
alter scripted/oracle cadence or semantics.

- [ ] **Step 4: Run GREEN tests and commit**

Run:

```bash
uv run pytest -q tests/logiv/test_gpt4o_grounding.py tests/logiv/test_shadow_runtime.py
git diff --check
```

Commit only Task 2 changes.

---

### Task 3: Add GPT-4o initial PDDL proposal and VAL-bound local repair

**Files:**
- Create: `src/pi05_libero_repro/logiv/gpt4o_planning.py`
- Create: `tests/logiv/test_gpt4o_planning.py`
- Modify: `src/pi05_libero_repro/logiv/initial_proposal.py`
- Modify: `src/pi05_libero_repro/logiv/proposal.py`
- Modify: `src/pi05_libero_repro/logiv/evaluation.py`
- Modify: `src/pi05_libero_repro/logiv/repair.py`
- Modify: `src/pi05_libero_repro/logiv/controller.py`
- Modify: `tests/logiv/test_initial_proposal.py`
- Modify: `tests/logiv/test_repair.py`
- Modify: `tests/logiv/test_controller.py`

**Interfaces:**
- Produces: `Gpt4oProposalProvider.propose(..., observation=...)` and
  `Gpt4oRepairOperator.repair(...)` compatible with existing initial/runtime
  call sites.
- Consumes: metadata-only scaffold from the existing proposal fixture, initial
  GPT-4o snapshot, `render_domain_pddl()`, `render_problem_pddl()`, allowed
  schemas, causal slice, retry exclusions, current images, and `ValWrapper`.

- [ ] **Step 1: Write failing initial proposal tests**

Require `run_initial_proposal(..., observation=observation)` to pass the images
to providers that declare the keyword while scripted providers remain backward
compatible. Test that GPT-4o receives the rendered Domain/Problem and returns:

```json
{
  "actions": [
    {"schema":"place-on", "arguments":["moka_pot_1","left_region","stove_region"],
     "instruction":"Place the left moka pot on the stove."}
  ]
}
```

Reject empty plans, unregistered objects, unsupported schemas, wrong arity or
types, duplicate action records, goal fields, DAG fields, and continuous action
fields. Assert official goal and objects stay byte-for-byte equal to metadata.

- [ ] **Step 2: Write failing local repair tests**

Construct a plan with protected prefix/suffix and a causal slice. Assert GPT-4o
sees only the current problem and affected obligations, may return only legal
replacement actions, and program code merges them without changing protected
actions or goal. Require full merged-plan VAL validation before returning
`CERTIFIED`; invalid VAL output is fed back for another model proposal within
the existing VAL-call budget, and exhausted/invalid responses never install a
DAG.

Add controller coverage showing later Node/Graph failures call the same GPT-4o
repair operator and high-level control never returns to BASE.

- [ ] **Step 3: Run RED tests**

Run:

```bash
uv run pytest -q tests/logiv/test_gpt4o_planning.py \
  tests/logiv/test_initial_proposal.py tests/logiv/test_repair.py \
  tests/logiv/test_controller.py -k 'gpt4o or observation or local_repair'
```

- [ ] **Step 4: Implement proposal and repair adapters**

Use the scripted fixture only for frozen BDDL metadata, object declarations,
task instruction, and official goal. Replace its initial facts and every
candidate action with GPT-derived/grounded values; do not read scripted actions
as a fallback.

`Gpt4oRepairOperator` wraps the same `ValWrapper`, retry ledger, causal slice,
edit distance, and global/local budgets. It emits the existing
`RepairResult/PlanCertificate` types so deterministic DAG compilation and atomic
installation remain unchanged.

- [ ] **Step 5: Run GREEN tests and commit**

Run focused tests, then:

```bash
uv run pytest -q tests/logiv/test_initial_proposal.py tests/logiv/test_repair.py \
  tests/logiv/test_controller.py tests/logiv/test_gpt4o_planning.py
git diff --check
```

Commit only Task 3 files.

---

### Task 4: Wire the GPT-4o backend into evaluation, artifacts, and launchers

**Files:**
- Modify: `scripts/eval_logiv_libero.py`
- Modify: `scripts/run_logiv_eval.sh`
- Create: `scripts/run_gpt4o_logiv_dual.sh`
- Create: `scripts/smoke_gpt4o_logiv.py`
- Modify: `src/pi05_libero_repro/logiv/shadow_runtime.py`
- Modify: `src/pi05_libero_repro/logiv/records.py`
- Modify: `tests/logiv/test_evaluator.py`
- Modify: `tests/logiv/test_records.py`
- Modify: `tests/test_launchers.py`

**Interfaces:**
- Produces: `--perception-backend {scripted-oracle,gpt4o}` and
  `--gpt4o-model gpt-4o`; Docker key forwarding; GPT request provenance and
  nonzero request accounting; a two-GPU paired runner.
- Consumes: one shared `Gpt4oClient` per episode across initial planning,
  observation, handoff repair, and later controller repair.

- [ ] **Step 1: Write failing backend/launcher/accounting tests**

Require:

- GPT mode rejects `--oracle-grounding` and missing environment key;
- scripted/oracle mode remains unchanged;
- GPT mode records `oracle_grounding=False`;
- `initial_proposal_requests`, `shadow_vlm_requests`, and
  `recovery_policy_requests` equal the shared client's per-purpose counters;
- schema validation allows those counters only for explicit GPT mode;
- `run_logiv_eval.sh` contains `-e OPENAI_API_KEY` but never expands or echoes
  the key;
- the dual runner defines seeds `7 17`, tasks 0..9 exactly once per shard,
  episode indices `0:50`, and runs BASE before GPT LOGIV against each unchanged
  server port.

- [ ] **Step 2: Run RED tests**

Run:

```bash
uv run pytest -q tests/logiv/test_evaluator.py tests/logiv/test_records.py \
  tests/test_launchers.py -k 'gpt4o or key or dual or accounting'
```

- [ ] **Step 3: Implement evaluator selection and provenance**

Keep default `scripted-oracle` for backward compatibility. In GPT mode, create
one shared client, GPT proposal provider, GPT grounder, and GPT repair operator;
pass the original certified graph and confirmed deviation into the handoff-local
repair provider. Do not call `LiberoOracleGrounder` in GPT mode.

Write `gpt4o_calls.json` per episode using only `Gpt4oCallRecord` values. Do not
write prompts, image bytes, raw response bodies, Authorization headers, or the
environment key.

- [ ] **Step 4: Implement safe Docker forwarding and dual runner**

Add `-e OPENAI_API_KEY` to Docker only when GPT mode is selected and fail before
container launch if the variable is absent. Split tasks as `0,2,4,6,8` on GPU 0
and `1,3,5,7,9` on GPU 1. For each shard and seed, run BASE then GPT-4o LOGIV on
the same long-lived policy-server process and separate fresh output directories.

- [ ] **Step 5: Implement one-call API smoke**

`scripts/smoke_gpt4o_logiv.py` sends a 2x2 synthetic RGB image and requires a
strict `{"status":"OK"}` response. It prints only model, response ID, usage,
latency, and hashes.

- [ ] **Step 6: Run GREEN tests and commit**

Run focused tests, `bash -n` on all changed shell scripts, and `git diff --check`.
Commit Task 4 files only.

---

### Task 5: Verify the complete implementation and GPT-4o capability

**Files:**
- Modify if needed: `README.md`
- Create through runtime: `results/gpt4o-logiv-api-smoke-20260812.json`

**Interfaces:**
- Consumes: all code from Tasks 1-4 and the existing API key.
- Produces: verified unit/integration baseline plus one paid capability smoke.

- [ ] **Step 1: Run the complete local verification suite**

Run:

```bash
uv run pytest -q
bash -n scripts/run_logiv_eval.sh scripts/run_gpt4o_logiv_dual.sh
git diff --check
```

Expected: at least the baseline 801 tests plus new tests, zero failures, silent
shell/diff checks.

- [ ] **Step 2: Run the real API smoke inside the evaluator image**

Run the smoke with the key passed by environment name. Expected: exit 0,
returned model `gpt-4o` or a documented GPT-4o snapshot, strict schema result
`OK`, and no secret in stdout/stderr/artifacts.

- [ ] **Step 3: Run one paired episode smoke per GPU**

Start policy servers on GPU 0/port 8010 and GPU 1/port 8020. On each server run
one BASE episode followed by the matching GPT-4o LOGIV episode with the same
task, episode index, and master seed. Require valid records, initial VAL
certificates, GPT observation requests from step 0, exact no-trigger prefix
parity, and no post-success action.

- [ ] **Step 4: Review failures before scaling**

If either smoke fails, preserve its artifacts, add a failing regression test,
fix only the demonstrated root cause, and repeat the complete verification.

---

### Task 6: Prompt-only tuning and two-GPU 1000-pair evaluation

**Files:**
- Create: `configs/logiv/gpt4o-prompts-v1.json`
- Create only if measured failures require them: `configs/logiv/gpt4o-prompts-v2.json`, `configs/logiv/gpt4o-prompts-v3.json`
- Create through evaluator: `runs/gpt4o-logiv-tuning-20260812/**`
- Create through evaluator: `runs/gpt4o-logiv-final-20260812/**`
- Create through reporter: `results/gpt4o-logiv-final-paired-20260812.json`
- Create through reporter: `results/gpt4o-logiv-final-paired-20260812.md`

**Interfaces:**
- Consumes: fixed LOGIV/pi0.5 parameters, GPT prompt config, tasks 0..9,
  episode indices 0..49, seeds 7/17, two policy servers.
- Produces: one honest fresh 1000-pair GPT-4o development report.

- [ ] **Step 1: Freeze v1 prompts and a predeclared diagnostic subset**

Store the initial, State Gate, and local repair prompts in one versioned JSON
file. Predeclare a balanced diagnostic subset containing at least one current
BASE success and one current BASE failure per task across seeds 7/17. Do not
select examples after seeing GPT output.

- [ ] **Step 2: Run v1 diagnostics concurrently on both GPUs**

Measure initial JSON validity, initial VAL certification, visual fact agreement
with oracle labels as development diagnostics only, strict-confirmation rate,
false interventions, repair VAL certification, negative flips, API latency, and
request count.

- [ ] **Step 3: Tune at most two prompt revisions**

Only change prompt wording or JSON-schema descriptions. Do not change LOGIV
thresholds, pi0.5 prompts, task routing, budgets, or seeds. Select
lexicographically by zero negative flips, higher success, fewer invalid/UNKNOWN
required facts, then lower latency/calls.

- [ ] **Step 4: Launch the full paired two-GPU run**

Use fresh output directories. Each GPU keeps one policy server alive. For its
five tasks, run BASE then selected GPT-4o LOGIV for seed 7 and seed 17, all
episode indices 0..49. Communicate progress at least hourly and preserve partial
append-only results for resume.

- [ ] **Step 5: Generate and audit the final report**

Require exactly 1000 unique paired keys `(seed, task_id, episode_idx)`, BASE 924
only if the fresh run reproduces it, no mixed scripted/oracle LOGIV records, no
record errors, `combined_actions <= 520`, and native success absorption.

Target assertions:

```bash
jq -e '
  .paired_episodes == 1000 and
  .online_successes >= 950 and
  .flips.negative == 0 and
  .errors == []
' results/gpt4o-logiv-final-paired-20260812.json
```

If the target misses, retain and report the actual result and failure taxonomy;
do not cherry-pick episodes, alter the fixed protocol, or merge old oracle
results.

- [ ] **Step 6: Final verification and review**

Run the full test suite again, validate every result/artifact, and review the
implementation against the design spec before making any completion claim.
