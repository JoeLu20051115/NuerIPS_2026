# LOGIV R2M Phase 0 paired smoke verification

Date executed: 2026-08-04 (Asia/Singapore)

Implementation commits:

- `854417de feat(logiv): add auditable shadow data collection arm`
- `4628f0a3 fix(logiv): admit canonical LIBERO observation keys`

## Static and unit verification

- `git diff --check`: passed.
- `bash -n scripts/run_logiv_eval.sh`: passed.
- `uv run python -m compileall -q src scripts`: passed.
- `uv run pytest -q`: **379 passed in 2.17s** after the smoke-discovered observation-key fix.

The focused recovery-record, LIBERO-adapter, and Shadow-runtime regression suite passed 117 tests. In particular, Task 4's `test_raw_candidate_and_stale_without_fresh_label_are_not_training_eligible` rejects anomaly candidates and stale roots without an exactly matching fresh label.

## Real paired Task 8 episode

Both arms used task 8, episode 0, master seed 7, the full checkpoint, policy server port 8010, and policy episode seed `2668564155`.

| Arm | Terminal status | Steps | Base requests | Native success | Valid row |
| --- | --- | ---: | ---: | --- | --- |
| BASE | `EPISODE_SUCCESS` | 420 | 84 | true | true |
| SHADOW_LOGIV | `EPISODE_SUCCESS` | 420 | 84 | true | true |

The first Shadow attempt correctly failed open because the audit-layer observation-key grammar excluded LIBERO's legitimate `robot0_proprio-state` key. Its complete artifact is preserved at `runs/r2m-phase0-shadow-task8-seed0-rejected-observation-key`. The minimal regression fix admitted hyphens while continuing to reject path-like keys, passed the full suite, and the paired Shadow smoke was then rerun from a fresh output directory.

## Base/Shadow parity

The prescribed ten-field JSON diff produced no output and exited 0. Both artifacts contain:

| Field | Shared value |
| --- | --- |
| `steps` | `420` |
| `base_policy_requests` | `84` |
| `done_signal` | `true` |
| `post_settling_success` | `true` |
| `initial_state_sha256` | `61a34575b44313156895ce73996cf18dd243c396572fd0e1cd476e0e9fbe70af` |
| `base_prompt_sha256` | `28f14f4b739c4a8e89fe0fb5e95f489ec7822cd03a82373cd9352e2b9ed40529` |
| `base_checkpoint_sha256` | `7fcf7ee6e020cbcb67ccad4c7d378b892e4303b0c3f09be59df74dbc1154210d` |
| `policy_client_config_sha256` | `5fb4116a1c920d94b16f575a0a225e0c60676f755502e38ad41c23997242cb9f` |
| `request_envelope_log_sha256` | `e83f11e9c22743452e8ea7ba375e95c68f9f2bb0567f55c0f21a6436ddef1179` |
| `actions_sha256` | `7f2464e7dcefce28bc1e31ccfa8e01e87157879b96ba6a3c4082760a41b56b91` |

## Shadow proposal, monitor, and compute

- Initial proposal: `ACCEPTED`, one request, 0.140769 seconds.
- Certificate: `1761fa6d4fda571dc82106e1c1c6e9dd668c40a4a906c4cdf1d1eb511fd93598`.
- Initial graph: `38920d646bfa294912bcf5559b7e33fa54180801301c4687718dc8976fca98ac`.
- Shadow callbacks: 421; audited snapshots: 85; monitor seconds excluding proposal: 2.216020.
- `shadow_parity_valid=true`.
- Protocol callback, proposal callback, provenance, snapshot, event tracker, evidence overflow, trigger callback, and root-writer error buckets were all zero.
- Base policy requests: 84; initial proposal requests: 1; Shadow VLM requests: 0; recovery policy requests: 0.
- The initial certificate became diagnostically stale during the native trajectory (`stale_certificates=1`); this did not authorize control, emit a root by itself, or affect Base parity.
- The monitor observed no stable deviation: anomaly candidates 0, confirmed deviations 0, and recovery roots 0. Therefore no recovery dataset validator was required for this episode. The deterministic root round-trip and split/leakage validators remain covered by the unit suite.

Raw recovery roots were not passed to a trainer.

**Phase 0 collected development evidence only; recovery capability and takeover are not enabled.**
