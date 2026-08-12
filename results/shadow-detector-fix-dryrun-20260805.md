# Shadow detector fix: simulator dry run (2026-08-05)

## Scope

- Simulator: LIBERO, full frozen pi0.5 checkpoint, episode-seeded policy server.
- Method: paired `BASE` / read-only `SHADOW_LOGIV`; no recovery actions and no takeover.
- Selection: ten fresh random `(task, episode, master seed)` cases from the frozen seed manifest.
- Purpose: validate Shadow topology and deviation evidence before enabling R2M control.

## Frozen random cases and paired outcomes

| Case | Task | Episode | Master seed | Policy seed | Simulator seed | Base | Shadow | Steps |
|---|---:|---:|---:|---:|---:|---|---|---:|
| t05-r00 | 5 | 19 | 3036482162 | 2971111349 | 3163201098 | success | success | 202 |
| t05-r01 | 5 | 24 | 233625248 | 797717263 | 842910671 | success | success | 194 |
| t05-r02 | 5 | 36 | 54804909 | 2359768197 | 1128446460 | fail | fail | 204 |
| t05-r03 | 5 | 18 | 3546047300 | 1005999849 | 3849861332 | fail | fail | 169 |
| t05-r04 | 5 | 42 | 1564298395 | 1136844513 | 3754895077 | fail | fail | 155 |
| t08-r00 | 8 | 48 | 768330798 | 1974508897 | 4019552122 | success | success | 414 |
| t08-r01 | 8 | 15 | 3286795451 | 991703280 | 2450184904 | success | success | 470 |
| t08-r02 | 8 | 20 | 2807206097 | 2134786826 | 1744545367 | success | success | 405 |
| t08-r03 | 8 | 46 | 428229111 | 529505684 | 3021668959 | success | success | 367 |
| t08-r04 | 8 | 31 | 1005154013 | 1226178698 | 3887839016 | success | success | 434 |

Paired read-only parity was 10/10: exact Base execution payload equality, identical native outcome,
and `shadow_parity_valid=true`. Both arms scored 7/10 on this deliberately small development sample.

## Root causes and fixes

1. Subsequent monitor snapshots incorrectly used strict exactly-one grounding while an object was
   legitimately held or in flight. This produced 34 snapshot errors across four successful runs.
   Shadow monitoring now uses the advisory partial snapshot: zero locations becomes `UNKNOWN`, while
   conflicting multiple locations still fails closed. Initial proposal grounding remains strict.
2. A place attempt previously started only after its destination effect was already true, so a failed
   release could never create an attempted-effect timeout. The v2 tracker now starts from an attributable
   gripper-open transition after confirmed holding, then waits until the frozen effect deadline.
3. Evidence-monitoring mode did not audit the simulator's ten settling observations. A task could be
   natively successful while the last stored topology still showed `GOAL=BLOCKED`. Both monitor modes
   now project settling snapshots onto the same fixed graph.
4. Artifacts now separate a confirmed subattempt failure from the terminal trajectory judgment via
   `terminal_topology_status`, `terminal_topology_success`, and
   `base_self_recovered_after_confirmed_deviation`.

## Focused simulator reruns

| Case | Native outcome | Snapshot errors | Attempts | Evidence records | Confirmed events | Terminal interpretation |
|---|---|---:|---:|---:|---:|---|
| t05-r02 | fail | 0 | 0 | 2 | 0 | Pick completed; Base stalled while the place node remained active. No release attempt occurred, so only weak progress evidence existed. |
| t05-r00 | success | 0 | 1 | 3 | 0 | During settling, the place node and `GOAL` both reached `COMPLETED`; Shadow now agrees with native success. |
| t08-r01 | success | 0 | 1 | 6 | 1 | A real place subattempt timed out at step 150, but Base later self-recovered and completed the goal at step 470. This event must not itself authorize takeover. |

The Task 8 case is the key permit-gating result: `confirmed_deviation` records historical failure, not
an automatic prediction that the whole episode will fail. R2M must additionally require fresh state,
a capability contract, and a calibrated intervention-advantage gate. In pure Shadow mode it performs
no physical action.

## Verification and current boundary

- Focused simulator reruns preserved the paired Base trajectory.
- Latest unit/integration suite: `469 passed`.
- `git diff --check` passed.
- This detector/topology tranche does not yet exceed Base because it intentionally contains no active
  recovery executor. The next performance-bearing step is terminal local recovery on Base failures,
  using an independent `pi_recover`, followed by effect and protected-invariant verification. The
  existing frozen pi0.5 overlay is only a diagnostic and must not be relabeled as true `LOGIV_R2M`.
