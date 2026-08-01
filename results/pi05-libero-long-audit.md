# π₀.₅ LIBERO-Long final audit

Audit date: 2026-08-01 (Asia/Singapore)

## Outcome

Acceptance: **PASS**

| Checkpoint | Successes | Rate | Reference | Accepted interval |
| --- | ---: | ---: | ---: | ---: |
| Full π₀.₅ | 460/500 | **92.0%** | 92.4% | 89.4%–95.4% |
| Early π₀.₅ (2k steps) | 218/500 | **43.6%** | 43% | 38%–48% |

The per-task result is in
[`pi05-libero-long-summary.md`](pi05-libero-long-summary.md), with the same
data in machine-readable form in
[`pi05-libero-long-summary.json`](pi05-libero-long-summary.json).

## Locked inputs

- OpenPI commit: `650c5b0283a49c42784fb5055a0507da2c6d347d`
- LIBERO commit: `f78abd68ee283de9f9be3c8f7e2a9ad60246e95c`
- Evaluator image: `pi05-libero-eval:650c5b0`, image ID
  `sha256:9793b4c42491e6e77864adee82ab90dc9daf0a1a60be10664d32af043bcc82ae`
- Runtime: MuJoCo 3.2.3, robosuite 1.4.1, BDDL 1.0.1
- Full normalization statistics SHA-256:
  `b3a44bb2810436fb62917decaea58bd4d9110255df527dea21e8fd40c960bd84`
- Early normalization statistics SHA-256:
  `eb3724313020a90e3c3e60682d16628f8dc0d7d387c12baf3fedbc54650ac7f5`

The distinct checkpoint-specific normalization statistics were used; they
were not shared or substituted between runs.

## Protocol audit

- 10 LIBERO-10/Long tasks, exactly 50 trials per task and checkpoint.
- Seed 7 and LIBERO's first 50 fixed initial states for every task.
- One uninterrupted OpenPI policy process per checkpoint, starting at JAX
  `key(0)`; no extra client requests were sent to either primary process.
- Native 256-pixel LIBERO rendering, official 180-degree camera rotation, then
  resize/pad to 224×224.
- RGB agent-view and wrist-view images plus the official 8D robot state.
- Ten dummy stabilization steps, 10-action policy horizon, execute/replan every
  five actions, and at most 520 policy-controlled steps.
- Success recorded only from LIBERO's native `done` and `check_success`
  predicates.

## Integrity checks

- Both JSONL logs contain exactly 500 valid, unique records over the complete
  `(task_id 0–9, episode_idx 0–49)` grid.
- All records use seed 7; `success == done == check_success` for all 1,000
  episodes; no `invalid.json` exists.
- Every action statistic and timing value is finite, and every inference count
  equals `ceil(control_steps / 5)`.
- All 1,000 videos decode as H.264 at 224×224. Every decoded frame count is
  exactly equal to the corresponding episode's control-step count.
- All 500 cross-checkpoint episode keys, initial-state SHA-256 hashes, and
  first-frame SHA-256 hashes match exactly.
- Fresh repository test run: `31 passed`.
- Fresh `report_results.py` run and all five payload/run manifest verification
  commands exited with status 0.

## Committed evidence hashes

| Evidence | SHA-256 |
| --- | --- |
| Full checkpoint manifest | `7fcf7ee6e020cbcb67ccad4c7d378b892e4303b0c3f09be59df74dbc1154210d` |
| Early checkpoint manifest | `f9dbb03363ce7e986ecaa8679491543844e371e58752925d73261b18c05232a0` |
| LIBERO-10 dataset manifest | `34793ba5c035b991bd3a506bb2f678180fbb3fa40c4c437aeaf298f3e35c0bb0` |
| Full primary-run manifest | `2290289b4159f0a6801d0ca631de5008ec0f218e499cac36ff0be2dde08ba782` |
| Early primary-run manifest | `ed609696aa713fe278c1615d312e193ce9724dbc865960a83b6cf09309bd343e` |

Large weights, datasets, raw logs, and videos remain outside Git. Their
manifests, source pins, evaluator, launchers, and final reports are committed.

The public early per-task percentages are coarse small-sample references, so
individual 50-trial task rates need not equal them exactly. The requested
500-trial aggregate is 43.6%, within 0.6 percentage points of the 43% reference.
