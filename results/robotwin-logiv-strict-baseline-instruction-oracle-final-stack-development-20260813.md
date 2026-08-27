# RoboTwin LOGIV final-stack oracle gap sweep (development)

- Final result: **63/100 = 63.0%** under the user-authorized historical
  per-seed/config development oracle.
- Previous audited oracle: **60/100**.
- Fresh sweep: **40/40** missing cells completed exactly once; **3** new native
  successes, **33** action-budget failures, and **4** unresolved-visual-fact
  failures.
- Evidence label: **development oracle, not one frozen configuration, not a
  fair generalization result, and not an independent holdout**.

All 63 selected records keep the exact frozen pi0.5 baseline episode
instruction for their `(task, seed)`. The final selection embeds every record,
its source path and digest, checkpoint/config/planner provenance, and the TACO
runtime commit `8de0ed9520989f9fd156904291d0895b9a361886`.

## New cells

| Task | Seed | Actions | GPT-4o observations | Result |
|---|---:|---:|---:|---|
| `turn_switch` | 100008 | 52 | 3 | native success |
| `stack_blocks_three` | 100002 | 517 | 12 | native success |
| `stack_blocks_three` | 100005 | 486 | 11 | native success |

All three have nonempty LOGIV monitoring events, native RoboTwin success, and
VAL-valid recorded observations. No task was retried and no setting was changed
after launch.

## Per-task development oracle

| Task | Selected successes |
|---|---:|
| `handover_block` | 3/10 |
| `open_microwave` | 2/10 |
| `place_dual_shoes` | 3/10 |
| `stamp_seal` | 8/10 |
| `blocks_ranking_size` | 7/10 |
| `move_can_pot` | 8/10 |
| `turn_switch` | 7/10 |
| `stack_blocks_three` | 7/10 |
| `stack_bowls_three` | 10/10 |
| `beat_block_hammer` | 8/10 |

## Failure interpretation

The remaining failures are not primarily caused by unresolved evidence. Among
the 37 failed fresh attempts, **33** ended with `ACTION_BUDGET_EXHAUSTED` and
only **4** with `UNRESOLVED_VISUAL_FACT`. The unresolved cells were all in
`open_microwave`: seeds 100001, 100004, 100007, and 100011. The dominant
bottleneck in this sweep was therefore execution failing to finish within the
action budget, not excessive unresolved retries.

## Fresh VAL audit

The report reconstructed the selected planning states and reran VAL. All
**574/574** selected event occurrences revalidated across 47 unique states.
The VAL binary SHA-256 is
`24bd37a5cc125923483183359264a69a14358afd1fccdded3aa55289f03ed7dd`;
the checkpoint SHA-256 is
`5af5866f0e5f2ca446ee28d935b0dfc07c72031b455a2318e79249b3278ab87a`.

## Re-audit

```bash
uv run python - <<'PY'
import importlib.util
import json
from pathlib import Path

root = Path.cwd()
module_path = root / "scripts" / "report_robotwin_logiv_oracle.py"
spec = importlib.util.spec_from_file_location("oracle", module_path)
assert spec is not None and spec.loader is not None
oracle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(oracle)

report = json.loads((root / "results/robotwin-logiv-strict-baseline-instruction-oracle-final-stack-development-20260813.json").read_text())
assert oracle.audit_embedded_report(report) == []
assert report["successes"] == 63
assert report["baseline_instruction_matches"] == 63
assert report["val_revalidation"]["event_occurrences"] == 574
assert report["val_revalidation"]["valid_event_occurrences"] == 574
print("PASS: 63/100 development oracle and 574/574 VAL evidence verified")
PY
```

Rebuild the report from the local event tree:

```bash
uv run python scripts/report_robotwin_logiv_oracle.py \
  --config configs/robotwin/logiv-pddl-10x10-v1.json \
  --events-root /mnt/data3/data_xingrui/lueq/NuerIPS_2026/.worktrees \
  --embed-records --require-baseline-instruction \
  --revalidate-val-binary /mnt/data3/data_xingrui/lueq/NuerIPS_2026/artifacts/tools/val-ubuntu22/Validate \
  --checkpoint-file /mnt/data3/data_xingrui/lueq/NuerIPS_2026/artifacts/checkpoints/pi05_TACO_robotwin2_finetuned/model.safetensors \
  --runtime-commit 8de0ed9520989f9fd156904291d0895b9a361886 \
  --output results/robotwin-logiv-strict-baseline-instruction-oracle-final-stack-development-20260813.json
```
