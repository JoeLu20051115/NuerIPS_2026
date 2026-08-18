# RoboTwin 2.0 LOGIV original-instruction oracle (development)

- Result: **60/100 = 60.0%** under the user-authorized historical
  per-seed/config union.
- Protocol: 10 tasks × 10 fixed simulator seeds. All 60 selected records keep
  the exact pi0.5 baseline **episode instruction** for their `(task, seed)`.
- Checkpoint: `rhodes-team-teleai/pi05_TACO_robotwin2_finetuned`;
  `model.safetensors` SHA-256
  `5af5866f0e5f2ca446ee28d935b0dfc07c72031b455a2318e79249b3278ab87a`.
- Environment: RoboTwin 2.0 `demo_clean`, `unseen` instructions.
- Evidence label: **development oracle, not one frozen configuration, not a
  fresh 100-episode run, and not an independent holdout**.

The selection takes at most one historical monitored success per frozen cell;
the lowest action count breaks ties. A record must have RoboTwin native success,
at least one GPT-4o request, monitoring beginning at epoch 0, and a VAL-valid
PDDL plan at every recorded observation.

| Task | Selected successes |
|---|---:|
| `handover_block` | 3/10 |
| `open_microwave` | 2/10 |
| `place_dual_shoes` | 3/10 |
| `stamp_seal` | 8/10 |
| `blocks_ranking_size` | 7/10 |
| `move_can_pot` | 8/10 |
| `turn_switch` | 6/10 |
| `stack_blocks_three` | 5/10 |
| `stack_bowls_three` | 10/10 |
| `beat_block_hammer` | 8/10 |

The comparable single-configuration result remains **45/100**; direct pi0.5
baseline inference is **41/100**. The 60/100 figure is deliberately optimized
on these same 100 development cells and must not be reported as a fair
single-configuration generalization result.

## What “same instruction” means

The frozen original episode instruction matches in all 60 records. LOGIV still
owns the PDDL node schedule and may send a node-specific local prompt to pi0.5.
In particular, the newly recovered `turn_switch/100000` cell has the original
baseline episode instruction but succeeds through the registered PDDL node
prompt. Historical event logs did not store every dispatched policy prompt, so
they cannot prove a stricter requirement that pi0.5 received the baseline text
verbatim at every dispatch. Under that stricter dispatch-prompt interpretation,
the defensible total is **59/100**, not 60/100.

All 60 selected records show the LOGIV monitor active from the first
observation. Across them, 548 planning observations were recorded:
`BASE_MONITORED` 219, `DAG_EXECUTION` 163, and `REPAIR` 166. Twelve selected
episodes enter explicit `REPAIR`, 28 contain `DAG_EXECUTION`, and 32 contain
`BASE_MONITORED` (these episode categories overlap). Thus LOGIV participates in
all 60; this does **not** mean LOGIV causally rescued all 60.

## Provenance and VAL revalidation

The JSON embeds the frozen protocol, all selected records, source paths and
record digests, the full list of 16 contributing run tags, checkpoint/config/
planner hashes, and runtime commit
`8de0ed9520989f9fd156904291d0895b9a361886`. Historical tags predate complete
per-run commit/config provenance, so this is an auditable historical selection,
not a promise that all 16 runs can be independently recreated byte-for-byte.
The current `turn_switch/100000` gap cell is rerunnable from the committed
one-cell config and patch chain.

Historical event records contain validity and certificate fields but omit the
PDDL text. To strengthen that evidence, the report generator reconstructed all
47 unique `(task, ordered facts, truth assignment)` states represented by the
548 event occurrences and ran VAL again. **548/548 occurrences revalidated**
with VAL SHA-256
`24bd37a5cc125923483183359264a69a14358afd1fccdded3aa55289f03ed7dd`.
The generated domain, problem, plan, VAL output, and stable certificate are
embedded for each unique state. These are fresh revalidations; their digests are
not asserted to equal the historical certificates.

## Re-audit

```bash
python - <<'PY'
import importlib.util, json
from pathlib import Path

module_path = Path("scripts/report_robotwin_logiv_oracle.py")
spec = importlib.util.spec_from_file_location("oracle", module_path)
oracle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(oracle)
report = json.loads(Path(
    "results/robotwin-logiv-strict-baseline-instruction-oracle-development-20260813.json"
).read_text())
assert report["successes"] == 60
assert report["baseline_instruction_matches"] == 60
assert report["val_revalidation"]["valid_event_occurrences"] == 548
assert oracle.audit_embedded_report(report) == []
print("PASS: original-instruction 60/100 and 548/548 VAL evidence verified")
PY
```

Rebuild the selection and fresh VAL evidence from the local experiment tree:

```bash
python scripts/report_robotwin_logiv_oracle.py \
  --config configs/robotwin/logiv-pddl-10x10-v1.json \
  --events-root /path/to/NuerIPS_2026/.worktrees \
  --embed-records --require-baseline-instruction \
  --revalidate-val-binary artifacts/tools/val-ubuntu22/Validate \
  --checkpoint-file artifacts/checkpoints/pi05_TACO_robotwin2_finetuned/model.safetensors \
  --runtime-commit 8de0ed9520989f9fd156904291d0895b9a361886 \
  --output results/robotwin-logiv-strict-baseline-instruction-oracle-development-20260813.json
```
