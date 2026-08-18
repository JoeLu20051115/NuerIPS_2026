from __future__ import annotations

from collections import defaultdict
import math
import re
from typing import Iterable

from pi05_libero_repro.records import EpisodeRecord, wilson_interval


DISPLAY_ORDER = (1, 3, 2, 4, 8, 7, 9, 0, 5, 6)
TASK_LABELS = {
    0: "Alphabet soup + Tomato sauce → Basket",
    1: "Cream cheese + Butter → Basket",
    2: "Turn on stove + Moka pot",
    3: "Black bowl → Bottom drawer + Close",
    4: "Two mugs → Left/Right plates",
    5: "Book → Back compartment of caddy",
    6: "Mug → Plate + Chocolate pudding right of plate",
    7: "Alphabet soup + Cream cheese → Basket",
    8: "Both moka pots → Stove",
    9: "Mug → Microwave + Close",
}
EARLY_PUBLIC_RATES = {0: 0.40, 1: 0.60, 2: 0.53, 3: 0.40, 4: 0.07, 5: 0.67, 6: 0.53, 7: 0.80, 8: 0.20, 9: 0.13}
TARGETS = {
    "full": {"reference": 0.924, "minimum": 0.894, "maximum": 0.954},
    "early": {"reference": 0.43, "minimum": 0.38, "maximum": 0.48},
}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _statistics(records: Iterable[EpisodeRecord]) -> dict:
    records = list(records)
    trials = len(records)
    successes = sum(record.success for record in records)
    rate = successes / trials if trials else None
    interval = list(wilson_interval(successes, trials)) if trials else [None, None]
    return {"successes": successes, "trials": trials, "rate": rate, "wilson_95": interval}


def build_report(records: Iterable[EpisodeRecord]) -> dict:
    records = list(records)
    errors = []
    grouped = defaultdict(list)
    paired = defaultdict(dict)
    seen = set()

    for record in records:
        if record.key in seen:
            errors.append({"code": "duplicate_episode", "episode": record.key_text})
        seen.add(record.key)
        if record.checkpoint not in TARGETS:
            errors.append({"code": "unknown_checkpoint", "episode": record.key_text})
            continue
        if record.task_id not in TASK_LABELS:
            errors.append({"code": "unknown_task", "episode": record.key_text})
            continue
        grouped[(record.checkpoint, record.task_id)].append(record)
        paired[(record.task_id, record.episode_idx)][record.checkpoint] = record
        if record.seed != 7:
            errors.append({"code": "seed_mismatch", "episode": record.key_text, "actual": record.seed})
        if record.episode_idx < 0 or record.episode_idx >= 50:
            errors.append({"code": "episode_index_mismatch", "episode": record.key_text})
        if not _SHA256.fullmatch(record.init_state_sha256) or not _SHA256.fullmatch(
            record.first_frame_sha256
        ):
            errors.append({"code": "invalid_hash", "episode": record.key_text})
        if not record.valid:
            errors.append({"code": "invalid_episode", "episode": record.key_text})
        if record.exception is not None:
            errors.append({"code": "episode_exception", "episode": record.key_text})
        if record.success != record.done or record.done != record.check_success:
            errors.append({"code": "predicate_mismatch", "episode": record.key_text})
        if not all(math.isfinite(value) for value in (record.action_min, record.action_max, record.action_mean)):
            errors.append({"code": "nonfinite_action", "episode": record.key_text})
        if record.inference_requests != math.ceil(record.steps / 5):
            errors.append({"code": "control_horizon_mismatch", "episode": record.key_text})

    for (task_id, episode_idx), pair in paired.items():
        if set(pair) != {"full", "early"}:
            continue
        full = pair["full"]
        early = pair["early"]
        if full.init_state_sha256 != early.init_state_sha256:
            errors.append(
                {"code": "init_state_mismatch", "task_id": task_id, "episode_idx": episode_idx}
            )
        if full.first_frame_sha256 != early.first_frame_sha256:
            errors.append(
                {"code": "first_frame_mismatch", "task_id": task_id, "episode_idx": episode_idx}
            )

    checkpoint_reports = {}
    for checkpoint, target in TARGETS.items():
        checkpoint_records = [record for record in records if record.checkpoint == checkpoint]
        stats = _statistics(checkpoint_records)
        if stats["trials"] != 500:
            errors.append(
                {"code": "count_mismatch", "checkpoint": checkpoint, "actual": stats["trials"], "expected": 500}
            )
        for task_id in range(10):
            count = len(grouped[(checkpoint, task_id)])
            if count != 50:
                errors.append(
                    {
                        "code": "count_mismatch",
                        "checkpoint": checkpoint,
                        "task_id": task_id,
                        "actual": count,
                        "expected": 50,
                    }
                )
        in_range = (
            stats["trials"] == 500
            and target["minimum"] <= stats["rate"] <= target["maximum"]
        )
        if stats["trials"] == 500 and not in_range:
            errors.append(
                {
                    "code": "rate_out_of_range",
                    "checkpoint": checkpoint,
                    "actual": stats["rate"],
                    "minimum": target["minimum"],
                    "maximum": target["maximum"],
                }
            )
        checkpoint_reports[checkpoint] = {**stats, **target, "accepted": in_range}

    tasks = []
    for task_id in DISPLAY_ORDER:
        tasks.append(
            {
                "task_id": task_id,
                "task": TASK_LABELS[task_id],
                "full": _statistics(grouped[("full", task_id)]),
                "early": _statistics(grouped[("early", task_id)]),
                "early_public_rate": EARLY_PUBLIC_RATES[task_id],
            }
        )
    return {"accepted": not errors, "errors": errors, "checkpoints": checkpoint_reports, "tasks": tasks}


def _rate(stats: dict) -> str:
    if not stats["trials"]:
        return "—"
    return f'{stats["successes"]}/{stats["trials"]} ({stats["rate"] * 100:.1f}%)'


def render_markdown(report: dict) -> str:
    lines = [
        "# π₀.₅ LIBERO-Long Reproduction",
        "",
        f'Acceptance: **{"PASS" if report["accepted"] else "FAIL"}**',
        "",
        "| # | LIBERO-Long task | Full π₀.₅ | Early π₀.₅ | Early public |",
        "| ---: | --- | ---: | ---: | ---: |",
    ]
    for index, row in enumerate(report["tasks"], start=1):
        lines.append(
            f'| {index} | {row["task"]} | {_rate(row["full"])} | {_rate(row["early"])} | '
            f'{row["early_public_rate"] * 100:.0f}% |'
        )
    lines.extend(
        [
            f'| — | **Overall** | **{_rate(report["checkpoints"]["full"])}** | '
            f'**{_rate(report["checkpoints"]["early"])}** | **43%** |',
            "",
        ]
    )
    if report["errors"]:
        lines.append("## Audit errors")
        lines.append("")
        lines.extend(f"- `{error['code']}`: `{error}`" for error in report["errors"])
        lines.append("")
    return "\n".join(lines)
