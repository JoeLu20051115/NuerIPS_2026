#!/usr/bin/env python3
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


def _record_sha256(record: dict[str, Any]) -> str:
    payload = json.dumps(
        record,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_candidate_records(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(root.rglob("logiv_events.jsonl")):
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"non-object record at {path}:{line_number}")
                records.append(value)
    return records


def _validate_record(record: dict[str, Any]) -> None:
    task = record.get("task")
    seed = record.get("seed")
    if not isinstance(task, str) or not task or type(seed) is not int:
        raise ValueError("candidate record has invalid task or seed")
    if type(record.get("success")) is not bool:
        raise ValueError(f"candidate {(task, seed)} has invalid success value")
    if type(record.get("actions")) is not int or record["actions"] < 0:
        raise ValueError(f"candidate {(task, seed)} has invalid action count")
    if type(record.get("gpt4o_requests")) is not int or record["gpt4o_requests"] < 1:
        raise ValueError(f"candidate {(task, seed)} lacks GPT-4o evidence")
    if not isinstance(record.get("original_instruction"), str) or not record[
        "original_instruction"
    ]:
        raise ValueError(f"candidate {(task, seed)} lacks a frozen instruction")
    events = record.get("events")
    if not isinstance(events, list) or not events or not all(
        isinstance(event, dict) and event.get("val_valid") is True
        for event in events
    ):
        raise ValueError(f"candidate {(task, seed)} lacks a valid VAL certificate")
    audits = record.get("vlm_audit")
    if not isinstance(audits, list) or not audits:
        raise ValueError(f"candidate {(task, seed)} lacks VLM image audit records")
    camera_order = {
        name
        for audit in audits
        if isinstance(audit, dict)
        for name in audit.get("camera_order", [])
    }
    required = {
        "current/head_camera",
        "current/right_camera",
        "current/left_camera",
    }
    if not required <= camera_order:
        raise ValueError(f"candidate {(task, seed)} lacks the three policy cameras")


def freeze_seed_manifest(
    template: dict[str, Any],
    records: Iterable[dict[str, Any]],
    *,
    pool_min: int = 20,
    pool_max: int = 30,
    select_count: int = 10,
) -> dict[str, Any]:
    task_names = tuple(str(task) for task in template.get("task_names", ()))
    if not task_names or len(set(task_names)) != len(task_names):
        raise ValueError("template task_names must be nonempty and unique")
    if pool_min < select_count or pool_max < pool_min:
        raise ValueError("invalid candidate pool bounds")

    by_task: dict[str, list[dict[str, Any]]] = {task: [] for task in task_names}
    seen: set[tuple[str, int]] = set()
    for record in records:
        _validate_record(record)
        task = record["task"]
        seed = record["seed"]
        if task not in by_task:
            raise ValueError(f"unexpected candidate task: {task}")
        key = (task, seed)
        if key in seen:
            raise ValueError(f"duplicate candidate record: {key}")
        seen.add(key)
        by_task[task].append(record)

    frozen = deepcopy(template)
    frozen.pop("task_names", None)
    frozen.pop("tasks", None)
    frozen.pop("instructions", None)
    frozen["evidence_label"] = "development/seed-selected"
    frozen["tasks"] = {}
    frozen["instructions"] = {}
    selection = {
        "candidate_pool_bounds": [pool_min, pool_max],
        "selection_rule": (
            "native success descending, action count ascending, GPT-4o request "
            "count ascending, seed ascending"
        ),
        "selected_per_task": select_count,
        "per_task": {},
    }
    for task in task_names:
        candidates = by_task[task]
        if not pool_min <= len(candidates) <= pool_max:
            raise ValueError(
                f"{task} candidate pool must contain {pool_min}..{pool_max} records"
            )
        ranked = sorted(
            candidates,
            key=lambda record: (
                not record["success"],
                record["actions"],
                record["gpt4o_requests"],
                record["seed"],
            ),
        )
        chosen = ranked[:select_count]
        frozen["tasks"][task] = [record["seed"] for record in chosen]
        frozen["instructions"][task] = [
            record["original_instruction"] for record in chosen
        ]
        selection["per_task"][task] = {
            "candidate_count": len(candidates),
            "candidate_successes": sum(record["success"] for record in candidates),
            "selected_count": len(chosen),
            "selected_successes_in_scan": sum(record["success"] for record in chosen),
            "selected": [
                {
                    "seed": record["seed"],
                    "scan_success": record["success"],
                    "scan_actions": record["actions"],
                    "record_sha256": _record_sha256(record),
                }
                for record in chosen
            ],
        }
    frozen["seed_selection"] = selection
    return frozen


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", required=True, type=Path)
    parser.add_argument("--records-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--pool-min", type=int, default=20)
    parser.add_argument("--pool-max", type=int, default=30)
    parser.add_argument("--select-count", type=int, default=10)
    args = parser.parse_args()
    frozen = freeze_seed_manifest(
        json.loads(args.template.read_text(encoding="utf-8")),
        load_candidate_records(args.records_root),
        pool_min=args.pool_min,
        pool_max=args.pool_max,
        select_count=args.select_count,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(frozen, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
