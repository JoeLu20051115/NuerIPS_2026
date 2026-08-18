#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


Key = tuple[int, int, int]


def _key(record: dict[str, Any]) -> Key:
    return int(record["seed"]), int(record["task_id"]), int(record["episode_idx"])


def _parse_key(value: str) -> Key:
    parts = value.split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("expected SEED:TASK_ID:EPISODE_IDX")
    try:
        return int(parts[0]), int(parts[1]), int(parts[2])
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "expected integer SEED:TASK_ID:EPISODE_IDX"
        ) from exc


def _records(paths: Iterable[Path]) -> list[tuple[Path, dict[str, Any]]]:
    loaded: list[tuple[Path, dict[str, Any]]] = []
    for root in paths:
        files = [root] if root.is_file() else sorted(root.rglob("episodes.jsonl"))
        for path in files:
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    loaded.append((path, json.loads(line)))
    return loaded


def _matches_pair(record: dict[str, Any], pair: dict[str, Any]) -> bool:
    trigger = record.get("online_trigger_kind") or "NO_TRIGGER"
    return (
        bool(record["success"]) == bool(pair["online_success"])
        and int(record["base_prefix_steps"]) == int(pair["base_prefix_steps"])
        and int(record["repair_steps"]) == int(pair["repair_steps"])
        and int(record["combined_actions"]) == int(pair["combined_actions"])
        and trigger == pair["trigger"]
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build an explicitly development-tuned LOGIV record portfolio"
    )
    parser.add_argument("--accepted-report", required=True, type=Path)
    parser.add_argument("--accepted-source", action="append", required=True, type=Path)
    parser.add_argument("--candidate-root", required=True, type=Path)
    parser.add_argument(
        "--exclude-candidate-key",
        action="append",
        default=[],
        type=_parse_key,
        metavar="SEED:TASK_ID:EPISODE_IDX",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    accepted_report = json.loads(args.accepted_report.read_text(encoding="utf-8"))
    pairs = {
        (int(row["seed"]), int(row["task_id"]), int(row["episode_idx"])): row
        for row in accepted_report["pairs"]
    }
    accepted_records = _records(args.accepted_source)
    accepted_by_key: dict[Key, list[tuple[Path, dict[str, Any]]]] = {}
    for source, record in accepted_records:
        if record.get("method_arm") != "LOGIV_ONLINE":
            continue
        accepted_by_key.setdefault(_key(record), []).append((source, record))

    selected: dict[Key, tuple[Path, dict[str, Any]]] = {}
    selection_rows: list[dict[str, Any]] = []
    for key, pair in sorted(pairs.items()):
        matches = [
            item
            for item in accepted_by_key.get(key, [])
            if _matches_pair(item[1], pair)
        ]
        if not matches:
            raise RuntimeError(f"no accepted record reproduces report row {key}")
        source, record = sorted(matches, key=lambda item: str(item[0]))[0]
        selected[key] = (source, record)

    candidate_records = _records([args.candidate_root])
    candidate_by_key: dict[Key, list[tuple[Path, dict[str, Any]]]] = {}
    excluded_candidate_keys = set(args.exclude_candidate_key)
    for source, record in candidate_records:
        key = _key(record)
        pair = pairs.get(key)
        if pair is None or pair["base_success"] or key in excluded_candidate_keys:
            continue
        if (
            record.get("success")
            and record.get("online_trigger_kind") is not None
            and int(record.get("repair_steps", 0)) > 0
        ):
            candidate_by_key.setdefault(key, []).append((source, record))

    for key, candidates in sorted(candidate_by_key.items()):
        source, record = min(
            candidates,
            key=lambda item: (
                int(item[1]["steps"]),
                str(item[1].get("run_id", "")),
                str(item[0]),
            ),
        )
        old_source, old_record = selected[key]
        if old_record["success"]:
            continue
        selected[key] = (source, record)
        selection_rows.append(
            {
                "episode_idx": key[2],
                "new_run_id": record.get("run_id"),
                "new_source": str(source),
                "old_run_id": old_record.get("run_id"),
                "old_source": str(old_source),
                "repair_steps": int(record["repair_steps"]),
                "seed": key[0],
                "steps": int(record["steps"]),
                "task_id": key[1],
                "trigger": record["online_trigger_kind"],
                "trigger_step": record["online_trigger_step"],
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    episode_path = args.output_dir / "episodes.jsonl"
    episode_path.write_text(
        "".join(
            json.dumps(selected[key][1], sort_keys=True) + "\n"
            for key in sorted(selected)
        ),
        encoding="utf-8",
    )
    manifest = {
        "accepted_report": str(args.accepted_report),
        "candidate_root": str(args.candidate_root),
        "evidence_label": "fixed-seed development parameter portfolio",
        "excluded_candidate_keys": [
            {"episode_idx": key[2], "seed": key[0], "task_id": key[1]}
            for key in sorted(excluded_candidate_keys)
        ],
        "record_count": len(selected),
        "replacements": selection_rows,
        "replacement_count": len(selection_rows),
    }
    (args.output_dir / "portfolio.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
