#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict, deque
from pathlib import Path


DEFAULT_INPUT = Path("data/agibot_easy400_for_droid/meta/agibot_easy400_dreamzero_manifest.json")
DEFAULT_OUTPUT = Path("data/agibot_easy400_for_droid/meta/agibot_easy400_dreamzero_manifest_balanced400.json")


def interleave(groups: dict[str, list[dict]]) -> list[dict]:
    queues = {k: deque(v) for k, v in groups.items()}
    ordered_keys = sorted(groups.keys())
    out: list[dict] = []
    while any(queues.values()):
        progressed = False
        for key in ordered_keys:
            if queues[key]:
                out.append(queues[key].popleft())
                progressed = True
        if not progressed:
            break
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a more balanced 400-episode AgiBot manifest from the current easy400 extraction.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--target-episodes", type=int, default=400)
    args = parser.parse_args()

    data = json.loads(args.input.read_text())
    episodes = data["episodes"]

    by_task: dict[str, list[dict]] = defaultdict(list)
    for row in episodes:
        by_task[row["english_task_name"]].append(row)

    quotas = {
        "Clear table in the restaurant": 148,
        "Pack in the supermarket": 148,
        "Restock supermarket items": 52,
        "Stamp the seal ": 52,
    }

    selected: dict[str, list[dict]] = {}
    for task_name, rows in by_task.items():
        limit = quotas.get(task_name, 0)
        if limit > 0:
            selected[task_name] = rows[: min(limit, len(rows))]

    balanced = interleave(selected)
    balanced = balanced[: args.target_episodes]

    payload = {
        "num_episodes": len(balanced),
        "source_manifest": str(args.input),
        "assumption": (
            "Balanced/interleaved version of the local AgiBot easy subset. "
            "Task variety is limited to the 4 task groups currently extracted on disk."
        ),
        "task_quotas": quotas,
        "task_counts": {k: len(v) for k, v in selected.items()},
        "episodes": balanced,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: len(v) for k, v in selected.items()}, ensure_ascii=False, indent=2))
    print(f"wrote {len(balanced)} episodes to {args.output}")


if __name__ == "__main__":
    main()
