#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict, deque
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE_MANIFEST = REPO_ROOT / "data/agibot_easy400_for_droid/meta/agibot_easy400_dreamzero_manifest.json"
DEFAULT_ADDITIONAL_MANIFEST = REPO_ROOT / "data/agibot_additional4_for_droid/meta/agibot_additional4_dreamzero_manifest.json"
DEFAULT_OUTPUT = REPO_ROOT / "data/agibot_easy400_for_droid/meta/agibot_8task_balanced400_manifest.json"


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path)


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
    parser = argparse.ArgumentParser(description="Build an 8-task balanced 400-episode AgiBot manifest.")
    parser.add_argument("--base-manifest", type=Path, default=DEFAULT_BASE_MANIFEST)
    parser.add_argument("--additional-manifest", type=Path, default=DEFAULT_ADDITIONAL_MANIFEST)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--episodes-per-task", type=int, default=50)
    args = parser.parse_args()

    base = json.loads(args.base_manifest.read_text())["episodes"]
    extra = json.loads(args.additional_manifest.read_text())["episodes"]
    all_rows = base + extra

    by_task: dict[str, list[dict]] = defaultdict(list)
    for row in all_rows:
        by_task[row["english_task_name"]].append(row)

    selected: dict[str, list[dict]] = {}
    for task_name, rows in sorted(by_task.items()):
        if len(rows) < args.episodes_per_task:
            raise RuntimeError(f"Task {task_name} only has {len(rows)} episodes, fewer than requested {args.episodes_per_task}")
        selected[task_name] = rows[: args.episodes_per_task]

    balanced = interleave(selected)
    payload = {
        "num_episodes": len(balanced),
        "episodes_per_task": args.episodes_per_task,
        "source_manifests": [
            _display_path(args.base_manifest),
            _display_path(args.additional_manifest),
        ],
        "task_counts": {k: len(v) for k, v in selected.items()},
        "episodes": balanced,
    }
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(payload["task_counts"], indent=2, ensure_ascii=False))
    print(f"wrote {len(balanced)} episodes to {args.output_path}")


if __name__ == "__main__":
    main()
