#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
META_ROOT = REPO_ROOT / "data/agibot_challenge_meta/Manipulation-SimData"
OUTPUT_ROOT = REPO_ROOT / "data/agibot_additional4_for_droid"

SELECTED = [
    ("clear_the_countertop_waste", 2810232),
    ("open_drawer_and_store_items", 2810187),
    ("pack_moving_objects_from_conveyor", 2810079),
    ("heat_the_food_in_the_microwave", 2600003),
]


def read_lfs_size(pointer_path: Path) -> int:
    for line in pointer_path.read_text(errors="ignore").splitlines():
        if line.startswith("size "):
            return int(line.split()[1])
    return 0


def main() -> None:
    selected_rows = []
    selected_episode_rows = []
    total_eps = 0
    total_size = 0

    for task_group, task_id in SELECTED:
        task_dir = META_ROOT / task_group
        rows = json.loads((task_dir / "task_train.json").read_text())
        group_rows = [r for r in rows if int(r["task_id"]) == task_id]

        size = 0
        archives = []
        for pointer in sorted(task_dir.glob(f"{task_id}*.tgz")):
            size += read_lfs_size(pointer)
            archives.append(pointer.relative_to(REPO_ROOT).as_posix())

        selected_rows.append(
            {
                "task_group": task_group,
                "task_id": task_id,
                "num_episodes": len(group_rows),
                "total_size_bytes": size,
                "total_size_gb": round(size / 1e9, 2),
                "archives": archives,
            }
        )
        total_eps += len(group_rows)
        total_size += size

        for ep in group_rows:
            selected_episode_rows.append(
                {
                    "task_group": task_group,
                    "task_id": task_id,
                    "episode_id": int(ep["episode_id"]),
                    "english_task_name": ep.get("english_task_name", task_group),
                    "job_id": ep.get("job_id"),
                    "sn_code": ep.get("sn_code"),
                }
            )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    manifest = {
        "assumption": "Additional AgiBot task groups for DreamZero-DROID cross-dataset evaluation.",
        "warning": "These tasks are not proven held-out for AgiBot-trained models; they are held-out only with respect to DreamZero-DROID training on DROID.",
        "selection_rationale": (
            "Add four more task groups with at least ~50 episodes each to increase task diversity beyond the initial easy400 subset."
        ),
        "selected_episode_capacity": total_eps,
        "selected_total_size_bytes": total_size,
        "selected_total_size_gb": round(total_size / 1e9, 2),
        "selected_task_ids": selected_rows,
        "camera_note": "MP4 camera videos and trajectory files must be verified after extraction.",
    }
    (OUTPUT_ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    (OUTPUT_ROOT / "episodes_capacity.json").write_text(json.dumps(selected_episode_rows, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
