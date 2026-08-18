#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
META_ROOT = REPO_ROOT / "data/agibot_challenge_meta/Manipulation-SimData"
OUTPUT_ROOT = REPO_ROOT / "data/agibot_easy400_for_droid"

# Chosen for DreamZero-DROID sanity/generalization evaluation:
# shorter manipulation routines, clear camera evidence, lower bytes/episode,
# and simpler object placement / sealing tasks than microwave/freezer/sandwich.
SELECTED = [
    ("clear_table_in_the_restaurant", 2810196),
    ("clear_table_in_the_restaurant", 2810125),
    ("clear_table_in_the_restaurant", 2810129),
    ("pack_in_the_supermarket", 2810128),
    ("pack_in_the_supermarket", 2810137),
    ("pack_in_the_supermarket", 2810138),
    ("restock_supermarket_items", 2810229),
    ("stamp_the_seal", 2810131),
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
        for pointer in sorted(task_dir.glob("*.tgz")):
            pointer_task_id = int(pointer.name.replace(".tgz", "").split("_part_")[0])
            if pointer_task_id != task_id:
                continue
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
        "assumption": "Held-out with respect to DreamZero-DROID training because the selected data comes from AgiBot World Challenge Manipulation-SimData instead of DROID.",
        "warning": "This does not prove held-out status for AgiBot-trained models. It is a DreamZero-DROID-friendly cross-dataset subset.",
        "selection_rationale": "Prefer shorter, camera-clear, lower-bytes-per-episode tasks that resemble simple pick-place / packing / sealing routines.",
        "selected_episode_capacity": total_eps,
        "selected_total_size_bytes": total_size,
        "selected_total_size_gb": round(total_size / 1e9, 2),
        "selected_task_ids": selected_rows,
        "camera_note": "Manipulation challenge data contains MP4 camera videos per extracted episode; verify after extraction.",
    }
    (OUTPUT_ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    (OUTPUT_ROOT / "episodes_capacity.json").write_text(json.dumps(selected_episode_rows, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
