#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
META_ROOT = REPO_ROOT / "data/agibot_challenge_meta/Manipulation-SimData"
OUTPUT_ROOT = REPO_ROOT / "data/agibot_heldout1000_for_droid"
TARGET_EPISODES = 1000


def read_lfs_size(pointer_path: Path) -> int:
    size = 0
    for line in pointer_path.read_text(errors="ignore").splitlines():
        if line.startswith("size "):
            size = int(line.split()[1])
            break
    return size


def main() -> None:
    task_dirs = sorted([p for p in META_ROOT.iterdir() if p.is_dir() and (p / "task_train.json").exists()])
    candidates = []

    for task_dir in task_dirs:
        rows = json.loads((task_dir / "task_train.json").read_text())
        counts = Counter(int(r["task_id"]) for r in rows)
        grouped_episodes = defaultdict(list)
        for r in rows:
            grouped_episodes[int(r["task_id"])].append(r)

        grouped_sizes = defaultdict(int)
        grouped_files = defaultdict(list)
        for pointer in sorted(task_dir.glob("*.tgz")):
            task_id = int(pointer.name.replace(".tgz", "").split("_part_")[0])
            grouped_sizes[task_id] += read_lfs_size(pointer)
            grouped_files[task_id].append(pointer.relative_to(REPO_ROOT).as_posix())

        for task_id, count in sorted(counts.items()):
            total_size = grouped_sizes[task_id]
            candidates.append(
                {
                    "task_group": task_dir.name,
                    "task_id": task_id,
                    "num_episodes": count,
                    "total_size_bytes": total_size,
                    "size_per_episode_bytes": total_size / max(count, 1),
                    "archive_paths": grouped_files[task_id],
                    "episodes": grouped_episodes[task_id],
                }
            )

    candidates.sort(key=lambda x: (x["size_per_episode_bytes"], x["total_size_bytes"], x["task_group"], x["task_id"]))

    selected = []
    total_eps = 0
    total_size = 0
    for row in candidates:
        selected.append(row)
        total_eps += row["num_episodes"]
        total_size += row["total_size_bytes"]
        if total_eps >= TARGET_EPISODES:
            break

    output = {
        "assumption": "Held-out with respect to DreamZero-DROID training, because the selected data comes from AgiBot World Challenge Manipulation-SimData rather than DROID.",
        "warning": "This manifest does not prove held-out status for any AgiBot-trained model. It is intended as a cross-dataset held-out subset for DROID-trained models.",
        "target_episodes": TARGET_EPISODES,
        "selected_episode_capacity": total_eps,
        "selected_total_size_bytes": total_size,
        "selected_total_size_gb": round(total_size / 1e9, 2),
        "selected_task_groups": sorted({row["task_group"] for row in selected}),
        "selected_task_ids": [
            {
                "task_group": row["task_group"],
                "task_id": row["task_id"],
                "num_episodes": row["num_episodes"],
                "total_size_gb": round(row["total_size_bytes"] / 1e9, 2),
                "archives": row["archive_paths"],
            }
            for row in selected
        ],
        "camera_note": "Challenge manipulation data is camera-based. Extracted episodes contain MP4 camera videos; use extraction verification after download to confirm per-episode camera presence.",
    }

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "manifest.json").write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")

    flattened = []
    for row in selected:
        for ep in row["episodes"]:
            flattened.append(
                {
                    "task_group": row["task_group"],
                    "task_id": row["task_id"],
                    "episode_id": int(ep["episode_id"]),
                    "english_task_name": ep.get("english_task_name", row["task_group"]),
                    "job_id": ep.get("job_id"),
                    "sn_code": ep.get("sn_code"),
                }
            )
    (OUTPUT_ROOT / "episodes_capacity.json").write_text(json.dumps(flattened, indent=2, ensure_ascii=False) + "\n")

    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
