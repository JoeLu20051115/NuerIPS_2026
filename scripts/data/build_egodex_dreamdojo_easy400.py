from __future__ import annotations

import json
from pathlib import Path

import h5py


ROOT = Path("/home/xingrui/lueq/NuerIPS_2026")
SOURCE_ROOT = ROOT / "data" / "egodex_test" / "extracted" / "test"
OUTPUT_ROOT = ROOT / "data" / "egodex_dreamdojo_easy400"
OUTPUT_META = OUTPUT_ROOT / "meta"


TASK_SPECS = [
    ("basic_pick_place", 90, "Pick up an object and place it at the target location."),
    ("vertical_pick_place", 80, "Pick up an object and place it vertically at the target location."),
    ("stack_unstack_plates", 50, "Stack or unstack the plates at the target location."),
    (
        "insert_remove_furniture_bench_cabinet",
        50,
        "Insert or remove the object from the cabinet on the furniture bench.",
    ),
    ("insert_remove_usb", 40, "Insert or remove the USB device from the port."),
    (
        "open_close_insert_remove_tupperware",
        35,
        "Open or close the tupperware container and insert or remove the object.",
    ),
    ("add_remove_lid", 25, "Add or remove the lid from the container."),
    ("pick_place_food", 30, "Pick up the food item and place it at the target location."),
]


def _sorted_episode_ids(task_dir: Path) -> list[str]:
    ids = {p.stem for p in task_dir.glob("*.mp4")} & {p.stem for p in task_dir.glob("*.hdf5")}
    return sorted(ids, key=lambda x: int(x))


def _num_frames(h5_path: Path) -> int:
    with h5py.File(h5_path, "r") as f:
        return int(f["transforms"]["camera"].shape[0])


def main() -> None:
    OUTPUT_META.mkdir(parents=True, exist_ok=True)

    episodes = []
    task_counts = {}

    for task_group, target_count, description in TASK_SPECS:
        task_dir = SOURCE_ROOT / task_group
        if not task_dir.exists():
            raise FileNotFoundError(f"Missing EgoDex task directory: {task_dir}")

        episode_ids = _sorted_episode_ids(task_dir)
        if len(episode_ids) < target_count:
            raise ValueError(
                f"Task {task_group} only has {len(episode_ids)} episodes, "
                f"but {target_count} were requested."
            )

        picked = episode_ids[:target_count]
        task_counts[task_group] = len(picked)

        for eid in picked:
            h5_path = task_dir / f"{eid}.hdf5"
            mp4_path = task_dir / f"{eid}.mp4"
            episodes.append(
                {
                    "episode_id": f"{task_group}/{eid}",
                    "task_group": task_group,
                    "english_task_name": description,
                    "mp4_path": str(mp4_path.relative_to(SOURCE_ROOT)),
                    "h5_path": str(h5_path.relative_to(SOURCE_ROOT)),
                    "num_frames": _num_frames(h5_path),
                }
            )

    manifest = {
        "source": "data/egodex_test/extracted/test",
        "selection_policy": "DreamDojo-friendly easy subset emphasizing static end-state tasks.",
        "num_tasks": len(TASK_SPECS),
        "target_episodes": 400,
        "num_episodes": len(episodes),
        "task_counts": task_counts,
        "episodes": episodes,
    }

    out_path = OUTPUT_META / "manifest.json"
    out_path.write_text(json.dumps(manifest, indent=2))

    readme = OUTPUT_ROOT / "README.txt"
    readme.write_text(
        "\n".join(
            [
                "EgoDex DreamDojo Easy400",
                "",
                "This subset is built from data/egodex_test/extracted/test.",
                "It favors static end-state tasks that are easier for DreamDojo to judge and compare:",
                "basic_pick_place, vertical_pick_place, stack_unstack_plates,",
                "insert_remove_furniture_bench_cabinet, insert_remove_usb,",
                "open_close_insert_remove_tupperware, add_remove_lid, pick_place_food.",
                "",
                f"Manifest: {out_path}",
            ]
        )
    )

    print(f"Wrote {len(episodes)} episodes across {len(TASK_SPECS)} tasks to {out_path}")


if __name__ == "__main__":
    main()
