#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import imageio.v2 as imageio


REPO_ROOT = Path(__file__).resolve().parents[2]
EXTRACT_ROOT = REPO_ROOT / "data/agibot_easy400_for_droid/extracted"
OUTPUT_ROOT = REPO_ROOT / "data/agibot_easy400_for_droid/meta"


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path)


def count_frames(video_path: Path) -> int:
    reader = imageio.get_reader(str(video_path))
    try:
        return reader.count_frames()
    finally:
        reader.close()


def load_h5_shapes(h5_path: Path) -> dict:
    with h5py.File(h5_path, "r") as f:
        return {
            "timestamps": int(f["timestamp"].shape[0]),
            "state_joint_shape": list(f["state/joint/position"].shape),
            "action_joint_shape": list(f["action/joint/position"].shape),
            "right_effector_shape": list(f["state/right_effector/position"].shape),
            "left_effector_shape": list(f["state/left_effector/position"].shape),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extract-root", type=Path, default=EXTRACT_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--manifest-name", type=str, default="agibot_easy400_dreamzero_manifest.json")
    parser.add_argument("--dataset-root-label", type=str, default="data/agibot_easy400_for_droid/extracted")
    parser.add_argument("--adapter-type", type=str, default="dreamzero_droid_right_arm_proxy")
    parser.add_argument("--fast-camera-frames", action="store_true", help="Use trajectory length as camera frame count instead of counting video frames.")
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    rows = []

    for episode_dir in sorted(args.extract_root.glob("*/*/*/*/*")):
        if not episode_dir.is_dir():
            continue
        h5_path = episode_dir / "aligned_joints.h5"
        data_info = episode_dir / "data_info.json"
        meta_info = episode_dir / "meta_info.json"
        head = episode_dir / "head_color.mp4"
        hand_right = episode_dir / "hand_right_color.mp4"
        hand_left = episode_dir / "hand_left_color.mp4"
        required = [h5_path, data_info, meta_info, head, hand_right, hand_left]
        if not all(p.exists() for p in required):
            continue

        data = json.loads(data_info.read_text())
        meta = json.loads(meta_info.read_text())
        shapes = load_h5_shapes(h5_path)
        action_cfg = data.get("label_info", {}).get("action_config", [])
        active_start = min((int(x.get("start_frame", 0)) for x in action_cfg), default=0)
        active_end = max((int(x.get("end_frame", shapes["timestamps"] - 1)) for x in action_cfg), default=shapes["timestamps"] - 1)
        action_plan = [x.get("english_action_text", "").strip() for x in action_cfg if x.get("english_action_text")]
        row = {
            "episode_id": str(data["episode_id"]),
            "task_group": episode_dir.parts[-5],
            "task_id": int(data["task_id"]),
            "job_id": int(data.get("job_id", 0)),
            "sn_code": str(data.get("sn_code", "")),
            "english_task_name": data.get("english_task_name", ""),
            "relative_episode_dir": _display_path(episode_dir),
            "camera_paths": {
                "head": _display_path(head),
                "hand_right": _display_path(hand_right),
                "hand_left": _display_path(hand_left),
            },
            "camera_frames": {
                "head": shapes["timestamps"] if args.fast_camera_frames else count_frames(head),
                "hand_right": shapes["timestamps"] if args.fast_camera_frames else count_frames(hand_right),
                "hand_left": shapes["timestamps"] if args.fast_camera_frames else count_frames(hand_left),
            },
            "duration_sec": float(meta.get("duration", 0.0)),
            "camera_list": meta.get("camera_list", []),
            "fps": meta.get("fps", {}),
            "h5_path": _display_path(h5_path),
            "data_info_path": _display_path(data_info),
            "meta_info_path": _display_path(meta_info),
            "num_steps": shapes["timestamps"],
            "active_frame_start": max(0, min(active_start, shapes["timestamps"] - 1)),
            "active_frame_end": max(0, min(active_end, shapes["timestamps"] - 1)),
            "action_plan": action_plan,
            "state_joint_shape": shapes["state_joint_shape"],
            "action_joint_shape": shapes["action_joint_shape"],
            "right_arm_joint_slice": [7, 14],
            "left_arm_joint_slice": [0, 7],
            "gripper_key": "state/right_effector/position",
            "action_gripper_key": "action/right_effector/position",
            "dreamzero_camera_mapping": {
                "observation/exterior_image_0_left": "head_color.mp4",
                "observation/exterior_image_1_left": "hand_left_color.mp4",
                "observation/wrist_image_left": "hand_right_color.mp4",
            },
            "has_all_cameras": True,
            "valid_for_dreamzero_adapter": True,
        }
        rows.append(row)

    rows.sort(key=lambda r: (r["task_group"], r["task_id"], int(r["episode_id"])))
    manifest = {
        "num_episodes": len(rows),
        "assumption": "Right-arm-only adapter using joint indices [7:14] and right_effector position as gripper proxy.",
        "camera_mapping": {
            "head_color.mp4": "observation/exterior_image_0_left",
            "hand_left_color.mp4": "observation/exterior_image_1_left",
            "hand_right_color.mp4": "observation/wrist_image_left",
        },
        "episodes": rows,
    }
    manifest_path = args.output_root / args.manifest_name
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    with (args.output_root / "episodes.jsonl").open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    info = {
        "dataset_root": args.dataset_root_label,
        "manifest": _display_path(manifest_path),
        "num_episodes": len(rows),
        "adapter_type": args.adapter_type,
    }
    (args.output_root / "info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(info, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
