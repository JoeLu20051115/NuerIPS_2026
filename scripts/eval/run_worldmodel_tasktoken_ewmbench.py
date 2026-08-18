#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
import uuid
from pathlib import Path

import h5py
import imageio.v2 as imageio
import numpy as np
from PIL import Image

try:
    from eval_utils.policy_client import WebsocketClientPolicy
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("policy_client is required") from exc


DEFAULT_INPUT_ROOT = Path("data/agibot_worldmodel_2026/extracted/validation/info_dataset")
DEFAULT_GT_ROOT = Path("data/agibot_worldmodel_2026/extracted/validation/gt_dataset")
DEFAULT_OUTPUT_ROOT = Path("results/dreamzero_worldmodel_val_task_token_only/ACWM_dataset")
DEFAULT_GENERATED_VIDEO_DIR = Path("checkpoints/real_world_eval_gen_worldmodel_20260320_0/DreamZero-DROID")


def _load_image(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def _resize_frame(frame: np.ndarray, width: int = 320, height: int = 180) -> np.ndarray:
    if frame.shape[1] == width and frame.shape[0] == height:
        return frame
    return np.asarray(Image.fromarray(frame).resize((width, height), Image.BILINEAR))


def _find_generated_video(generated_video_dir: Path, session_id: str, existing_files: set[str]) -> Path:
    candidates = sorted(
        generated_video_dir.glob(f"*{session_id}*.mp4"),
        key=lambda p: p.stat().st_mtime,
    )
    if candidates:
        return candidates[-1]
    fresh_candidates = sorted(
        [p for p in generated_video_dir.glob("*.mp4") if p.name not in existing_files],
        key=lambda p: p.stat().st_mtime,
    )
    if fresh_candidates:
        return fresh_candidates[-1]
    raise FileNotFoundError(f"No generated video found for session_id={session_id} in {generated_video_dir}")


def _read_video_frames(video_path: Path) -> list[np.ndarray]:
    reader = imageio.get_reader(str(video_path))
    frames: list[np.ndarray] = []
    try:
        for frame in reader:
            frames.append(np.asarray(frame))
    finally:
        reader.close()
    if not frames:
        raise RuntimeError(f"No frames decoded from {video_path}")
    return frames


def _resample_frames(frames: list[np.ndarray], target_count: int) -> list[np.ndarray]:
    if target_count <= 0:
        return frames
    if len(frames) == target_count:
        return frames
    idx = np.linspace(0, len(frames) - 1, target_count, dtype=int)
    return [frames[i] for i in idx]


def _save_frames_as_jpg(frames: list[np.ndarray], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for idx, frame in enumerate(frames):
        imageio.imwrite(output_dir / f"frame_{idx:05d}.jpg", frame, quality=95)


def _count_gt_frames(gt_episode_dir: Path) -> int:
    return len(sorted((gt_episode_dir / "video").glob("frame_*.png")))


def _load_state_sequence(h5_path: Path) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(h5_path, "r") as f:
        joint_position = f["state/joint/position"][:]
        effector_position = f["state/effector/position"][:]

    if joint_position.ndim != 2 or joint_position.shape[1] < 14:
        raise ValueError(f"Unexpected joint position shape in {h5_path}: {joint_position.shape}")
    if effector_position.ndim != 2 or effector_position.shape[1] < 1:
        raise ValueError(f"Unexpected effector position shape in {h5_path}: {effector_position.shape}")

    right_joint = joint_position[:, 7:14].astype(np.float64)
    right_gripper = effector_position[:, -1:].astype(np.float64)
    return right_joint, right_gripper


def _build_obs(frame: np.ndarray, joint_position: np.ndarray, gripper_position: np.ndarray, prompt: str, session_id: str) -> dict:
    frame = _resize_frame(frame)
    return {
        "observation/exterior_image_0_left": frame,
        "observation/exterior_image_1_left": frame,
        "observation/wrist_image_left": frame,
        "observation/joint_position": joint_position,
        "observation/gripper_position": gripper_position,
        "prompt": prompt,
        "session_id": session_id,
    }


def iter_episodes(input_root: Path) -> list[tuple[str, str, Path]]:
    items: list[tuple[str, str, Path]] = []
    for task_dir in sorted(p for p in input_root.iterdir() if p.is_dir()):
        for episode_dir in sorted(p for p in task_dir.iterdir() if p.is_dir()):
            items.append((task_dir.name, episode_dir.name, episode_dir))
    return items


def run_episode(
    policy: WebsocketClientPolicy,
    generated_video_dir: Path,
    task_id: str,
    episode_id: str,
    episode_dir: Path,
    gt_root: Path,
    output_root: Path,
    prompt_template: str,
    sleep_after_reset: float,
    max_state_steps: int,
) -> dict:
    prompt = prompt_template.format(task_id=task_id, episode_id=episode_id)
    init_frame = _load_image(episode_dir / "frame.png")
    joint_sequence, gripper_sequence = _load_state_sequence(episode_dir / "proprio_stats.h5")
    if max_state_steps > 0:
        joint_sequence = joint_sequence[:max_state_steps]
        gripper_sequence = gripper_sequence[:max_state_steps]
    gt_count = _count_gt_frames(gt_root / task_id / episode_id)
    session_id = f"wm_tasktoken_{task_id}_{episode_id}_{uuid.uuid4().hex[:8]}"
    existing_videos = {p.name for p in generated_video_dir.glob("*.mp4")}

    policy.reset({"session_id": session_id})
    for step_idx in range(len(joint_sequence)):
        obs = _build_obs(
            frame=init_frame,
            joint_position=joint_sequence[step_idx],
            gripper_position=gripper_sequence[step_idx],
            prompt=prompt,
            session_id=session_id,
        )
        _ = policy.infer(obs)

    policy.reset({"session_id": session_id})
    if sleep_after_reset > 0:
        time.sleep(sleep_after_reset)

    generated_video = _find_generated_video(generated_video_dir, session_id, existing_videos)
    pred_frames = _read_video_frames(generated_video)
    pred_frames = _resample_frames(pred_frames, gt_count)
    save_dir = output_root / task_id / episode_id / "0" / "video"
    _save_frames_as_jpg(pred_frames, save_dir)

    return {
        "task_id": task_id,
        "episode_id": episode_id,
        "prompt": prompt,
        "num_state_steps": int(len(joint_sequence)),
        "num_gt_frames": int(gt_count),
        "num_pred_frames_raw": int(len(_read_video_frames(generated_video))),
        "num_pred_frames_saved": int(len(pred_frames)),
        "generated_video": str(generated_video),
        "save_dir": str(save_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DreamZero task_token_only on WorldModel val and export EWMBench-format predictions.")
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--gt-root", type=Path, default=DEFAULT_GT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--generated-video-dir", type=Path, default=DEFAULT_GENERATED_VIDEO_DIR)
    parser.add_argument("--prompt-template", type=str, default="task_{task_id}")
    parser.add_argument("--num-episodes", type=int, default=0, help="0 means all episodes.")
    parser.add_argument("--task-id", type=str, default="", help="Optional single task filter.")
    parser.add_argument("--sleep-after-reset", type=float, default=1.0)
    parser.add_argument("--max-state-steps", type=int, default=0, help="0 means use all state steps from proprio_stats.h5.")
    args = parser.parse_args()

    policy = WebsocketClientPolicy(host=args.host, port=args.port)
    episodes = iter_episodes(args.input_root)
    if args.task_id:
        episodes = [row for row in episodes if row[0] == args.task_id]
    if args.num_episodes > 0:
        episodes = episodes[: args.num_episodes]

    args.output_root.mkdir(parents=True, exist_ok=True)
    results = []
    for idx, (task_id, episode_id, episode_dir) in enumerate(episodes, start=1):
        print(f"[{idx}/{len(episodes)}] task={task_id} episode={episode_id}")
        row = run_episode(
            policy=policy,
            generated_video_dir=args.generated_video_dir,
            task_id=task_id,
            episode_id=episode_id,
            episode_dir=episode_dir,
            gt_root=args.gt_root,
            output_root=args.output_root,
            prompt_template=args.prompt_template,
            sleep_after_reset=args.sleep_after_reset,
            max_state_steps=args.max_state_steps,
        )
        print(
            f"  prompt={row['prompt']} raw_frames={row['num_pred_frames_raw']} "
            f"saved_frames={row['num_pred_frames_saved']} -> {row['save_dir']}"
        )
        results.append(row)

    print(f"Finished {len(results)} episodes.")


if __name__ == "__main__":
    main()
