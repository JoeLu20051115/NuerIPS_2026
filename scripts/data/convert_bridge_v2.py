#!/usr/bin/env python3
"""
Convert BridgeData V2 (from Hugging Face OXE-style export) into the local
DreamZero evaluation layout used by DROID experiments.

Output layout:
- <output_dir>/data/chunk-xxx/episode_xxxxxx.parquet
- <output_dir>/videos/chunk-xxx/observation.images.<camera_key>/episode_xxxxxx.mp4
- <output_dir>/meta/tasks.jsonl
- <output_dir>/meta/episodes.jsonl
- <output_dir>/meta/info.json
- <output_dir>/meta/modality.json
- <canonical_dir>/episodes/episode_xxxxxx.npz
- <canonical_dir>/videos/<camera_key>/episode_xxxxxx.mp4
- <canonical_dir>/meta.jsonl

Also writes:
- <test_sets_dir>/L1_test_set.json
- <test_sets_dir>/L3_test_set.json

Notes on state/action mapping:
- bridge_eef (default): keeps Bridge native semantics (EE pose + gripper).
  This is safer for cross-dataset reporting (no fake joint labels).
- droid_compat: pads into DROID indices for strict script compatibility.
  This is convenient but not physically equivalent.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import get_dataset_config_names, get_dataset_split_names, load_dataset

DROID_CAMERA_KEYS = [
    "exterior_image_1_left",
    "exterior_image_2_left",
    "wrist_image_left",
]


@dataclass
class StepRecord:
    timestamp: float
    state: np.ndarray
    action: np.ndarray
    is_first: bool
    is_last: bool
    is_terminal: bool
    frame: np.ndarray


@dataclass
class EpisodeRecord:
    source_episode_idx: int
    task_text: str
    task_index: int
    atomic_actions: int
    steps: List[StepRecord]


@dataclass
class ConvertConfig:
    hf_dataset_id: str
    hf_config_name: Optional[str]
    hf_splits: List[str]
    output_dir: Path
    canonical_dir: Path
    test_sets_dir: Path
    fps: int
    max_episodes: int
    min_episode_len: int
    min_atomic_actions: int
    test_set_size: int
    mapping_mode: str
    camera_fill_mode: str
    zero_action_eps: float
    zero_action_run_threshold: int


def normalize_task_text(task: str) -> str:
    text = (task or "").strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"^(task\s*:\s*)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^(please\s+)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^(the robot should\s+)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^(you should\s+)", "", text, flags=re.IGNORECASE)
    text = text.strip(" .")
    if not text:
        return "not provided"
    return text[0].upper() + text[1:]


def count_atomic_actions(task: str) -> int:
    text = (task or "").lower()
    if not text:
        return 1
    parts = re.split(r"\b(?:and then|then|after that|and)\b|[,;]", text)
    chunks = [p.strip() for p in parts if p.strip()]
    return max(1, len(chunks))


def to_rgb_uint8(image_obj) -> Optional[np.ndarray]:
    if image_obj is None:
        return None
    arr = np.array(image_obj)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.ndim != 3:
        return None
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.shape[-1] != 3:
        return None
    return arr.astype(np.uint8)


def _to_pose6(value) -> Optional[np.ndarray]:
    if value is None:
        return None

    if isinstance(value, dict):
        keys = ["x", "y", "z", "roll", "pitch", "yaw"]
        if all(k in value for k in keys):
            try:
                return np.array([float(value[k]) for k in keys], dtype=np.float64)
            except Exception:
                return None
        return None

    try:
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
    except Exception:
        return None
    if arr.shape[0] < 6:
        return None
    return arr[:6]


def _to_scalar(value, default: float = 0.0) -> float:
    if value is None:
        return float(default)
    if isinstance(value, dict):
        # Common OXE patterns: {"value": ...}, {"grasp": ...}, or single-key dicts.
        for k in ("value", "grasp", "closedness", "closedness_action"):
            if k in value:
                try:
                    return float(value[k])
                except Exception:
                    pass
        if len(value) == 1:
            only_v = next(iter(value.values()))
            try:
                return float(only_v)
            except Exception:
                return float(default)
        return float(default)
    if isinstance(value, (list, tuple, np.ndarray)):
        if len(value) == 0:
            return float(default)
        try:
            return float(np.asarray(value, dtype=np.float64).reshape(-1)[0])
        except Exception:
            return float(default)
    try:
        return float(value)
    except Exception:
        return float(default)


def build_state_action(sample: dict, mapping_mode: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    try:
        state_obj = sample.get("state", {})
        abs_action = sample.get("absolute_action", {})
        rel_action = sample.get("action", {})

        eef = _to_pose6(state_obj.get("end_effector_pose"))
        act_pose = _to_pose6(abs_action.get("pose"))
        if act_pose is None:
            act_pose = _to_pose6(rel_action.get("pose"))

        # BridgeData exposes gripper in action space; mirror it to state gripper slot.
        grasp = _to_scalar(abs_action.get("grasp", rel_action.get("grasp", 0.0)), default=0.0)

        if eef is None or act_pose is None:
            return None, None

        if mapping_mode == "bridge_eef":
            state = np.concatenate([eef, np.array([grasp], dtype=np.float64)], axis=0)
            action = np.concatenate([act_pose, np.array([grasp], dtype=np.float64)], axis=0)
            return state, action

        # droid_compat (padded): [xyz+rpy, gripper, 7 x pseudo_joint]
        state = np.zeros(14, dtype=np.float64)
        state[:6] = eef
        state[6] = grasp

        action = np.zeros(21, dtype=np.float64)
        action[14:20] = act_pose
        action[20] = grasp
        return state, action
    except Exception:
        return None, None


def longest_zero_action_run(actions: List[np.ndarray], eps: float) -> int:
    best = 0
    cur = 0
    for a in actions:
        if float(np.linalg.norm(a)) <= eps:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def encode_video(frames: np.ndarray, output_path: Path, fps: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    options = {
        "threads": "1",
        "thread_type": "slice",
        "preset": "ultrafast",
        "tune": "zerolatency",
        "crf": "23",
    }

    container = av.open(str(output_path), mode="w")
    stream = container.add_stream("h264", rate=fps, options=options)
    stream.width = int(frames.shape[2])
    stream.height = int(frames.shape[1])
    stream.pix_fmt = "yuv420p"

    video_frame = av.VideoFrame(width=stream.width, height=stream.height, format="rgb24")
    frame_array = video_frame.to_ndarray(format="rgb24")

    for frame in frames:
        frame_array[:] = frame
        packet = stream.encode(video_frame)
        container.mux(packet)

    packet = stream.encode(None)
    container.mux(packet)
    container.close()


def ensure_dirs(cfg: ConvertConfig) -> None:
    (cfg.output_dir / "data").mkdir(parents=True, exist_ok=True)
    (cfg.output_dir / "meta").mkdir(parents=True, exist_ok=True)
    for cam in DROID_CAMERA_KEYS:
        (cfg.canonical_dir / "videos" / cam).mkdir(parents=True, exist_ok=True)
    (cfg.canonical_dir / "episodes").mkdir(parents=True, exist_ok=True)
    cfg.test_sets_dir.mkdir(parents=True, exist_ok=True)


def get_available_configs(dataset_id: str) -> List[str]:
    return get_dataset_config_names(dataset_id)


def get_available_splits(dataset_id: str, config_name: Optional[str]) -> List[str]:
    return get_dataset_split_names(dataset_id, config_name=config_name)


def iter_samples(dataset_id: str, config_name: Optional[str], split: str) -> Iterable[dict]:
    ds = load_dataset(dataset_id, name=config_name, split=split, streaming=True)
    for sample in ds:
        yield sample


def save_episode(
    ep_index: int,
    episode: EpisodeRecord,
    cfg: ConvertConfig,
    workspace_root: Path,
) -> dict:
    chunk_idx = ep_index // 1000
    chunk_dir = cfg.output_dir / "data" / f"chunk-{chunk_idx:03d}"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    timestamps = np.array([s.timestamp for s in episode.steps], dtype=np.float64)
    states = np.stack([s.state for s in episode.steps], axis=0)
    actions = np.stack([s.action for s in episode.steps], axis=0)
    is_first = np.array([s.is_first for s in episode.steps], dtype=bool)
    is_last = np.array([s.is_last for s in episode.steps], dtype=bool)
    is_terminal = np.array([s.is_terminal for s in episode.steps], dtype=bool)

    parquet_dict = {
        "timestamp": pa.array(timestamps.tolist(), type=pa.float64()),
        "observation.state": pa.array(states.tolist(), type=pa.list_(pa.float64())),
        "action": pa.array(actions.tolist(), type=pa.list_(pa.float64())),
        "task_index": pa.array(np.full(len(episode.steps), episode.task_index, dtype=np.int64).tolist(), type=pa.int64()),
        "episode_index": pa.array(np.full(len(episode.steps), ep_index, dtype=np.int64).tolist(), type=pa.int64()),
        "frame_index": pa.array(np.arange(len(episode.steps), dtype=np.int64).tolist(), type=pa.int64()),
        "is_first": pa.array(is_first.tolist(), type=pa.bool_()),
        "next.done": pa.array(is_last.tolist(), type=pa.bool_()),
        "is_terminal": pa.array(is_terminal.tolist(), type=pa.bool_()),
    }

    parquet_path = chunk_dir / f"episode_{ep_index:06d}.parquet"
    table = pa.Table.from_pydict(parquet_dict)
    pq.write_table(table, parquet_path)

    frames_primary = np.stack([s.frame for s in episode.steps], axis=0)

    for cam in DROID_CAMERA_KEYS:
        if cfg.camera_fill_mode == "zero" and cam != DROID_CAMERA_KEYS[0]:
            cam_frames = np.zeros_like(frames_primary)
        else:
            cam_frames = frames_primary

        final_video = (
            cfg.output_dir
            / "videos"
            / f"chunk-{chunk_idx:03d}"
            / f"observation.images.{cam}"
            / f"episode_{ep_index:06d}.mp4"
        )
        encode_video(cam_frames, final_video, cfg.fps)

        canonical_video = cfg.canonical_dir / "videos" / cam / f"episode_{ep_index:06d}.mp4"
        encode_video(cam_frames, canonical_video, cfg.fps)

    canonical_npz = cfg.canonical_dir / "episodes" / f"episode_{ep_index:06d}.npz"
    np.savez_compressed(
        canonical_npz,
        timestamp=timestamps,
        state=states,
        action=actions,
        task_text=episode.task_text,
        task_index=episode.task_index,
        source_episode_idx=episode.source_episode_idx,
        mapping_mode=cfg.mapping_mode,
    )

    rel_path = parquet_path
    try:
        rel_path = parquet_path.relative_to(workspace_root)
    except Exception:
        rel_path = parquet_path

    return {
        "episode_index": ep_index,
        "source_episode_idx": int(episode.source_episode_idx),
        "tasks": [episode.task_text],
        "length": int(len(episode.steps)),
        "success": True,
        "atomic_actions": int(episode.atomic_actions),
        "mapping_mode": cfg.mapping_mode,
        "camera_fill_mode": cfg.camera_fill_mode,
        "path": str(rel_path),
    }


def write_meta_files(
    cfg: ConvertConfig,
    episodes_meta: List[dict],
    task_to_idx: Dict[str, int],
    state_dim: int,
    action_dim: int,
) -> None:
    task_items = sorted(task_to_idx.items(), key=lambda kv: kv[1])
    with open(cfg.output_dir / "meta" / "tasks.jsonl", "w") as f:
        for task, task_idx in task_items:
            f.write(json.dumps({"task_index": int(task_idx), "task": task}, ensure_ascii=False) + "\n")

    with open(cfg.output_dir / "meta" / "episodes.jsonl", "w") as f:
        for rec in episodes_meta:
            payload = {
                "episode_index": rec["episode_index"],
                "tasks": rec["tasks"],
                "length": rec["length"],
                "success": rec["success"],
                "source_episode_idx": rec["source_episode_idx"],
                "atomic_actions": rec["atomic_actions"],
            }
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    with open(cfg.canonical_dir / "meta.jsonl", "w") as f:
        for rec in episodes_meta:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    modality = {
        "state": {
            "vector": {
                "dim": state_dim,
                "mapping_mode": cfg.mapping_mode,
                "notes": "bridge_eef keeps EE semantics; droid_compat is padded for script compatibility",
            }
        },
        "action": {"vector": {"dim": action_dim, "mapping_mode": cfg.mapping_mode}},
        "video": {cam: {"original_key": f"observation.images.{cam}"} for cam in DROID_CAMERA_KEYS},
        "annotation": {"language.task": {}},
    }

    with open(cfg.output_dir / "meta" / "modality.json", "w") as f:
        json.dump(modality, f, indent=2, ensure_ascii=False)

    info = {
        "codebase_version": "v2.0",
        "robot_type": "bridge_v2",
        "total_episodes": len(episodes_meta),
        "total_frames": int(sum(x["length"] for x in episodes_meta)),
        "total_tasks": len(task_to_idx),
        "total_videos": len(DROID_CAMERA_KEYS),
        "chunks_size": 1000,
        "fps": cfg.fps,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "bridge_mapping_mode": cfg.mapping_mode,
        "camera_fill_mode": cfg.camera_fill_mode,
    }

    with open(cfg.output_dir / "meta" / "info.json", "w") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)


def build_test_sets(cfg: ConvertConfig, episodes_meta: List[dict]) -> None:
    candidates = [x for x in episodes_meta if int(x.get("atomic_actions", 1)) >= cfg.min_atomic_actions]
    if not candidates:
        candidates = episodes_meta[:]

    # Difficulty proxy: prioritize longer + more atomic actions for L3.
    ranked = sorted(candidates, key=lambda x: (x["atomic_actions"], x["length"]), reverse=True)
    l3 = ranked[: cfg.test_set_size]
    l1 = list(reversed(ranked))[: cfg.test_set_size]

    def to_test_item(rec: dict) -> dict:
        ep_idx = int(rec["episode_index"])
        return {
            "episode_id": f"episode_{ep_idx:06d}",
            "path": rec["path"],
        }

    with open(cfg.test_sets_dir / "L3_test_set.json", "w") as f:
        json.dump([to_test_item(x) for x in l3], f, indent=2, ensure_ascii=False)

    with open(cfg.test_sets_dir / "L1_test_set.json", "w") as f:
        json.dump([to_test_item(x) for x in l1], f, indent=2, ensure_ascii=False)


def finalize_episode(
    source_episode_idx: int,
    task_text: str,
    steps: List[StepRecord],
    cfg: ConvertConfig,
    task_to_idx: Dict[str, int],
) -> Optional[EpisodeRecord]:
    if not steps or len(steps) < cfg.min_episode_len:
        return None

    actions = [s.action for s in steps]
    if longest_zero_action_run(actions, cfg.zero_action_eps) >= cfg.zero_action_run_threshold:
        return None

    frame_ok = sum(1 for s in steps if s.frame is not None)
    frame_readable_ratio = frame_ok / len(steps)
    if frame_readable_ratio < 0.99:
        return None

    norm_task = normalize_task_text(task_text)
    atomic_actions = count_atomic_actions(norm_task)

    if norm_task not in task_to_idx:
        task_to_idx[norm_task] = len(task_to_idx)

    return EpisodeRecord(
        source_episode_idx=source_episode_idx,
        task_text=norm_task,
        task_index=task_to_idx[norm_task],
        atomic_actions=atomic_actions,
        steps=steps,
    )


def run_conversion(cfg: ConvertConfig) -> None:
    ensure_dirs(cfg)

    available_configs = get_available_configs(cfg.hf_dataset_id)
    config_name = cfg.hf_config_name
    if config_name is None:
        config_name = available_configs[0] if available_configs else None
    if config_name and available_configs and config_name not in available_configs:
        raise ValueError(
            f"Config '{config_name}' not found in dataset {cfg.hf_dataset_id}. "
            f"Available configs: {available_configs}"
        )

    available = get_available_splits(cfg.hf_dataset_id, config_name)
    for split in cfg.hf_splits:
        if split not in available:
            raise ValueError(f"Split '{split}' not found in dataset {cfg.hf_dataset_id}. Available: {available}")

    task_to_idx: Dict[str, int] = {"not provided": 0}
    episodes: List[EpisodeRecord] = []

    current_source_ep: Optional[int] = None
    current_task: str = "not provided"
    current_steps: List[StepRecord] = []

    def flush_current() -> None:
        nonlocal current_steps, current_source_ep, current_task
        if current_source_ep is None:
            return
        ep = finalize_episode(current_source_ep, current_task, current_steps, cfg, task_to_idx)
        if ep is not None:
            episodes.append(ep)
        current_source_ep = None
        current_task = "not provided"
        current_steps = []

    for split in cfg.hf_splits:
        for sample in iter_samples(cfg.hf_dataset_id, config_name, split):
            src_ep = int(sample.get("episode_idx", -1))
            if src_ep < 0:
                continue

            if current_source_ep is None:
                current_source_ep = src_ep
            elif src_ep != current_source_ep:
                flush_current()
                current_source_ep = src_ep

            task_text = sample.get("observation", {}).get("task", "not provided")
            if task_text:
                current_task = str(task_text)

            state, action = build_state_action(sample, cfg.mapping_mode)
            frame = to_rgb_uint8(sample.get("observation", {}).get("image", sample.get("image")))
            if state is None or action is None or frame is None:
                continue

            timestamp = float(sample.get("timestamp", len(current_steps) / max(1, cfg.fps)))
            sdict = sample.get("state", {})
            step = StepRecord(
                timestamp=timestamp,
                state=state,
                action=action,
                is_first=bool(sdict.get("is_first", len(current_steps) == 0)),
                is_last=bool(sdict.get("is_last", False)),
                is_terminal=bool(sdict.get("is_terminal", False)),
                frame=frame,
            )
            current_steps.append(step)

            if step.is_last:
                flush_current()

            if len(episodes) >= cfg.max_episodes:
                break

        if len(episodes) >= cfg.max_episodes:
            break

    flush_current()

    if not episodes:
        raise RuntimeError("No valid episodes produced. Try lowering filters or checking source fields.")

    workspace_root = Path.cwd()
    episodes_meta: List[dict] = []

    for ep_idx, episode in enumerate(episodes):
        meta = save_episode(ep_idx, episode, cfg, workspace_root)
        episodes_meta.append(meta)

    state_dim = episodes[0].steps[0].state.shape[0]
    action_dim = episodes[0].steps[0].action.shape[0]

    write_meta_files(cfg, episodes_meta, task_to_idx, state_dim=state_dim, action_dim=action_dim)
    build_test_sets(cfg, episodes_meta)

    print("BridgeData V2 conversion complete")
    print(f"Episodes kept: {len(episodes_meta)}")
    print(f"Mapping mode: {cfg.mapping_mode}")
    print(f"Output dataset: {cfg.output_dir}")
    print(f"Canonical dir: {cfg.canonical_dir}")
    print(f"Test sets: {cfg.test_sets_dir}")


def parse_args() -> ConvertConfig:
    parser = argparse.ArgumentParser(description="Convert BridgeData V2 to DreamZero-compatible structure")
    parser.add_argument("--hf-dataset-id", default="mbodiai/oxe_bridge_v2", help="Hugging Face dataset id")
    parser.add_argument(
        "--hf-config",
        default=None,
        help="Hugging Face dataset config name (defaults to first available config)",
    )
    parser.add_argument(
        "--hf-splits",
        default="shard_0",
        help="Comma-separated split/config names (default: shard_0)",
    )
    parser.add_argument(
        "--output-dir",
        default="data/bridge_v2_lerobot",
        help="Final parquet/videos/meta output directory",
    )
    parser.add_argument(
        "--canonical-dir",
        default="data/bridge_v2_canonical",
        help="Intermediate canonical output directory",
    )
    parser.add_argument(
        "--test-sets-dir",
        default="test_sets_bridge",
        help="Directory to write L1/L3 test set json files",
    )
    parser.add_argument("--fps", type=int, default=10, help="Output fps for videos")
    parser.add_argument("--max-episodes", type=int, default=200, help="Max episodes to convert")
    parser.add_argument("--min-episode-len", type=int, default=20, help="Drop episodes shorter than this")
    parser.add_argument(
        "--min-atomic-actions",
        type=int,
        default=3,
        help="Long-chain threshold used for test set candidate filtering",
    )
    parser.add_argument("--test-set-size", type=int, default=20, help="Episode count per L1/L3 test set")
    parser.add_argument(
        "--mapping-mode",
        choices=["bridge_eef", "droid_compat"],
        default="bridge_eef",
        help="State/action mapping strategy",
    )
    parser.add_argument(
        "--camera-fill-mode",
        choices=["duplicate", "zero"],
        default="duplicate",
        help="How to fill missing extra camera views",
    )
    parser.add_argument("--zero-action-eps", type=float, default=1e-6, help="Zero-action threshold")
    parser.add_argument(
        "--zero-action-run-threshold",
        type=int,
        default=20,
        help="Drop episodes with long all-zero action segments",
    )

    args = parser.parse_args()

    return ConvertConfig(
        hf_dataset_id=args.hf_dataset_id,
        hf_config_name=args.hf_config,
        hf_splits=[x.strip() for x in args.hf_splits.split(",") if x.strip()],
        output_dir=Path(args.output_dir),
        canonical_dir=Path(args.canonical_dir),
        test_sets_dir=Path(args.test_sets_dir),
        fps=int(args.fps),
        max_episodes=int(args.max_episodes),
        min_episode_len=int(args.min_episode_len),
        min_atomic_actions=int(args.min_atomic_actions),
        test_set_size=int(args.test_set_size),
        mapping_mode=args.mapping_mode,
        camera_fill_mode=args.camera_fill_mode,
        zero_action_eps=float(args.zero_action_eps),
        zero_action_run_threshold=int(args.zero_action_run_threshold),
    )


if __name__ == "__main__":
    cfg = parse_args()
    run_conversion(cfg)
