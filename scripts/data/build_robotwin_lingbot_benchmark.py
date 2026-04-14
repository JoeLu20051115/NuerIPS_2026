#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import pickle
import random
import re
import zipfile
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import av
import cv2
import h5py
import numpy as np
import pandas as pd
from huggingface_hub import HfApi, hf_hub_download


REPO_ROOT = Path(__file__).resolve().parents[2]

ROBOTWIN_DATASET_ID = "TianxingChen/RoboTwin2.0"
LINGBOT_DATASET_ID = "robbyant/robotwin-clean-and-aug-lerobot"

SEED = 42
LEVELS = ("L1", "L2", "L3")
SOURCE_VARIANTS = ("aloha-agilex_clean_50", "aloha-agilex_randomized_500")
REQUIRED_CAMERA_MAP = {
    "head_camera": "observation.images.cam_high",
    "left_camera": "observation.images.cam_left_wrist",
    "right_camera": "observation.images.cam_right_wrist",
}
STATE_FEATURE_NAMES = [
    "left_x",
    "left_y",
    "left_z",
    "left_q1",
    "left_q2",
    "left_q3",
    "left_q4",
    "left_gripper",
    "right_x",
    "right_y",
    "right_z",
    "right_q1",
    "right_q2",
    "right_q3",
    "right_q4",
    "right_gripper",
]


@dataclass(frozen=True)
class ArchiveSpec:
    task_name: str
    variant: str
    repo_path: str
    size_bytes: int | None
    local_zip_path: Path


class ZipFileCache:
    def __init__(self) -> None:
        self._cache: dict[Path, zipfile.ZipFile] = {}

    def get(self, zip_path: Path) -> zipfile.ZipFile:
        if zip_path not in self._cache:
            self._cache[zip_path] = zipfile.ZipFile(zip_path)
        return self._cache[zip_path]

    def close(self) -> None:
        for handle in self._cache.values():
            handle.close()
        self._cache.clear()


class RunningImageStats:
    def __init__(self) -> None:
        self._min = np.full(3, np.inf, dtype=np.float64)
        self._max = np.full(3, -np.inf, dtype=np.float64)
        self._sum = np.zeros(3, dtype=np.float64)
        self._sumsq = np.zeros(3, dtype=np.float64)
        self._count_pixels = 0
        self._count_frames = 0

    def update(self, frame_rgb: np.ndarray) -> None:
        flat = frame_rgb.astype(np.float64).reshape(-1, 3) / 255.0
        self._min = np.minimum(self._min, flat.min(axis=0))
        self._max = np.maximum(self._max, flat.max(axis=0))
        self._sum += flat.sum(axis=0)
        self._sumsq += np.square(flat).sum(axis=0)
        self._count_pixels += int(flat.shape[0])
        self._count_frames += 1

    def serialize(self) -> dict[str, Any]:
        if self._count_pixels == 0:
            zeros = np.zeros(3, dtype=np.float64)
            return {
                "min": [[[float(v)]] for v in zeros],
                "max": [[[float(v)]] for v in zeros],
                "mean": [[[float(v)]] for v in zeros],
                "std": [[[float(v)]] for v in zeros],
                "count": [0],
            }
        mean = self._sum / self._count_pixels
        variance = np.maximum(self._sumsq / self._count_pixels - np.square(mean), 0.0)
        std = np.sqrt(variance)
        return {
            "min": [[[float(v)]] for v in self._min],
            "max": [[[float(v)]] for v in self._max],
            "mean": [[[float(v)]] for v in mean],
            "std": [[[float(v)]] for v in std],
            "count": [int(self._count_frames)],
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download official RoboTwin 2.0 Aloha archives, build a Seed=42 "
            "L1/L2/L3 round-robin benchmark, and export a LingBot-style LeRobot dataset."
        )
    )
    parser.add_argument(
        "--download-dir",
        type=Path,
        default=REPO_ROOT / "data/robotwin_source_official_aloha",
        help="Directory that stores official RoboTwin Aloha zip archives.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "data/robotwin_lingbot_eval_seed42_l123_300",
        help="Destination LeRobot-style benchmark dataset directory.",
    )
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=REPO_ROOT / "analysis_outputs/robotwin_lingbot_eval_seed42_l123_300",
        help="Directory that stores human-readable preprocessing summaries.",
    )
    parser.add_argument(
        "--prompt-cache-dir",
        type=Path,
        default=REPO_ROOT / "tmp/lingbot_prompt_bank",
        help="Cache directory for LingBot prompt-bank metadata.",
    )
    parser.add_argument(
        "--quota-per-level",
        type=int,
        default=100,
        help="How many episodes to retain per complexity level.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Global random seed used for round-robin ordering and prompt assignment.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=50,
        help="Output FPS used for the LingBot-compatible benchmark videos.",
    )
    parser.add_argument(
        "--image-width",
        type=int,
        default=640,
        help="Output video width.",
    )
    parser.add_argument(
        "--image-height",
        type=int,
        default=480,
        help="Output video height.",
    )
    parser.add_argument(
        "--max-download-workers",
        type=int,
        default=4,
        help="Number of concurrent Hugging Face downloads.",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="",
        help="Optional comma-separated task whitelist for smoke tests.",
    )
    parser.add_argument(
        "--variants",
        type=str,
        default=",".join(SOURCE_VARIANTS),
        help=(
            "Optional comma-separated source variants. "
            "Defaults to aloha-agilex_clean_50,aloha-agilex_randomized_500."
        ),
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip Hugging Face downloads and use already-downloaded archives only.",
    )
    parser.add_argument(
        "--keep-probe-extracts",
        action="store_true",
        help="Keep any intermediate probe directories if you use this script for debugging.",
    )
    return parser.parse_args()


def normalize_task_filter(raw: str) -> set[str] | None:
    items = {x.strip() for x in raw.split(",") if x.strip()}
    return items or None


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def list_aloha_archives(
    *,
    download_dir: Path,
    task_filter: set[str] | None,
    variant_filter: set[str],
) -> list[ArchiveSpec]:
    api = HfApi()
    info = api.dataset_info(ROBOTWIN_DATASET_ID, files_metadata=True)
    keep_variants = {f"{variant}.zip" for variant in variant_filter}
    specs: list[ArchiveSpec] = []
    for sibling in info.siblings:
        repo_path = sibling.rfilename
        if not (repo_path.startswith("dataset/") and repo_path.endswith(".zip")):
            continue
        _, task_name, filename = repo_path.split("/", 2)
        if filename not in keep_variants:
            continue
        if task_filter and task_name not in task_filter:
            continue
        local_zip_path = download_dir / repo_path
        specs.append(
            ArchiveSpec(
                task_name=task_name,
                variant=filename[:-4],
                repo_path=repo_path,
                size_bytes=sibling.size,
                local_zip_path=local_zip_path,
            )
        )
    specs.sort(key=lambda item: (item.task_name, item.variant))
    return specs


def download_archives(specs: list[ArchiveSpec], *, download_dir: Path, max_workers: int) -> None:
    download_dir.mkdir(parents=True, exist_ok=True)
    if not specs:
        return
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _download(spec: ArchiveSpec) -> tuple[str, str]:
        local_path = hf_hub_download(
            repo_id=ROBOTWIN_DATASET_ID,
            filename=spec.repo_path,
            repo_type="dataset",
            local_dir=download_dir,
        )
        return spec.repo_path, local_path

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_download, spec): spec for spec in specs}
        for future in as_completed(futures):
            repo_path, local_path = future.result()
            print(f"[downloaded] {repo_path} -> {local_path}")


def infer_zip_prefix(zf: zipfile.ZipFile) -> str:
    prefixes = sorted({name.split("/", 1)[0] for name in zf.namelist() if "/" in name})
    if not prefixes:
        raise RuntimeError(f"Could not infer root prefix from archive {zf.filename}")
    return prefixes[0]


def extract_episode_index(member_name: str) -> int:
    match = re.search(r"episode(\d+)\.(?:hdf5|mp4|pkl)$", member_name)
    if not match:
        raise ValueError(f"Could not parse episode id from {member_name}")
    return int(match.group(1))


def load_scene_info(zf: zipfile.ZipFile, prefix: str) -> dict[str, Any]:
    scene_path = f"{prefix}/scene_info.json"
    if scene_path not in zf.namelist():
        return {}
    return json.loads(zf.read(scene_path))


def success_from_traj_metadata(metadata: dict[str, Any]) -> bool:
    saw_any = False
    for key in ("left_joint_path", "right_joint_path"):
        arm_statuses = [
            str(item["status"])
            for item in (metadata.get(key, []) or [])
            if isinstance(item, dict) and "status" in item
        ]
        if not arm_statuses:
            continue
        saw_any = True
        if arm_statuses[-1].lower() != "success":
            return False
    return saw_any


def arm_usage_from_traj_metadata(metadata: dict[str, Any]) -> str:
    left = bool(metadata.get("left_joint_path"))
    right = bool(metadata.get("right_joint_path"))
    if left and right:
        return "both"
    if left:
        return "left_only"
    if right:
        return "right_only"
    return "none"


def arm_hint_from_scene_bindings(bindings: dict[str, Any], arm_usage: str) -> str | None:
    for value in bindings.values():
        lowered = str(value).strip().lower()
        if lowered in {"left", "right", "both"}:
            return lowered
    if arm_usage == "left_only":
        return "left"
    if arm_usage == "right_only":
        return "right"
    if arm_usage == "both":
        return "both"
    return None


def mp4_frame_count(payload: bytes) -> int:
    with av.open(io.BytesIO(payload)) as container:
        stream = container.streams.video[0]
        if stream.frames and int(stream.frames) > 0:
            return int(stream.frames)
        count = 0
        for _ in container.decode(stream):
            count += 1
        return count


def validate_archive_schema(zf: zipfile.ZipFile, prefix: str, hdf5_member: str) -> dict[str, Any]:
    required_dataset_paths = [
        "endpose/left_endpose",
        "endpose/left_gripper",
        "endpose/right_endpose",
        "endpose/right_gripper",
        "observation/head_camera/rgb",
        "observation/left_camera/rgb",
        "observation/right_camera/rgb",
    ]
    with h5py.File(io.BytesIO(zf.read(hdf5_member)), "r") as h5:
        missing = [path for path in required_dataset_paths if path not in h5]
        if missing:
            return {
                "valid": False,
                "reason": f"missing_datasets:{','.join(missing)}",
            }
        lengths = {
            "state": int(h5["endpose/left_endpose"].shape[0]),
            "cam_high": int(h5["observation/head_camera/rgb"].shape[0]),
            "cam_left_wrist": int(h5["observation/left_camera/rgb"].shape[0]),
            "cam_right_wrist": int(h5["observation/right_camera/rgb"].shape[0]),
        }
        if len(set(lengths.values())) != 1:
            return {
                "valid": False,
                "reason": f"inconsistent_lengths:{lengths}",
            }
        sample_frame = decode_jpeg_bytes(h5["observation/head_camera/rgb"][0].tobytes())
        return {
            "valid": True,
            "frame_height": int(sample_frame.shape[0]),
            "frame_width": int(sample_frame.shape[1]),
            "episode_length": lengths["state"],
        }


def index_archive(spec: ArchiveSpec) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    counters = Counter()
    schema_info: dict[str, Any] | None = None

    with zipfile.ZipFile(spec.local_zip_path) as zf:
        prefix = infer_zip_prefix(zf)
        scene_info = load_scene_info(zf, prefix)
        members_by_episode: dict[int, dict[str, str]] = defaultdict(dict)
        for member in zf.namelist():
            if member.endswith(".pkl") and f"{prefix}/_traj_data/" in member:
                members_by_episode[extract_episode_index(member)]["pkl"] = member
            elif member.endswith(".hdf5") and f"{prefix}/data/" in member:
                members_by_episode[extract_episode_index(member)]["hdf5"] = member
            elif member.endswith(".mp4") and f"{prefix}/video/" in member:
                members_by_episode[extract_episode_index(member)]["mp4"] = member

        if members_by_episode:
            first_hdf5_member = next(
                value["hdf5"] for _, value in sorted(members_by_episode.items()) if "hdf5" in value
            )
            schema_info = validate_archive_schema(zf, prefix, first_hdf5_member)
            if not schema_info.get("valid", False):
                return [], {
                    "task_name": spec.task_name,
                    "variant": spec.variant,
                    "repo_path": spec.repo_path,
                    "schema_probe": schema_info,
                    "counters": dict(counters),
                }

        for episode_idx, members in sorted(members_by_episode.items()):
            counters["raw_pool"] += 1
            if {"pkl", "hdf5", "mp4"} - members.keys():
                counters["missing_members"] += 1
                continue

            traj_metadata = pickle.loads(zf.read(members["pkl"]))
            if not success_from_traj_metadata(traj_metadata):
                counters["not_success"] += 1
                continue

            step_count = mp4_frame_count(zf.read(members["mp4"]))
            if step_count <= 0:
                counters["bad_preview_video"] += 1
                continue

            scene_payload = scene_info.get(f"episode_{episode_idx}", {})
            bindings = dict(scene_payload.get("info") or {})
            arm_usage = arm_usage_from_traj_metadata(traj_metadata)
            candidates.append(
                {
                    "task_name": spec.task_name,
                    "variant": spec.variant,
                    "repo_path": spec.repo_path,
                    "local_zip_path": str(spec.local_zip_path),
                    "zip_prefix": prefix,
                    "source_episode_index": int(episode_idx),
                    "hdf5_member": members["hdf5"],
                    "mp4_member": members["mp4"],
                    "pkl_member": members["pkl"],
                    "step_count": int(step_count),
                    "success": True,
                    "arm_usage": arm_usage,
                    "arm_hint": arm_hint_from_scene_bindings(bindings, arm_usage),
                    "scene_bindings": bindings,
                    "scene_texture_info": scene_payload.get("texture_info"),
                    "scene_cluttered_table_info": scene_payload.get("cluttered_table_info"),
                    "source_uid": f"{spec.task_name}:{spec.variant}:episode{episode_idx}",
                }
            )
            counters["eligible_pre_hdf5_validation"] += 1

    stats = {
        "task_name": spec.task_name,
        "variant": spec.variant,
        "repo_path": spec.repo_path,
        "size_bytes": spec.size_bytes,
        "schema_probe": schema_info,
        "counters": dict(counters),
    }
    return candidates, stats


def compute_cutoffs(step_counts: list[int]) -> tuple[int, int]:
    ordered = sorted(step_counts)
    q1 = ordered[len(ordered) // 3]
    q2 = ordered[(2 * len(ordered)) // 3]
    return int(q1), int(q2)


def assign_level(step_count: int, q1: int, q2: int) -> str:
    if step_count <= q1:
        return "L1"
    if step_count <= q2:
        return "L2"
    return "L3"


def build_round_robin_order(
    rows: list[dict[str, Any]],
    *,
    seed: int,
    level: str,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["task_name"]].append(row)

    ordered_tasks = sorted(grouped.keys())
    queues: dict[str, deque[dict[str, Any]]] = {}
    for task_name in ordered_tasks:
        task_rows = sorted(
            grouped[task_name],
            key=lambda item: (item["variant"], item["source_episode_index"], item["source_uid"]),
        )
        rng = random.Random(f"{seed}::{level}::{task_name}")
        rng.shuffle(task_rows)
        queues[task_name] = deque(task_rows)

    ordered_rows: list[dict[str, Any]] = []
    while any(queues.values()):
        progressed = False
        for task_name in ordered_tasks:
            if queues[task_name]:
                ordered_rows.append(queues[task_name].popleft())
                progressed = True
        if not progressed:
            break
    return ordered_rows


def decode_jpeg_bytes(payload: bytes) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("cv2.imdecode returned None")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def validate_selected_episode(row: dict[str, Any], zip_cache: ZipFileCache) -> tuple[bool, dict[str, Any]]:
    zf = zip_cache.get(Path(row["local_zip_path"]))
    with h5py.File(io.BytesIO(zf.read(row["hdf5_member"])), "r") as h5:
        required_paths = [
            "endpose/left_endpose",
            "endpose/left_gripper",
            "endpose/right_endpose",
            "endpose/right_gripper",
            "observation/head_camera/rgb",
            "observation/left_camera/rgb",
            "observation/right_camera/rgb",
        ]
        missing = [path for path in required_paths if path not in h5]
        if missing:
            return False, {"reason": f"missing_datasets:{','.join(missing)}"}

        lengths = [
            int(h5["endpose/left_endpose"].shape[0]),
            int(h5["endpose/left_gripper"].shape[0]),
            int(h5["endpose/right_endpose"].shape[0]),
            int(h5["endpose/right_gripper"].shape[0]),
            int(h5["observation/head_camera/rgb"].shape[0]),
            int(h5["observation/left_camera/rgb"].shape[0]),
            int(h5["observation/right_camera/rgb"].shape[0]),
        ]
        if len(set(lengths)) != 1:
            return False, {"reason": f"inconsistent_modal_lengths:{lengths}"}
        if lengths[0] != int(row["step_count"]):
            return False, {
                "reason": f"preview_hdf5_length_mismatch:{row['step_count']}!={lengths[0]}"
            }

        try:
            for raw_camera_name in REQUIRED_CAMERA_MAP:
                decode_jpeg_bytes(h5[f"observation/{raw_camera_name}/rgb"][0].tobytes())
        except Exception as exc:  # pragma: no cover
            return False, {"reason": f"jpeg_decode_failed:{type(exc).__name__}:{exc}"}

        return True, {
            "validated_step_count": int(lengths[0]),
        }


def choose_valid_round_robin_subset(
    rows: list[dict[str, Any]],
    *,
    level: str,
    quota: int,
    seed: int,
    zip_cache: ZipFileCache,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ordered_rows = build_round_robin_order(rows, seed=seed, level=level)
    selected: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for row in ordered_rows:
        valid, info = validate_selected_episode(row, zip_cache)
        if not valid:
            rejected.append(
                {
                    "source_uid": row["source_uid"],
                    "level": level,
                    "reason": info["reason"],
                }
            )
            continue
        validated = dict(row)
        validated.update(info)
        selected.append(validated)
        if len(selected) >= quota:
            break
    if len(selected) < quota:
        raise RuntimeError(
            f"Could not select {quota} valid episodes for {level}; only found {len(selected)}."
        )
    return selected, rejected


def normalize_sentence(text: str) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    if not compact:
        return compact
    compact = compact[0].upper() + compact[1:]
    if compact[-1] not in ".!?":
        compact += "."
    return compact


def load_lingbot_prompt_bank(cache_dir: Path) -> tuple[dict[str, list[str]], dict[str, str]]:
    api = HfApi()
    cache_dir.mkdir(parents=True, exist_ok=True)

    folder_map: dict[str, str] = {}
    for item in api.list_repo_tree(
        LINGBOT_DATASET_ID,
        repo_type="dataset",
        path_in_repo="lerobot_robotwin_eef_clean_50",
        recursive=False,
    ):
        path = getattr(item, "path", None)
        if not path:
            continue
        folder_name = Path(path).name
        task_name = folder_name.split("-", 1)[0]
        folder_map[task_name] = path

    prompt_bank: dict[str, list[str]] = {}
    for task_name, folder_path in sorted(folder_map.items()):
        tasks_path = hf_hub_download(
            repo_id=LINGBOT_DATASET_ID,
            filename=f"{folder_path}/meta/tasks.jsonl",
            repo_type="dataset",
            local_dir=cache_dir,
        )
        prompts = []
        for line in Path(tasks_path).read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            prompt = normalize_sentence(str(record["task"]))
            if prompt not in prompts:
                prompts.append(prompt)
        prompt_bank[task_name] = prompts
    return prompt_bank, folder_map


def heuristic_prompt(task_name: str, row: dict[str, Any]) -> str:
    task_text = task_name.replace("_", " ")
    arm_hint = row.get("arm_hint")
    if arm_hint in {"left", "right"}:
        return normalize_sentence(f"Use the {arm_hint} arm to {task_text}")
    if arm_hint == "both":
        return normalize_sentence(f"Use both arms to {task_text}")
    return normalize_sentence(task_text)


def prompt_arm_score(prompt: str, arm_hint: str | None) -> int:
    lowered = prompt.lower()
    has_left = "left arm" in lowered
    has_right = "right arm" in lowered
    if arm_hint == "left":
        if has_left and not has_right:
            return 0
        if not has_left and not has_right:
            return 1
        return 3
    if arm_hint == "right":
        if has_right and not has_left:
            return 0
        if not has_left and not has_right:
            return 1
        return 3
    if arm_hint == "both":
        if has_left and has_right:
            return 0
        if not has_left and not has_right:
            return 1
        return 2
    return 0


def assign_prompts(
    selected_rows: list[dict[str, Any]],
    *,
    prompt_bank: dict[str, list[str]],
    seed: int,
) -> None:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selected_rows:
        grouped[row["task_name"]].append(row)

    for task_name, rows in grouped.items():
        prompts = list(prompt_bank.get(task_name, []))
        rng = random.Random(f"{seed}::prompt::{task_name}")
        rng.shuffle(prompts)
        rows.sort(key=lambda item: (LEVELS.index(item["complexity_level"]), item["source_uid"]))
        unused_indices = list(range(len(prompts)))
        for row in rows:
            if not unused_indices:
                row["prompt"] = heuristic_prompt(task_name, row)
                row["prompt_source"] = "heuristic_fallback_reused"
                continue
            arm_hint = row.get("arm_hint")
            best_idx = min(
                unused_indices,
                key=lambda prompt_idx: (
                    prompt_arm_score(prompts[prompt_idx], arm_hint),
                    prompt_idx,
                ),
            )
            row["prompt"] = prompts[best_idx]
            row["prompt_source"] = "lingbot_clean_prompt_bank"
            unused_indices.remove(best_idx)


def state_from_hdf5(h5: h5py.File) -> np.ndarray:
    left_pose = np.asarray(h5["endpose/left_endpose"], dtype=np.float32)
    left_gripper = np.asarray(h5["endpose/left_gripper"], dtype=np.float32).reshape(-1, 1)
    right_pose = np.asarray(h5["endpose/right_endpose"], dtype=np.float32)
    right_gripper = np.asarray(h5["endpose/right_gripper"], dtype=np.float32).reshape(-1, 1)
    return np.concatenate([left_pose, left_gripper, right_pose, right_gripper], axis=1)


def next_state_action(state: np.ndarray) -> np.ndarray:
    return np.concatenate([state[1:], state[-1:]], axis=0).astype(np.float32)


def make_three_segment_action_config(length: int, prompt: str) -> list[dict[str, Any]]:
    if length <= 0:
        return []
    boundaries = [0, min(49, length), min(98, length), length]
    segments: list[dict[str, Any]] = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        if end <= start:
            continue
        segments.append(
            {
                "start_frame": int(start),
                "end_frame": int(end),
                "action_text": prompt,
                "skill": "",
            }
        )
    return segments


def numeric_feature_stats(array: np.ndarray) -> dict[str, Any]:
    array = np.asarray(array)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    return {
        "min": array.min(axis=0).astype(np.float64).tolist(),
        "max": array.max(axis=0).astype(np.float64).tolist(),
        "mean": array.mean(axis=0).astype(np.float64).tolist(),
        "std": array.std(axis=0).astype(np.float64).tolist(),
        "count": [int(array.shape[0])],
    }


def open_video_writer(output_path: Path, *, width: int, height: int, fps: int) -> tuple[av.container.OutputContainer, av.video.stream.VideoStream]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    container = av.open(str(output_path), mode="w")
    stream = container.add_stream(
        "h264",
        rate=fps,
        options={
            "threads": "1",
            "thread_type": "slice",
            "preset": "ultrafast",
            "tune": "zerolatency",
            "crf": "23",
        },
    )
    stream.width = int(width)
    stream.height = int(height)
    stream.pix_fmt = "yuv420p"
    return container, stream


def encode_rgb_frame(
    container: av.container.OutputContainer,
    stream: av.video.stream.VideoStream,
    frame_rgb: np.ndarray,
) -> None:
    video_frame = av.VideoFrame.from_ndarray(frame_rgb, format="rgb24")
    for packet in stream.encode(video_frame):
        container.mux(packet)


def finalize_video_writer(
    container: av.container.OutputContainer,
    stream: av.video.stream.VideoStream,
) -> None:
    for packet in stream.encode(None):
        container.mux(packet)
    container.close()


def convert_episode(
    row: dict[str, Any],
    *,
    output_dir: Path,
    episode_index: int,
    task_index: int,
    global_frame_offset: int,
    fps: int,
    image_width: int,
    image_height: int,
    zip_cache: ZipFileCache,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    zf = zip_cache.get(Path(row["local_zip_path"]))
    hdf5_bytes = zf.read(row["hdf5_member"])
    with h5py.File(io.BytesIO(hdf5_bytes), "r") as h5:
        state = state_from_hdf5(h5)
        action = next_state_action(state)
        length = int(state.shape[0])

        timestamps = np.arange(length, dtype=np.float32) / float(fps)
        frame_indices = np.arange(length, dtype=np.int64)
        global_indices = np.arange(global_frame_offset, global_frame_offset + length, dtype=np.int64)

        episode_chunk = episode_index // 1000
        data_dir = output_dir / f"data/chunk-{episode_chunk:03d}"
        data_dir.mkdir(parents=True, exist_ok=True)
        parquet_path = data_dir / f"episode_{episode_index:06d}.parquet"
        pd.DataFrame(
            {
                "observation.state": [value.astype(np.float32) for value in state],
                "action": [value.astype(np.float32) for value in action],
                "timestamp": timestamps,
                "frame_index": frame_indices,
                "episode_index": np.full(length, episode_index, dtype=np.int64),
                "index": global_indices,
                "task_index": np.full(length, task_index, dtype=np.int64),
            }
        ).to_parquet(parquet_path, index=False)

        image_stats: dict[str, dict[str, Any]] = {}
        for raw_camera_name, feature_name in REQUIRED_CAMERA_MAP.items():
            video_path = (
                output_dir
                / f"videos/chunk-{episode_chunk:03d}/{feature_name}/episode_{episode_index:06d}.mp4"
            )
            container, stream = open_video_writer(
                video_path,
                width=image_width,
                height=image_height,
                fps=fps,
            )
            stats = RunningImageStats()
            sample_count = min(100, length)
            sample_indices = set(np.linspace(0, length - 1, num=sample_count, dtype=int).tolist())
            rgb_dataset = h5[f"observation/{raw_camera_name}/rgb"]
            for frame_idx in range(length):
                frame_rgb = decode_jpeg_bytes(rgb_dataset[frame_idx].tobytes())
                if frame_rgb.shape[1] != image_width or frame_rgb.shape[0] != image_height:
                    frame_rgb = cv2.resize(
                        frame_rgb,
                        (image_width, image_height),
                        interpolation=cv2.INTER_LINEAR,
                    )
                encode_rgb_frame(container, stream, frame_rgb)
                if frame_idx in sample_indices:
                    stats.update(frame_rgb)
            finalize_video_writer(container, stream)
            image_stats[feature_name] = stats.serialize()

    prompt = str(row["prompt"])
    episode_meta = {
        "episode_index": int(episode_index),
        "tasks": [prompt],
        "length": int(length),
        "action_config": [
            {
                "start_frame": 0,
                "end_frame": int(length),
                "action_text": prompt,
                "skill": "",
            }
        ],
        "complexity_level": row["complexity_level"],
        "step_count": int(row["step_count"]),
        "task_name": row["task_name"],
        "source_variant": row["variant"],
        "source_episode_index": int(row["source_episode_index"]),
        "source_uid": row["source_uid"],
        "source_repo_path": row["repo_path"],
        "source_local_zip_path": row["local_zip_path"],
        "source_hdf5_member": row["hdf5_member"],
        "source_mp4_member": row["mp4_member"],
        "source_pkl_member": row["pkl_member"],
        "prompt_source": row["prompt_source"],
        "arm_usage": row["arm_usage"],
        "arm_hint": row["arm_hint"],
        "scene_bindings": row["scene_bindings"],
    }
    episodes_ori_meta = {
        "episode_index": int(episode_index),
        "tasks": [prompt],
        "length": int(length),
        "action_config": make_three_segment_action_config(length, prompt),
        "complexity_level": row["complexity_level"],
        "step_count": int(row["step_count"]),
        "task_name": row["task_name"],
        "source_variant": row["variant"],
        "source_episode_index": int(row["source_episode_index"]),
        "source_uid": row["source_uid"],
        "prompt_source": row["prompt_source"],
    }
    episode_stats = {
        "episode_index": int(episode_index),
        "stats": {
            "observation.state": numeric_feature_stats(state),
            "action": numeric_feature_stats(action),
            "timestamp": numeric_feature_stats(timestamps),
            "frame_index": numeric_feature_stats(frame_indices),
            "episode_index": numeric_feature_stats(np.full(length, episode_index, dtype=np.int64)),
            "index": numeric_feature_stats(global_indices),
            "task_index": numeric_feature_stats(np.full(length, task_index, dtype=np.int64)),
            **image_stats,
        },
    }
    return episode_meta, episodes_ori_meta, episode_stats


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_info_json(
    *,
    total_episodes: int,
    total_frames: int,
    total_tasks: int,
    fps: int,
    image_width: int,
    image_height: int,
) -> dict[str, Any]:
    video_feature_info = {
        "dtype": "video",
        "shape": [3, image_height, image_width],
        "names": ["channels", "height", "width"],
        "info": {
            "video.height": int(image_height),
            "video.width": int(image_width),
            "video.codec": "h264",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "video.fps": int(fps),
            "video.channels": 3,
            "has_audio": False,
        },
    }
    return {
        "codebase_version": "v2.1",
        "robot_type": "aloha",
        "total_episodes": int(total_episodes),
        "total_frames": int(total_frames),
        "total_tasks": int(total_tasks),
        "total_videos": int(total_episodes * len(REQUIRED_CAMERA_MAP)),
        "total_chunks": int(np.ceil(total_episodes / 1000)),
        "chunks_size": 1000,
        "fps": int(fps),
        "splits": {"train": f"0:{int(total_episodes)}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "observation.state": {
                "dtype": "float32",
                "shape": [16],
                "names": [STATE_FEATURE_NAMES],
            },
            "action": {
                "dtype": "float32",
                "shape": [16],
                "names": [STATE_FEATURE_NAMES],
            },
            "observation.images.cam_high": video_feature_info,
            "observation.images.cam_left_wrist": video_feature_info,
            "observation.images.cam_right_wrist": video_feature_info,
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        },
    }


def build_modality_json() -> dict[str, Any]:
    return {
        "state": {
            "left_xyz_quat_gripper": {"start": 0, "end": 8},
            "right_xyz_quat_gripper": {"start": 8, "end": 16},
        },
        "action": {
            "left_xyz_quat_gripper": {"start": 0, "end": 8},
            "right_xyz_quat_gripper": {"start": 8, "end": 16},
        },
        "video": {
            "cam_high": {"original_key": "observation.images.cam_high"},
            "cam_left_wrist": {"original_key": "observation.images.cam_left_wrist"},
            "cam_right_wrist": {"original_key": "observation.images.cam_right_wrist"},
        },
    }


def build_summary_markdown(
    *,
    archive_stats: list[dict[str, Any]],
    selected_rows: list[dict[str, Any]],
    q1: int,
    q2: int,
    seed: int,
    quota_per_level: int,
    prompt_sources: Counter[str],
) -> str:
    lines = [
        "# RoboTwin LingBot Benchmark",
        "",
        f"- Seed: `{seed}`",
        f"- Quantile cutoffs: `Q33={q1}`, `Q66={q2}`",
        f"- Quota per level: `{quota_per_level}`",
        f"- Total selected episodes: `{len(selected_rows)}`",
        "",
        "## Selected counts by level",
        "",
    ]
    for level in LEVELS:
        level_rows = [row for row in selected_rows if row["complexity_level"] == level]
        task_counts = Counter(row["task_name"] for row in level_rows)
        lines.append(
            f"- {level}: `{len(level_rows)}` episodes across `{len(task_counts)}` tasks "
            f"(min `{min(task_counts.values()) if task_counts else 0}`, "
            f"max `{max(task_counts.values()) if task_counts else 0}` per task)"
        )
    lines += [
        "",
        "## Prompt sources",
        "",
    ]
    for key, value in sorted(prompt_sources.items()):
        lines.append(f"- {key}: `{value}`")

    lines += [
        "",
        "## Archive schema probes",
        "",
    ]
    for stat in archive_stats[: min(12, len(archive_stats))]:
        probe = stat.get("schema_probe") or {}
        lines.append(
            f"- {stat['task_name']} / {stat['variant']}: "
            f"`valid={probe.get('valid')}` "
            f"`episode_length={probe.get('episode_length')}` "
            f"`frame={probe.get('frame_width')}x{probe.get('frame_height')}`"
        )
    return "\n".join(lines) + "\n"


def prepare_output_dirs(output_dir: Path, analysis_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "meta").mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)

    task_filter = normalize_task_filter(args.tasks)
    variant_filter = normalize_task_filter(args.variants) or set(SOURCE_VARIANTS)
    prepare_output_dirs(args.output_dir, args.analysis_dir)

    archive_specs = list_aloha_archives(
        download_dir=args.download_dir,
        task_filter=task_filter,
        variant_filter=variant_filter,
    )
    if not archive_specs:
        raise RuntimeError("No official RoboTwin Aloha archives matched the requested filters.")
    print(f"[plan] matched {len(archive_specs)} official RoboTwin Aloha archives")

    if not args.skip_download:
        download_archives(
            archive_specs,
            download_dir=args.download_dir,
            max_workers=args.max_download_workers,
        )
    else:
        missing = [spec.repo_path for spec in archive_specs if not spec.local_zip_path.exists()]
        if missing:
            raise FileNotFoundError(
                f"--skip-download was set, but {len(missing)} archives are missing locally. "
                f"Examples: {missing[:5]}"
            )

    all_candidates: list[dict[str, Any]] = []
    archive_stats: list[dict[str, Any]] = []
    for spec in archive_specs:
        candidates, stats = index_archive(spec)
        all_candidates.extend(candidates)
        archive_stats.append(stats)
        print(
            f"[indexed] {spec.task_name}/{spec.variant}: "
            f"{stats['counters'].get('eligible_pre_hdf5_validation', 0)} eligible"
        )

    if not all_candidates:
        raise RuntimeError("No candidate RoboTwin episodes were found after success filtering.")

    step_counts = [int(row["step_count"]) for row in all_candidates]
    q1, q2 = compute_cutoffs(step_counts)
    for row in all_candidates:
        row["complexity_level"] = assign_level(int(row["step_count"]), q1, q2)

    prompt_bank, prompt_folder_map = load_lingbot_prompt_bank(args.prompt_cache_dir)
    print(f"[prompt-bank] loaded {len(prompt_bank)} task-level prompt banks from LingBot reference data")

    zip_cache = ZipFileCache()
    try:
        selected_rows: list[dict[str, Any]] = []
        rejected_rows: list[dict[str, Any]] = []
        for level in LEVELS:
            level_rows = [row for row in all_candidates if row["complexity_level"] == level]
            chosen, rejected = choose_valid_round_robin_subset(
                level_rows,
                level=level,
                quota=args.quota_per_level,
                seed=args.seed,
                zip_cache=zip_cache,
            )
            for rank, row in enumerate(chosen):
                row["level_rank"] = rank
            selected_rows.extend(chosen)
            rejected_rows.extend(rejected)
            print(
                f"[select] {level}: kept {len(chosen)} / requested {args.quota_per_level}, "
                f"rejected {len(rejected)} invalid candidates"
            )

        assign_prompts(selected_rows, prompt_bank=prompt_bank, seed=args.seed)
        prompt_sources = Counter(row["prompt_source"] for row in selected_rows)

        prompt_to_idx: dict[str, int] = {}
        tasks_rows: list[dict[str, Any]] = []
        for row in selected_rows:
            prompt = row["prompt"]
            if prompt not in prompt_to_idx:
                prompt_to_idx[prompt] = len(prompt_to_idx)
                tasks_rows.append({"task_index": prompt_to_idx[prompt], "task": prompt})

        selected_rows.sort(key=lambda item: (LEVELS.index(item["complexity_level"]), item["level_rank"]))

        episodes_rows: list[dict[str, Any]] = []
        episodes_ori_rows: list[dict[str, Any]] = []
        episodes_stats_rows: list[dict[str, Any]] = []
        global_frame_offset = 0
        for episode_index, row in enumerate(selected_rows):
            task_index = prompt_to_idx[row["prompt"]]
            episode_row, episode_ori_row, episode_stats_row = convert_episode(
                row,
                output_dir=args.output_dir,
                episode_index=episode_index,
                task_index=task_index,
                global_frame_offset=global_frame_offset,
                fps=args.fps,
                image_width=args.image_width,
                image_height=args.image_height,
                zip_cache=zip_cache,
            )
            episodes_rows.append(episode_row)
            episodes_ori_rows.append(episode_ori_row)
            episodes_stats_rows.append(episode_stats_row)
            global_frame_offset += int(episode_row["length"])
            print(
                f"[convert] episode_{episode_index:06d} "
                f"{row['task_name']} {row['variant']} "
                f"{row['complexity_level']} H={row['step_count']}"
            )
    finally:
        zip_cache.close()

    write_jsonl(args.output_dir / "meta/tasks.jsonl", tasks_rows)
    write_jsonl(args.output_dir / "meta/episodes.jsonl", episodes_rows)
    write_jsonl(args.output_dir / "meta/episodes_ori.jsonl", episodes_ori_rows)
    write_jsonl(args.output_dir / "meta/episodes_stats.jsonl", episodes_stats_rows)
    (args.output_dir / "meta/info.json").write_text(
        json.dumps(
            build_info_json(
                total_episodes=len(episodes_rows),
                total_frames=sum(int(row["length"]) for row in episodes_rows),
                total_tasks=len(tasks_rows),
                fps=args.fps,
                image_width=args.image_width,
                image_height=args.image_height,
            ),
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )
    (args.output_dir / "meta/modality.json").write_text(
        json.dumps(build_modality_json(), indent=2, ensure_ascii=False) + "\n"
    )

    selected_manifest = {
        "seed": int(args.seed),
        "quota_per_level": int(args.quota_per_level),
        "source_dataset_id": ROBOTWIN_DATASET_ID,
        "source_scope": f"official Aloha archives only: {', '.join(sorted(variant_filter))}",
        "prompt_reference_dataset_id": LINGBOT_DATASET_ID,
        "quantile_cutoffs": {"q33": int(q1), "q66": int(q2)},
        "num_candidates_after_success_filter": int(len(all_candidates)),
        "num_selected": int(len(episodes_rows)),
        "archive_count": int(len(archive_specs)),
        "prompt_bank_task_count": int(len(prompt_bank)),
        "prompt_bank_folder_map": prompt_folder_map,
        "selected_episodes": episodes_rows,
        "rejected_candidates": rejected_rows,
        "archive_stats": archive_stats,
    }
    (args.output_dir / "meta/benchmark_summary.json").write_text(
        json.dumps(selected_manifest, indent=2, ensure_ascii=False) + "\n"
    )
    (args.analysis_dir / "summary.md").write_text(
        build_summary_markdown(
            archive_stats=archive_stats,
            selected_rows=selected_rows,
            q1=q1,
            q2=q2,
            seed=args.seed,
            quota_per_level=args.quota_per_level,
            prompt_sources=prompt_sources,
        )
    )

    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "num_candidates_after_success_filter": len(all_candidates),
                "num_selected": len(episodes_rows),
                "quantile_cutoffs": {"q33": q1, "q66": q2},
                "prompt_sources": dict(prompt_sources),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
