#!/usr/bin/env python3
"""
AgiBot DreamDojo 3-way evaluation: task_token_only | dual_llm | val_llm

DreamDojo generates a predicted video given (initial observations + task prompt).
Evaluation uses:
  1. Video-space L2: Compare predicted vs GT video frames pixel-by-pixel
  2. GPT-4o-mini judge: Compare initial / predicted-final / real-final frames

Loads 300 episodes from Agi_L1_150 + Agi_L3_150 manifests.

Usage:
  python scripts/eval/run_agibot_dreamdojo_3way_compare.py \\
      --num-episodes 300 \\
      --output-json evaluation_results_dualsystem/agibot_dreamdojo_3way_compare.json
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import h5py
import imageio.v2 as imageio
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent))

from final_frame_judge import FinalFrameJudge

try:
    from run_dualsystem_evaluation import LLMPlanner
except ImportError as exc:
    raise RuntimeError("LLMPlanner is required") from exc

from llm_planner_val import LLMPlannerWithVAL

# ── Key paths ─────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DREAMDOJO_ROOT = REPO_ROOT / "external_repos" / "DreamDojo"
DEFAULT_CHECKPOINT_DIR = (
    DEFAULT_DREAMDOJO_ROOT / "checkpoints" / "2B_AgiBot_post-train" / "2B_AgiBot_post-train"
)
DEFAULT_CHECKPOINT_PATH = DEFAULT_CHECKPOINT_DIR / "iter_000050000" / "model_ema_bf16.pt"
DEFAULT_DREAMDOJO_PYTHON = DEFAULT_DREAMDOJO_ROOT / ".venv" / "bin" / "python"
DEFAULT_MANIFEST_PATHS = [
    REPO_ROOT / "data" / "final_data1" / "Agi_L1_150" / "meta" / "manifest.json",
    REPO_ROOT / "data" / "final_data1" / "Agi_L3_150" / "meta" / "manifest.json",
]
DEFAULT_SHARED_META = DEFAULT_DREAMDOJO_ROOT / "shared_meta"

MODE_ALIASES = {
    "task_token_only": "task_token_only",
    "description_only": "task_token_only",
    "dual_llm": "dual_llm",
    "llm_val": "val_llm",
    "val_llm": "val_llm",
}
DEFAULT_MODES = ["task_token_only", "dual_llm", "val_llm"]


def canonicalize_mode(mode: str) -> str:
    key = mode.strip().lower()
    if key not in MODE_ALIASES:
        supported = ", ".join(sorted(MODE_ALIASES))
        raise ValueError(f"Unsupported mode '{mode}'. Supported modes: {supported}")
    return MODE_ALIASES[key]


def parse_modes(raw: str) -> list[str]:
    seen: set[str] = set()
    modes: list[str] = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        mode = canonicalize_mode(chunk)
        if mode not in seen:
            seen.add(mode)
            modes.append(mode)
    return modes or list(DEFAULT_MODES)


def resolve_repo_path(path_like: str | Path, repo_root: Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else repo_root / path


def absolute_path_preserve_symlink(path_like: str | Path) -> Path:
    """Make a path absolute without resolving symlinks.

    Virtualenv interpreters often live at `.venv/bin/python` as a symlink to a base
    Python binary. Resolving that symlink drops the virtualenv context and loses the
    environment-specific site-packages.
    """
    path = Path(path_like).expanduser()
    return path if path.is_absolute() else Path.cwd() / path


# ── Episode loading ───────────────────────────────────────────────────────────

def load_episodes(manifest_paths: list[Path], n: int | None) -> list[dict]:
    eps: list[dict] = []
    for mpath in manifest_paths:
        manifest = json.loads(mpath.read_text())
        if isinstance(manifest, dict):
            chunk = manifest.get("episodes", [])
        else:
            chunk = manifest
        eps.extend(chunk)
    return eps[:n] if n is not None else eps


# ── LeRobot dataset builder ───────────────────────────────────────────────────

def _make_lerobot_dataset(
    row: dict,
    tmp_root: Path,
    task_text: str,
    num_frames: int,
    repo_root: Path,
    shared_meta: Path,
    sampling_strategy: str,
) -> Path:
    """
    Build a minimal LeRobot-format dataset directory for one AgiBot episode.
    Returns the path to the created dataset directory.
    """
    episode_id = str(row["episode_id"])
    dataset_dir = tmp_root / f"ep_{episode_id}_{int(time.time() * 1000) % 10_000_000}"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    h5_path = resolve_repo_path(row["h5_path"], repo_root)
    with h5py.File(h5_path, "r") as f:
        # state: 20-dim = [left_arm(7), right_arm(7), left_eff(1), right_eff(1), head(2), waist_pitch(1), waist_lift(1)]
        left_arm = f["state/joint/position"][:, :7]   # (T,7)
        right_arm = f["state/joint/position"][:, 7:14]  # (T,7)
        left_eff = f["state/left_effector/position"][:]  # (T,1)
        right_eff = f["state/right_effector/position"][:]  # (T,1)
        head = f["state/head/position"][:]              # (T,2)
        waist = f["state/waist/position"][:]            # (T,2) [pitch, lift]

        # action: 22-dim
        a_left_arm = f["action/joint/position"][:, :7]
        a_right_arm = f["action/joint/position"][:, 7:14]
        a_left_eff = f["action/left_effector/position"][:]
        a_right_eff = f["action/right_effector/position"][:]
        a_head = f["action/head/position"][:]
        a_waist = f["action/waist/position"][:]
        a_robot_vel = f["action/robot/velocity"][:]    # (T,) scalar

    total_frames = left_arm.shape[0]

    # Ensure 2-D for 1-D arrays
    if left_eff.ndim == 1:
        left_eff = left_eff[:, None]
    if right_eff.ndim == 1:
        right_eff = right_eff[:, None]
    if a_left_eff.ndim == 1:
        a_left_eff = a_left_eff[:, None]
    if a_right_eff.ndim == 1:
        a_right_eff = a_right_eff[:, None]
    if a_robot_vel.ndim == 1:
        a_robot_vel = a_robot_vel[:, None]

    waist_pitch = waist[:, 0:1]
    waist_lift = waist[:, 1:2]
    a_waist_pitch = a_waist[:, 0:1]
    a_waist_lift = a_waist[:, 1:2]

    state_all = np.concatenate(
        [left_arm, right_arm, left_eff, right_eff, head, waist_pitch, waist_lift],
        axis=1,
    ).astype(np.float64)  # (T, 20)

    zeros_col = np.zeros((total_frames, 1), dtype=np.float64)
    action_all = np.concatenate(
        [a_left_arm, a_right_arm, a_left_eff, a_right_eff, a_head,
         a_waist_pitch, a_waist_lift, a_robot_vel, zeros_col],
        axis=1,
    ).astype(np.float64)  # (T, 22)

    fps_val = row.get("fps", 10)
    fps = int(fps_val["head"] if isinstance(fps_val, dict) else fps_val)

    active_start = int(np.clip(row.get("active_frame_start", 0), 0, max(total_frames - 1, 0)))
    active_end = int(
        np.clip(
            row.get("active_frame_end", total_frames - 1),
            active_start,
            max(total_frames - 1, 0),
        )
    )
    target_frames = max(int(num_frames), 1)
    if sampling_strategy == "linspace_active":
        sampled_indices = np.linspace(
            active_start,
            active_end,
            min(target_frames, active_end - active_start + 1),
            dtype=int,
        )
        sampled_indices = np.unique(sampled_indices)
    elif sampling_strategy == "prefix_active":
        sampled_indices = np.arange(
            active_start,
            min(active_start + target_frames, active_end + 1),
            dtype=int,
        )
    elif sampling_strategy == "prefix_full":
        sampled_indices = np.arange(
            0,
            min(target_frames, total_frames),
            dtype=int,
        )
    else:
        raise ValueError(f"Unsupported sampling_strategy: {sampling_strategy}")
    if sampled_indices.size == 0:
        sampled_indices = np.array([active_start], dtype=int)

    state = state_all[sampled_indices]
    action = action_all[sampled_indices]
    frame_indices = sampled_indices.tolist()
    T = len(frame_indices)

    # ── Parquet ───────────────────────────────────────────────────────────────
    data_dir = dataset_dir / "data" / "chunk-000"
    data_dir.mkdir(parents=True, exist_ok=True)
    rows_list = []
    for i, src_idx in enumerate(frame_indices):
        rows_list.append({
            "observation.state": state[i].tolist(),
            "action": action[i].tolist(),
            "frame_index": int(src_idx),
            "episode_index": 0,
            "index": i,
            "task_index": 0,
            "timestamp": float(src_idx) / fps,
        })
    df = pd.DataFrame(rows_list)
    table = pa.Table.from_pandas(df)
    pq.write_table(table, str(data_dir / "episode_000000.parquet"))

    # ── Videos ───────────────────────────────────────────────────────────────
    cam_map = {
        "head": "observation.images.top_head",
        "hand_left": "observation.images.hand_left",
        "hand_right": "observation.images.hand_right",
    }
    for cam_key, lerobot_key in cam_map.items():
        src = resolve_repo_path(row["camera_paths"][cam_key], repo_root)
        vid_dir = dataset_dir / "videos" / "chunk-000" / lerobot_key
        vid_dir.mkdir(parents=True, exist_ok=True)
        dst = vid_dir / "episode_000000.mp4"
        if src.exists():
            dst.symlink_to(src.resolve())
        else:
            raise FileNotFoundError(f"Camera video not found: {src}")

    # ── Metadata ─────────────────────────────────────────────────────────────
    meta_dir = dataset_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    # Symlink modality.json and stats.json from shared_meta
    for fname in ("AgiBot_modality.json", "AgiBot_stats.json"):
        src = shared_meta / fname
        # Target names without the "AgiBot_" prefix
        dst_name = fname.replace("AgiBot_", "")
        dst = meta_dir / dst_name
        if src.exists():
            dst.symlink_to(src.resolve())

    # Probe video dimensions from actual source files
    def _probe_video_shape(video_path: Path) -> tuple[int, int]:
        """Return (height, width) by reading first frame."""
        try:
            reader = imageio.get_reader(str(video_path))
            try:
                frame = reader.get_data(0)
                return int(frame.shape[0]), int(frame.shape[1])
            finally:
                reader.close()
        except Exception:
            return 720, 1280  # fallback

    cam_files = {
        "head": Path(row["camera_paths"]["head"]),
        "hand_left": Path(row["camera_paths"]["hand_left"]),
        "hand_right": Path(row["camera_paths"]["hand_right"]),
    }
    cam_lerobot_keys = {
        "head": "observation.images.top_head",
        "hand_left": "observation.images.hand_left",
        "hand_right": "observation.images.hand_right",
    }

    features = {}
    for cam_key, lerobot_key in cam_lerobot_keys.items():
        src = resolve_repo_path(cam_files[cam_key], repo_root)
        h, w = _probe_video_shape(src)
        features[lerobot_key] = {
            "dtype": "video",
            "shape": [h, w, 3],
            "names": ["height", "width", "channel"],
            "video_info": {
                "video.fps": float(fps),
                "video.codec": "h264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "has_audio": False,
            },
        }

    # info.json — must include features, data_path, video_path, chunks_size
    info = {
        "codebase_version": "v2.1",
        "robot_type": "agibot",
        "total_episodes": 1,
        "total_frames": T,
        "total_tasks": 1,
        "total_videos": len(features),
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": fps,
        "splits": {"full": f"0:{T}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
    }
    (meta_dir / "info.json").write_text(json.dumps(info, indent=2))

    # tasks.jsonl
    (meta_dir / "tasks.jsonl").write_text(
        json.dumps({"task_index": 0, "task": task_text}) + "\n"
    )

    # episodes.jsonl
    (meta_dir / "episodes.jsonl").write_text(
        json.dumps({"episode_index": 0, "tasks": [task_text], "length": T}) + "\n"
    )

    return dataset_dir


# ── DreamDojo inference ───────────────────────────────────────────────────────

def run_dreamdojo_inference(
    dataset_path: Path,
    save_dir: Path,
    num_frames: int,
    log_fn,
    dreamdojo_python: Path,
    dreamdojo_root: Path,
    checkpoint_dir: Path,
    checkpoint_path: Path,
    dreamdojo_timeout: int,
    single_base_index: bool,
    deterministic_uniform_sampling: bool,
    single_chunk: bool,
    chunk_size: int | None,
    start_frame_idx: int | None,
    num_latent_conditional_frames: int | None,
    experiment_override: str | None,
    guidance: int | None,
) -> Path | None:
    """
    Run DreamDojo subprocess inference.
    Returns path to the predicted video, or None on failure.
    """
    output_dir = save_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(dreamdojo_python),
        str(dreamdojo_root / "examples" / "action_conditioned.py"),
        "-o", str(output_dir),
        "--checkpoints-dir", str(checkpoint_dir),
        "--checkpoint-path", str(checkpoint_path),
        "--save-dir", str(save_dir),
        "--num-frames", str(num_frames),
        "--num-samples", "1",
        "--dataset-path", str(dataset_path),
        "--data-split", "full",
    ]
    if experiment_override:
        cmd.extend(["--experiment", str(experiment_override)])
    if single_base_index:
        cmd.append("--single-base-index")
    if deterministic_uniform_sampling:
        cmd.append("--deterministic-uniform-sampling")
    if single_chunk:
        cmd.append("--single-chunk")
    if guidance is not None:
        cmd.extend(["--guidance", str(guidance)])
    if chunk_size is not None:
        cmd.extend(["--chunk-size", str(chunk_size)])
    if start_frame_idx is not None:
        cmd.extend(["--start-frame-idx", str(start_frame_idx)])
    if num_latent_conditional_frames is not None:
        cmd.extend(
            [
                "--num-latent-conditional-frames",
                str(num_latent_conditional_frames),
            ]
        )

    env = os.environ.copy()
    env["PYTHONPATH"] = str(dreamdojo_root)
    # Do NOT force CUDA_VISIBLE_DEVICES — let the parent env manage it.
    timeout_s = None if dreamdojo_timeout <= 0 else dreamdojo_timeout

    try:
        result = subprocess.run(
            cmd,
            env=env,
            cwd=str(dreamdojo_root),
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        if result.returncode != 0:
            log_fn(f"    [DreamDojo] returncode={result.returncode}")
            log_fn(f"    [DreamDojo] stderr: {result.stderr[-600:]}")
            return None
    except subprocess.TimeoutExpired:
        timeout_label = "disabled" if timeout_s is None else f"{timeout_s}s"
        log_fn(f"    [DreamDojo] TIMEOUT after {timeout_label}")
        return None
    except Exception as exc:
        log_fn(f"    [DreamDojo] subprocess error: {exc}")
        return None

    iter_dir_name = checkpoint_path.parent.name
    pred_video = save_dir / iter_dir_name / "0000_pred.mp4"
    if not pred_video.exists():
        # Attempt fallback: any *.mp4 under iter dir
        iter_dir = save_dir / iter_dir_name
        if iter_dir.exists():
            hits = sorted(iter_dir.glob("*.mp4"))
            if hits:
                return hits[0]
        log_fn(f"    [DreamDojo] pred video not found at {pred_video}")
        return None

    return pred_video


# ── Video L2 computation ──────────────────────────────────────────────────────

def _count_frames(reader) -> int:
    """Count frames without loading them all (count_frames may return inf for some codecs)."""
    try:
        n = reader.count_frames()
        if n < 1 or n > 100_000:
            raise ValueError("bad count")
        return n
    except Exception:
        # Fallback: seek-based count
        count = 0
        while True:
            try:
                reader.get_data(count)
                count += 1
            except Exception:
                break
        return count


def compute_frame_l2s(
    gt_video: Path,
    pred_video: Path,
    eval_steps: int,
    gt_start_frame: int | None = None,
    gt_end_frame: int | None = None,
) -> list[float]:
    """Compare sampled frames between pred and GT using indexed reads (avoids loading all frames)."""
    from PIL import Image as _PIL_Image

    gt_reader = imageio.get_reader(str(gt_video))
    pred_reader = imageio.get_reader(str(pred_video))
    try:
        n_gt = _count_frames(gt_reader)
        n_pred = _count_frames(pred_reader)
    except Exception:
        gt_reader.close()
        pred_reader.close()
        return []

    gt_start = int(np.clip(0 if gt_start_frame is None else gt_start_frame, 0, max(n_gt - 1, 0)))
    gt_end = int(np.clip((n_gt - 1) if gt_end_frame is None else gt_end_frame, gt_start, max(n_gt - 1, 0)))
    gt_span = max(gt_end - gt_start + 1, 1)

    n = min(eval_steps, gt_span, n_pred)
    if n == 0:
        gt_reader.close()
        pred_reader.close()
        return []

    # Sample indices uniformly across the pred video; map proportionally to GT
    pred_indices = np.linspace(0, n_pred - 1, n, dtype=int)
    gt_indices = (
        gt_start + (pred_indices / max(n_pred - 1, 1) * max(gt_span - 1, 0)).astype(int)
    )
    gt_indices = np.clip(gt_indices, gt_start, gt_end)

    l2s: list[float] = []
    for gi, pi in zip(gt_indices, pred_indices):
        try:
            gt_f = gt_reader.get_data(int(gi)).astype(np.float32) / 255.0
            pred_f = pred_reader.get_data(int(pi)).astype(np.float32) / 255.0
        except Exception:
            continue
        # Resize GT to pred resolution if dimensions differ
        if gt_f.shape[:2] != pred_f.shape[:2]:
            ph, pw = pred_f.shape[:2]
            gt_img = _PIL_Image.fromarray((gt_f * 255).astype(np.uint8)).resize((pw, ph), _PIL_Image.BILINEAR)
            gt_f = np.array(gt_img).astype(np.float32) / 255.0
        l2s.append(float(np.sqrt(np.mean((gt_f - pred_f) ** 2))))

    gt_reader.close()
    pred_reader.close()
    return l2s


# ── Helper: read first / last frame from video ────────────────────────────────

def _first_frame(video_path: Path) -> np.ndarray:
    reader = imageio.get_reader(str(video_path))
    try:
        return reader.get_data(0)
    finally:
        reader.close()


def _last_frame(video_path: Path) -> np.ndarray:
    reader = imageio.get_reader(str(video_path))
    try:
        n = reader.count_frames()
        return reader.get_data(max(n - 1, 0))
    finally:
        reader.close()


# ── Planner ───────────────────────────────────────────────────────────────────

class PlannerSet:
    def __init__(self) -> None:
        self._dual: LLMPlanner | None = None
        self._val: LLMPlannerWithVAL | None = None

    @property
    def dual(self) -> LLMPlanner:
        if self._dual is None:
            self._dual = LLMPlanner(use_mock=False, temperature=0.0)
        return self._dual

    @property
    def val(self) -> LLMPlannerWithVAL:
        if self._val is None:
            self._val = LLMPlannerWithVAL()
        return self._val

    def plan(
        self,
        mode: str,
        task: str,
        max_retries: int = 3,
    ) -> tuple[list[str], dict, float]:
        if mode == "task_token_only":
            return [task], {"planner_mode": "disabled"}, 0.0

        for attempt in range(max_retries):
            try:
                t0 = time.perf_counter()
                if mode == "dual_llm":
                    sub, meta = self.dual.plan(task, {})
                else:  # val_llm
                    sub, meta = self.val.plan(task, {})
                return sub, meta, time.perf_counter() - t0
            except Exception as exc:
                if attempt < max_retries - 1:
                    time.sleep(2.0 * (attempt + 1))
                else:
                    raise


# ── Per-episode runner ────────────────────────────────────────────────────────

class AgiBotDreamDojoRunner:
    def __init__(
        self,
        save_root: Path,
        tmp_root: Path,
        repo_root: Path,
        shared_meta: Path,
        dreamdojo_root: Path,
        dreamdojo_python: Path,
        checkpoint_dir: Path,
        checkpoint_path: Path,
        dreamdojo_timeout: int,
        eval_steps: int,
        num_frames: int,
        judge_model: str,
        success_threshold: float,
        use_full_video_l2: bool,
        sampling_strategy: str,
        dd_single_base_index: bool,
        dd_deterministic_uniform_sampling: bool,
        dd_single_chunk: bool,
        dd_chunk_size: int | None,
        dd_start_frame_idx: int | None,
        dd_num_latent_conditional_frames: int | None,
        dd_experiment: str | None,
        dd_guidance: int | None,
        log_fn,
    ) -> None:
        self.save_root = save_root
        self.tmp_root = tmp_root
        self.repo_root = repo_root
        self.shared_meta = shared_meta
        self.dreamdojo_root = dreamdojo_root
        self.dreamdojo_python = dreamdojo_python
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_path = checkpoint_path
        self.dreamdojo_timeout = dreamdojo_timeout
        self.eval_steps = eval_steps
        self.num_frames = num_frames
        self.judge = FinalFrameJudge(judge_model)
        self.success_threshold = success_threshold
        self.use_full_video_l2 = use_full_video_l2
        self.sampling_strategy = sampling_strategy
        self.dd_single_base_index = dd_single_base_index
        self.dd_deterministic_uniform_sampling = dd_deterministic_uniform_sampling
        self.dd_single_chunk = dd_single_chunk
        self.dd_chunk_size = dd_chunk_size
        self.dd_start_frame_idx = dd_start_frame_idx
        self.dd_num_latent_conditional_frames = dd_num_latent_conditional_frames
        self.dd_experiment = dd_experiment
        self.dd_guidance = dd_guidance
        self.planners = PlannerSet()
        self.log = log_fn

        self.save_root.mkdir(parents=True, exist_ok=True)
        self.tmp_root.mkdir(parents=True, exist_ok=True)

    def run_episode(self, row: dict, mode: str) -> dict:
        episode_id = str(row["episode_id"])
        task = row.get("english_task_name") or row.get("task_group", "")

        # 1. Plan
        sub_instructions, planner_meta, plan_time = self.planners.plan(mode, task)

        # For DreamDojo, the generation prompt = first sub-instruction
        gen_prompt = sub_instructions[0] if sub_instructions else task

        # 2. Build LeRobot dataset with the overridden task text
        tmp_dataset = None
        save_dir = None
        try:
            tmp_dataset = _make_lerobot_dataset(
                row=row,
                tmp_root=self.tmp_root,
                task_text=gen_prompt,
                num_frames=self.num_frames,
                repo_root=self.repo_root,
                shared_meta=self.shared_meta,
                sampling_strategy=self.sampling_strategy,
            )

            # 3. Run DreamDojo inference
            save_dir = (
                self.save_root
                / f"{episode_id}_{mode}_{int(time.time() * 1000) % 10_000_000}"
            )
            save_dir.mkdir(parents=True, exist_ok=True)

            pred_video = run_dreamdojo_inference(
                dataset_path=tmp_dataset,
                save_dir=save_dir,
                num_frames=self.num_frames,
                log_fn=self.log,
                dreamdojo_python=self.dreamdojo_python,
                dreamdojo_root=self.dreamdojo_root,
                checkpoint_dir=self.checkpoint_dir,
                checkpoint_path=self.checkpoint_path,
                dreamdojo_timeout=self.dreamdojo_timeout,
                single_base_index=self.dd_single_base_index,
                deterministic_uniform_sampling=self.dd_deterministic_uniform_sampling,
                single_chunk=self.dd_single_chunk,
                chunk_size=self.dd_chunk_size,
                start_frame_idx=self.dd_start_frame_idx,
                num_latent_conditional_frames=self.dd_num_latent_conditional_frames,
                experiment_override=self.dd_experiment,
                guidance=self.dd_guidance,
            )

            if pred_video is None or not pred_video.exists():
                raise RuntimeError("DreamDojo produced no output video")

            # 4. GT video = head camera
            gt_video = resolve_repo_path(row["camera_paths"]["head"], self.repo_root)

            # 5. Video L2
            active_start = int(row.get("active_frame_start", 0))
            active_end = int(row.get("active_frame_end", max(int(row.get("camera_frames", {}).get("head", 1)) - 1, 0)))
            l2s = compute_frame_l2s(
                gt_video,
                pred_video,
                self.eval_steps,
                gt_start_frame=None if self.use_full_video_l2 else active_start,
                gt_end_frame=None if self.use_full_video_l2 else active_end,
            )

            # 6. Judge
            initial_frame = _first_frame(gt_video)
            real_final = _last_frame(gt_video)
            pred_final = _last_frame(pred_video)
            judged = self.judge.judge(task, initial_frame, real_final, pred_final)
            task_progress = float(judged["task_progress"])
            rule_success = bool(judged.get("rule_success", False))
            task_success = bool((task_progress > self.success_threshold) or rule_success)

        finally:
            # Clean up temp dataset dir
            if tmp_dataset is not None and tmp_dataset.exists():
                try:
                    shutil.rmtree(str(tmp_dataset))
                except Exception:
                    pass

        return {
            "episode_id": episode_id,
            "task": task,
            "mode": mode,
            "sub_instructions": sub_instructions,
            "planner_meta": planner_meta,
            "plan_time": plan_time,
            "mean_l2": float(np.mean(l2s)) if l2s else None,
            "num_steps": len(l2s),
            "num_step_pass_l2_lt_0_1": int(sum(v < 0.1 for v in l2s)),
            "step_alignment_l2_lt_0_1": (
                float(np.mean([v < 0.1 for v in l2s])) if l2s else None
            ),
            "task_progress": task_progress,
            "rule_success": rule_success,
            "task_success": task_success,
            "judge_reason": judged.get("reason", ""),
            "generated_video": str(pred_video),
            "l2_definition": "video_frame_rgb_l2",
        }


# ── Statistics helpers ────────────────────────────────────────────────────────

def _summarize(rows: list[dict]) -> dict:
    valid = [r for r in rows if r.get("mean_l2") is not None]
    if not rows:
        return {
            "num_episodes": 0,
            "mean_l2": None,
            "mean_task_progress": None,
            "success_rate": None,
            "rate_of_l2_lt_0_1": None,
        }
    total_steps = sum(r["num_steps"] for r in rows)
    total_pass = sum(r["num_step_pass_l2_lt_0_1"] for r in rows)
    return {
        "num_episodes": len(rows),
        "mean_l2": float(np.mean([r["mean_l2"] for r in valid])) if valid else None,
        "mean_task_progress": float(np.mean([r["task_progress"] for r in rows])),
        "success_rate": float(np.mean([1.0 if r["task_success"] else 0.0 for r in rows])),
        "rate_of_l2_lt_0_1": (
            float(total_pass / total_steps) if total_steps > 0 else None
        ),
    }


def _save(
    path: Path,
    results: list[dict],
    summary: dict | None = None,
    meta: dict | None = None,
) -> None:
    payload: dict = {"results": results}
    if summary is not None:
        payload["summary"] = summary
    if meta is not None:
        payload["meta"] = meta
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="AgiBot DreamDojo comparison: task_token_only | dual_llm | val_llm"
    )
    parser.add_argument("--manifest-path", type=Path, action="append", default=None,
                        help="Manifest path(s) to evaluate. Can be passed multiple times. Defaults to Agi_L1_150 + Agi_L3_150.")
    parser.add_argument("--dataset-name", default="AgiBot")
    parser.add_argument("--num-episodes", type=int, default=None,
                        help="Optional cap after concatenating manifest episodes. Default: use all episodes in the provided manifests.")
    parser.add_argument(
        "--episode-id",
        action="append",
        default=None,
        help="Optional exact episode_id filter. Can be passed multiple times.",
    )
    parser.add_argument("--eval-steps", type=int, default=49,
                        help="Number of frames to sample for video L2 (matches --num-frames)")
    parser.add_argument("--num-frames", type=int, default=49,
                        help="Number of frames generated by DreamDojo")
    parser.add_argument("--modes", default="task_token_only,dual_llm,val_llm",
                        help="Comma-separated modes. Accepts task_token_only, description_only, dual_llm, val_llm, llm_val.")
    parser.add_argument("--judge-model", default="gpt-4o-mini")
    parser.add_argument("--success-threshold", type=float, default=0.75)
    parser.add_argument("--dreamdojo-root", type=Path, default=DEFAULT_DREAMDOJO_ROOT)
    parser.add_argument("--dreamdojo-python", type=Path, default=DEFAULT_DREAMDOJO_PYTHON)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--checkpoint-path", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--shared-meta", type=Path, default=DEFAULT_SHARED_META)
    parser.add_argument(
        "--dreamdojo-timeout",
        type=int,
        default=1800,
        help="Timeout in seconds for each DreamDojo subprocess. Use 0 or a negative value to disable the timeout.",
    )
    parser.add_argument(
        "--use-full-video-l2",
        action="store_true",
        help="Compare against the whole GT video instead of only the active action window. Default uses active window for closer alignment with prior judged runs.",
    )
    parser.add_argument(
        "--sampling-strategy",
        choices=["linspace_active", "prefix_active", "prefix_full"],
        default="linspace_active",
        help="How to pick frames when building the temporary LeRobot dataset for DreamDojo.",
    )
    parser.add_argument(
        "--dd-single-base-index",
        dest="dd_single_base_index",
        action="store_true",
        default=True,
        help="Pass --single-base-index to DreamDojo (default: on).",
    )
    parser.add_argument(
        "--dd-multi-base-index",
        dest="dd_single_base_index",
        action="store_false",
        help="Do not pass --single-base-index to DreamDojo.",
    )
    parser.add_argument(
        "--dd-deterministic-uniform-sampling",
        dest="dd_deterministic_uniform_sampling",
        action="store_true",
        default=True,
        help="Pass --deterministic-uniform-sampling to DreamDojo (default: on).",
    )
    parser.add_argument(
        "--dd-no-deterministic-uniform-sampling",
        dest="dd_deterministic_uniform_sampling",
        action="store_false",
        help="Do not pass --deterministic-uniform-sampling to DreamDojo.",
    )
    parser.add_argument(
        "--dd-single-chunk",
        action="store_true",
        help="Pass --single-chunk to DreamDojo.",
    )
    parser.add_argument(
        "--dd-chunk-size",
        type=int,
        default=None,
        help="Optional DreamDojo --chunk-size override.",
    )
    parser.add_argument(
        "--dd-start-frame-idx",
        type=int,
        default=None,
        help="Optional DreamDojo --start-frame-idx override.",
    )
    parser.add_argument(
        "--dd-num-latent-conditional-frames",
        type=int,
        default=None,
        help="Optional DreamDojo --num-latent-conditional-frames override.",
    )
    parser.add_argument(
        "--dd-experiment",
        type=str,
        default="dreamdojo_2b_480_640_agibot",
        help="DreamDojo experiment override to pass through. Default preserves current behavior.",
    )
    parser.add_argument(
        "--dd-no-force-experiment",
        action="store_true",
        help="Do not pass --experiment to DreamDojo; let its default model config choose.",
    )
    parser.add_argument(
        "--dd-guidance",
        type=int,
        default=None,
        help="Optional DreamDojo guidance override.",
    )
    parser.add_argument(
        "--output-json",
        default="evaluation_results_dualsystem/agibot_dreamdojo_3way_compare.json",
    )
    parser.add_argument("--log", default="logs/agibot_dreamdojo_3way_compare.log")
    parser.add_argument(
        "--save-root",
        default="evaluation_results_dualsystem/agibot_dreamdojo_3way_videos",
        help="Root directory for DreamDojo save-dirs (predicted videos are kept here)",
    )
    parser.add_argument(
        "--tmp-root",
        default="tmp/agibot_dreamdojo_3way",
        help="Root directory for temporary LeRobot datasets (cleaned up after each episode)",
    )
    args = parser.parse_args()

    manifest_paths = [p.resolve() for p in (args.manifest_path or DEFAULT_MANIFEST_PATHS)]
    selected_modes = parse_modes(args.modes)
    dreamdojo_root = args.dreamdojo_root.resolve()
    dreamdojo_python = absolute_path_preserve_symlink(args.dreamdojo_python)
    checkpoint_dir = args.checkpoint_dir.resolve()
    checkpoint_path = args.checkpoint_path.resolve()
    shared_meta = args.shared_meta.resolve()

    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_root = Path(args.save_root).resolve()
    tmp_root = Path(args.tmp_root).resolve()

    def log(msg: str) -> None:
        print(msg, flush=True)
        with open(log_path, "a") as f:
            f.write(msg + "\n")

    # ── Resume: load already-completed (episode_id, mode) pairs ──────────────
    all_results: list[dict] = []
    running: dict[str, list[dict]] = {m: [] for m in selected_modes}
    done_pairs: set[tuple[str, str]] = set()

    if out_path.exists():
        try:
            prev = json.loads(out_path.read_text())
            for r in prev.get("results", []):
                normalized = dict(r)
                m = canonicalize_mode(str(r["mode"]))
                normalized["mode"] = m
                all_results.append(normalized)
                if m in running:
                    running[m].append(normalized)
                done_pairs.add((str(r["episode_id"]), m))
            if done_pairs:
                log(
                    f"[RESUME] Loaded {len(all_results)} results, "
                    f"{len(done_pairs)} (episode, mode) pairs already done."
                )
        except Exception as exc:
            log(f"Warning: could not load existing results ({exc}), starting fresh.")

    # ── Load episodes ─────────────────────────────────────────────────────────
    episodes = load_episodes(manifest_paths, args.num_episodes)
    if args.episode_id:
        requested_ids = {str(eid) for eid in args.episode_id}
        episodes = [row for row in episodes if str(row.get("episode_id")) in requested_ids]
        if not episodes:
            raise ValueError(
                "No episodes matched --episode-id filter: "
                + ", ".join(sorted(requested_ids))
            )
    meta = {
        "dataset_name": args.dataset_name,
        "manifest_paths": [str(p) for p in manifest_paths],
        "episode_ids": [str(eid) for eid in args.episode_id] if args.episode_id else None,
        "modes": selected_modes,
        "eval_steps": args.eval_steps,
        "num_frames": args.num_frames,
        "judge_model": args.judge_model,
        "success_threshold": args.success_threshold,
        "dreamdojo_root": str(dreamdojo_root),
        "dreamdojo_python": str(dreamdojo_python),
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_path": str(checkpoint_path),
        "shared_meta": str(shared_meta),
        "dreamdojo_timeout": int(args.dreamdojo_timeout),
        "use_full_video_l2": bool(args.use_full_video_l2),
        "sampling_strategy": args.sampling_strategy,
        "dd_single_base_index": bool(args.dd_single_base_index),
        "dd_deterministic_uniform_sampling": bool(args.dd_deterministic_uniform_sampling),
        "dd_single_chunk": bool(args.dd_single_chunk),
        "dd_chunk_size": args.dd_chunk_size,
        "dd_start_frame_idx": args.dd_start_frame_idx,
        "dd_num_latent_conditional_frames": args.dd_num_latent_conditional_frames,
        "dd_experiment": None if args.dd_no_force_experiment else args.dd_experiment,
        "dd_guidance": args.dd_guidance,
    }

    log("=" * 72)
    log(f"{args.dataset_name} DreamDojo compare: {' | '.join(selected_modes)}")
    log(
        f"Episodes: {len(episodes)}  |  "
        f"Eval steps: {args.eval_steps}  |  "
        f"Num frames: {args.num_frames}  |  "
        f"L2 window: {'full_video' if args.use_full_video_l2 else 'active_window'}"
    )
    log(f"Manifests: {', '.join(str(p) for p in manifest_paths)}")
    log("=" * 72)

    runner = AgiBotDreamDojoRunner(
        save_root=save_root,
        tmp_root=tmp_root,
        repo_root=REPO_ROOT,
        shared_meta=shared_meta,
        dreamdojo_root=dreamdojo_root,
        dreamdojo_python=dreamdojo_python,
        checkpoint_dir=checkpoint_dir,
        checkpoint_path=checkpoint_path,
        dreamdojo_timeout=args.dreamdojo_timeout,
        eval_steps=args.eval_steps,
        num_frames=args.num_frames,
        judge_model=args.judge_model,
        success_threshold=args.success_threshold,
        use_full_video_l2=args.use_full_video_l2,
        sampling_strategy=args.sampling_strategy,
        dd_single_base_index=args.dd_single_base_index,
        dd_deterministic_uniform_sampling=args.dd_deterministic_uniform_sampling,
        dd_single_chunk=args.dd_single_chunk,
        dd_chunk_size=args.dd_chunk_size,
        dd_start_frame_idx=args.dd_start_frame_idx,
        dd_num_latent_conditional_frames=args.dd_num_latent_conditional_frames,
        dd_experiment=None if args.dd_no_force_experiment else args.dd_experiment,
        dd_guidance=args.dd_guidance,
        log_fn=log,
    )

    for ep_i, row in enumerate(episodes, 1):
        eid = str(row["episode_id"])
        task = row.get("english_task_name") or row.get("task_group", "?")
        task_short = task[:50]

        # Skip if all requested modes are already done
        if all((eid, m) in done_pairs for m in selected_modes):
            log(f"\n[{ep_i:3d}/{len(episodes)}] ep={eid} | SKIP (all modes done)")
            continue

        log(f"\n[{ep_i:3d}/{len(episodes)}] ep={eid} | {task_short}")

        for mode in selected_modes:
            if (eid, mode) in done_pairs:
                log(f"  [{mode:<18}] skipped (already done)")
                continue

            try:
                result = runner.run_episode(row, mode)
                all_results.append(result)
                running[mode].append(result)
                done_pairs.add((eid, mode))

                # Running stats for this mode
                cur = _summarize(running[mode])
                l2_ep = result["mean_l2"]
                l2_str = f"{l2_ep:.4f}" if l2_ep is not None else "N/A"
                run_l2 = f"{cur['mean_l2']:.4f}" if cur["mean_l2"] is not None else "N/A"
                run_l01 = (
                    f"{cur['rate_of_l2_lt_0_1']:.3f}"
                    if cur["rate_of_l2_lt_0_1"] is not None
                    else "N/A"
                )
                sub_note = (
                    f"\n    sub={result['sub_instructions']}"
                    if mode != "task_token_only"
                    else ""
                )
                log(
                    f"  [{mode:<18}] "
                    f"l2={l2_str} "
                    f"progress={result['task_progress']:.2f} "
                    f"success={'Y' if result['task_success'] else 'N'} "
                    f"plan={result['plan_time']:.1f}s | "
                    f"run: l2={run_l2} "
                    f"prog={cur['mean_task_progress']:.3f} "
                    f"sr={cur['success_rate']:.3f} "
                    f"l2<0.1={run_l01}"
                    + sub_note
                )
            except Exception as exc:
                log(f"  [{mode:<18}] ERROR: {type(exc).__name__}: {str(exc)[:120]}")

        # Checkpoint every 10 episodes
        if ep_i % 10 == 0:
            summary_snap = {m: _summarize(running[m]) for m in selected_modes}
            _save(out_path, all_results, summary_snap, meta)
            log(f"\n  --- Running summary @ {ep_i} episodes ---")
            log(
                f"  {'Mode':<20} {'Mean L2':>8} {'Task Prog':>10} "
                f"{'Success':>8} {'L2<0.1':>8} {'Plan(s)':>8}"
            )
            log("  " + "-" * 66)
            for m in selected_modes:
                s = summary_snap[m]
                plan_times = [r["plan_time"] for r in running[m]]
                mean_pt = float(np.mean(plan_times)) if plan_times else float("nan")
                l2_v = s["mean_l2"] if s["mean_l2"] is not None else float("nan")
                prog_v = s["mean_task_progress"] if s["mean_task_progress"] is not None else float("nan")
                sr_v = s["success_rate"] if s["success_rate"] is not None else float("nan")
                l01_v = s["rate_of_l2_lt_0_1"] if s["rate_of_l2_lt_0_1"] is not None else float("nan")
                log(
                    f"  {m:<20} {l2_v:>8.4f} {prog_v:>10.4f} "
                    f"{sr_v:>8.3f} {l01_v:>8.3f} {mean_pt:>8.2f}"
                )

    # ── Final save and summary ─────────────────────────────────────────────────
    final_summary = {m: _summarize(running[m]) for m in selected_modes}
    _save(out_path, all_results, final_summary, meta)

    log("\n" + "=" * 72)
    log(f"FINAL SUMMARY — {len(episodes)} episodes")
    log("=" * 72)
    log(
        f"{'Dataset':<12} {'Mode':<20} {'Mean L2':>8} {'Task Prog':>10} "
        f"{'Success':>8} {'L2<0.1':>8} {'Plan(s)':>8}"
    )
    log("-" * 74)
    for m in selected_modes:
        s = final_summary[m]
        plan_times = [r["plan_time"] for r in running[m]]
        mean_pt = float(np.mean(plan_times)) if plan_times else float("nan")
        l2_v = s["mean_l2"] if s["mean_l2"] is not None else float("nan")
        prog_v = s["mean_task_progress"] if s["mean_task_progress"] is not None else float("nan")
        sr_v = s["success_rate"] if s["success_rate"] is not None else float("nan")
        l01_v = s["rate_of_l2_lt_0_1"] if s["rate_of_l2_lt_0_1"] is not None else float("nan")
        log(
            f"{args.dataset_name[:12]:<12} {m:<20} {l2_v:>8.4f} {prog_v:>10.4f} "
            f"{sr_v:>8.3f} {l01_v:>8.3f} {mean_pt:>8.2f}"
        )
    log(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
