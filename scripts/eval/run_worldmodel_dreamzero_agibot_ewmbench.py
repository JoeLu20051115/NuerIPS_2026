#!/usr/bin/env python3
"""Run DreamZero with EmbodimentTag.AGIBOT and GT-conditioned AgiBot trajectories.

Reads competition-format validation/test data (info_dataset layout) and emits
EWMBench-format predictions.  Key differences from the legacy OXE_DROID script:

  * Uses EmbodimentTag.AGIBOT (embodiment ID 26) — the correct AgiBot path.
  * Feeds the full dual-arm state:
      state.left_arm_joint_position  (7D)
      state.right_arm_joint_position (7D)
      state.left_effector_position   (1D gripper)
      state.right_effector_position  (1D gripper)
      state.head_position            (2D pan/tilt)
      state.waist_pitch              (1D)
      state.waist_lift               (1D)
  * Video key: video.top_head (from frame.png); hand views are zero-filled
    because competition data only ships the head camera.
  * Action keys mirror the state keys above plus action.robot_velocity (zeros,
    not provided in competition data).

Usage example (smoke test — 1 episode):
    cd /mnt/data3/data_xingrui/lueq/NuerIPS_2026
    python scripts/eval/run_worldmodel_dreamzero_agibot_ewmbench.py \\
        --model-path checkpoints/DreamZero-DROID \\
        --input-root data/agibot_challenge_2026/validation/info_dataset \\
        --gt-root    data/agibot_challenge_2026/validation/gt_dataset \\
        --output-root results/dreamzero_agibot_val_smoke/ACWM_dataset \\
        --num-episodes 1 \\
        --device cuda:0
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

import h5py
import imageio.v2 as imageio
import numpy as np
import torch
import torch.distributed as dist
from einops import rearrange
from PIL import Image
from tianshou.data import Batch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from groot.vla.data.schema import EmbodimentTag
from groot.vla.model.n1_5.sim_policy import GrootSimPolicy


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_INPUT_ROOT = Path("data/agibot_challenge_2026/validation/info_dataset")
DEFAULT_GT_ROOT = Path("data/agibot_challenge_2026/validation/gt_dataset")
DEFAULT_OUTPUT_ROOT = Path("results/dreamzero_agibot_val/ACWM_dataset")

# AgiBot action horizon as used during DreamZero training
_ACTION_HORIZON = 24
# Video resolution expected by the AgiBot transform
_VIDEO_HEIGHT = 176
_VIDEO_WIDTH = 320


# ---------------------------------------------------------------------------
# Distributed init
# ---------------------------------------------------------------------------

def _init_dist(master_port: int) -> None:
    for key, val in [
        ("RANK", "0"),
        ("WORLD_SIZE", "1"),
        ("MASTER_ADDR", "127.0.0.1"),
        ("MASTER_PORT", str(master_port)),
        ("LOCAL_RANK", "0"),
    ]:
        os.environ.setdefault(key, val)
    os.environ.setdefault("DISABLE_TORCH_COMPILE", "true")
    os.environ.setdefault("ATTENTION_BACKEND", "torch")
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
    try:
        import torch._dynamo as dynamo
        dynamo.config.disable = True
        dynamo.config.suppress_errors = True
    except Exception:
        pass
    if not dist.is_initialized():
        dist.init_process_group("nccl")


def _load_model(model_path: str, device: str, master_port: int) -> GrootSimPolicy:
    _init_dist(master_port=master_port)
    return GrootSimPolicy(
        model_path=model_path,
        embodiment_tag=EmbodimentTag.AGIBOT,   # ← KEY CHANGE: use AgiBot path
        device=device,
        lazy_load=False,
    )


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def _load_image(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def _resize_frame(frame: np.ndarray,
                  width: int = _VIDEO_WIDTH,
                  height: int = _VIDEO_HEIGHT) -> np.ndarray:
    if frame.shape[1] == width and frame.shape[0] == height:
        return frame
    return np.asarray(Image.fromarray(frame).resize((width, height), Image.BILINEAR))


def _resize_frames(frames: list[np.ndarray],
                   width: int = _VIDEO_WIDTH,
                   height: int = _VIDEO_HEIGHT) -> list[np.ndarray]:
    return [_resize_frame(f, width=width, height=height) for f in frames]


# ---------------------------------------------------------------------------
# Data loading: AgiBot-format proprioception
# ---------------------------------------------------------------------------

def _load_state_sequence(h5_path: Path) -> dict[str, np.ndarray]:
    """Load and split the full AgiBot dual-arm state from competition h5."""
    with h5py.File(h5_path, "r") as f:
        joint   = f["state/joint/position"][:]       # (T, 14)
        eff     = f["state/effector/position"][:]     # (T, 2)
        head    = f["state/head/position"][:]         # (T, 2)
        waist   = f["state/waist/position"][:]        # (T, 2)

    T = joint.shape[0]
    return {
        "left_arm_joint_position":  joint[:, :7].astype(np.float64),    # (T, 7)
        "right_arm_joint_position": joint[:, 7:14].astype(np.float64),  # (T, 7)
        "left_effector_position":   eff[:, 0:1].astype(np.float64),     # (T, 1)
        "right_effector_position":  eff[:, 1:2].astype(np.float64),     # (T, 1)
        "head_position":            head.astype(np.float64),             # (T, 2)
        "waist_pitch":              waist[:, 0:1].astype(np.float64),   # (T, 1)
        "waist_lift":               waist[:, 1:2].astype(np.float64),   # (T, 1)
        # robot_velocity not available in competition data — zeros
        "robot_velocity":           np.zeros((T, 2), dtype=np.float64),
    }


def _seq_len(states: dict[str, np.ndarray]) -> int:
    return states["left_arm_joint_position"].shape[0]


# ---------------------------------------------------------------------------
# Action building
# ---------------------------------------------------------------------------

def _interp_to_horizon(values: np.ndarray, horizon: int) -> np.ndarray:
    """Linear-interpolate a (K, D) array to exactly (horizon, D)."""
    if len(values) == 0:
        raise ValueError("Cannot interpolate an empty window.")
    if len(values) == horizon:
        return values
    idx = np.linspace(0, len(values) - 1, horizon)
    left  = np.floor(idx).astype(int)
    right = np.ceil(idx).astype(int)
    alpha = (idx - left).reshape(-1, 1)
    return (1.0 - alpha) * values[left] + alpha * values[right]


def _build_chunk_actions(
    states: dict[str, np.ndarray],
    start_idx: int,
    stride_frames: int,
    action_horizon: int,
) -> dict[str, np.ndarray]:
    T = _seq_len(states)
    end_idx = min(T, start_idx + stride_frames + 1)

    action_keys = [
        "left_arm_joint_position",
        "right_arm_joint_position",
        "left_effector_position",
        "right_effector_position",
        "head_position",
        "waist_pitch",
        "waist_lift",
        "robot_velocity",
    ]
    result = {}
    for k in action_keys:
        future = states[k][start_idx + 1 : end_idx]
        if len(future) == 0:
            future = states[k][-1:]
        result[k] = _interp_to_horizon(future, action_horizon).astype(np.float64)
    return result


# ---------------------------------------------------------------------------
# Observation building — AgiBot key format
# ---------------------------------------------------------------------------

def _build_obs(
    context_frames: np.ndarray,      # (N, H, W, 3)
    states: dict[str, np.ndarray],
    actions: dict[str, np.ndarray],
    start_idx: int,
    prompt: str,
) -> dict:
    # Competition data only has the head camera.
    # hand_left / hand_right are zero-filled (absent view).
    zeros = np.zeros_like(context_frames)

    state_at_t = {k: states[k][start_idx : start_idx + 1] for k in [
        "left_arm_joint_position",
        "right_arm_joint_position",
        "left_effector_position",
        "right_effector_position",
        "head_position",
        "waist_pitch",
        "waist_lift",
    ]}

    return {
        # Video — 3 views expected by AgiBot transform
        "video.top_head":   context_frames,
        "video.hand_left":  zeros,   # not available
        "video.hand_right": zeros,   # not available

        # State
        "state.left_arm_joint_position":  state_at_t["left_arm_joint_position"],
        "state.right_arm_joint_position": state_at_t["right_arm_joint_position"],
        "state.left_effector_position":   state_at_t["left_effector_position"],
        "state.right_effector_position":  state_at_t["right_effector_position"],
        "state.head_position":            state_at_t["head_position"],
        "state.waist_pitch":              state_at_t["waist_pitch"],
        "state.waist_lift":               state_at_t["waist_lift"],

        # Action — GT-conditioned
        "action.left_arm_joint_position":  actions["left_arm_joint_position"],
        "action.right_arm_joint_position": actions["right_arm_joint_position"],
        "action.left_effector_position":   actions["left_effector_position"],
        "action.right_effector_position":  actions["right_effector_position"],
        "action.head_position":            actions["head_position"],
        "action.waist_pitch":              actions["waist_pitch"],
        "action.waist_lift":               actions["waist_lift"],
        "action.robot_velocity":           actions["robot_velocity"],

        # Language
        "annotation.agibot.sub_task": prompt,
    }


# ---------------------------------------------------------------------------
# VAE decoding
# ---------------------------------------------------------------------------

def _decode_latents(policy: GrootSimPolicy,
                    video_latents: torch.Tensor) -> list[np.ndarray]:
    action_head = policy.trained_model.action_head
    with torch.no_grad():
        frames = action_head.vae.decode(
            video_latents,
            tiled=action_head.tiled,
            tile_size=(action_head.tile_size_height, action_head.tile_size_width),
            tile_stride=(action_head.tile_stride_height, action_head.tile_stride_width),
        )
    frames = rearrange(frames, "B C T H W -> B T H W C")[0]
    frames = ((frames.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)
    return [frame for frame in frames]


# ---------------------------------------------------------------------------
# Generation state management
# ---------------------------------------------------------------------------

def _reset_generation_state(policy: GrootSimPolicy) -> None:
    action_head = policy.trained_model.action_head
    if hasattr(action_head, "reset_generation_state"):
        action_head.reset_generation_state()
    else:
        for attr in ("current_start_frame", "language", "clip_feas", "ys",
                     "kv_cache1", "kv_cache_neg", "crossattn_cache", "crossattn_cache_neg"):
            setattr(action_head, attr, 0 if attr == "current_start_frame" else None)


def _should_reset_context(policy: GrootSimPolicy) -> bool:
    action_head = policy.trained_model.action_head
    local_attn_size = getattr(action_head.model, "local_attn_size", -1)
    current_start_frame = getattr(action_head, "current_start_frame", 0)
    return local_attn_size != -1 and current_start_frame >= local_attn_size


# ---------------------------------------------------------------------------
# GT frame count
# ---------------------------------------------------------------------------

def _count_gt_frames(gt_episode_dir: Path) -> int:
    video_dir = gt_episode_dir / "video"
    paths = []
    for pat in ("frame_*.png", "frame_*.jpg", "frame_*.jpeg"):
        paths.extend(video_dir.glob(pat))
    gt_count = len(sorted(paths))
    if gt_count == 0:
        raise FileNotFoundError(f"No GT frames found under {video_dir}")
    return gt_count


def _save_frames_as_jpg(frames: list[np.ndarray], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for idx, frame in enumerate(frames):
        imageio.imwrite(output_dir / f"frame_{idx:05d}.jpg", frame, quality=95)


def _resample_frames(frames: list[np.ndarray], target_count: int) -> list[np.ndarray]:
    if target_count <= 0 or len(frames) == target_count:
        return frames
    idx = np.linspace(0, len(frames) - 1, target_count, dtype=int)
    return [frames[i] for i in idx]


# ---------------------------------------------------------------------------
# Episode iteration
# ---------------------------------------------------------------------------

def iter_episodes(input_root: Path) -> list[tuple[str, str, Path]]:
    items: list[tuple[str, str, Path]] = []
    for task_dir in sorted(p for p in input_root.iterdir() if p.is_dir()):
        for ep_dir in sorted(p for p in task_dir.iterdir() if p.is_dir()):
            items.append((task_dir.name, ep_dir.name, ep_dir))
    return items


# ---------------------------------------------------------------------------
# Per-episode inference
# ---------------------------------------------------------------------------

def run_episode(
    policy: GrootSimPolicy,
    task_id: str,
    episode_id: str,
    episode_dir: Path,
    gt_root: Path,
    output_root: Path,
    prompt_template: str,
    stride_frames: int,
    context_frames_after_first: int,
    latent_carry_mode: str,
    reset_context_source: str,
    n_pred: int,
) -> dict:
    prompt = prompt_template.format(task_id=task_id, episode_id=episode_id)

    # Load initial frame and states
    init_frame = _resize_frame(_load_image(episode_dir / "frame.png"))
    states = _load_state_sequence(episode_dir / "proprio_stats.h5")
    T = _seq_len(states)

    gt_count = _count_gt_frames(gt_root / task_id / episode_id)
    num_chunks = max(1, math.ceil(max(gt_count - 1, 0) / stride_frames))

    t0 = time.perf_counter()
    all_trials: list[list[np.ndarray]] = []

    for trial in range(n_pred):
        _reset_generation_state(policy)
        all_latents: list[torch.Tensor] = []
        latent_video: torch.Tensor | None = None
        context_frames = np.expand_dims(init_frame, axis=0)   # (1, H, W, 3)

        for chunk_idx in range(num_chunks):
            if chunk_idx > 0 and _should_reset_context(policy):
                if reset_context_source == "init_frame":
                    context_frames = np.expand_dims(init_frame, axis=0)

            start_idx = min(chunk_idx * stride_frames, T - 1)
            actions = _build_chunk_actions(
                states=states,
                start_idx=start_idx,
                stride_frames=stride_frames,
                action_horizon=policy.trained_model.action_head.action_horizon,
            )
            obs = _build_obs(
                context_frames=context_frames,
                states=states,
                actions=actions,
                start_idx=start_idx,
                prompt=prompt,
            )
            batch = Batch(obs=obs)
            latent_input = latent_video if latent_carry_mode == "previous_chunk" else None
            with torch.no_grad():
                _, video_pred = policy.lazy_joint_forward_causal_gt_cond(
                    batch,
                    latent_video=latent_input,
                )
            video_chunk = video_pred.detach()
            all_latents.append(video_chunk)
            latent_video = video_chunk

            decoded_chunk = _decode_latents(policy, video_chunk)
            resized_chunk = _resize_frames(decoded_chunk)
            if len(resized_chunk) >= context_frames_after_first:
                context_frames = np.stack(resized_chunk[-context_frames_after_first:], axis=0)
            else:
                context_frames = np.stack(resized_chunk, axis=0)

        all_video_latents = torch.cat(all_latents, dim=2)
        pred_frames = _decode_latents(policy, all_video_latents)
        pred_frames = _resample_frames(pred_frames, gt_count)
        all_trials.append(pred_frames)

    # Save all n_pred trials
    for trial_idx, pred_frames in enumerate(all_trials):
        save_dir = output_root / task_id / episode_id / str(trial_idx) / "video"
        _save_frames_as_jpg(pred_frames, save_dir)

    return {
        "task_id": task_id,
        "episode_id": episode_id,
        "prompt": prompt,
        "n_pred": n_pred,
        "num_chunks": int(num_chunks),
        "num_gt_frames": int(gt_count),
        "num_pred_frames_saved": int(len(all_trials[0])),
        "elapsed_sec": float(time.perf_counter() - t0),
        "save_dir": str(output_root / task_id / episode_id),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="DreamZero AgiBot-embodiment world-model inference for competition eval."
    )
    parser.add_argument("--model-path", type=str, default="checkpoints/DreamZero-DROID")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--master-port", type=int, default=29512)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--gt-root", type=Path, default=DEFAULT_GT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--prompt-template", type=str, default="task_{task_id}",
        help="Format string with {task_id} and optionally {episode_id}.",
    )
    parser.add_argument("--num-episodes", type=int, default=0, help="0 = all.")
    parser.add_argument("--task-id", type=str, default="")
    parser.add_argument("--episode-ids", type=str, default="")
    parser.add_argument("--stride-frames", type=int, default=8)
    parser.add_argument("--context-frames-after-first", type=int, default=4)
    parser.add_argument("--n-pred", type=int, default=3,
                        help="Number of prediction trials per episode (diversity).")
    parser.add_argument(
        "--latent-carry-mode", type=str, default="previous_chunk",
        choices=("previous_chunk", "disabled"),
    )
    parser.add_argument(
        "--reset-context-source", type=str, default="init_frame",
        choices=("pred_tail", "init_frame"),
    )
    args = parser.parse_args()

    policy = _load_model(args.model_path, device=args.device, master_port=args.master_port)
    episodes = iter_episodes(args.input_root)

    if args.task_id:
        episodes = [r for r in episodes if r[0] == args.task_id]
    if args.episode_ids:
        wanted = {s.strip() for s in args.episode_ids.split(",") if s.strip()}
        episodes = [r for r in episodes if r[1] in wanted]
    if args.num_episodes > 0:
        episodes = episodes[: args.num_episodes]

    args.output_root.mkdir(parents=True, exist_ok=True)

    results = []
    for idx, (task_id, episode_id, episode_dir) in enumerate(episodes, start=1):
        print(f"[{idx}/{len(episodes)}] task={task_id} episode={episode_id}")
        row = run_episode(
            policy=policy,
            task_id=task_id,
            episode_id=episode_id,
            episode_dir=episode_dir,
            gt_root=args.gt_root,
            output_root=args.output_root,
            prompt_template=args.prompt_template,
            stride_frames=args.stride_frames,
            context_frames_after_first=args.context_frames_after_first,
            latent_carry_mode=args.latent_carry_mode,
            reset_context_source=args.reset_context_source,
            n_pred=args.n_pred,
        )
        print(f"  chunks={row['num_chunks']} frames={row['num_pred_frames_saved']} "
              f"elapsed={row['elapsed_sec']:.1f}s")
        results.append(row)

    print(f"\nFinished {len(results)} episodes. Output: {args.output_root}")


if __name__ == "__main__":
    main()
