#!/usr/bin/env python3
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


DEFAULT_INPUT_ROOT = Path("data/agibot_worldmodel_2026/extracted/validation/info_dataset")
DEFAULT_GT_ROOT = Path("data/agibot_worldmodel_2026/extracted/validation/gt_dataset")
DEFAULT_OUTPUT_ROOT = Path("results/dreamzero_worldmodel_val_gtcond_joint/ACWM_dataset")


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
        embodiment_tag=EmbodimentTag.OXE_DROID,
        device=device,
        lazy_load=False,
    )


def _load_image(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def _resize_frame(frame: np.ndarray, width: int = 320, height: int = 180) -> np.ndarray:
    if frame.shape[1] == width and frame.shape[0] == height:
        return frame
    return np.asarray(Image.fromarray(frame).resize((width, height), Image.BILINEAR))


def _count_gt_frames(gt_episode_dir: Path) -> int:
    video_dir = gt_episode_dir / "video"
    frame_paths = []
    for pattern in ("frame_*.png", "frame_*.jpg", "frame_*.jpeg"):
        frame_paths.extend(video_dir.glob(pattern))
    gt_count = len(sorted(frame_paths))
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


def _load_state_sequence(h5_path: Path) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(h5_path, "r") as f:
        joint_position = f["state/joint/position"][:]
        effector_position = f["state/effector/position"][:]

    right_joint = joint_position[:, 7:14].astype(np.float64)
    right_gripper = effector_position[:, -1:].astype(np.float64)
    return right_joint, right_gripper


def _repeat_to_horizon(values: np.ndarray, horizon: int) -> np.ndarray:
    if len(values) == 0:
        raise ValueError("Cannot upsample an empty trajectory window.")
    if len(values) == horizon:
        return values
    idx = np.linspace(0, len(values) - 1, horizon)
    left = np.floor(idx).astype(int)
    right = np.ceil(idx).astype(int)
    alpha = (idx - left).reshape(-1, 1)
    if values.ndim == 1:
        alpha = alpha[:, 0]
    return (1.0 - alpha) * values[left] + alpha * values[right]


def _build_chunk_actions(
    joint_sequence: np.ndarray,
    gripper_sequence: np.ndarray,
    start_idx: int,
    stride_frames: int,
    action_horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    end_idx = min(len(joint_sequence), start_idx + stride_frames + 1)
    future_joint = joint_sequence[start_idx + 1 : end_idx]
    future_gripper = gripper_sequence[start_idx + 1 : end_idx]

    if len(future_joint) == 0:
        future_joint = joint_sequence[-1:]
        future_gripper = gripper_sequence[-1:]

    action_joint = _repeat_to_horizon(future_joint, action_horizon).astype(np.float64)
    action_gripper = _repeat_to_horizon(future_gripper, action_horizon).astype(np.float64)
    return action_joint, action_gripper


def _build_obs(
    context_frames: np.ndarray,
    state_joint: np.ndarray,
    state_gripper: np.ndarray,
    action_joint: np.ndarray,
    action_gripper: np.ndarray,
    prompt: str,
) -> dict:
    return {
        "video.exterior_image_1_left": context_frames,
        "video.exterior_image_2_left": context_frames,
        "video.wrist_image_left": context_frames,
        "state.joint_position": state_joint.reshape(1, -1),
        "state.gripper_position": state_gripper.reshape(1, -1),
        "action.joint_position": action_joint,
        "action.gripper_position": action_gripper,
        "annotation.language.language_instruction": prompt,
        "annotation.language.language_instruction_2": prompt,
        "annotation.language.language_instruction_3": prompt,
    }


def _decode_latents(policy: GrootSimPolicy, video_latents: torch.Tensor) -> list[np.ndarray]:
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


def _resize_frames(frames: list[np.ndarray], width: int = 320, height: int = 180) -> list[np.ndarray]:
    return [_resize_frame(frame, width=width, height=height) for frame in frames]


def _reset_generation_state(policy: GrootSimPolicy) -> None:
    action_head = policy.trained_model.action_head
    if hasattr(action_head, "reset_generation_state"):
        action_head.reset_generation_state()
    else:
        action_head.current_start_frame = 0
        action_head.language = None
        action_head.clip_feas = None
        action_head.ys = None
        action_head.kv_cache1 = None
        action_head.kv_cache_neg = None
        action_head.crossattn_cache = None
        action_head.crossattn_cache_neg = None


def _should_use_reset_context(policy: GrootSimPolicy) -> bool:
    action_head = policy.trained_model.action_head
    local_attn_size = getattr(action_head.model, "local_attn_size", -1)
    current_start_frame = getattr(action_head, "current_start_frame", 0)
    return local_attn_size != -1 and current_start_frame >= local_attn_size


def iter_episodes(input_root: Path) -> list[tuple[str, str, Path]]:
    items: list[tuple[str, str, Path]] = []
    for task_dir in sorted(p for p in input_root.iterdir() if p.is_dir()):
        for episode_dir in sorted(p for p in task_dir.iterdir() if p.is_dir()):
            items.append((task_dir.name, episode_dir.name, episode_dir))
    return items


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
    force_single_view: bool,
) -> dict:
    prompt = prompt_template.format(task_id=task_id, episode_id=episode_id)
    init_frame = _resize_frame(_load_image(episode_dir / "frame.png"))
    joint_sequence, gripper_sequence = _load_state_sequence(episode_dir / "proprio_stats.h5")
    gt_count = _count_gt_frames(gt_root / task_id / episode_id)
    num_chunks = max(1, math.ceil(max(gt_count - 1, 0) / stride_frames))

    _reset_generation_state(policy)

    all_latents: list[torch.Tensor] = []
    latent_video: torch.Tensor | None = None
    context_frames = np.expand_dims(init_frame, axis=0)

    t0 = time.perf_counter()
    for chunk_idx in range(num_chunks):
        if chunk_idx > 0 and _should_use_reset_context(policy):
            if reset_context_source == "init_frame":
                context_frames = np.expand_dims(init_frame, axis=0)
        start_idx = min(chunk_idx * stride_frames, len(joint_sequence) - 1)
        action_joint, action_gripper = _build_chunk_actions(
            joint_sequence=joint_sequence,
            gripper_sequence=gripper_sequence,
            start_idx=start_idx,
            stride_frames=stride_frames,
            action_horizon=policy.trained_model.action_head.action_horizon,
        )
        obs = _build_obs(
            context_frames=context_frames,
            state_joint=joint_sequence[start_idx],
            state_gripper=gripper_sequence[start_idx],
            action_joint=action_joint,
            action_gripper=action_gripper,
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

    save_dir = output_root / task_id / episode_id / "0" / "video"
    _save_frames_as_jpg(pred_frames, save_dir)

    return {
        "task_id": task_id,
        "episode_id": episode_id,
        "prompt": prompt,
        "num_chunks": int(num_chunks),
        "num_gt_frames": int(gt_count),
        "num_pred_frames_saved": int(len(pred_frames)),
        "elapsed_sec": float(time.perf_counter() - t0),
        "save_dir": str(save_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run DreamZero with GT-conditioned joint/gripper trajectories on WorldModel val and export EWMBench-format predictions."
    )
    parser.add_argument("--model-path", type=str, default="checkpoints/DreamZero-DROID")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--master-port", type=int, default=29512)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--gt-root", type=Path, default=DEFAULT_GT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--prompt-template", type=str, default="task_{task_id}")
    parser.add_argument("--num-episodes", type=int, default=0, help="0 means all episodes.")
    parser.add_argument("--task-id", type=str, default="", help="Optional single task filter.")
    parser.add_argument(
        "--episode-ids",
        type=str,
        default="",
        help="Optional comma-separated list of episode ids to evaluate.",
    )
    parser.add_argument("--stride-frames", type=int, default=8)
    parser.add_argument("--context-frames-after-first", type=int, default=4)
    parser.add_argument(
        "--force-single-view",
        action="store_true",
        help="Treat the duplicated WorldModel input frame as a true single-view input inside DreamZero.",
    )
    parser.add_argument(
        "--latent-carry-mode",
        type=str,
        default="previous_chunk",
        choices=("previous_chunk", "disabled"),
        help="Whether to pass the previous latent chunk into the next forward pass.",
    )
    parser.add_argument(
        "--reset-context-source",
        type=str,
        default="init_frame",
        choices=("pred_tail", "init_frame"),
        help="Which RGB context to use when the model is about to reset its internal cache. Defaults to init_frame because it is more stable on long episodes.",
    )
    args = parser.parse_args()

    if args.force_single_view:
        os.environ["DREAMZERO_FORCE_SINGLE_VIEW"] = "true"

    policy = _load_model(args.model_path, device=args.device, master_port=args.master_port)
    episodes = iter_episodes(args.input_root)
    if args.task_id:
        episodes = [row for row in episodes if row[0] == args.task_id]
    if args.episode_ids:
        wanted = {item.strip() for item in args.episode_ids.split(",") if item.strip()}
        episodes = [row for row in episodes if row[1] in wanted]
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
            force_single_view=args.force_single_view,
        )
        print(
            f"  prompt={row['prompt']} chunks={row['num_chunks']} "
            f"saved_frames={row['num_pred_frames_saved']} elapsed={row['elapsed_sec']:.1f}s"
        )
        results.append(row)

    print(f"Finished {len(results)} episodes.")


if __name__ == "__main__":
    main()
