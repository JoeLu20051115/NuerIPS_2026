#!/usr/bin/env python3
import argparse
import faulthandler
import os
import sys
import time
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--num-frames", type=int, default=49)
    parser.add_argument("--chunk-size", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    dreamdojo_root = repo_root / "external_repos" / "DreamDojo"
    checkpoint_path = (
        dreamdojo_root
        / "checkpoints"
        / "2B_AgiBot_post-train"
        / "2B_AgiBot_post-train"
        / "iter_000050000"
        / "model_ema_bf16.pt"
    )

    os.chdir(dreamdojo_root)
    sys.path.insert(0, str(dreamdojo_root))
    faulthandler.enable(all_threads=True)

    from cosmos_predict2._src.predict2.inference.video2world import (  # noqa: PLC0415
        Video2WorldInference,
    )
    from cosmos_predict2.action_conditioned_config import (  # noqa: PLC0415
        ActionConditionedInferenceArguments,
    )
    from groot_dreams.dataloader import MultiVideoActionDataset  # noqa: PLC0415

    start = time.time()
    print(f"[debug_video2world_generate] cwd={Path.cwd()}", flush=True)
    print(f"[debug_video2world_generate] dataset={args.dataset_path}", flush=True)
    print(f"[debug_video2world_generate] checkpoint={checkpoint_path}", flush=True)
    print(
        "[debug_video2world_generate] env",
        {
            "LOCAL_RANK": os.getenv("LOCAL_RANK"),
            "RANK": os.getenv("RANK"),
            "WORLD_SIZE": os.getenv("WORLD_SIZE"),
            "CUDA_VISIBLE_DEVICES": os.getenv("CUDA_VISIBLE_DEVICES"),
        },
        flush=True,
    )

    cli = Video2WorldInference(
        experiment_name="dreamdojo_2b_480_640_agibot",
        ckpt_path=str(checkpoint_path),
        s3_credential_path="",
        context_parallel_size=1,
        config_file="cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py",
    )
    print(f"[debug_video2world_generate] init done after {time.time() - start:.2f}s", flush=True)

    dataset = MultiVideoActionDataset(
        num_frames=args.num_frames,
        dataset_path=args.dataset_path,
        data_split="full",
        single_base_index=True,
        restrict_len=1,
        deterministic_uniform_sampling=True,
    )
    print(f"[debug_video2world_generate] dataset len={len(dataset)}", flush=True)
    data = dataset[0]
    print(
        "[debug_video2world_generate] sample",
        {
            "video_shape": tuple(data["video"].shape),
            "action_shape": tuple(data["action"].shape),
            "lam_video_shape": tuple(data["lam_video"].shape),
            "video_dtype": str(data["video"].dtype),
            "action_dtype": str(data["action"].dtype),
            "lam_video_dtype": str(data["lam_video"].dtype),
        },
        flush=True,
    )

    inference_args = ActionConditionedInferenceArguments()
    img_array = data["video"].transpose(0, 1)[:1]
    actions = data["action"][: args.num_frames - 1].numpy()
    lam_video = data["lam_video"]
    actions_chunk = actions[: args.chunk_size]
    current_lam_video = lam_video[: args.chunk_size * 2]

    if actions_chunk.shape[0] != args.chunk_size:
        raise RuntimeError(
            f"Expected actions_chunk to have {args.chunk_size} rows, got {actions_chunk.shape[0]}"
        )

    num_video_frames = actions_chunk.shape[0] + 1
    vid_input = torch.cat(
        [img_array, torch.zeros_like(img_array).repeat(num_video_frames - 1, 1, 1, 1)],
        dim=0,
    )
    vid_input = vid_input.to(torch.uint8)
    vid_input = vid_input.unsqueeze(0).permute(0, 2, 1, 3, 4)
    print(
        "[debug_video2world_generate] prepared",
        {
            "img_array_shape": tuple(img_array.shape),
            "vid_input_shape": tuple(vid_input.shape),
            "actions_chunk_shape": tuple(actions_chunk.shape),
            "lam_video_chunk_shape": tuple(current_lam_video.shape),
        },
        flush=True,
    )

    faulthandler.dump_traceback_later(60, repeat=True)
    before_generate = time.time()
    print("[debug_video2world_generate] calling generate_vid2world", flush=True)
    video = cli.generate_vid2world(
        prompt="",
        input_path=vid_input,
        action=torch.from_numpy(actions_chunk).float(),
        guidance=inference_args.guidance,
        num_video_frames=num_video_frames,
        num_latent_conditional_frames=inference_args.num_latent_conditional_frames,
        resolution="480,640",
        seed=args.seed,
        negative_prompt=inference_args.negative_prompt,
        lam_video=current_lam_video,
    )
    faulthandler.cancel_dump_traceback_later()

    print(
        "[debug_video2world_generate] success",
        {
            "elapsed_total_s": round(time.time() - start, 3),
            "elapsed_generate_s": round(time.time() - before_generate, 3),
            "video_shape": tuple(video.shape),
            "video_dtype": str(video.dtype),
        },
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
