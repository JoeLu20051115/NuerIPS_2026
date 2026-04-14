#!/usr/bin/env python3
import faulthandler
import os
import sys
import time
from pathlib import Path


def main() -> int:
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
    faulthandler.dump_traceback_later(60, repeat=True)

    print(f"[debug_video2world_init] cwd={Path.cwd()}", flush=True)
    print(f"[debug_video2world_init] checkpoint={checkpoint_path}", flush=True)
    print(
        "[debug_video2world_init] env",
        {
            "LOCAL_RANK": os.getenv("LOCAL_RANK"),
            "RANK": os.getenv("RANK"),
            "WORLD_SIZE": os.getenv("WORLD_SIZE"),
            "CUDA_VISIBLE_DEVICES": os.getenv("CUDA_VISIBLE_DEVICES"),
        },
        flush=True,
    )

    start = time.time()
    print("[debug_video2world_init] importing Video2WorldInference", flush=True)
    from cosmos_predict2._src.predict2.inference.video2world import (  # noqa: PLC0415
        Video2WorldInference,
    )

    print(f"[debug_video2world_init] import done after {time.time() - start:.2f}s", flush=True)
    print("[debug_video2world_init] constructing Video2WorldInference", flush=True)
    cli = Video2WorldInference(
        experiment_name="dreamdojo_2b_480_640_agibot",
        ckpt_path=str(checkpoint_path),
        s3_credential_path="",
        context_parallel_size=1,
        config_file="cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py",
    )
    print(f"[debug_video2world_init] init done after {time.time() - start:.2f}s", flush=True)
    print(f"[debug_video2world_init] model type={type(cli.model).__name__}", flush=True)
    faulthandler.cancel_dump_traceback_later()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
