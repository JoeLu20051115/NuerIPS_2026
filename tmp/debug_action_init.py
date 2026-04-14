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

    print(f"[debug_action_init] cwd={Path.cwd()}", flush=True)
    print(f"[debug_action_init] checkpoint={checkpoint_path}", flush=True)
    print(
        "[debug_action_init] env",
        {
            "LOCAL_RANK": os.getenv("LOCAL_RANK"),
            "RANK": os.getenv("RANK"),
            "WORLD_SIZE": os.getenv("WORLD_SIZE"),
            "CUDA_VISIBLE_DEVICES": os.getenv("CUDA_VISIBLE_DEVICES"),
        },
        flush=True,
    )

    start = time.time()
    print("[debug_action_init] importing ActionVideo2WorldInference", flush=True)
    from cosmos_predict2._src.predict2.action.inference.inference_pipeline import (  # noqa: PLC0415
        ActionVideo2WorldInference,
    )

    print(f"[debug_action_init] import done after {time.time() - start:.2f}s", flush=True)
    print("[debug_action_init] constructing ActionVideo2WorldInference", flush=True)
    cli = ActionVideo2WorldInference(
        experiment_name="dreamdojo_2b_480_640_agibot",
        ckpt_path=str(checkpoint_path),
        s3_credential_path="credentials/s3_checkpoint.secret",
        context_parallel_size=1,
    )
    print(f"[debug_action_init] init done after {time.time() - start:.2f}s", flush=True)
    print(f"[debug_action_init] model type={type(cli.model).__name__}", flush=True)
    faulthandler.cancel_dump_traceback_later()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
