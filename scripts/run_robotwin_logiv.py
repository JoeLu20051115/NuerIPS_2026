#!/usr/bin/env python3
"""Run LOGIV on RoboTwin without episode selection or repository output files."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

from pi05_libero_repro.logiv.robotwin import ROBOTWIN_TASKS


ROOT = Path(__file__).resolve().parents[1]


def _api_key() -> str:
    value = os.environ.get("OPENAI_API_KEY", "")
    if not value:
        raise RuntimeError("OPENAI_API_KEY is required")
    return value


def _require_path(path: Path, *, directory: bool) -> Path:
    resolved = path.expanduser().resolve()
    valid = resolved.is_dir() if directory else resolved.is_file()
    if not valid:
        kind = "directory" if directory else "file"
        raise ValueError(f"missing {kind}: {resolved}")
    return resolved


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the LOGIV Origin controller on ordinary RoboTwin episodes."
    )
    parser.add_argument("--task", required=True, choices=tuple(ROBOTWIN_TASKS))
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--taco", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--val-binary", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--repair-cfn", type=Path)
    parser.add_argument("--task-config", default="demo_clean")
    parser.add_argument("--instruction-type", default="unseen")
    parser.add_argument("--tag", default="logiv-origin")
    parser.add_argument("--action-chunk-steps", type=int, default=50)
    parser.add_argument("--repair-action-chunk-steps", type=int, default=10)
    parser.add_argument("--base-stall-observations", type=int, default=4)
    parser.add_argument("--vlm-image-detail", choices=("low", "high"), default="low")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.episodes < 1:
        raise ValueError("--episodes must be positive")
    if args.seed < 0:
        raise ValueError("--seed must be nonnegative")
    if args.gpu < 0:
        raise ValueError("--gpu must be nonnegative")

    taco = _require_path(args.taco, directory=True)
    robotwin = _require_path(taco / "third_party/Robotwin", directory=True)
    checkpoint = _require_path(args.checkpoint, directory=True)
    tokenizer = _require_path(args.tokenizer, directory=False)
    val_binary = _require_path(args.val_binary, directory=False)
    python = _require_path(args.python, directory=False)

    env = dict(os.environ)
    env["OPENAI_API_KEY"] = _api_key()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["PYTHONPATH"] = os.pathsep.join(
        (
            ".",
            "../lerobot/src",
            str(taco),
            str(taco / "cfn"),
            str(ROOT / "src"),
        )
    )

    command = [
        str(python),
        "script/eval_lerobot_torch_pi05.py",
        "--config",
        "policy/pi05/deploy_policy.yml",
        "--overrides",
        "--policy_name",
        "pi05",
        "--task_name",
        args.task,
        "--task_config",
        args.task_config,
        "--ckpt_setting",
        "unified_50tasks",
        "--seed",
        str(args.seed),
        "--tag",
        args.tag,
        "--instruction_type",
        args.instruction_type,
        "--policy_path",
        str(checkpoint),
        "--test_num",
        str(args.episodes),
        "--tokenizer_path",
        str(tokenizer),
        "--record_videos",
        "False",
        "--logiv_root",
        str(ROOT),
        "--val_binary",
        str(val_binary),
        "--action_chunk_steps",
        str(args.action_chunk_steps),
        "--repair_action_chunk_steps",
        str(args.repair_action_chunk_steps),
        "--vlm_image_detail",
        args.vlm_image_detail,
        "--max_gpt4o_retries",
        "2",
        "--base_stall_observations",
        str(args.base_stall_observations),
        "--min_base_dispatches",
        "0",
        "--dag_from_start",
        "True",
        "--use_registered_dag_prompts",
        "True",
    ]
    if args.repair_cfn is not None:
        command.extend(
            ["--repair_cfn_path", str(_require_path(args.repair_cfn, directory=False))]
        )

    return subprocess.run(
        command,
        cwd=robotwin,
        env=env,
        check=False,
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
