#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


GPU_TASKS = (
    ("handover_block", "stamp_seal", "turn_switch", "beat_block_hammer"),
    ("open_microwave", "blocks_ranking_size", "stack_blocks_three"),
    ("place_dual_shoes", "move_can_pot", "stack_bowls_three"),
)
ALL_TASKS = frozenset(task for tasks in GPU_TASKS for task in tasks)


def _task_command(
    config: dict, task: str, args: argparse.Namespace
) -> list[str]:
    return [
        str(args.python),
        "script/eval_lerobot_torch_pi05.py",
        "--config", "policy/pi05/deploy_policy.yml",
        "--overrides",
        "--policy_name", "pi05",
        "--task_name", task,
        "--task_config", config["task_config"],
        "--ckpt_setting", "unified_50tasks",
        "--seed", "0",
        "--tag", args.tag,
        "--instruction_type", config["instruction_type"],
        "--policy_path", str(args.checkpoint),
        "--test_num", str(len(config["tasks"][task])),
        "--tokenizer_path", str(args.tokenizer),
        "--record_videos", "False",
        "--accepted_seeds", json.dumps(config["tasks"][task]),
        "--accepted_instructions", json.dumps(config["instructions"][task]),
        "--baseline_only", "True",
    ]


def _run_worker(gpu: int, tasks: tuple[str, ...], args: argparse.Namespace) -> int:
    config = json.loads(args.protocol.read_text())
    robotwin = args.taco / "third_party" / "Robotwin"
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["PYTHONPATH"] = os.pathsep.join(
        (".", "../lerobot/src", str(args.taco), str(args.taco / "cfn"))
    )
    args.output.mkdir(parents=True, exist_ok=True)
    for task in tasks:
        command = _task_command(config, task, args)
        log = args.output / f"{task}.log"
        with log.open("x", encoding="utf-8") as stream:
            result = subprocess.run(
                command,
                cwd=robotwin,
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if result.returncode:
            return result.returncode
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=int, choices=range(3), required=True)
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--taco", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+")
    args = parser.parse_args()
    tasks = tuple(args.tasks) if args.tasks else GPU_TASKS[args.worker]
    unknown = set(tasks) - ALL_TASKS
    if unknown:
        parser.error(f"unknown tasks: {', '.join(sorted(unknown))}")
    return _run_worker(
        args.worker if args.gpu is None else args.gpu,
        tasks,
        args,
    )


if __name__ == "__main__":
    sys.exit(main())
