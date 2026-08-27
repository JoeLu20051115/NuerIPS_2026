#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / "configs/robotwin/logiv-gpt4o-63-vs-pi05-56.json"
LOGIV_WORKER = ROOT / "scripts/run_robotwin_logiv_10x10.py"
BASELINE_WORKER = ROOT / "scripts/run_robotwin_baseline_10x10.py"
REPORTER = ROOT / "scripts/report_robotwin_logiv.py"
TAG_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def _api_key() -> str:
    value = os.environ.get("OPENAI_API_KEY", "")
    if value.startswith("sk-") and len(value) > 20:
        return value
    raise RuntimeError("set a valid OPENAI_API_KEY for the live LOGIV run")


def _run_tags(output: Path) -> tuple[str, str]:
    name = output.name
    if TAG_RE.fullmatch(name) is None:
        raise ValueError(
            "output directory name must use only letters, digits, '.', '_', and '-'"
        )
    return f"{name}-logiv", f"{name}-baseline"


def _worker_commands(
    args: argparse.Namespace, phase: str, tag: str
) -> list[list[str]]:
    if phase not in {"logiv", "baseline"}:
        raise ValueError(f"unknown phase: {phase}")
    worker = LOGIV_WORKER if phase == "logiv" else BASELINE_WORKER
    commands: list[list[str]] = []
    for worker_id, gpu in enumerate(args.gpus):
        command = [
            sys.executable,
            str(worker),
            "--worker", str(worker_id),
            "--gpu", str(gpu),
            "--protocol", str(args.protocol),
            "--output", str(args.output / "raw" / phase),
            "--tag", tag,
            "--taco", str(args.taco),
            "--checkpoint", str(args.checkpoint),
            "--python", str(args.python),
            "--tokenizer", str(args.tokenizer),
        ]
        if phase == "logiv":
            command.extend(
                (
                    "--logiv-root", str(args.logiv_root),
                    "--val-binary", str(args.val_binary),
                )
            )
        commands.append(command)
    return commands


def _expectations(
    report: dict[str, Any], manifest: dict[str, Any]
) -> list[str]:
    if report.get("strict_protocol_complete") is not True:
        return ["frozen protocol is incomplete"]
    expected = manifest["expected_successes"]
    errors: list[str] = []
    observed_logiv = report.get("successes")
    observed_baseline = report.get("baseline_successes")
    if observed_logiv != expected["logiv"]:
        errors.append(
            f"expected LOGIV {expected['logiv']}, observed {observed_logiv}"
        )
    if observed_baseline != expected["baseline"]:
        errors.append(
            "expected baseline "
            f"{expected['baseline']}, observed {observed_baseline}"
        )
    return errors


def _direct_child(root: Path, name: str, label: str) -> Path:
    resolved_root = root.resolve()
    candidate = (resolved_root / name).resolve()
    try:
        candidate.relative_to(resolved_root)
    except ValueError:
        raise ValueError(f"cleanup path is outside {label}") from None
    if candidate.parent != resolved_root:
        raise ValueError(f"cleanup path is outside {label}")
    return candidate


def _compact(output_root: Path, eval_root: Path, tags: list[str]) -> None:
    raw = _direct_child(output_root, "raw", "output root")
    if raw.exists():
        if not raw.is_dir():
            raise ValueError("output raw path is not a directory")
        shutil.rmtree(raw)
    for tag in tags:
        path = _direct_child(eval_root, tag, "eval root")
        if path.exists():
            if not path.is_dir():
                raise ValueError(f"eval tag path is not a directory: {tag}")
            shutil.rmtree(path)


def _validate_manifest(manifest: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    tasks = manifest.get("tasks")
    instructions = manifest.get("instructions")
    if not isinstance(tasks, dict) or len(tasks) != 10:
        errors.append("manifest must contain exactly ten tasks")
        return errors
    if not isinstance(instructions, dict) or set(instructions) != set(tasks):
        errors.append("manifest task and instruction sets differ")
        return errors
    for task, seeds in tasks.items():
        task_instructions = instructions[task]
        if (
            not isinstance(seeds, list)
            or len(seeds) != 10
            or len(set(seeds)) != 10
            or not isinstance(task_instructions, list)
            or len(task_instructions) != 10
        ):
            errors.append(f"manifest task is not a unique 10-cell set: {task}")
    if manifest.get("expected_successes") != {"logiv": 63, "baseline": 56}:
        errors.append("manifest expected counts are not LOGIV 63 / baseline 56")
    if "checkpoint" in manifest or "seed_selection" in manifest:
        errors.append("manifest contains a machine path or seed-selection history")
    return errors


def _preflight(
    args: argparse.Namespace,
    manifest: dict[str, Any],
    tags: tuple[str, str],
) -> None:
    errors = _validate_manifest(manifest)
    if len(args.gpus) != 3 or len(set(args.gpus)) != 3 or min(args.gpus) < 0:
        errors.append("--gpus must contain three distinct nonnegative integers")
    if args.output.exists():
        errors.append(f"output already exists: {args.output}")
    evaluator = (
        args.taco
        / "third_party/Robotwin/script/eval_lerobot_torch_pi05.py"
    )
    if not evaluator.is_file():
        errors.append(f"missing RoboTwin evaluator: {evaluator}")
    if not args.checkpoint.is_dir():
        errors.append(f"missing checkpoint directory: {args.checkpoint}")
    else:
        for task in manifest.get("repair_cfn_tasks", ()):
            cfn = args.checkpoint / "cfns" / f"{task}_cfn.pt"
            if not cfn.is_file():
                errors.append(f"missing checkpoint CFN: {cfn}")
    if not args.tokenizer.is_file():
        errors.append(f"missing tokenizer: {args.tokenizer}")
    if not args.val_binary.is_file() or not os.access(args.val_binary, os.X_OK):
        errors.append(f"VAL binary is missing or not executable: {args.val_binary}")
    if not args.python.is_file() or not os.access(args.python, os.X_OK):
        errors.append(f"RoboTwin Python is missing or not executable: {args.python}")
    eval_root = args.taco / "third_party/Robotwin/eval_result"
    for tag in tags:
        if _direct_child(eval_root, tag, "eval root").exists():
            errors.append(f"TACO result tag already exists: {tag}")
    try:
        _api_key()
    except RuntimeError as error:
        errors.append(str(error))
    if errors:
        raise RuntimeError("preflight failed:\n- " + "\n- ".join(errors))


def _run_phase(commands: list[list[str]]) -> int:
    processes: list[subprocess.Popen] = []
    try:
        processes = [subprocess.Popen(command) for command in commands]
    except OSError as error:
        for process in processes:
            process.terminate()
        for process in processes:
            process.wait()
        print(f"failed to start worker: {error}", file=sys.stderr)
        return 1
    returncodes = [process.wait() for process in processes]
    return next((code for code in returncodes if code), 0)


def _report_command(
    args: argparse.Namespace, logiv_tag: str
) -> list[str]:
    eval_root = args.taco / "third_party/Robotwin/eval_result"
    return [
        sys.executable,
        str(REPORTER),
        "--config", str(args.protocol),
        "--events-root", str(eval_root / logiv_tag),
        "--baseline-logs", str(args.output / "raw/baseline"),
        "--json", str(args.output / "summary.json"),
        "--markdown", str(args.output / "summary.md"),
    ]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the frozen RoboTwin LOGIV 63 vs pi0.5 56 protocol."
    )
    parser.add_argument("--taco", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--val-binary", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpus", type=int, nargs=3, required=True)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    args.logiv_root = ROOT
    try:
        manifest = json.loads(args.protocol.read_text(encoding="utf-8"))
        tags = _run_tags(args.output)
        _preflight(args, manifest, tags)
    except (OSError, ValueError, json.JSONDecodeError, RuntimeError) as error:
        parser.error(str(error))
    logiv_tag, baseline_tag = tags
    logiv_commands = _worker_commands(args, "logiv", logiv_tag)
    baseline_commands = _worker_commands(args, "baseline", baseline_tag)
    if args.dry_run:
        print(
            json.dumps(
                {"logiv": logiv_commands, "baseline": baseline_commands},
                indent=2,
            )
        )
        return 0

    args.output.mkdir(parents=True)
    shutil.copy2(args.protocol, args.output / "manifest.json")
    metadata = {
        "taco": str(args.taco.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "tokenizer": str(args.tokenizer.resolve()),
        "val_binary": str(args.val_binary.resolve()),
        "python": str(args.python.resolve()),
        "gpus": list(args.gpus),
        "logiv_tag": logiv_tag,
        "baseline_tag": baseline_tag,
    }
    (args.output / "run.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    if _run_phase(logiv_commands):
        print("LOGIV phase failed; raw outputs retained", file=sys.stderr)
        return 1
    if _run_phase(baseline_commands):
        print("baseline phase failed; raw outputs retained", file=sys.stderr)
        return 1
    report_result = subprocess.run(_report_command(args, logiv_tag), check=False)
    if report_result.returncode:
        print("report is incomplete; raw outputs retained", file=sys.stderr)
        return report_result.returncode

    report = json.loads((args.output / "summary.json").read_text(encoding="utf-8"))
    expectation_errors = _expectations(report, manifest)
    eval_root = args.taco / "third_party/Robotwin/eval_result"
    _compact(args.output, eval_root, [logiv_tag, baseline_tag])
    if expectation_errors:
        for error in expectation_errors:
            print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
