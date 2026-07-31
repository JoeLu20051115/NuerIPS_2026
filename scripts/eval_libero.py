#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import logging
from pathlib import Path
import time
import traceback
from typing import List, Sequence

import numpy as np

from pi05_libero_repro.protocol import EpisodeInvalid, run_episode
from pi05_libero_repro.records import EpisodeRecord, append_record, load_records


OPENPI_REVISION = "650c5b0283a49c42784fb5055a0507da2c6d347d"
LIBERO_REVISION = "f78abd68ee283de9f9be3c8f7e2a9ad60246e95c"
LIBERO_ENV_RESOLUTION = 256


def pending_episode_indices(
    records: Sequence[EpisodeRecord], checkpoint: str, task_id: int, trials: int
) -> List[int]:
    relevant = sorted(
        record.episode_idx
        for record in records
        if record.checkpoint == checkpoint and record.task_id == task_id
    )
    if relevant != list(range(len(relevant))) or len(relevant) > trials:
        raise ValueError("resume would change policy RNG sequence")
    return list(range(len(relevant), trials))


def video_name(task_id: int, episode_idx: int, success: bool) -> str:
    outcome = "success" if success else "failure"
    return f"task_{task_id:02d}_episode_{episode_idx:02d}_{outcome}.mp4"


def _sha256_array(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _task_ids(value: str, total: int) -> List[int]:
    if value == "all":
        return list(range(total))
    task_ids = [int(item) for item in value.split(",")]
    if len(task_ids) != len(set(task_ids)) or any(task_id < 0 or task_id >= total for task_id in task_ids):
        raise ValueError(f"invalid task ids: {value}")
    return task_ids


def _version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def evaluate(args: argparse.Namespace) -> int:
    import imageio
    from libero.libero import benchmark
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from openpi_client import image_tools
    from openpi_client import websocket_client_policy

    np.random.seed(args.seed)
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite]()
    selected_tasks = _task_ids(args.task_ids, task_suite.n_tasks)

    output_dir = args.output_dir.resolve()
    episodes_path = output_dir / "episodes.jsonl"
    videos_dir = output_dir / "videos"
    existing = load_records(episodes_path)
    if any(record.checkpoint != args.checkpoint_name for record in existing):
        raise ValueError("output directory contains a different checkpoint")
    expected_records = len(selected_tasks) * args.trials
    if existing and not args.diagnostic_resume and len(existing) != expected_records:
        raise ValueError("partial primary run cannot resume without restoring the policy RNG")
    if output_dir.exists() and any(output_dir.iterdir()) and not existing:
        raise ValueError(f"nonempty output directory has no episode log: {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        output_dir / "run.json",
        {
            "checkpoint": args.checkpoint_name,
            "diagnostic_resume": args.diagnostic_resume,
            "host": args.host,
            "libero_revision": LIBERO_REVISION,
            "max_steps": args.max_steps,
            "mujoco": _version("mujoco"),
            "openpi_revision": OPENPI_REVISION,
            "port": args.port,
            "replan_steps": args.replan_steps,
            "robosuite": _version("robosuite"),
            "seed": args.seed,
            "task_ids": selected_tasks,
            "task_suite": args.task_suite,
            "trials": args.trials,
            "wait_steps": args.wait_steps,
        },
    )

    if len(existing) == expected_records:
        logging.info("Evaluation already complete: %s records", len(existing))
        return 0

    client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    for task_id in selected_tasks:
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        if len(initial_states) < args.trials:
            raise ValueError(f"task {task_id} has only {len(initial_states)} initial states")
        task_bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        env = OffScreenRenderEnv(
            bddl_file_name=task_bddl_file,
            camera_heights=LIBERO_ENV_RESOLUTION,
            camera_widths=LIBERO_ENV_RESOLUTION,
        )
        env.seed(args.seed)
        try:
            existing = load_records(episodes_path)
            for episode_idx in pending_episode_indices(
                existing, args.checkpoint_name, task_id, args.trials
            ):
                started = time.monotonic()
                try:
                    outcome = run_episode(
                        env,
                        client,
                        initial_states[episode_idx],
                        task.language,
                        image_tools,
                        max_steps=args.max_steps,
                        wait_steps=args.wait_steps,
                        replan_steps=args.replan_steps,
                    )
                    filename = video_name(task_id, episode_idx, outcome.success)
                    final_video = videos_dir / filename
                    temporary_video = videos_dir / f".{filename}.tmp.mp4"
                    imageio.mimwrite(
                        temporary_video,
                        [np.asarray(frame) for frame in outcome.replay_frames],
                        fps=args.video_fps,
                    )
                    temporary_video.replace(final_video)
                    actions = np.stack(outcome.actions)
                    record = EpisodeRecord(
                        checkpoint=args.checkpoint_name,
                        task_id=task_id,
                        task_name=str(task.language),
                        episode_idx=episode_idx,
                        init_state_sha256=_sha256_array(np.asarray(initial_states[episode_idx])),
                        seed=args.seed,
                        success=outcome.success,
                        valid=True,
                        steps=outcome.steps,
                        inference_requests=outcome.inference_requests,
                        wall_seconds=time.monotonic() - started,
                        exception=None,
                        first_frame_sha256=_sha256_array(outcome.first_frame),
                        action_min=float(actions.min()),
                        action_max=float(actions.max()),
                        action_mean=float(actions.mean()),
                        done=outcome.done,
                        check_success=outcome.check_success,
                        video_path=str(final_video.relative_to(output_dir)),
                    )
                    append_record(episodes_path, record)
                    logging.info(
                        "checkpoint=%s task=%d episode=%d success=%s steps=%d",
                        args.checkpoint_name,
                        task_id,
                        episode_idx,
                        outcome.success,
                        outcome.steps,
                    )
                except Exception as error:
                    _write_json(
                        output_dir / "invalid.json",
                        {
                            "checkpoint": args.checkpoint_name,
                            "episode_idx": episode_idx,
                            "exception": str(error),
                            "task_id": task_id,
                            "traceback": traceback.format_exc(),
                        },
                    )
                    logging.exception("Invalid evaluation episode")
                    return 2
        finally:
            env.close()
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Auditable official-order π₀.₅ LIBERO-10 evaluation")
    parser.add_argument("--checkpoint-name", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--task-suite", default="libero_10", choices=("libero_10",))
    parser.add_argument("--task-ids", default="all")
    parser.add_argument("--seed", default=7, type=int)
    parser.add_argument("--trials", default=50, type=int)
    parser.add_argument("--max-steps", default=520, type=int)
    parser.add_argument("--wait-steps", default=10, type=int)
    parser.add_argument("--replan-steps", default=5, type=int)
    parser.add_argument("--video-fps", default=10, type=int)
    parser.add_argument("--diagnostic-resume", action="store_true")
    return parser


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    return evaluate(_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
