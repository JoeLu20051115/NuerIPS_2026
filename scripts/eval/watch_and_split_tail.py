#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_BIN = REPO_ROOT / "external_repos" / "DreamDojo" / ".venv" / "bin" / "python"
RUNNER = REPO_ROOT / "scripts" / "eval" / "run_robotwin_lingbot_smoke.py"
DEFAULT_MODES = ["task_token_only", "dual_llm", "llm_val"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Wait for GPU1 shard completion, then split all remaining rollout jobs across GPU1/GPU2."
    )
    parser.add_argument(
        "--gpu1-current-json",
        type=Path,
        default=REPO_ROOT / "evaluation_results_dualsystem/robotwin_lingbot_full_3way_gpu1_shard0of2.json",
    )
    parser.add_argument(
        "--gpu2-current-json",
        type=Path,
        default=REPO_ROOT / "evaluation_results_dualsystem/robotwin_lingbot_full_3way_gpu2_shard1of2.json",
    )
    parser.add_argument(
        "--gpu1-tail-json",
        type=Path,
        default=REPO_ROOT / "evaluation_results_dualsystem/robotwin_lingbot_full_3way_tail_gpu1.json",
    )
    parser.add_argument(
        "--gpu2-tail-json",
        type=Path,
        default=REPO_ROOT / "evaluation_results_dualsystem/robotwin_lingbot_full_3way_tail_gpu2.json",
    )
    parser.add_argument(
        "--gpu1-tail-save-root",
        type=Path,
        default=REPO_ROOT / "evaluation_results_dualsystem/robotwin_lingbot_full_3way_tail_gpu1_artifacts",
    )
    parser.add_argument(
        "--gpu2-tail-save-root",
        type=Path,
        default=REPO_ROOT / "evaluation_results_dualsystem/robotwin_lingbot_full_3way_tail_gpu2_artifacts",
    )
    parser.add_argument(
        "--manifest-json",
        type=Path,
        default=REPO_ROOT / "evaluation_results_dualsystem/robotwin_lingbot_full_3way_tail_manifest.json",
    )
    parser.add_argument("--gpu1-device", type=str, default="1")
    parser.add_argument("--gpu2-device", type=str, default="2")
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--terminate-timeout-seconds", type=float, default=20.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-wait", action="store_true")
    return parser.parse_args()


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if row.get("mean_l2") is not None]
    total_steps = sum(int(row.get("compared_steps", 0)) for row in valid)
    total_good_steps = sum(int(row.get("num_step_pass_l2_lt_0_1", 0)) for row in valid)
    return {
        "num_episodes": len(rows),
        "mean_l2": float(np.mean([row["mean_l2"] for row in valid])) if valid else None,
        "mean_task_progress": float(np.mean([row["task_progress"] for row in rows])) if rows else None,
        "success_rate": float(np.mean([1.0 if row["task_success"] else 0.0 for row in rows])) if rows else None,
        "rate_of_l2_lt_0_1": float(total_good_steps / total_steps) if total_steps > 0 else None,
    }


def load_payload(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing JSON payload: {path}")
    return json.loads(path.read_text())


def find_eval_pids(output_json: Path) -> list[int]:
    proc = subprocess.run(["ps", "-eo", "pid,args"], capture_output=True, text=True, check=False)
    pids: list[int] = []
    needle = str(output_json)
    for line in proc.stdout.splitlines():
        if "run_robotwin_lingbot_smoke.py" not in line:
            continue
        if needle not in line:
            continue
        parts = line.strip().split(None, 1)
        if not parts:
            continue
        try:
            pids.append(int(parts[0]))
        except ValueError:
            continue
    return pids


def wait_until_idle(output_json: Path, poll_seconds: float) -> None:
    while True:
        pids = find_eval_pids(output_json)
        if not pids:
            print(f"[tail-watch] {output_json.name} is idle; proceeding.", flush=True)
            return
        print(
            f"[tail-watch] waiting for {output_json.name} to finish; active pids={pids}",
            flush=True,
        )
        time.sleep(poll_seconds)


def terminate_eval(output_json: Path, poll_seconds: float, timeout_seconds: float) -> None:
    pids = find_eval_pids(output_json)
    if not pids:
        print(f"[tail-watch] no active process for {output_json.name}; nothing to stop.", flush=True)
        return
    print(f"[tail-watch] stopping {output_json.name}; pids={pids}", flush=True)
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        still_running = find_eval_pids(output_json)
        if not still_running:
            print(f"[tail-watch] {output_json.name} stopped cleanly.", flush=True)
            return
        time.sleep(poll_seconds)
    still_running = find_eval_pids(output_json)
    if still_running:
        print(f"[tail-watch] force-killing {output_json.name}; pids={still_running}", flush=True)
        for pid in still_running:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def combine_state(payloads: list[dict[str, Any]], modes: list[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    episodes_by_id: dict[int, dict[str, Any]] = {}
    all_results: list[dict[str, Any]] = []
    for payload in payloads:
        for row in payload.get("episodes", []):
            episodes_by_id[int(row["episode_index"])] = row
        all_results.extend(payload.get("results", []))

    completed: dict[int, set[str]] = defaultdict(set)
    for result in all_results:
        completed[int(result["episode_index"])].add(str(result["mode"]))

    remaining_rows: list[dict[str, Any]] = []
    for episode_index, row in episodes_by_id.items():
        remaining_modes = [mode for mode in modes if mode not in completed.get(episode_index, set())]
        if not remaining_modes:
            continue
        enriched = dict(row)
        enriched["_remaining_modes"] = remaining_modes
        enriched["_remaining_jobs"] = len(remaining_modes)
        remaining_rows.append(enriched)

    remaining_rows.sort(
        key=lambda row: (int(row["_remaining_jobs"]), int(row.get("step_count", 0)), int(row["episode_index"])),
        reverse=True,
    )
    return remaining_rows, all_results


def partition_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    buckets = [
        {"rows": [], "job_weight": 0, "step_weight": 0},
        {"rows": [], "job_weight": 0, "step_weight": 0},
    ]
    for row in rows:
        idx = 0
        left = (buckets[0]["job_weight"], buckets[0]["step_weight"])
        right = (buckets[1]["job_weight"], buckets[1]["step_weight"])
        if left > right:
            idx = 1
        buckets[idx]["rows"].append(row)
        buckets[idx]["job_weight"] += int(row["_remaining_jobs"])
        buckets[idx]["step_weight"] += int(row.get("step_count", 0))

    left_rows = sorted(buckets[0]["rows"], key=lambda row: int(row["episode_index"]))
    right_rows = sorted(buckets[1]["rows"], key=lambda row: int(row["episode_index"]))
    stats = {
        "gpu1_jobs": buckets[0]["job_weight"],
        "gpu2_jobs": buckets[1]["job_weight"],
        "gpu1_steps": buckets[0]["step_weight"],
        "gpu2_steps": buckets[1]["step_weight"],
        "gpu1_episodes": len(left_rows),
        "gpu2_episodes": len(right_rows),
    }
    return left_rows, right_rows, stats


def build_tail_payload(
    *,
    meta: dict[str, Any],
    parent_outputs: list[Path],
    assigned_rows: list[dict[str, Any]],
    all_results: list[dict[str, Any]],
    modes: list[str],
) -> dict[str, Any]:
    episode_ids = {int(row["episode_index"]) for row in assigned_rows}
    kept_rows = [{k: v for k, v in row.items() if not k.startswith("_")} for row in assigned_rows]
    kept_results = [result for result in all_results if int(result["episode_index"]) in episode_ids]
    running: dict[str, list[dict[str, Any]]] = {mode: [] for mode in modes}
    for result in kept_results:
        mode = str(result.get("mode"))
        if mode in running:
            running[mode].append(result)
    tail_meta = dict(meta)
    tail_meta["tail_parent_outputs"] = [str(path) for path in parent_outputs]
    tail_meta["tail_created_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    tail_meta["tail_mode"] = "remaining_work_split"
    return {
        "meta": tail_meta,
        "episodes": kept_rows,
        "results": kept_results,
        "summary": {mode: summarize(rows) for mode, rows in running.items()},
    }


def launch_runner(
    *,
    gpu_id: str,
    output_json: Path,
    save_root: Path,
    meta: dict[str, Any],
    modes: list[str],
) -> int:
    benchmark_dir = Path(str(meta["benchmark_dir"]))
    checkpoint_dir = Path(str(meta["checkpoint_dir"]))
    episodes_per_level = int(meta.get("episodes_per_level", 100))
    max_step_count = int(meta.get("max_step_count", 10000))
    seed = int(meta.get("seed", 42))
    judge_model = str(meta.get("judge_model", "gpt-5.4"))
    planner_model = str(meta.get("planner_model", "gpt-5.4"))

    cmd = [
        str(PYTHON_BIN),
        str(RUNNER),
        "--benchmark-dir",
        str(benchmark_dir),
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--modes",
        ",".join(modes),
        "--episodes-per-level",
        str(episodes_per_level),
        "--max-step-count",
        str(max_step_count),
        "--seed",
        str(seed),
        "--judge-model",
        judge_model,
        "--planner-model",
        planner_model,
        "--output-json",
        str(output_json),
        "--save-root",
        str(save_root),
        "--resume",
        "--no-save-video",
    ]
    save_root.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    proc = subprocess.Popen(cmd, env=env, start_new_session=True)
    print(
        f"[tail-watch] launched tail worker on GPU{gpu_id}: pid={proc.pid} output={output_json.name}",
        flush=True,
    )
    return int(proc.pid)


def main() -> None:
    args = parse_args()
    modes = list(DEFAULT_MODES)

    if not args.skip_wait:
        wait_until_idle(args.gpu1_current_json, args.poll_seconds)

    if not args.dry_run:
        terminate_eval(args.gpu2_current_json, args.poll_seconds, args.terminate_timeout_seconds)
        time.sleep(2.0)

    gpu1_payload = load_payload(args.gpu1_current_json)
    gpu2_payload = load_payload(args.gpu2_current_json)
    meta = dict(gpu1_payload.get("meta", {}))
    meta.setdefault("judge_model", "gpt-5.4")
    meta.setdefault("planner_model", "gpt-5.4")
    meta["modes"] = modes

    remaining_rows, all_results = combine_state([gpu1_payload, gpu2_payload], modes)
    total_remaining_jobs = sum(int(row["_remaining_jobs"]) for row in remaining_rows)
    print(
        f"[tail-watch] remaining episodes={len(remaining_rows)} remaining_jobs={total_remaining_jobs}",
        flush=True,
    )
    if not remaining_rows:
        print("[tail-watch] no remaining work; nothing to do.", flush=True)
        return

    gpu1_rows, gpu2_rows, stats = partition_rows(remaining_rows)
    print(f"[tail-watch] split stats: {stats}", flush=True)

    gpu1_tail_payload = build_tail_payload(
        meta=meta,
        parent_outputs=[args.gpu1_current_json, args.gpu2_current_json],
        assigned_rows=gpu1_rows,
        all_results=all_results,
        modes=modes,
    )
    gpu2_tail_payload = build_tail_payload(
        meta=meta,
        parent_outputs=[args.gpu1_current_json, args.gpu2_current_json],
        assigned_rows=gpu2_rows,
        all_results=all_results,
        modes=modes,
    )

    manifest = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_outputs": [str(args.gpu1_current_json), str(args.gpu2_current_json)],
        "tail_outputs": [str(args.gpu1_tail_json), str(args.gpu2_tail_json)],
        "remaining_jobs": total_remaining_jobs,
        "stats": stats,
        "gpu1_episode_indices": [int(row["episode_index"]) for row in gpu1_rows],
        "gpu2_episode_indices": [int(row["episode_index"]) for row in gpu2_rows],
    }

    if args.dry_run:
        print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)
        return

    args.gpu1_tail_json.write_text(json.dumps(gpu1_tail_payload, indent=2, ensure_ascii=False) + "\n")
    args.gpu2_tail_json.write_text(json.dumps(gpu2_tail_payload, indent=2, ensure_ascii=False) + "\n")
    args.manifest_json.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")

    gpu1_pid = launch_runner(
        gpu_id=args.gpu1_device,
        output_json=args.gpu1_tail_json,
        save_root=args.gpu1_tail_save_root,
        meta=meta,
        modes=modes,
    )
    gpu2_pid = launch_runner(
        gpu_id=args.gpu2_device,
        output_json=args.gpu2_tail_json,
        save_root=args.gpu2_tail_save_root,
        meta=meta,
        modes=modes,
    )
    manifest["launched_pids"] = {"gpu1": gpu1_pid, "gpu2": gpu2_pid}
    args.manifest_json.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print("[tail-watch] tail acceleration launched.", flush=True)


if __name__ == "__main__":
    main()
