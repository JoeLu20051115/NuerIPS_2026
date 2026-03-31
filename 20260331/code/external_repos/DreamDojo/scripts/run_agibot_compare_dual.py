#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "eval"))

from final_frame_judge import FinalFrameJudge
from run_agibot_dreamdojo_3way_compare import (
    DEFAULT_CHECKPOINT_DIR,
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_DREAMDOJO_PYTHON,
    DEFAULT_DREAMDOJO_ROOT,
    DEFAULT_MANIFEST_PATHS,
    DEFAULT_MODES,
    DEFAULT_SHARED_META,
    PlannerSet,
    _first_frame,
    _last_frame,
    _make_lerobot_dataset,
    _summarize,
    absolute_path_preserve_symlink,
    load_episodes,
    parse_modes,
    resolve_repo_path,
    run_dreamdojo_inference,
)


NOTES = {
    "mean_l2_definition": "video-space RGB frame L2 over generated-vs-ground-truth rollout frames",
    "task_progress_definition": "final-frame visual judge against task text and demo final frame",
    "success_definition": "task_progress > threshold OR visual rule_success",
}


def _count_frames(reader) -> int:
    try:
        n = reader.count_frames()
        if 1 <= n <= 100_000:
            return n
    except Exception:
        pass
    i = 0
    while True:
        try:
            reader.get_data(i)
            i += 1
        except Exception:
            return i


def compute_rollout_frame_l2s(gt_video: Path, pred_video: Path, eval_steps: int) -> list[float]:
    gt_reader = imageio.get_reader(str(gt_video))
    pred_reader = imageio.get_reader(str(pred_video))
    try:
        n_gt = _count_frames(gt_reader)
        n_pred = _count_frames(pred_reader)
        n = min(eval_steps, n_gt, n_pred)
        if n <= 0:
            return []

        gt_indices = np.linspace(0, n_gt - 1, n, dtype=int)
        pred_indices = np.linspace(0, n_pred - 1, n, dtype=int)
        l2s: list[float] = []
        for gi, pi in zip(gt_indices, pred_indices):
            gt_frame = gt_reader.get_data(int(gi)).astype(np.float32) / 255.0
            pred_frame = pred_reader.get_data(int(pi)).astype(np.float32) / 255.0
            if gt_frame.shape != pred_frame.shape:
                continue
            l2s.append(float(np.sqrt(np.mean((gt_frame - pred_frame) ** 2))))
        return l2s
    finally:
        gt_reader.close()
        pred_reader.close()


def flatten_pred_video(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src.resolve())


def _relative_to_repo(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except Exception:
        return str(path.resolve())


def resolve_checkpoint_paths(
    checkpoint_dir: Path,
    checkpoint_path: Path,
) -> tuple[Path, Path]:
    env_value = os.environ.get("DREAMDOJO_LOCAL_BASE_CHECKPOINT", "").strip()
    if not env_value:
        return checkpoint_dir.resolve(), checkpoint_path.resolve()

    candidate = Path(env_value).expanduser()
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    candidate = candidate.resolve()

    if candidate.is_file():
        inferred_dir = candidate.parent.parent if candidate.parent.name.startswith("iter_") else checkpoint_dir
        return inferred_dir.resolve(), candidate

    if candidate.is_dir():
        model_path = candidate / "model_ema_bf16.pt"
        if model_path.exists():
            inferred_dir = candidate.parent if (candidate.parent / "latest_checkpoint.txt").exists() else checkpoint_dir
            return inferred_dir.resolve(), model_path.resolve()
        latest_file = candidate / "latest_checkpoint.txt"
        if latest_file.exists():
            last_checkpoint = latest_file.read_text().strip()
            resolved_path = (candidate / last_checkpoint / "model_ema_bf16.pt").resolve()
            return candidate.resolve(), resolved_path

    raise FileNotFoundError(
        f"DREAMDOJO_LOCAL_BASE_CHECKPOINT is set to {candidate}, but it is neither "
        "a checkpoint file, an iter_* directory, nor a checkpoint root."
    )


class LegacyAgiBotCompareRunner:
    def __init__(
        self,
        video_dir: Path,
        work_root: Path,
        tmp_root: Path,
        repo_root: Path,
        shared_meta: Path,
        dreamdojo_root: Path,
        dreamdojo_python: Path,
        checkpoint_dir: Path,
        checkpoint_path: Path,
        dreamdojo_timeout: int,
        eval_steps: int,
        num_frames: int,
        judge_model: str,
        success_threshold: float,
        sampling_strategy: str,
        dd_single_base_index: bool,
        dd_deterministic_uniform_sampling: bool,
        dd_single_chunk: bool,
        dd_chunk_size: int | None,
        dd_start_frame_idx: int | None,
        dd_num_latent_conditional_frames: int | None,
        dd_experiment: str | None,
        dd_guidance: int | None,
        log_fn,
    ) -> None:
        self.video_dir = video_dir
        self.work_root = work_root
        self.tmp_root = tmp_root
        self.repo_root = repo_root
        self.shared_meta = shared_meta
        self.dreamdojo_root = dreamdojo_root
        self.dreamdojo_python = dreamdojo_python
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_path = checkpoint_path
        self.dreamdojo_timeout = dreamdojo_timeout
        self.eval_steps = eval_steps
        self.num_frames = num_frames
        self.judge = FinalFrameJudge(judge_model)
        self.success_threshold = success_threshold
        self.sampling_strategy = sampling_strategy
        self.dd_single_base_index = dd_single_base_index
        self.dd_deterministic_uniform_sampling = dd_deterministic_uniform_sampling
        self.dd_single_chunk = dd_single_chunk
        self.dd_chunk_size = dd_chunk_size
        self.dd_start_frame_idx = dd_start_frame_idx
        self.dd_num_latent_conditional_frames = dd_num_latent_conditional_frames
        self.dd_experiment = dd_experiment
        self.dd_guidance = dd_guidance
        self.planners = PlannerSet()
        self.log = log_fn

        self.video_dir.mkdir(parents=True, exist_ok=True)
        self.work_root.mkdir(parents=True, exist_ok=True)
        self.tmp_root.mkdir(parents=True, exist_ok=True)

    def run_episode(self, row: dict, mode: str) -> dict:
        episode_id = str(row["episode_id"])
        task = row.get("english_task_name") or row.get("task_group", "")
        sub_instructions, planner_meta, _plan_time = self.planners.plan(mode, task)
        gen_prompt = sub_instructions[0] if sub_instructions else task

        tmp_dataset = None
        try:
            tmp_dataset = _make_lerobot_dataset(
                row=row,
                tmp_root=self.tmp_root,
                task_text=gen_prompt,
                num_frames=self.num_frames,
                repo_root=self.repo_root,
                shared_meta=self.shared_meta,
                sampling_strategy=self.sampling_strategy,
            )

            save_dir = self.work_root / f"{episode_id}_{mode}_{int(time.time() * 1000) % 10_000_000}"
            save_dir.mkdir(parents=True, exist_ok=True)

            pred_video = run_dreamdojo_inference(
                dataset_path=tmp_dataset,
                save_dir=save_dir,
                num_frames=self.num_frames,
                log_fn=self.log,
                dreamdojo_python=self.dreamdojo_python,
                dreamdojo_root=self.dreamdojo_root,
                checkpoint_dir=self.checkpoint_dir,
                checkpoint_path=self.checkpoint_path,
                dreamdojo_timeout=self.dreamdojo_timeout,
                single_base_index=self.dd_single_base_index,
                deterministic_uniform_sampling=self.dd_deterministic_uniform_sampling,
                single_chunk=self.dd_single_chunk,
                chunk_size=self.dd_chunk_size,
                start_frame_idx=self.dd_start_frame_idx,
                num_latent_conditional_frames=self.dd_num_latent_conditional_frames,
                experiment_override=self.dd_experiment,
                guidance=self.dd_guidance,
            )
            if pred_video is None or not pred_video.exists():
                raise RuntimeError("DreamDojo produced no output video")

            rollout_gt = pred_video.with_name("0000_gt.mp4")
            if not rollout_gt.exists():
                raise FileNotFoundError(f"DreamDojo rollout GT video missing: {rollout_gt}")
            l2s = compute_rollout_frame_l2s(rollout_gt, pred_video, self.eval_steps)

            real_gt_video = resolve_repo_path(row["camera_paths"]["head"], self.repo_root)
            try:
                judged = self.judge.judge(
                    task,
                    _first_frame(real_gt_video),
                    _last_frame(real_gt_video),
                    _last_frame(pred_video),
                )
            except Exception as exc:
                judged = {
                    "task_progress": 0.0,
                    "rule_success": False,
                    "reason": f"judge_error: {type(exc).__name__}: {str(exc)[:200]}",
                }
            task_progress = float(judged["task_progress"])
            rule_success = bool(judged.get("rule_success", False))
            task_success = bool((task_progress > self.success_threshold) or rule_success)

            flat_pred = self.video_dir / f"{episode_id}_{mode}_pred.mp4"
            flatten_pred_video(pred_video, flat_pred)
            return {
                "episode_id": episode_id,
                "task": task,
                "mode": mode,
                "planner_meta": planner_meta,
                "sub_instructions": sub_instructions,
                "mean_l2": float(np.mean(l2s)) if l2s else None,
                "num_steps": len(l2s),
                "num_step_pass_l2_lt_0_1": int(sum(v < 0.1 for v in l2s)),
                "step_alignment_l2_lt_0_1": float(np.mean([v < 0.1 for v in l2s])) if l2s else None,
                "task_progress": task_progress,
                "rule_success": rule_success,
                "task_success": task_success,
                "judge_reason": judged.get("reason", ""),
                "generated_video": _relative_to_repo(flat_pred),
                "l2_definition": "video_frame_rgb_l2",
            }
        finally:
            if tmp_dataset is not None and tmp_dataset.exists():
                try:
                    shutil.rmtree(str(tmp_dataset))
                except Exception:
                    pass


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Legacy-style AgiBot DreamDojo comparison: task_token_only | dual_llm | val_llm"
    )
    parser.add_argument("--manifest-path", type=Path, action="append", default=None)
    parser.add_argument("--num-episodes", type=int, default=400)
    parser.add_argument("--episode-id", action="append", default=None)
    parser.add_argument(
        "--modes",
        default=",".join(DEFAULT_MODES),
        help="Comma-separated modes. Accepts task_token_only, description_only, dual_llm, val_llm, llm_val.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=REPO_ROOT / "evaluation_results_dualsystem" / "dreamdojo_agibot_compare_preview.json",
    )
    parser.add_argument(
        "--live-log",
        type=Path,
        default=REPO_ROOT / "logs" / "dreamdojo_agibot_compare_preview.log",
    )
    parser.add_argument(
        "--video-dir",
        type=Path,
        default=REPO_ROOT / "evaluation_results_dualsystem" / "dreamdojo_agibot_compare_preview_videos",
    )
    parser.add_argument(
        "--work-root",
        type=Path,
        default=REPO_ROOT / "evaluation_results_dualsystem" / "dreamdojo_agibot_compare_work",
    )
    parser.add_argument(
        "--tmp-root",
        type=Path,
        default=REPO_ROOT / "tmp" / "dreamdojo_agibot_compare_dual",
    )
    parser.add_argument("--eval-steps", type=int, default=49)
    parser.add_argument("--num-frames", type=int, default=49)
    parser.add_argument("--judge-model", default="gpt-4o-mini")
    parser.add_argument("--success-threshold", type=float, default=0.75)
    parser.add_argument("--dreamdojo-root", type=Path, default=DEFAULT_DREAMDOJO_ROOT)
    parser.add_argument("--dreamdojo-python", type=Path, default=DEFAULT_DREAMDOJO_PYTHON)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--checkpoint-path", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--shared-meta", type=Path, default=DEFAULT_SHARED_META)
    parser.add_argument("--dreamdojo-timeout", type=int, default=1800)
    parser.add_argument(
        "--sampling-strategy",
        choices=["linspace_active", "prefix_active", "prefix_full"],
        default="prefix_active",
    )
    parser.add_argument(
        "--dd-single-base-index",
        dest="dd_single_base_index",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--dd-multi-base-index",
        dest="dd_single_base_index",
        action="store_false",
    )
    parser.add_argument(
        "--dd-deterministic-uniform-sampling",
        dest="dd_deterministic_uniform_sampling",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--dd-no-deterministic-uniform-sampling",
        dest="dd_deterministic_uniform_sampling",
        action="store_false",
    )
    parser.add_argument("--dd-single-chunk", action="store_true")
    parser.add_argument("--dd-chunk-size", type=int, default=None)
    parser.add_argument("--dd-start-frame-idx", type=int, default=None)
    parser.add_argument("--dd-num-latent-conditional-frames", type=int, default=None)
    parser.add_argument("--dd-experiment", type=str, default="dreamdojo_2b_480_640_agibot")
    parser.add_argument("--dd-no-force-experiment", action="store_true")
    parser.add_argument("--dd-guidance", type=int, default=None)
    args = parser.parse_args()

    manifest_paths = [p.resolve() for p in (args.manifest_path or DEFAULT_MANIFEST_PATHS)]
    dreamdojo_root = args.dreamdojo_root.resolve()
    dreamdojo_python = absolute_path_preserve_symlink(args.dreamdojo_python)
    checkpoint_dir, checkpoint_path = resolve_checkpoint_paths(args.checkpoint_dir, args.checkpoint_path)
    shared_meta = args.shared_meta.resolve()
    out_path = args.output_json.resolve()
    log_path = args.live_log.resolve()
    video_dir = args.video_dir.resolve()
    work_root = args.work_root.resolve()
    tmp_root = args.tmp_root.resolve()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)
    work_root.mkdir(parents=True, exist_ok=True)
    tmp_root.mkdir(parents=True, exist_ok=True)

    def log(msg: str) -> None:
        print(msg, flush=True)
        with open(log_path, "a") as f:
            f.write(msg + "\n")

    episodes = load_episodes(manifest_paths, None)
    if args.episode_id:
        requested_ids = {str(eid) for eid in args.episode_id}
        episodes = [row for row in episodes if str(row.get("episode_id")) in requested_ids]
    if args.num_episodes is not None:
        episodes = episodes[: args.num_episodes]
    modes = parse_modes(args.modes)

    all_results: list[dict] = []
    running = {mode: [] for mode in modes}
    done_pairs: set[tuple[str, str]] = set()
    if out_path.exists():
        try:
            payload = json.loads(out_path.read_text())
            for row in payload.get("results", []):
                mode = str(row["mode"])
                if mode not in running:
                    continue
                all_results.append(row)
                running[mode].append(row)
                done_pairs.add((str(row["episode_id"]), mode))
        except Exception:
            pass

    runner = LegacyAgiBotCompareRunner(
        video_dir=video_dir,
        work_root=work_root,
        tmp_root=tmp_root,
        repo_root=REPO_ROOT,
        shared_meta=shared_meta,
        dreamdojo_root=dreamdojo_root,
        dreamdojo_python=dreamdojo_python,
        checkpoint_dir=checkpoint_dir,
        checkpoint_path=checkpoint_path,
        dreamdojo_timeout=args.dreamdojo_timeout,
        eval_steps=args.eval_steps,
        num_frames=args.num_frames,
        judge_model=args.judge_model,
        success_threshold=args.success_threshold,
        sampling_strategy=args.sampling_strategy,
        dd_single_base_index=args.dd_single_base_index,
        dd_deterministic_uniform_sampling=args.dd_deterministic_uniform_sampling,
        dd_single_chunk=args.dd_single_chunk,
        dd_chunk_size=args.dd_chunk_size,
        dd_start_frame_idx=args.dd_start_frame_idx,
        dd_num_latent_conditional_frames=args.dd_num_latent_conditional_frames,
        dd_experiment=None if args.dd_no_force_experiment else args.dd_experiment,
        dd_guidance=args.dd_guidance,
        log_fn=log,
    )

    log("=" * 72)
    log(f"DreamDojo AgiBot compare: {' | '.join(modes)}")
    log(f"Episodes: {len(episodes)} | eval_steps={args.eval_steps} | num_frames={args.num_frames}")
    log(f"modes={modes}")
    log(f"sampling_strategy={args.sampling_strategy} | dd_guidance={args.dd_guidance}")
    log(f"checkpoint_dir={checkpoint_dir}")
    log(f"checkpoint_path={checkpoint_path}")
    log("=" * 72)

    for ep_i, row in enumerate(episodes, 1):
        episode_id = str(row["episode_id"])
        task = row.get("english_task_name") or row.get("task_group", "?")
        log(f"\n[{ep_i:3d}/{len(episodes)}] ep={episode_id} | {task[:60]}")

        for mode in modes:
            if (episode_id, mode) in done_pairs:
                log(f"  [{mode:<16}] skipped (already done)")
                continue
            try:
                result = runner.run_episode(row, mode)
            except Exception as exc:
                log(f"  [{mode:<16}] ERROR: {type(exc).__name__}: {str(exc)[:160]}")
                continue

            all_results.append(result)
            running[mode].append(result)
            done_pairs.add((episode_id, mode))
            current = _summarize(running[mode])
            l2_str = f"{result['mean_l2']:.4f}" if result["mean_l2"] is not None else "N/A"
            cur_l2_str = f"{current['mean_l2']:.4f}" if current["mean_l2"] is not None else "N/A"
            cur_l01_str = (
                f"{current['rate_of_l2_lt_0_1']:.3f}"
                if current["rate_of_l2_lt_0_1"] is not None
                else "N/A"
            )
            log(
                f"  [{mode:<16}] l2={l2_str} "
                f"progress={result['task_progress']:.2f} "
                f"success={'Y' if result['task_success'] else 'N'} | "
                f"run_l2={cur_l2_str} run_prog={current['mean_task_progress']:.3f} "
                f"run_sr={current['success_rate']:.3f} run_l2<0.1={cur_l01_str}"
            )
            if mode != "task_token_only":
                log(f"    sub={result['sub_instructions']}")

            payload = {
                "results": all_results,
                "summary": {mode_name: _summarize(rows) for mode_name, rows in running.items()},
                "notes": NOTES,
            }
            out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")

    log("\nDone.")
    log(json.dumps({mode: _summarize(rows) for mode, rows in running.items()}, indent=2))


if __name__ == "__main__":
    main()
