#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import shutil
import subprocess
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pandas as pd
from openai import OpenAI
from PIL import Image

from run_dualsystem_evaluation import LLMPlanner


REPO_ROOT = Path("/home/xingrui/lueq/NuerIPS_2026")
DREAMDOJO_ROOT = REPO_ROOT / "external_repos" / "DreamDojo"
DEFAULT_DATASET = REPO_ROOT / "data" / "droid_easy400_dualfavored_dreamzero"
DEFAULT_CHECKPOINT_DIR = DREAMDOJO_ROOT / "checkpoints" / "2B_GR1_post-train"
DEFAULT_CHECKPOINT_PATH = DEFAULT_CHECKPOINT_DIR / "iter_000050000" / "model_ema_bf16.pt"
DEFAULT_OUTPUT_JSON = REPO_ROOT / "evaluation_results_dualsystem" / "dreamdojo_droid400_compare.json"
DEFAULT_LIVE_LOG = REPO_ROOT / "logs" / "dreamdojo_droid400_compare.log"
DEFAULT_SAVE_ROOT = REPO_ROOT / "evaluation_results_dualsystem" / "dreamdojo_droid400_compare"
DEFAULT_TMP_ROOT = REPO_ROOT / "tmp" / "dreamdojo_droid400_compare"
DEFAULT_DREAMDOJO_PYTHON = DREAMDOJO_ROOT / ".venv" / "bin" / "python"
DEFAULT_NUM_FRAMES = 13
DEFAULT_LAM_CKPT = REPO_ROOT / "checkpoints" / "DreamDojo" / "LAM_400k.ckpt"
DROID_VIDEO_KEYS = [
    "observation.images.exterior_image_1_left",
    "observation.images.exterior_image_2_left",
    "observation.images.wrist_image_left",
]
PRIMARY_VIDEO_KEY = "observation.images.exterior_image_1_left"
FALLBACK_DROID_TASKS = REPO_ROOT / "data" / "droid_lerobot" / "meta" / "tasks.jsonl"
FALLBACK_DROID_MODALITY = REPO_ROOT / "data" / "droid_lerobot" / "meta" / "modality.json"
FALLBACK_DROID_STATS = REPO_ROOT / "data" / "droid_lerobot" / "meta" / "stats.json"
OPENAI_KEY_PATH = REPO_ROOT / ".openai_key"


class FinalFrameJudge:
    def __init__(self, model: str) -> None:
        self.client = OpenAI()
        self.model = model

    @staticmethod
    def _frame_to_data_url(frame: np.ndarray) -> str:
        buf = io.BytesIO()
        imageio.imwrite(buf, frame, format="png")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/png;base64,{b64}"

    def judge(
        self,
        task: str,
        initial_frame: np.ndarray,
        real_final_frame: np.ndarray,
        predicted_final_frame: np.ndarray,
    ) -> dict:
        prompt = (
            "You are judging robot task completion from visual evidence.\n"
            "You will see three images:\n"
            "1. initial real observation\n"
            "2. real final observation from the demonstration (reference goal state)\n"
            "3. predicted final frame from the model rollout\n\n"
            f"Task: {task}\n\n"
            "Return strict JSON only with schema:\n"
            "{\"task_progress\": number, \"rule_success\": boolean, \"reason\": string}\n\n"
            "Rules:\n"
            "- task_progress must be in [0,1].\n"
            "- rule_success means: the final state satisfies the task goal, yes or no.\n"
            "- Judge the model's predicted final frame against the task goal, using the real final frame only as a helpful reference.\n"
            "- Be slightly lenient to small pose offsets if the task intent is clearly satisfied.\n"
            "- 1.0 means the predicted final frame clearly satisfies the task.\n"
            "- 0.7-0.9 means the task is almost fully achieved and should count as success in a lenient setting.\n"
            "- 0.4-0.6 means partial completion or the key object reached the target region but final relation is not fully convincing.\n"
            "- 0.0-0.3 means the task goal is not achieved.\n"
            "- If the final state visibly satisfies the task goal, set rule_success=true even if task_progress is below 1.0.\n"
        )
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0.0,
            max_tokens=300,
            messages=[
                {
                    "role": "system",
                    "content": "You are a careful robot-evaluation judge. Output JSON only.",
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "text", "text": "Initial real observation:"},
                        {"type": "image_url", "image_url": {"url": self._frame_to_data_url(initial_frame)}},
                        {"type": "text", "text": "Real final observation from the demonstration:"},
                        {"type": "image_url", "image_url": {"url": self._frame_to_data_url(real_final_frame)}},
                        {"type": "text", "text": "Predicted final frame from the model rollout:"},
                        {"type": "image_url", "image_url": {"url": self._frame_to_data_url(predicted_final_frame)}},
                    ],
                },
            ],
        )
        content = response.choices[0].message.content or "{}"
        try:
            start = content.find("{")
            end = content.rfind("}")
            payload = json.loads(content[start : end + 1] if start != -1 and end != -1 and end > start else content)
        except Exception:
            payload = {"task_progress": 0.0, "rule_success": False, "reason": f"non_json_response: {content[:200]}"}
        progress = float(np.clip(float(payload.get("task_progress", 0.0)), 0.0, 1.0))
        rule_success = bool(payload.get("rule_success", False))
        return {"task_progress": progress, "rule_success": rule_success, "reason": str(payload.get("reason", ""))}


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def extract_frame(video_path: Path, index: int) -> np.ndarray:
    reader = imageio.get_reader(str(video_path))
    try:
        frame = reader.get_data(index)
    finally:
        reader.close()
    return frame


def extract_first_frame(video_path: Path) -> np.ndarray:
    return extract_frame(video_path, 0)


def extract_final_frame(video_path: Path) -> np.ndarray:
    reader = imageio.get_reader(str(video_path))
    try:
        frame = reader.get_data(reader.count_frames() - 1)
    finally:
        reader.close()
    return frame


def compute_frame_l2s(gt_video: Path, pred_video: Path, eval_steps: int) -> list[float]:
    gt_reader = imageio.get_reader(str(gt_video))
    pred_reader = imageio.get_reader(str(pred_video))
    try:
        gt_len = gt_reader.count_frames()
        pred_len = pred_reader.count_frames()
        num = min(gt_len, pred_len)
        if num <= 0:
            return []
        indices = np.linspace(0, num - 1, min(num, eval_steps), dtype=int)
        vals = []
        for idx in indices:
            gt = gt_reader.get_data(int(idx)).astype(np.float32) / 255.0
            pred = pred_reader.get_data(int(idx)).astype(np.float32) / 255.0
            if gt.shape != pred.shape:
                target_h = min(gt.shape[0], pred.shape[0])
                target_w = min(gt.shape[1], pred.shape[1])
                gt = np.asarray(Image.fromarray((gt * 255).astype(np.uint8)).resize((target_w, target_h), Image.BILINEAR)).astype(np.float32) / 255.0
                pred = np.asarray(Image.fromarray((pred * 255).astype(np.uint8)).resize((target_w, target_h), Image.BILINEAR)).astype(np.float32) / 255.0
            vals.append(float(np.sqrt(np.mean((gt - pred) ** 2))))
        return vals
    finally:
        gt_reader.close()
        pred_reader.close()


def summarize(rows: list[dict]) -> dict:
    return {
        "num_episodes": len(rows),
        "mean_l2": float(np.mean([r["mean_l2"] for r in rows if r["mean_l2"] is not None])) if rows else None,
        "mean_task_progress": float(np.mean([r["task_progress"] for r in rows])) if rows else None,
        "success_rate": float(np.mean([1.0 if r["task_success"] else 0.0 for r in rows])) if rows else None,
        "rate_of_l2_lt_0_1": (
            float(sum(r["num_step_pass_l2_lt_0_1"] for r in rows) / sum(r["num_steps"] for r in rows))
            if rows and sum(r["num_steps"] for r in rows) > 0
            else None
        ),
    }


def build_dual_prompt(task: str, planner: LLMPlanner) -> tuple[str, list[str], dict]:
    subtasks, meta = planner.plan(task, {"task": task})
    subtasks = subtasks[:4]
    hybrid = task
    if subtasks:
        hybrid = f"{task}. Plan: " + " ; ".join(subtasks)
    return hybrid, subtasks, meta


def pick_task_text(row: dict) -> str:
    tasks = row.get("tasks") or []
    if isinstance(tasks, list) and tasks:
        return str(tasks[0])
    return "not provided"


def link_or_replace(dst: Path, src: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src)


def find_chunked_file(root: Path, category: str, leaf: str) -> Path | None:
    base = root / category
    if not base.exists():
        return None
    hits = sorted(base.glob(f"chunk-*/{leaf}"))
    return hits[0] if hits else None


def _slice_stats_block(block: dict, start: int, end: int) -> dict:
    sliced = {}
    for key, value in block.items():
        if isinstance(value, list):
            sliced[key] = value[start:end]
    return sliced


def build_droid_gr1_proxy_modality() -> dict:
    return {
        "state": {
            "left_arm": {"original_key": "observation.state", "start": 7, "end": 14},
            "right_arm": {"original_key": "observation.state", "start": 7, "end": 14},
            "left_hand": {"original_key": "observation.state", "start": 6, "end": 7},
            "right_hand": {"original_key": "observation.state", "start": 6, "end": 7},
            "waist": {"original_key": "observation.state", "start": 0, "end": 3},
        },
        "action": {
            "left_arm": {"original_key": "action", "start": 14, "end": 21},
            "right_arm": {"original_key": "action", "start": 14, "end": 21},
            "left_hand": {"original_key": "action", "start": 12, "end": 13},
            "right_hand": {"original_key": "action", "start": 12, "end": 13},
            "waist": {"original_key": "action", "start": 0, "end": 3},
        },
        "video": {
            "ego_view_freq20": {"original_key": PRIMARY_VIDEO_KEY},
        },
        "annotation": {
            "language.language_instruction": {},
        },
    }


def build_droid_gr1_proxy_stats(stats: dict) -> dict:
    obs = stats["observation.state"]
    act = stats["action"]
    return {
        "state": {
            "left_arm": _slice_stats_block(obs, 7, 14),
            "right_arm": _slice_stats_block(obs, 7, 14),
            "left_hand": _slice_stats_block(obs, 6, 7),
            "right_hand": _slice_stats_block(obs, 6, 7),
            "waist": _slice_stats_block(obs, 0, 3),
        },
        "action": {
            "left_arm": _slice_stats_block(act, 14, 21),
            "right_arm": _slice_stats_block(act, 14, 21),
            "left_hand": _slice_stats_block(act, 12, 13),
            "right_hand": _slice_stats_block(act, 12, 13),
            "waist": _slice_stats_block(act, 0, 3),
        },
    }


def resolve_source_root(src_root: Path) -> Path:
    info_path = src_root / "meta" / "info.json"
    if info_path.exists():
        try:
            info = json.loads(info_path.read_text())
            source_root = info.get("source_root")
            if source_root:
                source_root = Path(source_root)
                if not source_root.is_absolute():
                    source_root = (REPO_ROOT / source_root).resolve()
                if source_root.exists():
                    return source_root
        except Exception:
            pass
    return src_root


def choose_gt_video(src_root: Path, episode_id: str) -> Path:
    for key in DROID_VIDEO_KEYS:
        candidate = find_chunked_file(src_root, "videos", f"{key}/{episode_id}.mp4")
        if candidate is not None and candidate.exists():
            return candidate
    raise FileNotFoundError(f"Missing all DROID videos for {episode_id}")


def choose_gt_video_with_fallback(primary_root: Path, fallback_root: Path, episode_id: str) -> Path:
    try:
        return choose_gt_video(primary_root, episode_id)
    except FileNotFoundError:
        if fallback_root != primary_root:
            return choose_gt_video(fallback_root, episode_id)
        raise


def make_subset_dataset(src_root: Path, tmp_root: Path, episode_row: dict, task_override: str) -> tuple[Path, Path]:
    ep_idx = int(episode_row["episode_index"])
    episode_id = f"episode_{ep_idx:06d}"
    subset_episode_id = "episode_000000"
    subset_root = tmp_root / episode_id
    if subset_root.exists():
        shutil.rmtree(subset_root)
    (subset_root / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (subset_root / "videos" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (subset_root / "meta").mkdir(parents=True, exist_ok=True)

    resolved_source_root = resolve_source_root(src_root)

    modality = build_droid_gr1_proxy_modality()
    (subset_root / "meta" / "modality.json").write_text(json.dumps(modality, indent=2, ensure_ascii=False) + "\n")

    tasks_path = src_root / "meta" / "tasks.jsonl"
    if not tasks_path.exists():
        candidate = resolved_source_root / "meta" / "tasks.jsonl"
        if candidate.exists():
            tasks_path = candidate
    if not tasks_path.exists() and FALLBACK_DROID_TASKS.exists():
        tasks_path = FALLBACK_DROID_TASKS
    tasks = read_jsonl(tasks_path)

    info = {}
    info_src = src_root / "meta" / "info.json"
    if info_src.exists():
        info = json.loads(info_src.read_text())
    info.update(
        {
            "dataset_name": subset_root.name,
            "total_episodes": 1,
            "total_frames": int(episode_row["length"]),
            "total_tasks": len(tasks),
            "total_videos": len(DROID_VIDEO_KEYS),
            "total_chunks": 1,
            "chunks_size": 1,
            "splits": {"full": "0:1", "train": "0:1"},
            "source_root": str(resolve_source_root(src_root)),
        }
    )
    (subset_root / "meta" / "info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n")

    src_parquet = find_chunked_file(src_root, "data", f"{episode_id}.parquet")
    if src_parquet is None:
        raise FileNotFoundError(f"Missing parquet for {episode_id}")
    subset_parquet = subset_root / "data" / "chunk-000" / f"{subset_episode_id}.parquet"
    df = pd.read_parquet(src_parquet).copy()
    if "episode_index" in df.columns:
        df["episode_index"] = 0
    if "frame_index" in df.columns:
        df["frame_index"] = np.arange(len(df), dtype=np.int64)
    subset_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(subset_parquet, index=False)

    for key in DROID_VIDEO_KEYS:
        src_video = find_chunked_file(src_root, "videos", f"{key}/{episode_id}.mp4")
        if src_video is None:
            src_video = find_chunked_file(resolved_source_root, "videos", f"{key}/{episode_id}.mp4")
        if src_video is not None and src_video.exists():
            link_or_replace(subset_root / "videos" / "chunk-000" / key / f"{subset_episode_id}.mp4", src_video)

    task_index = None
    target_task = episode_row["tasks"][0] if episode_row.get("tasks") else None
    for row in tasks:
        if row.get("task") == target_task:
            task_index = int(row["task_index"])
            break
    if task_index is None:
        task_index = 0
    tasks[task_index]["task"] = task_override

    write_jsonl(subset_root / "meta" / "tasks.jsonl", tasks)
    subset_episode_row = dict(episode_row)
    subset_episode_row["episode_index"] = 0
    write_jsonl(subset_root / "meta" / "episodes.jsonl", [subset_episode_row])
    return subset_root, choose_gt_video_with_fallback(src_root, resolved_source_root, episode_id)


def run_official_inference(
    dataset_path: Path,
    save_dir: Path,
    dreamdojo_python: Path,
    checkpoint_dir: Path,
    checkpoint_path: Path,
    num_frames: int,
) -> Path:
    if save_dir.exists():
        shutil.rmtree(save_dir)
    output_dir = save_dir.parent / "driver_logs"
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(dreamdojo_python),
        str(DREAMDOJO_ROOT / "examples" / "action_conditioned.py"),
        "-o",
        str(output_dir),
        "--disable-guardrails",
        "--checkpoints-dir",
        str(checkpoint_dir),
        "--checkpoint-path",
        str(checkpoint_path),
        "--experiment",
        "dreamdojo_2b_480_640_gr1",
        "--save-dir",
        str(save_dir),
        "--num-frames",
        str(num_frames),
        "--num-samples",
        "1",
        "--dataset-path",
        str(dataset_path),
        "--data-split",
        "full",
        "--deterministic-uniform-sampling",
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(DREAMDOJO_ROOT)
    if DEFAULT_LAM_CKPT.exists():
        env["DREAMDOJO_LAM_CKPT"] = str(DEFAULT_LAM_CKPT)
    subprocess.run(cmd, check=True, env=env, cwd=DREAMDOJO_ROOT)
    return save_dir / "iter_000050000"


def append_log(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(line + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate DROID Easy400 with DreamDojo judged task_token_only vs dual-hybrid metrics.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--checkpoint-path", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--dreamdojo-python", type=Path, default=DEFAULT_DREAMDOJO_PYTHON)
    parser.add_argument("--num-frames", type=int, default=DEFAULT_NUM_FRAMES)
    parser.add_argument("--num-episodes", type=int, default=4)
    parser.add_argument("--eval-steps", type=int, default=10)
    parser.add_argument("--judge-model", type=str, default="gpt-4o-mini")
    parser.add_argument("--success-threshold", type=float, default=0.75)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--live-log", type=Path, default=DEFAULT_LIVE_LOG)
    parser.add_argument("--save-root", type=Path, default=DEFAULT_SAVE_ROOT)
    parser.add_argument("--tmp-root", type=Path, default=DEFAULT_TMP_ROOT)
    args = parser.parse_args()

    args.dataset_root = args.dataset_root.resolve()
    args.checkpoint_dir = args.checkpoint_dir.resolve()
    args.checkpoint_path = args.checkpoint_path.resolve()
    args.dreamdojo_python = args.dreamdojo_python.expanduser().absolute()
    args.output_json = args.output_json.resolve()
    args.live_log = args.live_log.resolve()
    args.save_root = args.save_root.resolve()
    args.tmp_root = args.tmp_root.resolve()

    if "OPENAI_API_KEY" not in os.environ and OPENAI_KEY_PATH.exists():
        os.environ["OPENAI_API_KEY"] = OPENAI_KEY_PATH.read_text().strip()

    episodes = read_jsonl(args.dataset_root / "meta" / "episodes.jsonl")[: args.num_episodes]
    planner = LLMPlanner(use_mock=False, temperature=0.0)
    judge = FinalFrameJudge(args.judge_model)
    all_results: list[dict] = []
    running = {"task_token_only": [], "dual_llm": []}

    if args.live_log.exists():
        args.live_log.unlink()

    for idx, episode_row in enumerate(episodes, 1):
        task = pick_task_text(episode_row)
        episode_label = f"episode_{int(episode_row['episode_index']):06d}"
        append_log(args.live_log, f"[{idx}/{len(episodes)}] {episode_label} | task={task}")
        for mode in ("task_token_only", "dual_llm"):
            if mode == "task_token_only":
                prompt = task
                subtasks = [task]
                planner_meta = {"planner_mode": "disabled"}
            else:
                prompt, subtasks, planner_meta = build_dual_prompt(task, planner)

            subset_root, gt_video = make_subset_dataset(args.dataset_root, args.tmp_root / mode, episode_row, prompt)
            save_dir = args.save_root / mode / episode_label
            infer_dir = run_official_inference(
                dataset_path=subset_root,
                save_dir=save_dir,
                dreamdojo_python=args.dreamdojo_python,
                checkpoint_dir=args.checkpoint_dir,
                checkpoint_path=args.checkpoint_path,
                num_frames=args.num_frames,
            )
            pred_video = infer_dir / "0000_pred.mp4"
            initial_frame = extract_first_frame(gt_video)
            real_final = extract_final_frame(gt_video)
            pred_final = extract_final_frame(pred_video)
            l2s = compute_frame_l2s(gt_video, pred_video, args.eval_steps)
            judged = judge.judge(task, initial_frame, real_final, pred_final)
            task_progress = float(judged["task_progress"])
            rule_success = bool(judged["rule_success"])
            task_success = bool((task_progress > args.success_threshold) or rule_success)
            result = {
                "episode_id": episode_label,
                "task_description": task,
                "mode": mode,
                "prompt": prompt,
                "sub_instructions": subtasks,
                "planner_meta": planner_meta,
                "mean_l2": float(np.mean(l2s)) if l2s else None,
                "num_steps": len(l2s),
                "num_step_pass_l2_lt_0_1": int(sum(v < 0.1 for v in l2s)),
                "step_alignment_l2_lt_0_1": float(np.mean([v < 0.1 for v in l2s])) if l2s else None,
                "task_progress": task_progress,
                "rule_success": rule_success,
                "task_success": task_success,
                "judge_reason": judged["reason"],
                "generated_video": str(pred_video),
            }
            all_results.append(result)
            running[mode].append(result)
            current = summarize(running[mode])
            append_log(
                args.live_log,
                f"  [{mode}] mean_l2={result['mean_l2']:.4f} | task_progress={task_progress:.3f} | "
                f"rule_success={'PASS' if rule_success else 'FAIL'} | success={'PASS' if task_success else 'FAIL'} | "
                f"running_success_rate={current['success_rate']:.3f} | running_mean_l2={current['mean_l2']:.4f} | "
                f"running_mean_progress={current['mean_task_progress']:.3f}"
            )
            if mode == "dual_llm":
                append_log(args.live_log, f"    sub_instructions={subtasks}")

        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps({"results": all_results}, indent=2, ensure_ascii=False) + "\n")

    summary = {mode: summarize(rows) for mode, rows in running.items()}
    args.output_json.write_text(json.dumps({"results": all_results, "summary": summary}, indent=2, ensure_ascii=False) + "\n")
    append_log(args.live_log, "=== Summary ===")
    append_log(args.live_log, json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
