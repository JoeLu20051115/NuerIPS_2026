#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from openai import OpenAI
from PIL import Image

from run_dualsystem_evaluation import LLMPlanner


REPO_ROOT = Path("/home/xingrui/lueq/NuerIPS_2026")
DREAMDOJO_ROOT = REPO_ROOT / "external_repos" / "DreamDojo"
DEFAULT_DATASET = REPO_ROOT / "data" / "egodex_eval_official" / "EgoDex_Eval"
DEFAULT_CHECKPOINT_DIR = DREAMDOJO_ROOT / "checkpoints" / "2B_GR1_post-train"
DEFAULT_CHECKPOINT_PATH = DEFAULT_CHECKPOINT_DIR / "iter_000050000" / "model_ema_bf16.pt"
DEFAULT_OUTPUT_JSON = REPO_ROOT / "evaluation_results_dualsystem" / "official_egodex_compare_preview.json"
DEFAULT_LIVE_LOG = REPO_ROOT / "logs" / "official_egodex_compare_preview.log"
DEFAULT_SAVE_ROOT = REPO_ROOT / "evaluation_results_dualsystem" / "official_egodex_compare_preview"
DEFAULT_TMP_ROOT = REPO_ROOT / "tmp" / "official_egodex_compare"
DEFAULT_DREAMDOJO_PYTHON = DREAMDOJO_ROOT / ".venv" / "bin" / "python"


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


def load_episode_entries(dataset_path: Path) -> tuple[list[dict], str]:
    if dataset_path.is_file() and dataset_path.suffix == ".json":
        manifest = json.loads(dataset_path.read_text())
        episodes = manifest.get("episodes", [])
        return episodes, "manifest"
    episodes = read_jsonl(dataset_path / "meta" / "episodes.jsonl")
    return episodes, "official"


def build_dual_prompt(task: str, planner: LLMPlanner) -> tuple[str, list[str], dict]:
    subtasks, meta = planner.plan(task, {"task": task})
    subtasks = subtasks[:4]
    hybrid = task
    if subtasks:
        hybrid = f"{task}. Plan: " + " ; ".join(subtasks)
    return hybrid, subtasks, meta


def infer_task_text(episode_row: dict) -> str:
    if episode_row.get("english_task_name"):
        return str(episode_row["english_task_name"])
    tasks = episode_row.get("tasks")
    if isinstance(tasks, list) and tasks:
        return str(tasks[0])
    if episode_row.get("task_group"):
        group = str(episode_row["task_group"]).replace("_", " ")
        return group
    if episode_row.get("description"):
        return str(episode_row["description"])
    return ""


def make_subset_dataset(
    src_root: Path,
    tmp_root: Path,
    episode_row: dict,
    task_override: str,
) -> tuple[Path, Path]:
    ep_idx = int(episode_row["episode_index"])
    episode_id = f"episode_{ep_idx:06d}"
    subset_root = tmp_root / episode_id
    if subset_root.exists():
        shutil.rmtree(subset_root)
    (subset_root / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (subset_root / "videos" / "chunk-000" / "observation.images.ego_view_freq20").mkdir(parents=True, exist_ok=True)
    (subset_root / "meta").mkdir(parents=True, exist_ok=True)

    for name in ("info.json", "stats.json", "modality.json"):
        src = src_root / "meta" / name
        dst = subset_root / "meta" / name
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src)

    src_parquet = src_root / "data" / "chunk-000" / f"{episode_id}.parquet"
    dst_parquet = subset_root / "data" / "chunk-000" / f"{episode_id}.parquet"
    dst_parquet.symlink_to(src_parquet)

    src_video = src_root / "videos" / "chunk-000" / "observation.images.ego_view_freq20" / f"{episode_id}.mp4"
    dst_video = subset_root / "videos" / "chunk-000" / "observation.images.ego_view_freq20" / f"{episode_id}.mp4"
    dst_video.symlink_to(src_video)

    tasks = read_jsonl(src_root / "meta" / "tasks.jsonl")
    task_index = 0
    for row in tasks:
        if row.get("task") == episode_row["tasks"][0]:
            task_index = int(row["task_index"])
            break
    tasks[task_index]["task"] = task_override

    write_jsonl(subset_root / "meta" / "tasks.jsonl", tasks)
    write_jsonl(subset_root / "meta" / "episodes.jsonl", [episode_row])
    return subset_root, src_video


def make_subset_manifest(
    manifest_path: Path,
    tmp_root: Path,
    episode_row: dict,
    task_override: str,
) -> tuple[Path, Path]:
    manifest = json.loads(manifest_path.read_text())
    source_root = Path(manifest["source"])
    if not source_root.is_absolute():
        repo_candidate = (REPO_ROOT / source_root).resolve()
        local_candidate = (manifest_path.parent.parent / source_root).resolve()
        source_root = repo_candidate if repo_candidate.exists() else local_candidate

    episode_id = str(episode_row["episode_id"]).replace("/", "__")
    subset_root = tmp_root / episode_id
    if subset_root.exists():
        shutil.rmtree(subset_root)
    subset_root.mkdir(parents=True, exist_ok=True)

    subset_manifest = {
        "source": str(source_root),
        "selection_policy": manifest.get("selection_policy", "single-episode subset"),
        "num_tasks": 1,
        "target_episodes": 1,
        "num_episodes": 1,
        "task_counts": {str(episode_row.get("task_group", "task")): 1},
        "episodes": [
            {
                **episode_row,
                "english_task_name": task_override,
            }
        ],
    }
    subset_manifest_path = subset_root / "manifest.json"
    subset_manifest_path.write_text(json.dumps(subset_manifest, indent=2, ensure_ascii=False) + "\n")
    gt_video = source_root / str(episode_row["mp4_path"])
    return subset_manifest_path, gt_video


def run_official_inference(
    dataset_path: Path,
    save_dir: Path,
    dreamdojo_python: Path,
    checkpoint_dir: Path,
    checkpoint_path: Path,
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
        "--checkpoints-dir",
        str(checkpoint_dir),
        "--checkpoint-path",
        str(checkpoint_path),
        "--experiment",
        "dreamdojo_2b_480_640_gr1",
        "--save-dir",
        str(save_dir),
        "--num-frames",
        "49",
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
    subprocess.run(cmd, check=True, env=env)
    return save_dir / "iter_000050000"


def append_log(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(line + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate official EgoDex_Eval with judged task_token_only vs dual-hybrid metrics.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--checkpoint-path", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--dreamdojo-python", type=Path, default=DEFAULT_DREAMDOJO_PYTHON)
    parser.add_argument("--num-episodes", type=int, default=4)
    parser.add_argument("--eval-steps", type=int, default=10)
    parser.add_argument("--judge-model", type=str, default="gpt-4o-mini")
    parser.add_argument("--success-threshold", type=float, default=0.75)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--live-log", type=Path, default=DEFAULT_LIVE_LOG)
    parser.add_argument("--save-root", type=Path, default=DEFAULT_SAVE_ROOT)
    parser.add_argument("--tmp-root", type=Path, default=DEFAULT_TMP_ROOT)
    args = parser.parse_args()

    episodes, dataset_kind = load_episode_entries(args.dataset_root)
    episodes = episodes[: args.num_episodes]
    planner = LLMPlanner(use_mock=False, temperature=0.0)
    judge = FinalFrameJudge(args.judge_model)
    all_results: list[dict] = []
    running = {"task_token_only": [], "dual_llm": []}

    if args.live_log.exists():
        args.live_log.unlink()

    for idx, episode_row in enumerate(episodes, 1):
        task = infer_task_text(episode_row)
        episode_label = (
            f"episode_{int(episode_row['episode_index']):06d}"
            if dataset_kind == "official"
            else str(episode_row["episode_id"])
        )
        append_log(args.live_log, f"[{idx}/{len(episodes)}] {episode_label} | task={task}")
        for mode in ("task_token_only", "dual_llm"):
            if mode == "task_token_only":
                prompt = task
                subtasks = [task]
                planner_meta = {"planner_mode": "disabled"}
            else:
                prompt, subtasks, planner_meta = build_dual_prompt(task, planner)

            if dataset_kind == "official":
                subset_root, gt_video = make_subset_dataset(args.dataset_root, args.tmp_root / mode, episode_row, prompt)
            else:
                subset_root, gt_video = make_subset_manifest(args.dataset_root, args.tmp_root / mode, episode_row, prompt)
            save_dir = args.save_root / mode / episode_label.replace("/", "__")
            infer_dir = run_official_inference(
                dataset_path=subset_root,
                save_dir=save_dir,
                dreamdojo_python=args.dreamdojo_python,
                checkpoint_dir=args.checkpoint_dir,
                checkpoint_path=args.checkpoint_path,
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
