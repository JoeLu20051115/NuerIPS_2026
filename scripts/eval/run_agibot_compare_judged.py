#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
import uuid
from pathlib import Path

import h5py
import imageio.v2 as imageio
import numpy as np
from PIL import Image

from run_agibot_tasktoken_judged import DEFAULT_GENERATED_VIDEO_DIR, DEFAULT_MANIFEST, FinalFrameJudge

try:
    from eval_utils.policy_client import WebsocketClientPolicy
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("policy_client is required") from exc

try:
    from run_dualsystem_evaluation import LLMPlanner
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("LLMPlanner is required") from exc


class AgiBotCompareRunner:
    def __init__(
        self,
        manifest_path: Path,
        generated_video_dir: Path,
        host: str,
        port: int,
        eval_steps: int,
        judge_model: str,
        success_threshold: float,
    ) -> None:
        self.manifest = json.loads(manifest_path.read_text())
        self.episodes = self.manifest["episodes"]
        self.generated_video_dir = generated_video_dir
        self.eval_steps = eval_steps
        self.success_threshold = success_threshold
        self.policy = WebsocketClientPolicy(host=host, port=port)
        self.judge = FinalFrameJudge(judge_model)
        self.planner = LLMPlanner(use_mock=False, temperature=0.0)

    @staticmethod
    def _load_episode_metadata(row: dict) -> tuple[int, int]:
        episode_dir = Path(row["relative_episode_dir"])
        data_info_path = Path(row.get("data_info_path", episode_dir / "data_info.json"))
        if not data_info_path.exists():
            return 0, int(row.get("num_steps", 1)) - 1
        data = json.loads(data_info_path.read_text())
        action_cfg = data.get("label_info", {}).get("action_config", [])
        if not action_cfg:
            return 0, int(row.get("num_steps", 1)) - 1
        active_start = min(int(x.get("start_frame", 0)) for x in action_cfg)
        active_end = max(int(x.get("end_frame", int(row.get("num_steps", 1)) - 1)) for x in action_cfg)
        return active_start, active_end

    @staticmethod
    def _load_frame(video_path: Path, frame_idx: int) -> np.ndarray:
        reader = imageio.get_reader(str(video_path))
        try:
            frame = reader.get_data(frame_idx)
        finally:
            reader.close()
        return frame

    @staticmethod
    def _extract_first_frame(video_path: Path) -> np.ndarray:
        reader = imageio.get_reader(str(video_path))
        try:
            frame = reader.get_data(0)
        finally:
            reader.close()
        return frame

    @staticmethod
    def _extract_final_frame(video_path: Path) -> np.ndarray:
        reader = imageio.get_reader(str(video_path))
        try:
            frame = reader.get_data(reader.count_frames() - 1)
        finally:
            reader.close()
        return frame

    @staticmethod
    def _resize_frame(frame: np.ndarray, width: int = 320, height: int = 180) -> np.ndarray:
        if frame.shape[1] == width and frame.shape[0] == height:
            return frame
        return np.asarray(Image.fromarray(frame).resize((width, height), Image.BILINEAR))

    @staticmethod
    def _sample_indices(start: int, end: int, eval_steps: int) -> np.ndarray:
        start = int(max(0, start))
        end = int(max(start, end))
        return np.linspace(start, end, min(end - start + 1, eval_steps), dtype=int)

    def _build_obs_images(self, head_frame: np.ndarray, right_frame: np.ndarray) -> dict:
        head_frame = self._resize_frame(head_frame)
        right_frame = self._resize_frame(right_frame)
        return {
            "observation/exterior_image_0_left": head_frame,
            "observation/exterior_image_1_left": head_frame,
            "observation/wrist_image_left": right_frame,
        }

    def _find_generated_video(self, session_id: str, existing_files: set[str]) -> Path:
        candidates = sorted(
            self.generated_video_dir.glob(f"*{session_id}*.mp4"),
            key=lambda p: p.stat().st_mtime,
        )
        if candidates:
            return candidates[-1]
        fresh_candidates = sorted(
            [p for p in self.generated_video_dir.glob("*.mp4") if p.name not in existing_files],
            key=lambda p: p.stat().st_mtime,
        )
        if fresh_candidates:
            return fresh_candidates[-1]
        raise FileNotFoundError(f"No generated video found for session_id={session_id}")

    def _plan(self, mode: str, task: str, episode_id: str) -> tuple[list[str], dict, float]:
        if mode == "task_token_only":
            return [task], {"planner_mode": "disabled"}, 0.0
        start = time.perf_counter()
        sub_instructions, planner_meta = self.planner.plan(task, {"episode_id": episode_id})
        return sub_instructions, planner_meta, time.perf_counter() - start

    def run_episode(self, row: dict, mode: str) -> dict:
        h5_path = Path(row["h5_path"])
        with h5py.File(h5_path, "r") as f:
            states = f["state/joint/position"][:]
            actions = f["action/joint/position"][:]
            right_gripper = f["state/right_effector/position"][:]

        head_path = Path(row["camera_paths"]["head"])
        right_path = Path(row["camera_paths"]["hand_right"])
        head_frames = int(row["camera_frames"]["head"])
        right_frames = int(row["camera_frames"]["hand_right"])

        active_start, active_end = self._load_episode_metadata(row)
        indices = self._sample_indices(active_start, active_end, self.eval_steps)
        task = row["english_task_name"] or row["task_group"]
        sub_instructions, planner_meta, plan_time = self._plan(mode, task, row["episode_id"])
        session_id = f"agibot_cmp_{mode}_{row['episode_id']}_{uuid.uuid4().hex[:8]}"
        existing_videos = {p.name for p in self.generated_video_dir.glob("*.mp4")}
        action_errors: list[float] = []

        initial_frame = self._extract_first_frame(head_path)
        real_final_frame = self._extract_final_frame(head_path)
        self.policy.reset({"session_id": session_id})

        for step_idx in indices:
            head_idx = min(int(step_idx), head_frames - 1)
            right_idx = min(int(step_idx), right_frames - 1)
            obs = self._build_obs_images(
                self._load_frame(head_path, head_idx),
                self._load_frame(right_path, right_idx),
            )
            sub_idx = min(int(step_idx * len(sub_instructions) / max(1, len(states))), len(sub_instructions) - 1)
            prompt = task if mode == "task_token_only" else sub_instructions[sub_idx]
            obs.update(
                {
                    "observation/joint_position": states[step_idx, 7:14].astype(np.float64),
                    "observation/gripper_position": right_gripper[step_idx].astype(np.float64),
                    "prompt": prompt,
                    "session_id": session_id,
                }
            )
            result = self.policy.infer(obs)
            pred_action = result.get("action", None) if isinstance(result, dict) else result
            if pred_action is None:
                continue
            pred_joint = pred_action[0, :7] if getattr(pred_action, "ndim", 1) == 2 else pred_action[:7]
            gt_joint = actions[step_idx, 7:14]
            action_errors.append(float(np.sqrt(np.sum((pred_joint - gt_joint) ** 2))))

        self.policy.reset({"session_id": session_id})
        time.sleep(1.0)
        generated_video = self._find_generated_video(session_id, existing_videos)
        predicted_final_frame = self._extract_final_frame(generated_video)
        judged = self.judge.judge(task, initial_frame, real_final_frame, predicted_final_frame)
        task_progress = float(judged["task_progress"])
        rule_success = bool(judged["rule_success"])
        task_success = bool((task_progress > self.success_threshold) or rule_success)
        return {
            "episode_id": row["episode_id"],
            "task": task,
            "mode": mode,
            "sub_instructions": sub_instructions,
            "planner_meta": planner_meta,
            "plan_time": plan_time,
            "mean_l2": float(np.mean(action_errors)) if action_errors else None,
            "num_steps": len(action_errors),
            "num_step_pass_l2_lt_0_1": int(sum(err < 0.1 for err in action_errors)),
            "step_alignment_l2_lt_0_1": float(np.mean([err < 0.1 for err in action_errors])) if action_errors else None,
            "task_progress": task_progress,
            "rule_success": rule_success,
            "task_success": task_success,
            "judge_reason": judged["reason"],
            "generated_video": str(generated_video),
        }


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare AgiBot task_token_only vs dual_llm with judged metrics.")
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--generated-video-dir", type=Path, default=DEFAULT_GENERATED_VIDEO_DIR)
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--eval-steps", type=int, default=3)
    parser.add_argument("--num-episodes", type=int, default=400)
    parser.add_argument("--judge-model", type=str, default="gpt-4o-mini")
    parser.add_argument("--success-threshold", type=float, default=0.75)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("evaluation_results_dualsystem/agibot_easy400_compare_judged.json"),
    )
    args = parser.parse_args()

    runner = AgiBotCompareRunner(
        manifest_path=args.manifest_path,
        generated_video_dir=args.generated_video_dir,
        host=args.host,
        port=args.port,
        eval_steps=args.eval_steps,
        judge_model=args.judge_model,
        success_threshold=args.success_threshold,
    )

    all_results: list[dict] = []
    running: dict[str, list[dict]] = {"task_token_only": [], "dual_llm": []}
    for idx, row in enumerate(runner.episodes[: args.num_episodes], 1):
        print(f"\n[{idx}/{args.num_episodes}] {row['episode_id']} | task={row['english_task_name']}")
        for mode in ("task_token_only", "dual_llm"):
            result = runner.run_episode(row, mode)
            running[mode].append(result)
            all_results.append(result)
            current = summarize(running[mode])
            print(
                "  "
                f"[{mode}] mean_l2={result['mean_l2']:.4f} | "
                f"task_progress={result['task_progress']:.3f} | "
                f"rule_success={'PASS' if result['rule_success'] else 'FAIL'} | "
                f"success={'PASS' if result['task_success'] else 'FAIL'} | "
                f"running_mean_l2={current['mean_l2']:.4f} | "
                f"running_mean_task_progress={current['mean_task_progress']:.3f} | "
                f"running_success_rate={current['success_rate']:.3f} | "
                f"running_rate_l2_lt_0_1={current['rate_of_l2_lt_0_1']:.3f}"
            )
            if mode == "dual_llm":
                print(f"    sub_instructions={result['sub_instructions']}")
        payload = {
            "results": all_results,
            "summary": {mode: summarize(rows) for mode, rows in running.items()},
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")

    final = {
        "results": all_results,
        "summary": {mode: summarize(rows) for mode, rows in running.items()},
    }
    args.output_json.write_text(json.dumps(final, indent=2, ensure_ascii=False) + "\n")
    print("\n=== Summary ===")
    print(json.dumps(final["summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
