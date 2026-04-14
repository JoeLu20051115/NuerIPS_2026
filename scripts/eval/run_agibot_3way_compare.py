#!/usr/bin/env python3
"""
3-way AgiBot evaluation: task_token_only | dual_llm | llm_val
Identical metric computation to run_agibot_compare_judged.py.
Metrics: Mean L2, Mean Task Progress, Success Rate, Rate of L2 < 0.1

Uses Agi_L1_150 + Agi_L3_150 manifests (= same 300 episodes as Agi_300_results.json).

Usage:
  python scripts/eval/run_agibot_3way_compare.py --port 8001 --num-episodes 300
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from pathlib import Path

import h5py
import imageio.v2 as imageio
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))

from run_agibot_tasktoken_judged import FinalFrameJudge

try:
    from eval_utils.policy_client import WebsocketClientPolicy
except ImportError as exc:
    raise RuntimeError("policy_client is required") from exc

try:
    from run_dualsystem_evaluation import LLMPlanner
except ImportError as exc:
    raise RuntimeError("LLMPlanner is required") from exc

from llm_planner_val import LLMPlannerWithVAL

_MANIFEST_L1 = Path("data/final_data1/Agi_L1_150/meta/manifest.json")
_MANIFEST_L3 = Path("data/final_data1/Agi_L3_150/meta/manifest.json")


def load_episodes(n: int) -> list[dict]:
    eps = []
    for mpath in (_MANIFEST_L1, _MANIFEST_L3):
        m = json.loads(mpath.read_text())
        eps.extend(m.get("episodes", m) if isinstance(m, dict) else m)
    return eps[:n]


class AgiBot3WayRunner:
    def __init__(self, host: str, port: int, generated_video_dir: Path,
                 eval_steps: int, judge_model: str, success_threshold: float) -> None:
        self.policy = WebsocketClientPolicy(host=host, port=port)
        self.judge = FinalFrameJudge(judge_model)
        self.planner_dual = LLMPlanner(use_mock=False, temperature=0.0)
        self.planner_val = LLMPlannerWithVAL()
        self.generated_video_dir = generated_video_dir
        self.eval_steps = eval_steps
        self.success_threshold = success_threshold

    @staticmethod
    def _load_frame(video_path: Path, frame_idx: int) -> np.ndarray:
        reader = imageio.get_reader(str(video_path))
        try:
            return reader.get_data(frame_idx)
        finally:
            reader.close()

    @staticmethod
    def _extract_first_frame(path: Path) -> np.ndarray:
        reader = imageio.get_reader(str(path))
        try:
            return reader.get_data(0)
        finally:
            reader.close()

    @staticmethod
    def _extract_final_frame(path: Path) -> np.ndarray:
        reader = imageio.get_reader(str(path))
        try:
            return reader.get_data(reader.count_frames() - 1)
        finally:
            reader.close()

    @staticmethod
    def _resize(frame: np.ndarray, w: int = 320, h: int = 180) -> np.ndarray:
        if frame.shape[1] == w and frame.shape[0] == h:
            return frame
        return np.asarray(Image.fromarray(frame).resize((w, h), Image.BILINEAR))

    @staticmethod
    def _sample_indices(start: int, end: int, n: int) -> np.ndarray:
        start, end = int(max(0, start)), int(max(int(start), end))
        return np.linspace(start, end, min(end - start + 1, n), dtype=int)

    def _build_obs(self, head: np.ndarray, right: np.ndarray) -> dict:
        head, right = self._resize(head), self._resize(right)
        return {
            "observation/exterior_image_0_left": head,
            "observation/exterior_image_1_left": head,
            "observation/wrist_image_left": right,
        }

    def _find_video(self, session_id: str, existing: set[str]) -> Path:
        cands = sorted(self.generated_video_dir.glob(f"*{session_id}*.mp4"),
                       key=lambda p: p.stat().st_mtime)
        if cands:
            return cands[-1]
        fresh = sorted([p for p in self.generated_video_dir.glob("*.mp4")
                        if p.name not in existing],
                       key=lambda p: p.stat().st_mtime)
        if fresh:
            return fresh[-1]
        raise FileNotFoundError(f"No video for session_id={session_id}")

    def _plan(self, mode: str, task: str) -> tuple[list[str], dict, float]:
        if mode == "task_token_only":
            return [task], {"planner_mode": "disabled"}, 0.0
        t0 = time.perf_counter()
        if mode == "dual_llm":
            sub, meta = self.planner_dual.plan(task, {})
        else:  # llm_val
            sub, meta = self.planner_val.plan(task, {})
        return sub, meta, time.perf_counter() - t0

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

        active_start = int(row.get("active_frame_start", 0))
        active_end = int(row.get("active_frame_end", row.get("num_steps", 1) - 1))
        indices = self._sample_indices(active_start, active_end, self.eval_steps)

        task = row.get("english_task_name") or row.get("task_group", "")
        sub_instructions, planner_meta, plan_time = self._plan(mode, task)
        session_id = f"agi3_{mode}_{row['episode_id']}_{uuid.uuid4().hex[:8]}"
        existing = {p.name for p in self.generated_video_dir.glob("*.mp4")}
        errors: list[float] = []

        initial_frame = self._extract_first_frame(head_path)
        real_final = self._extract_final_frame(head_path)
        self.policy.reset({"session_id": session_id})

        for step_idx in indices:
            h_idx = min(int(step_idx), head_frames - 1)
            r_idx = min(int(step_idx), right_frames - 1)
            obs = self._build_obs(
                self._load_frame(head_path, h_idx),
                self._load_frame(right_path, r_idx),
            )
            sub_idx = min(int(step_idx * len(sub_instructions) / max(1, len(states))),
                          len(sub_instructions) - 1)
            prompt = task if mode == "task_token_only" else sub_instructions[sub_idx]
            obs.update({
                "observation/joint_position": states[step_idx, 7:14].astype(np.float64),
                "observation/gripper_position": right_gripper[step_idx].astype(np.float64),
                "prompt": prompt,
                "session_id": session_id,
            })
            result = self.policy.infer(obs)
            pred = result.get("action") if isinstance(result, dict) else result
            if pred is None:
                continue
            pred_joint = pred[0, :7] if getattr(pred, "ndim", 1) == 2 else pred[:7]
            gt_joint = actions[step_idx, 7:14]
            errors.append(float(np.sqrt(np.sum((pred_joint - gt_joint) ** 2))))

        self.policy.reset({"session_id": session_id})
        time.sleep(1.0)
        gen_video = self._find_video(session_id, existing)
        pred_final = self._extract_final_frame(gen_video)
        judged = self.judge.judge(task, initial_frame, real_final, pred_final)
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
            "mean_l2": float(np.mean(errors)) if errors else None,
            "num_steps": len(errors),
            "num_step_pass_l2_lt_0_1": int(sum(e < 0.1 for e in errors)),
            "step_alignment_l2_lt_0_1": float(np.mean([e < 0.1 for e in errors])) if errors else None,
            "task_progress": task_progress,
            "rule_success": rule_success,
            "task_success": task_success,
            "judge_reason": judged["reason"],
            "generated_video": str(gen_video),
        }


def summarize(rows: list[dict]) -> dict:
    valid = [r for r in rows if r["mean_l2"] is not None]
    return {
        "num_episodes": len(rows),
        "mean_l2": float(np.mean([r["mean_l2"] for r in valid])) if valid else None,
        "mean_task_progress": float(np.mean([r["task_progress"] for r in rows])) if rows else None,
        "success_rate": float(np.mean([1.0 if r["task_success"] else 0.0 for r in rows])) if rows else None,
        "rate_of_l2_lt_0_1": (
            float(sum(r["num_step_pass_l2_lt_0_1"] for r in rows) /
                  sum(r["num_steps"] for r in rows))
            if rows and sum(r["num_steps"] for r in rows) > 0 else None
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--num-episodes", type=int, default=300)
    parser.add_argument("--eval-steps", type=int, default=3)
    parser.add_argument("--judge-model", default="gpt-4o-mini")
    parser.add_argument("--success-threshold", type=float, default=0.75)
    parser.add_argument("--generated-video-dir", type=Path,
                        default=Path("checkpoints/real_world_eval_gen_20260329_1/DreamZero-DROID"))
    parser.add_argument("--output-json", type=Path,
                        default=Path("evaluation_results_dualsystem/agibot_3way_compare.json"))
    parser.add_argument("--log", type=Path,
                        default=Path("logs/agibot_3way_compare.log"))
    args = parser.parse_args()

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.generated_video_dir.mkdir(parents=True, exist_ok=True)

    def log(msg: str):
        print(msg, flush=True)
        with args.log.open("a") as f:
            f.write(msg + "\n")

    # Resume logic: load existing results
    all_results: list[dict] = []
    running: dict[str, list[dict]] = {m: [] for m in ("task_token_only", "dual_llm", "llm_val")}
    done_pairs: set[tuple[str, str]] = set()
    if args.output_json.exists():
        try:
            prev = json.loads(args.output_json.read_text())
            prev_results = prev.get("results", [])
            for r in prev_results:
                done_pairs.add((r["episode_id"], r["mode"]))
                running[r["mode"]].append(r)
            all_results = list(prev_results)
            with args.log.open("a") as f:
                f.write(f"[RESUME] Loaded {len(prev_results)} results, {len(done_pairs)} done pairs\n")
            print(f"[RESUME] Loaded {len(prev_results)} results, {len(done_pairs)} done pairs", flush=True)
        except Exception as e:
            print(f"Warning: could not load existing results: {e}", flush=True)
    else:
        args.log.write_text("")

    log(f"{'='*70}")
    log(f"AgiBot 3-way compare: task_token_only | dual_llm | llm_val")
    log(f"Episodes: {args.num_episodes}  |  Port: {args.port}  |  Eval steps: {args.eval_steps}")
    log(f"{'='*70}\n")

    episodes = load_episodes(args.num_episodes)
    runner = AgiBot3WayRunner(
        host=args.host, port=args.port,
        generated_video_dir=args.generated_video_dir,
        eval_steps=args.eval_steps,
        judge_model=args.judge_model,
        success_threshold=args.success_threshold,
    )

    for idx, row in enumerate(episodes, 1):
        eid = row["episode_id"]
        task = row.get("english_task_name") or row.get("task_group", "")
        if all((eid, m) in done_pairs for m in ("task_token_only", "dual_llm", "llm_val")):
            log(f"\n[{idx:3d}/{args.num_episodes}] ep={eid} | SKIP (all done)")
            continue
        log(f"\n[{idx:3d}/{args.num_episodes}] ep={eid} | {task}")

        for mode in ("task_token_only", "dual_llm", "llm_val"):
            if (eid, mode) in done_pairs:
                log(f"  [{mode}] SKIP")
                continue
            try:
                result = runner.run_episode(row, mode)
            except Exception as e:
                log(f"  [{mode}] ERROR: {type(e).__name__}: {str(e)[:80]}")
                continue
            running[mode].append(result)
            all_results.append(result)
            cur = summarize(running[mode])
            log(
                f"  [{mode:17s}] l2={result['mean_l2']:.4f} "
                f"progress={result['task_progress']:.2f} "
                f"success={'Y' if result['task_success'] else 'N'} | "
                f"run: l2={cur['mean_l2']:.4f} prog={cur['mean_task_progress']:.3f} "
                f"sr={cur['success_rate']:.3f} l2<0.1={cur['rate_of_l2_lt_0_1']:.3f}"
            )
            if mode == "llm_val":
                log(f"    sub_instructions={result['sub_instructions']}")

        if idx % 10 == 0:
            payload = {"results": all_results,
                       "summary": {m: summarize(r) for m, r in running.items()}}
            args.output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
            log(f"\n  --- Running summary @ {idx} episodes ---")
            for m, r in running.items():
                s = summarize(r)
                log(f"  {m:17s}: l2={s['mean_l2']:.4f} prog={s['mean_task_progress']:.3f} "
                    f"sr={s['success_rate']:.3f} l2<0.1={s['rate_of_l2_lt_0_1']:.3f}")

    final = {"results": all_results,
             "summary": {m: summarize(r) for m, r in running.items()}}
    args.output_json.write_text(json.dumps(final, indent=2, ensure_ascii=False))

    log(f"\n{'='*70}")
    log(f"FINAL SUMMARY — {args.num_episodes} episodes")
    log(f"{'='*70}")
    log(f"{'Dataset':<6} {'Mode':<18} {'Mean L2':>8} {'Task Prog':>10} {'Success':>8} {'L2<0.1':>8}")
    log(f"{'-'*60}")
    for m, r in running.items():
        s = summarize(r)
        log(f"{'AgiBot':<6} {m:<18} {s['mean_l2']:>8.4f} {s['mean_task_progress']:>10.4f} "
            f"{s['success_rate']:>8.3f} {s['rate_of_l2_lt_0_1']:>8.3f}")
    log(f"\nSaved → {args.output_json}")


if __name__ == "__main__":
    main()
