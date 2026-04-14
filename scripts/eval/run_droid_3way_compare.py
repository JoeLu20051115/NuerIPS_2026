#!/usr/bin/env python3
"""
DROID 3-way evaluation: task_token_only | dual_llm | llm_val
Identical metric logic to run_agibot_3way_compare.py.
Metrics: Mean L2, Mean Task Progress, Success Rate, Rate of L2 < 0.1

Loads 400 episodes from DRO_L1_150 + DRO_L2_100 + DRO_L3_150.

Usage:
  python scripts/eval/run_droid_3way_compare.py --port 8001 --num-episodes 400
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pyarrow.parquet as pq
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

# ── Camera keys (LeRobot → RoboArena) ────────────────────────────────────────
CAMERA_KEYS = [
    "observation.images.exterior_image_1_left",
    "observation.images.exterior_image_2_left",
    "observation.images.wrist_image_left",
]
ROBOARENA_KEY_MAP = {
    "observation.images.exterior_image_1_left": "observation/exterior_image_0_left",
    "observation.images.exterior_image_2_left": "observation/exterior_image_1_left",
    "observation.images.wrist_image_left": "observation/wrist_image_left",
}

_ROOT = Path("data/final_data1")
_L1_DIR = _ROOT / "DRO_L1_150"
_L2_DIR = _ROOT / "DRO_L2_100"
_L3_DIR = _ROOT / "DRO_L3_150"
# Videos live in the shared source dataset (parquet files are symlinks)
_VIDEO_ROOT = Path("data/droid_easy400_dualfavored_dreamzero")


def _read_jsonl(path: Path) -> list[dict]:
    eps = []
    with open(path) as f:
        for line in f:
            eps.append(json.loads(line))
    return eps


def load_episodes(n: int) -> list[dict]:
    """Load up to n episodes from L1+L2+L3, annotated with dataset_dir."""
    eps = []
    for ddir in (_L1_DIR, _L2_DIR, _L3_DIR):
        for rec in _read_jsonl(ddir / "meta/episodes.jsonl"):
            eps.append({**rec, "_dataset_dir": str(ddir)})
    return eps[:n]


class Droid3WayRunner:
    def __init__(self, host: str, port: int, generated_video_dir: Path,
                 eval_steps: int, judge_model: str, success_threshold: float) -> None:
        self.policy = WebsocketClientPolicy(host=host, port=port)
        self.judge = FinalFrameJudge(judge_model)
        self.planner_dual = LLMPlanner(use_mock=False, temperature=0.0)
        self.planner_val = LLMPlannerWithVAL()
        self.generated_video_dir = generated_video_dir
        self.eval_steps = eval_steps
        self.success_threshold = success_threshold

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _resize(frame: np.ndarray, w: int = 320, h: int = 180) -> np.ndarray:
        if frame.shape[1] == w and frame.shape[0] == h:
            return frame
        return np.asarray(Image.fromarray(frame).resize((w, h), Image.BILINEAR))

    @staticmethod
    def _read_video_frame(video_path: str, frame_idx: int) -> np.ndarray:
        reader = imageio.get_reader(video_path)
        try:
            return reader.get_data(frame_idx)
        finally:
            reader.close()

    @staticmethod
    def _last_frame(video_path: str) -> np.ndarray:
        reader = imageio.get_reader(video_path)
        try:
            n = reader.count_frames()
            return reader.get_data(max(n - 1, 0))
        finally:
            reader.close()

    def _find_video(self, session_id: str, existing: set[str]) -> Path:
        hits = sorted(self.generated_video_dir.glob(f"*{session_id}*.mp4"),
                      key=lambda p: p.stat().st_mtime)
        if hits:
            return hits[-1]
        fresh = sorted([p for p in self.generated_video_dir.glob("*.mp4")
                        if p.name not in existing],
                       key=lambda p: p.stat().st_mtime)
        if fresh:
            return fresh[-1]
        raise FileNotFoundError(f"No video for session_id={session_id}")

    def _video_paths(self, ep_idx: int) -> dict[str, str | None]:
        chunk = ep_idx // 1000
        out = {}
        for cam in CAMERA_KEYS:
            p = _VIDEO_ROOT / f"videos/chunk-{chunk:03d}/{cam}/episode_{ep_idx:06d}.mp4"
            out[cam] = str(p) if p.exists() else None
        return out

    def _plan(self, mode: str, task: str, max_retries: int = 3) -> tuple[list[str], dict, float]:
        if mode == "task_token_only":
            return [task], {"planner_mode": "disabled"}, 0.0
        for attempt in range(max_retries):
            try:
                t0 = time.perf_counter()
                if mode == "dual_llm":
                    sub, meta = self.planner_dual.plan(task, {})
                else:  # llm_val
                    sub, meta = self.planner_val.plan(task, {})
                return sub, meta, time.perf_counter() - t0
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2.0 * (attempt + 1))
                else:
                    raise

    # ── episode runner ────────────────────────────────────────────────────────

    def run_episode(self, ep: dict, mode: str) -> dict:
        ep_idx = int(ep["episode_index"])
        dataset_dir = Path(ep["_dataset_dir"])
        chunk = ep_idx // 1000
        parquet_path = dataset_dir / f"data/chunk-{chunk:03d}/episode_{ep_idx:06d}.parquet"
        df = pq.read_table(str(parquet_path)).to_pandas()
        ep_len = len(df)

        # task_token: "task_N" format (numeric index) — matches old System1 baseline
        # task_desc: natural language from episodes.jsonl — used for LLM planner
        task_index = int(df["task_index"].iloc[0]) if "task_index" in df.columns else ep_idx
        task_token = f"task_{task_index}"
        task_desc = ep.get("tasks", [""])[0]
        if not task_desc:
            task_desc = task_token

        videos = self._video_paths(ep_idx)
        ref_cam = next((k for k in CAMERA_KEYS if videos.get(k)), None)
        if ref_cam is None:
            raise RuntimeError(f"No camera video for ep_idx={ep_idx}")

        sub_instructions, planner_meta, plan_time = self._plan(mode, task_desc)
        session_id = f"dro3_{mode}_{ep_idx}_{uuid.uuid4().hex[:8]}"
        existing = {p.name for p in self.generated_video_dir.glob("*.mp4")}

        eval_indices = np.linspace(0, ep_len - 1, min(self.eval_steps, ep_len), dtype=int)
        errors: list[float] = []
        initial_frame: np.ndarray | None = None

        self.policy.reset({"session_id": session_id})

        for step_idx in eval_indices:
            obs: dict = {}
            for cam_key, vpath in videos.items():
                robo_key = ROBOARENA_KEY_MAP[cam_key]
                if vpath:
                    frame = self._read_video_frame(vpath, int(step_idx))
                    frame = self._resize(frame)
                    obs[robo_key] = frame
                    if initial_frame is None and cam_key == ref_cam:
                        initial_frame = frame
                else:
                    obs[robo_key] = np.zeros((180, 320, 3), dtype=np.uint8)

            state = np.array(df["observation.state"].iloc[step_idx], dtype=np.float64)
            obs["observation/joint_position"] = state[7:14]
            obs["observation/gripper_position"] = state[6:7]

            sub_idx = min(int(step_idx * len(sub_instructions) / max(ep_len, 1)),
                          len(sub_instructions) - 1)
            obs["prompt"] = task_token if mode == "task_token_only" else sub_instructions[sub_idx]
            obs["session_id"] = session_id

            result = self.policy.infer(obs)
            pred = result.get("action") if isinstance(result, dict) else result
            if pred is None:
                continue
            pred_joint = pred[0, :7] if getattr(pred, "ndim", 1) == 2 else pred[:7]
            gt_action = np.array(df["action"].iloc[step_idx], dtype=np.float64)
            gt_joint = gt_action[14:21]
            errors.append(float(np.sqrt(np.sum((pred_joint - gt_joint) ** 2))))

        self.policy.reset({"session_id": session_id})
        time.sleep(1.0)
        gen_video = self._find_video(session_id, existing)
        pred_final = self._last_frame(str(gen_video))
        real_final = self._last_frame(videos[ref_cam])

        judged = self.judge.judge(task_desc, initial_frame, real_final, pred_final)
        task_progress = float(judged["task_progress"])
        rule_success = bool(judged.get("rule_success", False))
        task_success = bool((task_progress > self.success_threshold) or rule_success)

        return {
            "episode_id": str(ep_idx),
            "task": task_desc,
            "task_token": task_token,
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
            "judge_reason": judged.get("reason", ""),
            "generated_video": str(gen_video),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--num-episodes", type=int, default=400)
    parser.add_argument("--eval-steps", type=int, default=3)
    parser.add_argument("--judge-model", default="gpt-4o-mini")
    parser.add_argument("--success-threshold", type=float, default=0.6)
    parser.add_argument("--output-json", default="evaluation_results_dualsystem/droid_3way_compare.json")
    parser.add_argument("--log", default="logs/droid_3way_compare.log")
    args = parser.parse_args()

    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Use the server's existing video output directory (set at server startup)
    gen_dir_candidates = sorted(Path("checkpoints").glob("real_world_eval_gen_*/DreamZero-DROID"),
                                key=lambda p: p.stat().st_mtime, reverse=True)
    if not gen_dir_candidates:
        raise RuntimeError("No DreamZero video output directory found under checkpoints/")
    gen_dir = gen_dir_candidates[0]

    def log(msg: str) -> None:
        print(msg, flush=True)
        with open(log_path, "a") as f:
            f.write(msg + "\n")

    log(f"Using generated video dir: {gen_dir}")
    episodes = load_episodes(args.num_episodes)
    MODES = ["task_token_only", "dual_llm", "llm_val"]

    log("=" * 70)
    log("DROID 3-way compare: task_token_only | dual_llm | llm_val")
    log(f"Episodes: {len(episodes)}  |  Port: {args.port}  |  Eval steps: {args.eval_steps}")
    log("=" * 70)

    runner = Droid3WayRunner(
        host=args.host, port=args.port,
        generated_video_dir=gen_dir,
        eval_steps=args.eval_steps,
        judge_model=args.judge_model,
        success_threshold=args.success_threshold,
    )

    all_results: list[dict] = []
    run_stats: dict[str, dict] = {m: {"l2": [], "prog": [], "sr": [], "l01": [], "plan_time": []} for m in MODES}

    # Resume: load already-completed results
    done_pairs: set[tuple] = set()
    if out_path.exists():
        try:
            with open(out_path) as f:
                existing = json.load(f)
            for r in existing.get("results", []):
                all_results.append(r)
                m = r["mode"]
                if m in run_stats:
                    s = run_stats[m]
                    if r["mean_l2"] is not None:
                        s["l2"].append(r["mean_l2"])
                    s["prog"].append(r["task_progress"])
                    s["sr"].append(1 if r["task_success"] else 0)
                    if r["step_alignment_l2_lt_0_1"] is not None:
                        s["l01"].append(r["step_alignment_l2_lt_0_1"])
                    s["plan_time"].append(r["plan_time"])
                done_pairs.add((str(r["episode_id"]), r["mode"]))
            if done_pairs:
                log(f"Resuming: {len(done_pairs)//len(MODES)} episodes already done, skipping.")
        except Exception as e:
            log(f"Warning: could not load existing results ({e}), starting fresh.")

    for ep_i, ep in enumerate(episodes, 1):
        ep_idx = ep["episode_index"]
        task_short = ep.get("tasks", ["?"])[0][:45]

        # Skip fully-completed episodes
        if all(str(ep_idx) in [k[0] for k in done_pairs if k[1] == m] for m in MODES):
            continue

        log(f"\n[{ep_i:3d}/{len(episodes)}] ep={ep_idx} | {task_short}")

        for mode in MODES:
            if (str(ep_idx), mode) in done_pairs:
                log(f"  [{mode:<18}] skipped (already done)")
                continue
            try:
                res = runner.run_episode(ep, mode)
                all_results.append(res)
                s = run_stats[mode]
                if res["mean_l2"] is not None:
                    s["l2"].append(res["mean_l2"])
                s["prog"].append(res["task_progress"])
                s["sr"].append(1 if res["task_success"] else 0)
                if res["step_alignment_l2_lt_0_1"] is not None:
                    s["l01"].append(res["step_alignment_l2_lt_0_1"])
                s["plan_time"].append(res["plan_time"])

                l2_s   = f"{np.mean(s['l2']):.4f}" if s["l2"] else "N/A"
                prog_s = f"{np.mean(s['prog']):.3f}"
                sr_s   = f"{np.mean(s['sr']):.3f}"
                l01_s  = f"{np.mean(s['l01']):.3f}" if s["l01"] else "N/A"
                pt_s   = f"{np.mean(s['plan_time']):.2f}s"
                subs_note = f"\n    sub={res['sub_instructions']}" if mode == "llm_val" else ""
                log(f"  [{mode:<18}] l2={res['mean_l2'] or 0:.4f} "
                    f"progress={res['task_progress']:.2f} "
                    f"success={'Y' if res['task_success'] else 'N'} "
                    f"plan={res['plan_time']:.1f}s | "
                    f"run: l2={l2_s} prog={prog_s} sr={sr_s} l2<0.1={l01_s} plan={pt_s}"
                    + subs_note)
            except Exception as exc:
                log(f"  [{mode:<18}] ERROR: {exc}")

        # Save checkpoint every 10 episodes
        if ep_i % 10 == 0:
            _save(out_path, all_results)
            log(f"\n  --- Running summary @ {ep_i} episodes ---")
            for m in MODES:
                s = run_stats[m]
                l2v = np.mean(s["l2"]) if s["l2"] else float("nan")
                pt_v = np.mean(s["plan_time"]) if s["plan_time"] else float("nan")
                log(f"  {m:<20}: l2={l2v:.4f} prog={np.mean(s['prog']):.3f} "
                    f"sr={np.mean(s['sr']):.3f} l2<0.1={np.mean(s['l01']) if s['l01'] else float('nan'):.3f} "
                    f"plan={pt_v:.2f}s")

    _save(out_path, all_results)

    log("\n" + "=" * 70)
    log(f"FINAL SUMMARY — {len(episodes)} episodes")
    log("=" * 70)
    log(f"{'Dataset':<8} {'Mode':<20} {'Mean L2':>8} {'Task Prog':>10} {'Success':>8} {'L2<0.1':>8} {'Plan(s)':>8}")
    log("-" * 72)
    summary = {}
    for m in MODES:
        s = run_stats[m]
        mean_l2   = float(np.mean(s["l2"])) if s["l2"] else None
        mean_prog = float(np.mean(s["prog"])) if s["prog"] else None
        mean_sr   = float(np.mean(s["sr"])) if s["sr"] else None
        mean_l01  = float(np.mean(s["l01"])) if s["l01"] else None
        mean_pt   = float(np.mean(s["plan_time"])) if s["plan_time"] else None
        summary[m] = {"mean_l2": mean_l2, "mean_task_progress": mean_prog,
                      "success_rate": mean_sr, "rate_of_l2_lt_0_1": mean_l01,
                      "mean_plan_time": mean_pt}
        log(f"{'DROID':<8} {m:<20} "
            f"{(mean_l2 or 0):>8.4f} {(mean_prog or 0):>10.4f} "
            f"{(mean_sr or 0):>8.3f} {(mean_l01 or 0):>8.3f} {(mean_pt or 0):>8.2f}")

    _save(out_path, all_results, summary)
    log(f"\nSaved → {out_path}")


def _save(path: Path, results: list[dict], summary: dict | None = None) -> None:
    payload: dict = {"results": results}
    if summary:
        payload["summary"] = summary
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


if __name__ == "__main__":
    main()
