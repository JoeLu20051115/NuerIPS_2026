#!/usr/bin/env python3
"""
Full 200-episode comparison: System1 (natural language) vs Dual (hybrid).
OXE_testdataset — L3_sequential_200, δ=0.15.
Real-time progress printing.
"""

import json, sys, time, os, re
import numpy as np
import pyarrow.parquet as pq
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from eval_utils.policy_client import WebsocketClientPolicy
from groot.vla.common.utils import get_frames_by_timestamps

# ── Config ──────────────────────────────────────────────────────────
DATASET_NAME = "OXE_testdataset"
HOST, PORT = "localhost", 8000
DATASET_ROOT = Path("data/droid_lerobot")
TEST_SET = "test_sets_final/L3_sequential_200.json"
NUM_EPISODES = 200
EVAL_STEPS = 10
ERROR_THRESHOLD = 0.15

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

# ── LLM Planner ────────────────────────────────────────────────────
def llm_plan(task_description: str) -> list[str]:
    try:
        from openai import OpenAI
        client = OpenAI()
        prompt = (
            f"You are a robotic task planner. Break the following manipulation "
            f"task into 3-5 sequential sub-steps. Return ONLY a JSON list of strings.\n\n"
            f"Task: {task_description}\n\nSub-steps:"
        )
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=300,
        )
        text = resp.choices[0].message.content.strip()
        match = re.search(r'\[.*?\]', text, re.DOTALL)
        if match:
            return json.loads(match.group())
        return [task_description]
    except Exception as e:
        return [task_description]


# ── Episode loading ─────────────────────────────────────────────────
# Pre-load episode metadata for fast task lookup
_ep_meta = {}
_meta_loaded = False
def _ensure_meta():
    global _ep_meta, _meta_loaded
    if _meta_loaded:
        return
    meta_path = DATASET_ROOT / "meta" / "episodes.jsonl"
    if meta_path.exists():
        with open(meta_path) as f:
            for line in f:
                e = json.loads(line)
                _ep_meta[e["episode_index"]] = e
    _meta_loaded = True


def load_episode(ep_info):
    _ensure_meta()
    ep_id = ep_info["episode_id"]
    ep_idx = int(ep_id.split("_")[1])
    chunk = ep_idx // 1000
    parquet_path = DATASET_ROOT / f"data/chunk-{chunk:03d}/episode_{ep_idx:06d}.parquet"
    table = pq.read_table(str(parquet_path))
    df = table.to_pandas()
    video_paths = {}
    for cam_key in CAMERA_KEYS:
        vp = DATASET_ROOT / f"videos/chunk-{chunk:03d}/{cam_key}/episode_{ep_idx:06d}.mp4"
        video_paths[cam_key] = str(vp) if vp.exists() else None
    # Task description from metadata
    task_desc = ""
    meta = _ep_meta.get(ep_idx)
    if meta and meta["tasks"]:
        task_desc = meta["tasks"][0]
    if not task_desc:
        task_idx = int(df["task_index"].iloc[0]) if "task_index" in df.columns else 0
        task_desc = f"task_{task_idx}"
    task_idx = int(df["task_index"].iloc[0]) if "task_index" in df.columns else 0
    task_token = f"task_{task_idx}"
    return df, video_paths, task_desc, task_token, task_idx


def run_inference(client, df, video_paths, prompt_text, episode_id):
    """Run inference at eval_steps points, return per-step L2 errors."""
    timestamps = df["timestamp"].to_numpy()
    ep_len = len(df)
    eval_steps = min(EVAL_STEPS, ep_len)
    eval_indices = np.linspace(0, ep_len - 1, eval_steps, dtype=int)

    step_errors = []
    for step_idx in eval_indices:
        try:
            client.reset({"session_id": f"cmp_{episode_id}_{step_idx}"})
        except Exception:
            client = WebsocketClientPolicy(host=HOST, port=PORT)
            client.reset({"session_id": f"cmp_{episode_id}_{step_idx}"})

        obs = {}
        for cam_key, video_path in video_paths.items():
            rk = ROBOARENA_KEY_MAP[cam_key]
            if video_path:
                try:
                    ts = np.array([timestamps[step_idx]])
                    frame = get_frames_by_timestamps(video_path, ts, video_backend="ffmpeg")
                    obs[rk] = frame[0]
                except Exception:
                    obs[rk] = np.zeros((180, 320, 3), dtype=np.uint8)
            else:
                obs[rk] = np.zeros((180, 320, 3), dtype=np.uint8)

        state = np.array(df["observation.state"].iloc[step_idx], dtype=np.float64)
        obs["observation/joint_position"] = state[7:14].astype(np.float64)
        obs["observation/gripper_position"] = state[6:7].astype(np.float64)
        obs["prompt"] = prompt_text
        obs["session_id"] = f"cmp_{episode_id}_{step_idx}"

        try:
            action_result = client.infer(obs)
        except Exception:
            step_errors.append(float("inf"))
            continue

        if isinstance(action_result, dict):
            pred_action = action_result.get("action")
            if pred_action is None:
                for v in action_result.values():
                    if isinstance(v, np.ndarray) and v.ndim >= 1:
                        pred_action = v
                        break
        elif isinstance(action_result, np.ndarray):
            pred_action = action_result
        else:
            step_errors.append(float("inf"))
            continue

        if pred_action is None:
            step_errors.append(float("inf"))
            continue

        gt_actions = np.array(df["action"].iloc[step_idx], dtype=np.float64)
        gt_joint_pos = gt_actions[14:21]

        if pred_action.ndim == 2:
            pred_joint = pred_action[0, :7]
        elif pred_action.ndim == 1:
            pred_joint = pred_action[:7]
        else:
            step_errors.append(float("inf"))
            continue

        l2 = float(np.sqrt(np.sum((pred_joint - gt_joint_pos) ** 2)))
        step_errors.append(l2)

    return step_errors, client


def run_dual_inference(client, df, video_paths, task_desc, sub_instructions, episode_id):
    """Run dual inference with hybrid prompts (task_desc; subtask)."""
    timestamps = df["timestamp"].to_numpy()
    ep_len = len(df)
    eval_steps = min(EVAL_STEPS, ep_len)
    eval_indices = np.linspace(0, ep_len - 1, eval_steps, dtype=int)

    step_errors = []
    for step_idx in eval_indices:
        sub_idx = int(step_idx * len(sub_instructions) / ep_len)
        sub_idx = min(sub_idx, len(sub_instructions) - 1)
        hybrid_prompt = f"{task_desc}; {sub_instructions[sub_idx]}"

        try:
            client.reset({"session_id": f"dual_{episode_id}_{step_idx}"})
        except Exception:
            client = WebsocketClientPolicy(host=HOST, port=PORT)
            client.reset({"session_id": f"dual_{episode_id}_{step_idx}"})

        obs = {}
        for cam_key, video_path in video_paths.items():
            rk = ROBOARENA_KEY_MAP[cam_key]
            if video_path:
                try:
                    ts = np.array([timestamps[step_idx]])
                    frame = get_frames_by_timestamps(video_path, ts, video_backend="ffmpeg")
                    obs[rk] = frame[0]
                except Exception:
                    obs[rk] = np.zeros((180, 320, 3), dtype=np.uint8)
            else:
                obs[rk] = np.zeros((180, 320, 3), dtype=np.uint8)

        state = np.array(df["observation.state"].iloc[step_idx], dtype=np.float64)
        obs["observation/joint_position"] = state[7:14].astype(np.float64)
        obs["observation/gripper_position"] = state[6:7].astype(np.float64)
        obs["prompt"] = hybrid_prompt
        obs["session_id"] = f"dual_{episode_id}_{step_idx}"

        try:
            action_result = client.infer(obs)
        except Exception:
            step_errors.append(float("inf"))
            continue

        if isinstance(action_result, dict):
            pred_action = action_result.get("action")
            if pred_action is None:
                for v in action_result.values():
                    if isinstance(v, np.ndarray) and v.ndim >= 1:
                        pred_action = v
                        break
        elif isinstance(action_result, np.ndarray):
            pred_action = action_result
        else:
            step_errors.append(float("inf"))
            continue

        if pred_action is None:
            step_errors.append(float("inf"))
            continue

        gt_actions = np.array(df["action"].iloc[step_idx], dtype=np.float64)
        gt_joint_pos = gt_actions[14:21]

        if pred_action.ndim == 2:
            pred_joint = pred_action[0, :7]
        elif pred_action.ndim == 1:
            pred_joint = pred_action[:7]
        else:
            step_errors.append(float("inf"))
            continue

        l2 = float(np.sqrt(np.sum((pred_joint - gt_joint_pos) ** 2)))
        step_errors.append(l2)

    return step_errors, client


# ── Main ────────────────────────────────────────────────────────────
def main():
    print("=" * 80)
    print(f"  {DATASET_NAME} — System1(自然语言) vs Dual(hybrid) 全量评测")
    print(f"  数据集: L3_sequential_200 | Episodes: {NUM_EPISODES} | δ={ERROR_THRESHOLD}")
    print(f"  System1 prompt = task_description (自然语言)")
    print(f"  Dual prompt = task_description + LLM subtask (hybrid)")
    print("=" * 80)
    sys.stdout.flush()

    with open(TEST_SET) as f:
        all_eps = json.load(f)
    episodes = all_eps[:NUM_EPISODES]

    client = WebsocketClientPolicy(host=HOST, port=PORT)
    print(f"✓ 已连接推理服务器 {HOST}:{PORT}\n")
    sys.stdout.flush()

    sys1_results = []
    dual_results = []
    all_sys1_steps = []
    all_dual_steps = []

    start_time = time.time()

    for i, ep_info in enumerate(episodes, 1):
        ep_id = ep_info["episode_id"]
        df, video_paths, task_desc, task_token, task_idx = load_episode(ep_info)
        ep_len = len(df)

        # ── System1: natural language prompt ──
        prompt_sys1 = task_desc  # 自然语言！
        errors_s1, client = run_inference(client, df, video_paths, prompt_sys1, ep_id)
        valid_s1 = [e for e in errors_s1 if e != float("inf")]
        mean_s1 = np.mean(valid_s1) if valid_s1 else float("inf")
        sr_s1 = np.mean([1.0 if e < ERROR_THRESHOLD else 0.0 for e in valid_s1]) if valid_s1 else 0.0
        sys1_results.append({"ep": ep_id, "mean_l2": mean_s1, "sr": sr_s1})
        all_sys1_steps.extend(valid_s1)

        # ── Dual: hybrid prompt ──
        sub_instructions = llm_plan(task_desc)
        errors_dual, client = run_dual_inference(client, df, video_paths, task_desc, sub_instructions, ep_id)
        valid_dual = [e for e in errors_dual if e != float("inf")]
        mean_dual = np.mean(valid_dual) if valid_dual else float("inf")
        sr_dual = np.mean([1.0 if e < ERROR_THRESHOLD else 0.0 for e in valid_dual]) if valid_dual else 0.0
        dual_results.append({"ep": ep_id, "mean_l2": mean_dual, "sr": sr_dual})
        all_dual_steps.extend(valid_dual)

        # ── Real-time progress ──
        diff = mean_dual - mean_s1
        winner = "Dual" if diff < 0 else ("Sys1" if diff > 0 else "Tie")
        elapsed = time.time() - start_time
        eta = elapsed / i * (NUM_EPISODES - i)

        # Running averages
        run_s1_l2 = np.mean([r["mean_l2"] for r in sys1_results])
        run_dual_l2 = np.mean([r["mean_l2"] for r in dual_results])
        run_s1_sr = np.mean([r["sr"] for r in sys1_results])
        run_dual_sr = np.mean([r["sr"] for r in dual_results])

        print(
            f"[{i:3d}/{NUM_EPISODES}] {ep_id} len={ep_len:4d} | "
            f"S1={mean_s1:.4f} D={mean_dual:.4f} Δ={diff:+.4f} {winner:>4s} | "
            f"累计 S1={run_s1_l2:.4f} D={run_dual_l2:.4f} SR:{run_s1_sr:.1%}/{run_dual_sr:.1%} | "
            f"ETA {eta/60:.0f}m"
        )
        sys.stdout.flush()

        # Save checkpoint every 20 episodes
        if i % 20 == 0:
            _save_checkpoint(i, sys1_results, dual_results, all_sys1_steps, all_dual_steps)

    # ── Final summary ──
    _print_final(sys1_results, dual_results, all_sys1_steps, all_dual_steps, time.time() - start_time)


def _save_checkpoint(n, sys1_results, dual_results, all_sys1_steps, all_dual_steps):
    ckpt = {
        "n_episodes": n,
        "sys1_mean_l2": float(np.mean([r["mean_l2"] for r in sys1_results])),
        "dual_mean_l2": float(np.mean([r["mean_l2"] for r in dual_results])),
        "sys1_mean_sr": float(np.mean([r["sr"] for r in sys1_results])),
        "dual_mean_sr": float(np.mean([r["sr"] for r in dual_results])),
        "sys1_results": sys1_results,
        "dual_results": dual_results,
    }
    with open(f"evaluation_results/OXE_full200_checkpoint_{n}.json", "w") as f:
        json.dump(ckpt, f, indent=2)


def _print_final(sys1_results, dual_results, all_sys1_steps, all_dual_steps, elapsed):
    from scipy import stats as sp_stats

    s1_l2s = np.array([r["mean_l2"] for r in sys1_results])
    dl_l2s = np.array([r["mean_l2"] for r in dual_results])
    s1_srs = np.array([r["sr"] for r in sys1_results])
    dl_srs = np.array([r["sr"] for r in dual_results])

    print("\n" + "=" * 80)
    print(f"  {DATASET_NAME} — 最终结果 ({len(sys1_results)} episodes, {elapsed/60:.1f} min)")
    print("=" * 80)
    print(f"{'Metric':<25} {'System1':>15} {'Dual':>15} {'ΔDual-Sys1':>15}")
    print("-" * 80)
    print(f"{'Mean L2':<25} {np.mean(s1_l2s):>15.4f} {np.mean(dl_l2s):>15.4f} {np.mean(dl_l2s)-np.mean(s1_l2s):>+15.4f}")
    print(f"{'Std L2':<25} {np.std(s1_l2s):>15.4f} {np.std(dl_l2s):>15.4f}")
    print(f"{'SR@' + str(ERROR_THRESHOLD):<25} {np.mean(s1_srs):>15.1%} {np.mean(dl_srs):>15.1%} {np.mean(dl_srs)-np.mean(s1_srs):>+15.1%}")

    # Per-step significance
    s1_all = np.array(all_sys1_steps)
    dl_all = np.array(all_dual_steps)
    n_paired = min(len(s1_all), len(dl_all))
    if n_paired > 10:
        t_stat, p_t = sp_stats.ttest_rel(s1_all[:n_paired], dl_all[:n_paired])
        w_stat, p_w = sp_stats.wilcoxon(s1_all[:n_paired], dl_all[:n_paired])
        diff = s1_all[:n_paired] - dl_all[:n_paired]
        cohens_d = np.mean(diff) / np.std(diff, ddof=1)
        print(f"\n{'配对t检验 p值':<25} {p_t:>15.6f}")
        print(f"{'Wilcoxon p值':<25} {p_w:>15.6f}")
        label = "Cohen's d"
        print(f"{label:<25} {cohens_d:>15.4f}")
        print(f"{'样本量':<25} {n_paired:>15d}")

    # Episode-level
    if len(s1_l2s) > 2:
        t_ep, p_ep = sp_stats.ttest_rel(s1_l2s, dl_l2s)
        print(f"{'Episode级t检验 p值':<25} {p_ep:>15.6f}")

    # Win/loss
    dual_wins = int(np.sum(dl_l2s < s1_l2s))
    sys1_wins = int(np.sum(s1_l2s < dl_l2s))
    ties = int(np.sum(s1_l2s == dl_l2s))
    print(f"\nDual胜: {dual_wins}  System1胜: {sys1_wins}  平局: {ties}")
    print("=" * 80)

    # Save final results
    final = {
        "dataset": DATASET_NAME,
        "test_set": TEST_SET,
        "n_episodes": len(sys1_results),
        "error_threshold": ERROR_THRESHOLD,
        "sys1_prompt_mode": "task_description (natural language)",
        "dual_prompt_mode": "hybrid (task_desc; subtask)",
        "sys1_mean_l2": float(np.mean(s1_l2s)),
        "dual_mean_l2": float(np.mean(dl_l2s)),
        "sys1_mean_sr": float(np.mean(s1_srs)),
        "dual_mean_sr": float(np.mean(dl_srs)),
        "dual_wins": dual_wins,
        "sys1_wins": sys1_wins,
        "sys1_results": sys1_results,
        "dual_results": dual_results,
    }
    out = "evaluation_results/OXE_full200_final.json"
    with open(out, "w") as f:
        json.dump(final, f, indent=2)
    print(f"\n结果已保存: {out}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
