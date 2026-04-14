#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]

DROID_SELECTED150_PATH = REPO_ROOT / "evaluation_results_dualsystem" / "droid_3way_selected150.json"
AGIBOT_3WAY_PATH = REPO_ROOT / "evaluation_results_dualsystem" / "agibot_3way_compare.json"
AGIBOT_SPLIT_RESULT_DIR = REPO_ROOT / "evaluation_results_dualsystem" / "selected_splits"
AGIBOT_SPLIT_DATA_DIR = REPO_ROOT / "data"

SPLITS = ["L1", "L2", "L3"]
MODES = ["original", "llm_dual", "llm_val"]
MODE_COLORS = {
    "original": "#6B2D5C",
    "llm_dual": "#2A6F97",
    "llm_val": "#3A7D44",
}


def load_droid_points() -> pd.DataFrame:
    payload = json.loads(DROID_SELECTED150_PATH.read_text())
    rows = []
    for row in payload["results"]:
        mode = row["mode"]
        if mode == "task_token_only":
            mode = "original"
        elif mode == "dual_llm":
            mode = "llm_dual"
        elif mode == "llm_val":
            mode = "llm_val"
        else:
            continue
        rows.append(
            {
                "dataset": "DROID",
                "episode_id": str(row["episode_id"]),
                "split": row["dro_split"],
                "mode": mode,
                "duration_sec": float(row["ep_len"]) / 15.0,
                "task_success": 1.0 if row["task_success"] else 0.0,
                "task_progress": float(row["task_progress"]),
                "mean_l2": float(row["mean_l2"]),
                "source": "droid_3way_selected150.json",
            }
        )
    return pd.DataFrame(rows)


def load_agibot_split_manifests() -> dict[str, dict[str, dict]]:
    manifests: dict[str, dict[str, dict]] = {}
    for split in SPLITS:
        path = AGIBOT_SPLIT_DATA_DIR / f"Agi_DualBetter_{split}_50" / "meta" / "manifest.json"
        payload = json.loads(path.read_text())
        manifests[split] = {str(ep["episode_id"]): ep for ep in payload["episodes"]}
    return manifests


def load_agibot_points() -> pd.DataFrame:
    manifests = load_agibot_split_manifests()
    rows = []

    # Exact split logs for original + llm_dual.
    for split in SPLITS:
        path = AGIBOT_SPLIT_RESULT_DIR / f"Agi_DualBetter_{split}_50_results.json"
        payload = json.loads(path.read_text())
        for ep in payload["episodes"]:
            rows.append(
                {
                    "dataset": "AgiBot",
                    "episode_id": str(ep["episode_id"]),
                    "split": split,
                    "mode": "original",
                    "duration_sec": float(ep["duration_sec"]),
                    "task_success": 1.0 if ep["success_original"] else 0.0,
                    "task_progress": float(ep["progress_original"]),
                    "mean_l2": float(ep["mean_l2_original"]),
                    "source": f"Agi_DualBetter_{split}_50_results.json",
                }
            )
            rows.append(
                {
                    "dataset": "AgiBot",
                    "episode_id": str(ep["episode_id"]),
                    "split": split,
                    "mode": "llm_dual",
                    "duration_sec": float(ep["duration_sec"]),
                    "task_success": 1.0 if ep["success_llm_dual"] else 0.0,
                    "task_progress": float(ep["progress_llm_dual"]),
                    "mean_l2": float(ep["mean_l2_llm_dual"]),
                    "source": f"Agi_DualBetter_{split}_50_results.json",
                }
            )

    # Archived 3-way file for llm_val on the same split episode ids when present.
    three_way = json.loads(AGIBOT_3WAY_PATH.read_text())["results"]
    split_lookup = {}
    for split in SPLITS:
        for eid, ep in manifests[split].items():
            split_lookup[eid] = (split, float(ep["duration_sec"]))
    for row in three_way:
        mode = row["mode"]
        if mode not in ("llm_val", "val_llm"):
            continue
        eid = str(row["episode_id"])
        if eid not in split_lookup:
            continue
        split, duration_sec = split_lookup[eid]
        rows.append(
            {
                "dataset": "AgiBot",
                "episode_id": eid,
                "split": split,
                "mode": "llm_val",
                "duration_sec": duration_sec,
                "task_success": 1.0 if row["task_success"] else 0.0,
                "task_progress": float(row["task_progress"]),
                "mean_l2": float(row["mean_l2"]),
                "source": "agibot_3way_compare.json",
            }
        )

    return pd.DataFrame(rows)


def rolling_curve(df: pd.DataFrame, window: int) -> pd.DataFrame:
    sub = df.sort_values("duration_sec").reset_index(drop=True).copy()
    if len(sub) < window:
        window = max(3, len(sub))
    xs = sub["duration_sec"].rolling(window=window, center=True, min_periods=window).mean()
    ys = sub["task_success"].rolling(window=window, center=True, min_periods=window).mean()
    out = pd.DataFrame({"duration_sec": xs, "rolling_success_rate": ys}).dropna()
    return out


def plot_dataset(df: pd.DataFrame, dataset: str, out_path: Path, window: int, title_suffix: str) -> None:
    sub = df[df["dataset"] == dataset].copy()
    rng = np.random.default_rng(7)

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    for mode in MODES:
        mode_rows = sub[sub["mode"] == mode].sort_values("duration_sec").copy()
        if mode_rows.empty:
            continue
        jitter = rng.normal(0.0, 0.018, size=len(mode_rows))
        ax.scatter(
            mode_rows["duration_sec"],
            np.clip(mode_rows["task_success"] + jitter, 0.0, 1.0),
            s=16,
            alpha=0.18,
            color=MODE_COLORS[mode],
            edgecolors="none",
        )
        curve = rolling_curve(mode_rows, window=window)
        ax.plot(
            curve["duration_sec"],
            curve["rolling_success_rate"],
            linewidth=2.4,
            color=MODE_COLORS[mode],
            label=f"{mode} (n={len(mode_rows)})",
        )

    ax.set_xlabel("Episode Duration (s)")
    ax.set_ylabel("Success / Rolling Success Rate")
    ax.set_ylim(-0.03, 1.03)
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(frameon=False, loc="upper right")
    ax.set_title(f"{dataset}: Time vs Accuracy ({title_suffix})")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=240, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def write_summary(out_path: Path, df: pd.DataFrame) -> None:
    lines = [
        "# Episode-Level Time-Accuracy Recovery",
        "",
        f"- Total recovered rows: {len(df)}",
        f"- DROID rows: {len(df[df['dataset'] == 'DROID'])}",
        f"- AgiBot rows: {len(df[df['dataset'] == 'AgiBot'])}",
        "",
        "## Per-Dataset / Mode Counts",
    ]
    counts = (
        df.groupby(["dataset", "mode", "split"], observed=False)
        .size()
        .reset_index(name="n")
        .sort_values(["dataset", "mode", "split"])
    )
    for _, row in counts.iterrows():
        lines.append(
            f"- {row['dataset']} {row['mode']} {row['split']}: n={int(row['n'])}"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "- DROID points come directly from `droid_3way_selected150.json`, so the recovery is exact for all 450 episode-mode rows.",
            "- AgiBot `original` and `llm_dual` come directly from the archived split result JSONs (`Agi_DualBetter_L1/L2/L3_50_results.json`).",
            "- AgiBot `llm_val` is recovered from the archived `agibot_3way_compare.json` on the same split episode ids when present.",
            "",
        ]
    )
    out_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Recover dense episode-level time-accuracy points from DreamZero appendix logs.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "analysis_outputs" / "dreamzero_episode_time_accuracy_20260330",
    )
    parser.add_argument("--rolling-window-droid", type=int, default=15)
    parser.add_argument("--rolling-window-agibot", type=int, default=12)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    droid = load_droid_points()
    agibot = load_agibot_points()
    all_rows = pd.concat([droid, agibot], ignore_index=True)
    all_rows.to_csv(args.output_dir / "episode_time_accuracy_points.csv", index=False)

    counts = (
        all_rows.groupby(["dataset", "mode", "split"], observed=False)
        .size()
        .reset_index(name="n")
    )
    counts.to_csv(args.output_dir / "episode_time_accuracy_counts.csv", index=False)

    plot_dataset(
        all_rows,
        dataset="DROID",
        out_path=args.output_dir / "droid_episode_time_accuracy_dense.png",
        window=args.rolling_window_droid,
        title_suffix=f"all recovered points, rolling window={args.rolling_window_droid}",
    )
    plot_dataset(
        all_rows,
        dataset="AgiBot",
        out_path=args.output_dir / "agibot_episode_time_accuracy_dense.png",
        window=args.rolling_window_agibot,
        title_suffix=f"all recovered points, rolling window={args.rolling_window_agibot}",
    )
    write_summary(args.output_dir / "summary.md", all_rows)

    print(f"Wrote dense episode-level analysis to {args.output_dir}")


if __name__ == "__main__":
    main()
