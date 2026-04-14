#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
SPLIT_ORDER = ["L1", "L2", "L3"]
MODE_ORDER = ["original", "llm_dual", "llm_val"]
MODE_COLORS = {
    "original": "#6B2D5C",
    "llm_dual": "#2A6F97",
    "llm_val": "#3A7D44",
}
DATASET_ORDER = [
    "DreamZero-DROID",
    "DreamZero-AgiBot",
    "DreamDojo-AgiBot",
    "DreamDojo-EgoDex",
]
DATASET_SLUGS = {
    "DreamZero-DROID": "dreamzero_droid",
    "DreamZero-AgiBot": "dreamzero_agibot",
    "DreamDojo-AgiBot": "dreamdojo_agibot",
    "DreamDojo-EgoDex": "dreamdojo_egodex",
}


ROWS = [
    # DreamZero-DROID
    {"dataset": "DreamZero-DROID", "split": "L1", "mode": "original", "success_rate": 0.30},
    {"dataset": "DreamZero-DROID", "split": "L1", "mode": "llm_dual", "success_rate": 0.44},
    {"dataset": "DreamZero-DROID", "split": "L1", "mode": "llm_val", "success_rate": 0.50},
    {"dataset": "DreamZero-DROID", "split": "L2", "mode": "original", "success_rate": 0.18},
    {"dataset": "DreamZero-DROID", "split": "L2", "mode": "llm_dual", "success_rate": 0.32},
    {"dataset": "DreamZero-DROID", "split": "L2", "mode": "llm_val", "success_rate": 0.38},
    {"dataset": "DreamZero-DROID", "split": "L3", "mode": "original", "success_rate": 0.22},
    {"dataset": "DreamZero-DROID", "split": "L3", "mode": "llm_dual", "success_rate": 0.34},
    {"dataset": "DreamZero-DROID", "split": "L3", "mode": "llm_val", "success_rate": 0.48},
    # DreamZero-AgiBot
    {"dataset": "DreamZero-AgiBot", "split": "L1", "mode": "original", "success_rate": 0.30},
    {"dataset": "DreamZero-AgiBot", "split": "L1", "mode": "llm_dual", "success_rate": 0.52},
    {"dataset": "DreamZero-AgiBot", "split": "L1", "mode": "llm_val", "success_rate": 0.62},
    {"dataset": "DreamZero-AgiBot", "split": "L2", "mode": "original", "success_rate": 0.22},
    {"dataset": "DreamZero-AgiBot", "split": "L2", "mode": "llm_dual", "success_rate": 0.42},
    {"dataset": "DreamZero-AgiBot", "split": "L2", "mode": "llm_val", "success_rate": 0.48},
    {"dataset": "DreamZero-AgiBot", "split": "L3", "mode": "original", "success_rate": 0.16},
    {"dataset": "DreamZero-AgiBot", "split": "L3", "mode": "llm_dual", "success_rate": 0.48},
    {"dataset": "DreamZero-AgiBot", "split": "L3", "mode": "llm_val", "success_rate": 0.58},
    # DreamDojo-AgiBot
    {"dataset": "DreamDojo-AgiBot", "split": "L1", "mode": "original", "success_rate": 0.66},
    {"dataset": "DreamDojo-AgiBot", "split": "L1", "mode": "llm_dual", "success_rate": 0.74},
    {"dataset": "DreamDojo-AgiBot", "split": "L1", "mode": "llm_val", "success_rate": 0.76},
    {"dataset": "DreamDojo-AgiBot", "split": "L2", "mode": "original", "success_rate": 0.34},
    {"dataset": "DreamDojo-AgiBot", "split": "L2", "mode": "llm_dual", "success_rate": 0.44},
    {"dataset": "DreamDojo-AgiBot", "split": "L2", "mode": "llm_val", "success_rate": 0.48},
    {"dataset": "DreamDojo-AgiBot", "split": "L3", "mode": "original", "success_rate": 0.30},
    {"dataset": "DreamDojo-AgiBot", "split": "L3", "mode": "llm_dual", "success_rate": 0.46},
    {"dataset": "DreamDojo-AgiBot", "split": "L3", "mode": "llm_val", "success_rate": 0.50},
    # DreamDojo-EgoDex
    {"dataset": "DreamDojo-EgoDex", "split": "L1", "mode": "original", "success_rate": 0.36},
    {"dataset": "DreamDojo-EgoDex", "split": "L1", "mode": "llm_dual", "success_rate": 0.34},
    {"dataset": "DreamDojo-EgoDex", "split": "L1", "mode": "llm_val", "success_rate": 0.38},
    {"dataset": "DreamDojo-EgoDex", "split": "L2", "mode": "original", "success_rate": 0.30},
    {"dataset": "DreamDojo-EgoDex", "split": "L2", "mode": "llm_dual", "success_rate": 0.44},
    {"dataset": "DreamDojo-EgoDex", "split": "L2", "mode": "llm_val", "success_rate": 0.48},
    {"dataset": "DreamDojo-EgoDex", "split": "L3", "mode": "original", "success_rate": 0.22},
    {"dataset": "DreamDojo-EgoDex", "split": "L3", "mode": "llm_dual", "success_rate": 0.34},
    {"dataset": "DreamDojo-EgoDex", "split": "L3", "mode": "llm_val", "success_rate": 0.40},
]


def build_frame() -> pd.DataFrame:
    df = pd.DataFrame(ROWS)
    df["split"] = pd.Categorical(df["split"], categories=SPLIT_ORDER, ordered=True)
    df["mode"] = pd.Categorical(df["mode"], categories=MODE_ORDER, ordered=True)
    return df.sort_values(["dataset", "split", "mode"]).reset_index(drop=True)


def annotate_bars(ax: plt.Axes, bars: list[plt.Rectangle]) -> None:
    for bar in bars:
        value = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            value + 1.5,
            f"{value:.0f}%",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color="#333333",
        )


def draw_dataset_bars(ax: plt.Axes, sub: pd.DataFrame, dataset: str) -> None:
    x = np.arange(len(SPLIT_ORDER))
    width = 0.22
    offsets = [-width, 0.0, width]
    for offset, mode in zip(offsets, MODE_ORDER):
        mode_rows = sub[sub["mode"] == mode].sort_values("split")
        values = mode_rows["success_rate"].to_numpy() * 100.0
        bars = ax.bar(
            x + offset,
            values,
            width=width,
            color=MODE_COLORS[mode],
            label=mode,
            alpha=0.92,
            edgecolor="white",
            linewidth=0.8,
        )
        annotate_bars(ax, list(bars))

    ax.set_title(dataset, fontsize=12.5, weight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(SPLIT_ORDER, fontsize=10.5)
    ax.set_ylim(0.0, 90.0)
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel("Split", fontsize=10.5)


def build_legend_handles() -> list[plt.Rectangle]:
    return [
        plt.Rectangle((0, 0), 1, 1, color=MODE_COLORS[mode], alpha=0.92, label=mode)
        for mode in MODE_ORDER
    ]


def plot(df: pd.DataFrame, out_png: Path, out_pdf: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.5), sharey=True)
    for ax, dataset in zip(axes.flat, DATASET_ORDER):
        sub = df[df["dataset"] == dataset]
        draw_dataset_bars(ax, sub, dataset)

    axes[0, 0].set_ylabel("Success Rate (%)", fontsize=11)
    axes[1, 0].set_ylabel("Success Rate (%)", fontsize=11)

    fig.legend(
        handles=build_legend_handles(),
        labels=MODE_ORDER,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.995),
    )
    fig.suptitle("Success Rate by Split and Mode Across Four Datasets", fontsize=15, weight="bold", y=0.98)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.94])

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=240, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def plot_single_dataset(df: pd.DataFrame, dataset: str, out_png: Path, out_pdf: Path) -> None:
    sub = df[df["dataset"] == dataset]
    fig, ax = plt.subplots(figsize=(6.6, 4.9))
    draw_dataset_bars(ax, sub, dataset)
    ax.set_ylabel("Success Rate (%)", fontsize=11)
    ax.legend(handles=build_legend_handles(), labels=MODE_ORDER, frameon=False, loc="upper right")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=240, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot SR-only grouped bars for four datasets.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "20260331" / "figures",
        help="Directory for figure and data exports.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df = build_frame()
    df["success_rate_pct"] = df["success_rate"] * 100.0
    df.to_csv(out_dir / "sr_bars_four_datasets.tsv", sep="\t", index=False)

    plot(
        df=df,
        out_png=out_dir / "sr_bars_four_datasets.png",
        out_pdf=out_dir / "sr_bars_four_datasets.pdf",
    )
    for dataset in DATASET_ORDER:
        slug = DATASET_SLUGS[dataset]
        sub = df[df["dataset"] == dataset].copy()
        sub.to_csv(out_dir / f"sr_bars_{slug}.tsv", sep="\t", index=False)
        plot_single_dataset(
            df=df,
            dataset=dataset,
            out_png=out_dir / f"sr_bars_{slug}.png",
            out_pdf=out_dir / f"sr_bars_{slug}.pdf",
        )


if __name__ == "__main__":
    main()
