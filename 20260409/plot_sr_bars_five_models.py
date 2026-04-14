#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SPLIT_ORDER = ["L1", "L2", "L3"]
MODE_ORDER = ["original", "llm_dual", "llm_val"]
MODE_LABELS = {
    "original": "Base",
    "llm_dual": "LLM-Plan",
    "llm_val": "Full LOGIV",
}
MODE_COLORS = {
    "original": "#7A4567",
    "llm_dual": "#3E7EA5",
    "llm_val": "#4D8E5E",
}
DISPLAY_TITLES = {
    "DreamZero-DROID": "DreamZero-DROID",
    "DreamZero-AgiBot": "DreamZero-AgiBot",
    "DreamDojo-AgiBot": "DreamDojo-AgiBot",
    "DreamDojo-EgoDex": "DreamDojo-EgoDex",
    "LingBot-VA-RoboTwin": "LingBot-VA - RoboTwin",
}
DATASET_ORDER = [
    "DreamZero-DROID",
    "DreamZero-AgiBot",
    "DreamDojo-AgiBot",
    "DreamDojo-EgoDex",
    "LingBot-VA-RoboTwin",
]

PNG_DPI = 360
BAR_WIDTH = 0.23
TITLE_SIZE = 12.2
AXIS_LABEL_SIZE = 10.2
TICK_LABEL_SIZE = 8.8
ANNOTATION_SIZE = 7.7
LEGEND_SIZE = 13.2
MAIN_FIGSIZE = (19.2, 4.25)


ROWS = [
    {"dataset": "DreamZero-DROID", "split": "L1", "mode": "original", "success_rate": 0.30},
    {"dataset": "DreamZero-DROID", "split": "L1", "mode": "llm_dual", "success_rate": 0.44},
    {"dataset": "DreamZero-DROID", "split": "L1", "mode": "llm_val", "success_rate": 0.50},
    {"dataset": "DreamZero-DROID", "split": "L2", "mode": "original", "success_rate": 0.18},
    {"dataset": "DreamZero-DROID", "split": "L2", "mode": "llm_dual", "success_rate": 0.32},
    {"dataset": "DreamZero-DROID", "split": "L2", "mode": "llm_val", "success_rate": 0.38},
    {"dataset": "DreamZero-DROID", "split": "L3", "mode": "original", "success_rate": 0.22},
    {"dataset": "DreamZero-DROID", "split": "L3", "mode": "llm_dual", "success_rate": 0.34},
    {"dataset": "DreamZero-DROID", "split": "L3", "mode": "llm_val", "success_rate": 0.48},
    {"dataset": "DreamZero-AgiBot", "split": "L1", "mode": "original", "success_rate": 0.30},
    {"dataset": "DreamZero-AgiBot", "split": "L1", "mode": "llm_dual", "success_rate": 0.52},
    {"dataset": "DreamZero-AgiBot", "split": "L1", "mode": "llm_val", "success_rate": 0.62},
    {"dataset": "DreamZero-AgiBot", "split": "L2", "mode": "original", "success_rate": 0.22},
    {"dataset": "DreamZero-AgiBot", "split": "L2", "mode": "llm_dual", "success_rate": 0.42},
    {"dataset": "DreamZero-AgiBot", "split": "L2", "mode": "llm_val", "success_rate": 0.48},
    {"dataset": "DreamZero-AgiBot", "split": "L3", "mode": "original", "success_rate": 0.16},
    {"dataset": "DreamZero-AgiBot", "split": "L3", "mode": "llm_dual", "success_rate": 0.48},
    {"dataset": "DreamZero-AgiBot", "split": "L3", "mode": "llm_val", "success_rate": 0.58},
    {"dataset": "DreamDojo-AgiBot", "split": "L1", "mode": "original", "success_rate": 0.66},
    {"dataset": "DreamDojo-AgiBot", "split": "L1", "mode": "llm_dual", "success_rate": 0.74},
    {"dataset": "DreamDojo-AgiBot", "split": "L1", "mode": "llm_val", "success_rate": 0.76},
    {"dataset": "DreamDojo-AgiBot", "split": "L2", "mode": "original", "success_rate": 0.34},
    {"dataset": "DreamDojo-AgiBot", "split": "L2", "mode": "llm_dual", "success_rate": 0.44},
    {"dataset": "DreamDojo-AgiBot", "split": "L2", "mode": "llm_val", "success_rate": 0.48},
    {"dataset": "DreamDojo-AgiBot", "split": "L3", "mode": "original", "success_rate": 0.30},
    {"dataset": "DreamDojo-AgiBot", "split": "L3", "mode": "llm_dual", "success_rate": 0.46},
    {"dataset": "DreamDojo-AgiBot", "split": "L3", "mode": "llm_val", "success_rate": 0.50},
    {"dataset": "DreamDojo-EgoDex", "split": "L1", "mode": "original", "success_rate": 0.36},
    {"dataset": "DreamDojo-EgoDex", "split": "L1", "mode": "llm_dual", "success_rate": 0.34},
    {"dataset": "DreamDojo-EgoDex", "split": "L1", "mode": "llm_val", "success_rate": 0.38},
    {"dataset": "DreamDojo-EgoDex", "split": "L2", "mode": "original", "success_rate": 0.30},
    {"dataset": "DreamDojo-EgoDex", "split": "L2", "mode": "llm_dual", "success_rate": 0.44},
    {"dataset": "DreamDojo-EgoDex", "split": "L2", "mode": "llm_val", "success_rate": 0.48},
    {"dataset": "DreamDojo-EgoDex", "split": "L3", "mode": "original", "success_rate": 0.22},
    {"dataset": "DreamDojo-EgoDex", "split": "L3", "mode": "llm_dual", "success_rate": 0.34},
    {"dataset": "DreamDojo-EgoDex", "split": "L3", "mode": "llm_val", "success_rate": 0.40},
    {"dataset": "LingBot-VA-RoboTwin", "split": "L1", "mode": "original", "success_rate": 0.72},
    {"dataset": "LingBot-VA-RoboTwin", "split": "L1", "mode": "llm_dual", "success_rate": 0.78},
    {"dataset": "LingBot-VA-RoboTwin", "split": "L1", "mode": "llm_val", "success_rate": 0.86},
    {"dataset": "LingBot-VA-RoboTwin", "split": "L2", "mode": "original", "success_rate": 0.70},
    {"dataset": "LingBot-VA-RoboTwin", "split": "L2", "mode": "llm_dual", "success_rate": 0.74},
    {"dataset": "LingBot-VA-RoboTwin", "split": "L2", "mode": "llm_val", "success_rate": 0.84},
    {"dataset": "LingBot-VA-RoboTwin", "split": "L3", "mode": "original", "success_rate": 0.66},
    {"dataset": "LingBot-VA-RoboTwin", "split": "L3", "mode": "llm_dual", "success_rate": 0.74},
    {"dataset": "LingBot-VA-RoboTwin", "split": "L3", "mode": "llm_val", "success_rate": 0.78},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot paper-ready success-rate bar charts for the five-model comparison."
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "figures",
        help="Directory for figure exports.",
    )
    return parser.parse_args()


def build_frame() -> pd.DataFrame:
    df = pd.DataFrame(ROWS)
    df["dataset"] = pd.Categorical(df["dataset"], categories=DATASET_ORDER, ordered=True)
    df["split"] = pd.Categorical(df["split"], categories=SPLIT_ORDER, ordered=True)
    df["mode"] = pd.Categorical(df["mode"], categories=MODE_ORDER, ordered=True)
    return df.sort_values(["dataset", "split", "mode"]).reset_index(drop=True)


def set_panel_title(ax: plt.Axes, title: str, panel_letter: str | None = None) -> None:
    label = f"{panel_letter}. {title}" if panel_letter else title
    ax.set_title(label, loc="left", fontsize=TITLE_SIZE, fontweight="bold", pad=10, color="#111827")


def style_axes(
    ax: plt.Axes,
    *,
    xticks: np.ndarray,
    xticklabels: list[str],
    xlim: tuple[float, float],
    show_ylabel: bool,
    show_xlabel: bool,
) -> None:
    ax.set_facecolor("#FCFCFD")
    ax.set_axisbelow(True)
    ax.set_ylim(0.0, 90.0)
    ax.set_yticks(np.arange(0.0, 91.0, 20.0))
    ax.set_xlim(*xlim)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, fontsize=TICK_LABEL_SIZE)
    ax.tick_params(axis="x", labelsize=TICK_LABEL_SIZE, pad=3, colors="#344054")
    ax.tick_params(axis="y", labelsize=TICK_LABEL_SIZE, colors="#344054")
    ax.grid(axis="y", linestyle=(0, (3, 3)), linewidth=0.75, color="#D9E0E6")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#A9B3BD")
        ax.spines[side].set_linewidth(0.9)

    if show_ylabel:
        ax.set_ylabel("Success Rate (%)", fontsize=AXIS_LABEL_SIZE, color="#111827")
    if show_xlabel:
        ax.set_xlabel("Split", fontsize=AXIS_LABEL_SIZE, labelpad=5, color="#111827")


def annotate_bars(ax: plt.Axes, bars: list[plt.Rectangle]) -> None:
    for bar in bars:
        value = bar.get_height()
        x = bar.get_x() + bar.get_width() / 2.0
        ax.text(
            x,
            value + 1.2,
            f"{value:.0f}%",
            ha="center",
            va="bottom",
            fontsize=ANNOTATION_SIZE,
            fontweight="semibold",
            color="#111111",
            clip_on=False,
        )


def build_mode_arrays(sub: pd.DataFrame) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for mode in MODE_ORDER:
        mode_rows = sub[sub["mode"] == mode].sort_values("split")
        arrays[mode] = mode_rows["success_rate"].to_numpy(dtype=float) * 100.0
    return arrays


def draw_mode_bars(
    ax: plt.Axes,
    x_positions: np.ndarray,
    values_by_mode: dict[str, np.ndarray],
) -> None:
    offsets = np.array([-BAR_WIDTH, 0.0, BAR_WIDTH])
    for offset, mode in zip(offsets, MODE_ORDER):
        bars = ax.bar(
            x_positions + offset,
            values_by_mode[mode],
            width=BAR_WIDTH * 0.94,
            color=MODE_COLORS[mode],
            edgecolor="white",
            linewidth=0.85,
            zorder=3,
        )
        annotate_bars(ax, list(bars))


def draw_standard_panel(
    ax: plt.Axes,
    sub: pd.DataFrame,
    title: str,
    *,
    panel_letter: str | None,
    show_ylabel: bool,
    show_xlabel: bool,
) -> None:
    x = np.arange(len(SPLIT_ORDER), dtype=float)
    draw_mode_bars(ax, x, build_mode_arrays(sub))
    style_axes(
        ax,
        xticks=x,
        xticklabels=SPLIT_ORDER,
        xlim=(-0.55, 2.55),
        show_ylabel=show_ylabel,
        show_xlabel=show_xlabel,
    )
    set_panel_title(ax, title, panel_letter)


def build_legend_handles() -> list[mpatches.Patch]:
    return [
        mpatches.Patch(
            facecolor=MODE_COLORS[mode],
            edgecolor="white",
            linewidth=0.8,
            label=MODE_LABELS[mode],
        )
        for mode in MODE_ORDER
    ]


def save_figure(fig: plt.Figure, png_path: Path, pdf_path: Path) -> None:
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=PNG_DPI, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.03)


def plot_main_figure(df: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 5, figsize=MAIN_FIGSIZE, sharey=True)
    fig.patch.set_facecolor("white")
    panel_letters = ["A", "B", "C", "D", "E"]
    for ax, dataset, panel_letter in zip(axes, DATASET_ORDER, panel_letters):
        draw_standard_panel(
            ax,
            df[df["dataset"] == dataset],
            DISPLAY_TITLES[dataset],
            panel_letter=panel_letter,
            show_ylabel=(dataset == DATASET_ORDER[0]),
            show_xlabel=False,
        )

    fig.legend(
        handles=build_legend_handles(),
        labels=[MODE_LABELS[mode] for mode in MODE_ORDER],
        loc="upper right",
        bbox_to_anchor=(0.995, 1.02),
        ncol=3,
        frameon=False,
        prop={"size": LEGEND_SIZE, "weight": "semibold"},
        handlelength=1.7,
        columnspacing=1.4,
        handletextpad=0.55,
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.83], w_pad=0.9)
    save_figure(fig, out_dir / "sr_bars_five_models.png", out_dir / "sr_bars_five_models.pdf")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    df = build_frame()
    df["success_rate_pct"] = df["success_rate"] * 100.0
    df.to_csv(out_dir / "sr_bars_five_models.tsv", sep="\t", index=False)

    plot_main_figure(df, out_dir)


if __name__ == "__main__":
    main()
