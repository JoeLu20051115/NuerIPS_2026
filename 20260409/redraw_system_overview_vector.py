#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle


OUT_DIR = Path(__file__).resolve().parent / "figures"
PDF_PATH = OUT_DIR / "system_overview_vector.pdf"
SVG_PATH = OUT_DIR / "system_overview_vector.svg"
PNG_PATH = OUT_DIR / "system_overview_vector.png"

W, H = 100.0, 90.0

TOP_BG = "#ECF3FA"
BOTTOM_BG = "#E9FAF6"
TOP_BOX = "#2C6791"
TOP_BOX_EDGE = "#1E4F74"
OP_BOX = "#F7B248"
OP_BOX_EDGE = "#B77217"
MINT_BOX = "#DDF4F0"
MINT_BOX_STRONG = "#89CCC7"
MINT_BOX_EDGE = "#4E8E92"
TEXT = "#101418"
ACCENT = "#1C4966"
MID_GREY = "#647484"
DARK_GREY = "#29333D"
ROBOT_GREY = "#A6AFB9"
ROBOT_DARK = "#6F7882"

HEADER_FS = 17
PANEL_FS = 16
BOX_FS = 14
SMALL_FS = 12
MATH_FS = 18


def add_round_box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    fc: str,
    ec: str,
    lw: float = 1.8,
    rounding: float = 1.6,
    dash: tuple[int, ...] | None = None,
    z: int = 1,
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.02,rounding_size={rounding}",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        linestyle=(0, dash) if dash else "solid",
        joinstyle="round",
        zorder=z,
    )
    ax.add_patch(patch)
    return patch


def add_text(
    ax: plt.Axes,
    x: float,
    y: float,
    text: str,
    *,
    fs: float,
    weight: str = "normal",
    color: str = TEXT,
    ha: str = "center",
    va: str = "center",
    linespacing: float = 1.0,
    z: int = 5,
) -> None:
    ax.text(
        x,
        y,
        text,
        fontsize=fs,
        fontweight=weight,
        color=color,
        ha=ha,
        va=va,
        linespacing=linespacing,
        zorder=z,
    )


def draw_poly_arrow(
    ax: plt.Axes,
    points: list[tuple[float, float]],
    *,
    color: str = ACCENT,
    lw: float = 2.0,
    head_scale: float = 22.0,
    z: int = 4,
) -> None:
    if len(points) < 2:
        return
    if len(points) > 2:
        xs = [pt[0] for pt in points[:-1]]
        ys = [pt[1] for pt in points[:-1]]
        ax.add_line(Line2D(xs, ys, color=color, linewidth=lw, zorder=z))
    arrow = FancyArrowPatch(
        points[-2],
        points[-1],
        arrowstyle="-|>",
        mutation_scale=head_scale,
        linewidth=lw,
        color=color,
        shrinkA=0.0,
        shrinkB=0.0,
        capstyle="round",
        joinstyle="round",
        zorder=z + 0.1,
    )
    ax.add_patch(arrow)


def draw_robot_icon(ax: plt.Axes, *, x_shift: float = 0.0, y_shift: float = 0.0) -> None:
    def pt(x: float, y: float) -> tuple[float, float]:
        return x + x_shift, y + y_shift

    ax.add_patch(
        Polygon(
            [pt(77.4, 28.3), pt(87.0, 28.3), pt(91.7, 31.0), pt(82.0, 31.0)],
            closed=True,
            facecolor="#DDE1E6",
            edgecolor="#9AA4AE",
            linewidth=1.2,
            zorder=2,
        )
    )
    for x, y, s in [(79.2, 29.7, 1.7), (82.6, 29.0, 1.5), (85.7, 29.8, 1.8)]:
        ax.add_patch(
            Rectangle(
                pt(x, y),
                s,
                s,
                facecolor="#B6C3CF",
                edgecolor="#7A8794",
                linewidth=1.0,
                zorder=3,
            )
        )

    ax.add_patch(Circle(pt(89.6, 28.9), 1.25, facecolor=ROBOT_DARK, edgecolor="#535C66", linewidth=1.0, zorder=4))
    joints = [pt(90.0, 30.8), pt(88.8, 34.0), pt(87.5, 37.0), pt(89.3, 39.7)]
    base = pt(89.6, 28.9)
    segments = [(base, joints[0]), (joints[0], joints[1]), (joints[1], joints[2]), (joints[2], joints[3])]
    for (x1, y1), (x2, y2) in segments:
        ax.add_line(Line2D([x1, x2], [y1, y2], color=ROBOT_GREY, linewidth=5.0, zorder=4, solid_capstyle="round"))
        ax.add_line(Line2D([x1, x2], [y1, y2], color=ROBOT_DARK, linewidth=1.2, zorder=4.1, solid_capstyle="round"))
    for x, y in joints:
        ax.add_patch(Circle((x, y), 0.95, facecolor="#D7DDE4", edgecolor=ROBOT_DARK, linewidth=1.0, zorder=5))
    x1, y1 = pt(89.3, 39.7)
    x2, y2 = pt(90.8, 41.7)
    ax.add_line(Line2D([x1, x2], [y1, y2], color=ROBOT_GREY, linewidth=4.2, zorder=4, solid_capstyle="round"))
    ax.add_line(Line2D([x2, x_shift + 91.4], [y2, y_shift + 42.2], color=ROBOT_DARK, linewidth=2.0, zorder=5))
    ax.add_line(Line2D([x2, x_shift + 91.55], [y2, y_shift + 41.2], color=ROBOT_DARK, linewidth=2.0, zorder=5))


def draw_cycle(ax: plt.Axes, *, x_shift: float = 0.0, y_shift: float = 0.0) -> None:
    draw_poly_arrow(
        ax,
        [
            (83.4 + x_shift, 6.5 + y_shift),
            (81.4 + x_shift, 9.0 + y_shift),
            (81.0 + x_shift, 12.0 + y_shift),
            (84.5 + x_shift, 13.2 + y_shift),
        ],
        color="#2E7580",
        lw=2.1,
        head_scale=18.0,
        z=4,
    )
    draw_poly_arrow(
        ax,
        [
            (87.8 + x_shift, 13.0 + y_shift),
            (90.7 + x_shift, 11.0 + y_shift),
            (90.8 + x_shift, 7.4 + y_shift),
            (87.3 + x_shift, 6.0 + y_shift),
        ],
        color="#2E7580",
        lw=2.1,
        head_scale=18.0,
        z=4,
    )


def build_figure() -> plt.Figure:
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    fig, ax = plt.subplots(figsize=(11.1, 10.0))
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")
    ax.axis("off")

    # Outer system panels.
    add_round_box(ax, 2.2, 41.8, 81.8, 47.4, fc=TOP_BG, ec=DARK_GREY, lw=1.8, rounding=3.2, dash=(1.6, 1.6), z=0)
    add_round_box(ax, 2.2, 2.0, 96.0, 38.2, fc=BOTTOM_BG, ec=DARK_GREY, lw=1.8, rounding=3.2, dash=(1.6, 1.6), z=0)

    # Section titles.
    add_text(ax, 4.6, 85.0, "System 1: High Level Planner", fs=HEADER_FS, weight="semibold", ha="left")
    add_text(ax, 6.8, 5.8, "System 2: Frozen WAM Executor", fs=PANEL_FS, weight="semibold", ha="left")

    # Top planner blocks.
    add_round_box(ax, 8.6, 68.6, 25.7, 12.6, fc=TOP_BOX, ec=TOP_BOX_EDGE, rounding=1.4, z=2)
    add_round_box(ax, 52.6, 68.6, 19.3, 12.6, fc=TOP_BOX, ec=TOP_BOX_EDGE, rounding=1.4, z=2)
    add_text(ax, 21.45, 74.8, "LLM Predictor", fs=BOX_FS + 1, weight="medium", color="white")
    add_text(ax, 62.25, 74.8, "VAL", fs=BOX_FS + 3, weight="medium", color="white")

    draw_poly_arrow(ax, [(34.3, 74.9), (52.6, 74.9)], color=ACCENT, lw=1.9, head_scale=22.0, z=4)
    add_text(ax, 43.3, 77.3, r"$q_1, q_2, \ldots, q_k$", fs=SMALL_FS + 2, weight="medium")
    add_text(ax, 43.3, 71.7, r"$g=(V, E)$", fs=SMALL_FS + 2, weight="medium")

    # Repair operator module.
    add_round_box(ax, 5.3, 46.6, 46.7, 16.2, fc="#F7FBFF", ec="#54789B", lw=1.6, rounding=1.4, z=1)
    add_text(ax, 28.6, 60.0, "Repair Operator", fs=BOX_FS + 2, weight="medium")
    op_specs = [
        (6.4, "INSERT"),
        (17.4, "DELETE"),
        (28.7, "REORDER"),
        (39.8, "MODIFY"),
    ]
    for x, label in op_specs:
        add_round_box(ax, x, 48.0, 10.0, 7.5, fc=OP_BOX, ec=OP_BOX_EDGE, lw=1.4, rounding=1.0, z=2)
        add_text(ax, x + 5.0, 51.75, label, fs=SMALL_FS + 1, weight="medium")

    # Planner feedback arrows and annotations.
    draw_poly_arrow(ax, [(57.4, 68.6), (57.4, 55.6), (52.0, 55.6)], color=ACCENT, lw=2.0, head_scale=22.0, z=4)
    draw_poly_arrow(ax, [(60.7, 68.6), (60.7, 37.6)], color=ACCENT, lw=2.0, head_scale=22.0, z=4)
    draw_poly_arrow(ax, [(17.8, 62.8), (17.8, 68.0)], color=ACCENT, lw=2.0, head_scale=22.0, z=4)
    add_text(ax, 45.6, 64.8, "Error?", fs=BOX_FS + 3, weight="medium")
    add_text(ax, 69.0, 64.8, "Correct?", fs=BOX_FS + 3, weight="medium")

    # Correct plan list.
    list_boxes = [
        (61.8, 57.5, r"$L_1, W_1$"),
        (61.8, 51.0, r"$L_2, W_2$"),
        (61.8, 41.7, r"$L_k, W_k$"),
    ]
    for x, y, label in list_boxes:
        add_round_box(ax, x, y, 10.6, 4.3, fc="white", ec="#69788A", lw=1.3, rounding=0.8, z=2)
        add_text(ax, x + 5.3, y + 2.15, label, fs=BOX_FS + 1, weight="medium")
    add_text(ax, 67.0, 47.2, r"$\vdots$", fs=SMALL_FS + 7)

    # Bottom executor inputs.
    add_round_box(ax, 8.6, 29.3, 19.2, 8.8, fc=MINT_BOX, ec=MINT_BOX_EDGE, lw=1.4, rounding=1.0, z=2)
    add_round_box(ax, 34.0, 29.3, 9.6, 8.8, fc=MINT_BOX, ec=MINT_BOX_EDGE, lw=1.4, rounding=1.0, z=2)
    add_round_box(ax, 50.5, 29.3, 21.0, 8.8, fc=MINT_BOX_STRONG, ec=MINT_BOX_EDGE, lw=1.4, rounding=1.0, z=2)
    add_text(ax, 18.2, 33.7, "Observation", fs=BOX_FS + 2, weight="medium")
    add_text(ax, 38.8, 33.7, "State", fs=BOX_FS + 1, weight="medium")
    add_text(ax, 61.0, 33.7, "Atom\nInstructions", fs=BOX_FS + 1, weight="medium")

    draw_poly_arrow(ax, [(18.2, 29.3), (18.2, 27.2)], color="#347E86", lw=2.0, head_scale=20.0, z=4)
    draw_poly_arrow(ax, [(38.8, 29.3), (38.8, 27.2)], color="#347E86", lw=2.0, head_scale=20.0, z=4)
    draw_poly_arrow(ax, [(61.0, 37.6), (61.0, 29.3)], color="#347E86", lw=2.0, head_scale=20.0, z=4)

    # Main executor box.
    add_round_box(ax, 4.3, 12.8, 68.7, 14.8, fc="#BFE9E3", ec=MINT_BOX_EDGE, lw=1.6, rounding=1.2, z=1)
    ax.add_line(Line2D([42.4, 42.4], [14.0, 26.4], color="#2D8C8E", linewidth=1.7, zorder=3))
    add_text(ax, 24.0, 21.5, "Frozen Policy", fs=BOX_FS + 3, weight="medium")
    add_text(ax, 24.0, 17.1, r"$\mathcal{F}_{\theta}$", fs=MATH_FS + 6, weight="medium")
    add_text(ax, 57.5, 20.6, "World Modeling &\nAction Prediction", fs=BOX_FS + 2, weight="medium")

    draw_poly_arrow(ax, [(18.2, 27.2), (18.2, 23.9)], color="#347E86", lw=2.0, head_scale=20.0, z=4)
    draw_poly_arrow(ax, [(38.8, 27.2), (38.8, 23.9)], color="#347E86", lw=2.0, head_scale=20.0, z=4)
    draw_poly_arrow(ax, [(61.0, 29.3), (61.0, 23.9)], color="#347E86", lw=2.0, head_scale=20.0, z=4)

    # Real observation and environment.
    add_round_box(ax, 78.8, 12.9, 14.0, 14.2, fc="#F6FEFE", ec=MINT_BOX_EDGE, lw=1.4, rounding=1.0, z=2)
    add_text(ax, 85.8, 20.0, "Real\nObservation", fs=BOX_FS, weight="medium", linespacing=0.92)
    draw_poly_arrow(ax, [(73.0, 20.2), (78.8, 20.2)], color="#347E86", lw=2.0, head_scale=22.0, z=4)

    draw_robot_icon(ax, x_shift=2.1, y_shift=0.8)
    add_text(ax, 66.2, 7.7, "Cache-Preserving\nTransition", fs=BOX_FS + 1, weight="medium", ha="center", linespacing=0.92)
    draw_cycle(ax, x_shift=1.0, y_shift=-1.0)

    return fig


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig = build_figure()
    fig.savefig(PDF_PATH, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(SVG_PATH, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(PNG_PATH, dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


if __name__ == "__main__":
    main()
