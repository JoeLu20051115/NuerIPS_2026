#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import fisher_exact, norm


REPO_ROOT = Path(__file__).resolve().parents[2]
DROID_SELECTED150_PATH = REPO_ROOT / "evaluation_results_dualsystem" / "droid_3way_selected150.json"

SPLIT_ORDER = ["L1", "L2", "L3"]
MODE_ORDER = ["original", "llm_dual", "llm_val"]
MODE_COLORS = {
    "original": "#6B2D5C",
    "llm_dual": "#2A6F97",
    "llm_val": "#3A7D44",
}
DATASET_COLORS = {
    "DROID": "#22577A",
    "AgiBot": "#C44536",
    "Pooled": "#1F4E79",
}


TABLE_ROWS = [
    # Dreamzero-DROID (verbatim from the user-verified appendix table).
    {"dataset": "DROID", "split": "L1", "mode": "original", "n": 50, "mean_l2": 0.1291, "task_progress": 0.492, "success_rate": 0.30, "l2_lt_0_1": 0.487},
    {"dataset": "DROID", "split": "L1", "mode": "llm_dual", "n": 50, "mean_l2": 0.1095, "task_progress": 0.520, "success_rate": 0.44, "l2_lt_0_1": 0.613},
    {"dataset": "DROID", "split": "L1", "mode": "llm_val", "n": 50, "mean_l2": 0.1075, "task_progress": 0.574, "success_rate": 0.50, "l2_lt_0_1": 0.600},
    {"dataset": "DROID", "split": "L2", "mode": "original", "n": 50, "mean_l2": 0.1255, "task_progress": 0.420, "success_rate": 0.18, "l2_lt_0_1": 0.520},
    {"dataset": "DROID", "split": "L2", "mode": "llm_dual", "n": 50, "mean_l2": 0.1089, "task_progress": 0.480, "success_rate": 0.32, "l2_lt_0_1": 0.587},
    {"dataset": "DROID", "split": "L2", "mode": "llm_val", "n": 50, "mean_l2": 0.1032, "task_progress": 0.524, "success_rate": 0.38, "l2_lt_0_1": 0.627},
    {"dataset": "DROID", "split": "L3", "mode": "original", "n": 50, "mean_l2": 0.1389, "task_progress": 0.434, "success_rate": 0.22, "l2_lt_0_1": 0.480},
    {"dataset": "DROID", "split": "L3", "mode": "llm_dual", "n": 50, "mean_l2": 0.1125, "task_progress": 0.464, "success_rate": 0.34, "l2_lt_0_1": 0.520},
    {"dataset": "DROID", "split": "L3", "mode": "llm_val", "n": 50, "mean_l2": 0.1071, "task_progress": 0.538, "success_rate": 0.48, "l2_lt_0_1": 0.580},
    # Dreamzero-AgiBot (verbatim from the user-verified appendix table).
    {"dataset": "AgiBot", "split": "L1", "mode": "original", "n": 50, "mean_l2": 0.0845, "task_progress": 0.460, "success_rate": 0.30, "l2_lt_0_1": 0.720, "avg_duration_sec": 22.0},
    {"dataset": "AgiBot", "split": "L1", "mode": "llm_dual", "n": 50, "mean_l2": 0.0685, "task_progress": 0.574, "success_rate": 0.52, "l2_lt_0_1": 0.840, "avg_duration_sec": 22.0},
    {"dataset": "AgiBot", "split": "L1", "mode": "llm_val", "n": 50, "mean_l2": 0.0672, "task_progress": 0.624, "success_rate": 0.62, "l2_lt_0_1": 0.893, "avg_duration_sec": 22.0},
    {"dataset": "AgiBot", "split": "L2", "mode": "original", "n": 50, "mean_l2": 0.0942, "task_progress": 0.416, "success_rate": 0.22, "l2_lt_0_1": 0.520, "avg_duration_sec": 42.0},
    {"dataset": "AgiBot", "split": "L2", "mode": "llm_dual", "n": 50, "mean_l2": 0.0745, "task_progress": 0.502, "success_rate": 0.42, "l2_lt_0_1": 0.747, "avg_duration_sec": 42.0},
    {"dataset": "AgiBot", "split": "L2", "mode": "llm_val", "n": 50, "mean_l2": 0.0704, "task_progress": 0.542, "success_rate": 0.48, "l2_lt_0_1": 0.800, "avg_duration_sec": 42.0},
    {"dataset": "AgiBot", "split": "L3", "mode": "original", "n": 50, "mean_l2": 0.0748, "task_progress": 0.406, "success_rate": 0.16, "l2_lt_0_1": 0.773, "avg_duration_sec": 126.0},
    {"dataset": "AgiBot", "split": "L3", "mode": "llm_dual", "n": 50, "mean_l2": 0.0614, "task_progress": 0.536, "success_rate": 0.48, "l2_lt_0_1": 0.907, "avg_duration_sec": 126.0},
    {"dataset": "AgiBot", "split": "L3", "mode": "llm_val", "n": 50, "mean_l2": 0.0593, "task_progress": 0.584, "success_rate": 0.58, "l2_lt_0_1": 0.967, "avg_duration_sec": 126.0},
]


@dataclass
class FitResult:
    terms: list[str]
    beta: np.ndarray
    se: np.ndarray
    p_value: np.ndarray
    odds_ratio: np.ndarray
    cov: np.ndarray


def bootstrap_rate_ci(success_count: int, n: int, n_boot: int, seed: int) -> tuple[float, float]:
    if n <= 0:
        return np.nan, np.nan
    values = np.array([1.0] * success_count + [0.0] * (n - success_count), dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(n_boot, values.size))
    means = values[idx].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(lo), float(hi)


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    if not p_values:
        return []
    arr = np.asarray(p_values, dtype=float)
    order = np.argsort(arr)
    ranked = np.empty_like(arr)
    prev = 1.0
    m = len(arr)
    for rank, idx in reversed(list(enumerate(order, start=1))):
        adj = arr[idx] * m / rank
        prev = min(prev, adj)
        ranked[idx] = min(prev, 1.0)
    return ranked.tolist()


def fit_logistic_regression(X: np.ndarray, y: np.ndarray, terms: list[str]) -> FitResult:
    def nll(beta: np.ndarray) -> float:
        z = X @ beta
        return float(np.sum(np.logaddexp(0.0, z) - y * z))

    def grad(beta: np.ndarray) -> np.ndarray:
        z = X @ beta
        p = 1.0 / (1.0 + np.exp(-z))
        return X.T @ (p - y)

    def hess(beta: np.ndarray) -> np.ndarray:
        z = X @ beta
        p = 1.0 / (1.0 + np.exp(-z))
        w = p * (1.0 - p)
        return X.T @ (X * w[:, None])

    res = minimize(nll, np.zeros(X.shape[1], dtype=float), jac=grad, method="BFGS")
    beta = res.x
    cov = np.linalg.pinv(hess(beta))
    se = np.sqrt(np.diag(cov))
    z_scores = np.divide(beta, se, out=np.zeros_like(beta), where=se > 0)
    p_values = 2.0 * (1.0 - norm.cdf(np.abs(z_scores)))
    odds_ratio = np.exp(beta)
    return FitResult(
        terms=terms,
        beta=beta,
        se=se,
        p_value=p_values,
        odds_ratio=odds_ratio,
        cov=cov,
    )


def load_droid_split_mean_durations() -> dict[str, float]:
    import json

    with DROID_SELECTED150_PATH.open() as f:
        payload = json.load(f)
    episodes: dict[str, tuple[str, float]] = {}
    for row in payload["results"]:
        eid = str(row["episode_id"])
        split = row.get("dro_split")
        ep_len = float(row.get("ep_len", 0.0))
        if eid not in episodes:
            episodes[eid] = (split, ep_len)

    out = {}
    for split in SPLIT_ORDER:
        vals = [ep_len / 15.0 for s, ep_len in episodes.values() if s == split]
        out[split] = float(np.mean(vals))
    return out


def linear_combo_from_fit(fit: FitResult, weights: np.ndarray, label: str) -> dict:
    estimate = float(weights @ fit.beta)
    variance = float(weights @ fit.cov @ weights)
    se = float(np.sqrt(max(variance, 0.0)))
    z_value = estimate / se if se > 0 else 0.0
    p_value = float(2.0 * (1.0 - norm.cdf(abs(z_value))))
    return {
        "term": label,
        "beta": estimate,
        "se": se,
        "z": z_value,
        "p_value": p_value,
        "odds_ratio": float(np.exp(estimate)),
    }


def build_input_table(n_boot: int, seed: int) -> pd.DataFrame:
    rows = []
    for idx, row in enumerate(TABLE_ROWS):
        success_count = int(round(row["n"] * row["success_rate"]))
        ci_lo, ci_hi = bootstrap_rate_ci(success_count, int(row["n"]), n_boot=n_boot, seed=seed + idx)
        rows.append(
            {
                **row,
                "success_count": success_count,
                "fail_count": int(row["n"]) - success_count,
                "success_rate_ci_lo": ci_lo,
                "success_rate_ci_hi": ci_hi,
                "split_order": SPLIT_ORDER.index(row["split"]),
            }
        )
    df = pd.DataFrame(rows)
    df["split"] = pd.Categorical(df["split"], categories=SPLIT_ORDER, ordered=True)
    df["mode"] = pd.Categorical(df["mode"], categories=MODE_ORDER, ordered=True)
    return df.sort_values(["dataset", "mode", "split"]).reset_index(drop=True)


def build_pooled_table(df: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    rows = []
    grouped = df.groupby(["split", "mode"], observed=False, sort=False)
    for idx, ((split, mode), group) in enumerate(grouped):
        n = int(group["n"].sum())
        success_count = int(group["success_count"].sum())
        ci_lo, ci_hi = bootstrap_rate_ci(success_count, n, n_boot=n_boot, seed=seed + 100 + idx)
        rows.append(
            {
                "dataset": "Pooled",
                "split": split,
                "mode": mode,
                "n": n,
                "success_count": success_count,
                "fail_count": n - success_count,
                "success_rate": success_count / n,
                "success_rate_ci_lo": ci_lo,
                "success_rate_ci_hi": ci_hi,
                "mean_l2": float(np.average(group["mean_l2"], weights=group["n"])),
                "task_progress": float(np.average(group["task_progress"], weights=group["n"])),
                "l2_lt_0_1": float(np.average(group["l2_lt_0_1"], weights=group["n"])),
                "split_order": SPLIT_ORDER.index(split),
            }
        )
    out = pd.DataFrame(rows)
    out["split"] = pd.Categorical(out["split"], categories=SPLIT_ORDER, ordered=True)
    out["mode"] = pd.Categorical(out["mode"], categories=MODE_ORDER, ordered=True)
    return out.sort_values(["mode", "split"]).reset_index(drop=True)


def build_pairwise_tests(df: pd.DataFrame, dataset_col: str = "dataset") -> pd.DataFrame:
    tests = []
    grouped = df.groupby([dataset_col, "mode"], observed=False, sort=False)
    for (dataset, mode), group in grouped:
        group = group.sort_values("split")
        for left, right in zip(SPLIT_ORDER, SPLIT_ORDER[1:]):
            left_row = group[group["split"] == left]
            right_row = group[group["split"] == right]
            if left_row.empty or right_row.empty:
                continue
            left_row = left_row.iloc[0]
            right_row = right_row.iloc[0]
            odds_ratio, p_value = fisher_exact(
                [
                    [int(left_row["success_count"]), int(left_row["fail_count"])],
                    [int(right_row["success_count"]), int(right_row["fail_count"])],
                ]
            )
            tests.append(
                {
                    dataset_col: dataset,
                    "mode": mode,
                    "split_left": left,
                    "split_right": right,
                    "n_left": int(left_row["n"]),
                    "n_right": int(right_row["n"]),
                    "success_left": int(left_row["success_count"]),
                    "success_right": int(right_row["success_count"]),
                    "success_rate_left": float(left_row["success_rate"]),
                    "success_rate_right": float(right_row["success_rate"]),
                    "odds_ratio": float(odds_ratio),
                    "p_value": float(p_value),
                }
            )
    out = pd.DataFrame(tests)
    if out.empty:
        return out
    out["p_value_bh"] = benjamini_hochberg(out["p_value"].tolist())
    return out.sort_values([dataset_col, "mode", "split_left"]).reset_index(drop=True)


def build_trend_rows(df: pd.DataFrame, label: str, with_dataset: bool, with_interaction: bool) -> list[dict]:
    X = []
    y = []
    for _, row in df.iterrows():
        split_ord = float(row["split_order"])
        is_agibot = 1.0 if row["dataset"] == "AgiBot" else 0.0
        for idx in range(int(row["n"])):
            success = 1.0 if idx < int(row["success_count"]) else 0.0
            if with_dataset and with_interaction:
                X.append([1.0, split_ord, is_agibot, split_ord * is_agibot])
            elif with_dataset:
                X.append([1.0, split_ord, is_agibot])
            else:
                X.append([1.0, split_ord])
            y.append(success)

    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if with_dataset and with_interaction:
        terms = ["Intercept", "split_ord", "dataset_AgiBot", "split_ord:dataset_AgiBot"]
    elif with_dataset:
        terms = ["Intercept", "split_ord", "dataset_AgiBot"]
    else:
        terms = ["Intercept", "split_ord"]

    fit = fit_logistic_regression(X_arr, y_arr, terms=terms)
    rows = [
        {
            "model": label,
            "term": term,
            "beta": float(beta),
            "se": float(se),
            "p_value": float(p_value),
            "odds_ratio": float(odds_ratio),
        }
        for term, beta, se, p_value, odds_ratio in zip(
            fit.terms,
            fit.beta,
            fit.se,
            fit.p_value,
            fit.odds_ratio,
        )
    ]
    if with_dataset and with_interaction:
        rows.append(
            linear_combo_from_fit(
                fit,
                weights=np.array([0.0, 1.0, 0.0, 0.0]),
                label="split_ord | DROID slope",
            )
            | {"model": label}
        )
        rows.append(
            linear_combo_from_fit(
                fit,
                weights=np.array([0.0, 1.0, 0.0, 1.0]),
                label="split_ord | AgiBot slope",
            )
            | {"model": label}
        )
    return rows


def plot_dataset_sr(df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharey=True)
    for ax, dataset in zip(axes, ["DROID", "AgiBot"]):
        sub = df[df["dataset"] == dataset]
        xs = np.arange(len(SPLIT_ORDER))
        for mode in MODE_ORDER:
            mode_rows = sub[sub["mode"] == mode].sort_values("split")
            ys = mode_rows["success_rate"].to_numpy()
            err_lo = ys - mode_rows["success_rate_ci_lo"].to_numpy()
            err_hi = mode_rows["success_rate_ci_hi"].to_numpy() - ys
            ax.errorbar(
                xs,
                ys,
                yerr=np.vstack([err_lo, err_hi]),
                fmt="-o",
                capsize=4,
                linewidth=2.0,
                markersize=7,
                color=MODE_COLORS[mode],
                label=mode,
            )
        ax.set_xticks(xs)
        ax.set_xticklabels(SPLIT_ORDER)
        ax.set_ylim(0.0, 1.02)
        ax.grid(alpha=0.25, linestyle="--")
        ax.set_title(dataset)
        ax.set_xlabel("Table Split")
    axes[0].set_ylabel("Success Rate")
    axes[1].legend(frameon=False, loc="upper right")
    fig.suptitle("DreamZero Appendix Table Splits: SR by Dataset and Mode")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_pooled_sr(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    xs = np.arange(len(SPLIT_ORDER))
    for mode in MODE_ORDER:
        mode_rows = df[df["mode"] == mode].sort_values("split")
        ys = mode_rows["success_rate"].to_numpy()
        err_lo = ys - mode_rows["success_rate_ci_lo"].to_numpy()
        err_hi = mode_rows["success_rate_ci_hi"].to_numpy() - ys
        ax.errorbar(
            xs,
            ys,
            yerr=np.vstack([err_lo, err_hi]),
            fmt="-o",
            capsize=4,
            linewidth=2.0,
            markersize=7,
            color=MODE_COLORS[mode],
            label=mode,
        )
    ax.set_xticks(xs)
    ax.set_xticklabels(SPLIT_ORDER)
    ax.set_ylim(0.0, 1.02)
    ax.grid(alpha=0.25, linestyle="--")
    ax.set_xlabel("Table Split")
    ax.set_ylabel("Success Rate")
    ax.set_title("DreamZero Appendix Table Splits: Pooled SR by Mode")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_pooled_metrics(df: pd.DataFrame, out_path: Path) -> None:
    metrics = [("task_progress", "Mean Task Progress"), ("mean_l2", "Mean L2")]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    xs = np.arange(len(SPLIT_ORDER))
    for ax, (metric, title) in zip(axes, metrics):
        for mode in MODE_ORDER:
            mode_rows = df[df["mode"] == mode].sort_values("split")
            ax.plot(
                xs,
                mode_rows[metric].to_numpy(),
                "-o",
                linewidth=2.0,
                markersize=7,
                color=MODE_COLORS[mode],
                label=mode,
            )
        ax.set_xticks(xs)
        ax.set_xticklabels(SPLIT_ORDER)
        ax.grid(alpha=0.25, linestyle="--")
        ax.set_title(title)
        ax.set_xlabel("Table Split")
        if metric == "task_progress":
            ax.set_ylim(0.0, 1.02)
    axes[0].legend(frameon=False, loc="upper right")
    fig.suptitle("DreamZero Appendix Table Splits: Pooled Metrics by Mode")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_original_story_figure(
    table_df: pd.DataFrame,
    pooled_df: pd.DataFrame,
    trend_df: pd.DataFrame,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))
    x = np.arange(len(SPLIT_ORDER))

    datasets = [
        ("DROID", table_df[(table_df["dataset"] == "DROID") & (table_df["mode"] == "original")].sort_values("split")),
        ("AgiBot", table_df[(table_df["dataset"] == "AgiBot") & (table_df["mode"] == "original")].sort_values("split")),
        ("Pooled", pooled_df[pooled_df["mode"] == "original"].sort_values("split")),
    ]

    ax = axes[0]
    for name, frame in datasets:
        ys = frame["success_rate"].to_numpy()
        err_lo = ys - frame["success_rate_ci_lo"].to_numpy()
        err_hi = frame["success_rate_ci_hi"].to_numpy() - ys
        ax.errorbar(
            x,
            ys,
            yerr=np.vstack([err_lo, err_hi]),
            fmt="-o",
            capsize=4,
            linewidth=2.2,
            markersize=7,
            color=DATASET_COLORS[name],
            label=name,
        )
        for xi, yi in zip(x, ys):
            ax.text(xi, yi + 0.035, f"{yi * 100:.0f}%", ha="center", va="bottom", fontsize=9, color=DATASET_COLORS[name])
    ax.set_xticks(x)
    ax.set_xticklabels(SPLIT_ORDER)
    ax.set_ylim(0.0, 0.7)
    ax.set_ylabel("Success Rate")
    ax.set_xlabel("Appendix Split")
    ax.set_title("Original Success Rate Trend")
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(frameon=False, loc="upper right")

    trend_lookup = {
        (row["mode"], row["model"], row["term"]): row
        for _, row in trend_df.iterrows()
    }
    story_text = "\n".join(
        [
            f"DROID: OR={trend_lookup[('original', 'DROID_only', 'split_ord')]['odds_ratio']:.3f}, p={trend_lookup[('original', 'DROID_only', 'split_ord')]['p_value']:.3f}",
            f"AgiBot: OR={trend_lookup[('original', 'AgiBot_only', 'split_ord')]['odds_ratio']:.3f}, p={trend_lookup[('original', 'AgiBot_only', 'split_ord')]['p_value']:.3f}",
            f"Pooled: OR={trend_lookup[('original', 'pooled_no_interaction', 'split_ord')]['odds_ratio']:.3f}, p={trend_lookup[('original', 'pooled_no_interaction', 'split_ord')]['p_value']:.3f}",
        ]
    )
    ax.text(
        0.03,
        0.03,
        story_text,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.92, "edgecolor": "#cccccc"},
    )

    ax2 = axes[1]
    for name, frame in datasets:
        ys = frame["task_progress"].to_numpy()
        ax2.plot(
            x,
            ys,
            "-o",
            linewidth=2.2,
            markersize=7,
            color=DATASET_COLORS[name],
            label=name,
        )
        for xi, yi in zip(x, ys):
            ax2.text(xi, yi + 0.022, f"{yi:.3f}", ha="center", va="bottom", fontsize=9, color=DATASET_COLORS[name])
    ax2.set_xticks(x)
    ax2.set_xticklabels(SPLIT_ORDER)
    ax2.set_ylim(0.30, 0.56)
    ax2.set_ylabel("Task Progress")
    ax2.set_xlabel("Appendix Split")
    ax2.set_title("Original Task-Progress Trend")
    ax2.grid(alpha=0.25, linestyle="--")

    fig.suptitle("Original Mode Degrades on Longer/Harder Appendix Splits", fontsize=14)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=240, bbox_inches="tight")
    if out_path.suffix.lower() == ".png":
        fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def write_summary(
    out_path: Path,
    table_df: pd.DataFrame,
    pooled_df: pd.DataFrame,
    tests_df: pd.DataFrame,
    pooled_tests_df: pd.DataFrame,
    trend_df: pd.DataFrame,
) -> None:
    lines = [
        "# DreamZero Appendix-Table Split Analysis",
        "",
        "## Source Convention",
        "- Inputs follow the user-verified appendix table values verbatim.",
        "- DROID is analyzed with the archived table splits `L1/L2/L3`.",
        "- AgiBot is analyzed with the appendix levels `L1/L2/L3`, whose labels carry approximate average durations `22s / 42s / 126s` in the table.",
        "- Pooling is done only after computing split-wise statistics separately for each dataset.",
        "",
        "## Separate Results",
    ]

    for dataset in ["DROID", "AgiBot"]:
        lines.append(f"### {dataset}")
        sub = table_df[table_df["dataset"] == dataset].sort_values(["mode", "split"])
        for mode in MODE_ORDER:
            mode_rows = sub[sub["mode"] == mode].sort_values("split")
            parts = []
            for _, row in mode_rows.iterrows():
                parts.append(
                    f"{row['split']}: SR={row['success_rate']:.3f}, "
                    f"95% CI=[{row['success_rate_ci_lo']:.3f}, {row['success_rate_ci_hi']:.3f}], "
                    f"progress={row['task_progress']:.3f}, L2={row['mean_l2']:.4f}"
                )
            lines.append(f"- {mode}: " + " | ".join(parts))
        lines.append("")

    lines.extend(["## Pooled Results"])
    for mode in MODE_ORDER:
        mode_rows = pooled_df[pooled_df["mode"] == mode].sort_values("split")
        parts = []
        for _, row in mode_rows.iterrows():
            parts.append(
                f"{row['split']}: SR={row['success_rate']:.3f}, "
                f"95% CI=[{row['success_rate_ci_lo']:.3f}, {row['success_rate_ci_hi']:.3f}], "
                f"progress={row['task_progress']:.3f}, L2={row['mean_l2']:.4f}"
            )
        lines.append(f"- {mode}: " + " | ".join(parts))
    lines.append("")

    lines.extend(["## Adjacent-Split Fisher Tests"])
    if tests_df.empty:
        lines.append("- No per-dataset tests were available.")
    else:
        for _, row in tests_df.iterrows():
            lines.append(
                f"- {row['dataset']} {row['mode']} {row['split_left']} vs {row['split_right']}: "
                f"p={row['p_value']:.4g}, BH p={row['p_value_bh']:.4g}, "
                f"SR {row['success_rate_left']:.3f} -> {row['success_rate_right']:.3f}"
            )
    lines.append("")
    if pooled_tests_df.empty:
        lines.append("- No pooled tests were available.")
    else:
        for _, row in pooled_tests_df.iterrows():
            lines.append(
                f"- pooled {row['mode']} {row['split_left']} vs {row['split_right']}: "
                f"p={row['p_value']:.4g}, BH p={row['p_value_bh']:.4g}, "
                f"SR {row['success_rate_left']:.3f} -> {row['success_rate_right']:.3f}"
            )
    lines.append("")

    lines.extend(["## Ordered-Split Logistic Trend"])
    for mode in MODE_ORDER:
        lines.append(f"### {mode}")
        sub = trend_df[trend_df["mode"] == mode]
        for _, row in sub.iterrows():
            lines.append(
                f"- {row['model']} | {row['term']}: beta={row['beta']:.4f}, "
                f"OR={row['odds_ratio']:.3f}, p={row['p_value']:.4g}"
            )
        lines.append("")

    lines.extend(
        [
            "## Readout",
            "- On the table-aligned bins, the `original` pooled SR drops from `30%` at `L1` to `20%` at `L2` and `19%` at `L3`, so the visual trend is cleaner than the earlier absolute-seconds analysis.",
            "- AgiBot alone also shows a clean `30% -> 22% -> 16%` drop for `original`.",
            "- DROID alone is weaker: `30% -> 18% -> 22%`, so it still does not give a perfectly monotone decline.",
            "- The stronger modes do not show the same clean deterioration pattern under pooling, which is consistent with the story that the original method is the one that is most sensitive to longer/harder splits.",
            "- Statistically, the split-aligned evidence is still suggestive rather than definitive: adjacent Fisher tests are not significant after correction, and the pooled `original` ordered-split trend is only marginal (`p` around the `0.05-0.10` band).",
            "",
        ]
    )
    out_path.write_text("\n".join(lines) + "\n")


def build_original_trend_presentation(
    table_df: pd.DataFrame,
    pooled_df: pd.DataFrame,
    trend_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    orig_sep = table_df[table_df["mode"] == "original"].copy()
    orig_pooled = pooled_df[pooled_df["mode"] == "original"].copy()

    main_rows = []
    progress_rows = []
    config = [
        ("DROID", orig_sep[orig_sep["dataset"] == "DROID"].copy(), "DROID_only"),
        ("AgiBot", orig_sep[orig_sep["dataset"] == "AgiBot"].copy(), "AgiBot_only"),
        ("Pooled", orig_pooled.copy(), "pooled_no_interaction"),
    ]
    for dataset, frame, model_name in config:
        frame = frame.sort_values("split")
        sr_vals = frame["success_rate"].to_numpy()
        prog_vals = frame["task_progress"].to_numpy()
        trend_row = trend_df[
            (trend_df["mode"] == "original")
            & (trend_df["model"] == model_name)
            & (trend_df["term"] == "split_ord")
        ].iloc[0]
        main_rows.append(
            {
                "dataset": dataset,
                "L1_SR": float(sr_vals[0]),
                "L2_SR": float(sr_vals[1]),
                "L3_SR": float(sr_vals[2]),
                "delta_L1_to_L3_pp": float((sr_vals[2] - sr_vals[0]) * 100.0),
                "monotone_decline": bool((sr_vals[0] >= sr_vals[1]) and (sr_vals[1] >= sr_vals[2])),
                "ordered_split_or": float(trend_row["odds_ratio"]),
                "ordered_split_p": float(trend_row["p_value"]),
                "readout": (
                    "supports claim"
                    if bool((sr_vals[0] >= sr_vals[1]) and (sr_vals[1] >= sr_vals[2]))
                    else "mixed trend"
                ),
            }
        )
        progress_rows.append(
            {
                "dataset": dataset,
                "L1_progress": float(prog_vals[0]),
                "L2_progress": float(prog_vals[1]),
                "L3_progress": float(prog_vals[2]),
                "delta_L1_to_L3": float(prog_vals[2] - prog_vals[0]),
            }
        )

    main_df = pd.DataFrame(main_rows)
    progress_df = pd.DataFrame(progress_rows)
    return main_df, progress_df


def write_original_trend_presentation(
    out_path: Path,
    latex_path: Path,
    main_df: pd.DataFrame,
    progress_df: pd.DataFrame,
) -> None:
    fmt_pct = lambda x: f"{x * 100:.0f}%"
    fmt_pp = lambda x: f"{x:+.0f} pp"
    fmt_p = lambda x: f"{x:.3f}"
    fmt_or = lambda x: f"{x:.3f}"
    fmt_prog = lambda x: f"{x:.3f}"

    lines = [
        "# Original-Mode Trend Table",
        "",
        "## Main Trend Table",
        "",
        "| Dataset | L1 SR | L2 SR | L3 SR | Delta (L1->L3) | Monotone Decline | OR / split | p-value | Readout |",
        "| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- |",
    ]
    for _, row in main_df.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["dataset"]),
                    fmt_pct(row["L1_SR"]),
                    fmt_pct(row["L2_SR"]),
                    fmt_pct(row["L3_SR"]),
                    fmt_pp(row["delta_L1_to_L3_pp"]),
                    "Yes" if bool(row["monotone_decline"]) else "No",
                    fmt_or(row["ordered_split_or"]),
                    fmt_p(row["ordered_split_p"]),
                    str(row["readout"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Supporting Task-Progress Table",
            "",
            "| Dataset | L1 Progress | L2 Progress | L3 Progress | Delta (L1->L3) |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for _, row in progress_df.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["dataset"]),
                    fmt_prog(row["L1_progress"]),
                    fmt_prog(row["L2_progress"]),
                    fmt_prog(row["L3_progress"]),
                    f"{row['delta_L1_to_L3']:+.3f}",
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Suggested Claim",
            "",
            "- In the original setting, AgiBot and the pooled unseen-test-set analysis both show a cleaner decline from `L1` to `L3` than the earlier absolute-duration bucket analysis.",
            "- The pooled original trend is `30% -> 20% -> 19%`, with ordered-split `OR=0.731` and `p=0.066`, so it is best described as supportive or marginal rather than fully conclusive.",
            "- DROID alone trends in the same general direction from `L1` to `L2` but is not perfectly monotone because `L3` rebounds slightly.",
            "",
        ]
    )
    out_path.write_text("\n".join(lines) + "\n")

    latex_lines = [
        "\\begin{tabular}{lccccccc}",
        "\\hline",
        "Dataset & L1 SR & L2 SR & L3 SR & $\\Delta$(L1$\\to$L3) & Monotone & OR/split & $p$ \\\\",
        "\\hline",
    ]
    for _, row in main_df.iterrows():
        latex_lines.append(
            f"{row['dataset']} & "
            f"{fmt_pct(row['L1_SR'])} & "
            f"{fmt_pct(row['L2_SR'])} & "
            f"{fmt_pct(row['L3_SR'])} & "
            f"{fmt_pp(row['delta_L1_to_L3_pp'])} & "
            f"{'Yes' if bool(row['monotone_decline']) else 'No'} & "
            f"{fmt_or(row['ordered_split_or'])} & "
            f"{fmt_p(row['ordered_split_p'])} \\\\"
        )
    latex_lines.extend(
        [
            "\\hline",
            "\\end{tabular}",
            "",
            "% Readout: pooled original is 30% -> 20% -> 19%, OR=0.731, p=0.066 (marginal/supportive).",
        ]
    )
    latex_path.write_text("\n".join(latex_lines) + "\n")


def build_time_accuracy_points(table_df: pd.DataFrame, mode_filter: str | None = None) -> pd.DataFrame:
    droid_durations = load_droid_split_mean_durations()
    rows = []
    sub_df = table_df.copy()
    if mode_filter is not None:
        sub_df = sub_df[sub_df["mode"] == mode_filter].copy()
    for _, row in sub_df.iterrows():
        if row["dataset"] == "DROID":
            avg_duration_sec = droid_durations[str(row["split"])]
        else:
            avg_duration_sec = float(row.get("avg_duration_sec", np.nan))
        rows.append(
            {
                "dataset": row["dataset"],
                "split": str(row["split"]),
                "mode": str(row["mode"]),
                "point_label": f"{row['dataset']}-{row['split']}",
                "avg_duration_sec": avg_duration_sec,
                "success_rate": float(row["success_rate"]),
                "task_progress": float(row["task_progress"]),
                "mean_l2": float(row["mean_l2"]),
                "success_rate_ci_lo": float(row["success_rate_ci_lo"]),
                "success_rate_ci_hi": float(row["success_rate_ci_hi"]),
            }
        )
    out = pd.DataFrame(rows).sort_values(["dataset", "mode", "avg_duration_sec"]).reset_index(drop=True)
    return out


def write_time_accuracy_table(md_path: Path, tex_path: Path, csv_path: Path, points_df: pd.DataFrame) -> None:
    points_df.to_csv(csv_path, index=False)

    lines = [
        "# Time-Accuracy Table (Original Mode)",
        "",
        "| Point | Avg Duration (s) | Success Rate | 95% CI | Task Progress | Mean L2 |",
        "| --- | ---: | ---: | --- | ---: | ---: |",
    ]
    for _, row in points_df.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["point_label"]),
                    f"{row['avg_duration_sec']:.1f}",
                    f"{row['success_rate'] * 100:.0f}%",
                    f"[{row['success_rate_ci_lo'] * 100:.0f}%, {row['success_rate_ci_hi'] * 100:.0f}%]",
                    f"{row['task_progress']:.3f}",
                    f"{row['mean_l2']:.4f}",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "- The six points are sorted by average episode duration, so the table can be read directly as a short-to-long horizon trend.",
            "- The overall picture is downward: short DROID splits start around `30%`, while the longest AgiBot split ends at `16%`.",
            "",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n")

    tex_lines = [
        "\\begin{tabular}{lccccc}",
        "\\hline",
        "Point & Avg Duration (s) & Success Rate & 95\\% CI & Task Progress & Mean L2 \\\\",
        "\\hline",
    ]
    for _, row in points_df.iterrows():
        tex_lines.append(
            f"{row['point_label']} & "
            f"{row['avg_duration_sec']:.1f} & "
            f"{row['success_rate'] * 100:.0f}\\% & "
            f"[{row['success_rate_ci_lo'] * 100:.0f}\\%, {row['success_rate_ci_hi'] * 100:.0f}\\%] & "
            f"{row['task_progress']:.3f} & "
            f"{row['mean_l2']:.4f} \\\\"
        )
    tex_lines.extend(["\\hline", "\\end{tabular}"])
    tex_path.write_text("\n".join(tex_lines) + "\n")


def plot_time_accuracy_story(points_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 4.8))

    for dataset in ["DROID", "AgiBot"]:
        sub = points_df[points_df["dataset"] == dataset].sort_values("avg_duration_sec")
        xs = sub["avg_duration_sec"].to_numpy()
        ys = sub["success_rate"].to_numpy()
        err_lo = ys - sub["success_rate_ci_lo"].to_numpy()
        err_hi = sub["success_rate_ci_hi"].to_numpy() - ys
        ax.errorbar(
            xs,
            ys,
            yerr=np.vstack([err_lo, err_hi]),
            fmt="-o",
            capsize=4,
            linewidth=2.2,
            markersize=7,
            color=DATASET_COLORS[dataset],
            label=dataset,
        )
        for _, row in sub.iterrows():
            ax.annotate(
                row["split"],
                (row["avg_duration_sec"], row["success_rate"]),
                textcoords="offset points",
                xytext=(0, 7),
                ha="center",
                fontsize=9,
                color=DATASET_COLORS[dataset],
            )

    xs = points_df["avg_duration_sec"].to_numpy()
    ys = points_df["success_rate"].to_numpy()
    coef = np.polyfit(xs, ys, 1)
    x_fit = np.linspace(xs.min(), xs.max(), 200)
    y_fit = coef[0] * x_fit + coef[1]
    ax.plot(
        x_fit,
        y_fit,
        "--",
        linewidth=1.8,
        color="#444444",
        alpha=0.9,
        label="6-point fit",
    )

    ax.set_xlabel("Average Duration (s)")
    ax.set_ylabel("Success Rate")
    ax.set_ylim(0.0, 0.42)
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(frameon=False, loc="upper right")
    ax.set_title("Original Mode: Longer-Horizon Splits Tend to Have Lower Success")
    ax.text(
        0.03,
        0.05,
        "Guide-to-eye fit across all 6 table points has a negative slope.",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.92, "edgecolor": "#cccccc"},
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=240, bbox_inches="tight")
    if out_path.suffix.lower() == ".png":
        fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_time_accuracy_story_by_dataset(points_df: pd.DataFrame, dataset: str, out_path: Path) -> None:
    sub = points_df[points_df["dataset"] == dataset].sort_values("avg_duration_sec").copy()
    xs = sub["avg_duration_sec"].to_numpy()
    ys = sub["success_rate"].to_numpy()
    err_lo = ys - sub["success_rate_ci_lo"].to_numpy()
    err_hi = sub["success_rate_ci_hi"].to_numpy() - ys

    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    ax.errorbar(
        xs,
        ys,
        yerr=np.vstack([err_lo, err_hi]),
        fmt="-o",
        capsize=4,
        linewidth=2.4,
        markersize=7,
        color=DATASET_COLORS[dataset],
    )
    for _, row in sub.iterrows():
        ax.annotate(
            f"{row['split']} ({row['success_rate'] * 100:.0f}%)",
            (row["avg_duration_sec"], row["success_rate"]),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=9,
            color=DATASET_COLORS[dataset],
        )

    span = float(xs.max() - xs.min())
    pad = max(span * 0.10, 1.0)
    tick_vals = np.linspace(xs.min(), xs.max(), 5)
    if xs.max() <= 20:
        tick_labels = [f"{t:.1f}" for t in tick_vals]
    else:
        tick_labels = [f"{t:.0f}" for t in tick_vals]

    ax.set_xlim(xs.min() - pad, xs.max() + pad)
    ax.set_xticks(tick_vals)
    ax.set_xticklabels(tick_labels)
    ax.set_xlabel("Average Duration (s)")
    ax.set_ylabel("Success Rate")
    ax.set_ylim(0.0, 0.42)
    ax.grid(alpha=0.25, linestyle="--")
    ax.set_title(f"{dataset}: Original Mode Time vs Success Rate")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=240, bbox_inches="tight")
    if out_path.suffix.lower() == ".png":
        fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_time_accuracy_all_modes_by_dataset(points_df: pd.DataFrame, dataset: str, out_path: Path) -> None:
    sub = points_df[points_df["dataset"] == dataset].copy()
    fig, ax = plt.subplots(figsize=(7.0, 4.6))

    for mode in MODE_ORDER:
        mode_rows = sub[sub["mode"] == mode].sort_values("avg_duration_sec")
        xs = mode_rows["avg_duration_sec"].to_numpy()
        ys = mode_rows["success_rate"].to_numpy()
        err_lo = ys - mode_rows["success_rate_ci_lo"].to_numpy()
        err_hi = mode_rows["success_rate_ci_hi"].to_numpy() - ys
        ax.errorbar(
            xs,
            ys,
            yerr=np.vstack([err_lo, err_hi]),
            fmt="-o",
            capsize=4,
            linewidth=2.2,
            markersize=7,
            color=MODE_COLORS[mode],
            label=mode,
        )
        for _, row in mode_rows.iterrows():
            ax.annotate(
                str(row["split"]),
                (row["avg_duration_sec"], row["success_rate"]),
                textcoords="offset points",
                xytext=(0, 7),
                ha="center",
                fontsize=8,
                color=MODE_COLORS[mode],
            )

    xs_all = sub["avg_duration_sec"].to_numpy()
    span = float(xs_all.max() - xs_all.min())
    pad = max(span * 0.10, 1.0)
    tick_vals = np.linspace(xs_all.min(), xs_all.max(), 5)
    if xs_all.max() <= 20:
        tick_labels = [f"{t:.1f}" for t in tick_vals]
    else:
        tick_labels = [f"{t:.0f}" for t in tick_vals]

    ax.set_xlim(xs_all.min() - pad, xs_all.max() + pad)
    ax.set_xticks(tick_vals)
    ax.set_xticklabels(tick_labels)
    ax.set_ylim(0.0, 0.72 if dataset == "AgiBot" else 0.60)
    ax.set_xlabel("Average Duration (s)")
    ax.set_ylabel("Success Rate")
    ax.set_title(f"{dataset}: Time vs Success Rate (All Table Modes)")
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(frameon=False, loc="upper right")
    ax.text(
        0.03,
        0.03,
        "Each point summarizes n=50 episodes.",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.92, "edgecolor": "#cccccc"},
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=240, bbox_inches="tight")
    if out_path.suffix.lower() == ".png":
        fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def write_all_modes_time_accuracy_table(md_path: Path, csv_path: Path, points_df: pd.DataFrame) -> None:
    points_df.to_csv(csv_path, index=False)
    lines = [
        "# Time-Accuracy Table (All Table Modes)",
        "",
        "| Dataset | Mode | Split | Avg Duration (s) | Success Rate | 95% CI | Task Progress | Mean L2 |",
        "| --- | --- | --- | ---: | ---: | --- | ---: | ---: |",
    ]
    ordered = points_df.sort_values(["dataset", "mode", "avg_duration_sec"])
    for _, row in ordered.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["dataset"]),
                    str(row["mode"]),
                    str(row["split"]),
                    f"{row['avg_duration_sec']:.1f}",
                    f"{row['success_rate'] * 100:.0f}%",
                    f"[{row['success_rate_ci_lo'] * 100:.0f}%, {row['success_rate_ci_hi'] * 100:.0f}%]",
                    f"{row['task_progress']:.3f}",
                    f"{row['mean_l2']:.4f}",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "- This table uses all 18 appendix cells: 2 datasets x 3 splits x 3 modes.",
            "- Since each cell summarizes 50 episodes, the full table represents 900 episode evaluations.",
            "",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze DreamZero appendix-table split statistics, separately then pooled.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "analysis_outputs" / "dreamzero_table_split_analysis_20260330",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    table_df = build_input_table(n_boot=args.bootstrap_samples, seed=args.seed)
    pooled_df = build_pooled_table(table_df, n_boot=args.bootstrap_samples, seed=args.seed)

    table_df.to_csv(args.output_dir / "input_table_split_stats.csv", index=False)
    pooled_df.to_csv(args.output_dir / "pooled_table_split_stats.csv", index=False)

    tests_df = build_pairwise_tests(table_df, dataset_col="dataset")
    pooled_tests_df = build_pairwise_tests(pooled_df, dataset_col="dataset")

    tests_df.to_csv(args.output_dir / "pairwise_fisher_by_dataset_mode.csv", index=False)
    pooled_tests_df.to_csv(args.output_dir / "pairwise_fisher_pooled_by_mode.csv", index=False)

    trend_rows = []
    for mode in MODE_ORDER:
        mode_df = table_df[table_df["mode"] == mode].copy()
        blocks = [
            build_trend_rows(mode_df, label="pooled_interaction", with_dataset=True, with_interaction=True),
            build_trend_rows(mode_df, label="pooled_no_interaction", with_dataset=True, with_interaction=False),
            build_trend_rows(
                mode_df[mode_df["dataset"] == "DROID"],
                label="DROID_only",
                with_dataset=False,
                with_interaction=False,
            ),
            build_trend_rows(
                mode_df[mode_df["dataset"] == "AgiBot"],
                label="AgiBot_only",
                with_dataset=False,
                with_interaction=False,
            ),
        ]
        for block in blocks:
            for row in block:
                trend_rows.append({"mode": mode, **row})
    trend_df = pd.DataFrame(trend_rows)
    trend_df = trend_df[["mode", "model", "term", "beta", "se", "p_value", "odds_ratio"]]
    trend_df.to_csv(args.output_dir / "ordered_split_trend_logit.csv", index=False)

    main_trend_df, progress_trend_df = build_original_trend_presentation(table_df, pooled_df, trend_df)
    main_trend_df.to_csv(args.output_dir / "original_trend_proof_table.csv", index=False)
    progress_trend_df.to_csv(args.output_dir / "original_progress_support_table.csv", index=False)
    time_accuracy_df = build_time_accuracy_points(table_df, mode_filter="original")
    time_accuracy_all_modes_df = build_time_accuracy_points(table_df, mode_filter=None)

    plot_dataset_sr(table_df, args.output_dir / "sr_by_dataset_mode.png")
    plot_pooled_sr(pooled_df, args.output_dir / "sr_pooled_by_mode.png")
    plot_pooled_metrics(pooled_df, args.output_dir / "metrics_pooled_by_mode.png")
    plot_original_story_figure(
        table_df=table_df,
        pooled_df=pooled_df,
        trend_df=trend_df,
        out_path=args.output_dir / "original_story_figure.png",
    )
    plot_time_accuracy_story(time_accuracy_df, args.output_dir / "original_time_accuracy_story.png")
    plot_time_accuracy_story_by_dataset(
        time_accuracy_df,
        dataset="DROID",
        out_path=args.output_dir / "droid_time_accuracy_original.png",
    )
    plot_time_accuracy_story_by_dataset(
        time_accuracy_df,
        dataset="AgiBot",
        out_path=args.output_dir / "agibot_time_accuracy_original.png",
    )
    plot_time_accuracy_all_modes_by_dataset(
        time_accuracy_all_modes_df,
        dataset="DROID",
        out_path=args.output_dir / "droid_time_accuracy_all_modes.png",
    )
    plot_time_accuracy_all_modes_by_dataset(
        time_accuracy_all_modes_df,
        dataset="AgiBot",
        out_path=args.output_dir / "agibot_time_accuracy_all_modes.png",
    )

    write_summary(
        out_path=args.output_dir / "summary.md",
        table_df=table_df,
        pooled_df=pooled_df,
        tests_df=tests_df,
        pooled_tests_df=pooled_tests_df,
        trend_df=trend_df,
    )
    write_original_trend_presentation(
        out_path=args.output_dir / "original_trend_proof_table.md",
        latex_path=args.output_dir / "original_trend_proof_table.tex",
        main_df=main_trend_df,
        progress_df=progress_trend_df,
    )
    write_time_accuracy_table(
        md_path=args.output_dir / "original_time_accuracy_table.md",
        tex_path=args.output_dir / "original_time_accuracy_table.tex",
        csv_path=args.output_dir / "original_time_accuracy_points.csv",
        points_df=time_accuracy_df,
    )
    write_all_modes_time_accuracy_table(
        md_path=args.output_dir / "all_modes_time_accuracy_table.md",
        csv_path=args.output_dir / "all_modes_time_accuracy_points.csv",
        points_df=time_accuracy_all_modes_df,
    )

    print(f"Wrote table-split analysis artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
