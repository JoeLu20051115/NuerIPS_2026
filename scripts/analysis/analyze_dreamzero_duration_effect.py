#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import fisher_exact, norm


REPO_ROOT = Path(__file__).resolve().parents[2]

DROID_RESULT_PATH = REPO_ROOT / "evaluation_results_dualsystem" / "droid_3way_compare.json"
AGIBOT_RESULT_PATH = REPO_ROOT / "evaluation_results_dualsystem" / "agibot_3way_compare.json"

DROID_META_PATHS = {
    "L1": REPO_ROOT / "data" / "final_data1" / "DRO_L1_150" / "meta" / "episodes.jsonl",
    "L2": REPO_ROOT / "data" / "final_data1" / "DRO_L2_100" / "meta" / "episodes.jsonl",
    "L3": REPO_ROOT / "data" / "final_data1" / "DRO_L3_150" / "meta" / "episodes.jsonl",
}
AGIBOT_META_PATHS = {
    "L1": REPO_ROOT / "data" / "final_data1" / "Agi_L1_150" / "meta" / "manifest.json",
    "L3": REPO_ROOT / "data" / "final_data1" / "Agi_L3_150" / "meta" / "manifest.json",
}

ABS_BUCKET_EDGES = [0.0, 30.0, 60.0, 120.0, np.inf]
ABS_BUCKET_LABELS = ["0-30s", "30-60s", "60-120s", "120s+"]
MODE_LABELS = {
    "task_token_only": "original",
    "dual_llm": "llm_dual",
    "val_llm": "llm_val",
}
PLOT_COLORS = {
    "DROID": "#22577A",
    "AgiBot": "#C44536",
    "original": "#6B2D5C",
    "llm_dual": "#2A6F97",
    "llm_val": "#3A7D44",
}


@dataclass
class FitResult:
    terms: list[str]
    beta: np.ndarray
    se: np.ndarray
    p_value: np.ndarray
    odds_ratio: np.ndarray
    cov: np.ndarray


def canonical_mode(mode: str) -> str:
    mode = mode.strip().lower()
    if mode == "llm_val":
        return "val_llm"
    return mode


def bucket_duration(duration_sec: float) -> str:
    for idx, label in enumerate(ABS_BUCKET_LABELS):
        if ABS_BUCKET_EDGES[idx] <= duration_sec < ABS_BUCKET_EDGES[idx + 1]:
            return label
    raise ValueError(f"Unexpected duration: {duration_sec}")


def load_droid_full_compare() -> pd.DataFrame:
    meta: dict[str, dict] = {}
    for level, path in DROID_META_PATHS.items():
        with path.open() as f:
            for line in f:
                row = json.loads(line)
                episode_index = str(row["episode_index"])
                meta[episode_index] = {
                    "dataset": "DROID",
                    "level": level,
                    # Confirmed from parquet timestamps: 0.0667s between frames => 15 Hz.
                    "duration_sec": float(row["length"]) / 15.0,
                    "task_meta": (row.get("tasks") or [""])[0],
                }

    with DROID_RESULT_PATH.open() as f:
        payload = json.load(f)

    records: list[dict] = []
    for row in payload["results"]:
        episode_id = str(row["episode_id"])
        if episode_id not in meta:
            continue
        mode = canonical_mode(row["mode"])
        records.append(
            {
                "dataset": "DROID",
                "episode_id": episode_id,
                "task": row.get("task") or meta[episode_id]["task_meta"],
                "mode": mode,
                "mode_label": MODE_LABELS[mode],
                "level": meta[episode_id]["level"],
                "duration_sec": meta[episode_id]["duration_sec"],
                "duration_bucket_abs": bucket_duration(meta[episode_id]["duration_sec"]),
                "task_progress": float(row["task_progress"]),
                "mean_l2": float(row["mean_l2"]) if row.get("mean_l2") is not None else np.nan,
                "task_success": bool(row["task_success"]),
            }
        )

    return pd.DataFrame.from_records(records)


def load_agibot_full_compare() -> pd.DataFrame:
    meta: dict[str, dict] = {}
    for level, path in AGIBOT_META_PATHS.items():
        with path.open() as f:
            manifest = json.load(f)
        for row in manifest["episodes"]:
            episode_id = str(row["episode_id"])
            meta[episode_id] = {
                "dataset": "AgiBot",
                "level": level,
                "duration_sec": float(row["duration_sec"]),
                "task_meta": row["english_task_name"],
            }

    with AGIBOT_RESULT_PATH.open() as f:
        payload = json.load(f)

    records: list[dict] = []
    for row in payload["results"]:
        episode_id = str(row["episode_id"])
        if episode_id not in meta:
            continue
        mode = canonical_mode(row["mode"])
        records.append(
            {
                "dataset": "AgiBot",
                "episode_id": episode_id,
                "task": row.get("task") or meta[episode_id]["task_meta"],
                "mode": mode,
                "mode_label": MODE_LABELS[mode],
                "level": meta[episode_id]["level"],
                "duration_sec": meta[episode_id]["duration_sec"],
                "duration_bucket_abs": bucket_duration(meta[episode_id]["duration_sec"]),
                "task_progress": float(row["task_progress"]),
                "mean_l2": float(row["mean_l2"]) if row.get("mean_l2") is not None else np.nan,
                "task_success": bool(row["task_success"]),
            }
        )

    return pd.DataFrame.from_records(records)


def bootstrap_mean_ci(values: np.ndarray, n_boot: int, seed: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    sample_idx = rng.integers(0, values.size, size=(n_boot, values.size))
    sample_means = values[sample_idx].mean(axis=1)
    lo, hi = np.quantile(sample_means, [0.025, 0.975])
    return float(lo), float(hi)


def build_bucket_summary(
    df: pd.DataFrame,
    group_cols: list[str],
    bucket_col: str,
    n_boot: int,
    seed: int,
    bucket_categories: list[str] | None = None,
) -> pd.DataFrame:
    rows: list[dict] = []
    for key, group in df.groupby(group_cols + [bucket_col], dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        row = dict(zip(group_cols + [bucket_col], key))
        success = group["task_success"].astype(float).to_numpy()
        sr = float(success.mean()) if len(success) else np.nan
        sr_lo, sr_hi = bootstrap_mean_ci(success, n_boot=n_boot, seed=seed)
        row.update(
            {
                "n": int(len(group)),
                "success_rate": sr,
                "success_rate_ci_lo": sr_lo,
                "success_rate_ci_hi": sr_hi,
                "mean_task_progress": float(group["task_progress"].mean()) if len(group) else np.nan,
                "mean_l2": float(group["mean_l2"].mean()) if len(group) else np.nan,
            }
        )
        rows.append(row)

    out = pd.DataFrame(rows)
    if bucket_col in out.columns:
        categories = bucket_categories
        if categories is None:
            source = df[bucket_col]
            if hasattr(source.dtype, "categories"):
                categories = list(source.dtype.categories)
            else:
                categories = list(pd.unique(out[bucket_col]))
        out[bucket_col] = pd.Categorical(out[bucket_col], categories=categories, ordered=True)
        out = out.sort_values(group_cols + [bucket_col]).reset_index(drop=True)
    return out


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


def build_adjacent_bucket_tests(
    df: pd.DataFrame,
    dataset_col: str,
    bucket_col: str,
) -> pd.DataFrame:
    tests: list[dict] = []
    for dataset, group in df.groupby(dataset_col):
        for left, right in zip(ABS_BUCKET_LABELS, ABS_BUCKET_LABELS[1:]):
            left_group = group[group[bucket_col] == left]
            right_group = group[group[bucket_col] == right]
            if left_group.empty or right_group.empty:
                continue

            success_left = int(left_group["task_success"].sum())
            fail_left = int(len(left_group) - success_left)
            success_right = int(right_group["task_success"].sum())
            fail_right = int(len(right_group) - success_right)
            odds_ratio, p_value = fisher_exact(
                [[success_left, fail_left], [success_right, fail_right]]
            )
            tests.append(
                {
                    "dataset": dataset,
                    "bucket_left": left,
                    "bucket_right": right,
                    "n_left": int(len(left_group)),
                    "n_right": int(len(right_group)),
                    "success_left": success_left,
                    "success_right": success_right,
                    "success_rate_left": float(left_group["task_success"].mean()),
                    "success_rate_right": float(right_group["task_success"].mean()),
                    "odds_ratio": float(odds_ratio),
                    "p_value": float(p_value),
                }
            )

    out = pd.DataFrame(tests)
    if out.empty:
        return out
    out["p_value_bh"] = benjamini_hochberg(out["p_value"].tolist())
    return out.sort_values(["dataset", "bucket_left"]).reset_index(drop=True)


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


def linear_combo_from_fit(
    fit: FitResult,
    weights: np.ndarray,
    label: str,
) -> dict:
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


def droid_short_quartile_labels(df: pd.DataFrame) -> pd.DataFrame:
    droid = df[df["dataset"] == "DROID"].copy()
    quantiles = np.quantile(droid["duration_sec"].to_numpy(), [0.0, 0.25, 0.5, 0.75, 1.0])
    labels: list[str] = []
    bins: list[tuple[float, float, str]] = []
    for idx in range(4):
        lo = float(quantiles[idx])
        hi = float(quantiles[idx + 1])
        label = f"Q{idx + 1} ({lo:.1f}-{hi:.1f}s)"
        bins.append((lo, hi, label))
        labels.append(label)

    def assign_bucket(duration_sec: float) -> str:
        for idx, (lo, hi, label) in enumerate(bins):
            if idx == len(bins) - 1:
                if lo <= duration_sec <= hi:
                    return label
            elif lo <= duration_sec < hi:
                return label
        raise ValueError(f"Unexpected DROID duration {duration_sec}")

    droid["duration_bucket_quartile"] = droid["duration_sec"].map(assign_bucket)
    droid["duration_bucket_quartile"] = pd.Categorical(
        droid["duration_bucket_quartile"], categories=labels, ordered=True
    )
    return droid


def plot_sr_curve(
    summary: pd.DataFrame,
    out_path: Path,
    title_prefix: str,
    dataset_order: list[str],
    bucket_col: str,
) -> None:
    fig, axes = plt.subplots(1, len(dataset_order), figsize=(13, 4.5), sharey=True)
    if len(dataset_order) == 1:
        axes = [axes]

    for ax, dataset in zip(axes, dataset_order):
        sub = summary[summary["dataset"] == dataset].copy()
        categories = list(sub[bucket_col].cat.categories) if not sub.empty else ABS_BUCKET_LABELS
        sub = sub.set_index(bucket_col)
        xs = np.arange(len(categories))
        ys = []
        err_lo = []
        err_hi = []
        counts = []
        for label in categories:
            if label in sub.index:
                row = sub.loc[label]
                ys.append(float(row["success_rate"]))
                err_lo.append(float(row["success_rate"]) - float(row["success_rate_ci_lo"]))
                err_hi.append(float(row["success_rate_ci_hi"]) - float(row["success_rate"]))
                counts.append(int(row["n"]))
            else:
                ys.append(np.nan)
                err_lo.append(np.nan)
                err_hi.append(np.nan)
                counts.append(0)

        ax.errorbar(
            xs,
            ys,
            yerr=np.vstack([err_lo, err_hi]),
            fmt="-o",
            capsize=4,
            linewidth=2.0,
            markersize=7,
            color=PLOT_COLORS[dataset],
        )
        for x, y, n in zip(xs, ys, counts):
            if np.isnan(y):
                ax.text(x, 0.03, f"n={n}", ha="center", va="bottom", fontsize=9, color="#777777")
            else:
                ax.text(x, 0.03, f"n={n}", ha="center", va="bottom", fontsize=9, color="#444444")

        ax.set_xticks(xs)
        ax.set_xticklabels(categories, rotation=20)
        ax.set_ylim(0.0, 1.02)
        ax.grid(alpha=0.25, linestyle="--")
        ax.set_title(dataset)
        ax.set_xlabel("Duration Bucket")

    axes[0].set_ylabel("Success Rate")
    fig.suptitle(title_prefix)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_metric_curves(
    summary: pd.DataFrame,
    out_path: Path,
    dataset_order: list[str],
    bucket_col: str,
) -> None:
    metrics = [
        ("mean_task_progress", "Mean Task Progress"),
        ("mean_l2", "Mean L2"),
    ]
    fig, axes = plt.subplots(
        len(dataset_order),
        len(metrics),
        figsize=(12, 7),
        sharex=False,
        sharey="col",
    )
    if len(dataset_order) == 1:
        axes = np.array([axes])

    for row_idx, dataset in enumerate(dataset_order):
        sub = summary[summary["dataset"] == dataset].copy().set_index(bucket_col)
        categories = list(sub.index.categories) if hasattr(sub.index, "categories") else ABS_BUCKET_LABELS
        xs = np.arange(len(categories))
        for col_idx, (metric, label) in enumerate(metrics):
            ax = axes[row_idx, col_idx]
            ys = [
                float(sub.loc[label_name][metric]) if label_name in sub.index else np.nan
                for label_name in categories
            ]
            ax.plot(
                xs,
                ys,
                "-o",
                linewidth=2.0,
                markersize=7,
                color=PLOT_COLORS[dataset],
            )
            ax.set_xticks(xs)
            ax.set_xticklabels(categories, rotation=20)
            ax.grid(alpha=0.25, linestyle="--")
            if row_idx == 0:
                ax.set_title(label)
            if col_idx == 0:
                ax.set_ylabel(dataset)
            if metric == "mean_task_progress":
                ax.set_ylim(0.0, 1.02)
            ax.set_xlabel("Duration Bucket")

    fig.suptitle("Old DreamZero Data: Original Mode Metrics by Duration Bucket")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_pooled_sr_curve(
    summary: pd.DataFrame,
    out_path: Path,
    title: str,
    bucket_col: str,
) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.5))
    categories = list(summary[bucket_col].cat.categories) if not summary.empty else ABS_BUCKET_LABELS
    sub = summary.set_index(bucket_col)
    xs = np.arange(len(categories))
    ys = []
    err_lo = []
    err_hi = []
    counts = []
    for label in categories:
        if label in sub.index:
            row = sub.loc[label]
            ys.append(float(row["success_rate"]))
            err_lo.append(float(row["success_rate"]) - float(row["success_rate_ci_lo"]))
            err_hi.append(float(row["success_rate_ci_hi"]) - float(row["success_rate"]))
            counts.append(int(row["n"]))
        else:
            ys.append(np.nan)
            err_lo.append(np.nan)
            err_hi.append(np.nan)
            counts.append(0)

    ax.errorbar(
        xs,
        ys,
        yerr=np.vstack([err_lo, err_hi]),
        fmt="-o",
        capsize=4,
        linewidth=2.0,
        markersize=7,
        color="#1F4E79",
    )
    for x, y, n in zip(xs, ys, counts):
        if np.isnan(y):
            ax.text(x, 0.03, f"n={n}", ha="center", va="bottom", fontsize=9, color="#777777")
        else:
            ax.text(x, 0.03, f"n={n}", ha="center", va="bottom", fontsize=9, color="#444444")
    ax.set_xticks(xs)
    ax.set_xticklabels(categories, rotation=20)
    ax.set_ylim(0.0, 1.02)
    ax.grid(alpha=0.25, linestyle="--")
    ax.set_xlabel("Duration Bucket")
    ax.set_ylabel("Success Rate")
    ax.set_title(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_markdown_summary(
    out_path: Path,
    droid_abs: pd.DataFrame,
    agibot_abs: pd.DataFrame,
    pairwise_abs: pd.DataFrame,
    combined_rows: list[dict],
    pooled_abs: pd.DataFrame,
    pairwise_pooled: pd.DataFrame,
    pooled_rows: list[dict],
    droid_rows: list[dict],
    agibot_rows: list[dict],
) -> None:
    droid_note = (
        "DROID old DreamZero episodes span only 3.1s to 37.7s; the requested absolute 0-30/30-60/60-120/120+ "
        "bucket analysis therefore leaves only 3 episodes outside the first bucket."
    )
    lines = [
        "# DreamZero Duration-Effect Analysis",
        "",
        "## Sources",
        f"- DROID: `{DROID_RESULT_PATH.relative_to(REPO_ROOT)}` joined with `data/final_data1/DRO_L1_150`, `DRO_L2_100`, `DRO_L3_150` metadata.",
        f"- AgiBot: `{AGIBOT_RESULT_PATH.relative_to(REPO_ROOT)}` joined with `data/final_data1/Agi_L1_150` and `Agi_L3_150` manifests.",
        "",
        "## Core Read",
        f"- {droid_note}",
        "- AgiBot spans 14.3s to 264.2s, so the requested absolute buckets are meaningful there.",
        "- The analysis below uses the `original/task_token_only` mode for the main claim, because the target hypothesis is that the original single-prompt method degrades with longer tasks.",
        "",
        "## Descriptive Result",
    ]

    for dataset, frame in [("DROID", droid_abs), ("AgiBot", agibot_abs)]:
        lines.append(f"### {dataset}")
        for _, row in frame.iterrows():
            lines.append(
                f"- {row['duration_bucket_abs']}: n={int(row['n'])}, SR={row['success_rate']:.3f}, "
                f"95% CI=[{row['success_rate_ci_lo']:.3f}, {row['success_rate_ci_hi']:.3f}], "
                f"mean progress={row['mean_task_progress']:.3f}, mean L2={row['mean_l2']:.4f}"
            )
        lines.append("")

    lines.extend(
        [
            "## Pairwise Bucket Tests",
            "- Fisher exact tests compare adjacent duration buckets on success rate only.",
            "- Benjamini-Hochberg is applied across all available adjacent-bucket tests.",
            "",
        ]
    )
    if pairwise_abs.empty:
        lines.append("- No adjacent-bucket tests were possible.")
        lines.append("")
    else:
        for _, row in pairwise_abs.iterrows():
            lines.append(
                f"- {row['dataset']} {row['bucket_left']} vs {row['bucket_right']}: "
                f"p={row['p_value']:.4g}, BH-adjusted p={row['p_value_bh']:.4g}, "
                f"SR {row['success_rate_left']:.3f} -> {row['success_rate_right']:.3f}"
            )
        lines.append("")

    lines.extend(
        [
            "## Pooled Across Datasets",
            "- This combines DROID and AgiBot into one unseen-test-set pool.",
            "- Important caveat: the pooled 0-30s bucket is DROID-heavy, while the long-duration buckets are effectively AgiBot-only.",
            "",
        ]
    )
    for _, row in pooled_abs.iterrows():
        lines.append(
            f"- {row['duration_bucket_abs']}: n={int(row['n'])}, SR={row['success_rate']:.3f}, "
            f"95% CI=[{row['success_rate_ci_lo']:.3f}, {row['success_rate_ci_hi']:.3f}], "
            f"mean progress={row['mean_task_progress']:.3f}, mean L2={row['mean_l2']:.4f}"
        )
    lines.append("")
    if pairwise_pooled.empty:
        lines.append("- No pooled adjacent-bucket tests were possible.")
        lines.append("")
    else:
        for _, row in pairwise_pooled.iterrows():
            lines.append(
                f"- pooled {row['bucket_left']} vs {row['bucket_right']}: "
                f"p={row['p_value']:.4g}, BH-adjusted p={row['p_value_bh']:.4g}, "
                f"SR {row['success_rate_left']:.3f} -> {row['success_rate_right']:.3f}"
            )
        lines.append("")

    lines.extend(
        [
            "## Regression Read",
            "- Combined model: `logit(success) ~ duration + level + dataset + duration:dataset` on original mode.",
            "- A pooled no-interaction model is also reported because a single mixed statistic can be useful as a compact appendix summary.",
            "- Additional per-dataset models are reported for interpretability.",
            "",
            "### Combined Original-Mode Model",
        ]
    )
    for row in combined_rows:
        lines.append(
            f"- {row['term']}: beta={row['beta']:.4f}, OR={row['odds_ratio']:.3f}, p={row['p_value']:.4g}"
        )

    lines.extend(["", "### Pooled No-Interaction Model"])
    for row in pooled_rows:
        lines.append(
            f"- {row['term']}: beta={row['beta']:.4f}, OR={row['odds_ratio']:.3f}, p={row['p_value']:.4g}"
        )

    lines.extend(["", "### DROID-Only Model"])
    for row in droid_rows:
        lines.append(
            f"- {row['term']}: beta={row['beta']:.4f}, OR={row['odds_ratio']:.3f}, p={row['p_value']:.4g}"
        )

    lines.extend(["", "### AgiBot-Only Model"])
    for row in agibot_rows:
        lines.append(
            f"- {row['term']}: beta={row['beta']:.4f}, OR={row['odds_ratio']:.3f}, p={row['p_value']:.4g}"
        )

    lines.extend(
        [
            "",
            "## Verdict",
            "- AgiBot shows a descriptive drop in original success once duration exceeds 60s, but the effect is not monotonic at the short end and the adjacent-bucket tests are not significant after correction.",
            "- After controlling for level, the original-mode AgiBot duration coefficient is not significantly negative.",
            "- DROID does not provide meaningful long-horizon coverage in the archived old DreamZero run, so it cannot strongly support a 'longer tasks cause collapse' claim on an absolute-seconds axis.",
            "- The pooled curve is also not convincingly monotonic: 0-30s -> 30-60s rises before later buckets fall, and the pooled no-interaction duration coefficient is near zero and not significant.",
            "- Overall, the current archived old DreamZero data provide at most a suggestive AgiBot trend, not a strong standalone causal argument.",
            "",
        ]
    )
    out_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze duration effects in archived DreamZero DROID/AgiBot results.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "analysis_outputs" / "duration_effect_dreamzero_20260330",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    droid = load_droid_full_compare()
    agibot = load_agibot_full_compare()
    all_rows = pd.concat([droid, agibot], ignore_index=True)
    all_rows.to_csv(args.output_dir / "joined_episode_results.csv", index=False)

    original = all_rows[all_rows["mode"] == "task_token_only"].copy()
    original.to_csv(args.output_dir / "joined_episode_results_original.csv", index=False)

    abs_summary_all_modes = build_bucket_summary(
        all_rows,
        group_cols=["dataset", "mode_label"],
        bucket_col="duration_bucket_abs",
        n_boot=args.bootstrap_samples,
        seed=args.seed,
        bucket_categories=ABS_BUCKET_LABELS,
    )
    abs_summary_all_modes.to_csv(args.output_dir / "absolute_bucket_summary_all_modes.csv", index=False)

    abs_summary_original = build_bucket_summary(
        original,
        group_cols=["dataset"],
        bucket_col="duration_bucket_abs",
        n_boot=args.bootstrap_samples,
        seed=args.seed,
        bucket_categories=ABS_BUCKET_LABELS,
    )
    abs_summary_original.to_csv(args.output_dir / "absolute_bucket_summary_original.csv", index=False)

    abs_tests_original = build_adjacent_bucket_tests(
        original,
        dataset_col="dataset",
        bucket_col="duration_bucket_abs",
    )
    abs_tests_original.to_csv(args.output_dir / "absolute_bucket_tests_original.csv", index=False)

    pooled_original = original.assign(pooled_dataset="Pooled")
    pooled_abs_summary_original = build_bucket_summary(
        pooled_original,
        group_cols=["pooled_dataset"],
        bucket_col="duration_bucket_abs",
        n_boot=args.bootstrap_samples,
        seed=args.seed,
        bucket_categories=ABS_BUCKET_LABELS,
    )
    pooled_abs_summary_original.to_csv(
        args.output_dir / "absolute_bucket_summary_original_pooled.csv", index=False
    )
    pooled_abs_tests_original = build_adjacent_bucket_tests(
        pooled_original,
        dataset_col="pooled_dataset",
        bucket_col="duration_bucket_abs",
    )
    pooled_abs_tests_original.to_csv(
        args.output_dir / "absolute_bucket_tests_original_pooled.csv", index=False
    )

    droid_quart = droid_short_quartile_labels(original)
    droid_quart_summary = build_bucket_summary(
        droid_quart,
        group_cols=["dataset"],
        bucket_col="duration_bucket_quartile",
        n_boot=args.bootstrap_samples,
        seed=args.seed,
        bucket_categories=list(droid_quart["duration_bucket_quartile"].cat.categories),
    )
    droid_quart_summary.to_csv(args.output_dir / "droid_short_duration_quartiles_original.csv", index=False)

    plot_sr_curve(
        summary=abs_summary_original,
        out_path=args.output_dir / "duration_sr_original_absolute_buckets.png",
        title_prefix="Old DreamZero Data: Original Mode SR vs Absolute Duration Bucket",
        dataset_order=["DROID", "AgiBot"],
        bucket_col="duration_bucket_abs",
    )
    plot_metric_curves(
        summary=abs_summary_original,
        out_path=args.output_dir / "duration_metrics_original_absolute_buckets.png",
        dataset_order=["DROID", "AgiBot"],
        bucket_col="duration_bucket_abs",
    )
    plot_pooled_sr_curve(
        summary=pooled_abs_summary_original,
        out_path=args.output_dir / "duration_sr_original_absolute_buckets_pooled.png",
        title="Old DreamZero Data: Original Mode SR vs Duration Bucket (Pooled)",
        bucket_col="duration_bucket_abs",
    )
    plot_sr_curve(
        summary=droid_quart_summary,
        out_path=args.output_dir / "droid_sr_original_short_duration_quartiles.png",
        title_prefix="DROID Only: Original Mode SR vs Short-Duration Quartiles",
        dataset_order=["DROID"],
        bucket_col="duration_bucket_quartile",
    )

    # Combined original-mode regression.
    X_combined = []
    y_combined = []
    for _, row in original.iterrows():
        duration_30 = float(row["duration_sec"]) / 30.0
        is_agibot = 1.0 if row["dataset"] == "AgiBot" else 0.0
        level_l2 = 1.0 if row["level"] == "L2" else 0.0
        level_l3 = 1.0 if row["level"] == "L3" else 0.0
        X_combined.append([1.0, duration_30, level_l2, level_l3, is_agibot, duration_30 * is_agibot])
        y_combined.append(float(row["task_success"]))
    fit_combined = fit_logistic_regression(
        np.asarray(X_combined, dtype=float),
        np.asarray(y_combined, dtype=float),
        terms=[
            "Intercept",
            "duration_per_30s",
            "level_L2",
            "level_L3",
            "dataset_AgiBot",
            "duration_per_30s:dataset_AgiBot",
        ],
    )
    combined_rows = []
    for term, beta, se, p_value, odds_ratio in zip(
        fit_combined.terms,
        fit_combined.beta,
        fit_combined.se,
        fit_combined.p_value,
        fit_combined.odds_ratio,
    ):
        combined_rows.append(
            {
                "term": term,
                "beta": float(beta),
                "se": float(se),
                "p_value": float(p_value),
                "odds_ratio": float(odds_ratio),
            }
        )

    combined_rows.append(
        linear_combo_from_fit(
            fit_combined,
            weights=np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
            label="duration_per_30s | DROID slope",
        )
    )
    combined_rows.append(
        linear_combo_from_fit(
            fit_combined,
            weights=np.array([0.0, 1.0, 0.0, 0.0, 0.0, 1.0]),
            label="duration_per_30s | AgiBot net slope",
        )
    )
    pd.DataFrame(combined_rows).to_csv(args.output_dir / "logit_original_combined.csv", index=False)

    X_pooled = []
    y_pooled = []
    for _, row in original.iterrows():
        X_pooled.append(
            [
                1.0,
                float(row["duration_sec"]) / 30.0,
                1.0 if row["level"] == "L2" else 0.0,
                1.0 if row["level"] == "L3" else 0.0,
                1.0 if row["dataset"] == "AgiBot" else 0.0,
            ]
        )
        y_pooled.append(float(row["task_success"]))
    fit_pooled = fit_logistic_regression(
        np.asarray(X_pooled, dtype=float),
        np.asarray(y_pooled, dtype=float),
        terms=["Intercept", "duration_per_30s", "level_L2", "level_L3", "dataset_AgiBot"],
    )
    pooled_rows = [
        {
            "term": term,
            "beta": float(beta),
            "se": float(se),
            "p_value": float(p_value),
            "odds_ratio": float(odds_ratio),
        }
        for term, beta, se, p_value, odds_ratio in zip(
            fit_pooled.terms, fit_pooled.beta, fit_pooled.se, fit_pooled.p_value, fit_pooled.odds_ratio
        )
    ]
    pd.DataFrame(pooled_rows).to_csv(
        args.output_dir / "logit_original_pooled_no_interaction.csv", index=False
    )

    # DROID-only regression.
    droid_original = original[original["dataset"] == "DROID"].copy()
    X_droid = []
    y_droid = []
    for _, row in droid_original.iterrows():
        X_droid.append(
            [
                1.0,
                float(row["duration_sec"]) / 10.0,
                1.0 if row["level"] == "L2" else 0.0,
                1.0 if row["level"] == "L3" else 0.0,
            ]
        )
        y_droid.append(float(row["task_success"]))
    fit_droid = fit_logistic_regression(
        np.asarray(X_droid, dtype=float),
        np.asarray(y_droid, dtype=float),
        terms=["Intercept", "duration_per_10s", "level_L2", "level_L3"],
    )
    droid_rows = [
        {
            "term": term,
            "beta": float(beta),
            "se": float(se),
            "p_value": float(p_value),
            "odds_ratio": float(odds_ratio),
        }
        for term, beta, se, p_value, odds_ratio in zip(
            fit_droid.terms, fit_droid.beta, fit_droid.se, fit_droid.p_value, fit_droid.odds_ratio
        )
    ]
    pd.DataFrame(droid_rows).to_csv(args.output_dir / "logit_original_droid_only.csv", index=False)

    # AgiBot-only regression.
    agibot_original = original[original["dataset"] == "AgiBot"].copy()
    X_agibot = []
    y_agibot = []
    for _, row in agibot_original.iterrows():
        X_agibot.append(
            [
                1.0,
                float(row["duration_sec"]) / 30.0,
                1.0 if row["level"] == "L3" else 0.0,
            ]
        )
        y_agibot.append(float(row["task_success"]))
    fit_agibot = fit_logistic_regression(
        np.asarray(X_agibot, dtype=float),
        np.asarray(y_agibot, dtype=float),
        terms=["Intercept", "duration_per_30s", "level_L3"],
    )
    agibot_rows = [
        {
            "term": term,
            "beta": float(beta),
            "se": float(se),
            "p_value": float(p_value),
            "odds_ratio": float(odds_ratio),
        }
        for term, beta, se, p_value, odds_ratio in zip(
            fit_agibot.terms, fit_agibot.beta, fit_agibot.se, fit_agibot.p_value, fit_agibot.odds_ratio
        )
    ]
    pd.DataFrame(agibot_rows).to_csv(args.output_dir / "logit_original_agibot_only.csv", index=False)

    write_markdown_summary(
        out_path=args.output_dir / "summary.md",
        droid_abs=abs_summary_original[abs_summary_original["dataset"] == "DROID"],
        agibot_abs=abs_summary_original[abs_summary_original["dataset"] == "AgiBot"],
        pairwise_abs=abs_tests_original,
        combined_rows=combined_rows,
        pooled_abs=pooled_abs_summary_original,
        pairwise_pooled=pooled_abs_tests_original,
        pooled_rows=pooled_rows,
        droid_rows=droid_rows,
        agibot_rows=agibot_rows,
    )

    print(f"Wrote analysis artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
