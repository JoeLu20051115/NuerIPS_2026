#!/usr/bin/env python3
from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import pandas as pd
from scipy.stats import binomtest, fisher_exact, wilcoxon


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "analysis_outputs" / "dreamzero_significance_20260330"
DROID_SELECTED150_PATH = REPO_ROOT / "evaluation_results_dualsystem" / "droid_3way_selected150.json"
AGIBOT_3WAY_PATH = REPO_ROOT / "evaluation_results_dualsystem" / "agibot_3way_compare.json"
AGIBOT_SPLIT_RESULTS_DIR = REPO_ROOT / "evaluation_results_dualsystem" / "selected_splits"

SPLIT_ORDER = ["L1", "L2", "L3"]
MODE_ORDER = ["original", "llm_dual", "llm_val"]
COMPARISONS = [("original", "llm_dual"), ("original", "llm_val")]

SCREENSHOT_TABLE_ROWS = [
    {"dataset": "DROID", "split": "L1", "mode": "original", "n": 50, "success_rate": 0.30},
    {"dataset": "DROID", "split": "L1", "mode": "llm_dual", "n": 50, "success_rate": 0.44},
    {"dataset": "DROID", "split": "L1", "mode": "llm_val", "n": 50, "success_rate": 0.50},
    {"dataset": "DROID", "split": "L2", "mode": "original", "n": 50, "success_rate": 0.18},
    {"dataset": "DROID", "split": "L2", "mode": "llm_dual", "n": 50, "success_rate": 0.32},
    {"dataset": "DROID", "split": "L2", "mode": "llm_val", "n": 50, "success_rate": 0.38},
    {"dataset": "DROID", "split": "L3", "mode": "original", "n": 50, "success_rate": 0.22},
    {"dataset": "DROID", "split": "L3", "mode": "llm_dual", "n": 50, "success_rate": 0.34},
    {"dataset": "DROID", "split": "L3", "mode": "llm_val", "n": 50, "success_rate": 0.48},
    {"dataset": "AgiBot", "split": "L1", "mode": "original", "n": 50, "success_rate": 0.30},
    {"dataset": "AgiBot", "split": "L1", "mode": "llm_dual", "n": 50, "success_rate": 0.52},
    {"dataset": "AgiBot", "split": "L1", "mode": "llm_val", "n": 50, "success_rate": 0.62},
    {"dataset": "AgiBot", "split": "L2", "mode": "original", "n": 50, "success_rate": 0.22},
    {"dataset": "AgiBot", "split": "L2", "mode": "llm_dual", "n": 50, "success_rate": 0.42},
    {"dataset": "AgiBot", "split": "L2", "mode": "llm_val", "n": 50, "success_rate": 0.48},
    {"dataset": "AgiBot", "split": "L3", "mode": "original", "n": 50, "success_rate": 0.16},
    {"dataset": "AgiBot", "split": "L3", "mode": "llm_dual", "n": 50, "success_rate": 0.48},
    {"dataset": "AgiBot", "split": "L3", "mode": "llm_val", "n": 50, "success_rate": 0.58},
]


def bh_adjust(p_values: list[float]) -> list[float]:
    if not p_values:
        return []
    m = len(p_values)
    order = sorted(range(m), key=lambda idx: p_values[idx])
    adjusted = [0.0] * m
    running = 1.0
    for rank, idx in reversed(list(enumerate(order, start=1))):
        value = min(running, p_values[idx] * m / rank)
        adjusted[idx] = value
        running = value
    return adjusted


def sig_label(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def load_droid_raw() -> pd.DataFrame:
    payload = json.loads(DROID_SELECTED150_PATH.read_text())
    rows = []
    for row in payload["results"]:
        mode = {
            "task_token_only": "original",
            "dual_llm": "llm_dual",
            "llm_val": "llm_val",
        }.get(row["mode"])
        if not mode:
            continue
        rows.append(
            {
                "dataset": "DROID",
                "split": row["dro_split"],
                "episode_id": str(row["episode_id"]),
                "mode": mode,
                "success": 1.0 if row["task_success"] else 0.0,
                "task_progress": float(row["task_progress"]),
                "mean_l2": float(row["mean_l2"]),
                "l2step": float(row["step_alignment_l2_lt_0_1"]),
            }
        )
    return pd.DataFrame(rows)


def load_agibot_recoverable_raw() -> pd.DataFrame:
    three_way = json.loads(AGIBOT_3WAY_PATH.read_text())["results"]
    val_lookup = {
        str(row["episode_id"]): row
        for row in three_way
        if row["mode"] in ("llm_val", "val_llm")
    }
    rows = []
    for split in SPLIT_ORDER:
        path = AGIBOT_SPLIT_RESULTS_DIR / f"Agi_DualBetter_{split}_50_results.json"
        payload = json.loads(path.read_text())["episodes"]
        for ep in payload:
            episode_id = str(ep["episode_id"])
            rows.append(
                {
                    "dataset": "AgiBot",
                    "split": split,
                    "episode_id": episode_id,
                    "mode": "original",
                    "success": 1.0 if ep["success_original"] else 0.0,
                    "task_progress": float(ep["progress_original"]),
                    "mean_l2": float(ep["mean_l2_original"]),
                    "l2step": float(ep["task_token_only"]["step_alignment_l2_lt_0_1"]),
                }
            )
            rows.append(
                {
                    "dataset": "AgiBot",
                    "split": split,
                    "episode_id": episode_id,
                    "mode": "llm_dual",
                    "success": 1.0 if ep["success_llm_dual"] else 0.0,
                    "task_progress": float(ep["progress_llm_dual"]),
                    "mean_l2": float(ep["mean_l2_llm_dual"]),
                    "l2step": float(ep["dual_llm"]["step_alignment_l2_lt_0_1"]),
                }
            )
            if episode_id in val_lookup:
                row = val_lookup[episode_id]
                rows.append(
                    {
                        "dataset": "AgiBot",
                        "split": split,
                        "episode_id": episode_id,
                        "mode": "llm_val",
                        "success": 1.0 if row["task_success"] else 0.0,
                        "task_progress": float(row["task_progress"]),
                        "mean_l2": float(row["mean_l2"]),
                        "l2step": float(row["step_alignment_l2_lt_0_1"]),
                    }
                )
    return pd.DataFrame(rows)


def exact_mcnemar(a: pd.Series, b: pd.Series) -> tuple[int, int, float]:
    gain = int(((a == 0) & (b == 1)).sum())
    loss = int(((a == 1) & (b == 0)).sum())
    discordant = gain + loss
    if discordant == 0:
        return gain, loss, 1.0
    p = float(binomtest(min(gain, loss), discordant, 0.5, alternative="two-sided").pvalue)
    return gain, loss, p


def paired_wilcoxon(a: pd.Series, b: pd.Series) -> float:
    if len(a) == 0 or (a == b).all():
        return 1.0
    try:
        return float(wilcoxon(a, b, alternative="two-sided", zero_method="wilcox").pvalue)
    except ValueError:
        return 1.0


def build_exact_tables(df: pd.DataFrame, dataset_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_specs = [
        ("success", "Success Rate", True, "McNemar exact"),
        ("task_progress", "Task Progress", True, "Wilcoxon paired"),
        ("mean_l2", "Mean L2", False, "Wilcoxon paired"),
        ("l2step", "L2<0.1 Step Rate", True, "Wilcoxon paired"),
    ]
    rows = []
    for split in SPLIT_ORDER:
        split_df = df[df["split"] == split]
        for base_mode, compare_mode in COMPARISONS:
            merged = (
                split_df[split_df["mode"] == base_mode]
                .merge(
                    split_df[split_df["mode"] == compare_mode],
                    on=["dataset", "split", "episode_id"],
                    suffixes=("_base", "_cmp"),
                )
                .sort_values("episode_id")
            )
            for metric, label, higher_is_better, test_name in metric_specs:
                base = merged[f"{metric}_base"]
                cmp_ = merged[f"{metric}_cmp"]
                if metric == "success":
                    gain, loss, p_value = exact_mcnemar(base, cmp_)
                else:
                    gain, loss = None, None
                    p_value = paired_wilcoxon(base, cmp_)
                delta = float(cmp_.mean() - base.mean())
                if not higher_is_better:
                    delta = -delta
                rows.append(
                    {
                        "dataset": dataset_name,
                        "split": split,
                        "comparison": f"{base_mode} -> {compare_mode}",
                        "metric": label,
                        "n_pairs": len(merged),
                        "baseline_value": float(base.mean()),
                        "compare_value": float(cmp_.mean()),
                        "effect_better_direction": delta,
                        "test": test_name,
                        "gain_pairs": gain,
                        "loss_pairs": loss,
                        "p_value": p_value,
                    }
                )
    out = pd.DataFrame(rows)
    out["p_value_bh"] = bh_adjust(out["p_value"].tolist())
    out["sig_bh"] = out["p_value_bh"].map(sig_label)
    display = out.copy()
    for col in ["baseline_value", "compare_value", "effect_better_direction", "p_value", "p_value_bh"]:
        display[col] = display[col].map(lambda v: round(float(v), 4))
    return out, display


def build_screenshot_sr_table() -> tuple[pd.DataFrame, pd.DataFrame]:
    table_df = pd.DataFrame(SCREENSHOT_TABLE_ROWS)
    rows = []
    for dataset in ["DROID", "AgiBot"]:
        for split in SPLIT_ORDER:
            split_df = table_df[(table_df["dataset"] == dataset) & (table_df["split"] == split)].copy()
            by_mode = {row["mode"]: row for row in split_df.to_dict("records")}
            for base_mode, compare_mode in COMPARISONS:
                base = by_mode[base_mode]
                cmp_ = by_mode[compare_mode]
                base_success = int(round(base["n"] * base["success_rate"]))
                cmp_success = int(round(cmp_["n"] * cmp_["success_rate"]))
                contingency = [
                    [cmp_success, cmp_["n"] - cmp_success],
                    [base_success, base["n"] - base_success],
                ]
                _, p_value = fisher_exact(contingency, alternative="two-sided")
                rows.append(
                    {
                        "dataset": dataset,
                        "split": split,
                        "comparison": f"{base_mode} -> {compare_mode}",
                        "n_base": int(base["n"]),
                        "n_compare": int(cmp_["n"]),
                        "baseline_successes": base_success,
                        "compare_successes": cmp_success,
                        "baseline_sr": float(base["success_rate"]),
                        "compare_sr": float(cmp_["success_rate"]),
                        "delta_sr_pp": float((cmp_["success_rate"] - base["success_rate"]) * 100.0),
                        "test": "Fisher exact (count-based)",
                        "p_value": float(p_value),
                    }
                )
    out = pd.DataFrame(rows)
    out["p_value_bh"] = bh_adjust(out["p_value"].tolist())
    out["sig_bh"] = out["p_value_bh"].map(sig_label)
    display = out.copy()
    for col in ["baseline_sr", "compare_sr", "delta_sr_pp", "p_value", "p_value_bh"]:
        display[col] = display[col].map(lambda v: round(float(v), 4))
    return out, display


def write_markdown_table(df: pd.DataFrame, path: Path) -> None:
    path.write_text(df.to_markdown(index=False) + "\n")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    droid_raw = load_droid_raw()
    droid_exact, droid_exact_display = build_exact_tables(droid_raw, "DROID")
    agibot_recoverable_raw = load_agibot_recoverable_raw()
    agibot_exact, agibot_exact_display = build_exact_tables(agibot_recoverable_raw, "AgiBot")
    sr_count, sr_count_display = build_screenshot_sr_table()

    droid_csv = OUTPUT_DIR / "droid_exact_paired_significance.csv"
    droid_md = OUTPUT_DIR / "droid_exact_paired_significance.md"
    agibot_csv = OUTPUT_DIR / "agibot_recoverable_paired_significance.csv"
    agibot_md = OUTPUT_DIR / "agibot_recoverable_paired_significance.md"
    sr_csv = OUTPUT_DIR / "screenshot_sr_count_significance.csv"
    sr_md = OUTPUT_DIR / "screenshot_sr_count_significance.md"
    summary_md = OUTPUT_DIR / "README.md"

    droid_exact_display.to_csv(droid_csv, index=False)
    write_markdown_table(droid_exact_display, droid_md)

    agibot_exact_display.to_csv(agibot_csv, index=False)
    write_markdown_table(agibot_exact_display, agibot_md)

    sr_count_display.to_csv(sr_csv, index=False)
    write_markdown_table(sr_count_display, sr_md)

    summary_lines = [
        "# DreamZero Significance Tables",
        "",
        "Files:",
        f"- Exact paired DROID significance: `{droid_csv.relative_to(REPO_ROOT)}` / `{droid_md.relative_to(REPO_ROOT)}`",
        f"- Recoverable paired AgiBot significance: `{agibot_csv.relative_to(REPO_ROOT)}` / `{agibot_md.relative_to(REPO_ROOT)}`",
        f"- Screenshot-level SR significance (count-based for both DROID and AgiBot): `{sr_csv.relative_to(REPO_ROOT)}` / `{sr_md.relative_to(REPO_ROOT)}`",
        "",
        "Notes:",
        "- `droid_exact_paired_significance.*` uses exact episode-level raw data from `evaluation_results_dualsystem/droid_3way_selected150.json`.",
        "- `agibot_recoverable_paired_significance.*` uses exact split raw for `original`/`llm_dual` from `Agi_DualBetter_L1/L2/L3_50_results.json`, plus recoverable overlapping `llm_val` rows from `agibot_3way_compare.json` on the same episode ids.",
        "- The DROID raw file gives `L3/original` success rate `20%`; the screenshot shows `22%`. The exact-paired table uses the raw file as source of truth.",
        "- The recoverable AgiBot table is useful for paired testing, but it is not guaranteed to be identical to the screenshot table because the exact screenshot-era 3-mode AgiBot split file is not currently recoverable from the repo/archive.",
        "- `screenshot_sr_count_significance.*` uses the 50-episode success-rate cells shown in the screenshot tables and applies Fisher exact tests on success counts.",
        "- The screenshot-level SR table is conservative and does not use episode pairing, because the exact paired 3-mode Dreamzero-AgiBot split file is not currently recoverable from the repo/archive.",
    ]
    summary_md.write_text("\n".join(summary_lines) + "\n")


if __name__ == "__main__":
    main()
