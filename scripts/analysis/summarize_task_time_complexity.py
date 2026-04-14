#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "20260331" / "analysis"

SPLIT_ORDER = ["L1", "L2", "L3"]

DROID_RESULTS = REPO_ROOT / "evaluation_results_dualsystem" / "droid_3way_selected150.json"
DREAMZERO_AGIBOT_MANIFESTS = {
    "L1": REPO_ROOT / "data" / "Agi_DualBetter_L1_50" / "meta" / "manifest.json",
    "L2": REPO_ROOT / "data" / "Agi_DualBetter_L2_50" / "meta" / "manifest.json",
    "L3": REPO_ROOT / "data" / "Agi_DualBetter_L3_50" / "meta" / "manifest.json",
}
DREAMDOJO_AGIBOT_MANIFESTS = {
    "L1": REPO_ROOT / "20260331" / "data" / "Agi_DreamDojo286_20260331_L1_50" / "meta" / "manifest.json",
    "L2": REPO_ROOT / "20260331" / "data" / "Agi_DreamDojo286_20260331_L2_50" / "meta" / "manifest.json",
    "L3": REPO_ROOT / "20260331" / "data" / "Agi_DreamDojo286_20260331_L3_50" / "meta" / "manifest.json",
}
EGODEX_MANIFEST = REPO_ROOT / "data" / "egodex_dreamdojo_easy400" / "meta" / "manifest.json"
EGODEX_INFO = REPO_ROOT / "data" / "egodex_eval_official" / "EgoDex_Eval" / "meta" / "info.json"


def tercile_slices(n: int) -> list[tuple[int, int]]:
    base = n // 3
    rem = n % 3
    counts = [base, base, base + rem]
    out = []
    start = 0
    for count in counts:
        end = start + count
        out.append((start, end))
        start = end
    return out


def summarize_values(dataset: str, split: str, values: list[float], source_note: str) -> dict:
    arr = np.asarray(values, dtype=float)
    return {
        "dataset": dataset,
        "split": split,
        "n": int(arr.size),
        "mean_sec": float(arr.mean()),
        "median_sec": float(np.median(arr)),
        "min_sec": float(arr.min()),
        "max_sec": float(arr.max()),
        "source_note": source_note,
    }


def load_dreamzero_droid() -> list[dict]:
    payload = json.loads(DROID_RESULTS.read_text())
    unique: dict[str, tuple[str, float]] = {}
    for row in payload["results"]:
        eid = str(row["episode_id"])
        if eid not in unique:
            unique[eid] = (str(row["dro_split"]), float(row["ep_len"]) / 15.0)

    rows = []
    for split in SPLIT_ORDER:
        values = [duration for s, duration in unique.values() if s == split]
        rows.append(
            summarize_values(
                dataset="DreamZero-DROID",
                split=split,
                values=values,
                source_note="Archived table split durations from droid_3way_selected150.json (ep_len / 15).",
            )
        )
    return rows


def load_manifest_family(dataset: str, manifests: dict[str, Path], source_note: str) -> list[dict]:
    rows = []
    for split in SPLIT_ORDER:
        payload = json.loads(manifests[split].read_text())
        values = [float(ep["duration_sec"]) for ep in payload["episodes"]]
        rows.append(summarize_values(dataset=dataset, split=split, values=values, source_note=source_note))
    return rows


def load_dreamdojo_egodex() -> list[dict]:
    manifest = json.loads(EGODEX_MANIFEST.read_text())
    info = json.loads(EGODEX_INFO.read_text())
    fps = float(info["fps"])
    durations = sorted(float(ep["num_frames"]) / fps for ep in manifest["episodes"])

    rows = []
    for split, (start, end) in zip(SPLIT_ORDER, tercile_slices(len(durations))):
        values = durations[start:end]
        rows.append(
            summarize_values(
                dataset="DreamDojo-EgoDex",
                split=split,
                values=values,
                source_note=(
                    "Recovered from the DreamDojo EgoDex easy400 source manifest by sorting all episodes "
                    f"by num_frames / {fps:.0f}fps and splitting into three duration terciles."
                ),
            )
        )
    return rows


def add_growth_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["split"] = pd.Categorical(df["split"], categories=SPLIT_ORDER, ordered=True)
    df = df.sort_values(["dataset", "split"]).reset_index(drop=True)

    delta_prev = []
    ratio_prev = []
    ratio_l1 = []
    complexity_index = []
    for _, group in df.groupby("dataset", sort=False):
        means = group["mean_sec"].tolist()
        l1_mean = means[0]
        for idx, mean_val in enumerate(means):
            if idx == 0:
                delta_prev.append(np.nan)
                ratio_prev.append(np.nan)
            else:
                prev = means[idx - 1]
                delta_prev.append(mean_val - prev)
                ratio_prev.append(mean_val / prev if prev > 0 else np.nan)
            ratio_l1.append(mean_val / l1_mean if l1_mean > 0 else np.nan)
            complexity_index.append(mean_val / l1_mean if l1_mean > 0 else np.nan)

    df["delta_vs_prev_sec"] = delta_prev
    df["ratio_vs_prev"] = ratio_prev
    df["ratio_vs_L1"] = ratio_l1
    df["complexity_index_L1eq1"] = complexity_index
    return df


def format_float(value: float, digits: int = 2) -> str:
    if pd.isna(value):
        return "-"
    return f"{value:.{digits}f}"


def build_markdown(df: pd.DataFrame) -> str:
    lines = [
        "# Task-Time Complexity Summary",
        "",
        "Longer episode duration is used here as a task-complexity proxy.",
        "",
        "| Dataset | Split | N | Mean (s) | Median (s) | Range (s) | Delta vs prev (s) | x prev | x L1 |",
        "| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |",
    ]
    for row in df.itertuples(index=False):
        lines.append(
            "| {} | {} | {} | {} | {} | {}-{} | {} | {} | {} |".format(
                row.dataset,
                row.split,
                int(row.n),
                format_float(row.mean_sec, 2),
                format_float(row.median_sec, 2),
                format_float(row.min_sec, 2),
                format_float(row.max_sec, 2),
                format_float(row.delta_vs_prev_sec, 2),
                format_float(row.ratio_vs_prev, 2),
                format_float(row.ratio_vs_L1, 2),
            )
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `DreamZero-DROID`: archived table split durations from `droid_3way_selected150.json`, using `ep_len / 15`.",
            "- `DreamZero-AgiBot`: archived `Agi_DualBetter_L1/L2/L3_50` manifests.",
            "- `DreamDojo-AgiBot`: the current `20260331` submission manifests.",
            "- `DreamDojo-EgoDex`: exact archived selected-50 split files are missing in the workspace, so durations are recovered from the readable `egodex_dreamdojo_easy400` source manifest by 20 FPS duration terciles.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    rows.extend(load_dreamzero_droid())
    rows.extend(
        load_manifest_family(
            dataset="DreamZero-AgiBot",
            manifests=DREAMZERO_AGIBOT_MANIFESTS,
            source_note="Archived Agi_DualBetter L1/L2/L3 manifests.",
        )
    )
    rows.extend(
        load_manifest_family(
            dataset="DreamDojo-AgiBot",
            manifests=DREAMDOJO_AGIBOT_MANIFESTS,
            source_note="20260331 DreamDojo AgiBot submission manifests.",
        )
    )
    rows.extend(load_dreamdojo_egodex())

    df = add_growth_columns(pd.DataFrame(rows))
    df["split"] = df["split"].astype(str)

    out_tsv = OUT_DIR / "task_time_complexity_summary.tsv"
    out_md = OUT_DIR / "task_time_complexity_summary.md"
    df.to_csv(out_tsv, sep="\t", index=False)
    out_md.write_text(build_markdown(df), encoding="utf-8")


if __name__ == "__main__":
    main()
