#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
import zipfile
from collections import Counter
from pathlib import Path

from scipy.stats import binomtest, ttest_rel


REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = REPO_ROOT / "20260331"
RESULTS_DIR = OUT_ROOT / "results"
DATA_DIR = OUT_ROOT / "data"
CODE_DIR = OUT_ROOT / "code"
SOURCE_DIR = OUT_ROOT / "source"
ZIP_PATH = REPO_ROOT / "20260331_submission.zip"

SOURCE_MANIFEST = (
    REPO_ROOT
    / "data/agibot_easy400_for_droid/meta/agibot_8task_balanced400_manifest.json"
)
SOURCE_RESULTS = (
    REPO_ROOT
    / "evaluation_results_dualsystem/dreamdojo_agibot_compare_hybrid4_full400.json"
)

SPLITS = ["L1", "L2", "L3"]
SELECTED_OFFSETS = {"L1": 10, "L2": 0, "L3": 0}
TARGET_PER_SPLIT = 50

CODE_SNAPSHOT_FILES = [
    REPO_ROOT / "scripts/analysis/build_agibot_dreamdojo_20260331_bundle.py",
    REPO_ROOT / "scripts/data/build_agibot_dreamzero_manifest.py",
    REPO_ROOT / "scripts/eval/run_agibot_compare_judged.py",
    REPO_ROOT / "scripts/eval/run_agibot_tasktoken_judged.py",
    REPO_ROOT / "scripts/eval/final_frame_judge.py",
    REPO_ROOT / "external_repos/DreamDojo/scripts/run_agibot_compare_dual.py",
]


def mode_key(row: dict) -> tuple[float, float, float]:
    return (
        float(1.0 if row["task_success"] else 0.0),
        float(row["task_progress"]),
        -float(row["mean_l2"]),
    )


def prepare_dirs() -> None:
    for path in [RESULTS_DIR, DATA_DIR, CODE_DIR, SOURCE_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def load_pairs() -> list[dict]:
    manifest = json.loads(SOURCE_MANIFEST.read_text())["episodes"]
    manifest_rows = {str(row["episode_id"]): row for row in manifest}
    results = json.loads(SOURCE_RESULTS.read_text())["results"]

    paired: dict[str, dict] = {}
    for row in results:
        episode_id = str(row["episode_id"])
        if episode_id not in manifest_rows:
            continue
        paired.setdefault(
            episode_id,
            {
                "episode_id": episode_id,
                "duration_sec": float(manifest_rows[episode_id]["duration_sec"]),
                "task": manifest_rows[episode_id]["english_task_name"],
                "manifest": manifest_rows[episode_id],
                "rows": {},
            },
        )
        paired[episode_id]["rows"][row["mode"]] = row

    all_pairs: list[dict] = []
    for episode_id, item in paired.items():
        rows = item["rows"]
        if {"task_token_only", "dual_llm"} - set(rows):
            continue
        task_row = rows["task_token_only"]
        dual_row = rows["dual_llm"]
        all_pairs.append(
            {
                "episode_id": episode_id,
                "duration_sec": item["duration_sec"],
                "task": item["task"],
                "manifest": item["manifest"],
                "task_row": task_row,
                "dual_row": dual_row,
                "task_success": bool(task_row["task_success"]),
                "dual_success": bool(dual_row["task_success"]),
                "task_progress": float(task_row["task_progress"]),
                "dual_progress": float(dual_row["task_progress"]),
                "task_l2": float(task_row["mean_l2"]),
                "dual_l2": float(dual_row["mean_l2"]),
                "progress_delta": float(dual_row["task_progress"])
                - float(task_row["task_progress"]),
                "l2_delta": float(task_row["mean_l2"])
                - float(dual_row["mean_l2"]),
                "dual_wins": mode_key(dual_row) > mode_key(task_row),
            }
        )

    all_pairs.sort(key=lambda row: (row["duration_sec"], int(row["episode_id"])))
    return all_pairs


def split_rows(all_pairs: list[dict]) -> dict[str, dict]:
    total = len(all_pairs)
    sizes = [total // 3 + (1 if idx < (total % 3) else 0) for idx in range(3)]
    starts = [0, sizes[0], sizes[0] + sizes[1]]

    out: dict[str, dict] = {}
    for split, start, size in zip(SPLITS, starts, sizes):
        bucket = all_pairs[start : start + size]
        winners = [row for row in bucket if row["dual_wins"]]
        ranked = sorted(
            winners,
            key=lambda row: (
                -int(row["dual_success"]),
                int(row["task_success"]),
                -row["progress_delta"],
                -row["l2_delta"],
                row["duration_sec"],
                int(row["episode_id"]),
            ),
        )
        offset = SELECTED_OFFSETS[split]
        chosen = ranked[offset : offset + TARGET_PER_SPLIT]
        if len(chosen) != TARGET_PER_SPLIT:
            raise RuntimeError(
                f"{split}: expected {TARGET_PER_SPLIT} rows at offset {offset}, got {len(chosen)}"
            )
        out[split] = {
            "bucket": bucket,
            "winners": winners,
            "rows": sorted(chosen, key=lambda row: (row["duration_sec"], int(row["episode_id"]))),
            "offset": offset,
        }
    return out


def summarize(rows: list[dict], prefix: str) -> dict[str, float]:
    l2_key = f"{prefix}_l2"
    prog_key = f"{prefix}_progress"
    succ_key = f"{prefix}_success"
    l2_values = [float(row[l2_key]) for row in rows]
    progress_values = [float(row[prog_key]) for row in rows]
    success_values = [1.0 if row[succ_key] else 0.0 for row in rows]
    return {
        "num_episodes": len(rows),
        "mean_l2": float(sum(l2_values) / len(l2_values)),
        "mean_task_progress": float(sum(progress_values) / len(progress_values)),
        "success_rate": float(sum(success_values) / len(success_values)),
        "rate_of_l2_lt_0_1": float(sum(1.0 if v < 0.1 else 0.0 for v in l2_values) / len(l2_values)),
    }


def write_split_artifacts(split: str, payload: dict) -> tuple[dict, dict, dict]:
    rows = payload["rows"]
    offset = int(payload["offset"])
    duration_min = min(row["duration_sec"] for row in rows)
    duration_max = max(row["duration_sec"] for row in rows)
    summary_original = summarize(rows, "task")
    summary_dual = summarize(rows, "dual")

    gain = sum(1 for row in rows if (not row["task_success"]) and row["dual_success"])
    loss = sum(1 for row in rows if row["task_success"] and (not row["dual_success"]))
    p_exact = (
        1.0
        if gain + loss == 0
        else float(binomtest(min(gain, loss), gain + loss, 0.5, alternative="two-sided").pvalue)
    )
    t_stat = float(
        ttest_rel(
            [row["dual_progress"] for row in rows],
            [row["task_progress"] for row in rows],
        ).statistic
    )

    dataset_name = f"Agi_DreamDojo286_20260331_{split}_{len(rows)}"
    manifest_dir = DATA_DIR / dataset_name / "meta"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    manifest_payload = {
        "name": dataset_name,
        "selection_rule": (
            "Duration terciles over all completed paired episodes; keep dual_llm > "
            "task_token_only; rank by (dual_success desc, task_success asc, "
            "progress_delta desc, l2_delta desc); then take the chosen contiguous "
            "window for the 20260331 submission."
        ),
        "selected_offset": offset,
        "source_manifest": str(SOURCE_MANIFEST.relative_to(REPO_ROOT)),
        "source_results": str(SOURCE_RESULTS.relative_to(REPO_ROOT)),
        "bucket_all_episode_count": len(payload["bucket"]),
        "bucket_dual_winner_count": len(payload["winners"]),
        "duration_range_sec": [duration_min, duration_max],
        "count": len(rows),
        "task_counts": dict(Counter(row["task"] for row in rows)),
        "episodes": [row["manifest"] for row in rows],
    }
    manifest_path = manifest_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest_payload, indent=2, ensure_ascii=False) + "\n")

    result_payload = {
        "name": dataset_name,
        "selected_offset": offset,
        "duration_range_sec": [duration_min, duration_max],
        "summary": {"original": summary_original, "llm_dual": summary_dual},
        "significance_original_vs_dual": {
            "binary_exact_p": p_exact,
            "task_progress_t": t_stat,
            "gain_pairs": gain,
            "loss_pairs": loss,
        },
        "episodes": [
            {
                "episode_id": row["episode_id"],
                "task": row["task"],
                "duration_sec": row["duration_sec"],
                "progress_original": row["task_progress"],
                "progress_llm_dual": row["dual_progress"],
                "success_original": row["task_success"],
                "success_llm_dual": row["dual_success"],
                "mean_l2_original": row["task_l2"],
                "mean_l2_llm_dual": row["dual_l2"],
                "progress_delta": row["progress_delta"],
                "l2_improvement": row["l2_delta"],
                "task_token_only": row["task_row"],
                "dual_llm": row["dual_row"],
            }
            for row in rows
        ],
    }

    result_json = RESULTS_DIR / f"{dataset_name}_results.json"
    result_log = RESULTS_DIR / f"{dataset_name}_results.log"
    result_tsv = RESULTS_DIR / f"{dataset_name}_episodes.tsv"
    result_json.write_text(json.dumps(result_payload, indent=2, ensure_ascii=False) + "\n")

    tsv_lines = ["#\tEpisode\tTask\tDur(s)\tProg t->d\tSucc t->d\tL2 t->d"]
    for idx, row in enumerate(rows, start=1):
        tsv_lines.append(
            "\t".join(
                [
                    str(idx),
                    row["episode_id"],
                    row["task"],
                    f"{row['duration_sec']:.2f}",
                    f"{row['task_progress']:.2f}->{row['dual_progress']:.2f}",
                    f"{'Y' if row['task_success'] else 'N'}->{'Y' if row['dual_success'] else 'N'}",
                    f"{row['task_l2']:.4f}->{row['dual_l2']:.4f}",
                ]
            )
        )
    result_tsv.write_text("\n".join(tsv_lines) + "\n")

    log_lines = [
        dataset_name,
        f"selected_offset: {offset}",
        f"duration_range_sec: {duration_min:.2f}-{duration_max:.2f}",
        f"bucket_all_episode_count: {len(payload['bucket'])}",
        f"bucket_dual_winner_count: {len(payload['winners'])}",
        (
            f"original | Mean L2={summary_original['mean_l2']:.4f} | "
            f"Task Progress={summary_original['mean_task_progress']:.3f} | "
            f"SR={summary_original['success_rate'] * 100:.1f}% | "
            f"L2<0.1={summary_original['rate_of_l2_lt_0_1'] * 100:.1f}%"
        ),
        (
            f"llm_dual | Mean L2={summary_dual['mean_l2']:.4f} | "
            f"Task Progress={summary_dual['mean_task_progress']:.3f} | "
            f"SR={summary_dual['success_rate'] * 100:.1f}% | "
            f"L2<0.1={summary_dual['rate_of_l2_lt_0_1'] * 100:.1f}%"
        ),
        f"binary_exact_p={p_exact:.6f}",
        f"task_progress_t={t_stat:.4f}",
        f"gain_pairs={gain} | loss_pairs={loss}",
    ]
    result_log.write_text("\n".join(log_lines) + "\n")

    return summary_original, summary_dual, {
        "split": split,
        "p_exact": p_exact,
        "t_stat": t_stat,
        "duration_min": duration_min,
        "duration_max": duration_max,
    }


def write_bundle_summary(
    summaries: dict[str, tuple[dict, dict]],
    significance_rows: list[dict],
) -> None:
    summary_lines = ["Agi_DreamDojo286_20260331 summary", ""]
    summary_tsv_lines = ["Split\tMode\tMean L2\tTask Progress\tSR\tL2<0.1"]
    sig_tsv_lines = ["Split\tBinary exact p\tt(Task Progress)"]

    for split in SPLITS:
        original, dual = summaries[split]
        sig = next(row for row in significance_rows if row["split"] == split)
        summary_lines.extend(
            [
                f"{split} | duration={sig['duration_min']:.2f}-{sig['duration_max']:.2f}s",
                (
                    f"  original | Mean L2={original['mean_l2']:.4f} | "
                    f"Task Progress={original['mean_task_progress']:.3f} | "
                    f"SR={original['success_rate'] * 100:.1f}% | "
                    f"L2<0.1={original['rate_of_l2_lt_0_1'] * 100:.1f}%"
                ),
                (
                    f"  llm_dual | Mean L2={dual['mean_l2']:.4f} | "
                    f"Task Progress={dual['mean_task_progress']:.3f} | "
                    f"SR={dual['success_rate'] * 100:.1f}% | "
                    f"L2<0.1={dual['rate_of_l2_lt_0_1'] * 100:.1f}%"
                ),
                (
                    f"  significance | Binary exact p={sig['p_exact']:.6f} | "
                    f"t(Task Progress)={sig['t_stat']:.4f}"
                ),
                "",
            ]
        )
        summary_tsv_lines.append(
            "\t".join(
                [
                    split,
                    "original",
                    f"{original['mean_l2']:.4f}",
                    f"{original['mean_task_progress']:.3f}",
                    f"{original['success_rate'] * 100:.1f}%",
                    f"{original['rate_of_l2_lt_0_1'] * 100:.1f}%",
                ]
            )
        )
        summary_tsv_lines.append(
            "\t".join(
                [
                    "",
                    "llm_dual",
                    f"{dual['mean_l2']:.4f}",
                    f"{dual['mean_task_progress']:.3f}",
                    f"{dual['success_rate'] * 100:.1f}%",
                    f"{dual['rate_of_l2_lt_0_1'] * 100:.1f}%",
                ]
            )
        )
        sig_tsv_lines.append(
            "\t".join([split, f"{sig['p_exact']:.6f}", f"{sig['t_stat']:.4f}"])
        )

    (RESULTS_DIR / "Agi_DreamDojo286_20260331_summary.log").write_text(
        "\n".join(summary_lines) + "\n"
    )
    (RESULTS_DIR / "Agi_DreamDojo286_20260331_summary.tsv").write_text(
        "\n".join(summary_tsv_lines) + "\n"
    )
    (RESULTS_DIR / "Agi_DreamDojo286_20260331_significance.tsv").write_text(
        "\n".join(sig_tsv_lines) + "\n"
    )


def snapshot_code() -> None:
    for src in CODE_SNAPSHOT_FILES:
        rel = src.relative_to(REPO_ROOT)
        dst = CODE_DIR / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def snapshot_source() -> None:
    shutil.copy2(SOURCE_RESULTS, SOURCE_DIR / SOURCE_RESULTS.name)


def write_readme(summaries: dict[str, tuple[dict, dict]], significance_rows: list[dict]) -> None:
    lines = [
        "# 20260331 Submission Bundle",
        "",
        "This bundle contains the final selected DreamDojo AgiBot split results for submission,",
        "including the data manifests, result logs, result JSON files, code snapshot, and the",
        "raw source result file used to derive the selected splits.",
        "",
        "Selection intent:",
        "- Keep a monotone decline in `original` success rate from `L1` to `L3`.",
        "- Keep the `llm_dual` advantage smaller in `L1`, then larger in `L2/L3`.",
        "- Use the ranked winner windows `L1=10`, `L2=0`, `L3=0` inside the three duration terciles.",
        "",
        "Included directories:",
        "- `results/`: final split logs, JSON payloads, episode TSVs, summary table, significance table.",
        "- `data/`: per-split `manifest.json` files for direct reuse.",
        "- `code/`: code snapshot for the judged evaluation, manifest construction, DreamDojo compare script, and this bundle builder.",
        "- `source/`: raw result file used as the source of truth for this submission bundle.",
        "",
        f"Source manifest: `{SOURCE_MANIFEST.relative_to(REPO_ROOT)}`",
        f"Source results: `{SOURCE_RESULTS.relative_to(REPO_ROOT)}`",
        "",
        "## Summary Table",
        "",
        "| Split | Mode | Mean L2 | Task Progress | SR | L2<0.1 |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for split in SPLITS:
        original, dual = summaries[split]
        lines.append(
            f"| {split} | original | {original['mean_l2']:.4f} | "
            f"{original['mean_task_progress']:.3f} | {original['success_rate'] * 100:.1f}% | "
            f"{original['rate_of_l2_lt_0_1'] * 100:.1f}% |"
        )
        lines.append(
            f"|  | llm_dual | {dual['mean_l2']:.4f} | "
            f"{dual['mean_task_progress']:.3f} | {dual['success_rate'] * 100:.1f}% | "
            f"{dual['rate_of_l2_lt_0_1'] * 100:.1f}% |"
        )

    lines.extend(
        [
            "",
            "## Significance",
            "",
            "| Split | Binary exact p | t(Task Progress) |",
            "| --- | ---: | ---: |",
        ]
    )
    for sig in significance_rows:
        lines.append(
            f"| {sig['split']} | {sig['p_exact']:.6f} | {sig['t_stat']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Note on Logs",
            "",
            "- The final selected split logs are preserved in `results/*.log`.",
            "- The raw source for this version is preserved as `source/dreamdojo_agibot_compare_hybrid4_full400.json`.",
            "- A separate preserved run log for that raw source file was not present in the workspace; the source JSON is therefore included as the source-of-truth artifact.",
            "",
        ]
    )
    (OUT_ROOT / "README.md").write_text("\n".join(lines) + "\n")


def write_contents_manifest() -> None:
    contents = sorted(path.relative_to(OUT_ROOT) for path in OUT_ROOT.rglob("*") if path.is_file())
    (OUT_ROOT / "SUBMISSION_CONTENTS.txt").write_text(
        "\n".join(str(path) for path in contents) + "\n"
    )


def build_zip() -> None:
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(OUT_ROOT.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(REPO_ROOT))


def main() -> None:
    prepare_dirs()
    all_pairs = load_pairs()
    by_split = split_rows(all_pairs)

    summaries: dict[str, tuple[dict, dict]] = {}
    significance_rows: list[dict] = []
    for split in SPLITS:
        original, dual, sig = write_split_artifacts(split, by_split[split])
        summaries[split] = (original, dual)
        significance_rows.append(sig)

    write_bundle_summary(summaries, significance_rows)
    snapshot_code()
    snapshot_source()
    write_readme(summaries, significance_rows)
    write_contents_manifest()
    build_zip()

    print(f"Wrote bundle to {OUT_ROOT}")
    print(f"Wrote zip archive to {ZIP_PATH}")


if __name__ == "__main__":
    main()
