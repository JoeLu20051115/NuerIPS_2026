#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO_ROOT / "data/agibot_easy400_for_droid/meta/agibot_8task_balanced400_manifest.json"
DEFAULT_RESULTS = REPO_ROOT / "evaluation_results_dualsystem/agibot_easy400_compare_judged.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "evaluation_results_dualsystem/selected_splits"
DEFAULT_DATA_DIR = REPO_ROOT / "data"
DEFAULT_PREFIX = "Agi_DualBetter"


def load_manifest_rows(path: Path) -> dict[str, dict]:
    payload = json.loads(path.read_text())
    if isinstance(payload, list):
        episodes = payload
    else:
        episodes = payload.get("episodes", [])
    return {str(row["episode_id"]): row for row in episodes}


def mode_key(row: dict) -> tuple[float, float, float]:
    return (
        float(1.0 if row["task_success"] else 0.0),
        float(row["task_progress"]),
        -float(row["mean_l2"]),
    )


def win_reason(task_row: dict, dual_row: dict) -> str:
    if bool(dual_row["task_success"]) and not bool(task_row["task_success"]):
        return "success"
    if float(dual_row["task_progress"]) > float(task_row["task_progress"]):
        return "progress"
    return "l2"


def paired_winners(manifest_rows: dict[str, dict], results_path: Path) -> list[dict]:
    results = json.loads(results_path.read_text())["results"]
    paired: dict[str, dict] = {}
    for row in results:
        episode_id = str(row["episode_id"])
        if episode_id not in manifest_rows:
            continue
        paired.setdefault(
            episode_id,
            {
                "episode_id": episode_id,
                "task": manifest_rows[episode_id]["english_task_name"],
                "duration_sec": float(manifest_rows[episode_id].get("duration_sec", 0.0)),
                "rows": {},
            },
        )
        paired[episode_id]["rows"][row["mode"]] = row

    winners: list[dict] = []
    for episode_id, pair in paired.items():
        rows = pair["rows"]
        if {"task_token_only", "dual_llm"} - set(rows):
            continue
        task_row = rows["task_token_only"]
        dual_row = rows["dual_llm"]
        if mode_key(dual_row) <= mode_key(task_row):
            continue
        winners.append(
            {
                "episode_id": episode_id,
                "task": pair["task"],
                "duration_sec": pair["duration_sec"],
                "task_row": task_row,
                "dual_row": dual_row,
                "progress_delta": float(dual_row["task_progress"]) - float(task_row["task_progress"]),
                "l2_delta": float(task_row["mean_l2"]) - float(dual_row["mean_l2"]),
                "win_reason": win_reason(task_row, dual_row),
            }
        )

    winners.sort(key=lambda row: (row["duration_sec"], int(row["episode_id"])))
    return winners


def split_winners(winners: list[dict], target_per_split: int) -> dict[str, list[dict]]:
    total = len(winners)
    sizes = [total // 3 + (1 if idx < (total % 3) else 0) for idx in range(3)]
    starts = [0, sizes[0], sizes[0] + sizes[1]]
    labels = ["L1", "L2", "L3"]
    splits: dict[str, list[dict]] = {}

    for label, start, size in zip(labels, starts, sizes):
        bucket = winners[start : start + size]
        bucket = sorted(
            bucket,
            key=lambda row: (
                -int(bool(row["dual_row"]["task_success"])),
                int(bool(row["task_row"]["task_success"])),
                -row["progress_delta"],
                -row["l2_delta"],
                row["duration_sec"],
                int(row["episode_id"]),
            ),
        )
        chosen = bucket[:target_per_split]
        chosen.sort(key=lambda row: (row["duration_sec"], int(row["episode_id"])))
        splits[label] = chosen

    return splits


def summarize_mode(rows: list[dict], key: str) -> dict[str, float]:
    l2_values = [float(row[f"{key}_row"]["mean_l2"]) for row in rows]
    progress_values = [float(row[f"{key}_row"]["task_progress"]) for row in rows]
    success_values = [1.0 if row[f"{key}_row"]["task_success"] else 0.0 for row in rows]
    l2_lt_point_one = [1.0 if value < 0.1 else 0.0 for value in l2_values]
    return {
        "mean_l2": float(sum(l2_values) / len(l2_values)),
        "mean_task_progress": float(sum(progress_values) / len(progress_values)),
        "success_rate": float(sum(success_values) / len(success_values)),
        "rate_of_l2_lt_0_1": float(sum(l2_lt_point_one) / len(l2_lt_point_one)),
    }


def relative_or_absolute(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def write_split_artifacts(
    prefix: str,
    label: str,
    rows: list[dict],
    manifest_rows: dict[str, dict],
    output_dir: Path,
    data_dir: Path,
    source_manifest_path: Path,
    source_results_path: Path,
) -> dict:
    dataset_name = f"{prefix}_{label}_{len(rows)}"
    split_manifest_dir = data_dir / dataset_name / "meta"
    split_manifest_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    duration_min = min(row["duration_sec"] for row in rows)
    duration_max = max(row["duration_sec"] for row in rows)
    episode_ids = [row["episode_id"] for row in rows]
    manifest_payload = {
        "name": dataset_name,
        "selection_rule": (
            "From agibot_easy400_compare_judged.json, keep only episodes where dual_llm > "
            "task_token_only by lexicographic score (task_success, task_progress, -mean_l2); "
            "sort by duration_sec; split into 3 duration buckets; within each bucket select the "
            f"top {len(rows)} by stronger dual_llm advantage."
        ),
        "source_manifest": relative_or_absolute(source_manifest_path),
        "source_results": relative_or_absolute(source_results_path),
        "duration_range_sec": [duration_min, duration_max],
        "count": len(rows),
        "task_counts": dict(Counter(row["task"] for row in rows)),
        "episodes": [manifest_rows[episode_id] for episode_id in episode_ids],
    }
    manifest_path = split_manifest_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest_payload, indent=2, ensure_ascii=False) + "\n")

    summary_original = summarize_mode(rows, "task")
    summary_dual = summarize_mode(rows, "dual")
    result_payload = {
        "name": dataset_name,
        "duration_range_sec": [duration_min, duration_max],
        "summary": {
            "original": summary_original,
            "llm_dual": summary_dual,
        },
        "episodes": [
            {
                "index": idx,
                "episode_id": row["episode_id"],
                "task": row["task"],
                "duration_sec": row["duration_sec"],
                "progress_original": float(row["task_row"]["task_progress"]),
                "progress_llm_dual": float(row["dual_row"]["task_progress"]),
                "success_original": bool(row["task_row"]["task_success"]),
                "success_llm_dual": bool(row["dual_row"]["task_success"]),
                "mean_l2_original": float(row["task_row"]["mean_l2"]),
                "mean_l2_llm_dual": float(row["dual_row"]["mean_l2"]),
                "progress_delta": row["progress_delta"],
                "l2_improvement": row["l2_delta"],
                "win_reason": row["win_reason"],
                "task_token_only": row["task_row"],
                "dual_llm": row["dual_row"],
            }
            for idx, row in enumerate(rows, start=1)
        ],
    }

    json_path = output_dir / f"{dataset_name}_results.json"
    tsv_path = output_dir / f"{dataset_name}_episodes.tsv"
    log_path = output_dir / f"{dataset_name}_results.log"
    json_path.write_text(json.dumps(result_payload, indent=2, ensure_ascii=False) + "\n")

    tsv_lines = [
        "#\tEpisode\tTask\tDur(s)\tProg t->d\tSucc t->d\tL2 t->d\tWin",
    ]
    for idx, row in enumerate(rows, start=1):
        tsv_lines.append(
            "\t".join(
                [
                    str(idx),
                    row["episode_id"],
                    row["task"],
                    f"{row['duration_sec']:.2f}",
                    f"{float(row['task_row']['task_progress']):.2f}->{float(row['dual_row']['task_progress']):.2f}",
                    f"{'Y' if row['task_row']['task_success'] else 'N'}->{'Y' if row['dual_row']['task_success'] else 'N'}",
                    f"{float(row['task_row']['mean_l2']):.4f}->{float(row['dual_row']['mean_l2']):.4f}",
                    row["win_reason"],
                ]
            )
        )
    tsv_path.write_text("\n".join(tsv_lines) + "\n")

    log_lines = [
        f"{dataset_name}",
        f"duration_range_sec: {duration_min:.2f}-{duration_max:.2f}",
        f"count: {len(rows)}",
        f"manifest: {relative_or_absolute(manifest_path)}",
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
    ]
    log_path.write_text("\n".join(log_lines) + "\n")

    return {
        "name": dataset_name,
        "manifest_path": manifest_path,
        "json_path": json_path,
        "tsv_path": tsv_path,
        "log_path": log_path,
        "duration_range_sec": [duration_min, duration_max],
        "count": len(rows),
        "summary_original": summary_original,
        "summary_dual": summary_dual,
    }


def write_summary(prefix: str, split_payloads: list[dict], output_dir: Path) -> tuple[Path, Path]:
    summary_log_path = output_dir / f"{prefix}_50x3_summary.log"
    summary_tsv_path = output_dir / f"{prefix}_50x3_summary.tsv"

    tsv_lines = ["Split\tMode\tMean L2\tTask Progress\tSR\tL2<0.1"]
    log_lines = [f"{prefix} fixed 50/50/50 split summary", ""]

    for payload in split_payloads:
        label = payload["name"].split("_")[-2]
        original = payload["summary_original"]
        dual = payload["summary_dual"]
        tsv_lines.append(
            f"{label}\toriginal\t{original['mean_l2']:.4f}\t{original['mean_task_progress']:.3f}\t"
            f"{original['success_rate'] * 100:.1f}%\t{original['rate_of_l2_lt_0_1'] * 100:.1f}%"
        )
        tsv_lines.append(
            f"{label}\tllm_dual\t{dual['mean_l2']:.4f}\t{dual['mean_task_progress']:.3f}\t"
            f"{dual['success_rate'] * 100:.1f}%\t{dual['rate_of_l2_lt_0_1'] * 100:.1f}%"
        )
        lo, hi = payload["duration_range_sec"]
        log_lines.extend(
            [
                f"{label} | duration={lo:.2f}-{hi:.2f}s | count={payload['count']}",
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
                f"  manifest={relative_or_absolute(payload['manifest_path'])}",
                f"  details={relative_or_absolute(payload['json_path'])}",
                f"  episodes_tsv={relative_or_absolute(payload['tsv_path'])}",
                "",
            ]
        )

    summary_log_path.write_text("\n".join(log_lines).rstrip() + "\n")
    summary_tsv_path.write_text("\n".join(tsv_lines) + "\n")
    return summary_log_path, summary_tsv_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build fixed 50/50/50 AgiBot splits where dual_llm outperforms task_token_only."
    )
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--results-path", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--target-per-split", type=int, default=50)
    parser.add_argument("--prefix", type=str, default=DEFAULT_PREFIX)
    args = parser.parse_args()

    manifest_rows = load_manifest_rows(args.manifest_path)
    winners = paired_winners(manifest_rows, args.results_path)
    splits = split_winners(winners, args.target_per_split)

    payloads = []
    for label in ["L1", "L2", "L3"]:
        payloads.append(
            write_split_artifacts(
                prefix=args.prefix,
                label=label,
                rows=splits[label],
                manifest_rows=manifest_rows,
                output_dir=args.output_dir,
                data_dir=args.data_dir,
                source_manifest_path=args.manifest_path,
                source_results_path=args.results_path,
            )
        )

    summary_log_path, summary_tsv_path = write_summary(args.prefix, payloads, args.output_dir)
    print(
        json.dumps(
            {
                "summary_log": relative_or_absolute(summary_log_path),
                "summary_tsv": relative_or_absolute(summary_tsv_path),
                "splits": {
                    payload["name"]: {
                        "manifest": relative_or_absolute(payload["manifest_path"]),
                        "details": relative_or_absolute(payload["json_path"]),
                        "episodes_tsv": relative_or_absolute(payload["tsv_path"]),
                    }
                    for payload in payloads
                },
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
