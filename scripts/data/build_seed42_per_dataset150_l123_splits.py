#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import random
import zipfile
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any

import h5py
import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


REPO_ROOT = Path(__file__).resolve().parents[2]
SEED = 42
LEVELS = ["L1", "L2", "L3"]
LEVEL_QUOTA = 50
OUTPUT_ROOT = REPO_ROOT / "data/seed42_per_dataset150_l123"
SUMMARY_PATH = REPO_ROOT / "analysis_outputs/seed42_per_dataset150_l123/summary.md"


def set_global_seed(seed: int) -> dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    status = {
        "seed": seed,
        "torch_available": torch is not None,
        "cuda_seeded": False,
    }
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            status["cuda_seeded"] = True
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return status


def canonical_task(tasks: list[str] | None, fallback: str = "<missing>") -> str:
    for task in tasks or []:
        normalized = " ".join((task or "").split()).strip()
        if normalized and normalized.lower() != "not provided":
            return normalized
    normalized_fallback = " ".join(fallback.split()).strip()
    return normalized_fallback or "<missing>"


def compute_cutoffs(step_counts: list[int]) -> tuple[int, int]:
    ordered = sorted(step_counts)
    q1 = ordered[len(ordered) // 3]
    q2 = ordered[(2 * len(ordered)) // 3]
    return q1, q2


def assign_level(step_count: int, q1: int, q2: int) -> str:
    if step_count <= q1:
        return "L1"
    if step_count <= q2:
        return "L2"
    return "L3"


def make_rng(*parts: str) -> random.Random:
    return random.Random("::".join([str(SEED), *parts]))


def sample_with_diversity(candidates: list[dict[str, Any]], *, dataset_name: str, level: str, quota: int) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        grouped[row["sample_group"]].append(row)

    group_keys = sorted(grouped)
    group_rng = make_rng(dataset_name, level, "group-order")
    group_rng.shuffle(group_keys)

    queues: dict[str, deque[dict[str, Any]]] = {}
    for group_key in group_keys:
        rows = sorted(grouped[group_key], key=lambda row: row["uid"])
        row_rng = make_rng(dataset_name, level, group_key)
        row_rng.shuffle(rows)
        queues[group_key] = deque(rows)

    selected: list[dict[str, Any]] = []
    while len(selected) < quota and any(queues[group_key] for group_key in group_keys):
        progressed = False
        for group_key in group_keys:
            if queues[group_key]:
                selected.append(queues[group_key].popleft())
                progressed = True
                if len(selected) >= quota:
                    break
        if not progressed:
            break
    return selected


def droid_relative_paths(info: dict[str, Any], episode_index: int) -> tuple[Path, dict[str, Path]]:
    dataset_root = REPO_ROOT / "data/droid_lerobot"
    episode_chunk = episode_index // int(info["chunks_size"])
    parquet_path = dataset_root / info["data_path"].format(
        episode_chunk=episode_chunk,
        episode_index=episode_index,
    )
    video_paths: dict[str, Path] = {}
    for key, feature in info["features"].items():
        if feature.get("dtype") != "video":
            continue
        video_paths[key] = dataset_root / info["video_path"].format(
            episode_chunk=episode_chunk,
            episode_index=episode_index,
            video_key=key,
        )
    return parquet_path, video_paths


def load_droid_candidates() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dataset_root = REPO_ROOT / "data/droid_lerobot"
    info = json.loads((dataset_root / "meta/info.json").read_text())

    raw_pool = 0
    dropped = Counter()
    candidates: list[dict[str, Any]] = []

    for line in (dataset_root / "meta/episodes.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        raw_pool += 1
        record = json.loads(line)
        task_text = canonical_task(record.get("tasks"))
        if task_text == "<missing>":
            dropped["missing_task"] += 1
            continue
        if record.get("success") is not True:
            dropped["not_success"] += 1
            continue
        step_count = int(record["length"])
        if step_count < 10:
            dropped["too_short"] += 1
            continue

        episode_index = int(record["episode_index"])
        parquet_path, video_paths = droid_relative_paths(info, episode_index)
        if not parquet_path.exists():
            dropped["missing_parquet"] += 1
            continue
        if any(not path.exists() for path in video_paths.values()):
            dropped["missing_sensor"] += 1
            continue

        candidates.append(
            {
                "uid": f"droid:{episode_index}",
                "dataset": "DROID",
                "source_split": "public_pool_only",
                "source_note": "No public official DROID val/test split exists locally, so the official public pool is used as the closest substitute.",
                "episode_index": episode_index,
                "task_text": task_text,
                "all_tasks": record.get("tasks") or [],
                "step_count": step_count,
                "sample_group": task_text,
                "success_flag": True,
                "relative_parquet_path": str(parquet_path.relative_to(REPO_ROOT)),
                "relative_video_paths": {key: str(path.relative_to(REPO_ROOT)) for key, path in video_paths.items()},
            }
        )

    stats = {
        "raw_pool": raw_pool,
        "eligible_after_cleaning": len(candidates),
        "dropped": dict(dropped),
        "source_isolation": "Unavailable: local DROID metadata exposes only the public pool, not a public non-train split.",
        "success_rule": "Keep success == True from data/droid_lerobot/meta/episodes.jsonl.",
        "validity_rule": "Require step_count >= 10 plus parquet and all declared videos on disk.",
    }
    return candidates, stats


def agibot_step_count(h5_path: Path) -> int:
    with h5py.File(h5_path, "r") as h5:
        return int(h5["timestamp"].shape[0])


def load_agibot_candidates() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    val_root = REPO_ROOT / "data/agibot_challenge_2026/val"
    required = {
        "frame.png",
        "head_color.mp4",
        "head_extrinsic_params_aligned.json",
        "head_intrinsic_params.json",
        "proprio_stats.h5",
    }

    raw_pool = 0
    dropped = Counter()
    candidates: list[dict[str, Any]] = []

    for episode_dir in sorted(p for p in val_root.iterdir() if p.is_dir()):
        raw_pool += 1
        names = {p.name for p in episode_dir.iterdir() if p.is_file()}
        if not required.issubset(names):
            dropped["missing_sensor"] += 1
            continue
        step_count = agibot_step_count(episode_dir / "proprio_stats.h5")
        if step_count < 10:
            dropped["too_short"] += 1
            continue

        task_id, episode_id, segment_id = episode_dir.name.split("-")
        candidates.append(
            {
                "uid": f"agibot:{episode_dir.name}",
                "dataset": "AgiBot",
                "source_split": "official_val_only",
                "source_note": "Use the official AgiBot Challenge 2026 val split only; local official test files are not populated.",
                "task_id": task_id,
                "episode_id": episode_id,
                "segment_id": segment_id,
                "task_text": f"task_id_{task_id}",
                "step_count": step_count,
                "sample_group": task_id,
                "success_flag": None,
                "relative_episode_dir": str(episode_dir.relative_to(REPO_ROOT)),
            }
        )

    stats = {
        "raw_pool": raw_pool,
        "eligible_after_cleaning": len(candidates),
        "dropped": dict(dropped),
        "source_isolation": "Satisfied: use only the official AgiBot 2026 val split; train is never touched.",
        "success_rule": "No explicit success/reward field is provided in the official val files, so only non-train, sensor-complete episodes are retained.",
        "validity_rule": "Require step_count >= 10 and the full five-file episode bundle.",
    }
    return candidates, stats


def egodex_step_count_from_bytes(data: bytes) -> int:
    with h5py.File(io.BytesIO(data), "r") as h5:
        for candidate_key in ["transforms/camera", "confidences/hip"]:
            if candidate_key in h5:
                return int(h5[candidate_key].shape[0])

        found = {"length": None}

        def visit(name: str, obj: Any) -> None:
            if found["length"] is not None:
                return
            shape = getattr(obj, "shape", ())
            if shape and len(shape) >= 1:
                found["length"] = int(shape[0])

        h5.visititems(visit)
        if found["length"] is None:
            raise ValueError("No sequence dataset found in EgoDex HDF5.")
        return found["length"]


def load_egodex_candidates() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    zip_path = REPO_ROOT / "data/egodex_test/test.zip"
    raw_pool = 0
    dropped = Counter()
    candidates: list[dict[str, Any]] = []

    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
        h5_entries = sorted(name for name in names if name.endswith(".hdf5"))
        raw_pool = len(h5_entries)
        for h5_entry in h5_entries:
            mp4_entry = h5_entry[:-5] + ".mp4"
            if mp4_entry not in names:
                dropped["missing_sensor"] += 1
                continue

            with zf.open(h5_entry) as f:
                h5_bytes = f.read()
            step_count = egodex_step_count_from_bytes(h5_bytes)
            if step_count < 10:
                dropped["too_short"] += 1
                continue

            task_group = Path(h5_entry).parent.name
            episode_id = Path(h5_entry).stem
            candidates.append(
                {
                    "uid": f"egodex:{task_group}:{episode_id}",
                    "dataset": "EgoDex",
                    "source_split": "official_test_zip",
                    "source_note": "Use the official EgoDex test.zip pool.",
                    "task_group": task_group,
                    "episode_id": episode_id,
                    "task_text": task_group,
                    "step_count": step_count,
                    "sample_group": task_group,
                    "success_flag": None,
                    "relative_zip_path": str(zip_path.relative_to(REPO_ROOT)),
                    "zip_h5_path": h5_entry,
                    "zip_mp4_path": mp4_entry,
                }
            )

    stats = {
        "raw_pool": raw_pool,
        "eligible_after_cleaning": len(candidates),
        "dropped": dict(dropped),
        "source_isolation": "Satisfied: use only the official EgoDex test.zip pool.",
        "success_rule": "The official test zip does not provide explicit success/reward labels, so all sensor-complete official test episodes are retained.",
        "validity_rule": "Require paired .hdf5/.mp4 entries and step_count >= 10.",
    }
    return candidates, stats


def build_dataset_manifest(dataset_name: str, candidates: list[dict[str, Any]], cleaning_stats: dict[str, Any]) -> dict[str, Any]:
    q1, q2 = compute_cutoffs([row["step_count"] for row in candidates])
    level_counts = Counter()
    enriched: list[dict[str, Any]] = []
    for row in candidates:
        payload = dict(row)
        payload["level"] = assign_level(payload["step_count"], q1, q2)
        level_counts[payload["level"]] += 1
        enriched.append(payload)

    selected: dict[str, list[dict[str, Any]]] = {}
    selected_task_counts: dict[str, dict[str, int]] = {}
    for level in LEVELS:
        pool = [row for row in enriched if row["level"] == level]
        if len(pool) < LEVEL_QUOTA:
            raise SystemExit(f"{dataset_name} {level} has {len(pool)} candidates, fewer than quota {LEVEL_QUOTA}.")
        chosen = sample_with_diversity(pool, dataset_name=dataset_name, level=level, quota=LEVEL_QUOTA)
        selected[level] = chosen
        selected_task_counts[level] = dict(Counter(row["sample_group"] for row in chosen))

    return {
        "dataset": dataset_name,
        "seed": SEED,
        "level_quotas": {level: LEVEL_QUOTA for level in LEVELS},
        "cleaning": cleaning_stats,
        "level_cutoffs": {
            "q33_max": q1,
            "q66_max": q2,
            "candidate_counts": dict(level_counts),
        },
        "selected_counts": {level: len(selected[level]) for level in LEVELS},
        "selected_unique_groups": {level: len(selected_task_counts[level]) for level in LEVELS},
        "selected_group_counts": selected_task_counts,
        "selected": selected,
    }


def write_dataset_outputs(dataset_slug: str, manifest: dict[str, Any], output_root: Path) -> None:
    meta_root = output_root / dataset_slug / "meta"
    meta_root.mkdir(parents=True, exist_ok=True)
    manifest_path = meta_root / "manifest.json"
    jsonl_path = meta_root / "episodes.jsonl"

    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    with jsonl_path.open("w", encoding="utf-8") as f:
        for level in LEVELS:
            for row in manifest["selected"][level]:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_summary(manifests: list[dict[str, Any]], seed_status: dict[str, Any]) -> str:
    lines = [
        "# Seed-42 Per-Dataset 150 Splits",
        "",
        f"- Seed: `{seed_status['seed']}`",
        "- Target per dataset: `150` episodes.",
        "- Split per dataset: `L1=50`, `L2=50`, `L3=50`.",
        "- Level definition: per-dataset internal step-count tertiles.",
        "",
    ]
    for manifest in manifests:
        lines.extend(
            [
                f"## {manifest['dataset']}",
                "",
                f"- Cleaning: `{manifest['cleaning']}`",
                f"- Cutoffs: `{manifest['level_cutoffs']}`",
                f"- Selected counts: `{manifest['selected_counts']}`",
                f"- Unique task groups in selected split: `{manifest['selected_unique_groups']}`",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build per-dataset 150-episode Seed-42 L1/L2/L3 splits.")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--summary-path", type=Path, default=SUMMARY_PATH)
    args = parser.parse_args()

    seed_status = set_global_seed(SEED)

    droid_candidates, droid_stats = load_droid_candidates()
    agibot_candidates, agibot_stats = load_agibot_candidates()
    egodex_candidates, egodex_stats = load_egodex_candidates()

    manifests = [
        build_dataset_manifest("DROID", droid_candidates, droid_stats),
        build_dataset_manifest("AgiBot", agibot_candidates, agibot_stats),
        build_dataset_manifest("EgoDex", egodex_candidates, egodex_stats),
    ]

    args.output_root.mkdir(parents=True, exist_ok=True)
    slug_map = {"DROID": "droid", "AgiBot": "agibot", "EgoDex": "egodex"}
    overview = {
        "seed_status": seed_status,
        "datasets": {manifest["dataset"]: {
            "manifest_path": str((args.output_root / slug_map[manifest["dataset"]] / "meta/manifest.json").relative_to(REPO_ROOT)),
            "selected_counts": manifest["selected_counts"],
            "level_cutoffs": manifest["level_cutoffs"],
        } for manifest in manifests},
    }

    for manifest in manifests:
        write_dataset_outputs(slug_map[manifest["dataset"]], manifest, args.output_root)

    overview_path = args.output_root / "overview.json"
    overview_path.write_text(json.dumps(overview, indent=2, ensure_ascii=False) + "\n")

    args.summary_path.parent.mkdir(parents=True, exist_ok=True)
    args.summary_path.write_text(build_summary(manifests, seed_status), encoding="utf-8")

    for dataset_slug in ["droid", "agibot", "egodex"]:
        print(args.output_root / dataset_slug / "meta/manifest.json")
    print(overview_path)
    print(args.summary_path)


if __name__ == "__main__":
    main()
