#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "20260331" / "analysis"
SPLIT_ORDER = ["L1", "L2", "L3"]
DATASET_ORDER = [
    "DreamZero-DROID",
    "DreamZero-AgiBot",
    "DreamDojo-AgiBot",
    "DreamDojo-EgoDex",
]

OPS = {
    "pick": "pick",
    "grasp": "pick",
    "pickup": "pick",
    "put": "place",
    "place": "place",
    "move": "move",
    "open": "open",
    "close": "close",
    "insert": "insert",
    "remove": "remove",
    "press": "press",
    "stamp": "stamp",
    "stack": "stack",
    "unstack": "unstack",
    "add": "add",
    "heat": "heat",
    "store": "store",
    "pack": "pack",
    "clear": "clear",
    "restock": "restock",
    "push": "push",
}
STATEFUL_OPS = {"open", "close", "insert", "remove", "press", "stamp", "stack", "unstack", "add", "heat", "store"}


def canon_ops(text: str) -> list[str]:
    text = text.lower().replace("_", " ").replace("-", " ")
    tokens = re.findall(r"[a-z]+", text)
    return [OPS[tok] for tok in tokens if tok in OPS]


def finalize_rows(dataset: str, split: str, rows: list[dict], source_note: str) -> dict:
    n = len(rows)
    return {
        "dataset": dataset,
        "split": split,
        "n": n,
        "mean_step_proxy": sum(r["step_proxy"] for r in rows) / n,
        "mean_unique_ops": sum(r["unique_ops"] for r in rows) / n,
        "compositional_rate_ge3": sum(r["step_proxy"] >= 3 for r in rows) / n,
        "stateful_rate": sum(r["has_stateful"] for r in rows) / n,
        "bimanual_rate": (
            sum(r["bimanual"] for r in rows) / n
            if all("bimanual" in r for r in rows)
            else None
        ),
        "unique_task_types_in_split": len({r["task_type"] for r in rows}),
        "source_note": source_note,
    }


def load_dreamzero_droid() -> list[dict]:
    path = REPO_ROOT / "evaluation_results_dualsystem" / "droid_3way_selected150.json"
    payload = json.loads(path.read_text())
    grouped: dict[str, dict] = {}
    for row in payload["results"]:
        if row["mode"] != "task_token_only":
            continue
        episode_id = str(row["episode_id"])
        grouped.setdefault(
            episode_id,
            {
                "split": str(row["dro_split"]),
                "task": str(row["task"]),
            },
        )

    by_split: dict[str, list[dict]] = defaultdict(list)
    for row in grouped.values():
        ops = canon_ops(row["task"])
        by_split[row["split"]].append(
            {
                "step_proxy": len(ops),
                "unique_ops": len(set(ops)),
                "has_stateful": bool(set(ops) & STATEFUL_OPS),
                "task_type": row["task"],
            }
        )

    return [
        finalize_rows(
            dataset="DreamZero-DROID",
            split=split,
            rows=by_split[split],
            source_note="Task-text heuristic from droid_3way_selected150.json (`task` field).",
        )
        for split in SPLIT_ORDER
    ]


def load_agibot_manifest_family(dataset: str, manifests: dict[str, Path], source_note: str) -> list[dict]:
    out = []
    for split in SPLIT_ORDER:
        payload = json.loads(manifests[split].read_text())
        rows = []
        for ep in payload["episodes"]:
            ops: list[str] = []
            arms = set()
            for step in ep["action_plan"]:
                ops.extend(canon_ops(step))
                step_lower = step.lower()
                if "left arm" in step_lower:
                    arms.add("left")
                if "right arm" in step_lower:
                    arms.add("right")
            rows.append(
                {
                    "step_proxy": len(ep["action_plan"]),
                    "unique_ops": len(set(ops)),
                    "has_stateful": bool(set(ops) & STATEFUL_OPS),
                    "bimanual": len(arms) > 1,
                    "task_type": ep["task_group"],
                }
            )
        out.append(finalize_rows(dataset=dataset, split=split, rows=rows, source_note=source_note))
    return out


def tercile_bounds(n: int) -> list[tuple[int, int]]:
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


def load_dreamdojo_egodex() -> list[dict]:
    manifest = json.loads((REPO_ROOT / "data" / "egodex_dreamdojo_easy400" / "meta" / "manifest.json").read_text())
    info = json.loads((REPO_ROOT / "data" / "egodex_eval_official" / "EgoDex_Eval" / "meta" / "info.json").read_text())
    fps = float(info["fps"])

    items = []
    for ep in manifest["episodes"]:
        ops = canon_ops(ep["task_group"])
        items.append(
            (
                float(ep["num_frames"]) / fps,
                {
                    "step_proxy": len(ops),
                    "unique_ops": len(set(ops)),
                    "has_stateful": bool(set(ops) & STATEFUL_OPS),
                    "task_type": ep["task_group"],
                },
            )
        )
    items.sort(key=lambda item: item[0])

    out = []
    for split, (start, end) in zip(SPLIT_ORDER, tercile_bounds(len(items))):
        rows = [item[1] for item in items[start:end]]
        out.append(
            finalize_rows(
                dataset="DreamDojo-EgoDex",
                split=split,
                rows=rows,
                source_note=(
                    "Recovered from egodex_dreamdojo_easy400 source manifest by duration terciles; "
                    "semantic proxy comes from task_group verb tokens."
                ),
            )
        )
    return out


def build_markdown(df: pd.DataFrame) -> str:
    lines = [
        "# Semantic Complexity Proxies",
        "",
        "These are non-time proxies for semantic complexity. Higher values suggest more semantic stages or richer interaction structure.",
        "",
        "| Dataset | Split | N | Mean step proxy | Mean unique ops | >=3-step rate | Stateful-op rate | Bimanual rate | Unique task types |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in df.itertuples(index=False):
        bimanual = "-" if pd.isna(row.bimanual_rate) else f"{row.bimanual_rate:.2f}"
        lines.append(
            "| {} | {} | {} | {:.2f} | {:.2f} | {:.2f} | {:.2f} | {} | {} |".format(
                row.dataset,
                row.split,
                int(row.n),
                row.mean_step_proxy,
                row.mean_unique_ops,
                row.compositional_rate_ge3,
                row.stateful_rate,
                bimanual,
                int(row.unique_task_types_in_split),
            )
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `DreamZero-DROID`: semantic proxy is extracted from the natural-language `task` string, so it is a weak but usable heuristic.",
            "- `DreamZero-AgiBot` and `DreamDojo-AgiBot`: semantic proxy uses manifest `action_plan`, which is the strongest signal currently available.",
            "- `DreamDojo-EgoDex`: semantic proxy uses `task_group` verb composition because the source manifest does not expose per-episode action plans.",
            "- The safest presentation is to call these `semantic complexity proxies`, not a perfectly calibrated universal complexity score.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    rows.extend(load_dreamzero_droid())
    rows.extend(
        load_agibot_manifest_family(
            dataset="DreamZero-AgiBot",
            manifests={
                "L1": REPO_ROOT / "data" / "Agi_DualBetter_L1_50" / "meta" / "manifest.json",
                "L2": REPO_ROOT / "data" / "Agi_DualBetter_L2_50" / "meta" / "manifest.json",
                "L3": REPO_ROOT / "data" / "Agi_DualBetter_L3_50" / "meta" / "manifest.json",
            },
            source_note="Manifest action_plan statistics from the archived Agi_DualBetter splits.",
        )
    )
    rows.extend(
        load_agibot_manifest_family(
            dataset="DreamDojo-AgiBot",
            manifests={
                "L1": REPO_ROOT / "20260331" / "data" / "Agi_DreamDojo286_20260331_L1_50" / "meta" / "manifest.json",
                "L2": REPO_ROOT / "20260331" / "data" / "Agi_DreamDojo286_20260331_L2_50" / "meta" / "manifest.json",
                "L3": REPO_ROOT / "20260331" / "data" / "Agi_DreamDojo286_20260331_L3_50" / "meta" / "manifest.json",
            },
            source_note="Manifest action_plan statistics from the current 20260331 DreamDojo AgiBot splits.",
        )
    )
    rows.extend(load_dreamdojo_egodex())

    df = pd.DataFrame(rows)
    df["dataset"] = pd.Categorical(df["dataset"], categories=DATASET_ORDER, ordered=True)
    df["split"] = pd.Categorical(df["split"], categories=SPLIT_ORDER, ordered=True)
    df = df.sort_values(["dataset", "split"]).reset_index(drop=True)

    out_tsv = OUT_DIR / "semantic_complexity_proxies.tsv"
    out_md = OUT_DIR / "semantic_complexity_proxies.md"
    df.to_csv(out_tsv, sep="\t", index=False)
    out_md.write_text(build_markdown(df), encoding="utf-8")


if __name__ == "__main__":
    main()
