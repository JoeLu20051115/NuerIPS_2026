from __future__ import annotations

import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def _tracked_files() -> tuple[str, ...]:
    output = subprocess.check_output(
        ["git", "ls-files"], cwd=ROOT, text=True
    )
    return tuple(output.splitlines())


def test_origin_tree_has_no_generated_records_or_seed_selection() -> None:
    files = _tracked_files()
    forbidden_roots = (
        "results/",
        "evaluation_results/",
        "evaluation_results_dualsystem/",
    )
    assert not any(path.startswith(forbidden_roots) for path in files)

    deployment = "\n".join(
        (ROOT / path).read_text(encoding="utf-8", errors="ignore")
        for path in files
        if path.startswith(("scripts/", "configs/logiv/", "patches/robotwin/"))
    )
    forbidden_names = (
        "accepted_seeds",
        "candidate_seed",
        "seed_pool",
        "seed_scan",
        "freeze_robotwin_logiv_seeds",
    )
    assert not any(name in deployment for name in forbidden_names)


def test_origin_configs_are_resolved() -> None:
    paths = sorted((ROOT / "configs/logiv/origin").glob("*.json"))
    assert paths
    for path in paths:
        assert "extends" not in json.loads(path.read_text(encoding="utf-8"))


def test_origin_runtime_defaults_only_reference_origin_configs() -> None:
    sources = (
        ROOT / "src/pi05_libero_repro/logiv/prompts.py",
        ROOT / "src/pi05_libero_repro/logiv/proposal.py",
        ROOT / "src/pi05_libero_repro/logiv/libero_adapter.py",
        ROOT / "scripts/eval_logiv_libero.py",
    )
    text = "\n".join(path.read_text(encoding="utf-8") for path in sources)
    assert "configs/logiv/origin/" in text
    assert "online-v" not in text
