from __future__ import annotations

import json
from pathlib import Path
import subprocess
import tomllib


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
        if path.startswith(
            (
                "scripts/",
                "src/pi05_libero_repro/logiv/config/",
                "patches/robotwin/",
            )
        )
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
    paths = sorted(
        (ROOT / "src/pi05_libero_repro/logiv/config").glob("*.json")
    )
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
    assert "origin_config_path(" in text
    assert "online-v" not in text


def test_public_release_identity_and_docs_are_origin_only() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert project["project"]["name"] == "logiv-origin"
    assert (ROOT / "README.md").read_text(encoding="utf-8").startswith(
        "# LOGIV Origin\n"
    )

    docs = {
        path for path in _tracked_files() if path.startswith("docs/")
    }
    assert docs == {"docs/DEPLOYMENT.md", "docs/METHOD.md"}


def test_origin_configs_are_packaged_with_the_runtime() -> None:
    from pi05_libero_repro.logiv.configuration import origin_config_path

    expected = (
        "coverage.json",
        "domain.pddl",
        "monitor-evidence.json",
        "prompts.json",
        "proposals.json",
        "terminal-recovery.json",
        "val-build.json",
    )
    assert all(origin_config_path(name).is_file() for name in expected)
