from __future__ import annotations

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_origin_deployment_entrypoints_exist() -> None:
    expected = (
        ROOT / "scripts/serve_policy.py",
        ROOT / "scripts/run_policy_server.sh",
        ROOT / "scripts/run_logiv_eval.sh",
        ROOT / "scripts/run_robotwin_logiv.py",
        ROOT / "patches/robotwin/logiv-origin.patch",
    )
    assert all(path.is_file() for path in expected)


def test_deployment_sources_have_no_seed_selector_or_log_sink() -> None:
    paths = (
        ROOT / "scripts/run_robotwin_logiv.py",
        ROOT / "scripts/run_policy_server.sh",
        ROOT / "patches/robotwin/logiv-origin.patch",
    )
    text = "\n".join(path.read_text(encoding="utf-8") for path in paths)
    forbidden = (
        "accepted_seeds",
        "candidate_seed",
        "seed_pool",
        "seed_scan",
        "server.log",
        "log_dir",
    )
    assert not any(value in text for value in forbidden)


def test_python_deployment_entrypoints_expose_help() -> None:
    for path in (
        ROOT / "scripts/eval_logiv_libero.py",
        ROOT / "scripts/run_robotwin_logiv.py",
    ):
        result = subprocess.run(
            [sys.executable, str(path), "--help"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert "usage:" in result.stdout.lower()
