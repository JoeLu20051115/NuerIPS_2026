from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).parents[1]


@pytest.mark.parametrize("name", ["run_policy_server.sh", "run_libero_eval.sh"])
def test_shell_launcher_requires_arguments(name: str) -> None:
    completed = subprocess.run(
        ["bash", str(ROOT / "scripts" / name)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 64
    assert completed.stderr.startswith("usage:")


def test_architecture_snapshot_help_does_not_load_model() -> None:
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "snapshot_architecture.py"), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0
    assert "--checkpoint-dir" in completed.stdout
