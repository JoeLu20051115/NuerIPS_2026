from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).parents[1]


@pytest.mark.parametrize(
    "name",
    [
        "run_policy_server.sh",
        "run_libero_eval.sh",
        "run_logiv_eval.sh",
        "run_logiv_online_shard.sh",
        "run_gpt4o_logiv_dual.sh",
    ],
)
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


def test_libero_image_pins_python38_build_backend() -> None:
    dockerfile = (ROOT / "docker" / "Dockerfile.libero").read_text()
    backend = "uv pip install setuptools==75.3.0 wheel==0.45.1"
    sync = "uv pip sync"

    assert backend in dockerfile
    assert dockerfile.count("--no-build-isolation") == 2
    assert dockerfile.index(backend) < dockerfile.index(sync)


def test_libero_image_allows_unprivileged_python_execution() -> None:
    dockerfile = (ROOT / "docker" / "Dockerfile.libero").read_text()

    assert "chmod o+x /root" in dockerfile


def test_policy_server_persists_identity_and_runtime_log() -> None:
    launcher = (ROOT / "scripts" / "run_policy_server.sh").read_text()

    assert 'server.log' in launcher
    assert 'tee' in launcher


def test_logiv_launcher_accepts_the_online_repair_arm() -> None:
    launcher = (ROOT / "scripts" / "run_logiv_eval.sh").read_text()

    assert "LOGIV_ONLINE" in launcher


def test_logiv_launcher_forwards_openai_key_by_name_only() -> None:
    launcher = (ROOT / "scripts" / "run_logiv_eval.sh").read_text()

    assert "-e OPENAI_API_KEY" in launcher
    assert "OPENAI_API_KEY=" not in launcher


def test_online_shard_launcher_defines_two_seeds_and_all_ten_tasks() -> None:
    launcher = (ROOT / "scripts" / "run_logiv_online_shard.sh").read_text()

    assert "seeds=(7 17)" in launcher
    assert "pi05-subtasks-online-v5.json" in launcher
    assert "--online-recovery-requires-goal-task-ids 6" in launcher
    assert "--online-stall-requires-handempty-task-ids 8" in launcher
    for task_id in range(10):
        assert str(task_id) in launcher
