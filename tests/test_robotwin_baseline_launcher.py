from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import sys
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
SPEC = spec_from_file_location(
    "run_robotwin_baseline_10x10",
    ROOT / "scripts" / "run_robotwin_baseline_10x10.py",
)
assert SPEC is not None and SPEC.loader is not None
BASELINE = module_from_spec(SPEC)
SPEC.loader.exec_module(BASELINE)


def test_task_command_uses_the_explicit_checkpoint(tmp_path: Path) -> None:
    config = json.loads(
        (ROOT / "configs/robotwin/logiv-gpt4o-63-vs-pi05-56.json").read_text()
    )
    args = SimpleNamespace(
        python=tmp_path / "python",
        checkpoint=tmp_path / "checkpoint",
        tokenizer=tmp_path / "tokenizer.model",
        tag="test-baseline",
    )

    command = BASELINE._task_command(config, "turn_switch", args)

    assert command[command.index("--policy_path") + 1] == str(args.checkpoint)
    assert "--baseline_only" in command


def test_launcher_accepts_physical_gpu_number_above_worker_range(
    monkeypatch,
    tmp_path: Path,
) -> None:
    calls = []

    def record_run(gpu, tasks, args):
        calls.append((gpu, tasks, args))
        return 0

    monkeypatch.setattr(BASELINE, "_run_worker", record_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_robotwin_baseline_10x10.py",
            "--worker", "0",
            "--gpu", "4",
            "--protocol", str(tmp_path / "protocol.json"),
            "--output", str(tmp_path / "output"),
            "--tag", "test",
            "--taco", str(tmp_path / "taco"),
            "--checkpoint", str(tmp_path / "checkpoint"),
            "--python", sys.executable,
            "--tokenizer", str(tmp_path / "tokenizer.model"),
        ],
    )

    assert BASELINE.main() == 0
    assert len(calls) == 1
    gpu, tasks, _args = calls[0]
    assert gpu == 4
    assert tasks == BASELINE.GPU_TASKS[0]
