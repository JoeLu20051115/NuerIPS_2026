from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
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
