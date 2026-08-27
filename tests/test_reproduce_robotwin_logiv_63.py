from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import json
import os
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/reproduce_robotwin_logiv_63.py"
DRIVER = None
if SCRIPT.exists():
    SPEC = spec_from_file_location("reproduce_robotwin_logiv_63", SCRIPT)
    assert SPEC is not None and SPEC.loader is not None
    DRIVER = module_from_spec(SPEC)
    SPEC.loader.exec_module(DRIVER)


@pytest.fixture
def args(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        python=tmp_path / "robotwin-python",
        protocol=ROOT / "configs/robotwin/logiv-gpt4o-63-vs-pi05-56.json",
        output=tmp_path / "run-63",
        taco=tmp_path / "TACO",
        checkpoint=tmp_path / "checkpoint",
        tokenizer=tmp_path / "tokenizer.model",
        val_binary=tmp_path / "Validate",
        gpus=[2, 4, 6],
        logiv_root=ROOT,
    )


def test_worker_commands_cover_three_gpus_and_both_phases(
    args: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert DRIVER is not None
    secret = "sk-" + "s" * 48
    monkeypatch.setenv("OPENAI_API_KEY", secret)

    logiv = DRIVER._worker_commands(args, "logiv", "run-63-logiv")
    baseline = DRIVER._worker_commands(args, "baseline", "run-63-baseline")

    assert [command[command.index("--gpu") + 1] for command in logiv] == [
        "2",
        "4",
        "6",
    ]
    assert all("run_robotwin_logiv_10x10.py" in command[1] for command in logiv)
    assert all(
        "run_robotwin_baseline_10x10.py" in command[1]
        for command in baseline
    )
    assert all("--val-binary" in command for command in logiv)
    assert all("--val-binary" not in command for command in baseline)
    assert all(secret not in argument for command in logiv for argument in command)


def test_expectations_report_live_mismatch() -> None:
    assert DRIVER is not None
    errors = DRIVER._expectations(
        {
            "strict_protocol_complete": True,
            "successes": 62,
            "baseline_successes": 56,
        },
        {"expected_successes": {"logiv": 63, "baseline": 56}},
    )

    assert errors == ["expected LOGIV 63, observed 62"]


def test_expectations_reject_incomplete_protocol() -> None:
    assert DRIVER is not None
    errors = DRIVER._expectations(
        {
            "strict_protocol_complete": False,
            "successes": 63,
            "baseline_successes": 56,
        },
        {"expected_successes": {"logiv": 63, "baseline": 56}},
    )

    assert errors == ["frozen protocol is incomplete"]


def test_compaction_removes_only_raw_and_current_tag_directories(
    tmp_path: Path,
) -> None:
    assert DRIVER is not None
    output = tmp_path / "output"
    eval_root = tmp_path / "eval_result"
    (output / "raw").mkdir(parents=True)
    (output / "raw/task.log").write_text("raw", encoding="utf-8")
    (output / "summary.json").write_text("{}", encoding="utf-8")
    for name in ("run-logiv", "run-baseline", "unrelated"):
        path = eval_root / name
        path.mkdir(parents=True)
        (path / "data").write_text(name, encoding="utf-8")

    DRIVER._compact(output, eval_root, ["run-logiv", "run-baseline"])

    assert not (output / "raw").exists()
    assert (output / "summary.json").is_file()
    assert not (eval_root / "run-logiv").exists()
    assert not (eval_root / "run-baseline").exists()
    assert (eval_root / "unrelated/data").read_text() == "unrelated"


def test_compaction_refuses_paths_outside_the_eval_root(tmp_path: Path) -> None:
    assert DRIVER is not None
    with pytest.raises(ValueError, match="outside eval root"):
        DRIVER._compact(
            tmp_path / "output",
            tmp_path / "eval_result",
            ["../other"],
        )


def test_dry_run_preflights_without_creating_output_or_printing_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert DRIVER is not None
    taco = tmp_path / "TACO"
    evaluator = taco / "third_party/Robotwin/script/eval_lerobot_torch_pi05.py"
    evaluator.parent.mkdir(parents=True)
    evaluator.touch()
    checkpoint = tmp_path / "checkpoint"
    for task in ("handover_block", "move_can_pot", "beat_block_hammer"):
        cfn = checkpoint / "cfns" / f"{task}_cfn.pt"
        cfn.parent.mkdir(parents=True, exist_ok=True)
        cfn.touch()
    tokenizer = tmp_path / "tokenizer.model"
    tokenizer.touch()
    val_binary = tmp_path / "Validate"
    val_binary.touch(mode=0o755)
    robotwin_python = tmp_path / "robotwin-python"
    robotwin_python.touch(mode=0o755)
    output = tmp_path / "dry-run-63"
    secret = "sk-" + "k" * 48
    monkeypatch.setenv("OPENAI_API_KEY", secret)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "--taco", str(taco),
            "--checkpoint", str(checkpoint),
            "--tokenizer", str(tokenizer),
            "--val-binary", str(val_binary),
            "--python", str(robotwin_python),
            "--output", str(output),
            "--gpus", "0", "1", "2",
            "--dry-run",
        ],
    )

    assert DRIVER.main() == 0

    printed = capsys.readouterr().out
    payload = json.loads(printed)
    assert len(payload["logiv"]) == 3
    assert len(payload["baseline"]) == 3
    assert secret not in printed
    assert not output.exists()
    assert os.environ["OPENAI_API_KEY"] == secret
