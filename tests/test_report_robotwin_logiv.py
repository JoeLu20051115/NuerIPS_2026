from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = spec_from_file_location(
    "report_robotwin_logiv",
    ROOT / "scripts" / "report_robotwin_logiv.py",
)
assert SPEC is not None and SPEC.loader is not None
REPORTER = module_from_spec(SPEC)
SPEC.loader.exec_module(REPORTER)


def _config(*, seeds: list[int] | None = None) -> dict:
    selected = [7] if seeds is None else seeds
    return {
        "evidence_label": "development/frozen-rerun",
        "tasks": {"turn_switch": selected},
        "instructions": {
            "turn_switch": [f"press-{seed}" for seed in selected]
        },
    }


def _write_events(root: Path, name: str, records: list[dict]) -> None:
    path = root / name / "logiv_events.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def _write_baseline(root: Path, rows: list[tuple[int, int, int]]) -> None:
    root.mkdir(parents=True)
    text = "".join(
        f"Success rate: {successes}/{total} current seed: {seed}\n"
        for seed, successes, total in rows
    )
    (root / "turn_switch.log").write_text(text, encoding="utf-8")


def test_minimal_complete_report_needs_no_val_provenance_or_images(
    tmp_path: Path,
) -> None:
    _write_events(
        tmp_path,
        "episode",
        [
            {
                "task": "turn_switch",
                "seed": 7,
                "original_instruction": "press-7",
                "success": True,
            }
        ],
    )
    baseline = tmp_path / "baseline"
    _write_baseline(baseline, [(7, 0, 1)])

    report = REPORTER.build_report(_config(), tmp_path, baseline)

    assert report["strict_protocol_complete"] is True
    assert report["successes"] == 1
    assert report["baseline_successes"] == 0
    assert report["positive_flips"] == 1
    assert report["negative_flips"] == 0
    assert report["errors"] == []


def test_report_rejects_duplicate_and_wrong_instruction(tmp_path: Path) -> None:
    wrong = {
        "task": "turn_switch",
        "seed": 7,
        "original_instruction": "wrong",
        "success": True,
    }
    _write_events(tmp_path, "first", [wrong])
    _write_events(tmp_path, "second", [{**wrong, "original_instruction": "press-7"}])

    report = REPORTER.build_report(_config(), tmp_path, None)

    assert any("duplicate LOGIV record" in error for error in report["errors"])
    assert any("instruction mismatch" in error for error in report["errors"])
    assert report["strict_protocol_complete"] is False


def test_report_records_malformed_and_unexpected_rows_as_errors(
    tmp_path: Path,
) -> None:
    _write_events(
        tmp_path,
        "records",
        [
            {"task": "turn_switch", "seed": "not-an-int", "success": True},
            {
                "task": "other_task",
                "seed": 9,
                "original_instruction": "other",
                "success": False,
            },
        ],
    )

    report = REPORTER.build_report(_config(), tmp_path, None)

    assert any("malformed LOGIV record" in error for error in report["errors"])
    assert any("unexpected LOGIV record" in error for error in report["errors"])
    assert any("missing LOGIV record" in error for error in report["errors"])


def test_report_records_invalid_json_as_an_error(tmp_path: Path) -> None:
    path = tmp_path / "broken" / "logiv_events.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text("{not-json}\n", encoding="utf-8")

    report = REPORTER.build_report(_config(), tmp_path, None)

    assert any("malformed JSON event" in error for error in report["errors"])
    assert report["strict_protocol_complete"] is False


def test_baseline_must_contain_the_exact_frozen_seed_set(tmp_path: Path) -> None:
    _write_events(
        tmp_path,
        "episodes",
        [
            {
                "task": "turn_switch",
                "seed": seed,
                "original_instruction": f"press-{seed}",
                "success": seed == 7,
            }
            for seed in (7, 8)
        ],
    )
    baseline = tmp_path / "baseline"
    _write_baseline(baseline, [(7, 1, 1)])

    report = REPORTER.build_report(_config(seeds=[7, 8]), tmp_path, baseline)

    assert "baseline seed set does not match frozen protocol" in report["errors"]
    assert report["strict_protocol_complete"] is False


def test_markdown_labels_the_output_as_a_frozen_rerun(tmp_path: Path) -> None:
    _write_events(
        tmp_path,
        "episode",
        [
            {
                "task": "turn_switch",
                "seed": 7,
                "original_instruction": "press-7",
                "success": True,
            }
        ],
    )
    report = REPORTER.build_report(_config(), tmp_path, None)

    markdown = REPORTER.render_markdown(report)

    assert "development/frozen-rerun" in markdown
    assert "1/1" in markdown
    assert "VAL" not in markdown
    assert "audit" not in markdown.lower()
