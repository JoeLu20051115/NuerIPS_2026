from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from pi05_libero_repro.records import (
    EpisodeRecord,
    append_record,
    load_records,
    validate_records,
    wilson_interval,
)


def make_record(episode_idx: int = 0, success: bool = True) -> EpisodeRecord:
    return EpisodeRecord(
        checkpoint="full",
        task_id=0,
        task_name="task",
        episode_idx=episode_idx,
        init_state_sha256="0" * 64,
        seed=7,
        success=success,
        valid=True,
        steps=12,
        inference_requests=3,
        wall_seconds=1.5,
        exception=None,
        first_frame_sha256="1" * 64,
        action_min=-0.5,
        action_max=0.5,
        action_mean=0.0,
        done=success,
        check_success=success,
        video_path="video.mp4",
    )


def test_jsonl_round_trip_and_duplicate_detection(tmp_path: Path) -> None:
    path = tmp_path / "episodes.jsonl"
    append_record(path, make_record())

    assert load_records(path) == [make_record()]
    with pytest.raises(ValueError, match="duplicate episode: full/0/0"):
        append_record(path, make_record())


def test_validation_and_wilson_interval() -> None:
    records = [make_record(episode_idx=i, success=i < 46) for i in range(50)]

    assert validate_records(records, expected_trials=50) == []
    low, high = wilson_interval(46, 50)
    assert 0.81 < low < 0.82
    assert 0.96 < high < 0.97


def test_validation_rejects_invalid_predicates_hashes_and_numbers() -> None:
    record = make_record()
    cases = [
        (replace(record, valid=False), "invalid episode: full/0/0"),
        (replace(record, check_success=False), "predicate mismatch: full/0/0"),
        (replace(record, first_frame_sha256="bad"), "invalid first-frame hash: full/0/0"),
        (replace(record, action_mean=float("nan")), "non-finite action statistics: full/0/0"),
    ]

    for broken, expected in cases:
        assert expected in validate_records([broken], expected_trials=1)


def test_wilson_interval_rejects_impossible_counts() -> None:
    with pytest.raises(ValueError, match="successes must be between zero and total"):
        wilson_interval(2, 1)
    with pytest.raises(ValueError, match="total must be positive"):
        wilson_interval(0, 0)
