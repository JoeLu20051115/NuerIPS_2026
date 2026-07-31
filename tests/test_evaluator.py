from __future__ import annotations

from dataclasses import replace

import pytest

from pi05_libero_repro.records import EpisodeRecord
from scripts.eval_libero import pending_episode_indices, video_name


def record(episode_idx: int) -> EpisodeRecord:
    return EpisodeRecord(
        checkpoint="full",
        task_id=0,
        task_name="task",
        episode_idx=episode_idx,
        init_state_sha256="0" * 64,
        seed=7,
        success=False,
        valid=True,
        steps=520,
        inference_requests=104,
        wall_seconds=1.0,
        exception=None,
        first_frame_sha256="1" * 64,
        action_min=-1.0,
        action_max=1.0,
        action_mean=0.0,
        done=False,
        check_success=False,
        video_path="video.mp4",
    )


def test_pending_indices_require_a_contiguous_prefix() -> None:
    assert pending_episode_indices([], "full", 0, 3) == [0, 1, 2]
    assert pending_episode_indices([record(0)], "full", 0, 3) == [1, 2]
    assert pending_episode_indices([record(0), record(1), record(2)], "full", 0, 3) == []

    with pytest.raises(ValueError, match="resume would change policy RNG sequence"):
        pending_episode_indices([record(1)], "full", 0, 3)


def test_pending_indices_ignore_other_checkpoint_and_task() -> None:
    unrelated = [replace(record(0), checkpoint="early"), replace(record(0), task_id=1)]
    assert pending_episode_indices(unrelated, "full", 0, 2) == [0, 1]


def test_video_names_are_unique_and_outcome_labeled() -> None:
    assert video_name(0, 0, True) == "task_00_episode_00_success.mp4"
    assert video_name(0, 0, False) == "task_00_episode_00_failure.mp4"
    assert video_name(9, 49, True) == "task_09_episode_49_success.mp4"
