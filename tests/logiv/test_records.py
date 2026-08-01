from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from pi05_libero_repro.logiv.model import ContextEnvelope, ContextPhase, GoalMode
from pi05_libero_repro.logiv.records import (
    EventJournal,
    LogivEpisodeRecord,
    append_episode_record,
    load_episode_records,
    paired_task_stratified_bootstrap,
    validate_event_chain,
    validate_episode_records,
)


def _context() -> ContextEnvelope:
    return ContextEnvelope(
        phase=ContextPhase.PRE_DISPATCH_FACTS,
        goal_mode=GoalMode.METADATA_ASSISTED,
        request_id="request-1",
        request_generation=0,
        episode_id="run/full/METADATA_ASSISTED/8/0",
        goal_id="goal-8",
        goal_epoch=0,
        epoch_id=12,
        graph_version="graph-abc",
        occurrence_id="occurrence-1",
        attempt_id=None,
        certificate_hash="a" * 64,
        safety_epoch=None,
    )


def _record(*, arm: str = "FULL_LOGIV", episode_idx: int = 0, success: bool = True):
    return LogivEpisodeRecord(
        schema_version=1,
        run_id="run-1",
        checkpoint="full",
        method_arm=arm,
        goal_mode="METADATA_ASSISTED",
        deviation_mode="NOMINAL",
        task_id=8,
        task_name="put both moka pots on the stove",
        episode_idx=episode_idx,
        seed=7,
        allocated=True,
        valid=True,
        success=success,
        terminal_status="EPISODE_SUCCESS" if success else "EPISODE_FAIL",
        terminal_cause="EPISODE_SUCCESS" if success else "EPISODE_FAIL",
        evaluator_status="EPISODE_SUCCESS" if success else "EPISODE_FAIL",
        init_state_sha256="0" * 64,
        first_frame_sha256="1" * 64,
        prompt_version="pi05-subtasks-v1",
        prompt_config_sha256="2" * 64,
        proposal_config_sha256="3" * 64,
        coverage_manifest_sha256="4" * 64,
        domain_sha256="5" * 64,
        initial_certificate_hash="6" * 64,
        initial_graph_hash="7" * 64,
        initial_graph_width=2,
        final_graph_hash="8" * 64,
        physical_attempts=2,
        repair_rounds=0,
        total_val_calls=1,
        graph_installs=1,
        steps=40,
        inference_requests=8,
        wall_seconds=2.0,
        safety_permits=2,
        halt_acknowledged=None,
        receipts_count=2,
        event_count=2,
        event_chain_head="9" * 64,
        artifact_dir="episodes/task08/episode000",
        video_path="videos/task08-episode000.mp4",
        exception=None,
        oracle_grounding=True,
        development_only=True,
    )


def test_event_journal_has_context_and_verifiable_hash_chain(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    journal = EventJournal(path)
    first = journal.append("FACTS_GROUNDED", _context(), {"facts": 8})
    second = journal.append("EXECUTION_AUTHORIZED", _context(), {"safety_epoch": 1})

    assert first.previous_hash == "0" * 64
    assert second.previous_hash == first.event_hash
    assert validate_event_chain(path) == (2, second.event_hash)
    payload = json.loads(path.read_text().splitlines()[0])
    assert payload["context"]["request_id"] == "request-1"

    lines = path.read_text().splitlines()
    lines[0] = lines[0].replace('"facts":8', '"facts":9')
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match="event hash mismatch"):
        validate_event_chain(path)


def test_episode_jsonl_round_trip_and_full_comparison_key(tmp_path: Path) -> None:
    path = tmp_path / "episodes.jsonl"
    append_episode_record(path, _record())
    append_episode_record(path, _record(arm="STAGE_ONLY"))

    records = load_episode_records(path)
    assert records == [_record(), _record(arm="STAGE_ONLY")]
    assert records[0].key != records[1].key
    with pytest.raises(ValueError, match="duplicate LOGIV episode"):
        append_episode_record(path, _record())


def test_record_contract_requires_task8_non_chain_width_and_failure_in_denominator() -> None:
    successful = _record()
    failed = replace(
        _record(episode_idx=1, success=False),
        valid=False,
        terminal_status="TERMINAL_NO_FURTHER_DISPATCH",
        terminal_cause="STATE_GROUNDING_FAILURE",
        evaluator_status="NOT_CALLED",
    )
    assert validate_episode_records([successful, failed]) == []

    broken = replace(successful, initial_graph_width=1)
    assert "task 8 Full LOGIV graph must have width >= 2" in validate_episode_records([broken])


def test_task_stratified_paired_bootstrap_uses_equal_task_weight() -> None:
    full = []
    comparator = []
    for task_id in range(2):
        for episode_idx in range(4):
            base = replace(_record(episode_idx=episode_idx), task_id=task_id)
            full.append(replace(base, success=(task_id == 0 or episode_idx < 2)))
            comparator.append(
                replace(base, method_arm="BASE", success=(task_id == 0 and episode_idx < 2))
            )
    result = paired_task_stratified_bootstrap(full, comparator, samples=2000, seed=19)
    # task 0: +0.5, task 1: +0.5, hence equal-weight macro difference +0.5.
    assert result["estimate"] == pytest.approx(0.5)
    assert result["paired_episodes"] == 8
    assert len(result["percentile_95"]) == 2
