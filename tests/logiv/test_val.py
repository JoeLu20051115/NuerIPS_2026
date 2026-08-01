from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

from pi05_libero_repro.logiv.model import (
    ContextEnvelope,
    ContextPhase,
    Fact,
    GoalMode,
)
from pi05_libero_repro.logiv.proposal import ScriptedProposalProvider
from pi05_libero_repro.logiv.val import (
    SignedTraceStatus,
    ValidationStatus,
    ValWrapper,
    run_signed_trace,
    verify_certificate,
)


REAL_VAL = Path("/home/xingrui/.local/bin/Validate")


def preinstall_context(request_id: str = "request-1") -> ContextEnvelope:
    return ContextEnvelope(
        phase=ContextPhase.PREINSTALL_VAL,
        goal_mode=GoalMode.METADATA_ASSISTED,
        request_id=request_id,
        request_generation=0,
        episode_id="episode-3-0",
        goal_id="libero10-task-3-official-goal-v1",
        goal_epoch=0,
        epoch_id=11,
        graph_version=None,
        occurrence_id=None,
        attempt_id=None,
        certificate_hash=None,
        safety_epoch=None,
    )


def package_and_plan(task_id: int = 3):
    package = ScriptedProposalProvider().propose(task_id=task_id, epoch_id=11)
    plan = tuple(item.action for item in package.proposal.candidate_subtasks)
    sidecar = json.dumps(
        [
            {
                "occurrence_id": item.occurrence_id,
                "schema": item.action.schema,
                "arguments": list(item.action.arguments),
            }
            for item in package.proposal.candidate_subtasks
        ],
        sort_keys=True,
    ).encode("utf-8")
    return package, plan, sidecar


def test_signed_trace_accepts_nominal_task3() -> None:
    package, plan, _ = package_and_plan()

    trace = run_signed_trace(package.problem, plan)

    assert trace.status is SignedTraceStatus.VALID
    assert package.problem.goal <= trace.final_true


def test_signed_trace_distinguishes_known_false_precondition_from_unknown() -> None:
    package, plan, _ = package_and_plan()
    place_first = plan[1:]

    known_false = run_signed_trace(package.problem, place_first)
    unknown_problem = replace(
        package.problem,
        initial_false=package.problem.initial_false - {Fact("holding", ("akita_black_bowl_1",))},
    )
    unknown = run_signed_trace(unknown_problem, place_first)

    assert known_false.status is SignedTraceStatus.ACTION_PRECONDITION_FAILURE
    assert known_false.obligations[0].fact == Fact("holding", ("akita_black_bowl_1",))
    assert unknown.status is SignedTraceStatus.PLAN_GROUNDING_INCOMPLETE


def test_signed_trace_reports_final_goal_failure_and_negative_goal_unknown() -> None:
    package, plan, _ = package_and_plan(task_id=8)

    incomplete = run_signed_trace(package.problem, plan[:1])
    negative_goal_problem = replace(
        package.problem,
        negative_goal=frozenset({Fact("holding", ("moka_pot_1",))}),
        initial_false=package.problem.initial_false - {Fact("holding", ("moka_pot_1",))},
    )
    negative_unknown = run_signed_trace(negative_goal_problem, plan)

    assert incomplete.status is SignedTraceStatus.FINAL_GOAL_FAILURE
    assert negative_unknown.status is SignedTraceStatus.PLAN_GROUNDING_INCOMPLETE


def test_real_val_certifies_task3_and_binds_every_input() -> None:
    package, plan, sidecar = package_and_plan()
    wrapper = ValWrapper(REAL_VAL, timeout_seconds=5.0)
    context = preinstall_context()

    result = wrapper.validate(package.problem, plan, sidecar, context)

    assert result.status is ValidationStatus.VALID
    assert result.certificate is not None
    assert "Plan valid" in result.stdout
    assert verify_certificate(
        result.certificate,
        problem=package.problem,
        plan=plan,
        occurrence_sidecar=sidecar,
        context=context,
        val_binary=REAL_VAL,
        timeout_seconds=5.0,
        forbidden_retry_keys=frozenset(),
        retry_ledger_version=0,
    )
    assert not verify_certificate(
        result.certificate,
        problem=package.problem,
        plan=plan,
        occurrence_sidecar=sidecar + b" ",
        context=context,
        val_binary=REAL_VAL,
        timeout_seconds=5.0,
        forbidden_retry_keys=frozenset(),
        retry_ledger_version=0,
    )
    assert not verify_certificate(
        replace(result.certificate, certificate_hash="0" * 64),
        problem=package.problem,
        plan=plan,
        occurrence_sidecar=sidecar,
        context=context,
        val_binary=REAL_VAL,
        timeout_seconds=5.0,
        forbidden_retry_keys=frozenset(),
        retry_ledger_version=0,
    )
    assert not verify_certificate(
        result.certificate,
        problem=package.problem,
        plan=plan,
        occurrence_sidecar=sidecar,
        context=context,
        val_binary=REAL_VAL,
        timeout_seconds=5.0,
        forbidden_retry_keys=frozenset(),
        retry_ledger_version=1,
    )


def test_real_val_certifies_all_ten_scripted_plans() -> None:
    wrapper = ValWrapper(REAL_VAL, timeout_seconds=5.0)

    for task_id in range(10):
        package, plan, sidecar = package_and_plan(task_id)
        context = replace(
            preinstall_context(request_id=f"request-{task_id}"),
            episode_id=f"episode-{task_id}-0",
            goal_id=package.frozen_goal.goal_id,
        )
        result = wrapper.validate(package.problem, plan, sidecar, context)

        assert result.status is ValidationStatus.VALID, (task_id, result.reason, result.stdout)
    assert not verify_certificate(
        result.certificate,
        problem=package.problem,
        plan=plan,
        occurrence_sidecar=sidecar,
        context=replace(context, request_id="stale-request"),
        val_binary=REAL_VAL,
        timeout_seconds=5.0,
        forbidden_retry_keys=frozenset(),
        retry_ledger_version=0,
    )


def _fake_val(path: Path, body: str) -> Path:
    path.write_text("#!/usr/bin/env python3\n" + body, encoding="utf-8")
    path.chmod(0o755)
    return path


def test_val_timeout_and_unrecognized_outputs_fail_closed(tmp_path: Path) -> None:
    package, plan, sidecar = package_and_plan()
    context = preinstall_context()
    timeout_binary = _fake_val(
        tmp_path / "timeout-val",
        "import time\ntime.sleep(1)\n",
    )
    malformed_binary = _fake_val(
        tmp_path / "malformed-val",
        "print('maybe valid')\n",
    )
    crash_binary = _fake_val(
        tmp_path / "crash-val",
        "import sys\nprint('boom')\nsys.exit(2)\n",
    )

    timeout = ValWrapper(timeout_binary, timeout_seconds=0.01, val_version="fake-v1").validate(
        package.problem, plan, sidecar, context
    )
    malformed = ValWrapper(
        malformed_binary, timeout_seconds=1.0, val_version="fake-v1"
    ).validate(package.problem, plan, sidecar, context)
    crash = ValWrapper(crash_binary, timeout_seconds=1.0, val_version="fake-v1").validate(
        package.problem, plan, sidecar, context
    )

    assert timeout.status is ValidationStatus.VALIDATION_ERROR
    assert malformed.status is ValidationStatus.VALIDATION_ERROR
    assert crash.status is ValidationStatus.VALIDATION_ERROR
    assert timeout.certificate is malformed.certificate is crash.certificate is None


def test_known_val_invalid_marker_is_invalid_not_wrapper_error(tmp_path: Path) -> None:
    package, plan, sidecar = package_and_plan()
    binary = _fake_val(
        tmp_path / "invalid-val",
        "import sys\nprint('Plan invalid')\nsys.exit(1)\n",
    )

    result = ValWrapper(binary, timeout_seconds=1.0, val_version="fake-v1").validate(
        package.problem, plan, sidecar, preinstall_context()
    )

    assert result.status is ValidationStatus.INVALID
    assert result.certificate is None


def test_malformed_occurrence_sidecar_fails_before_val(tmp_path: Path) -> None:
    package, plan, _ = package_and_plan()
    marker = tmp_path / "called"
    binary = _fake_val(
        tmp_path / "must-not-run",
        f"from pathlib import Path\nPath({str(marker)!r}).touch()\nprint('Plan valid')\n",
    )

    result = ValWrapper(binary, timeout_seconds=1.0, val_version="fake-v1").validate(
        package.problem, plan, b"not-json", preinstall_context()
    )

    assert result.status is ValidationStatus.VALIDATION_ERROR
    assert not marker.exists()
