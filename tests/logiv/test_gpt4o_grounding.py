from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

import numpy as np
import pytest

from pi05_libero_repro.logiv.controller import GroundingStatus
from pi05_libero_repro.logiv.gpt4o_grounding import (
    Gpt4oGrounder,
    Gpt4oGroundingError,
)
from pi05_libero_repro.logiv.libero_adapter import LiberoObservationStore
from pi05_libero_repro.logiv.model import (
    ContextEnvelope,
    ContextPhase,
    Fact,
    GoalMode,
    GroundAction,
    ObjectDecl,
    TaskProblem,
    TruthValue,
)


OBJECT = "moka_pot_1"
LEFT = "left_region"
STOVE = "stove_region"
AT_LEFT = Fact("at", (OBJECT, LEFT))
AT_STOVE = Fact("at", (OBJECT, STOVE))
HOLDING = Fact("holding", (OBJECT,))
HANDEMPTY = Fact("handempty")
UNIVERSE = frozenset({AT_LEFT, AT_STOVE, HOLDING, HANDEMPTY})
PROBLEM = TaskProblem(
    name="gpt4o_grounding_test",
    objects=(
        ObjectDecl(OBJECT, "movable"),
        ObjectDecl(LEFT, "surface"),
        ObjectDecl(STOVE, "surface"),
    ),
    initial_state=frozenset({AT_LEFT, HANDEMPTY}),
    initial_false=frozenset({AT_STOVE, HOLDING}),
    goal=frozenset({AT_STOVE}),
)


def _observation(pixel: int = 0) -> dict[str, Any]:
    return {
        "agentview_image": np.full((3, 4, 3), pixel, dtype=np.uint8),
        "robot0_eye_in_hand_image": np.full((3, 4, 3), pixel + 1, dtype=np.uint8),
    }


def _response(
    *,
    epoch_id: int = 0,
    graph_version: str | None = "graph-v1",
    values: Mapping[Fact, TruthValue] | None = None,
) -> dict[str, Any]:
    selected = values or {
        AT_LEFT: TruthValue.TRUE,
        AT_STOVE: TruthValue.FALSE,
        HOLDING: TruthValue.FALSE,
        HANDEMPTY: TruthValue.TRUE,
    }
    return {
        "epoch_id": epoch_id,
        "graph_version": graph_version,
        "facts": [
            {
                "fact": fact.pddl(),
                "truth": selected[fact].value,
                "evidence": "both",
            }
            for fact in sorted(selected, key=lambda item: item.pddl())
        ],
    }


class FakeClient:
    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def complete_json(self, **kwargs: Any) -> Mapping[str, Any]:
        self.calls.append(kwargs)
        return self.responses.pop(0)


class ImageTools:
    @staticmethod
    def resize_with_pad(image: np.ndarray, height: int, width: int) -> np.ndarray:
        assert (height, width) == (224, 224)
        return image

    @staticmethod
    def convert_to_uint8(image: np.ndarray) -> np.ndarray:
        return np.asarray(image, dtype=np.uint8)


def _grounder(client: FakeClient, observation: dict[str, Any] | None = None):
    return Gpt4oGrounder(
        client=client,
        observation_store=LiberoObservationStore(observation or _observation()),
        problem=PROBLEM,
        monitored_facts=UNIVERSE,
        task_instruction="Put the moka pot on the stove.",
        graph_version="graph-v1",
        image_tools=ImageTools(),
    )


def _context(*, epoch_id: int = 0, graph_version: str | None = "graph-v1"):
    return ContextEnvelope(
        phase=ContextPhase.PRE_DISPATCH_FACTS,
        goal_mode=GoalMode.METADATA_ASSISTED,
        request_id="grounding-test",
        request_generation=0,
        episode_id="episode-test",
        goal_id="goal-test",
        goal_epoch=0,
        epoch_id=epoch_id,
        graph_version=graph_version,
        occurrence_id="a0",
        attempt_id=None,
        certificate_hash=None,
        safety_epoch=None,
    )


def test_snapshot_is_complete_audited_and_uses_both_oriented_images() -> None:
    client = FakeClient([_response()])
    grounder = _grounder(client)

    snapshot = grounder.peek_snapshot()

    assert snapshot.true_facts == frozenset({AT_LEFT, HANDEMPTY})
    assert snapshot.false_facts == frozenset({AT_STOVE, HOLDING})
    assert snapshot.fact_universe == UNIVERSE
    assert snapshot.truth(AT_LEFT) is TruthValue.TRUE
    payload = json.loads(snapshot.evidence_payload_json)
    assert payload["epoch_id"] == 0
    assert len(payload["observation_hash"]) == 64
    assert hashlib.sha256(snapshot.evidence_payload_json.encode()).hexdigest() == snapshot.evidence_hash
    call = client.calls[0]
    assert call["purpose"] == "state_gate"
    assert len(call["images"]) == 2
    np.testing.assert_array_equal(
        call["images"][0], _observation()["agentview_image"][::-1, ::-1]
    )
    np.testing.assert_array_equal(
        call["images"][1],
        _observation()["robot0_eye_in_hand_image"][::-1, ::-1],
    )
    assert set(call["schema"]["properties"]) == {"epoch_id", "graph_version", "facts"}


def test_unknown_remains_unknown_and_blocks_required_grounding() -> None:
    values = {
        AT_LEFT: TruthValue.UNKNOWN,
        AT_STOVE: TruthValue.FALSE,
        HOLDING: TruthValue.UNKNOWN,
        HANDEMPTY: TruthValue.UNKNOWN,
    }
    grounder = _grounder(FakeClient([_response(values=values)]))

    response = grounder.ground(
        ContextPhase.PRE_DISPATCH_FACTS,
        _context(),
        frozenset({AT_LEFT}),
    )

    assert response.status is GroundingStatus.STATE_GROUNDING_FAILURE
    assert response.snapshot is None
    assert response.reason == "required facts UNKNOWN: at(moka_pot_1, left_region)"


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (
            lambda: {
                **_response(),
                "facts": _response()["facts"][:-1],
            },
            "complete partition",
        ),
        (
            lambda: {
                **_response(),
                "facts": _response()["facts"] + [_response()["facts"][0]],
            },
            "duplicate fact",
        ),
        (
            lambda: _response(epoch_id=1),
            "epoch mismatch",
        ),
        (
            lambda: _response(graph_version="stale-graph"),
            "graph version mismatch",
        ),
        (
            lambda: _response(
                values={
                    AT_LEFT: TruthValue.TRUE,
                    AT_STOVE: TruthValue.TRUE,
                    HOLDING: TruthValue.FALSE,
                    HANDEMPTY: TruthValue.TRUE,
                }
            ),
            "multiple locations",
        ),
    ],
)
def test_snapshot_rejects_unusable_model_partitions(response, message: str) -> None:
    grounder = _grounder(FakeClient([response()]))

    with pytest.raises(Gpt4oGroundingError, match=message):
        grounder.peek_snapshot()


def test_same_observation_is_cached_but_strict_read_is_fresh() -> None:
    client = FakeClient([_response(), _response()])
    grounder = _grounder(client)

    first = grounder.peek_snapshot()
    cached = grounder.peek_snapshot()
    strict = grounder.read_strict_snapshot()

    assert cached is first
    assert strict == first
    assert len(client.calls) == 2


def test_ground_rejects_stale_context_without_calling_model() -> None:
    client = FakeClient([_response()])
    grounder = _grounder(client)

    response = grounder.ground(
        ContextPhase.PRE_DISPATCH_FACTS,
        _context(epoch_id=1),
        frozenset({AT_LEFT}),
    )

    assert response.status is GroundingStatus.STATE_GROUNDING_FAILURE
    assert response.reason == "context/epoch mismatch"
    assert client.calls == []


def test_action_progress_uses_visual_fact_snapshot() -> None:
    grounder = _grounder(FakeClient([_response()]))
    action = GroundAction(
        schema="place-on",
        arguments=(OBJECT, LEFT, STOVE),
        preconditions=frozenset({AT_LEFT, HANDEMPTY}),
        add_effects=frozenset({AT_STOVE}),
        del_effects=frozenset({AT_LEFT}),
        repeatable=True,
    )

    completion, primary, diverged = grounder.observe_action_progress_details(action)

    assert not completion and not primary and not diverged
    assert grounder.detector_calls == 1
