from __future__ import annotations

import hashlib
import json
from typing import Any, FrozenSet, Iterable, Mapping

import numpy as np

from pi05_libero_repro.logiv.controller import GroundingResponse, GroundingStatus
from pi05_libero_repro.logiv.gpt4o import Gpt4oClient
from pi05_libero_repro.logiv.libero_adapter import LiberoObservationStore
from pi05_libero_repro.logiv.model import (
    ContextEnvelope,
    ContextPhase,
    Fact,
    FactSnapshot,
    GroundAction,
    TaskProblem,
    TruthValue,
    fact_pddl_sort_key,
    fact_universe_sha256,
    parse_pddl_fact,
)
class Gpt4oGroundingError(RuntimeError):
    """GPT-4o did not return a usable complete symbolic observation."""


class Gpt4oGrounder:
    """Evidence-backed VLM grounding over a fixed, finite fact universe."""

    def __init__(
        self,
        *,
        client: Gpt4oClient,
        observation_store: LiberoObservationStore,
        problem: TaskProblem,
        monitored_facts: Iterable[Fact],
        task_instruction: str,
        graph_version: str | None,
        image_tools: Any | None = None,
    ) -> None:
        facts = frozenset(monitored_facts)
        if not facts:
            raise ValueError("GPT-4o grounding requires a nonempty fact universe")
        if not task_instruction:
            raise ValueError("task instruction must be nonempty")
        self.client = client
        self.store = observation_store
        self.problem = problem
        self.monitored_facts = facts
        self.task_instruction = task_instruction
        self.graph_version = graph_version
        self.image_tools = image_tools
        self.detector_calls = 0
        self.ground_calls = 0
        self._cache: dict[tuple[int, str, str | None], FactSnapshot] = {}

    def acquire_epoch(self, phase: ContextPhase) -> int:
        del phase
        return self.store.epoch_id

    def set_graph_version(self, graph_version: str | None) -> None:
        self.graph_version = graph_version

    def _schema(self) -> dict[str, Any]:
        fact_names = [
            fact.pddl()
            for fact in sorted(self.monitored_facts, key=fact_pddl_sort_key)
        ]
        count = len(fact_names)
        return {
            "type": "object",
            "properties": {
                "epoch_id": {"type": "integer"},
                "graph_version": (
                    {"type": "string"}
                    if self.graph_version is not None
                    else {"type": "null"}
                ),
                "facts": {
                    "type": "array",
                    "minItems": count,
                    "maxItems": count,
                    "items": {
                        "type": "object",
                        "properties": {
                            "fact": {"type": "string", "enum": fact_names},
                            "truth": {
                                "type": "string",
                                "enum": [value.value for value in TruthValue],
                            },
                            "evidence": {
                                "type": "string",
                                "enum": ["main", "wrist", "both", "unclear"],
                            },
                        },
                        "required": ["fact", "truth", "evidence"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["epoch_id", "graph_version", "facts"],
            "additionalProperties": False,
        }

    def _images(self, observation: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        try:
            main = np.asarray(observation["agentview_image"])[::-1, ::-1]
            wrist = np.asarray(observation["robot0_eye_in_hand_image"])[::-1, ::-1]
        except (KeyError, TypeError, ValueError) as error:
            raise Gpt4oGroundingError("observation is missing the two RGB cameras") from error
        images: list[np.ndarray] = []
        for image in (main, wrist):
            if self.image_tools is not None:
                image = self.image_tools.resize_with_pad(image, 224, 224)
                image = self.image_tools.convert_to_uint8(image)
            else:
                image = np.asarray(image, dtype=np.uint8)
            if image.ndim != 3 or image.shape[2] != 3:
                raise Gpt4oGroundingError("camera image must have shape HxWx3")
            images.append(image)
        return images[0], images[1]

    @staticmethod
    def _observation_hash(observation: Mapping[str, Any]) -> str:
        digest = hashlib.sha256(b"LOGIV_GPT4O_VISUAL_OBSERVATION_V1\0")
        for key in ("agentview_image", "robot0_eye_in_hand_image"):
            try:
                image = np.asarray(observation[key])
            except (KeyError, TypeError, ValueError) as error:
                raise Gpt4oGroundingError(
                    "observation is missing the two RGB cameras"
                ) from error
            digest.update(key.encode("utf-8") + b"\0")
            digest.update(str(image.dtype).encode("ascii") + b"\0")
            digest.update(json.dumps(image.shape).encode("ascii") + b"\0")
            digest.update(np.ascontiguousarray(image).tobytes(order="C"))
        return digest.hexdigest()

    def _prompt(self, epoch_id: int) -> str:
        values = "\n".join(
            fact.pddl()
            for fact in sorted(self.monitored_facts, key=fact_pddl_sort_key)
        )
        return (
            f"Task instruction: {self.task_instruction}\n"
            f"Physical epoch: {epoch_id}\n"
            f"Graph version: {self.graph_version}\n"
            "Observe the main camera and wrist camera. Classify every listed "
            "PDDL fact. Use UNKNOWN whenever the images do not provide enough "
            "evidence; absence from view is not FALSE. Return each fact exactly once.\n"
            f"Facts:\n{values}"
        )

    def _validate_partition(
        self,
        response: Mapping[str, Any],
        *,
        epoch_id: int,
    ) -> dict[Fact, TruthValue]:
        if set(response) != {"epoch_id", "graph_version", "facts"}:
            raise Gpt4oGroundingError("response fields mismatch")
        if response["epoch_id"] != epoch_id:
            raise Gpt4oGroundingError("epoch mismatch")
        if response["graph_version"] != self.graph_version:
            raise Gpt4oGroundingError("graph version mismatch")
        records = response["facts"]
        if not isinstance(records, list):
            raise Gpt4oGroundingError("facts are not a complete partition")
        values: dict[Fact, TruthValue] = {}
        for record in records:
            if not isinstance(record, Mapping) or set(record) != {
                "fact",
                "truth",
                "evidence",
            }:
                raise Gpt4oGroundingError("fact record is malformed")
            try:
                fact = parse_pddl_fact(record["fact"])
            except (TypeError, ValueError) as error:
                raise Gpt4oGroundingError("fact record is malformed") from error
            if fact in values:
                raise Gpt4oGroundingError("duplicate fact")
            if fact not in self.monitored_facts:
                raise Gpt4oGroundingError("fact is outside the monitored universe")
            try:
                values[fact] = TruthValue(record["truth"])
            except (TypeError, ValueError) as error:
                raise Gpt4oGroundingError("fact truth value is invalid") from error
            if record["evidence"] not in {"main", "wrist", "both", "unclear"}:
                raise Gpt4oGroundingError("fact evidence label is invalid")
        if frozenset(values) != self.monitored_facts:
            raise Gpt4oGroundingError("facts do not form a complete partition")
        self._validate_exclusivity(values)
        return values

    @staticmethod
    def _validate_exclusivity(values: Mapping[Fact, TruthValue]) -> None:
        movable_names = {
            fact.arguments[0]
            for fact in values
            if fact.predicate in {"at", "holding"} and fact.arguments
        }
        for object_name in movable_names:
            locations = [
                fact
                for fact, truth in values.items()
                if truth is TruthValue.TRUE
                and fact.arguments
                and fact.arguments[0] == object_name
                and fact.predicate in {"at", "holding"}
            ]
            if len(locations) > 1:
                raise Gpt4oGroundingError(
                    f"multiple locations for {object_name}"
                )
        handempty = values.get(Fact("handempty"), TruthValue.UNKNOWN)
        if handempty is TruthValue.TRUE and any(
            fact.predicate == "holding" and truth is TruthValue.TRUE
            for fact, truth in values.items()
        ):
            raise Gpt4oGroundingError("handempty conflicts with holding")
        for positive, negative in (("open", "closed"), ("powered-on", "powered-off")):
            names = {
                fact.arguments[0]
                for fact in values
                if fact.predicate in {positive, negative} and fact.arguments
            }
            for name in names:
                if (
                    values.get(Fact(positive, (name,))) is TruthValue.TRUE
                    and values.get(Fact(negative, (name,))) is TruthValue.TRUE
                ):
                    raise Gpt4oGroundingError(
                        f"conflicting {positive}/{negative} facts for {name}"
                    )

    def _snapshot(self, *, force_refresh: bool = False) -> FactSnapshot:
        epoch_id, observation, _ = self.store.read()
        observation_hash = self._observation_hash(observation)
        cache_key = (epoch_id, observation_hash, self.graph_version)
        if not force_refresh and cache_key in self._cache:
            return self._cache[cache_key]
        response = self.client.complete_json(
            purpose="state_gate",
            system=(
                "You are LOGIV's conservative visual state gate. Report only "
                "image-supported symbolic facts and preserve uncertainty."
            ),
            text=self._prompt(epoch_id),
            images=self._images(observation),
            schema_name="logiv_state_gate",
            schema=self._schema(),
        )
        values = self._validate_partition(response, epoch_id=epoch_id)
        payload = {
            "epoch_id": epoch_id,
            "observation_hash": observation_hash,
            "values": [
                [fact.pddl(), values[fact].value]
                for fact in sorted(values, key=fact_pddl_sort_key)
            ],
            "dominance_overrides": [],
        }
        payload_json = json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        universe_version = "logiv-gpt4o-grounding-v1"
        snapshot = FactSnapshot(
            epoch_id=epoch_id,
            true_facts=frozenset(
                fact for fact, truth in values.items() if truth is TruthValue.TRUE
            ),
            false_facts=frozenset(
                fact for fact, truth in values.items() if truth is TruthValue.FALSE
            ),
            evidence_hash=hashlib.sha256(payload_json.encode("utf-8")).hexdigest(),
            fact_universe=self.monitored_facts,
            fact_universe_version=universe_version,
            fact_universe_sha256=fact_universe_sha256(
                universe_version, self.monitored_facts
            ),
            evidence_payload_json=payload_json,
        )
        self._cache[cache_key] = snapshot
        return snapshot

    def ground(
        self,
        phase: ContextPhase,
        context: ContextEnvelope,
        required_facts: FrozenSet[Fact],
    ) -> GroundingResponse:
        self.ground_calls += 1
        failure = (
            GroundingStatus.POST_STOP_GROUNDING_FAILURE
            if phase is ContextPhase.POST_STOP_FACTS
            else GroundingStatus.STATE_GROUNDING_FAILURE
        )
        if (
            phase is not context.phase
            or context.epoch_id != self.store.epoch_id
            or context.graph_version != self.graph_version
        ):
            return GroundingResponse(
                status=failure, context=context, reason="context/epoch mismatch"
            )
        try:
            snapshot = self._snapshot()
        except (Gpt4oGroundingError, TypeError, ValueError) as error:
            return GroundingResponse(status=failure, context=context, reason=str(error))
        unknown = snapshot.unknown(required_facts)
        if unknown:
            return GroundingResponse(
                status=failure,
                context=context,
                reason="required facts UNKNOWN: "
                + ", ".join(map(str, sorted(unknown))),
            )
        return GroundingResponse(
            status=GroundingStatus.OK, context=context, snapshot=snapshot
        )

    def peek_snapshot(self) -> FactSnapshot:
        return self._snapshot()

    def peek_advisory_partial_snapshot(self) -> FactSnapshot:
        return self._snapshot()

    def read_strict_snapshot(self) -> FactSnapshot:
        return self._snapshot(force_refresh=True)

    def effects_satisfied(self, action: GroundAction) -> bool:
        return self.observe_action_progress(action)[0]

    def observe_action_progress(
        self,
        action: GroundAction,
        *,
        completion_positive: FrozenSet[Fact] | None = None,
        completion_negative: FrozenSet[Fact] | None = None,
    ) -> tuple[bool, bool]:
        completion, _, diverged = self.observe_action_progress_details(
            action,
            completion_positive=completion_positive,
            completion_negative=completion_negative,
        )
        return completion, diverged

    def observe_action_progress_details(
        self,
        action: GroundAction,
        *,
        completion_positive: FrozenSet[Fact] | None = None,
        completion_negative: FrozenSet[Fact] | None = None,
    ) -> tuple[bool, bool, bool]:
        self.detector_calls += 1
        positive = action.add_effects if completion_positive is None else completion_positive
        negative = action.del_effects if completion_negative is None else completion_negative
        try:
            snapshot = self._snapshot()
        except (Gpt4oGroundingError, TypeError, ValueError):
            return False, False, False
        completion = snapshot.satisfies(positive=positive, negative=negative)
        primary = snapshot.satisfies(
            positive=action.add_effects, negative=action.del_effects
        )
        sourced_place = action.schema in {"place-on", "place-in", "place-relative"}
        held_place = action.schema in {
            "place-held-on",
            "place-held-in",
            "place-held-relative",
            "put-down",
        }
        if not (sourced_place or held_place):
            return completion, primary, False
        object_name = action.arguments[0]
        source = action.arguments[1] if sourced_place else None
        target = action.arguments[2] if sourced_place else action.arguments[1]
        required_false = {
            Fact("at", (object_name, target)),
            Fact("holding", (object_name,)),
        }
        if source is not None:
            required_false.add(Fact("at", (object_name, source)))
        nominal = {target} | ({source} if source is not None else set())
        alternative = any(
            fact.predicate == "at"
            and fact.arguments[0] == object_name
            and fact.arguments[1] not in nominal
            for fact in snapshot.true_facts
        )
        return completion, primary, required_false <= snapshot.false_facts and alternative
