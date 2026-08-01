from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import threading
import time
from typing import Any, FrozenSet, Iterable, Mapping

import numpy as np

from pi05_libero_repro.logiv.controller import (
    DispatchStart,
    DispatchStatus,
    ExecutorOutcome,
    ExecutorStatus,
    GroundingResponse,
    GroundingStatus,
)
from pi05_libero_repro.logiv.domain import _is_subtype
from pi05_libero_repro.logiv.model import (
    ContextEnvelope,
    ContextPhase,
    Fact,
    FactSnapshot,
    GroundAction,
    TaskProblem,
    TruthValue,
)
from pi05_libero_repro.logiv.prompts import SubtaskPromptRenderer
from pi05_libero_repro.protocol import EpisodeInvalid, prepare_observation


def _inner_env(env: Any) -> Any:
    current = env
    seen: set[int] = set()
    while not hasattr(current, "_eval_predicate") and hasattr(current, "env"):
        if id(current) in seen:
            break
        seen.add(id(current))
        current = current.env
    return current


def _observation_hash(observation: Mapping[str, Any], epoch_id: int) -> str:
    digest = hashlib.sha256(str(epoch_id).encode("ascii"))
    for key in sorted(observation):
        value = observation[key]
        digest.update(key.encode("utf-8"))
        if isinstance(value, np.ndarray):
            digest.update(str(value.shape).encode("ascii"))
            digest.update(str(value.dtype).encode("ascii"))
            digest.update(np.ascontiguousarray(value).tobytes())
        else:
            digest.update(repr(value).encode("utf-8"))
    return digest.hexdigest()


@dataclass(frozen=True)
class TaskBinding:
    task_id: int
    frozen: bool
    registered_objects: tuple[str, ...]
    symbol_bindings: Mapping[str, Mapping[str, str]]
    supported_action_schemas: FrozenSet[str]
    recovery_schemas: FrozenSet[str]

    @classmethod
    def from_manifest(cls, path: Path | str, task_id: int) -> "TaskBinding":
        payload = json.loads(Path(path).read_text())
        tasks = payload.get("tasks", [])
        task_ids = [int(item["task_id"]) for item in tasks]
        if payload.get("frozen") is not True or task_ids != list(range(10)):
            raise ValueError("coverage manifest must be frozen with exact task IDs 0..9")
        try:
            item = next(entry for entry in tasks if int(entry["task_id"]) == task_id)
        except StopIteration as error:
            raise ValueError(f"task {task_id} is outside the frozen manifest") from error
        return cls(
            task_id=task_id,
            frozen=True,
            registered_objects=tuple(item["registered_objects"]),
            symbol_bindings={
                str(key): {str(k): str(v) for k, v in value.items()}
                for key, value in item.get("symbol_bindings", {}).items()
            },
            supported_action_schemas=frozenset(item["supported_action_schemas"]),
            recovery_schemas=frozenset(item["recovery_schemas"]),
        )

    def resolve(self, symbol: str) -> tuple[str, str | None]:
        binding = self.symbol_bindings.get(symbol, {})
        return binding.get("libero_id", symbol), binding.get("kind")


def monitored_fact_universe(problem: TaskProblem) -> FrozenSet[Fact]:
    types = problem.object_types
    movable = tuple(name for name, kind in types.items() if _is_subtype(kind, "movable"))
    locations = tuple(name for name, kind in types.items() if _is_subtype(kind, "location"))
    accesses = tuple(name for name, kind in types.items() if _is_subtype(kind, "access"))
    containers = tuple(
        name for name, kind in types.items() if _is_subtype(kind, "container-region")
    )
    devices = tuple(name for name, kind in types.items() if _is_subtype(kind, "switchable"))
    facts = {Fact("handempty")}
    facts.update(Fact("holding", (obj,)) for obj in movable)
    facts.update(Fact("at", (obj, location)) for obj in movable for location in locations)
    facts.update(Fact(predicate, (access,)) for access in accesses for predicate in ("open", "closed"))
    facts.update(
        Fact(predicate, (device,))
        for device in devices
        for predicate in ("powered-on", "powered-off")
    )
    facts.update(Fact("accessible", (container, access)) for container in containers for access in accesses)
    facts.update(problem.initial_state | problem.initial_false | problem.goal | problem.negative_goal)
    return frozenset(facts)


class LiberoObservationStore:
    """Thread-safe latest synchronous simulator observation and physical epoch."""

    def __init__(self, observation: Mapping[str, Any], *, epoch_id: int = 0) -> None:
        if epoch_id < 0:
            raise ValueError("epoch_id must be nonnegative")
        self._lock = threading.RLock()
        self._observation = dict(observation)
        self._epoch_id = epoch_id
        self._updated_at = time.monotonic()

    @property
    def epoch_id(self) -> int:
        with self._lock:
            return self._epoch_id

    @property
    def age_seconds(self) -> float:
        with self._lock:
            return time.monotonic() - self._updated_at

    def read(self) -> tuple[int, dict[str, Any], float]:
        with self._lock:
            return self._epoch_id, dict(self._observation), self._updated_at

    def update(self, observation: Mapping[str, Any]) -> int:
        with self._lock:
            self._observation = dict(observation)
            self._epoch_id += 1
            self._updated_at = time.monotonic()
            return self._epoch_id


class GroundingError(RuntimeError):
    pass


class LiberoOracleGrounder:
    """Development-only privileged grounding from current LIBERO simulator state."""

    def __init__(
        self,
        env: Any,
        observation_store: LiberoObservationStore,
        binding: TaskBinding,
        monitored_facts: Iterable[Fact],
    ) -> None:
        self.env = env
        self.inner = _inner_env(env)
        self.store = observation_store
        self.binding = binding
        self.monitored_facts = frozenset(monitored_facts)
        self.detector_calls = 0
        self.ground_calls = 0

    def acquire_epoch(self, phase: ContextPhase) -> int:
        del phase
        return self.store.epoch_id

    @staticmethod
    def _location_predicate(location: str) -> str:
        if any(token in location for token in ("contain_region", "heating_region", "bottom_region")):
            return "in"
        return "on"

    def _predicate(self, state: list[str]) -> TruthValue:
        try:
            value = self.inner._eval_predicate(state)
        except Exception:
            return TruthValue.UNKNOWN
        if isinstance(value, (bool, np.bool_)):
            return TruthValue.TRUE if bool(value) else TruthValue.FALSE
        return TruthValue.UNKNOWN

    def _holding(self, object_name: str) -> TruthValue:
        try:
            object_model = self.inner.objects_dict[object_name]
            robot = self.inner.robots[0]
            value = self.inner._check_grasp(robot.gripper, object_model.contact_geoms)
        except (KeyError, IndexError, TypeError, AttributeError):
            return TruthValue.UNKNOWN
        return TruthValue.TRUE if bool(value) else TruthValue.FALSE

    def _accessible(self, container: str, access: str) -> TruthValue:
        resolved, kind = self.binding.resolve(access)
        if kind == "always_open_access" and resolved == container:
            return TruthValue.TRUE
        access_root = access[: -len("_access")] if access.endswith("_access") else access
        if access_root in container or resolved in container or container in resolved:
            return TruthValue.TRUE
        return TruthValue.FALSE

    def _support_contact(self, object_name: str, support_name: str) -> TruthValue:
        """Ground workspace support from the current MuJoCo contact graph.

        LIBERO's kitchen table is an arena workspace, not an entry in
        ``object_states_dict``.  Consequently ``On(obj, kitchen_table)`` is not
        a valid ordinary BDDL predicate query.  Contact pairs are a deterministic
        geometry source for that registered alias and avoid interpreting absent
        region membership as a table relation.
        """

        try:
            model = self.inner.sim.model
            data = self.inner.sim.data
            object_model = self.inner.get_object(object_name)
            object_geom_ids = {
                int(model.geom_name2id(name)) for name in object_model.contact_geoms
            }
            support_geom_ids = {
                geom_id
                for geom_id in range(int(model.ngeom))
                if (
                    (geom_name := model.geom_id2name(geom_id)) is not None
                    and (
                        geom_name == f"{support_name}_collision"
                        or (
                            geom_name.endswith("table_collision")
                            and support_name.endswith("table")
                        )
                    )
                )
            }
            if not object_geom_ids or not support_geom_ids:
                return TruthValue.UNKNOWN
            for index in range(int(data.ncon)):
                contact = data.contact[index]
                pair = {int(contact.geom1), int(contact.geom2)}
                if pair & object_geom_ids and pair & support_geom_ids:
                    return TruthValue.TRUE
            return TruthValue.FALSE
        except (KeyError, IndexError, TypeError, ValueError, AttributeError):
            return TruthValue.UNKNOWN

    def _support_geometry(self, object_name: str, support_name: str) -> TruthValue:
        """Check a registered workspace plane without using task-goal regions."""

        if not support_name.endswith("table"):
            return TruthValue.UNKNOWN
        try:
            workspace = np.asarray(self.inner.workspace_offset, dtype=np.float64)
            size_value = getattr(self.inner, f"{support_name}_full_size", None)
            if size_value is None:
                size_value = self.inner.table_full_size
            full_size = np.asarray(
                size_value,
                dtype=np.float64,
            )
            object_model = self.inner.get_object(object_name)
            position = np.asarray(
                self.inner.sim.data.body_xpos[self.inner.obj_body_id[object_name]],
                dtype=np.float64,
            )
            bottom_offset = np.asarray(object_model.bottom_offset, dtype=np.float64)
        except (KeyError, IndexError, TypeError, ValueError, AttributeError):
            return TruthValue.UNKNOWN
        if (
            workspace.shape != (3,)
            or full_size.ndim != 1
            or full_size.size < 2
            or position.shape != (3,)
            or bottom_offset.shape != (3,)
            or not all(
                np.isfinite(value).all()
                for value in (workspace, full_size, position, bottom_offset)
            )
        ):
            return TruthValue.UNKNOWN
        inside_xy = bool(
            np.all(np.abs(position[:2] - workspace[:2]) <= full_size[:2] / 2.0)
        )
        bottom_height = float(position[2] + bottom_offset[2])
        near_tabletop = abs(bottom_height - float(workspace[2])) <= 0.04
        return TruthValue.TRUE if inside_xy and near_tabletop else TruthValue.FALSE

    def _switch_truth(self, resolved: str, *, powered_on: bool) -> TruthValue:
        """Use the raw joint and declared fixture ranges to close LIBERO's qpos==0 gap."""

        try:
            state = self.inner.object_states_dict[resolved]
            model = self.inner.get_object(resolved)
            articulation = model.object_properties["articulation"]
            on_threshold = min(articulation["default_turnon_ranges"])
            off_threshold = max(articulation["default_turnoff_ranges"])
            qposes = []
            for joint in model.joints:
                address = state.env.sim.model.get_joint_qpos_addr(joint)
                qposes.append(float(state.env.sim.data.qpos[address]))
        except (KeyError, IndexError, TypeError, ValueError, AttributeError):
            return self._predicate(
                ["turnon" if powered_on else "turnoff", resolved]
            )
        if any(value >= on_threshold for value in qposes):
            return TruthValue.TRUE if powered_on else TruthValue.FALSE
        if qposes and all(value <= off_threshold for value in qposes):
            return TruthValue.FALSE if powered_on else TruthValue.TRUE
        return TruthValue.UNKNOWN

    def _truth(self, fact: Fact) -> TruthValue:
        predicate = fact.predicate
        if predicate == "at":
            object_name, location = fact.arguments
            resolved, kind = self.binding.resolve(location)
            if kind == "support_surface_alias":
                other_locations = sorted(
                    candidate.arguments[1]
                    for candidate in self.monitored_facts
                    if candidate.predicate == "at"
                    and candidate.arguments[0] == object_name
                    and candidate.arguments[1] != location
                )
                other_values = [
                    self._predicate(
                        [
                            self._location_predicate(other),
                            object_name,
                            self.binding.resolve(other)[0],
                        ]
                    )
                    for other in other_locations
                ]
                if any(value is TruthValue.TRUE for value in other_values):
                    return TruthValue.FALSE
                broad = self._predicate(["on", object_name, resolved])
                contact = self._support_contact(object_name, resolved)
                geometry = self._support_geometry(object_name, resolved)
                held = self._holding(object_name)
                supported = (
                    TruthValue.TRUE
                    if TruthValue.TRUE in {broad, contact, geometry}
                    else TruthValue.FALSE
                    if all(
                        value is TruthValue.FALSE
                        for value in (broad, contact, geometry)
                    )
                    else TruthValue.UNKNOWN
                )
                if supported is TruthValue.TRUE and held is TruthValue.FALSE and all(
                    value is TruthValue.FALSE for value in other_values
                ):
                    return TruthValue.TRUE
                if supported is TruthValue.FALSE or held is TruthValue.TRUE:
                    return TruthValue.FALSE
                return TruthValue.UNKNOWN
            return self._predicate([self._location_predicate(location), object_name, resolved])
        if predicate == "holding":
            return self._holding(fact.arguments[0])
        if predicate == "open" or predicate == "closed":
            access = fact.arguments[0]
            resolved, kind = self.binding.resolve(access)
            if kind == "always_open_access":
                return TruthValue.TRUE if predicate == "open" else TruthValue.FALSE
            return self._predicate(["open" if predicate == "open" else "close", resolved])
        if predicate in {"powered-on", "powered-off"}:
            device, = fact.arguments
            resolved, _ = self.binding.resolve(device)
            return self._switch_truth(resolved, powered_on=predicate == "powered-on")
        if predicate == "accessible":
            return self._accessible(*fact.arguments)
        return TruthValue.UNKNOWN

    def _snapshot(self, required: FrozenSet[Fact]) -> FactSnapshot:
        epoch_id, observation, _ = self.store.read()
        values: dict[Fact, TruthValue] = {}
        for fact in sorted(self.monitored_facts | required):
            if fact.predicate != "handempty":
                values[fact] = self._truth(fact)

        holding_facts = sorted(
            fact for fact in self.monitored_facts | required if fact.predicate == "holding"
        )
        dominance_overrides: list[tuple[str, str, str]] = []
        for holding_fact in holding_facts:
            if values.get(holding_fact) is not TruthValue.TRUE:
                continue
            object_name = holding_fact.arguments[0]
            for fact, value in tuple(values.items()):
                if (
                    fact.predicate == "at"
                    and fact.arguments[0] == object_name
                    and value is TruthValue.TRUE
                ):
                    values[fact] = TruthValue.FALSE
                    dominance_overrides.append(
                        (holding_fact.pddl(), fact.pddl(), "reliable-holding-over-at")
                    )
        holding_values = [values.get(fact, TruthValue.UNKNOWN) for fact in holding_facts]
        if any(value is TruthValue.TRUE for value in holding_values):
            values[Fact("handempty")] = TruthValue.FALSE
        elif holding_values and all(value is TruthValue.FALSE for value in holding_values):
            values[Fact("handempty")] = TruthValue.TRUE
        else:
            values[Fact("handempty")] = TruthValue.UNKNOWN

        true_facts = frozenset(fact for fact, value in values.items() if value is TruthValue.TRUE)
        false_facts = frozenset(fact for fact, value in values.items() if value is TruthValue.FALSE)
        movable = {fact.arguments[0] for fact in holding_facts}
        for object_name in movable:
            candidates = {
                fact
                for fact in values
                if (fact.predicate == "holding" and fact.arguments == (object_name,))
                or (fact.predicate == "at" and fact.arguments[0] == object_name)
            }
            confirmed = candidates & true_facts
            if len(confirmed) != 1:
                raise GroundingError(
                    f"exactly-one violation for {object_name}: "
                    f"confirmed={sorted(map(str, confirmed))}"
                )
        for positive_name, negative_name, predicate in (
            ("open", "closed", "access"),
            ("powered-on", "powered-off", "switchable"),
        ):
            names = {
                fact.arguments[0]
                for fact in values
                if fact.predicate in {positive_name, negative_name}
            }
            for name in names:
                pair = {
                    Fact(positive_name, (name,)),
                    Fact(negative_name, (name,)),
                }
                if len(pair & true_facts) != 1:
                    raise GroundingError(
                        f"exactly-one {predicate} violation for {name}"
                    )

        evidence_payload = {
            "epoch_id": epoch_id,
            "observation_hash": _observation_hash(observation, epoch_id),
            "values": [(fact.pddl(), values[fact].value) for fact in sorted(values)],
            "dominance_overrides": dominance_overrides,
        }
        evidence_hash = hashlib.sha256(
            json.dumps(evidence_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return FactSnapshot(epoch_id, true_facts, false_facts, evidence_hash)

    def ground(
        self,
        phase: ContextPhase,
        context: ContextEnvelope,
        required_facts: FrozenSet[Fact],
    ) -> GroundingResponse:
        self.ground_calls += 1
        status = (
            GroundingStatus.POST_STOP_GROUNDING_FAILURE
            if phase is ContextPhase.POST_STOP_FACTS
            else GroundingStatus.STATE_GROUNDING_FAILURE
        )
        if phase is not context.phase or context.epoch_id != self.store.epoch_id:
            return GroundingResponse(status=status, context=context, reason="context/epoch mismatch")
        try:
            snapshot = self._snapshot(required_facts)
        except GroundingError as error:
            return GroundingResponse(status=status, context=context, reason=str(error))
        unknown = snapshot.unknown(required_facts)
        if unknown:
            return GroundingResponse(
                status=status,
                context=context,
                reason="required facts UNKNOWN: " + ", ".join(map(str, sorted(unknown))),
            )
        return GroundingResponse(status=GroundingStatus.OK, context=context, snapshot=snapshot)

    def peek_snapshot(self) -> FactSnapshot:
        return self._snapshot(frozenset())

    def effects_satisfied(self, action: GroundAction) -> bool:
        return self.observe_action_progress(action)[0]

    def observe_action_progress(self, action: GroundAction) -> tuple[bool, bool]:
        """Return (declared effects satisfied, object at a different known location).

        The second signal is intentionally limited to grounded place macros that
        carry an explicit source and target.  It does not infer a location from
        missing predicates: a different registered ``at`` fact must be TRUE and
        source, target, and holding must all be explicitly FALSE.
        """

        self.detector_calls += 1
        required = action.add_effects | action.del_effects
        location_schema = action.schema in {"place-on", "place-in", "place-relative"}
        if location_schema and len(action.arguments) >= 3:
            object_name, source, target = action.arguments[:3]
            required |= frozenset(
                {
                    Fact("at", (object_name, source)),
                    Fact("at", (object_name, target)),
                    Fact("holding", (object_name,)),
                }
            )
        try:
            snapshot = self._snapshot(required)
        except GroundingError:
            return False, False
        effects_satisfied = snapshot.satisfies(
            positive=action.add_effects,
            negative=action.del_effects,
        )
        if not location_schema or len(action.arguments) < 3:
            return effects_satisfied, False
        object_name, source, target = action.arguments[:3]
        required_false = {
            Fact("at", (object_name, source)),
            Fact("at", (object_name, target)),
            Fact("holding", (object_name,)),
        }
        alternative_location = any(
            fact.predicate == "at"
            and fact.arguments[0] == object_name
            and fact.arguments[1] not in {source, target}
            for fact in snapshot.true_facts
        )
        target_diverged = required_false <= snapshot.false_facts and alternative_location
        return effects_satisfied, target_diverged


@dataclass(frozen=True)
class SafetyDecision:
    status: DispatchStatus
    safety_epoch: int | None = None
    reason: str = ""


class SimulatorSafetySupervisor:
    """Independent fail-closed checks around the synchronous simulator command queue."""

    def __init__(
        self,
        *,
        watchdog_seconds: float,
        action_limit: float = 1.1,
        workspace_low: tuple[float, float, float] = (-2.0, -2.0, -0.1),
        workspace_high: tuple[float, float, float] = (2.0, 2.0, 2.5),
    ) -> None:
        if watchdog_seconds <= 0 or action_limit <= 0:
            raise ValueError("watchdog_seconds and action_limit must be positive")
        self.watchdog_seconds = watchdog_seconds
        self.action_limit = action_limit
        self.workspace_low = np.asarray(workspace_low, dtype=np.float64)
        self.workspace_high = np.asarray(workspace_high, dtype=np.float64)
        self._safety_epoch = 0

    def authorize(
        self,
        observation_store: LiberoObservationStore,
        context: ContextEnvelope,
    ) -> SafetyDecision:
        epoch_id, observation, updated_at = observation_store.read()
        if time.monotonic() - updated_at > self.watchdog_seconds:
            return SafetyDecision(DispatchStatus.WATCHDOG_EXPIRED, reason="observation watchdog expired")
        if context.epoch_id != epoch_id:
            return SafetyDecision(DispatchStatus.SAFETY_VETO, reason="stale physical epoch")
        try:
            eef = np.asarray(observation["robot0_eef_pos"], dtype=np.float64)
            state_parts = [
                eef,
                np.asarray(observation["robot0_eef_quat"], dtype=np.float64),
                np.asarray(observation["robot0_gripper_qpos"], dtype=np.float64),
            ]
        except (KeyError, TypeError, ValueError):
            return SafetyDecision(DispatchStatus.SAFETY_VETO, reason="missing robot state")
        if eef.shape != (3,) or any(not np.isfinite(part).all() for part in state_parts):
            return SafetyDecision(DispatchStatus.SAFETY_VETO, reason="non-finite robot state")
        if np.any(eef < self.workspace_low) or np.any(eef > self.workspace_high):
            return SafetyDecision(DispatchStatus.SAFETY_VETO, reason="EEF outside workspace")
        self._safety_epoch += 1
        return SafetyDecision(DispatchStatus.ENQUEUED, self._safety_epoch)

    def validate_action(self, action: np.ndarray) -> bool:
        return (
            action.shape == (7,)
            and np.isfinite(action).all()
            and bool(np.all(np.abs(action) <= self.action_limit))
        )


@dataclass(frozen=True)
class AttemptResult:
    attempt_id: str
    context: ContextEnvelope
    action: GroundAction
    prompt: str
    prompt_history: tuple[str, ...]
    pre_epoch: int
    post_epoch: int | None
    executor_status: ExecutorStatus
    stopped: bool
    stop_evidence: str
    actions: tuple[np.ndarray, ...]
    frames: tuple[np.ndarray, ...]
    inference_requests: int
    detector_calls: int
    unused_actions_flushed: int
    post_snapshot: FactSnapshot | None
    post_snapshot_error: str | None = None
    reason: str = ""


@dataclass
class _QueuedAttempt:
    action: GroundAction
    context: ContextEnvelope
    pre_epoch: int


class Pi05MacroExecutor:
    """Turn one grounded occurrence into a bounded synchronous π0.5 macro attempt."""

    def __init__(
        self,
        *,
        env: Any,
        client: Any,
        image_tools: Any,
        observation_store: LiberoObservationStore,
        grounder: LiberoOracleGrounder,
        prompt_renderer: SubtaskPromptRenderer,
        safety_supervisor: SimulatorSafetySupervisor,
        replan_steps: int,
        max_action_steps: int,
        settling_steps: int,
        stop_on_effects: bool = True,
        max_total_action_steps: int | None = None,
    ) -> None:
        if replan_steps <= 0 or max_action_steps <= 0 or settling_steps < 0:
            raise ValueError("invalid executor bounds")
        self.env = env
        self.client = client
        self.image_tools = image_tools
        self.store = observation_store
        self.grounder = grounder
        self.prompt_renderer = prompt_renderer
        self.safety = safety_supervisor
        self.replan_steps = replan_steps
        self.max_action_steps = max_action_steps
        self.settling_steps = settling_steps
        self.stop_on_effects = bool(stop_on_effects)
        self.max_total_action_steps = (
            max_action_steps if max_total_action_steps is None else max_total_action_steps
        )
        if self.max_total_action_steps <= 0:
            raise ValueError("max_total_action_steps must be positive")
        self.total_action_steps = 0
        self._lock = threading.RLock()
        self._next_attempt = 0
        self._active: str | None = None
        self._queued: dict[str, _QueuedAttempt] = {}
        self._completed: dict[str, ExecutorOutcome] = {}
        self.results: list[AttemptResult] = []
        self._last_gripper_command = -1.0
        self._halt_ack_possible = True

    def consume_permit_and_enqueue(
        self,
        action: GroundAction,
        context: ContextEnvelope,
        snapshot: FactSnapshot,
    ) -> DispatchStart:
        with self._lock:
            if self._active is not None:
                return DispatchStart(DispatchStatus.EXECUTOR_REJECTED_NOT_ENQUEUED, context=context)
            if snapshot.epoch_id != context.epoch_id or snapshot.epoch_id != self.store.epoch_id:
                return DispatchStart(DispatchStatus.SAFETY_VETO, context=context)
            if self.total_action_steps >= self.max_total_action_steps:
                return DispatchStart(DispatchStatus.ACTION_BUDGET_EXHAUSTED, context=context)
            decision = self.safety.authorize(self.store, context)
            if decision.status is not DispatchStatus.ENQUEUED or decision.safety_epoch is None:
                return DispatchStart(decision.status, context=context)
            attempt_id = f"sim-attempt-{self._next_attempt:06d}"
            self._next_attempt += 1
            attempt_context = replace(
                context,
                attempt_id=attempt_id,
                safety_epoch=decision.safety_epoch,
            )
            self._queued[attempt_id] = _QueuedAttempt(action, attempt_context, snapshot.epoch_id)
            self._active = attempt_id
            return DispatchStart(
                DispatchStatus.ENQUEUED,
                attempt_id=attempt_id,
                safety_epoch=decision.safety_epoch,
                context=attempt_context,
            )

    def _outcome(
        self,
        queued: _QueuedAttempt,
        attempt_id: str,
        status: ExecutorStatus,
        settled_epoch: int | None,
        reason: str,
    ) -> ExecutorOutcome:
        outcome = ExecutorOutcome(
            status=status,
            attempt_id=attempt_id,
            stopped=status in {ExecutorStatus.SUCCEEDED, ExecutorStatus.EXECUTOR_FAILED},
            stop_ack_attempt_id=(
                attempt_id if status in {ExecutorStatus.SUCCEEDED, ExecutorStatus.EXECUTOR_FAILED} else None
            ),
            settled_epoch=settled_epoch,
            reason=reason,
            context=queued.context,
        )
        self._completed[attempt_id] = outcome
        self._queued.pop(attempt_id, None)
        self._active = None
        return outcome

    def await_outcome(self, start: DispatchStart) -> ExecutorOutcome:
        if start.attempt_id is None:
            raise ValueError("cannot await a non-enqueued dispatch")
        with self._lock:
            if start.attempt_id in self._completed:
                return self._completed[start.attempt_id]
            queued = self._queued.get(start.attempt_id)
            if queued is None or start.context != queued.context:
                return ExecutorOutcome(
                    ExecutorStatus.OUTCOME_UNKNOWN,
                    start.attempt_id,
                    stopped=False,
                    reason="attempt/context mismatch",
                    context=start.context,
                )

        action_queue: deque[np.ndarray] = deque()
        actions: list[np.ndarray] = []
        frames: list[np.ndarray] = []
        inference_requests = 0
        detector_start = self.grounder.detector_calls
        phase_flushed = 0
        phase = "acquire"
        prompt = self.prompt_renderer.render_phase(queued.action, phase)
        prompt_history = [prompt]
        place_schemas = {
            "place-on",
            "place-in",
            "place-relative",
            "place-held-on",
            "place-held-in",
            "place-held-relative",
            "put-down",
        }
        status = ExecutorStatus.SUCCEEDED
        reason = "macro bound reached"

        try:
            for _ in range(self.max_action_steps):
                if self.total_action_steps >= self.max_total_action_steps:
                    reason = "episode-global low-level action budget reached"
                    break
                _, observation, _ = self.store.read()
                element, main_image = prepare_observation(observation, prompt, self.image_tools)
                frames.append(main_image)
                if not action_queue:
                    response = self.client.infer(element)
                    inference_requests += 1
                    chunk = np.asarray(response["actions"], dtype=np.float64)
                    if (
                        chunk.ndim != 2
                        or chunk.shape[0] < self.replan_steps
                        or chunk.shape[1] != 7
                    ):
                        raise EpisodeInvalid(
                            f"action chunk shape {chunk.shape} cannot supply {self.replan_steps}x7"
                        )
                    selected = chunk[: self.replan_steps]
                    if any(not self.safety.validate_action(row) for row in selected):
                        finite = bool(np.isfinite(selected).all())
                        finite_values = selected[np.isfinite(selected)]
                        minimum = (
                            float(finite_values.min()) if finite_values.size else float("nan")
                        )
                        maximum = (
                            float(finite_values.max()) if finite_values.size else float("nan")
                        )
                        raise EpisodeInvalid(
                            "policy action violates finite/shape/limit checks: "
                            f"shape={selected.shape}, finite={finite}, "
                            f"min={minimum:.6g}, max={maximum:.6g}, "
                            f"limit={self.safety.action_limit:.6g}"
                        )
                    action_queue.extend(np.asarray(row, dtype=np.float64) for row in selected)

                low_level_action = action_queue.popleft()
                if not self.safety.validate_action(low_level_action):
                    raise EpisodeInvalid("queued policy action failed safety recheck")
                self._last_gripper_command = float(low_level_action[-1])
                post_observation, _, done, _ = self.env.step(low_level_action.tolist())
                actions.append(low_level_action.copy())
                self.total_action_steps += 1
                self.store.update(post_observation)
                if self.stop_on_effects:
                    effects_satisfied, target_diverged = (
                        self.grounder.observe_action_progress(queued.action)
                    )
                    if effects_satisfied:
                        reason = "observed declared effects"
                        break
                    if target_diverged:
                        reason = "observed target-location divergence"
                        break
                if self.prompt_renderer.has_phase(queued.action, "finish"):
                    try:
                        phase_snapshot = self.grounder.peek_snapshot()
                    except GroundingError:
                        phase_snapshot = None
                    if phase_snapshot is not None:
                        object_name = queued.action.arguments[0]
                        intended_holding = Fact("holding", (object_name,))
                        held_objects = {
                            fact.arguments[0]
                            for fact in phase_snapshot.true_facts
                            if fact.predicate == "holding"
                        }
                        if intended_holding in phase_snapshot.true_facts and phase != "finish":
                            phase_flushed += len(action_queue)
                            action_queue.clear()
                            phase = "finish"
                            prompt = self.prompt_renderer.render_phase(queued.action, phase)
                            prompt_history.append(prompt)
                        elif held_objects and object_name not in held_objects:
                            reason = "observed unexpected-object holding divergence"
                            break
                # LIBERO overloads ``done`` with the current BDDL task-success
                # predicate; its state remains step-able.  An effect-gated
                # LOGIV occurrence therefore cannot use it as STOPPED evidence.
                if bool(done) and not self.stop_on_effects:
                    reason = "simulator task predicate became true"
                    break
        except (EpisodeInvalid, KeyError, TypeError, ValueError) as error:
            # A malformed policy response is outside the action contract.  It is
            # not a task-level effect failure and must never enter automatic retry.
            status = ExecutorStatus.OUTCOME_UNKNOWN
            reason = str(error)
        except Exception as error:  # simulator or transport state is no longer trustworthy
            status = ExecutorStatus.OUTCOME_UNKNOWN
            reason = f"untrusted executor exception: {error}"
            self._halt_ack_possible = False

        unused_actions_flushed = phase_flushed + len(action_queue)
        action_queue.clear()
        if status in {ExecutorStatus.SUCCEEDED, ExecutorStatus.EXECUTOR_FAILED}:
            try:
                hold = np.zeros(7, dtype=np.float64)
                release_completion = (
                    self.stop_on_effects
                    and queued.action.schema in place_schemas
                    and reason == "observed declared effects"
                )
                hold[-1] = -1.0 if release_completion else self._last_gripper_command
                for _ in range(self.settling_steps):
                    observation, _, _, _ = self.env.step(hold.tolist())
                    self.store.update(observation)
            except Exception as error:
                status = ExecutorStatus.SETTLING_TIMEOUT
                reason = f"settling failed: {error}"

        settled_epoch = (
            self.store.epoch_id
            if status in {ExecutorStatus.SUCCEEDED, ExecutorStatus.EXECUTOR_FAILED}
            else None
        )
        post_snapshot = None
        post_snapshot_error = None
        if settled_epoch is not None:
            try:
                post_snapshot = self.grounder.peek_snapshot()
            except GroundingError as error:
                post_snapshot_error = str(error)
        self.results.append(
            AttemptResult(
                attempt_id=start.attempt_id,
                context=queued.context,
                action=queued.action,
                prompt=prompt_history[0],
                prompt_history=tuple(prompt_history),
                pre_epoch=queued.pre_epoch,
                post_epoch=settled_epoch,
                executor_status=status,
                stopped=status in {ExecutorStatus.SUCCEEDED, ExecutorStatus.EXECUTOR_FAILED},
                stop_evidence=(
                    "synchronous env.step returned and local action deque was flushed"
                    if status in {ExecutorStatus.SUCCEEDED, ExecutorStatus.EXECUTOR_FAILED}
                    else "no STOPPED acknowledgement"
                ),
                actions=tuple(actions),
                frames=tuple(frames),
                inference_requests=inference_requests,
                detector_calls=self.grounder.detector_calls - detector_start,
                unused_actions_flushed=unused_actions_flushed,
                post_snapshot=post_snapshot,
                post_snapshot_error=post_snapshot_error,
                reason=reason,
            )
        )
        with self._lock:
            return self._outcome(
                queued,
                start.attempt_id,
                status,
                settled_epoch,
                reason,
            )

    def request_emergency_halt(self, attempt_id: str | None) -> bool:
        with self._lock:
            if attempt_id is not None and self._active not in {None, attempt_id}:
                return False
            self._queued.clear()
            self._active = None
            return self._halt_ack_possible
