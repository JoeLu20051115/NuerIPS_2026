from __future__ import annotations

from collections import deque
from contextlib import contextmanager
import copy
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import threading
import time
from typing import Any, Callable, FrozenSet, Iterable, Iterator, Mapping

import numpy as np

from pi05_libero_repro.logiv.configuration import load_extended_json
from pi05_libero_repro.logiv.controller import (
    DispatchStart,
    DispatchStatus,
    ExecutionCompletionHint,
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
    fact_pddl_sort_key,
    fact_universe_sha256,
)
from pi05_libero_repro.logiv.prompts import SubtaskPromptRenderer
from pi05_libero_repro.protocol import EpisodeInvalid, prepare_observation
from pi05_libero_repro.logiv.recovery_records import observation_sha256
from pi05_libero_repro.logiv.shadow_monitor import (
    ActionTransitionFeatures,
    MonitorEvidenceContract,
)


def _inner_env(env: Any) -> Any:
    current = env
    seen: set[int] = set()
    while not hasattr(current, "_eval_predicate") and hasattr(current, "env"):
        if id(current) in seen:
            break
        seen.add(id(current))
        current = current.env
    return current


def _observation_hash(observation: Mapping[str, Any]) -> str:
    return observation_sha256(observation)


@dataclass(frozen=True)
class TaskBinding:
    task_id: int
    frozen: bool
    registered_objects: tuple[str, ...]
    symbol_bindings: Mapping[str, Mapping[str, str]]
    supported_action_schemas: FrozenSet[str]
    recovery_schemas: FrozenSet[str]
    decompose_macro_sources: FrozenSet[str]

    @classmethod
    def from_manifest(cls, path: Path | str, task_id: int) -> "TaskBinding":
        payload = load_extended_json(path)
        tasks = payload.get("tasks", [])
        task_ids = [int(item["task_id"]) for item in tasks]
        if payload.get("frozen") is not True or task_ids != list(range(10)):
            raise ValueError("coverage manifest must be frozen with exact task IDs 0..9")
        try:
            item = next(entry for entry in tasks if int(entry["task_id"]) == task_id)
        except StopIteration as error:
            raise ValueError(f"task {task_id} is outside the frozen manifest") from error
        registered_objects = tuple(item["registered_objects"])
        decompose_macro_sources = frozenset(item.get("decompose_macro_sources", ()))
        if not decompose_macro_sources <= frozenset(registered_objects):
            raise ValueError("decompose_macro_sources must be registered objects")
        return cls(
            task_id=task_id,
            frozen=True,
            registered_objects=registered_objects,
            symbol_bindings={
                str(key): {str(k): str(v) for k, v in value.items()}
                for key, value in item.get("symbol_bindings", {}).items()
            },
            supported_action_schemas=frozenset(item["supported_action_schemas"]),
            recovery_schemas=frozenset(item["recovery_schemas"]),
            decompose_macro_sources=decompose_macro_sources,
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


@dataclass(frozen=True)
class ReadOnlyLiberoStateView:
    """Narrow, read-only simulator access frozen for transition monitoring."""

    holding_reader: Callable[[str], TruthValue]
    contact_count_reader: Callable[[str], int | None]
    region_truth_reader: Callable[[str, str], TruthValue]
    region_distance_reader: Callable[[str, str], float | None]
    object_position_reader: Callable[[str], np.ndarray | None]


def _truth_from_optional_bool(value: Any) -> TruthValue:
    if isinstance(value, TruthValue):
        return value
    if isinstance(value, (bool, np.bool_)):
        return TruthValue.TRUE if bool(value) else TruthValue.FALSE
    return TruthValue.UNKNOWN


def _finite_scalar(value: Any, *, nonnegative: bool = False) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not np.isfinite(result) or (nonnegative and result < 0):
        return None
    return result


def _finite_position(value: Any) -> np.ndarray | None:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if result.shape != (3,) or not np.isfinite(result).all():
        return None
    return np.array(result, copy=True)


def _build_read_only_transition_view(
    env: Any,
    binding: TaskBinding,
) -> ReadOnlyLiberoStateView:
    inner = _inner_env(env)
    custom_holding = getattr(inner, "read_logiv_holding", None)
    custom_contacts = getattr(inner, "read_logiv_contact_count", None)
    custom_region_truth = getattr(inner, "read_logiv_region_truth", None)
    custom_region_distance = getattr(inner, "read_logiv_region_distance", None)
    custom_object_position = getattr(inner, "read_logiv_object_position", None)

    def holding(object_id: str) -> TruthValue:
        try:
            if callable(custom_holding):
                return _truth_from_optional_bool(custom_holding(object_id))
            object_model = inner.objects_dict[object_id]
            robot = inner.robots[0]
            return _truth_from_optional_bool(
                inner._check_grasp(robot.gripper, object_model.contact_geoms)
            )
        except (KeyError, IndexError, TypeError, ValueError, AttributeError):
            return TruthValue.UNKNOWN

    def contact_count(object_id: str) -> int | None:
        try:
            if callable(custom_contacts):
                value = custom_contacts(object_id)
                if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                    return None
                return int(value) if int(value) >= 0 else None
            value = holding(object_id)
            return 1 if value is TruthValue.TRUE else 0 if value is TruthValue.FALSE else None
        except (KeyError, IndexError, TypeError, ValueError, AttributeError):
            return None

    def support_distance(object_id: str, support_name: str) -> float | None:
        try:
            workspace = _finite_position(inner.workspace_offset)
            size_value = getattr(inner, f"{support_name}_full_size", None)
            if size_value is None:
                size_value = inner.table_full_size
            full_size = np.asarray(size_value, dtype=np.float64).reshape(-1)
            position = _finite_position(
                inner.sim.data.body_xpos[inner.obj_body_id[object_id]]
            )
            bottom_offset = _finite_position(
                inner.get_object(object_id).bottom_offset
            )
        except (KeyError, IndexError, TypeError, ValueError, AttributeError):
            return None
        if (
            workspace is None
            or position is None
            or bottom_offset is None
            or full_size.size < 2
            or not np.isfinite(full_size).all()
            or np.any(full_size[:2] <= 0)
        ):
            return None
        bottom = position + bottom_offset
        horizontal = np.maximum(
            np.abs(bottom[:2] - workspace[:2]) - full_size[:2] / 2.0,
            0.0,
        )
        vertical = abs(float(bottom[2] - workspace[2]))
        return float(np.linalg.norm(np.r_[horizontal, vertical]))

    def region_truth(object_id: str, region: str) -> TruthValue:
        try:
            if callable(custom_region_truth):
                return _truth_from_optional_bool(custom_region_truth(object_id, region))
            resolved, kind = binding.resolve(region)
            if kind == "support_surface_alias":
                distance = support_distance(object_id, resolved)
                if distance is None:
                    return TruthValue.UNKNOWN
                return (
                    TruthValue.TRUE
                    if distance <= 0.04
                    else TruthValue.FALSE
                )
            predicate = (
                "in"
                if any(
                    token in region
                    for token in ("contain_region", "heating_region", "bottom_region")
                )
                else "on"
            )
            return _truth_from_optional_bool(
                inner._eval_predicate([predicate, object_id, resolved])
            )
        except (KeyError, IndexError, TypeError, ValueError, AttributeError):
            return TruthValue.UNKNOWN

    def object_position(object_id: str) -> np.ndarray | None:
        try:
            if callable(custom_object_position):
                return _finite_position(custom_object_position(object_id))
            return _finite_position(
                inner.sim.data.body_xpos[inner.obj_body_id[object_id]]
            )
        except (KeyError, IndexError, TypeError, ValueError, AttributeError):
            return None

    def region_distance(object_id: str, region: str) -> float | None:
        try:
            if callable(custom_region_distance):
                return _finite_scalar(
                    custom_region_distance(object_id, region), nonnegative=True
                )
            position = object_position(object_id)
            resolved, kind = binding.resolve(region)
            if kind == "support_surface_alias":
                return support_distance(object_id, resolved)
            target = _finite_position(inner.sim.data.get_site_xpos(resolved))
            if position is None or target is None:
                return None
            return float(np.linalg.norm(position - target))
        except (KeyError, IndexError, TypeError, ValueError, AttributeError):
            return None

    return ReadOnlyLiberoStateView(
        holding_reader=holding,
        contact_count_reader=contact_count,
        region_truth_reader=region_truth,
        region_distance_reader=region_distance,
        object_position_reader=object_position,
    )


class _LiberoTransitionFeatureReader:
    def __init__(
        self,
        view: ReadOnlyLiberoStateView,
        monitor_contract: MonitorEvidenceContract,
    ) -> None:
        self._view = view
        self._rules = monitor_contract.action_event_rules
        self._abnormal_surfaces = monitor_contract.abnormal_support_surfaces
        self.monitor_contract_sha256 = monitor_contract.contract_sha256
        self.tracker_version = monitor_contract.tracker_version
        self.rule_ids = tuple(rule.rule_id for rule in self._rules)
        self._previous_object_positions: dict[str, np.ndarray] = {}
        self._previous_eef_position: np.ndarray | None = None
        self._previous_policy_step: int | None = None

    @staticmethod
    def _observation_scalar(
        observation: Mapping[str, Any], key: str
    ) -> float | None:
        try:
            value = np.asarray(observation[key], dtype=np.float64).reshape(-1)
        except (KeyError, TypeError, ValueError):
            return None
        if not value.size or not np.isfinite(value).all():
            return None
        return float(value[0])

    @staticmethod
    def _motion_correlation(
        previous_object: np.ndarray | None,
        current_object: np.ndarray | None,
        previous_eef: np.ndarray | None,
        current_eef: np.ndarray | None,
    ) -> float | None:
        if any(
            value is None
            for value in (previous_object, current_object, previous_eef, current_eef)
        ):
            return None
        assert previous_object is not None
        assert current_object is not None
        assert previous_eef is not None
        assert current_eef is not None
        object_delta = current_object - previous_object
        eef_delta = current_eef - previous_eef
        denominator = float(np.linalg.norm(object_delta) * np.linalg.norm(eef_delta))
        if denominator <= 0 or not np.isfinite(denominator):
            return None
        value = float(np.dot(object_delta, eef_delta) / denominator)
        return float(np.clip(value, -1.0, 1.0)) if np.isfinite(value) else None

    def __call__(
        self, context: ShadowStepContext
    ) -> tuple[ActionTransitionFeatures, ...]:
        observation = context.observation
        gripper = self._observation_scalar(observation, "robot0_gripper_qpos")
        current_eef = _finite_position(observation.get("robot0_eef_pos"))
        consecutive = (
            self._previous_policy_step is not None
            and context.policy_step == self._previous_policy_step + 1
        )
        object_positions = {
            object_id: self._view.object_position_reader(object_id)
            for object_id in {rule.object_id for rule in self._rules}
        }
        correlations = {
            object_id: self._motion_correlation(
                self._previous_object_positions.get(object_id) if consecutive else None,
                position,
                self._previous_eef_position if consecutive else None,
                current_eef,
            )
            for object_id, position in object_positions.items()
        }
        rows = []
        for rule in self._rules:
            abnormal_values = [
                (
                    surface,
                    self._view.region_truth_reader(rule.object_id, surface),
                    self._view.region_distance_reader(rule.object_id, surface),
                )
                for surface in self._abnormal_surfaces
            ]
            confirmed = [item for item in abnormal_values if item[1] is TruthValue.TRUE]
            if confirmed:
                surface, abnormal_truth, abnormal_distance = min(
                    confirmed,
                    key=lambda item: (
                        float("inf") if item[2] is None else item[2], item[0]
                    ),
                )
                abnormal_region_id = surface
            else:
                abnormal_truth = (
                    TruthValue.FALSE
                    if abnormal_values
                    and all(item[1] is TruthValue.FALSE for item in abnormal_values)
                    else TruthValue.UNKNOWN
                )
                abnormal_region_id = None
                abnormal_distance = None
            rows.append(
                ActionTransitionFeatures.create(
                    policy_step=context.policy_step,
                    monitor_contract_sha256=self.monitor_contract_sha256,
                    tracker_version=self.tracker_version,
                    rule_id=rule.rule_id,
                    object_id=rule.object_id,
                    source_region=rule.source_region,
                    destination_region=rule.destination_region,
                    gripper_qpos=gripper,
                    contact_count=self._view.contact_count_reader(rule.object_id),
                    holding=self._view.holding_reader(rule.object_id),
                    source_region_truth=(
                        self._view.region_truth_reader(rule.object_id, rule.source_region)
                        if rule.source_region is not None
                        else TruthValue.UNKNOWN
                    ),
                    destination_region_truth=(
                        self._view.region_truth_reader(
                            rule.object_id, rule.destination_region
                        )
                        if rule.destination_region is not None
                        else TruthValue.UNKNOWN
                    ),
                    abnormal_region_truth=abnormal_truth,
                    source_region_distance=(
                        self._view.region_distance_reader(
                            rule.object_id, rule.source_region
                        )
                        if rule.source_region is not None
                        else None
                    ),
                    destination_region_distance=(
                        self._view.region_distance_reader(
                            rule.object_id, rule.destination_region
                        )
                        if rule.destination_region is not None
                        else None
                    ),
                    abnormal_region_id=abnormal_region_id,
                    abnormal_region_distance=abnormal_distance,
                    object_eef_motion_correlation=correlations[rule.object_id],
                )
            )
        self._previous_object_positions = {
            object_id: position
            for object_id, position in object_positions.items()
            if position is not None
        }
        self._previous_eef_position = current_eef
        self._previous_policy_step = context.policy_step
        return tuple(rows)


def build_libero_transition_feature_reader(
    env: Any,
    binding: TaskBinding,
    monitor_contract: MonitorEvidenceContract,
) -> _LiberoTransitionFeatureReader:
    if binding.task_id != monitor_contract.task_id:
        raise ValueError("transition reader binding/monitor task mismatch")
    registered = frozenset(binding.registered_objects)
    required = set(monitor_contract.object_ids) | set(
        monitor_contract.abnormal_support_surfaces
    )
    for rule in monitor_contract.action_event_rules:
        required.update(
            value
            for value in (rule.source_region, rule.destination_region)
            if value is not None
        )
    missing = required - registered
    if missing:
        raise ValueError(
            "transition reader contract uses unregistered IDs: "
            + ", ".join(sorted(missing))
        )
    return _LiberoTransitionFeatureReader(
        _build_read_only_transition_view(env, binding), monitor_contract
    )


class LiberoObservationStore:
    """Thread-safe latest synchronous simulator observation and physical epoch."""

    def __init__(self, observation: Mapping[str, Any], *, epoch_id: int = 0) -> None:
        if epoch_id < 0:
            raise ValueError("epoch_id must be nonnegative")
        self._lock = threading.RLock()
        self._observation = self._freeze(observation)
        self._epoch_id = epoch_id
        self._updated_at = time.monotonic()

    @staticmethod
    def _freeze(observation: Mapping[str, Any]) -> dict[str, Any]:
        frozen: dict[str, Any] = {}
        for key, value in observation.items():
            if isinstance(value, np.ndarray):
                array = np.array(value, copy=True)
                array.setflags(write=False)
                frozen[key] = array
            else:
                frozen[key] = copy.deepcopy(value)
        return frozen

    def _read_locked(self) -> tuple[int, dict[str, Any], float]:
        return self._epoch_id, self._freeze(self._observation), self._updated_at

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
            return self._read_locked()

    @contextmanager
    def capture(self) -> Iterator[tuple[int, dict[str, Any], float]]:
        with self._lock:
            yield self._read_locked()

    def update(self, observation: Mapping[str, Any]) -> int:
        with self._lock:
            self._observation = self._freeze(observation)
            self._epoch_id += 1
            self._updated_at = time.monotonic()
            return self._epoch_id

    def advance(self, step: Callable[[], tuple[Any, ...]]) -> tuple[Any, ...]:
        """Run one synchronous simulator step and publish its observation atomically."""

        with self._lock:
            result = step()
            if not isinstance(result, tuple) or not result:
                raise ValueError("synchronous simulator step returned no observation")
            self._observation = self._freeze(result[0])
            self._epoch_id += 1
            self._updated_at = time.monotonic()
            return result


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

    def _settling_tolerant_region_truth(
        self, object_name: str, location: str
    ) -> TruthValue:
        """Confirm a nominal init site with 5 mm of simulator-settling tolerance."""

        try:
            parsed = self.inner.parsed_problem
            initial_state = parsed["initial_state"]
            support_name = str(parsed["regions"][location]["target"])
            site = self.inner.object_sites_dict[location]
            site_position = np.asarray(
                self.inner.sim.data.get_site_xpos(location), dtype=np.float64
            )
            site_matrix = np.asarray(
                self.inner.sim.data.get_site_xmat(location), dtype=np.float64
            )
            size = np.asarray(site.size, dtype=np.float64)
            object_position = np.asarray(
                self.inner.sim.data.body_xpos[self.inner.obj_body_id[object_name]],
                dtype=np.float64,
            )
        except (KeyError, IndexError, TypeError, ValueError, AttributeError):
            return TruthValue.UNKNOWN
        nominal = any(
            len(state) == 3
            and str(state[0]).lower() == "on"
            and str(state[1]) == object_name
            and str(state[2]) == location
            for state in initial_state
        )
        if not nominal or not support_name.endswith("table"):
            return TruthValue.UNKNOWN
        if (
            site_position.shape != (3,)
            or site_matrix.shape != (3, 3)
            or size.shape != (3,)
            or object_position.shape != (3,)
            or not all(
                np.isfinite(value).all()
                for value in (site_position, site_matrix, size, object_position)
            )
        ):
            return TruthValue.UNKNOWN
        held = self._holding(object_name)
        if held is TruthValue.TRUE:
            return TruthValue.FALSE
        if held is TruthValue.UNKNOWN:
            return TruthValue.UNKNOWN
        tolerance = 0.005
        total_size = np.abs(site_matrix @ size)
        delta = site_matrix @ (object_position - site_position)
        inside_xy = bool(
            np.all(np.abs(delta[:2]) <= total_size[:2] + tolerance)
        )
        above_site = bool(
            total_size[2] - 0.005 - tolerance
            < delta[2]
            < total_size[2] + 0.10 + tolerance
        )
        return TruthValue.TRUE if inside_xy and above_site else TruthValue.FALSE

    def _direct_location_truth(
        self, object_name: str, location: str
    ) -> TruthValue:
        resolved, kind = self.binding.resolve(location)
        direct = self._predicate(
            [self._location_predicate(location), object_name, resolved]
        )
        if direct is TruthValue.TRUE or kind == "support_surface_alias":
            return direct
        tolerant = self._settling_tolerant_region_truth(object_name, location)
        return TruthValue.TRUE if tolerant is TruthValue.TRUE else direct

    def _workspace_support_truth(
        self,
        object_name: str,
        support_name: str,
        other_locations: Iterable[str],
    ) -> TruthValue:
        other_values = [
            self._direct_location_truth(object_name, other)
            for other in other_locations
        ]
        if any(value is TruthValue.TRUE for value in other_values):
            return TruthValue.FALSE
        broad = self._predicate(["on", object_name, support_name])
        contact = self._support_contact(object_name, support_name)
        geometry = self._support_geometry(object_name, support_name)
        held = self._holding(object_name)
        supported = (
            TruthValue.TRUE
            if TruthValue.TRUE in {broad, contact, geometry}
            else TruthValue.FALSE
            if all(value is TruthValue.FALSE for value in (broad, contact, geometry))
            else TruthValue.UNKNOWN
        )
        if supported is TruthValue.TRUE and held is TruthValue.FALSE and all(
            value is TruthValue.FALSE for value in other_values
        ):
            return TruthValue.TRUE
        if supported is TruthValue.FALSE or held is TruthValue.TRUE:
            return TruthValue.FALSE
        return TruthValue.UNKNOWN

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
            other_locations = sorted(
                candidate.arguments[1]
                for candidate in self.monitored_facts
                if candidate.predicate == "at"
                and candidate.arguments[0] == object_name
                and candidate.arguments[1] != location
            )
            if kind == "support_surface_alias":
                return self._workspace_support_truth(
                    object_name, resolved, other_locations
                )
            return self._direct_location_truth(object_name, location)
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

    def _snapshot(
        self,
        required: FrozenSet[Fact],
        *,
        expected_epoch: int | None = None,
        advisory_partial: bool = False,
    ) -> FactSnapshot:
        unregistered = required - self.monitored_facts
        if unregistered:
            raise GroundingError(
                "required facts are outside the registered fact universe: "
                + ", ".join(fact.pddl() for fact in sorted(unregistered))
            )
        with self.store.capture() as (epoch_id, observation, _):
            if expected_epoch is not None and epoch_id != expected_epoch:
                raise GroundingError("context/epoch mismatch")
            snapshot = self._snapshot_at(
                required, epoch_id, observation, advisory_partial=advisory_partial
            )
            if self.store.epoch_id != epoch_id:
                raise GroundingError("fact snapshot epoch changed during capture")
            return snapshot

    def _snapshot_at(
        self,
        required: FrozenSet[Fact],
        epoch_id: int,
        observation: Mapping[str, Any],
        *,
        advisory_partial: bool = False,
    ) -> FactSnapshot:
        values: dict[Fact, TruthValue] = {}
        for fact in sorted(self.monitored_facts, key=fact_pddl_sort_key):
            if fact.predicate != "handempty":
                values[fact] = self._truth(fact)

        holding_facts = sorted(
            (fact for fact in self.monitored_facts if fact.predicate == "holding"),
            key=fact_pddl_sort_key,
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

        def exactly_one(candidates: set[Fact]) -> bool:
            confirmed = candidates & true_facts
            return len(confirmed) == 1 or (advisory_partial and not confirmed)

        movable = {fact.arguments[0] for fact in holding_facts}
        for object_name in movable:
            candidates = {
                fact
                for fact in values
                if (fact.predicate == "holding" and fact.arguments == (object_name,))
                or (fact.predicate == "at" and fact.arguments[0] == object_name)
            }
            confirmed = candidates & true_facts
            valid = len(confirmed) <= 1 if advisory_partial else len(confirmed) == 1
            if not valid:
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
                if not exactly_one(pair):
                    raise GroundingError(
                        f"exactly-one {predicate} violation for {name}"
                    )

        fact_universe = frozenset(values)
        fact_universe_version = (
            f"logiv-libero-grounding-v1/task-{self.binding.task_id}"
        )
        evidence_payload = {
            "epoch_id": epoch_id,
            "observation_hash": _observation_hash(observation),
            "values": [
                (fact.pddl(), values[fact].value)
                for fact in sorted(values, key=fact_pddl_sort_key)
            ],
            "dominance_overrides": sorted(dominance_overrides),
        }
        evidence_payload_json = json.dumps(
            evidence_payload, sort_keys=True, separators=(",", ":")
        )
        evidence_hash = hashlib.sha256(evidence_payload_json.encode("utf-8")).hexdigest()
        return FactSnapshot(
            epoch_id=epoch_id,
            true_facts=true_facts,
            false_facts=false_facts,
            evidence_hash=evidence_hash,
            fact_universe=fact_universe,
            fact_universe_version=fact_universe_version,
            fact_universe_sha256=fact_universe_sha256(
                fact_universe_version, fact_universe
            ),
            evidence_payload_json=evidence_payload_json,
        )

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
        if phase is not context.phase:
            return GroundingResponse(status=status, context=context, reason="context/epoch mismatch")
        try:
            snapshot = self._snapshot(
                required_facts, expected_epoch=context.epoch_id
            )
        except (GroundingError, TypeError, ValueError) as error:
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

    def peek_advisory_partial_snapshot(self) -> FactSnapshot:
        """Read a topology-only partial snapshot without authorizing control."""

        return self._snapshot(frozenset(), advisory_partial=True)

    def effects_satisfied(self, action: GroundAction) -> bool:
        return self.observe_action_progress(action)[0]

    def observe_action_progress(
        self,
        action: GroundAction,
        *,
        completion_positive: FrozenSet[Fact] | None = None,
        completion_negative: FrozenSet[Fact] | None = None,
    ) -> tuple[bool, bool]:
        completion, _, target_diverged = self.observe_action_progress_details(
            action,
            completion_positive=completion_positive,
            completion_negative=completion_negative,
        )
        return completion, target_diverged

    def observe_action_progress_details(
        self,
        action: GroundAction,
        *,
        completion_positive: FrozenSet[Fact] | None = None,
        completion_negative: FrozenSet[Fact] | None = None,
    ) -> tuple[bool, bool, bool]:
        """Return completion, primary-effect, and factual-divergence signals.

        The tuple contains completion-effects, primary-effects, and divergence.
        The third signal is intentionally limited to grounded place macros that
        carry an explicit source and target.  It does not infer a location from
        missing predicates: a different registered ``at`` fact must be TRUE and
        source, target, and holding must all be explicitly FALSE.
        """

        self.detector_calls += 1
        positive = action.add_effects if completion_positive is None else completion_positive
        negative = action.del_effects if completion_negative is None else completion_negative
        required = positive | negative
        sourced_place = action.schema in {"place-on", "place-in", "place-relative"}
        held_place = action.schema in {
            "place-held-on",
            "place-held-in",
            "place-held-relative",
            "put-down",
        }
        location_schema = sourced_place or held_place
        source = action.arguments[1] if sourced_place else None
        target = (
            action.arguments[2]
            if sourced_place
            else action.arguments[1]
            if held_place
            else None
        )
        if location_schema:
            object_name = action.arguments[0]
            assert target is not None
            location_evidence = {
                Fact("at", (object_name, target)),
                Fact("holding", (object_name,)),
            }
            if source is not None:
                location_evidence.add(Fact("at", (object_name, source)))
            required |= frozenset(location_evidence)
        try:
            snapshot = self._snapshot(required)
        except GroundingError:
            return False, False, False
        effects_satisfied = snapshot.satisfies(
            positive=positive,
            negative=negative,
        )
        primary_effects_satisfied = snapshot.satisfies(
            positive=action.add_effects,
            negative=action.del_effects,
        )
        if not location_schema:
            return effects_satisfied, primary_effects_satisfied, False
        assert target is not None
        required_false = {
            Fact("at", (object_name, target)),
            Fact("holding", (object_name,)),
        }
        if source is not None:
            required_false.add(Fact("at", (object_name, source)))
        nominal_locations = {target}
        if source is not None:
            nominal_locations.add(source)
        alternative_location = any(
            fact.predicate == "at"
            and fact.arguments[0] == object_name
            and fact.arguments[1] not in nominal_locations
            for fact in snapshot.true_facts
        )
        target_diverged = required_false <= snapshot.false_facts and alternative_location
        return effects_satisfied, primary_effects_satisfied, target_diverged


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
    completion_mode: str
    recovery_frontier: bool
    completion_occurrence_ids: tuple[str, ...]
    completion_actions: tuple[GroundAction, ...]
    completion_positive: FrozenSet[Fact]
    completion_negative: FrozenSet[Fact]
    primary_effect_first_step: int | None
    frontier_followup_limit: int | None
    frontier_completion_step: int | None
    frontier_completion_prompt: str | None
    frontier_fallback_step: int | None
    frontier_fallback_prompt: str | None
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
    post_stop_reobservation_steps: int
    post_snapshot: FactSnapshot | None
    post_snapshot_error: str | None = None
    reason: str = ""


@dataclass
class _QueuedAttempt:
    action: GroundAction
    context: ContextEnvelope
    pre_epoch: int
    policy_prompt: str
    recovery_frontier: bool
    completion_mode: str
    completion_occurrence_ids: tuple[str, ...]
    completion_actions: tuple[GroundAction, ...]
    completion_positive: FrozenSet[Fact]
    completion_negative: FrozenSet[Fact]


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
        post_stop_grounding_reobservation_steps: int = 0,
        effect_confirmation_steps: int = 1,
        place_effect_confirmation_steps: int | None = None,
        access_effect_stabilization_steps: int = 0,
        place_effect_stabilization_steps: int = 0,
        target_divergence_confirmation_steps: int | None = None,
        held_target_divergence_confirmation_steps: int = 0,
        frontier_followup_steps: int | None = None,
        frontier_completion_followup_steps: int | None = None,
        frontier_completion_recovery_only: bool = False,
        frontier_recovery_max_consumed_steps: int | None = None,
        frontier_fallback_after_steps: int | None = None,
        frontier_fallback_followup_steps: int | None = None,
        stop_on_effects: bool = True,
        max_total_action_steps: int | None = None,
    ) -> None:
        if (
            replan_steps <= 0
            or max_action_steps <= 0
            or settling_steps < 0
            or post_stop_grounding_reobservation_steps < 0
            or effect_confirmation_steps <= 0
            or (
                place_effect_confirmation_steps is not None
                and place_effect_confirmation_steps <= 0
            )
            or access_effect_stabilization_steps < 0
            or place_effect_stabilization_steps < 0
            or held_target_divergence_confirmation_steps < 0
            or (
                target_divergence_confirmation_steps is not None
                and target_divergence_confirmation_steps <= 0
            )
            or (frontier_followup_steps is not None and frontier_followup_steps <= 0)
            or (
                frontier_completion_followup_steps is not None
                and frontier_completion_followup_steps <= 0
            )
            or (
                frontier_recovery_max_consumed_steps is not None
                and frontier_recovery_max_consumed_steps < 0
            )
            or (
                frontier_fallback_after_steps is not None
                and frontier_fallback_after_steps <= 0
            )
            or (
                frontier_fallback_followup_steps is not None
                and frontier_fallback_followup_steps <= 0
            )
        ):
            raise ValueError("invalid executor bounds")
        self.env = env
        self.client = client
        self.image_tools = image_tools
        self.store = observation_store
        self._initial_graph_version: str | None = None
        self.grounder = grounder
        self.prompt_renderer = prompt_renderer
        self.safety = safety_supervisor
        self.replan_steps = replan_steps
        self.max_action_steps = max_action_steps
        self.settling_steps = settling_steps
        self.post_stop_grounding_reobservation_steps = (
            post_stop_grounding_reobservation_steps
        )
        self.effect_confirmation_steps = effect_confirmation_steps
        self.place_effect_confirmation_steps = place_effect_confirmation_steps
        self.access_effect_stabilization_steps = access_effect_stabilization_steps
        self.place_effect_stabilization_steps = place_effect_stabilization_steps
        self.target_divergence_confirmation_steps = (
            effect_confirmation_steps
            if target_divergence_confirmation_steps is None
            else target_divergence_confirmation_steps
        )
        self.held_target_divergence_confirmation_steps = (
            held_target_divergence_confirmation_steps
        )
        self.frontier_followup_steps = frontier_followup_steps
        self.frontier_completion_followup_steps = (
            frontier_completion_followup_steps
        )
        self.frontier_completion_recovery_only = bool(
            frontier_completion_recovery_only
        )
        self.frontier_recovery_max_consumed_steps = (
            frontier_recovery_max_consumed_steps
        )
        self.frontier_fallback_after_steps = frontier_fallback_after_steps
        self.frontier_fallback_followup_steps = frontier_fallback_followup_steps
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
        *,
        completion_hint: ExecutionCompletionHint | None = None,
    ) -> DispatchStart:
        with self._lock:
            if self._active is not None:
                return DispatchStart(DispatchStatus.EXECUTOR_REJECTED_NOT_ENQUEUED, context=context)
            if snapshot.epoch_id != context.epoch_id or snapshot.epoch_id != self.store.epoch_id:
                return DispatchStart(DispatchStatus.SAFETY_VETO, context=context)
            if self.total_action_steps >= self.max_total_action_steps:
                return DispatchStart(DispatchStatus.ACTION_BUDGET_EXHAUSTED, context=context)
            if completion_hint is not None and (
                completion_hint.actions[0] != action
                or completion_hint.occurrence_ids[0] != context.occurrence_id
            ):
                return DispatchStart(
                    DispatchStatus.EXECUTOR_REJECTED_NOT_ENQUEUED,
                    context=context,
                )
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
            completion_mode = "OCCURRENCE"
            recovery_frontier = (
                self._initial_graph_version is not None
                and context.graph_version != self._initial_graph_version
                and (
                    self.frontier_recovery_max_consumed_steps is None
                    or self.total_action_steps
                    <= self.frontier_recovery_max_consumed_steps
                )
            )
            policy_prompt = self.prompt_renderer.render_phase(action, "acquire")
            completion_occurrence_ids = (
                (context.occurrence_id,) if context.occurrence_id is not None else ()
            )
            completion_actions = (action,)
            if completion_hint is not None:
                render_frontier = (
                    self.prompt_renderer.render_recovery_frontier
                    if recovery_frontier
                    else self.prompt_renderer.render_frontier
                )
                frontier_prompts = tuple(
                    render_frontier(item) for item in completion_hint.actions
                )
                if (
                    len(completion_hint.actions) >= 2
                    and len(set(frontier_prompts)) == 1
                ):
                    completion_mode = "DAG_FRONTIER"
                    policy_prompt = frontier_prompts[0]
                    completion_occurrence_ids = completion_hint.occurrence_ids
                    completion_actions = completion_hint.actions
            completion_positive = frozenset().union(
                *(item.add_effects for item in completion_actions)
            )
            completion_negative = frozenset().union(
                *(item.del_effects for item in completion_actions)
            )
            if self._initial_graph_version is None:
                self._initial_graph_version = context.graph_version
            self._queued[attempt_id] = _QueuedAttempt(
                action=action,
                context=attempt_context,
                pre_epoch=snapshot.epoch_id,
                policy_prompt=policy_prompt,
                recovery_frontier=recovery_frontier,
                completion_mode=completion_mode,
                completion_occurrence_ids=completion_occurrence_ids,
                completion_actions=completion_actions,
                completion_positive=completion_positive,
                completion_negative=completion_negative,
            )
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
            stopped=status
            in {
                ExecutorStatus.SUCCEEDED,
                ExecutorStatus.EPISODE_SUCCESS,
                ExecutorStatus.EXECUTOR_FAILED,
            },
            stop_ack_attempt_id=(
                attempt_id
                if status
                in {
                    ExecutorStatus.SUCCEEDED,
                    ExecutorStatus.EPISODE_SUCCESS,
                    ExecutorStatus.EXECUTOR_FAILED,
                }
                else None
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
        confirmed_effect_steps = 0
        stabilized_effect_steps = 0
        effect_stabilizing = False
        confirmed_divergence_steps = 0
        primary_effect_streak = 0
        primary_effect_first_step = None
        frontier_fallback_step = None
        frontier_fallback_prompt = None
        frontier_fallback_triggered = False
        frontier_completion_step = None
        frontier_completion_prompt = None
        frontier_completion_triggered = False
        active_frontier_followup_steps = self.frontier_followup_steps
        prompt = queued.policy_prompt
        prompt_history = [prompt]
        sourced_place_schemas = {
            "place-on",
            "place-in",
            "place-relative",
        }
        held_place_schemas = {
            "place-held-on",
            "place-held-in",
            "place-held-relative",
            "put-down",
        }
        place_schemas = sourced_place_schemas | held_place_schemas
        divergence_confirmation_steps = (
            self.held_target_divergence_confirmation_steps
            if queued.action.schema in held_place_schemas
            else self.target_divergence_confirmation_steps
        )
        effect_stabilization_steps = 0
        confirmation_steps = self.effect_confirmation_steps
        if queued.action.schema == "close-access":
            effect_stabilization_steps = self.access_effect_stabilization_steps
        elif queued.action.schema in sourced_place_schemas:
            effect_stabilization_steps = self.place_effect_stabilization_steps
            if self.place_effect_confirmation_steps is not None:
                confirmation_steps = self.place_effect_confirmation_steps
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
                if effect_stabilizing:
                    low_level_action = np.zeros(7, dtype=np.float64)
                    low_level_action[-1] = self._last_gripper_command
                elif not action_queue:
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

                if not effect_stabilizing:
                    low_level_action = action_queue.popleft()
                if not self.safety.validate_action(low_level_action):
                    raise EpisodeInvalid("queued policy action failed safety recheck")
                self._last_gripper_command = float(low_level_action[-1])
                _, _, done, _ = self.store.advance(
                    lambda: self.env.step(low_level_action.tolist())
                )
                actions.append(low_level_action.copy())
                self.total_action_steps += 1
                if bool(done):
                    status = ExecutorStatus.EPISODE_SUCCESS
                    reason = "simulator task predicate became true"
                    break
                if self.stop_on_effects:
                    effects_satisfied, primary_satisfied, target_diverged = (
                        self.grounder.observe_action_progress_details(
                            queued.action,
                            completion_positive=queued.completion_positive,
                            completion_negative=queued.completion_negative,
                        )
                    )
                    if primary_satisfied:
                        primary_effect_streak += 1
                        if primary_effect_first_step is None:
                            primary_effect_first_step = len(actions)
                    else:
                        primary_effect_streak = 0
                    if effect_stabilizing:
                        if effects_satisfied:
                            stabilized_effect_steps += 1
                            if (
                                stabilized_effect_steps
                                >= effect_stabilization_steps
                            ):
                                reason = "observed stabilized declared effects"
                                break
                        else:
                            effect_stabilizing = False
                            stabilized_effect_steps = 0
                            confirmed_effect_steps = 0
                        continue
                    if effects_satisfied:
                        confirmed_effect_steps += 1
                        if confirmed_effect_steps >= confirmation_steps:
                            if (
                                effect_stabilization_steps > 0
                            ):
                                effect_stabilizing = True
                                stabilized_effect_steps = 0
                                phase_flushed += len(action_queue)
                                action_queue.clear()
                                continue
                            reason = (
                                "observed frontier effects"
                                if queued.completion_mode == "DAG_FRONTIER"
                                else "observed declared effects"
                            )
                            break
                    else:
                        confirmed_effect_steps = 0
                    if (
                        queued.completion_mode == "DAG_FRONTIER"
                        and self.frontier_completion_followup_steps is not None
                        and (
                            not self.frontier_completion_recovery_only
                            or queued.recovery_frontier
                        )
                        and not frontier_completion_triggered
                        and not frontier_fallback_triggered
                        and primary_effect_streak >= confirmation_steps
                    ):
                        frontier_completion_triggered = True
                        frontier_completion_step = len(actions)
                        completion_prompt = (
                            self.prompt_renderer.render_frontier_completion(
                                queued.action
                            )
                        )
                        if completion_prompt != prompt:
                            phase_flushed += len(action_queue)
                            action_queue.clear()
                            prompt = completion_prompt
                            prompt_history.append(prompt)
                        frontier_completion_prompt = completion_prompt
                        active_frontier_followup_steps = (
                            self.frontier_completion_followup_steps
                        )
                        confirmed_divergence_steps = 0
                    if (
                        not effects_satisfied
                        and queued.completion_mode == "DAG_FRONTIER"
                        and active_frontier_followup_steps is not None
                        and primary_effect_streak >= active_frontier_followup_steps
                    ):
                        reason = "frontier follow-up deadline reached"
                        break
                    if (
                        not effects_satisfied
                        and target_diverged
                        and divergence_confirmation_steps > 0
                    ):
                        confirmed_divergence_steps += 1
                        if (
                            confirmed_divergence_steps
                            >= divergence_confirmation_steps
                        ):
                            reason = "observed target-location divergence"
                            break
                    else:
                        confirmed_divergence_steps = 0
                    if (
                        self.frontier_fallback_after_steps is not None
                        and not frontier_fallback_triggered
                        and not frontier_completion_triggered
                        and primary_effect_first_step is None
                        and len(actions) >= self.frontier_fallback_after_steps
                    ):
                        frontier_fallback_triggered = True
                        frontier_fallback_step = len(actions)
                        fallback_prompt = self.prompt_renderer.render_frontier_fallback(
                            queued.action
                        )
                        if fallback_prompt != prompt:
                            phase_flushed += len(action_queue)
                            action_queue.clear()
                            prompt = fallback_prompt
                            prompt_history.append(prompt)
                        frontier_fallback_prompt = fallback_prompt
                        if self.frontier_fallback_followup_steps is not None:
                            active_frontier_followup_steps = (
                                self.frontier_fallback_followup_steps
                            )
                        confirmed_divergence_steps = 0
                if (
                    not frontier_fallback_triggered
                    and not frontier_completion_triggered
                    and self.prompt_renderer.has_phase(queued.action, "finish")
                ):
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
                    and reason
                    in {
                        "observed declared effects",
                        "observed frontier effects",
                    }
                )
                hold[-1] = -1.0 if release_completion else self._last_gripper_command
                for _ in range(self.settling_steps):
                    _, _, settling_done, _ = self.store.advance(
                        lambda: self.env.step(hold.tolist())
                    )
                    if bool(settling_done):
                        status = ExecutorStatus.EPISODE_SUCCESS
                        reason = "simulator task predicate became true during settling"
                        break
            except Exception as error:
                status = ExecutorStatus.SETTLING_TIMEOUT
                reason = f"settling failed: {error}"

        settled_epoch = (
            self.store.epoch_id
            if status
            in {
                ExecutorStatus.SUCCEEDED,
                ExecutorStatus.EPISODE_SUCCESS,
                ExecutorStatus.EXECUTOR_FAILED,
            }
            else None
        )
        post_snapshot = None
        post_snapshot_error = None
        post_stop_reobservation_steps = 0
        if settled_epoch is not None and status is not ExecutorStatus.EPISODE_SUCCESS:
            try:
                post_snapshot = self.grounder.peek_snapshot()
            except GroundingError as error:
                post_snapshot_error = str(error)

            needs_reobservation = post_snapshot is None or not post_snapshot.satisfies(
                positive=queued.completion_positive,
                negative=queued.completion_negative,
            )
            if needs_reobservation and self.post_stop_grounding_reobservation_steps:
                try:
                    for _ in range(self.post_stop_grounding_reobservation_steps):
                        observation, _, _, _ = self.env.step(hold.tolist())
                        self.store.update(observation)
                        post_stop_reobservation_steps += 1
                except Exception as reobservation_error:
                    status = ExecutorStatus.SETTLING_TIMEOUT
                    settled_epoch = None
                    reason = f"post-stop reobservation failed: {reobservation_error}"
                if settled_epoch is not None and post_stop_reobservation_steps:
                    settled_epoch = self.store.epoch_id
                    try:
                        post_snapshot = self.grounder.peek_snapshot()
                        post_snapshot_error = None
                    except GroundingError as retry_error:
                        post_snapshot_error = str(retry_error)
        self.results.append(
            AttemptResult(
                attempt_id=start.attempt_id,
                context=queued.context,
                action=queued.action,
                prompt=prompt_history[0],
                prompt_history=tuple(prompt_history),
                completion_mode=queued.completion_mode,
                recovery_frontier=queued.recovery_frontier,
                completion_occurrence_ids=queued.completion_occurrence_ids,
                completion_actions=queued.completion_actions,
                completion_positive=queued.completion_positive,
                completion_negative=queued.completion_negative,
                primary_effect_first_step=primary_effect_first_step,
                frontier_followup_limit=(
                    active_frontier_followup_steps
                    if queued.completion_mode == "DAG_FRONTIER"
                    else None
                ),
                frontier_completion_step=frontier_completion_step,
                frontier_completion_prompt=frontier_completion_prompt,
                frontier_fallback_step=frontier_fallback_step,
                frontier_fallback_prompt=frontier_fallback_prompt,
                pre_epoch=queued.pre_epoch,
                post_epoch=settled_epoch,
                executor_status=status,
                stopped=status
                in {
                    ExecutorStatus.SUCCEEDED,
                    ExecutorStatus.EPISODE_SUCCESS,
                    ExecutorStatus.EXECUTOR_FAILED,
                },
                stop_evidence=(
                    "native success returned by synchronous env.step"
                    if status is ExecutorStatus.EPISODE_SUCCESS
                    else "synchronous env.step returned and local action deque was flushed"
                    if status
                    in {ExecutorStatus.SUCCEEDED, ExecutorStatus.EXECUTOR_FAILED}
                    else "no STOPPED acknowledgement"
                ),
                actions=tuple(actions),
                frames=tuple(frames),
                inference_requests=inference_requests,
                detector_calls=self.grounder.detector_calls - detector_start,
                unused_actions_flushed=unused_actions_flushed,
                post_stop_reobservation_steps=post_stop_reobservation_steps,
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
