from __future__ import annotations

import copy
from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from typing import Any, Callable, Mapping

from pi05_libero_repro.logiv.dag import CausalGraph, NodeKind
from pi05_libero_repro.logiv.model import (
    Fact,
    FactSnapshot,
    TaskProblem,
    TruthValue,
)


_HAND_EMPTY = Fact("handempty", ())


class OnlineDeviationKind(str, Enum):
    GOAL_REGRESSION = "GOAL_REGRESSION"
    UNPLANNED_RECOVERY_SURFACE = "UNPLANNED_RECOVERY_SURFACE"
    FRONTIER_STALL = "FRONTIER_STALL"


@dataclass(frozen=True)
class OnlineRepairRequest:
    kind: OnlineDeviationKind
    policy_step: int
    first_observed_step: int
    signature: tuple[str, ...]
    snapshot: FactSnapshot
    observation: Mapping[str, Any]
    source_graph_hash: str
    request_sha256: str


class OnlineGraphDeviationDetector:
    """Latch the first strictly confirmed deviation from a certified task graph."""

    def __init__(
        self,
        problem: TaskProblem,
        graph: CausalGraph,
        *,
        confirmation_count: int,
        recovery_surface_confirmation_count: int | None = None,
        min_intervention_step: int,
        stall_steps: int,
        recovery_requires_achieved_goal: bool = False,
        stall_requires_achieved_goal: bool = False,
        stall_ignores_holding: bool = False,
        stall_requires_handempty: bool = False,
    ) -> None:
        if confirmation_count <= 0 or min_intervention_step < 0 or stall_steps <= 0:
            raise ValueError("invalid online deviation detector bounds")
        if (
            recovery_surface_confirmation_count is not None
            and recovery_surface_confirmation_count <= 0
        ):
            raise ValueError("invalid recovery-surface confirmation count")
        self.problem = problem
        self.graph = graph
        self.confirmation_count = confirmation_count
        self.recovery_surface_confirmation_count = (
            recovery_surface_confirmation_count
        )
        self.min_intervention_step = min_intervention_step
        self.stall_steps = stall_steps
        self.recovery_requires_achieved_goal = recovery_requires_achieved_goal
        self.stall_requires_achieved_goal = stall_requires_achieved_goal
        self.stall_ignores_holding = stall_ignores_holding
        self.stall_requires_handempty = stall_requires_handempty
        relevant = set(problem.goal | problem.negative_goal)
        for node in graph.nodes:
            if node.action is None:
                continue
            relevant.update(
                node.action.preconditions
                | node.action.negative_preconditions
                | node.action.add_effects
                | node.action.del_effects
            )
        self._relevant_facts = frozenset(relevant)
        self._achieved_positive: set[Fact] = set()
        self._achieved_negative: set[Fact] = set()
        self._strict_achieved_positive: set[Fact] = set()
        self._strict_achieved_negative: set[Fact] = set()
        self._progress_signature: tuple[Any, ...] | None = None
        self._progress_started_step: int | None = None
        self._candidate_key: tuple[str, ...] | None = None
        self._candidate_count = 0
        self._candidate_first_step = 0
        self.latched_request: OnlineRepairRequest | None = None

    def _goal_satisfied(self, snapshot: FactSnapshot) -> bool:
        return snapshot.satisfies(
            positive=self.problem.goal,
            negative=self.problem.negative_goal,
        )

    def _update_achieved_goals(self, snapshot: FactSnapshot) -> None:
        self._achieved_positive.update(self.problem.goal & snapshot.true_facts)
        self._achieved_negative.update(
            self.problem.negative_goal & snapshot.false_facts
        )

    def _update_strict_achieved_goals(self, snapshot: FactSnapshot) -> None:
        self._strict_achieved_positive.update(
            self.problem.goal & snapshot.true_facts
        )
        self._strict_achieved_negative.update(
            self.problem.negative_goal & snapshot.false_facts
        )

    def _goal_regression(self, snapshot: FactSnapshot) -> tuple[str, ...]:
        return tuple(
            sorted(
                [
                    f"+{fact.pddl()}"
                    for fact in self._achieved_positive
                    if snapshot.truth(fact) is TruthValue.FALSE
                ]
                + [
                    f"-{fact.pddl()}"
                    for fact in self._achieved_negative
                    if snapshot.truth(fact) is TruthValue.TRUE
                ]
            )
        )

    def _recovery_surface(self, snapshot: FactSnapshot) -> tuple[str, ...]:
        if self.recovery_requires_achieved_goal and not (
            self._achieved_positive or self._achieved_negative
        ):
            return ()
        excluded = self.problem.initial_state | self.problem.goal
        return tuple(
            sorted(
                fact.pddl()
                for fact in snapshot.true_facts
                if fact.predicate == "at"
                and len(fact.arguments) == 2
                and fact.arguments[1].endswith("recovery_surface")
                and fact not in excluded
            )
        )

    def _state_signature(
        self, snapshot: FactSnapshot, graph_state: Mapping[str, Any]
    ) -> tuple[Any, ...]:
        completed_nodes = tuple(
            str(item.get("node_id"))
            for item in graph_state.get("nodes", ())
            if item.get("node_id") in self.graph.canonical_agenda
            and item.get("status") == "COMPLETED"
        )
        holdings = (
            ()
            if self.stall_ignores_holding
            else tuple(
                fact.pddl()
                for fact in sorted(
                    snapshot.true_facts, key=lambda item: item.pddl()
                )
                if fact.predicate == "holding"
            )
        )
        return (
            completed_nodes,
            tuple(sorted(fact.pddl() for fact in self._achieved_positive)),
            tuple(sorted(fact.pddl() for fact in self._achieved_negative)),
            holdings,
        )

    @staticmethod
    def _verified_handempty(snapshot: FactSnapshot) -> bool:
        return (
            snapshot.truth(_HAND_EMPTY) is TruthValue.TRUE
            and not any(
                fact.predicate == "holding"
                for fact in snapshot.true_facts
            )
        )

    def _stalled_frontier(
        self,
        snapshot: FactSnapshot,
        graph_state: Mapping[str, Any],
        policy_step: int,
    ) -> tuple[str, ...]:
        signature = self._state_signature(snapshot, graph_state)
        if policy_step < self.min_intervention_step:
            self._progress_signature = None
            self._progress_started_step = None
            return ()
        if signature != self._progress_signature:
            self._progress_signature = signature
            self._progress_started_step = policy_step
            return ()
        if self._progress_started_step is None:
            self._progress_started_step = policy_step
            return ()
        if self.stall_requires_achieved_goal and not (
            self._achieved_positive or self._achieved_negative
        ):
            return ()
        if (
            self.stall_requires_handempty
            and not self._verified_handempty(snapshot)
        ):
            return ()
        active = tuple(
            sorted(
                str(item.get("node_id"))
                for item in graph_state.get("nodes", ())
                if item.get("node_id") in self.graph.canonical_agenda
                and item.get("status") in {"READY", "ACTIVE"}
            )
        )
        if (
            not active
            or self._goal_satisfied(snapshot)
            or policy_step - self._progress_started_step < self.stall_steps
        ):
            return ()
        return active

    def _candidate(
        self,
        snapshot: FactSnapshot,
        graph_state: Mapping[str, Any],
        policy_step: int,
    ) -> tuple[OnlineDeviationKind, tuple[str, ...]] | None:
        regression = self._goal_regression(snapshot)
        if regression:
            return OnlineDeviationKind.GOAL_REGRESSION, regression
        recovery = self._recovery_surface(snapshot)
        if recovery:
            return OnlineDeviationKind.UNPLANNED_RECOVERY_SURFACE, recovery
        stalled = self._stalled_frontier(snapshot, graph_state, policy_step)
        if stalled:
            return OnlineDeviationKind.FRONTIER_STALL, stalled
        return None

    def _strictly_confirmed(
        self,
        kind: OnlineDeviationKind,
        signature: tuple[str, ...],
        snapshot: FactSnapshot,
    ) -> bool:
        if kind is OnlineDeviationKind.GOAL_REGRESSION:
            expected = self._goal_regression(snapshot)
            return expected == signature
        if kind is OnlineDeviationKind.UNPLANNED_RECOVERY_SURFACE:
            if self.recovery_requires_achieved_goal and not (
                self._strict_achieved_positive
                or self._strict_achieved_negative
            ):
                return False
            return self._recovery_surface(snapshot) == signature
        if self._goal_satisfied(snapshot):
            return False
        if self.stall_requires_achieved_goal and not (
            self._strict_achieved_positive or self._strict_achieved_negative
        ):
            return False
        if (
            self.stall_requires_handempty
            and not self._verified_handempty(snapshot)
        ):
            return False
        node_map = self.graph.node_map
        actions = [
            node_map[node_id].action
            for node_id in signature
            if node_id in node_map and node_map[node_id].kind is NodeKind.ACTION
        ]
        if not actions or any(action is None for action in actions):
            return False
        for action in actions:
            assert action is not None
            effects = action.add_effects | action.del_effects
            if snapshot.unknown(effects) or snapshot.satisfies(
                positive=action.add_effects,
                negative=action.del_effects,
            ):
                return False
        return True

    @staticmethod
    def _request_hash(
        kind: OnlineDeviationKind,
        policy_step: int,
        first_observed_step: int,
        signature: tuple[str, ...],
        snapshot: FactSnapshot,
        graph_hash: str,
    ) -> str:
        payload = {
            "first_observed_step": first_observed_step,
            "graph_hash": graph_hash,
            "kind": kind.value,
            "policy_step": policy_step,
            "signature": list(signature),
            "snapshot_evidence_hash": snapshot.evidence_hash,
        }
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(b"LOGIV_ONLINE_REPAIR_REQUEST_V1\0" + encoded).hexdigest()

    def _reset_candidate(self) -> None:
        self._candidate_key = None
        self._candidate_count = 0

    def observe(
        self,
        *,
        snapshot: FactSnapshot,
        graph_state: Mapping[str, Any],
        observation: Mapping[str, Any],
        strict_snapshot_reader: Callable[[Mapping[str, Any]], FactSnapshot],
    ) -> OnlineRepairRequest | None:
        if self.latched_request is not None:
            return None
        if graph_state.get("graph_hash") != self.graph.graph_hash:
            self._reset_candidate()
            return None
        self._update_achieved_goals(snapshot)
        candidate = self._candidate(
            snapshot, graph_state, int(graph_state.get("policy_step", 0))
        )
        if candidate is None:
            self._reset_candidate()
            return None
        kind, signature = candidate
        key = (kind.value, *signature)
        if key != self._candidate_key:
            self._candidate_key = key
            self._candidate_count = 1
            self._candidate_first_step = int(graph_state.get("policy_step", 0))
        else:
            self._candidate_count += 1
        required_confirmations = (
            self.recovery_surface_confirmation_count
            if kind is OnlineDeviationKind.UNPLANNED_RECOVERY_SURFACE
            and self.recovery_surface_confirmation_count is not None
            else self.confirmation_count
        )
        if self._candidate_count < required_confirmations:
            return None
        try:
            strict_snapshot = strict_snapshot_reader(observation)
        except Exception:
            self._reset_candidate()
            return None
        self._update_strict_achieved_goals(strict_snapshot)
        if not self._strictly_confirmed(kind, signature, strict_snapshot):
            self._reset_candidate()
            return None
        policy_step = int(graph_state.get("policy_step", 0))
        request = OnlineRepairRequest(
            kind=kind,
            policy_step=policy_step,
            first_observed_step=self._candidate_first_step,
            signature=signature,
            snapshot=strict_snapshot,
            observation=copy.deepcopy(observation),
            source_graph_hash=self.graph.graph_hash,
            request_sha256=self._request_hash(
                kind,
                policy_step,
                self._candidate_first_step,
                signature,
                strict_snapshot,
                self.graph.graph_hash,
            ),
        )
        self.latched_request = request
        return request
