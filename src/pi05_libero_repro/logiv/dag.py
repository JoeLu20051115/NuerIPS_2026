from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from pathlib import Path
from typing import AbstractSet, Any, FrozenSet, Mapping, Sequence, Tuple

from pi05_libero_repro.logiv.model import ContextEnvelope, Fact, GroundAction, TaskProblem
from pi05_libero_repro.logiv.val import PlanCertificate, verify_certificate


class CompilerError(RuntimeError):
    pass


class NodeKind(str, Enum):
    INIT = "INIT"
    ACTION = "ACTION"
    GOAL = "GOAL"


@dataclass(frozen=True, order=True)
class SignedLiteral:
    fact: Fact
    positive: bool

    def __str__(self) -> str:
        return str(self.fact) if self.positive else f"not {self.fact}"


@dataclass(frozen=True)
class GraphNode:
    node_id: str
    kind: NodeKind
    canonical_rank: int
    action: GroundAction | None = None
    lineage_root: str | None = None
    instruction: str | None = None


@dataclass(frozen=True, order=True)
class CausalLink:
    source: str
    literal: SignedLiteral
    target: str


@dataclass(frozen=True, order=True)
class ConflictReason:
    literal: SignedLiteral
    protected_source: str
    protected_target: str
    threat: str


@dataclass(frozen=True)
class GraphEdge:
    source: str
    target: str
    support_literals: FrozenSet[SignedLiteral] = frozenset()
    conflict_reasons: FrozenSet[ConflictReason] = frozenset()


@dataclass(frozen=True)
class CausalGraph:
    graph_version: str
    graph_hash: str
    source_epoch: int
    certificate_hash: str
    nodes: Tuple[GraphNode, ...]
    edges: Tuple[GraphEdge, ...]
    causal_links: Tuple[CausalLink, ...]
    canonical_agenda: Tuple[str, ...]

    @property
    def node_map(self) -> dict[str, GraphNode]:
        return {node.node_id: node for node in self.nodes}

    def edge(self, source: str, target: str) -> GraphEdge | None:
        return next(
            (edge for edge in self.edges if edge.source == source and edge.target == target),
            None,
        )

    def action_layer_width(self) -> int:
        action_ids = set(self.canonical_agenda)
        outgoing = {node_id: set() for node_id in action_ids}
        indegree = {node_id: 0 for node_id in action_ids}
        for edge in self.edges:
            if edge.source in action_ids and edge.target in action_ids:
                outgoing[edge.source].add(edge.target)
                indegree[edge.target] += 1
        remaining = set(action_ids)
        maximum = 0
        while remaining:
            layer = sorted(
                (node_id for node_id in remaining if indegree[node_id] == 0),
                key=lambda node_id: self.node_map[node_id].canonical_rank,
            )
            if not layer:
                raise CompilerError("cycle in action subgraph")
            maximum = max(maximum, len(layer))
            for node_id in layer:
                remaining.remove(node_id)
                for target in outgoing[node_id]:
                    indegree[target] -= 1
        return maximum

    def ready_action_ids(self, committed: AbstractSet[str]) -> Tuple[str, ...]:
        """Return the uncommitted action antichain enabled by graph precedence."""
        action_ids = set(self.canonical_agenda)
        action_predecessors = {node_id: set() for node_id in action_ids}
        for edge in self.edges:
            if edge.source in action_ids and edge.target in action_ids:
                action_predecessors[edge.target].add(edge.source)
        return tuple(
            node_id
            for node_id in self.canonical_agenda
            if node_id not in committed
            and action_predecessors[node_id].issubset(committed)
        )


@dataclass
class _EdgeBuilder:
    support_literals: set[SignedLiteral]
    conflict_reasons: set[ConflictReason]


def _read_sidecar(sidecar: bytes, plan: Sequence[GroundAction]) -> list[Mapping[str, Any]]:
    try:
        payload = json.loads(sidecar.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CompilerError(f"occurrence sidecar parse error: {error}") from error
    if not isinstance(payload, list) or len(payload) != len(plan):
        raise CompilerError("occurrence sidecar length mismatch")
    result = []
    seen = set()
    for index, (item, action) in enumerate(zip(payload, plan)):
        if not isinstance(item, dict):
            raise CompilerError(f"occurrence sidecar item {index} is not an object")
        occurrence_id = item.get("occurrence_id")
        if not isinstance(occurrence_id, str) or not occurrence_id or occurrence_id in seen:
            raise CompilerError(f"invalid or duplicate occurrence_id at index {index}")
        if item.get("schema") != action.schema or item.get("arguments") != list(action.arguments):
            raise CompilerError(f"occurrence sidecar action mismatch at index {index}")
        seen.add(occurrence_id)
        result.append(item)
    return result


def _topological_node_ids(graph: CausalGraph) -> Tuple[str, ...]:
    node_map = graph.node_map
    outgoing = {node_id: set() for node_id in node_map}
    indegree = {node_id: 0 for node_id in node_map}
    for edge in graph.edges:
        if edge.target not in outgoing[edge.source]:
            outgoing[edge.source].add(edge.target)
            indegree[edge.target] += 1
    available = {node_id for node_id, degree in indegree.items() if degree == 0}
    ordered = []
    while available:
        node_id = min(
            available,
            key=lambda value: (node_map[value].canonical_rank, value),
        )
        available.remove(node_id)
        ordered.append(node_id)
        for target in outgoing[node_id]:
            indegree[target] -= 1
            if indegree[target] == 0:
                available.add(target)
    if len(ordered) != len(node_map):
        raise CompilerError("graph contains a cycle")
    return tuple(ordered)


def validate_graph(graph: CausalGraph) -> None:
    node_map = graph.node_map
    if len(node_map) != len(graph.nodes):
        raise CompilerError("duplicate graph node ID")
    if node_map.get("INIT", GraphNode("", NodeKind.ACTION, 0)).kind is not NodeKind.INIT:
        raise CompilerError("graph must contain virtual INIT")
    if node_map.get("GOAL", GraphNode("", NodeKind.ACTION, 0)).kind is not NodeKind.GOAL:
        raise CompilerError("graph must contain virtual GOAL")
    pairs = set()
    for edge in graph.edges:
        if edge.source not in node_map or edge.target not in node_map:
            raise CompilerError("edge references an unknown node")
        if edge.source == edge.target:
            raise CompilerError("self-loop is forbidden")
        pair = (edge.source, edge.target)
        if pair in pairs:
            raise CompilerError("parallel graph edges are forbidden")
        pairs.add(pair)
        if node_map[edge.source].canonical_rank >= node_map[edge.target].canonical_rank:
            raise CompilerError("reverse canonical rank edge is forbidden")
        if not edge.support_literals and not edge.conflict_reasons:
            raise CompilerError("edge must contain support or conflict provenance")
    ordered = _topological_node_ids(graph)
    agenda = tuple(node_id for node_id in ordered if node_map[node_id].kind is NodeKind.ACTION)
    if agenda != graph.canonical_agenda:
        raise CompilerError("canonical agenda does not match stable topological order")
    if set(agenda) != {
        node.node_id for node in graph.nodes if node.kind is NodeKind.ACTION
    }:
        raise CompilerError("canonical agenda/action node mismatch")


def _graph_payload(
    source_epoch: int,
    certificate_hash: str,
    nodes: Sequence[GraphNode],
    edges: Sequence[GraphEdge],
    causal_links: Sequence[CausalLink],
) -> bytes:
    payload = {
        "source_epoch": source_epoch,
        "certificate_hash": certificate_hash,
        "nodes": [
            {
                "id": node.node_id,
                "kind": node.kind.value,
                "rank": node.canonical_rank,
                "action": node.action.pddl() if node.action else None,
                "lineage_root": node.lineage_root,
            }
            for node in nodes
        ],
        "edges": [
            {
                "source": edge.source,
                "target": edge.target,
                "support": sorted(str(item) for item in edge.support_literals),
                "conflicts": sorted(
                    (
                        str(item.literal),
                        item.protected_source,
                        item.protected_target,
                        item.threat,
                    )
                    for item in edge.conflict_reasons
                ),
            }
            for edge in edges
        ],
        "causal_links": [
            (link.source, str(link.literal), link.target) for link in causal_links
        ],
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


class CausalDagCompiler:
    def __init__(self, val_binary: Path | str, *, timeout_seconds: float) -> None:
        self.val_binary = Path(val_binary)
        self.timeout_seconds = float(timeout_seconds)

    def compile(
        self,
        problem: TaskProblem,
        plan: Sequence[GroundAction],
        occurrence_sidecar: bytes,
        certificate: PlanCertificate | None,
        context: ContextEnvelope,
        *,
        forbidden_retry_keys: FrozenSet[Any] = frozenset(),
        retry_ledger_version: int = 0,
    ) -> CausalGraph:
        if certificate is None or not verify_certificate(
            certificate,
            problem=problem,
            plan=plan,
            occurrence_sidecar=occurrence_sidecar,
            context=context,
            val_binary=self.val_binary,
            timeout_seconds=self.timeout_seconds,
            forbidden_retry_keys=forbidden_retry_keys,
            retry_ledger_version=retry_ledger_version,
        ):
            raise CompilerError("certificate mismatch")
        return self._compile_bound(
            problem,
            plan,
            occurrence_sidecar,
            certificate.certificate_hash,
            context,
        )

    def _compile_bound(
        self,
        problem: TaskProblem,
        plan: Sequence[GroundAction],
        occurrence_sidecar: bytes,
        certificate_hash: str,
        context: ContextEnvelope,
    ) -> CausalGraph:
        sidecar = _read_sidecar(occurrence_sidecar, plan)
        nodes = [GraphNode("INIT", NodeKind.INIT, -1)]
        for rank, (action, metadata) in enumerate(zip(plan, sidecar)):
            nodes.append(
                GraphNode(
                    node_id=str(metadata["occurrence_id"]),
                    kind=NodeKind.ACTION,
                    canonical_rank=rank,
                    action=action,
                    lineage_root=(
                        str(metadata["lineage_root"])
                        if metadata.get("lineage_root") is not None
                        else None
                    ),
                    instruction=(
                        str(metadata["instruction"])
                        if metadata.get("instruction") is not None
                        else None
                    ),
                )
            )
        nodes.append(GraphNode("GOAL", NodeKind.GOAL, len(plan)))
        ranks = {node.node_id: node.canonical_rank for node in nodes}
        action_nodes = [node for node in nodes if node.kind is NodeKind.ACTION]

        producers: dict[SignedLiteral, str] = {}
        for fact in problem.initial_state:
            producers[SignedLiteral(fact, True)] = "INIT"
        for fact in problem.initial_false:
            producers[SignedLiteral(fact, False)] = "INIT"
        causal_links = []
        edges: dict[tuple[str, str], _EdgeBuilder] = {}

        def add_support(source: str, target: str, literal: SignedLiteral) -> None:
            causal_links.append(CausalLink(source, literal, target))
            builder = edges.setdefault((source, target), _EdgeBuilder(set(), set()))
            builder.support_literals.add(literal)

        for node in action_nodes:
            assert node.action is not None
            required = [SignedLiteral(fact, True) for fact in node.action.preconditions]
            required.extend(
                SignedLiteral(fact, False) for fact in node.action.negative_preconditions
            )
            for literal in sorted(required):
                source = producers.get(literal)
                if source is None:
                    raise CompilerError(f"missing nominal producer for {literal} -> {node.node_id}")
                add_support(source, node.node_id, literal)
            for fact in node.action.del_effects:
                producers.pop(SignedLiteral(fact, True), None)
                producers[SignedLiteral(fact, False)] = node.node_id
            for fact in node.action.add_effects:
                producers.pop(SignedLiteral(fact, False), None)
                producers[SignedLiteral(fact, True)] = node.node_id

        goal_literals = [SignedLiteral(fact, True) for fact in problem.goal]
        goal_literals.extend(SignedLiteral(fact, False) for fact in problem.negative_goal)
        for literal in sorted(goal_literals):
            source = producers.get(literal)
            if source is None:
                raise CompilerError(f"missing nominal producer for {literal} -> GOAL")
            add_support(source, "GOAL", literal)

        for link in causal_links:
            for threat_node in action_nodes:
                if threat_node.node_id in {link.source, link.target}:
                    continue
                assert threat_node.action is not None
                threatening_effects = (
                    threat_node.action.del_effects
                    if link.literal.positive
                    else threat_node.action.add_effects
                )
                if link.literal.fact not in threatening_effects:
                    continue
                threat_rank = ranks[threat_node.node_id]
                source_rank = ranks[link.source]
                target_rank = ranks[link.target]
                reason = ConflictReason(
                    literal=link.literal,
                    protected_source=link.source,
                    protected_target=link.target,
                    threat=threat_node.node_id,
                )
                if threat_rank < source_rank:
                    ordered_pair = (threat_node.node_id, link.source)
                elif threat_rank > target_rank:
                    ordered_pair = (link.target, threat_node.node_id)
                else:
                    raise CompilerError(f"causal link is threatened in certified order: {link}")
                edges.setdefault(ordered_pair, _EdgeBuilder(set(), set())).conflict_reasons.add(
                    reason
                )

        graph_edges = tuple(
            GraphEdge(
                source=source,
                target=target,
                support_literals=frozenset(builder.support_literals),
                conflict_reasons=frozenset(builder.conflict_reasons),
            )
            for (source, target), builder in sorted(
                edges.items(), key=lambda item: (ranks[item[0][0]], ranks[item[0][1]])
            )
        )
        causal_links_tuple = tuple(
            sorted(causal_links, key=lambda link: (ranks[link.source], ranks[link.target], str(link.literal)))
        )
        node_tuple = tuple(nodes)
        payload = _graph_payload(
            context.epoch_id,
            certificate_hash,
            node_tuple,
            graph_edges,
            causal_links_tuple,
        )
        graph_hash = hashlib.sha256(payload).hexdigest()
        provisional = CausalGraph(
            graph_version=f"graph-{graph_hash[:16]}",
            graph_hash=graph_hash,
            source_epoch=context.epoch_id,
            certificate_hash=certificate_hash,
            nodes=node_tuple,
            edges=graph_edges,
            causal_links=causal_links_tuple,
            canonical_agenda=(),
        )
        ordered = _topological_node_ids(provisional)
        canonical_agenda = tuple(
            node_id
            for node_id in ordered
            if provisional.node_map[node_id].kind is NodeKind.ACTION
        )
        graph = CausalGraph(
            graph_version=provisional.graph_version,
            graph_hash=provisional.graph_hash,
            source_epoch=provisional.source_epoch,
            certificate_hash=provisional.certificate_hash,
            nodes=provisional.nodes,
            edges=provisional.edges,
            causal_links=provisional.causal_links,
            canonical_agenda=canonical_agenda,
        )
        validate_graph(graph)
        return graph


class SchemaOnlyCausalDagCompiler:
    """Ablation compiler that shares graph semantics but never accepts a VAL certificate."""

    def compile(
        self,
        problem: TaskProblem,
        plan: Sequence[GroundAction],
        occurrence_sidecar: bytes,
        context: ContextEnvelope,
    ) -> CausalGraph:
        binding = hashlib.sha256(
            b"schema-only\0"
            + _graph_payload(context.epoch_id, "NO_VAL_CERTIFICATE", (), (), ())
            + occurrence_sidecar
            + b"".join(action.pddl().encode("utf-8") + b"\n" for action in plan)
        ).hexdigest()
        helper = object.__new__(CausalDagCompiler)
        return helper._compile_bound(
            problem,
            plan,
            occurrence_sidecar,
            binding,
            context,
        )
