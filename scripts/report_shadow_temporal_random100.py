#!/usr/bin/env python3
"""Audit a topology-only temporal Shadow rerun against a saved Base cohort."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


BASE_EXECUTION_FIELDS = (
    "steps",
    "base_policy_requests",
    "done_signal",
    "post_settling_success",
    "initial_state_sha256",
    "base_prompt_sha256",
    "base_checkpoint_sha256",
    "policy_client_config_sha256",
    "request_envelope_log_sha256",
    "actions_sha256",
)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _load_single_jsonl(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        values = [json.loads(line) for line in stream if line.strip()]
    if len(values) != 1 or not isinstance(values[0], dict):
        raise ValueError(f"expected exactly one JSONL object: {path}")
    return values[0]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _episode_artifact_dir(
    case_root: Path, *, task_id: int, episode_idx: int
) -> Path:
    return (
        case_root
        / "artifacts"
        / f"task_{task_id:02d}"
        / f"episode_{episode_idx:03d}"
    )


def _terminal_status(trace_item: dict[str, Any], node_id: str) -> str | None:
    return next(
        (
            str(node["status"])
            for node in trace_item.get("nodes", ())
            if node.get("node_id") == node_id
        ),
        None,
    )


def _stats(successes: Iterable[bool]) -> dict[str, Any]:
    values = list(successes)
    count = sum(values)
    return {
        "successes": count,
        "total": len(values),
        "rate": count / len(values) if values else None,
    }


def _format_rate(value: dict[str, Any]) -> str:
    return f"{value['successes']}/{value['total']}"


def _case_artifacts(
    *,
    manifest_case: dict[str, Any],
    baseline_root: Path,
    shadow_root: Path,
) -> dict[str, Any]:
    case_id = str(manifest_case["case_id"])
    task_id = int(manifest_case["task_id"])
    episode_idx = int(manifest_case["episode_idx"])
    base_case_root = baseline_root / "cases" / case_id / "base"
    shadow_case_root = shadow_root / "cases" / case_id / "shadow"
    base_artifacts = _episode_artifact_dir(
        base_case_root, task_id=task_id, episode_idx=episode_idx
    )
    shadow_artifacts = _episode_artifact_dir(
        shadow_case_root, task_id=task_id, episode_idx=episode_idx
    )

    base_record = _load_single_jsonl(base_case_root / "episodes.jsonl")
    shadow_record = _load_single_jsonl(shadow_case_root / "episodes.jsonl")
    base_execution = _load_json(base_artifacts / "base_execution.json")
    shadow_execution = _load_json(shadow_artifacts / "base_execution.json")
    base_rng = _load_json(base_artifacts / "policy_rng.json")
    shadow_rng = _load_json(shadow_artifacts / "policy_rng.json")
    graph = _load_json(shadow_artifacts / "graph.json")
    monitor = _load_json(shadow_artifacts / "shadow_monitor.json")
    trace = graph.get("state_trace")
    if not isinstance(trace, list) or not trace:
        raise ValueError(f"empty topology trace: {shadow_artifacts / 'graph.json'}")

    expected_seed_fields = {
        "master_seed": int(manifest_case["master_seed"]),
        "episode_seed": int(manifest_case["policy_seed"]),
        "simulator_seed": int(manifest_case["simulator_seed"]),
        "task_id": task_id,
        "episode_idx": episode_idx,
    }
    seed_exact = all(
        base_rng.get(field) == value and shadow_rng.get(field) == value
        for field, value in expected_seed_fields.items()
    )
    record_exact = all(
        base_record.get(field) == shadow_record.get(field)
        for field in ("task_id", "episode_idx", "init_state_sha256", "first_frame_sha256")
    )
    field_exact = {
        field: base_execution.get(field) == shadow_execution.get(field)
        for field in BASE_EXECUTION_FIELDS
    }

    graph_hash = graph.get("graph_hash")
    graph_version = graph.get("graph_version")
    fixed_graph = all(
        item.get("graph_hash") == graph_hash
        and item.get("graph_version") == graph_version
        for item in trace
    )
    trace_count_exact = len(trace) == int(monitor.get("callback_calls", -1))
    final = trace[-1]
    terminal_goal_status = _terminal_status(final, "GOAL")
    shadow_success = bool(shadow_record.get("success"))
    terminal_goal_success = terminal_goal_status == "COMPLETED"
    terminal_exact = terminal_goal_success == shadow_success
    settling_terminal_exact = (
        final.get("phase") == "SETTLING"
        and int(final.get("settling_step", -1)) == 10
        and int(final.get("policy_step", -1)) == int(shadow_record.get("steps", -2))
    )

    status_counts: Counter[str] = Counter()
    transition_counts: Counter[str] = Counter()
    certificate_counts: Counter[str] = Counter()
    certificate_transitions: Counter[str] = Counter()
    previous_nodes: dict[str, str] = {}
    previous_certificate: str | None = None
    for item in trace:
        nodes = {
            str(node["node_id"]): str(node["status"])
            for node in item.get("nodes", ())
        }
        status_counts.update(nodes.values())
        for node_id, status in nodes.items():
            previous = previous_nodes.get(node_id)
            if previous is not None and previous != status:
                transition_counts[f"{previous}->{status}"] += 1
        previous_nodes = nodes
        certificate = str(item.get("certificate_state"))
        certificate_counts[certificate] += 1
        if previous_certificate is not None and previous_certificate != certificate:
            certificate_transitions[f"{previous_certificate}->{certificate}"] += 1
        previous_certificate = certificate

    policy_terminal = next(
        (item for item in reversed(trace) if item.get("phase") == "POLICY"), None
    )
    policy_terminal_goal = (
        _terminal_status(policy_terminal, "GOAL")
        if policy_terminal is not None
        else None
    )

    return {
        **manifest_case,
        "task_name": shadow_record.get("task_name"),
        "base_success": bool(base_record.get("success")),
        "shadow_success": shadow_success,
        "base_steps": int(base_record.get("steps", 0)),
        "shadow_steps": int(shadow_record.get("steps", 0)),
        "valid": bool(base_record.get("valid")) and bool(shadow_record.get("valid")),
        "seed_exact": seed_exact,
        "record_identity_exact": record_exact,
        "base_execution_exact": all(field_exact.values()),
        "base_execution_field_exact": field_exact,
        "graph_path": str(shadow_artifacts / "graph.json"),
        "graph_hash": graph_hash,
        "fixed_graph": fixed_graph,
        "trace_snapshots": len(trace),
        "monitor_calls": int(monitor.get("callback_calls", 0)),
        "monitor_errors": int(monitor.get("aggregate_errors", 0)),
        "trace_count_exact": trace_count_exact,
        "terminal_goal_status": terminal_goal_status,
        "terminal_exact": terminal_exact,
        "settling_terminal_exact": settling_terminal_exact,
        "policy_terminal_goal_status": policy_terminal_goal,
        "completed_during_settling": (
            policy_terminal_goal != "COMPLETED"
            and terminal_goal_status == "COMPLETED"
        ),
        "has_precondition_unknown": status_counts["PRECONDITION_UNKNOWN"] > 0,
        "status_counts": dict(sorted(status_counts.items())),
        "transition_counts": dict(sorted(transition_counts.items())),
        "certificate_counts": dict(sorted(certificate_counts.items())),
        "certificate_transitions": dict(sorted(certificate_transitions.items())),
    }


def build_report(
    *, manifest: dict[str, Any], baseline_root: Path, shadow_root: Path
) -> dict[str, Any]:
    manifest_cases = manifest.get("cases")
    if not isinstance(manifest_cases, list) or len(manifest_cases) != 100:
        raise ValueError("the seed manifest must contain exactly 100 cases")
    cases = [
        _case_artifacts(
            manifest_case=item,
            baseline_root=baseline_root,
            shadow_root=shadow_root,
        )
        for item in manifest_cases
    ]

    status_counts: Counter[str] = Counter()
    transition_counts: Counter[str] = Counter()
    certificate_counts: Counter[str] = Counter()
    certificate_transitions: Counter[str] = Counter()
    field_exact_counts: Counter[str] = Counter()
    for item in cases:
        status_counts.update(item["status_counts"])
        transition_counts.update(item["transition_counts"])
        certificate_counts.update(item["certificate_counts"])
        certificate_transitions.update(item["certificate_transitions"])
        field_exact_counts.update(
            field
            for field, exact in item["base_execution_field_exact"].items()
            if exact
        )

    confusion = Counter()
    for item in cases:
        predicted = item["terminal_goal_status"] == "COMPLETED"
        actual = item["shadow_success"]
        confusion[(actual, predicted)] += 1

    by_task = []
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for item in cases:
        grouped[int(item["task_id"])].append(item)
    for task_id, task_cases in sorted(grouped.items()):
        by_task.append(
            {
                "task_id": task_id,
                "task_name": task_cases[0]["task_name"],
                "base": _stats(item["base_success"] for item in task_cases),
                "shadow": _stats(item["shadow_success"] for item in task_cases),
                "terminal_correct": sum(item["terminal_exact"] for item in task_cases),
                "settling_completions": sum(
                    item["completed_during_settling"] for item in task_cases
                ),
            }
        )

    base_stats = _stats(item["base_success"] for item in cases)
    shadow_stats = _stats(item["shadow_success"] for item in cases)
    trace_snapshots = sum(item["trace_snapshots"] for item in cases)
    monitor_calls = sum(item["monitor_calls"] for item in cases)
    audit = {
        "valid_pairs": sum(item["valid"] for item in cases),
        "seed_exact_pairs": sum(item["seed_exact"] for item in cases),
        "record_identity_exact_pairs": sum(
            item["record_identity_exact"] for item in cases
        ),
        "base_execution_exact_pairs": sum(
            item["base_execution_exact"] for item in cases
        ),
        "base_execution_field_exact_counts": dict(
            sorted(field_exact_counts.items())
        ),
        "fixed_graph_cases": sum(item["fixed_graph"] for item in cases),
        "settling_terminal_cases": sum(
            item["settling_terminal_exact"] for item in cases
        ),
        "trace_count_exact_cases": sum(item["trace_count_exact"] for item in cases),
        "monitor_calls": monitor_calls,
        "trace_snapshots": trace_snapshots,
        "trace_coverage": trace_snapshots / monitor_calls if monitor_calls else None,
        "monitor_errors": sum(item["monitor_errors"] for item in cases),
    }
    topology = {
        "terminal_correct": sum(item["terminal_exact"] for item in cases),
        "confusion": {
            "tp": confusion[(True, True)],
            "fn": confusion[(True, False)],
            "tn": confusion[(False, False)],
            "fp": confusion[(False, True)],
        },
        "successful_goal_recall": (
            confusion[(True, True)]
            / (confusion[(True, True)] + confusion[(True, False)])
            if confusion[(True, True)] + confusion[(True, False)]
            else None
        ),
        "settling_completions": sum(
            item["completed_during_settling"] for item in cases
        ),
        "precondition_unknown_cases": sum(
            item["has_precondition_unknown"] for item in cases
        ),
        "status_counts": dict(sorted(status_counts.items())),
        "transition_counts": dict(sorted(transition_counts.items())),
        "certificate_counts": dict(sorted(certificate_counts.items())),
        "certificate_transitions": dict(sorted(certificate_transitions.items())),
    }
    gates = {
        "valid_pairs_100": audit["valid_pairs"] == 100,
        "seed_exact_100": audit["seed_exact_pairs"] == 100,
        "base_execution_exact_100": audit["base_execution_exact_pairs"] == 100,
        "fixed_graph_100": audit["fixed_graph_cases"] == 100,
        "settling_terminal_100": audit["settling_terminal_cases"] == 100,
        "trace_count_exact_100": audit["trace_count_exact_cases"] == 100,
        "zero_monitor_errors": audit["monitor_errors"] == 0,
        "terminal_goal_exact_100": topology["terminal_correct"] == 100,
    }
    topology_gate_names = (
        "valid_pairs_100",
        "seed_exact_100",
        "base_execution_exact_100",
        "fixed_graph_100",
        "settling_terminal_100",
        "trace_count_exact_100",
        "zero_monitor_errors",
        "terminal_goal_exact_100",
    )
    gates["topology_judgment_ready"] = all(
        gates[name] for name in topology_gate_names
    )
    gates["ready_for_r2m_takeover_evaluation"] = (
        gates["topology_judgment_ready"]
        and gates["base_execution_exact_100"]
    )
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "manifest_sha256": manifest.get("sha256"),
            "baseline_root": str(baseline_root),
            "shadow_root": str(shadow_root),
            "arm": "SHADOW_LOGIV",
            "topology_only": True,
            "settling_steps": 10,
            "intervention": False,
        },
        "outcome": {
            "base": base_stats,
            "shadow": shadow_stats,
            "shadow_minus_base": shadow_stats["successes"]
            - base_stats["successes"],
            "base_success_shadow_failure": sum(
                item["base_success"] and not item["shadow_success"] for item in cases
            ),
            "base_failure_shadow_success": sum(
                not item["base_success"] and item["shadow_success"] for item in cases
            ),
        },
        "audit": audit,
        "topology": topology,
        "gates": gates,
        "by_task": by_task,
        "cases": cases,
    }


def render_markdown(report: dict[str, Any]) -> str:
    outcome = report["outcome"]
    audit = report["audit"]
    topology = report["topology"]
    gates = report["gates"]
    confusion = topology["confusion"]
    lines = [
        "# Temporal Shadow LOGIV：100 个随机仿真 case 拓扑判定复跑",
        "",
        "## 结论",
        "",
        (
            f"同一份随机 manifest 上，本次配对 Base 为 **{_format_rate(outcome['base'])}**，"
            f"本次只读 Shadow LOGIV 为 **{_format_rate(outcome['shadow'])}**。"
            "Shadow 没有接管或干预动作；本报告只验证拓扑判断，不把它解释为 R2M 收益。"
        ),
        "",
        (
            f"终态 GOAL 相对本次仿真原生 success：TP={confusion['tp']}、"
            f"FN={confusion['fn']}、TN={confusion['tn']}、FP={confusion['fp']}，"
            f"正确 **{topology['terminal_correct']}/100**；其中 "
            f"{topology['settling_completions']} 个 case 是在 settling 阶段才被确认完成。"
        ),
        "",
        (
            "Shadow 拓扑判定与只读同轨门槛：**"
            + ("通过" if gates["topology_judgment_ready"] else "未通过")
            + "**；进入 R2M 接管评估的前置门槛：**"
            + ("通过" if gates["ready_for_r2m_takeover_evaluation"] else "未通过")
            + "**。未通过的单项会在下面的审计数字中保留，不用成功率掩盖。"
        ),
        "",
        "## 每任务结果",
        "",
        "| Task | Base | Shadow | Shadow−Base | 终态图正确 | settling 才完成 |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in report["by_task"]:
        difference = item["shadow"]["successes"] - item["base"]["successes"]
        lines.append(
            f"| {item['task_id']} | {_format_rate(item['base'])} | "
            f"{_format_rate(item['shadow'])} | {difference:+d} | "
            f"{item['terminal_correct']}/10 | {item['settling_completions']} |"
        )

    field_counts = audit["base_execution_field_exact_counts"]
    lines.extend(
        [
            "",
            "## 拓扑与审计",
            "",
            f"- 有效配对：{audit['valid_pairs']}/100；seed 对齐：{audit['seed_exact_pairs']}/100；初态/首帧对齐：{audit['record_identity_exact_pairs']}/100。",
            f"- 固定图结构：{audit['fixed_graph_cases']}/100；终态均包含完整 settling：{audit['settling_terminal_cases']}/100。",
            f"- 实时回调 {audit['monitor_calls']} 次、图状态快照 {audit['trace_snapshots']} 个、覆盖率 {audit['trace_coverage']:.2%}、错误 {audit['monitor_errors']}。",
            f"- 终态成功召回率：{topology['successful_goal_recall']:.2%}；出现 `PRECONDITION_UNKNOWN` 的 case：{topology['precondition_unknown_cases']}/100。",
            "- 证书变化："
            + "、".join(
                f"`{key}`={value}"
                for key, value in topology["certificate_transitions"].items()
            )
            + "；这些候选 deviation 留给后续 R2M 接管时机评估。",
            f"- 与配对 Base 的 10 字段全量逐 case 完全一致：{audit['base_execution_exact_pairs']}/100。各字段一致数："
            + "、".join(f"`{field}`={field_counts.get(field, 0)}" for field in BASE_EXECUTION_FIELDS)
            + "。",
            "- 状态计数："
            + "、".join(f"`{key}`={value}" for key, value in topology["status_counts"].items())
            + "。",
            "- 主要状态变化："
            + "、".join(
                f"`{key}`={value}"
                for key, value in sorted(
                    topology["transition_counts"].items(),
                    key=lambda pair: (-pair[1], pair[0]),
                )[:12]
            )
            + "。",
            "",
            "## 随机种子与逐 case 结果",
            "",
            "`B/S/G` 分别表示配对 Base 成功、本次 Shadow 成功、Shadow 终态 GOAL 完成。",
            "",
            "| Case | Task | Episode | Master seed | Policy seed | Simulator seed | B/S/G |",
            "| --- | ---: | ---: | ---: | ---: | ---: | :---: |",
        ]
    )
    for item in report["cases"]:
        goal = item["terminal_goal_status"] == "COMPLETED"
        lines.append(
            f"| {item['case_id']} | {item['task_id']} | {item['episode_idx']} | "
            f"{item['master_seed']} | {item['policy_seed']} | {item['simulator_seed']} | "
            f"{int(item['base_success'])}/{int(item['shadow_success'])}/{int(goal)} |"
        )
    lines.extend(
        [
            "",
            "## 证据边界",
            "",
            "这是 `development_only`、`METADATA_ASSISTED`、oracle-grounded 的 simulator 评估。Shadow 只输出固定图及其时序节点状态，不含接管动作；真正的 LOGIV_R2M 仍需在 confirmed deviation 后调用独立 `π_recover`，并在 effect 与 protected invariants 验证后交还 Base。",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("runs/shadow-logiv-random100-20260804/seed_manifest.json"),
    )
    parser.add_argument(
        "--baseline-root",
        type=Path,
        default=Path("runs/base-random100-paired-v4-20260805"),
    )
    parser.add_argument(
        "--shadow-root",
        type=Path,
        default=Path("runs/shadow-logiv-random100-temporal-final-v4-20260805"),
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("results/logiv-shadow-temporal-random100-20260805.json"),
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=Path("results/logiv-shadow-temporal-random100-20260805.md"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = _load_json(args.manifest)
    manifest["sha256"] = _sha256(args.manifest)
    report = build_report(
        manifest=manifest,
        baseline_root=args.baseline_root,
        shadow_root=args.shadow_root,
    )
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.markdown_output.write_text(render_markdown(report), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
