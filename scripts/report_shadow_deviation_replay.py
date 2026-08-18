#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping


def _node_statuses(frame: Mapping[str, Any]) -> dict[str, str]:
    return {
        str(node["node_id"]): str(node["status"])
        for node in frame.get("nodes", ())
    }


def analyze_case(case: Mapping[str, Any], graph: Mapping[str, Any]) -> dict[str, Any]:
    trace = graph.get("state_trace")
    if not isinstance(trace, list) or not trace:
        raise ValueError("graph state_trace must be a nonempty list")

    transition_index = next(
        (
            index
            for index in range(1, len(trace))
            if trace[index - 1].get("certificate_state") == "CURRENT"
            and trace[index].get("certificate_state") == "STALE"
        ),
        None,
    )
    failed = not bool(case.get("base_success"))
    if transition_index is None:
        label = "STALL_WITHOUT_STALE" if failed else "NO_DEVIATION_CANDIDATE"
    else:
        label = (
            "PERSISTENT_FAILURE_CANDIDATE"
            if failed
            else "TRANSIENT_OR_SELF_RECOVERED"
        )

    first_stale = trace[transition_index] if transition_index is not None else None
    before = trace[transition_index - 1] if transition_index is not None else None
    action_nodes = {
        str(node["node_id"])
        for node in graph.get("nodes", ())
        if node.get("kind") == "ACTION"
    }
    progress_after_stale = False
    node_changes: list[dict[str, str]] = []
    if first_stale is not None and before is not None:
        before_statuses = _node_statuses(before)
        stale_statuses = _node_statuses(first_stale)
        node_changes = [
            {
                "node_id": node_id,
                "before": before_statuses.get(node_id, "MISSING"),
                "after": stale_statuses.get(node_id, "MISSING"),
            }
            for node_id in sorted(set(before_statuses) | set(stale_statuses))
            if before_statuses.get(node_id) != stale_statuses.get(node_id)
        ]
        completed_at_stale = {
            node_id
            for node_id, status in stale_statuses.items()
            if node_id in action_nodes and status == "COMPLETED"
        }
        progress_after_stale = any(
            any(
                node_id in action_nodes
                and status == "COMPLETED"
                and node_id not in completed_at_stale
                for node_id, status in _node_statuses(frame).items()
            )
            or _node_statuses(frame).get("GOAL") == "COMPLETED"
            for frame in trace[transition_index + 1 :]
        )

    first_stale_step = (
        int(first_stale["policy_step"]) if first_stale is not None else None
    )
    base_steps = int(case.get("base_steps", 0))
    requires_evidence = transition_index is not None or failed
    return {
        **dict(case),
        "retrospective_label": label,
        "first_stale_policy_step": first_stale_step,
        "first_stale_phase": (
            str(first_stale.get("phase", "POLICY"))
            if first_stale is not None
            else None
        ),
        "policy_steps_remaining": (
            max(0, base_steps - first_stale_step)
            if first_stale_step is not None
            else None
        ),
        "node_changes_at_stale": node_changes,
        "progress_after_stale": progress_after_stale,
        "confirmed_deviation": False,
        "requires_action_evidence": requires_evidence,
    }


def build_report(summary: Mapping[str, Any], *, trace_root: Path) -> dict[str, Any]:
    cases = summary.get("cases")
    if not isinstance(cases, list):
        raise ValueError("summary cases must be a list")
    analyzed = []
    for case in cases:
        graph_path = case.get("graph_path")
        if not isinstance(graph_path, str) or not graph_path:
            raise ValueError("case graph_path must be nonempty")
        graph = json.loads((trace_root / graph_path).read_text(encoding="utf-8"))
        analyzed.append(analyze_case(case, graph))

    relevant = [
        case
        for case in analyzed
        if case["retrospective_label"] != "NO_DEVIATION_CANDIDATE"
    ]
    labels = Counter(case["retrospective_label"] for case in relevant)
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "mode": "offline_topology_replay",
            "physical_intervention": False,
            "confirmation_rule": (
                "Topology-only evidence can nominate candidates but cannot confirm takeover."
            ),
        },
        "summary": {
            "source_cases": len(analyzed),
            "review_cases": len(relevant),
            "label_counts": dict(sorted(labels.items())),
            "confirmed_deviations": 0,
            "requires_action_evidence": sum(
                bool(case["requires_action_evidence"]) for case in relevant
            ),
        },
        "cases": relevant,
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Shadow confirmed-deviation offline replay",
        "",
        "本报告只重放拓扑状态，不执行动作；拓扑候选不能单独升级为主动接管。",
        "",
        f"- 来源：{summary['source_cases']} cases",
        f"- 需要复核：{summary['review_cases']} cases",
        f"- Confirmed deviation：{summary['confirmed_deviations']}",
        "",
        "| Case | Task | Seed | Base | 首次 STALE | 剩余步数 | 后续拓扑进展 | 回溯标签 |",
        "|---|---:|---:|---|---:|---:|---|---|",
    ]
    for case in report["cases"]:
        lines.append(
            "| {case_id} | {task_id} | {master_seed} | {base} | {stale} | "
            "{remaining} | {progress} | {label} |".format(
                case_id=case.get("case_id"),
                task_id=case.get("task_id"),
                master_seed=case.get("master_seed"),
                base="成功" if case.get("base_success") else "失败",
                stale=case.get("first_stale_policy_step"),
                remaining=case.get("policy_steps_remaining"),
                progress="是" if case.get("progress_after_stale") else "否",
                label=case.get("retrospective_label"),
            )
        )
    lines.extend(
        [
            "",
            "## 接管约束",
            "",
            "`CURRENT → STALE`、长期无拓扑进展以及 `PRECONDITION_UNKNOWN` 均只能产生候选。",
            "真正的 `CONFIRMED_DEVIATION` 还必须具有动作归因后的强证据，例如 "
            "`ATTEMPTED_EFFECT_TIMEOUT`、`ABNORMAL_TRANSFER_AFTER_MANIPULATION` 或 "
            "`GOAL_REGRESSION`。",
            "",
        ]
    )
    return "\n".join(lines)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay Shadow topology traces without physical intervention"
    )
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--trace-root", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", required=True, type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    report = build_report(summary, trace_root=args.trace_root)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.output_md.write_text(render_markdown(report), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
