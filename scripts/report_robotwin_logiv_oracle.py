#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Iterable


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _frozen_cells(config: dict[str, Any]) -> dict[tuple[str, int], str]:
    cells: dict[tuple[str, int], str] = {}
    for task, seeds in config["tasks"].items():
        if len(seeds) != 10 or len(set(map(int, seeds))) != 10:
            raise ValueError(f"{task} must contain 10 distinct seeds")
        instructions = config["instructions"][task]
        if len(instructions) != len(seeds):
            raise ValueError(f"instruction count does not match seeds for {task}")
        cells.update(
            ((task, int(seed)), str(instructions[index]))
            for index, seed in enumerate(seeds)
        )
    return cells


def _records(roots: Iterable[Path]) -> Iterable[dict[str, Any]]:
    for root in roots:
        root = root.resolve()
        for path in sorted(root.rglob("logiv_events.jsonl")):
            with path.open(encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, 1):
                    if line.strip():
                        record = json.loads(line)
                        record["source_record_sha256"] = hashlib.sha256(
                            line.strip().encode()
                        ).hexdigest()
                        record["source_event_file"] = str(
                            path.resolve().relative_to(root)
                        )
                        record["source_line"] = line_number
                        yield record


def _is_compliant_success(record: dict[str, Any]) -> bool:
    events = record.get("events")
    return bool(
        record.get("success") is True
        and record.get("original_instruction")
        and int(record.get("gpt4o_requests", 0)) > 0
        and isinstance(events, list)
        and events
        and events[0].get("epoch") == 0
        and all(
            event.get("val_valid") is True
            and len(str(event.get("certificate_sha256", ""))) == 64
            and event.get("control_mode")
            in {"BASE_MONITORED", "DAG_EXECUTION", "REPAIR"}
            for event in events
        )
    )


def build_oracle_report(
    config: dict[str, Any],
    event_roots: Iterable[Path],
    *,
    embed_records: bool = False,
    require_baseline_instruction: bool = False,
) -> dict[str, Any]:
    cells = _frozen_cells(config)
    candidates: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for record in _records(event_roots):
        key = (str(record.get("task")), int(record.get("seed", -1)))
        instruction_matches = record.get("original_instruction") == cells.get(key)
        if (
            key in cells
            and _is_compliant_success(record)
            and (instruction_matches or not require_baseline_instruction)
        ):
            candidates[key].append(record)

    selected = []
    missing = []
    per_task: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"episodes": 0, "successes": 0, "successful_seeds": []}
    )
    for (task, seed), _baseline_instruction in sorted(cells.items()):
        per_task[task]["episodes"] += 1
        if not candidates[(task, seed)]:
            missing.append({"task": task, "seed": seed})
            continue
        record = min(
            candidates[(task, seed)],
            key=lambda item: (
                int(item.get("actions", 10**9)),
                item["source_event_file"],
                item["source_line"],
            ),
        )
        per_task[task]["successes"] += 1
        per_task[task]["successful_seeds"].append(seed)
        selection = {
            "task": task,
            "seed": seed,
            "instruction": record["original_instruction"],
            "actions": record.get("actions"),
            "gpt4o_requests": record.get("gpt4o_requests"),
            "source_event_file": record["source_event_file"],
            "source_line": record["source_line"],
            "source_record_sha256": record["source_record_sha256"],
        }
        if embed_records:
            selection["record"] = {
                key: value
                for key, value in record.items()
                if not key.startswith("source_")
            }
            selection["embedded_record_sha256"] = _canonical_sha256(
                selection["record"]
            )
        selected.append(selection)
    successes = len(selected)
    baseline_instruction_matches = sum(
        selection["instruction"] == cells[(selection["task"], selection["seed"])]
        for selection in selected
    )
    source_runs: dict[str, int] = defaultdict(int)
    for selection in selected:
        parts = Path(selection["source_event_file"]).parts
        try:
            run_name = parts[parts.index("eval_result") + 1]
        except (ValueError, IndexError):
            run_name = selection["source_event_file"]
        source_runs[run_name] += 1
    return {
        "evidence_label": (
            "per-seed/config development oracle with frozen baseline instructions "
            "(not a single config)"
            if require_baseline_instruction
            else "per-seed/prompt development oracle (not a single config)"
        ),
        "selection_rule": (
            "one historical monitored success per frozen (task, seed), "
            + (
                "requiring the exact frozen baseline instruction; "
                if require_baseline_instruction
                else "including historical prompt choice; "
            )
            + "minimum action count breaks ties"
        ),
        "require_baseline_instruction": require_baseline_instruction,
        "instruction_scope": (
            "original episode instruction only; PDDL-selected policy dispatch "
            "prompts may differ and are not present in historical event records"
        ),
        "baseline_instruction_matches": baseline_instruction_matches,
        "protocol": {
            "canonical_config": config,
            "canonical_config_sha256": _canonical_sha256(config),
            "tasks": config["tasks"],
            "baseline_instructions": config["instructions"],
        },
        "expected": len(cells),
        "successes": successes,
        "success_rate": successes / len(cells) if cells else None,
        "per_task": dict(per_task),
        "source_runs": dict(sorted(source_runs.items())),
        "selected": selected,
        "missing": missing,
    }


def _stable_val_stdout(stdout: str) -> str:
    return re.sub(
        r"/tmp/logiv-robotwin-val-[^/\s]+/",
        "",
        stdout,
    )


def revalidate_embedded_report(
    report: dict[str, Any], val_binary: Path
) -> dict[str, Any]:
    """Regenerate embedded fact-state plans and run VAL again.

    Historical records did not retain their PDDL text.  This audit therefore
    preserves their original certificate separately and reconstructs a linear
    registered DAG from each record's ordered fact universe.  It is a fresh
    validity check, not a claim that the new digest equals the historical one.
    """
    source_root = Path(__file__).resolve().parents[1] / "src"
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    from pi05_libero_repro.logiv.robotwin import (
        RobotwinPddlPlanner,
        RobotwinStage,
        RobotwinTask,
        TruthValue,
    )

    planner = RobotwinPddlPlanner(val_binary, timeout_seconds=5.0)
    states: dict[str, dict[str, Any]] = {}
    event_occurrences = 0
    valid_event_occurrences = 0
    for selection in report.get("selected", []):
        task_name = str(selection["task"])
        for event in selection.get("record", {}).get("events", []):
            event_occurrences += 1
            raw_facts = event.get("facts", {})
            facts = {
                str(name): "UNRESOLVED" if str(value) == "UNKNOWN" else str(value)
                for name, value in raw_facts.items()
            }
            state_id = _canonical_sha256({"task": task_name, "facts": facts})
            if state_id not in states:
                task = RobotwinTask(
                    task_name,
                    tuple(
                        RobotwinStage(name, "revalidation", "revalidation")
                        for name in facts
                    ),
                )
                plan = planner.plan(
                    task,
                    {name: TruthValue(value) for name, value in facts.items()},
                )
                states[state_id] = {
                    "state_sha256": state_id,
                    "task": task_name,
                    "facts": facts,
                    "occurrences": 0,
                    "val_valid": plan.valid,
                    "certificate_sha256": plan.certificate_sha256,
                    "domain_pddl": plan.domain_pddl,
                    "problem_pddl": plan.problem_pddl,
                    "plan_pddl": plan.plan_pddl,
                    "val_stdout": _stable_val_stdout(plan.val_stdout),
                    "val_stderr": _stable_val_stdout(plan.val_stderr),
                }
            states[state_id]["occurrences"] += 1
            if states[state_id]["val_valid"]:
                valid_event_occurrences += 1
    return {
        "method": (
            "fresh VAL run after regenerating a bounded linear PDDL problem "
            "from each embedded ordered fact universe and truth assignment"
        ),
        "historical_certificate_match_expected": False,
        "val_binary": str(val_binary),
        "val_binary_sha256": hashlib.sha256(val_binary.read_bytes()).hexdigest(),
        "event_occurrences": event_occurrences,
        "valid_event_occurrences": valid_event_occurrences,
        "unique_states": len(states),
        "states": sorted(states.values(), key=lambda row: row["state_sha256"]),
    }


def audit_embedded_report(report: dict[str, Any]) -> list[str]:
    errors = []
    baseline_instructions = report.get("protocol", {}).get(
        "baseline_instructions", {}
    )
    tasks = report.get("protocol", {}).get("tasks", {})
    canonical_config = report.get("protocol", {}).get("canonical_config")
    if canonical_config is not None and report.get("protocol", {}).get(
        "canonical_config_sha256"
    ) != _canonical_sha256(canonical_config):
        errors.append("canonical protocol config hash mismatch")
    frozen_instructions = {
        (task, seed): baseline_instructions.get(task, [])[index]
        for task, seeds in tasks.items()
        for index, seed in enumerate(seeds)
        if index < len(baseline_instructions.get(task, []))
    }
    keys = [
        (selection.get("task"), selection.get("seed"))
        for selection in report.get("selected", [])
    ]
    if len(keys) != len(set(keys)):
        errors.append("duplicate selected cell")
    for selection in report.get("selected", []):
        record = selection.get("record")
        key = (selection.get("task"), selection.get("seed"))
        if not isinstance(record, dict) or not _is_compliant_success(record):
            errors.append(f"selected record is not compliant: {key}")
        elif (record.get("task"), record.get("seed")) != key:
            errors.append(f"selected record key mismatch: {key}")
        elif selection.get("embedded_record_sha256") != _canonical_sha256(record):
            errors.append(f"embedded record hash mismatch: {key}")
        elif any(
            selection.get(field) != record.get(record_field)
            for field, record_field in (
                ("instruction", "original_instruction"),
                ("actions", "actions"),
                ("gpt4o_requests", "gpt4o_requests"),
            )
        ):
            errors.append(f"selection metadata mismatch: {key}")
        elif (
            report.get("require_baseline_instruction") is True
            and record.get("original_instruction") != frozen_instructions.get(key)
        ):
            errors.append(f"baseline instruction mismatch: {key}")
    if len(report.get("selected", [])) != report.get("successes"):
        errors.append("success count does not match selected records")
    expected = int(report.get("expected", 0))
    expected_rate = len(report.get("selected", [])) / expected if expected else None
    if report.get("success_rate") != expected_rate:
        errors.append("success rate does not match selected records")
    selected_keys = set(keys)
    missing_keys = {
        (row.get("task"), row.get("seed")) for row in report.get("missing", [])
    }
    if selected_keys & missing_keys:
        errors.append("cell appears in both selected and missing")
    if selected_keys | missing_keys != set(frozen_instructions):
        errors.append("selected and missing cells do not cover the protocol")
    actual_baseline_matches = sum(
        selection.get("instruction") == frozen_instructions.get(key)
        for selection, key in zip(report.get("selected", []), keys)
    )
    if actual_baseline_matches != report.get("baseline_instruction_matches"):
        errors.append("baseline instruction match count mismatch")
    revalidation = report.get("val_revalidation")
    if revalidation is not None:
        event_count = sum(
            len(selection.get("record", {}).get("events", []))
            for selection in report.get("selected", [])
        )
        state_occurrences = sum(
            int(state.get("occurrences", 0))
            for state in revalidation.get("states", [])
        )
        state_evidence_valid = True
        for state in revalidation.get("states", []):
            expected_state_id = _canonical_sha256(
                {"task": state.get("task"), "facts": state.get("facts")}
            )
            expected_certificate = _canonical_sha256(
                {
                    "domain": state.get("domain_pddl"),
                    "problem": state.get("problem_pddl"),
                    "plan": state.get("plan_pddl"),
                    "val_binary_sha256": revalidation.get("val_binary_sha256"),
                    "val_stdout": state.get("val_stdout"),
                    "valid": state.get("val_valid"),
                }
            )
            state_evidence_valid &= (
                state.get("state_sha256") == expected_state_id
                and state.get("certificate_sha256") == expected_certificate
            )
        if (
            revalidation.get("event_occurrences") != event_count
            or revalidation.get("valid_event_occurrences") != event_count
            or state_occurrences != event_count
            or revalidation.get("unique_states")
            != len(revalidation.get("states", []))
            or not all(
                state.get("val_valid") is True
                and "Plan valid" in state.get("val_stdout", "")
                for state in revalidation.get("states", [])
            )
            or not state_evidence_valid
        ):
            errors.append("VAL revalidation count mismatch")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--events-root", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--embed-records", action="store_true")
    parser.add_argument("--require-baseline-instruction", action="store_true")
    parser.add_argument("--revalidate-val-binary", type=Path)
    parser.add_argument("--checkpoint-file", type=Path)
    parser.add_argument("--runtime-commit")
    args = parser.parse_args()
    report = build_oracle_report(
        json.loads(args.config.read_text()),
        args.events_root,
        embed_records=args.embed_records,
        require_baseline_instruction=args.require_baseline_instruction,
    )
    source_root = Path(__file__).resolve().parents[1]
    report["artifact_provenance"] = {
        "protocol_config_file": {
            "path": str(args.config.resolve()),
            "sha256": _file_sha256(args.config.resolve()),
        },
        "report_generator_sha256": _file_sha256(Path(__file__).resolve()),
        "planner_source_sha256": _file_sha256(
            source_root / "src/pi05_libero_repro/logiv/robotwin.py"
        ),
    }
    if args.revalidate_val_binary is not None:
        if not args.embed_records:
            parser.error("--revalidate-val-binary requires --embed-records")
        report["val_revalidation"] = revalidate_embedded_report(
            report, args.revalidate_val_binary.resolve()
        )
    if args.checkpoint_file is not None:
        checkpoint_file = args.checkpoint_file.resolve()
        report["artifact_provenance"]["checkpoint_file"] = {
            "path": str(checkpoint_file),
            "sha256": _file_sha256(checkpoint_file),
        }
    if args.runtime_commit:
        report["artifact_provenance"]["robotwin_runtime_commit"] = (
            args.runtime_commit
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
