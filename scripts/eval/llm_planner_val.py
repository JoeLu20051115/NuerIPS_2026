#!/usr/bin/env python3
"""
LLM planner with structured output + VAL validation + light correction.

Each plan step has:
  - instruction: natural-language short phrase
  - action: fixed action primitive
  - args: list of argument strings
  - weight: relative time duration (positive integer); converted to time [start, end] by
            cumulative-sum normalisation over the step sequence

Flow:
  1. LLM one-shot generates structured JSON plan
  2. Convert to PDDL (domain + problem + plan)
  3. Run VAL (Validate) for logical validation
  4. If VAL finds errors → give errors back to LLM for minimal correction
  5. Return final plan
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Fixed PDDL domain for robot manipulation
# ---------------------------------------------------------------------------
PDDL_DOMAIN = """\
(define (domain robot-manipulation)
  (:requirements :strips :typing)
  (:types object)
  (:predicates
    (gripper-open)
    (holding ?o - object)
    (drawer-open ?o - object)
    (pressed ?o - object)
    (placed-in ?o1 - object ?o2 - object)
    (visited ?o - object)
  )

  ; Spatial/no-op actions: (and) = unconditional; effect marks visited (re-asserting is ok)
  (:action move
    :parameters (?target - object)
    :precondition (and)
    :effect (visited ?target)
  )
  (:action pull
    :parameters (?o - object)
    :precondition (and)
    :effect (visited ?o)
  )
  (:action push
    :parameters (?o - object)
    :precondition (and)
    :effect (visited ?o)
  )
  (:action lower
    :parameters (?o - object)
    :precondition (and)
    :effect (visited ?o)
  )

  ; Gripper actions: validated
  (:action grasp
    :parameters (?o - object)
    :precondition (gripper-open)
    :effect (and (holding ?o) (not (gripper-open)))
  )
  (:action place
    :parameters (?o - object ?target - object)
    :precondition (holding ?o)
    :effect (and (not (holding ?o)) (gripper-open) (placed-in ?o ?target))
  )
  (:action release
    :parameters (?o - object)
    :precondition (holding ?o)
    :effect (and (not (holding ?o)) (gripper-open))
  )
  (:action lift
    :parameters (?o - object)
    :precondition (holding ?o)
    :effect (visited ?o)
  )

  ; Container actions: validated
  (:action open
    :parameters (?o - object)
    :precondition (not (drawer-open ?o))
    :effect (drawer-open ?o)
  )
  (:action close
    :parameters (?o - object)
    :precondition (drawer-open ?o)
    :effect (not (drawer-open ?o))
  )

  ; Press: unconditional (button state can be reset)
  (:action press
    :parameters (?o - object)
    :precondition (and)
    :effect (pressed ?o)
  )
)
"""

VALID_ACTIONS = {
    "move", "grasp", "place", "release",
    "open", "close", "press", "pull", "push", "lift", "lower",
}

# Args count expected per action (min, max); -1 = any
ACTION_ARGS = {
    "move":    (1, 1),
    "grasp":   (1, 1),
    "place":   (2, 2),
    "release": (1, 1),
    "open":    (1, 1),
    "close":   (1, 1),
    "press":   (1, 1),
    "pull":    (1, 1),
    "push":    (1, 1),
    "lift":    (1, 1),
    "lower":   (1, 1),
}

# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------
_SYSTEM = "You are a robot task planning assistant. Output ONLY the requested JSON format—no extra text."

_PLAN_PROMPT_TEMPLATE = """\
Task: {task}

Robot: Franka arm, indoor manipulation scene.
Available action primitives:
  move(target)          — move gripper/arm to a target position or object
  grasp(object)         — close gripper to hold an object  (requires gripper open)
  place(object, target) — put held object at target        (requires holding object)
  release(object)       — open gripper to let go           (requires holding object)
  open(container)       — open a drawer or container       (requires it is closed)
  close(container)      — close a drawer or container      (requires it is open)
  press(target)         — press a button or target
  pull(object)          — pull an object toward the robot
  push(object)          — push an object away

Break the task into 3-6 atomic steps. For each step output:
  "instruction"  : short natural-language phrase (≤10 words)
  "action"       : one of the primitives above (exact name)
  "args"         : list of snake_case argument strings matching the primitive
  "weight"       : positive integer for relative duration (move≈1, grasp/place/press≈2-3)
  "pre"          : list of state facts that must hold BEFORE this step
                   (use strings like "gripper_open", "holding(marker)", "drawer_open(drawer)")
  "eff"          : list of state facts that become true AFTER this step
                   (prefix with "~" for facts that become false, e.g. "~gripper_open")

Rules:
- weights must be positive integers (1–5 range recommended)
- grasp requires "gripper_open"; its effects include "holding(obj)" and "~gripper_open"
- place requires "holding(obj)"; its effects include "~holding(obj)" and "gripper_open"
- open requires "~drawer_open(c)"; close requires "drawer_open(c)"
- effects of step i must satisfy preconditions of step i+1 where relevant

Output a JSON array only:
[
  {{"instruction": "move to the button", "action": "move",  "args": ["button"], "weight": 1,
    "pre": [], "eff": []}},
  {{"instruction": "press the button",   "action": "press", "args": ["button"], "weight": 3,
    "pre": [], "eff": ["pressed(button)"]}},
  ...
]
"""

# ---------------------------------------------------------------------------
# Repair operator system
# ---------------------------------------------------------------------------

# Operator types
REPAIR_DELETE      = "delete"        # Remove a step — deterministic, no LLM
REPAIR_INSERT      = "insert"        # Insert one new step — LLM generates the single step
REPAIR_REORDER     = "reorder"       # Move a step to a different index — LLM selects target index
REPAIR_MODIFY_ARGS = "modify_args"   # Fix args of one step — LLM fixes one field only


@dataclass
class RepairOp:
    op:         str             # one of REPAIR_* constants
    target_idx: int             # index of the problematic step in current plan
    insert_pos: int  = -1       # for INSERT: position to insert before (-1 = before target_idx)
    context:    str  = ""       # human-readable description of why this repair is needed
    new_args:   List[str] = field(default_factory=list)   # for MODIFY_ARGS: candidate args


# --- VAL error parsing ---

def _extract_failed_actions(val_errors: str) -> List[Tuple[str, str]]:
    """
    Parse VAL -v output → list of (action_str, predicate_str) for each failure.

    VAL format:
      Plan failed because of unsatisfied precondition
        in:
      (action arg1 arg2 ...)
      ...
      Set (predicate) to true
    """
    lines = val_errors.splitlines()
    results: List[Tuple[str, str]] = []
    i = 0
    while i < len(lines):
        if "plan failed because of unsatisfied precondition" in lines[i].lower():
            # Skip whitespace / "in:" label lines to find the actual action line
            j = i + 1
            while j < len(lines) and (
                not lines[j].strip() or lines[j].strip().rstrip(":").lower() == "in"
            ):
                j += 1
            if j < len(lines):
                action_line = lines[j].strip().lstrip("(").rstrip(")")
                pred_str = ""
                for k in range(j + 1, min(j + 8, len(lines))):
                    m = re.search(r"Set\s+\(([^)]+)\)\s+to\s+true", lines[k], re.I)
                    if m:
                        pred_str = m.group(1).strip()
                        break
                results.append((action_line, pred_str))
        i += 1
    return results


def _find_step_idx(steps: List[Dict], action_str: str) -> int:
    """Find the first step whose action+args match the VAL failure action string."""
    parts = action_str.strip().split()
    if not parts:
        return -1
    act = parts[0].lower()
    args = [p.lower() for p in parts[1:]]
    for i, s in enumerate(steps):
        if s.get("action", "") == act:
            step_args = [_sanitize(a) for a in s.get("args", [])]
            if not args or step_args[:len(args)] == args:
                return i
    # fallback: match action name only
    for i, s in enumerate(steps):
        if s.get("action", "") == act:
            return i
    return len(steps) - 1


def _simulate_graph_state(steps: List[Dict]) -> Dict[int, set]:
    """
    Forward-simulate the task graph using each node's 'eff' field.
    Returns a dict: step_index → set of true predicates BEFORE that step executes.
    This lets repair operators reason about graph state at any node.
    """
    # Seed: initial world state
    state: set = {"gripper-open"}
    snapshots: Dict[int, set] = {0: set(state)}

    for i, step in enumerate(steps):
        for eff in step.get("eff", []):
            e = eff.strip().lower()
            if e.startswith("~"):
                state.discard(e[1:].strip())
            else:
                state.add(e)
        snapshots[i + 1] = set(state)

    return snapshots


def _classify_repair_ops(val_errors: str, steps: List[Dict]) -> List[RepairOp]:
    """
    Graph-aware repair operator classification.

    For each VAL failure, we:
      1. Identify the failing graph node (step index)
      2. Simulate the graph state up to that node using pre/eff fields
      3. Identify which precondition of that node is unsatisfied
      4. Choose the minimal repair that restores graph consistency:

    Graph state violation      → RepairOp
    ──────────────────────────────────────────────────────────────────
    Node has 'holding X' in pre but state has no holding X
      and node is release(X)   → DELETE  (no predecessor gives holding X → node unreachable)
    Node has 'holding X' in pre but state has no holding X
      and node is place(X)     → INSERT  grasp(X) before this node
    Node has 'gripper-open' in pre but state has gripper closed
      and node is grasp(X)     → INSERT  release(held) before this node
    Node has 'drawer-open X' in pre but state has no drawer-open X
      and node is close(X)     → INSERT  open(X) before this node
    Node has '~drawer-open X' in pre but state already has drawer-open X
      and node is open(X)      → DELETE  (duplicate open)
    fallback                   → INSERT  with graph-state context
    """
    failed = _extract_failed_actions(val_errors)
    if not failed:
        return []

    # Simulate graph state to get state-before-each-node
    state_at = _simulate_graph_state(steps)
    ops: List[RepairOp] = []

    for action_str, pred_str in failed:
        parts   = action_str.strip().split()
        act     = parts[0].lower() if parts else ""
        args    = parts[1:] if len(parts) > 1 else []
        obj     = args[0] if args else "?"
        pred_lo = pred_str.lower()

        idx = _find_step_idx(steps, action_str)
        state_before = state_at.get(idx, set())

        # ── Case 1: release(X) but graph state has no 'holding X' ──────────
        # Node is logically unreachable — no predecessor produces 'holding X'
        if act == "release" and "holding" in pred_lo:
            ops.append(RepairOp(
                op=REPAIR_DELETE, target_idx=idx,
                context=(
                    f"Graph node release({obj}): precondition 'holding {obj}' "
                    f"is not in graph state at node {idx} "
                    f"(state={sorted(state_before)}). "
                    f"No predecessor produces this effect — node is unreachable. DELETE."
                )
            ))

        # ── Case 2: open(X) but graph state already has 'drawer-open X' ────
        elif act == "open" and "drawer" in pred_lo:
            ops.append(RepairOp(
                op=REPAIR_DELETE, target_idx=idx,
                context=(
                    f"Graph node open({obj}): graph state at node {idx} already "
                    f"satisfies 'drawer-open {obj}'. Duplicate node — DELETE."
                )
            ))

        # ── Case 3: place(X) but graph state has no 'holding X' ─────────────
        # Missing predecessor node: insert grasp(X) to satisfy the dependency edge
        elif act == "place" and "holding" in pred_lo:
            ops.append(RepairOp(
                op=REPAIR_INSERT, target_idx=idx, insert_pos=idx,
                context=(
                    f"Graph node place({obj}): requires 'holding {obj}' but "
                    f"graph state at node {idx} has no such predicate "
                    f"(state={sorted(state_before)}). "
                    f"INSERT grasp({obj}) node before this node to close the dependency gap."
                )
            ))

        # ── Case 4: close(X) but graph state has no 'drawer-open X' ─────────
        elif act == "close" and "drawer" in pred_lo:
            ops.append(RepairOp(
                op=REPAIR_INSERT, target_idx=idx, insert_pos=idx,
                context=(
                    f"Graph node close({obj}): requires 'drawer-open {obj}' but "
                    f"graph state at node {idx} has no such predicate. "
                    f"INSERT open({obj}) node before this node."
                )
            ))

        # ── Case 5: grasp(X) but graph state has gripper closed ──────────────
        # Identify the currently held object from graph state, insert release
        elif act == "grasp" and "gripper" in pred_lo:
            held = _infer_currently_held(steps, idx) or "object"
            ops.append(RepairOp(
                op=REPAIR_INSERT, target_idx=idx, insert_pos=idx,
                context=(
                    f"Graph node grasp({obj}): requires 'gripper-open' but "
                    f"graph state at node {idx} has gripper holding '{held}' "
                    f"(state={sorted(state_before)}). "
                    f"INSERT release({held}) node before this node to restore gripper-open."
                )
            ))

        else:
            # Generic fallback with full graph state context
            ops.append(RepairOp(
                op=REPAIR_INSERT, target_idx=idx, insert_pos=idx,
                context=(
                    f"Graph node ({action_str}): precondition '{pred_str}' not in "
                    f"graph state at node {idx} (state={sorted(state_before)}). "
                    f"Insert or reorder predecessor nodes to satisfy this dependency."
                )
            ))

    return ops


# --- Targeted repair prompts (LLM operates in restricted scope) ---

_INSERT_PROMPT = """\
Robot task: "{task}"

The plan has a logical error at step {insert_pos} (0-indexed):
{context}

Existing step at that position:
{target_step}

Generate ONLY the single missing step to insert BEFORE the step above.
Output a single JSON object (not an array):
{{"instruction": "...", "action": "<primitive>", "args": [...], "weight": <int>, "pre": [...], "eff": [...]}}

Available actions: move, grasp, place, release, open, close, press, pull, push
Output JSON only."""

_MODIFY_ARGS_PROMPT = """\
Robot task: "{task}"

Step {idx} has wrong arguments:
{step_json}

{context}

Fix ONLY the "args" field of this step. Keep instruction, action, weight, pre, eff unchanged.
Output the corrected single JSON object only."""


def _infer_currently_held(steps: List[Dict], before_idx: int) -> Optional[str]:
    """Scan backward through steps[0:before_idx] to find the currently held object."""
    held = None
    for i in range(before_idx):
        act  = steps[i].get("action", "").lower()
        args = steps[i].get("args", [])
        obj  = args[0] if args else None
        if act == "grasp" and obj:
            held = obj
        elif act in ("place", "release"):
            held = None
    return held


def _deterministic_insert(op: RepairOp, steps: List[Dict]) -> Optional[Dict]:
    """
    Generate an insert step deterministically for known error patterns — no LLM call.

    Known patterns:
      place(X) without holding X  → insert grasp(X) before place
      close(X) without open       → insert open(X) before close
      grasp(X) while gripper closed → insert release(held) before grasp
    Returns None for unknown patterns (falls through to LLM).
    """
    idx = op.target_idx
    if idx < 0 or idx >= len(steps):
        return None
    failing = steps[idx]
    act  = failing.get("action", "").lower()
    args = failing.get("args", [])
    obj  = args[0] if args else "object"

    if act == "place":
        # Must grasp obj before placing it
        return {
            "instruction": f"grasp the {obj.replace('_', ' ')}",
            "action": "grasp",
            "args": [obj],
            "weight": 1,
            "pre": ["gripper-open"],
            "eff": [f"holding {obj}", "~gripper-open"],
        }

    if act == "close":
        # Must open container before closing
        return {
            "instruction": f"open the {obj.replace('_', ' ')}",
            "action": "open",
            "args": [obj],
            "weight": 1,
            "pre": [f"~drawer-open {obj}"],
            "eff": [f"drawer-open {obj}"],
        }

    if act == "grasp":
        # Gripper is closed — release whatever is currently held
        held = _infer_currently_held(steps, idx) or "object"
        return {
            "instruction": f"release the {held.replace('_', ' ')}",
            "action": "release",
            "args": [held],
            "weight": 1,
            "pre": [f"holding {held}"],
            "eff": [f"~holding {held}", "gripper-open"],
        }

    return None  # unknown pattern — fall through to LLM


def _apply_repair_ops(
    steps: List[Dict], ops: List[RepairOp], task: str, llm_fn
) -> List[Dict]:
    """
    Apply repair operators to the step list.
    - DELETE:       pure deletion, no LLM call.
    - INSERT:       deterministic for known patterns; LLM fallback for unknown.
    - MODIFY_ARGS:  LLM fixes only the args field of one step.
    - REORDER:      moves step to a new position, no LLM call.
    """
    result = list(steps)  # work on a copy

    # Process ops in reverse index order so earlier inserts don't shift later indices
    for op in sorted(ops, key=lambda o: o.target_idx, reverse=True):
        idx = op.target_idx
        if idx < 0 or idx >= len(result):
            continue

        if op.op == REPAIR_DELETE:
            result.pop(idx)

        elif op.op == REPAIR_INSERT:
            pos = op.insert_pos if op.insert_pos >= 0 else idx
            pos = max(0, min(pos, len(result)))

            # Try deterministic repair first (no LLM)
            new_step = _deterministic_insert(op, result)
            if new_step is None:
                # Fallback: ask LLM for the missing step
                target_step = json.dumps(result[idx], ensure_ascii=False) if idx < len(result) else "{}"
                prompt = _INSERT_PROMPT.format(
                    task=task,
                    insert_pos=pos,
                    context=op.context,
                    target_step=target_step,
                )
                raw = llm_fn(prompt)
                new_step = _parse_single_step_json(raw)
            if new_step:
                result.insert(pos, new_step)

        elif op.op == REPAIR_MODIFY_ARGS:
            prompt = _MODIFY_ARGS_PROMPT.format(
                task=task,
                idx=idx,
                step_json=json.dumps(result[idx], ensure_ascii=False),
                context=op.context,
            )
            raw = llm_fn(prompt)
            fixed = _parse_single_step_json(raw)
            if fixed and fixed.get("action") == result[idx].get("action"):
                result[idx]["args"] = fixed.get("args", result[idx]["args"])

        elif op.op == REPAIR_REORDER:
            # Move step at target_idx to insert_pos
            if op.insert_pos >= 0:
                step = result.pop(idx)
                new_pos = max(0, min(op.insert_pos, len(result)))
                result.insert(new_pos, step)

    return result


def _parse_single_step_json(raw: str) -> Optional[Dict]:
    """Parse a single step JSON object from LLM output."""
    raw = re.sub(r"```[a-z]*\n?", "", raw).strip().strip("`")
    # Try to extract a {...} object
    m = re.search(r"\{[^{}]+\}", raw, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict) or "action" not in obj:
        return None
    action = str(obj.get("action", "move")).lower().strip()
    if action not in VALID_ACTIONS:
        action = "move"
    args = obj.get("args", [])
    if isinstance(args, str):
        args = [args]
    args = [str(a) for a in args]
    lo, hi = ACTION_ARGS.get(action, (1, 2))
    while len(args) < lo:
        args.append("target")
    if hi > 0:
        args = args[:hi]
    try:
        weight = max(1, int(float(obj.get("weight", 1))))
    except (TypeError, ValueError):
        weight = 1
    return {
        "instruction": str(obj.get("instruction", action)),
        "action":      action,
        "args":        args,
        "weight":      weight,
        "pre":         list(obj.get("pre", [])),
        "eff":         list(obj.get("eff", [])),
    }


# ---------------------------------------------------------------------------
# PDDL helpers
# ---------------------------------------------------------------------------

def _sanitize(name: str) -> str:
    """Make a string safe for PDDL identifiers."""
    s = re.sub(r"[^a-zA-Z0-9_]", "_", name.strip())
    s = re.sub(r"_+", "_", s).strip("_").lower()
    if not s:
        s = "obj"
    if s[0].isdigit():
        s = "o_" + s
    return s


def _collect_objects(steps: List[Dict]) -> List[str]:
    """Return unique PDDL object names from all args in the plan."""
    seen = set()
    order = []
    for step in steps:
        for arg in step.get("args", []):
            name = _sanitize(arg)
            if name not in seen:
                seen.add(name)
                order.append(name)
    return order


def _infer_init(steps: List[Dict], objects: List[str]) -> List[str]:
    """
    Infer initial state by tracking the FIRST open/close action per container.

    Rules:
    - First action on container is 'close'  → container starts OPEN   → add (drawer-open obj)
    - First action on container is 'open'   → container starts CLOSED  → omit (drawer-open obj)
    - Only 'close' appears (no 'open')      → container starts OPEN   → add (drawer-open obj)
    - Only 'open' appears (no 'close')      → container starts CLOSED  → omit

    This prevents false positives for plans like open→...→close where the
    container starts closed and both actions are legitimate.
    """
    facts = ["(gripper-open)"]
    # Map each container object → first open/close action seen in plan order
    first_oc: Dict[str, str] = {}
    for step in steps:
        action = step.get("action", "")
        if action in ("open", "close") and step.get("args"):
            obj = _sanitize(step["args"][0])
            if obj not in first_oc:
                first_oc[obj] = action

    for obj, first_action in first_oc.items():
        if first_action == "close":
            # Container must start open so the close precondition is met
            facts.append(f"(drawer-open {obj})")
        # if first_action == "open": starts closed — (not (drawer-open obj)) is default, nothing to add

    return facts


def _infer_goal(steps: List[Dict]) -> str:
    """Infer goal from the last meaningful action."""
    for step in reversed(steps):
        action = step.get("action", "")
        args = [_sanitize(a) for a in step.get("args", [])]
        if action == "place" and len(args) >= 2:
            return f"(placed-in {args[0]} {args[1]})"
        if action == "close" and args:
            return f"(not (drawer-open {args[0]}))"
        if action == "open" and args:
            return f"(drawer-open {args[0]})"
        if action == "press" and args:
            return f"(pressed {args[0]})"
        if action in ("release",) and args:
            return "(gripper-open)"
        if action == "grasp" and args:
            return f"(holding {args[0]})"
    return "(gripper-open)"


def _make_pddl_action_line(step: Dict) -> str:
    """Produce one STRIPS plan line: (action arg1 arg2 ...)"""
    action = step.get("action", "move")
    args = [_sanitize(a) for a in step.get("args", [])]
    args_str = (" " + " ".join(args)) if args else ""
    return f"({action}{args_str})"


def _build_pddl_problem(task: str, steps: List[Dict]) -> str:
    objects = _collect_objects(steps)
    if not objects:
        objects = ["dummy_obj"]
    obj_str = " ".join(objects) + " - object"
    init_facts = _infer_init(steps, objects)
    init_str = " ".join(init_facts)
    goal_str = _infer_goal(steps)
    return (
        f"(define (problem manipulation-task)\n"
        f"  (:domain robot-manipulation)\n"
        f"  (:objects {obj_str})\n"
        f"  (:init {init_str})\n"
        f"  (:goal {goal_str})\n"
        f")\n"
    )


def _build_pddl_plan(steps: List[Dict]) -> str:
    return "\n".join(_make_pddl_action_line(s) for s in steps) + "\n"


# ---------------------------------------------------------------------------
# VAL runner
# ---------------------------------------------------------------------------

def run_val(steps: List[Dict], task: str, validate_path: str = "Validate") -> Tuple[bool, str]:
    """
    Run VAL on the plan. Returns (is_valid, error_text).
    Uses -v (verbose) to get per-action precondition details.
    If VAL cannot run, returns (True, "") to skip validation gracefully.
    """
    domain_str = PDDL_DOMAIN
    problem_str = _build_pddl_problem(task, steps)
    plan_str = _build_pddl_plan(steps)

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            d = Path(tmpdir) / "domain.pddl"
            p = Path(tmpdir) / "problem.pddl"
            pl = Path(tmpdir) / "plan.pddl"
            d.write_text(domain_str)
            p.write_text(problem_str)
            pl.write_text(plan_str)

            result = subprocess.run(
                [validate_path, "-v", str(d), str(p), str(pl)],
                capture_output=True,
                text=True,
                timeout=10,
            )
            output = result.stdout + result.stderr
            is_valid = "Plan valid" in output or "executed successfully" in output

            if is_valid:
                return True, ""

            # Extract actionable error lines from verbose output
            errors = []
            capture = False
            for line in output.splitlines():
                low = line.strip().lower()
                # Trigger capture on key failure lines
                if "plan failed because" in low or "unsatisfied precondition" in low:
                    capture = True
                if capture and line.strip():
                    errors.append(line.strip())
                # Also grab Plan Repair Advice section
                if "plan repair advice" in low:
                    capture = True
            # Fallback: grab any error-like lines
            if not errors:
                for line in output.splitlines():
                    low = line.lower()
                    if any(kw in low for kw in ("failed", "precondition", "unsatisfied", "invalid")):
                        errors.append(line.strip())
            error_text = "\n".join(errors) if errors else "Plan failed to execute"
            return False, error_text
    except Exception as exc:
        return True, f"VAL skipped: {exc}"


# ---------------------------------------------------------------------------
# JSON plan parser
# ---------------------------------------------------------------------------

def _parse_plan_json(raw: str, task: str) -> List[Dict]:
    """Parse LLM output into list of step dicts. Returns best-effort result."""
    # Strip markdown code fences
    raw = re.sub(r"```[a-z]*\n?", "", raw).strip()
    raw = raw.strip("`").strip()

    # Find the JSON array
    start = raw.find("[")
    end = raw.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return _fallback_plan(task)

    try:
        steps = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return _fallback_plan(task)

    if not isinstance(steps, list) or not steps:
        return _fallback_plan(task)

    cleaned = []
    for i, s in enumerate(steps):
        if not isinstance(s, dict):
            continue
        action = str(s.get("action", "move")).lower().strip()
        if action not in VALID_ACTIONS:
            action = "move"
        args = s.get("args", [])
        if isinstance(args, str):
            args = [args]
        args = [str(a) for a in args]

        # Enforce args count
        lo, hi = ACTION_ARGS.get(action, (1, 2))
        while len(args) < lo:
            args.append("target")
        if hi > 0:
            args = args[:hi]

        # Parse weight (positive int); fall back to 1 if missing/invalid
        raw_weight = s.get("weight", None)
        # Also accept old "time" field from legacy outputs for backwards compat
        if raw_weight is None and isinstance(s.get("time"), list):
            dur = s["time"][1] - s["time"][0] if len(s["time"]) >= 2 else 0.2
            raw_weight = max(1, round(dur * 10))
        try:
            weight = max(1, int(float(raw_weight)))
        except (TypeError, ValueError):
            weight = 1

        pre = s.get("pre", [])
        eff = s.get("eff", [])
        if not isinstance(pre, list): pre = []
        if not isinstance(eff, list): eff = []

        cleaned.append({
            "instruction": str(s.get("instruction", action)),
            "action":      action,
            "args":        args,
            "weight":      weight,
            "pre":         [str(p) for p in pre],
            "eff":         [str(e) for e in eff],
        })

    if not cleaned:
        return _fallback_plan(task)

    # Convert weights → time intervals via cumulative-sum normalisation
    total = sum(c["weight"] for c in cleaned)
    cursor = 0.0
    for c in cleaned:
        start = cursor
        end   = cursor + c["weight"] / total
        c["time"] = [round(start, 4), round(end, 4)]
        cursor = end
    cleaned[-1]["time"][1] = 1.0  # guarantee exact 1.0

    return cleaned


def _fallback_plan(task: str) -> List[Dict]:
    return [{
        "instruction": task,
        "action": "move",
        "args": ["target"],
        "weight": 1,
        "pre": [], "eff": [],
        "time": [0.0, 1.0],
    }]


def _recompute_time(steps: List[Dict]) -> List[Dict]:
    """Re-run weight → time normalisation after structural edits."""
    if not steps:
        return steps
    total = sum(max(1, s.get("weight", 1)) for s in steps)
    cursor = 0.0
    for s in steps:
        w = max(1, s.get("weight", 1))
        s["time"] = [round(cursor, 4), round(cursor + w / total, 4)]
        cursor += w / total
    steps[-1]["time"][1] = 1.0
    return steps


# ---------------------------------------------------------------------------
# Main planner class
# ---------------------------------------------------------------------------

class LLMPlannerWithVAL:
    """
    Structured LLM planner with PDDL/VAL validation and light correction.

    Compatible with existing LLMPlanner interface:
      plan(task, context) -> (sub_instructions, meta)

    Extended interface:
      plan_structured(task) -> full structured result dict
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        validate_path: str = "Validate",
        max_correction_rounds: int = 1,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.validate_path = validate_path
        self.max_correction_rounds = max_correction_rounds

        from openai import OpenAI
        self.client = OpenAI()

    def _llm(self, user_prompt: str, max_tokens: int = 600) -> str:
        # Sanitize prompt: remove non-UTF8 and control chars that break JSON body
        user_prompt = user_prompt.encode("utf-8", errors="replace").decode("utf-8")
        user_prompt = "".join(c for c in user_prompt if c >= " " or c in "\n\t")
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": _SYSTEM},
                    {"role": "user", "content": user_prompt},
                ],
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            print(f"[LLM error] {type(e).__name__}: {str(e)[:120]}", flush=True)
            return ""

    def plan_structured(self, task: str) -> Dict:
        """
        Full structured planning with VAL validation.

        Returns:
          steps            : list of {instruction, action, args, time}
          sub_instructions : list[str]  (for downstream compatibility)
          start_fractions  : list[float]
          val_result       : "ok" | "corrected" | "val_failed" | "val_skipped"
          val_errors       : str | None
          corrected        : bool
          raw_response     : str
          plan_time        : float
        """
        t0 = time.perf_counter()
        errors_log: List[str] = []

        # Step 1: Generate candidate plan
        prompt = _PLAN_PROMPT_TEMPLATE.format(task=task)
        raw = self._llm(prompt)
        steps = _parse_plan_json(raw, task)

        # Step 2: PDDL validation
        val_ok, val_errors = run_val(steps, task, self.validate_path)
        corrected = False

        repair_ops_log: List[Dict] = []

        if "skipped" in val_errors.lower():
            val_result = "val_skipped"
        elif val_ok:
            val_result = "ok"
        else:
            # Step 3: Graph-structured repair via finite operator set
            for _ in range(self.max_correction_rounds):
                ops = _classify_repair_ops(val_errors, steps)
                # Log the repair operators chosen
                repair_ops_log.append({
                    "ops": [{"op": o.op, "target_idx": o.target_idx,
                             "insert_pos": o.insert_pos, "context": o.context}
                            for o in ops]
                })
                if not ops:
                    val_result = "corrected"
                    corrected = True
                    break
                repaired_steps = _apply_repair_ops(steps, ops, task, self._llm)
                # Re-normalise weights → time after structural repair
                repaired_steps = _recompute_time(repaired_steps)
                val_ok2, val_errors2 = run_val(repaired_steps, task, self.validate_path)
                # Safety: only commit repair if it doesn't make things worse
                if val_ok2 or (not val_ok):
                    steps = repaired_steps
                    corrected = True
                    if val_ok2:
                        val_result = "corrected_valid"
                        break
                    val_errors = val_errors2
                    val_result = "corrected"
                else:
                    # Repair made a valid plan invalid — rollback and stop
                    val_result = "ok"
                    break

        plan_time = time.perf_counter() - t0
        return {
            "steps": steps,
            "sub_instructions": [s["instruction"] for s in steps],
            "start_fractions": [s["time"][0] for s in steps],
            "val_result": val_result,
            "val_errors": val_errors if not val_ok else None,
            "corrected": corrected,
            "repair_ops": repair_ops_log,
            "raw_response": raw,
            "plan_time": plan_time,
        }

    # ------------------------------------------------------------------
    # Compatibility shim: old LLMPlanner interface
    # ------------------------------------------------------------------
    def plan(self, task: str, context: Optional[Dict] = None) -> Tuple[List[str], Dict]:
        """Drop-in replacement for LLMPlanner.plan()"""
        result = self.plan_structured(task)
        meta = {
            "planner_mode": "val_validated" if not result["corrected"] else "val_corrected",
            "api_called": True,
            "api_success": True,
            "error": None,
            "model": self.model,
            "raw_response": result["raw_response"],
            "val_result": result["val_result"],
            "val_errors": result["val_errors"],
            "steps": result["steps"],
            "start_fractions": result["start_fractions"],
        }
        return result["sub_instructions"], meta
