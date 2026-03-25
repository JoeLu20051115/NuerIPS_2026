#!/usr/bin/env python3
"""
LLM planner with structured output + VAL validation + light correction.

Each plan step has:
  - instruction: natural-language short phrase
  - action: fixed action primitive
  - args: list of argument strings
  - time: [start_fraction, end_fraction]  both in [0, 1]

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
  "time"         : [start_fraction, end_fraction] in [0.0, 1.0], non-overlapping

Rules:
- time[0] of step 0 must be 0.0; time[1] of last step must be 1.0
- steps must cover [0,1] contiguously (end of step i == start of step i+1)
- grasp must come before any place/release on that object
- open must come before close on the same container
- reflect natural timing (approach phases are shorter than manipulation phases)

Output a JSON array only:
[
  {{"instruction": "...", "action": "move", "args": ["button"], "time": [0.0, 0.2]}},
  ...
]
"""

_CORRECTION_PROMPT_TEMPLATE = """\
You generated this robot plan for the task: "{task}"

Current plan:
{plan_json}

The validator found these logical errors and suggests the following fixes:
{actionable_hints}

Domain rules reminder:
- grasp(obj)         requires gripper OPEN  → after grasp gripper is CLOSED
- place(obj, target) requires HOLDING obj   → after place gripper is OPEN (released automatically)
- release(obj)       requires HOLDING obj   → after release gripper is OPEN
- open(container)    requires container CLOSED
- close(container)   requires container OPEN

Please apply ONLY the suggested fixes above (keep all other steps unchanged).
Output the corrected full plan as a JSON array in the same format.
Output JSON only.
"""


# ---------------------------------------------------------------------------
# VAL error → actionable hint translator
# ---------------------------------------------------------------------------

# Maps (failed_action_pat, unsatisfied_predicate_keyword) → human-readable fix instruction
# pred_keyword is a plain substring match against the predicate string (no backrefs)
_ERROR_HINT_MAP = [
    # release after place: holding is false because place already released
    (r"release\s+(\S+)",   "holding",
     lambda m, pred: f"Remove the 'release({m.group(1)})' step — "
                     f"'place' already opens the gripper, so 'release' afterward is redundant and invalid."),
    # grasp when gripper already closed
    (r"grasp\s+(\S+)",     "gripper",
     lambda m, pred: f"Cannot 'grasp({m.group(1)})' — gripper is already closed (holding another object). "
                     f"Add a 'release' or 'place' step before this grasp to open the gripper first."),
    # place without holding the object
    (r"place\s+(\S+)\s+(\S+)", "holding",
     lambda m, pred: f"Cannot 'place({m.group(1)}, {m.group(2)})' — not holding '{m.group(1)}'. "
                     f"Add a 'grasp({m.group(1)})' step before this place."),
    # open on an already-open container
    (r"open\s+(\S+)",      "drawer",
     lambda m, pred: f"Cannot 'open({m.group(1)})' — container is already open. "
                     f"Remove this 'open' step (the container starts open for this task)."),
    # close on an already-closed container (drawer-open not satisfied)
    (r"close\s+(\S+)",     "drawer",
     lambda m, pred: f"Cannot 'close({m.group(1)})' — container is not open. "
                     f"Add an 'open({m.group(1)})' step before this close."),
]


def _translate_val_errors(val_errors: str, steps: List[Dict]) -> str:
    """
    Convert raw VAL verbose output into actionable fix instructions for the LLM.
    Falls back to a cleaned version of the original error if no pattern matches.
    """
    hints: List[str] = []

    # Extract the failing action and the unsatisfied predicate from VAL output
    # VAL -v format:
    #   "Plan failed because of unsatisfied precondition in:\n(action args)"
    #   "(action args) has an unsatisfied precondition at time N"
    #   "(Set (predicate args) to true)"

    lines = val_errors.splitlines()
    failed_actions: List[Tuple[str, str]] = []  # (action_str, predicate_str)

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        # Capture "Plan failed because of unsatisfied precondition in:"
        if "plan failed because of unsatisfied precondition" in line.lower():
            # Next non-empty line is the action
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines):
                action_line = lines[j].strip().lstrip("(").rstrip(")")
                # Look for "Set (predicate...) to true" a few lines ahead
                pred_str = ""
                for k in range(j + 1, min(j + 6, len(lines))):
                    set_m = re.search(r"Set\s+\(([^)]+)\)\s+to\s+true", lines[k], re.I)
                    if set_m:
                        pred_str = set_m.group(1).strip()
                        break
                failed_actions.append((action_line, pred_str))
        i += 1

    for action_str, pred_str in failed_actions:
        matched = False
        for act_pat, pred_kw, hint_fn in _ERROR_HINT_MAP:
            act_m = re.search(act_pat, action_str, re.I)
            if act_m and pred_kw.lower() in pred_str.lower():
                hints.append(f"• {hint_fn(act_m, pred_str)}")
                matched = True
                break
        if not matched:
            # Generic fallback: strip confusing "Set X to true" advice, just name the failing step
            clean_pred = pred_str.replace("_", " ") if pred_str else "unknown condition"
            hints.append(
                f"• Step '({action_str})' failed because '{clean_pred}' was not satisfied. "
                f"Check the step order and preconditions listed in the domain rules above."
            )

    if not hints:
        # Last resort: strip Set-advice lines and return cleaned errors
        cleaned = "\n".join(
            l for l in lines
            if l.strip() and "Set (" not in l and "Failed plans" not in l
        )
        return cleaned or val_errors

    return "\n".join(hints)


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

        time_seg = s.get("time", None)
        if not isinstance(time_seg, list) or len(time_seg) < 2:
            time_seg = [i / max(len(steps), 1), (i + 1) / max(len(steps), 1)]
        time_seg = [float(max(0.0, min(1.0, t))) for t in time_seg[:2]]

        cleaned.append({
            "instruction": str(s.get("instruction", action)),
            "action": action,
            "args": args,
            "time": time_seg,
        })

    if not cleaned:
        return _fallback_plan(task)

    # Fix time: force start[0]=0.0, end[-1]=1.0, strictly non-overlapping
    cleaned[0]["time"][0] = 0.0
    cleaned[-1]["time"][1] = 1.0
    for i in range(1, len(cleaned)):
        cleaned[i]["time"][0] = cleaned[i - 1]["time"][1]
        if cleaned[i]["time"][1] <= cleaned[i]["time"][0]:
            cleaned[i]["time"][1] = min(1.0, cleaned[i]["time"][0] + 0.1)
    cleaned[-1]["time"][1] = 1.0

    return cleaned


def _fallback_plan(task: str) -> List[Dict]:
    return [{
        "instruction": task,
        "action": "move",
        "args": ["target"],
        "time": [0.0, 1.0],
    }]


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

        if "skipped" in val_errors.lower():
            val_result = "val_skipped"
        elif val_ok:
            val_result = "ok"
        else:
            # Step 3: Light correction
            for _ in range(self.max_correction_rounds):
                plan_json = json.dumps(steps, indent=2)
                actionable_hints = _translate_val_errors(val_errors, steps)
                correction_prompt = _CORRECTION_PROMPT_TEMPLATE.format(
                    task=task,
                    plan_json=plan_json,
                    actionable_hints=actionable_hints,
                )
                corrected_raw = self._llm(correction_prompt)
                corrected_steps = _parse_plan_json(corrected_raw, task)
                val_ok2, val_errors2 = run_val(corrected_steps, task, self.validate_path)
                steps = corrected_steps
                corrected = True
                if val_ok2:
                    val_result = "corrected_valid"
                    break
                val_errors = val_errors2
                val_result = "corrected"

        plan_time = time.perf_counter() - t0
        return {
            "steps": steps,
            "sub_instructions": [s["instruction"] for s in steps],
            "start_fractions": [s["time"][0] for s in steps],
            "val_result": val_result,
            "val_errors": val_errors if not val_ok else None,
            "corrected": corrected,
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
