#!/usr/bin/env python3
"""
Offline planner comparison on DROID 300 episodes.

Compares 3 methods (no policy inference required):
  1. llm_raw       — LLM one-shot, no validation or correction
  2. llm_self      — LLM generates, then LLM reviews and fixes its own plan (blind)
  3. llm_val       — LLM generates, VAL validates, LLM corrects with VAL error hints

Logged per-episode:
  - task, episode_id
  - for each method: steps with {action, args, time}, VAL validity, plan_time
  - summary metrics at end

Usage:
  python scripts/eval/compare_planners_droid300.py [--num-episodes 300] [--log logs/planner_compare_droid300.log]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from llm_planner_val import (
    _PLAN_PROMPT_TEMPLATE,
    _CORRECTION_PROMPT_TEMPLATE,
    _parse_plan_json,
    _translate_val_errors,
    run_val,
    VALID_ACTIONS,
)

from openai import OpenAI

_CLIENT = OpenAI()
_MODEL  = "gpt-4o-mini"
_SYSTEM = "You are a robot task planning assistant. Output ONLY the requested JSON format—no extra text."

_SELF_CORRECT_PROMPT = """\
You generated this robot plan for the task: "{task}"

Plan:
{plan_json}

Review the plan for logical errors using these rules:
- grasp(obj)         requires the gripper to be OPEN
- place(obj, target) requires HOLDING obj  →  after place the gripper is OPEN (no need for release)
- release(obj)       requires HOLDING obj
- open(container)    requires the container to be CLOSED
- close(container)   requires the container to be OPEN

Fix any errors you find. If the plan is already correct, return it unchanged.
Output the full plan as a JSON array in the same format. Output JSON only.
"""


def _llm(prompt: str) -> str:
    r = _CLIENT.chat.completions.create(
        model=_MODEL,
        temperature=0.0,
        max_tokens=600,
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user",   "content": prompt},
        ],
    )
    return r.choices[0].message.content or ""


def plan_llm_raw(task: str) -> dict:
    """Method 1: LLM one-shot, no correction."""
    t0 = time.perf_counter()
    raw = _llm(_PLAN_PROMPT_TEMPLATE.format(task=task))
    steps = _parse_plan_json(raw, task)
    val_ok, val_errors = run_val(steps, task)
    return {
        "steps": steps,
        "sub_instructions": [s["instruction"] for s in steps],
        "start_fractions":  [round(s["time"][0], 3) for s in steps],
        "durations":        [round(s["time"][1] - s["time"][0], 3) for s in steps],
        "val_valid": val_ok,
        "val_errors": val_errors if not val_ok else None,
        "corrected": False,
        "plan_time": round(time.perf_counter() - t0, 3),
        "llm_calls": 1,
    }


def plan_llm_self(task: str) -> dict:
    """Method 2: LLM generates, then reviews and self-corrects (blind, no VAL feedback)."""
    t0 = time.perf_counter()
    raw1 = _llm(_PLAN_PROMPT_TEMPLATE.format(task=task))
    steps1 = _parse_plan_json(raw1, task)

    # Self-review round
    self_prompt = _SELF_CORRECT_PROMPT.format(
        task=task,
        plan_json=json.dumps(steps1, indent=2),
    )
    raw2 = _llm(self_prompt)
    steps2 = _parse_plan_json(raw2, task)

    val_ok, val_errors = run_val(steps2, task)
    changed = (
        [s["action"] for s in steps1] != [s["action"] for s in steps2]
        or [s["args"] for s in steps1] != [s["args"] for s in steps2]
    )
    return {
        "steps": steps2,
        "sub_instructions": [s["instruction"] for s in steps2],
        "start_fractions":  [round(s["time"][0], 3) for s in steps2],
        "durations":        [round(s["time"][1] - s["time"][0], 3) for s in steps2],
        "val_valid": val_ok,
        "val_errors": val_errors if not val_ok else None,
        "corrected": changed,
        "plan_time": round(time.perf_counter() - t0, 3),
        "llm_calls": 2,
        "steps_before_self_correct": steps1,
    }


def plan_llm_val(task: str) -> dict:
    """Method 3: LLM generates, VAL validates, LLM corrects with precise hints."""
    t0 = time.perf_counter()
    raw1 = _llm(_PLAN_PROMPT_TEMPLATE.format(task=task))
    steps1 = _parse_plan_json(raw1, task)
    val_ok, val_errors = run_val(steps1, task)

    if val_ok:
        return {
            "steps": steps1,
            "sub_instructions": [s["instruction"] for s in steps1],
            "start_fractions":  [round(s["time"][0], 3) for s in steps1],
            "durations":        [round(s["time"][1] - s["time"][0], 3) for s in steps1],
            "val_result": "ok",
            "val_errors": None,
            "corrected": False,
            "plan_time": round(time.perf_counter() - t0, 3),
            "llm_calls": 1,
        }

    hints = _translate_val_errors(val_errors, steps1)
    corr_prompt = _CORRECTION_PROMPT_TEMPLATE.format(
        task=task,
        plan_json=json.dumps(steps1, indent=2),
        actionable_hints=hints,
    )
    raw2 = _llm(corr_prompt)
    steps2 = _parse_plan_json(raw2, task)
    val_ok2, val_errors2 = run_val(steps2, task)

    return {
        "steps": steps2,
        "sub_instructions": [s["instruction"] for s in steps2],
        "start_fractions":  [round(s["time"][0], 3) for s in steps2],
        "durations":        [round(s["time"][1] - s["time"][0], 3) for s in steps2],
        "val_result": "corrected_valid" if val_ok2 else "corrected_still_invalid",
        "val_errors": val_errors2 if not val_ok2 else None,
        "corrected": True,
        "val_hints_given": hints,
        "steps_before_correction": steps1,
        "plan_time": round(time.perf_counter() - t0, 3),
        "llm_calls": 2,
    }


def _fmt_steps(steps: list[dict]) -> str:
    parts = []
    for s in steps:
        act  = s.get("action", "?")
        args = ", ".join(s.get("args", []))
        t    = s.get("time", [0, 0])
        dur  = round(t[1] - t[0], 2)
        instr = s.get("instruction", "")
        parts.append(f"    {act}({args})  [{t[0]:.2f}-{t[1]:.2f} | dur={dur:.2f}]  \"{instr}\"")
    return "\n".join(parts)


def load_episodes(meta_path: Path, n: int) -> list[dict]:
    eps = []
    with meta_path.open() as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            r = json.loads(line)
            eps.append({
                "episode_id": f"episode_{r['episode_index']:06d}",
                "task": r.get("tasks", [""])[0] if r.get("tasks") else "",
            })
    return eps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path,
                        default=Path("data/droid_easy400_dualfavored_dreamzero"))
    parser.add_argument("--num-episodes", type=int, default=300)
    parser.add_argument("--log",  type=Path,
                        default=Path("logs/planner_compare_droid300.log"))
    parser.add_argument("--out",  type=Path,
                        default=Path("evaluation_results_dualsystem/planner_compare_droid300.json"))
    args = parser.parse_args()

    args.log.parent.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    episodes = load_episodes(args.dataset_root / "meta/episodes.jsonl", args.num_episodes)
    all_results = []

    # counters  {method: {valid, invalid, corrected, still_invalid}}
    counters: dict[str, dict] = {
        m: {"valid": 0, "invalid": 0, "corrected_valid": 0,
            "corrected_still_invalid": 0, "self_changed": 0, "total": 0}
        for m in ("llm_raw", "llm_self", "llm_val")
    }

    def log(line: str):
        print(line, flush=True)
        with args.log.open("a") as f:
            f.write(line + "\n")

    # Clear log
    args.log.write_text("")
    log(f"{'='*80}")
    log(f"DROID {args.num_episodes} episodes — Planner comparison")
    log(f"Methods: llm_raw | llm_self | llm_val")
    log(f"{'='*80}\n")

    for idx, ep in enumerate(episodes, 1):
        task = ep["task"]
        eid  = ep["episode_id"]

        log(f"[{idx:3d}/{args.num_episodes}] {eid}  |  {task}")

        r_raw  = plan_llm_raw(task)
        r_self = plan_llm_self(task)
        r_val  = plan_llm_val(task)

        # --- log method 1 ---
        valid1 = "✓ VALID" if r_raw["val_valid"] else "✗ INVALID"
        log(f"  ── llm_raw   {valid1}  ({len(r_raw['steps'])} steps, {r_raw['plan_time']}s)")
        log(_fmt_steps(r_raw["steps"]))
        if not r_raw["val_valid"]:
            log(f"    VAL error: {(r_raw['val_errors'] or '').splitlines()[0]}")

        # --- log method 2 ---
        valid2   = "✓ VALID" if r_self["val_valid"] else "✗ INVALID"
        changed2 = " [self-changed]" if r_self["corrected"] else " [unchanged]"
        log(f"  ── llm_self  {valid2}  ({len(r_self['steps'])} steps){changed2}")
        log(_fmt_steps(r_self["steps"]))
        if not r_self["val_valid"]:
            log(f"    VAL error: {(r_self['val_errors'] or '').splitlines()[0]}")

        # --- log method 3 ---
        vr3   = r_val.get("val_result", "ok")
        log(f"  ── llm_val   [{vr3}]  ({len(r_val['steps'])} steps, corrected={r_val['corrected']})")
        log(_fmt_steps(r_val["steps"]))
        if r_val.get("val_hints_given"):
            log(f"    hints: {r_val['val_hints_given']}")

        log("")

        # update counters
        for method, r in [("llm_raw", r_raw), ("llm_self", r_self), ("llm_val", r_val)]:
            c = counters[method]
            c["total"] += 1
            if method == "llm_val":
                vr = r.get("val_result", "ok")
                if vr == "ok" or vr == "corrected_valid":
                    c["valid"] += 1
                else:
                    c["invalid"] += 1
                if r["corrected"]:
                    if vr == "corrected_valid":
                        c["corrected_valid"] += 1
                    else:
                        c["corrected_still_invalid"] += 1
            else:
                if r["val_valid"]:
                    c["valid"] += 1
                else:
                    c["invalid"] += 1
                if method == "llm_self" and r["corrected"]:
                    c["self_changed"] += 1

        # Running summary every 50 episodes
        if idx % 50 == 0:
            log(f"  {'─'*60}")
            log(f"  Running summary after {idx} episodes:")
            for method, c in counters.items():
                vr = c["valid"] / c["total"] * 100
                log(f"    {method:10s}: valid={c['valid']}/{c['total']} ({vr:.1f}%)")
            log(f"  {'─'*60}\n")

        all_results.append({
            "episode_id": eid,
            "task": task,
            "llm_raw":  r_raw,
            "llm_self": r_self,
            "llm_val":  r_val,
        })

        # Save incrementally every 10 episodes
        if idx % 10 == 0:
            args.out.write_text(json.dumps(all_results, indent=2, ensure_ascii=False))

    # Final summary
    log(f"\n{'='*80}")
    log(f"FINAL SUMMARY — {args.num_episodes} episodes")
    log(f"{'='*80}")
    for method, c in counters.items():
        vr = c["valid"] / c["total"] * 100
        inv = c["invalid"] / c["total"] * 100
        line = f"  {method:12s}: valid={c['valid']}/{c['total']} ({vr:.1f}%)  invalid={c['invalid']} ({inv:.1f}%)"
        if method == "llm_val":
            line += f"  corrected_valid={c['corrected_valid']}  still_invalid={c['corrected_still_invalid']}"
        elif method == "llm_self":
            line += f"  self_changed={c['self_changed']}"
        log(line)

    # Per-task breakdown
    log(f"\nPer-task valid rate (llm_raw / llm_self / llm_val):")
    task_stats: dict[str, dict] = defaultdict(
        lambda: {m: {"valid": 0, "total": 0} for m in ("llm_raw", "llm_self", "llm_val")}
    )
    for rec in all_results:
        t = rec["task"]
        task_stats[t]["llm_raw"]["total"]  += 1
        task_stats[t]["llm_self"]["total"] += 1
        task_stats[t]["llm_val"]["total"]  += 1
        if rec["llm_raw"]["val_valid"]:
            task_stats[t]["llm_raw"]["valid"]  += 1
        if rec["llm_self"]["val_valid"]:
            task_stats[t]["llm_self"]["valid"] += 1
        vr = rec["llm_val"].get("val_result", "ok")
        if vr in ("ok", "corrected_valid"):
            task_stats[t]["llm_val"]["valid"] += 1

    for task, ms in sorted(task_stats.items(), key=lambda x: -x[1]["llm_val"]["total"]):
        n = ms["llm_val"]["total"]
        if n < 2:
            continue
        r1 = ms["llm_raw"]["valid"]  / n * 100
        r2 = ms["llm_self"]["valid"] / n * 100
        r3 = ms["llm_val"]["valid"]  / n * 100
        log(f"  [{n:3d}ep] {task:<50} raw={r1:.0f}%  self={r2:.0f}%  val={r3:.0f}%")

    args.out.write_text(json.dumps(all_results, indent=2, ensure_ascii=False))
    log(f"\nResults saved → {args.out}")
    log(f"Log saved     → {args.log}")


if __name__ == "__main__":
    main()
