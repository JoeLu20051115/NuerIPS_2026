#!/usr/bin/env python3
"""
3-way comparison on Ego4D GoalStep 1000 samples:
  llm_raw  : LLM one-shot, no correction
  llm_self : LLM self-review (second call to review own plan)
  llm_val  : VAL graph-structured repair operators

Score per sample = 0.5 * val_validity + 0.5 * gt_similarity
  val_validity  : 1.0 if VAL passes, else 0.0
  gt_similarity : token-F1 recall of GT step descriptions vs generated instructions
                  (for each GT step, find best-matching generated instruction)

Outputs:
  logs/ego4d_compare.log          real-time log
  evaluation_results_dualsystem/ego4d_compare_1000.json  full results
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent))
from llm_planner_val import (
    LLMPlannerWithVAL,
    run_val,
    _parse_plan_json,
    _classify_repair_ops,
    _apply_repair_ops,
    _recompute_time,
    _PLAN_PROMPT_TEMPLATE,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_PATH   = "data/ego4d_goalstep_1000/goalstep_1000.json"
OUT_JSON    = "evaluation_results_dualsystem/ego4d_compare_1000.json"
LOG_PATH    = "logs/ego4d_compare.log"
SAVE_EVERY  = 20
MAX_SAMPLES = 1000

# ---------------------------------------------------------------------------
# Self-review prompt
# ---------------------------------------------------------------------------
_SELF_REVIEW_PROMPT = """\
You generated this robot plan for the task: "{task}"

{plan_json}

Review your plan for logical errors:
- grasp(obj) requires gripper OPEN; after grasp gripper is CLOSED
- place(obj, target) requires HOLDING obj; after place gripper is OPEN
- release(obj) requires HOLDING obj
- open(container) requires it is CLOSED
- close(container) requires it is OPEN
- Do NOT add 'release' after 'place' — place already releases

If the plan is correct, return it unchanged.
If there are errors, fix ONLY the problematic steps.
Output the plan as a JSON array with the same format (weight fields, not time).
Output JSON only."""


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z]+", text.lower())


def _token_f1(pred: str, ref: str) -> float:
    p_toks = set(_tokenize(pred))
    r_toks = set(_tokenize(ref))
    if not p_toks or not r_toks:
        return 0.0
    inter = len(p_toks & r_toks)
    prec  = inter / len(p_toks)
    rec   = inter / len(r_toks)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def gt_similarity(gen_steps: List[Dict], gt_steps: List[Dict]) -> float:
    """
    For each GT step, find best-matching generated instruction (token-F1).
    Return mean over GT steps.
    """
    if not gen_steps or not gt_steps:
        return 0.0
    gen_instrs = [s.get("instruction", "") for s in gen_steps]
    scores = []
    for gt in gt_steps:
        ref = gt.get("step_description", "")
        best = max(_token_f1(instr, ref) for instr in gen_instrs)
        scores.append(best)
    return sum(scores) / len(scores)


def combined_score(
    steps: List[Dict], task: str, gt_steps: List[Dict]
) -> Tuple[float, float, float]:
    """Returns (combined, val_score, gt_score)."""
    val_ok, _ = run_val(steps, task)
    v = 1.0 if val_ok else 0.0
    g = gt_similarity(steps, gt_steps)
    return 0.5 * v + 0.5 * g, v, g


# ---------------------------------------------------------------------------
# Planning methods
# ---------------------------------------------------------------------------

def plan_llm_raw(task: str, planner: LLMPlannerWithVAL) -> Tuple[List[Dict], bool, str]:
    raw = planner._llm(_PLAN_PROMPT_TEMPLATE.format(task=task))
    steps = _parse_plan_json(raw, task)
    val_ok, val_err = run_val(steps, task, planner.validate_path)
    return steps, val_ok, val_err


def plan_llm_self(task: str, planner: LLMPlannerWithVAL) -> Tuple[List[Dict], bool, str]:
    raw1 = planner._llm(_PLAN_PROMPT_TEMPLATE.format(task=task))
    steps1 = _parse_plan_json(raw1, task)
    plan_json = json.dumps(
        [{"instruction": s["instruction"], "action": s["action"],
          "args": s["args"], "weight": s["weight"]} for s in steps1],
        indent=2
    )
    review_prompt = _SELF_REVIEW_PROMPT.format(task=task, plan_json=plan_json)
    raw2 = planner._llm(review_prompt)
    steps2 = _parse_plan_json(raw2, task)
    val_ok, val_err = run_val(steps2, task, planner.validate_path)
    return steps2, val_ok, val_err


def plan_llm_val(task: str, planner: LLMPlannerWithVAL,
                 initial_steps: List[Dict] = None) -> Tuple[List[Dict], bool, str, bool, List]:
    # Reuse raw steps if provided — avoids regression from independent re-generation
    if initial_steps is not None:
        steps = [s.copy() for s in initial_steps]
    else:
        raw = planner._llm(_PLAN_PROMPT_TEMPLATE.format(task=task))
        steps = _parse_plan_json(raw, task)
    val_ok, val_err = run_val(steps, task, planner.validate_path)
    corrected = False
    ops_log = []
    if not val_ok:
        ops = _classify_repair_ops(val_err, steps)
        ops_log = [{"op": o.op, "target_idx": o.target_idx, "context": o.context[:80]}
                   for o in ops]
        repaired = _apply_repair_ops(steps, ops, task, planner._llm)
        repaired = _recompute_time(repaired)
        val_ok2, val_err2 = run_val(repaired, task, planner.validate_path)
        # Safety rollback: only commit if repair actually helped
        if val_ok2 or (not val_ok):
            steps = repaired
            val_ok = val_ok2
            val_err = val_err2
        corrected = True
    return steps, val_ok, val_err, corrected, ops_log


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs("logs", exist_ok=True)
    os.makedirs("evaluation_results_dualsystem", exist_ok=True)

    samples = json.load(open(DATA_PATH))["samples"][:MAX_SAMPLES]
    planner = LLMPlannerWithVAL()

    # ── Resume from existing results ─────────────────────────────────────
    done_ids: set = set()
    results: list = []
    if os.path.exists(OUT_JSON):
        try:
            existing = json.load(open(OUT_JSON))
            old_results = existing.get("results", existing) if isinstance(existing, dict) else existing
            for r in old_results:
                done_ids.add(r["id"])
                results.append(r)
            print(f"[resume] loaded {len(results)} existing results, skipping them.")
        except Exception as e:
            print(f"[resume] could not load existing results: {e}")

    running   = defaultdict(lambda: {"score": 0.0, "val": 0.0, "gt": 0.0, "n": 0})
    # Seed running stats from existing results
    for r in results:
        for m in ("llm_raw", "llm_self", "llm_val"):
            running[m]["score"] += r[m]["score"]
            running[m]["val"]   += r[m]["val_ok"]
            running[m]["gt"]    += r[m]["gt_score"]
            running[m]["n"]     += 1
    log_lines = []

    def log(msg: str):
        print(msg, flush=True)
        log_lines.append(msg)
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    # Append to log (don't clear)
    log(f"\n[resume] Ego4D GoalStep 3-way comparison  ({len(samples)} samples, {len(done_ids)} already done)")
    log("=" * 70)

    for ep_i, sample in enumerate(samples):
        task     = sample["goal"]
        gt_steps = sample["steps"]
        sid      = sample["id"]
        t0       = time.perf_counter()

        # Skip already-done samples
        if sid in done_ids:
            continue

        log(f"\n[{ep_i+1:4d}/{len(samples)}] {sid}")
        log(f"  task: {task}")
        log(f"  GT steps ({len(gt_steps)}): " +
            " | ".join(s["step_description"][:25] for s in gt_steps[:4]))

        try:
            # ── llm_raw ──────────────────────────────────────────────
            t1 = time.perf_counter()
            steps_raw, raw_val_ok, raw_err = plan_llm_raw(task, planner)
            sc_raw, v_raw, g_raw = combined_score(steps_raw, task, gt_steps)
            t_raw = time.perf_counter() - t1
            log(f"  llm_raw  {'✓' if raw_val_ok else '✗'}  "
                f"val={v_raw:.2f} gt={g_raw:.2f} score={sc_raw:.3f}  "
                f"({len(steps_raw)} steps, {t_raw:.1f}s)")
            log(f"    instrs: " + " → ".join(s["instruction"] for s in steps_raw))

            # ── llm_self ─────────────────────────────────────────────
            t1 = time.perf_counter()
            steps_self, self_val_ok, self_err = plan_llm_self(task, planner)
            sc_self, v_self, g_self = combined_score(steps_self, task, gt_steps)
            t_self = time.perf_counter() - t1
            log(f"  llm_self {'✓' if self_val_ok else '✗'}  "
                f"val={v_self:.2f} gt={g_self:.2f} score={sc_self:.3f}  "
                f"({len(steps_self)} steps, {t_self:.1f}s)")
            log(f"    instrs: " + " → ".join(s["instruction"] for s in steps_self))

            # ── llm_val ──────────────────────────────────────────────
            t1 = time.perf_counter()
            steps_val, val_val_ok, val_err2, corrected, ops_log = plan_llm_val(task, planner, initial_steps=steps_raw)
            sc_val, v_val, g_val = combined_score(steps_val, task, gt_steps)
            t_val = time.perf_counter() - t1
            repair_tag = f" [corrected:{'+'.join(o['op'] for o in ops_log)}]" if corrected else ""
            log(f"  llm_val  {'✓' if val_val_ok else '✗'}  "
                f"val={v_val:.2f} gt={g_val:.2f} score={sc_val:.3f}  "
                f"({len(steps_val)} steps, {t_val:.1f}s){repair_tag}")
            log(f"    instrs: " + " → ".join(s["instruction"] for s in steps_val))
        except Exception as e:
            log(f"  [SKIP] error on sample {sid}: {type(e).__name__}: {str(e)[:100]}")
            continue

        # ── update running averages ───────────────────────────────
        for method, sc, v, g in [
            ("raw",  sc_raw,  v_raw,  g_raw),
            ("self", sc_self, v_self, g_self),
            ("val",  sc_val,  v_val,  g_val),
        ]:
            running[method]["score"] += sc
            running[method]["val"]   += v
            running[method]["gt"]    += g
            running[method]["n"]     += 1

        # ── running summary every 20 ─────────────────────────────
        if (ep_i + 1) % 20 == 0:
            n = ep_i + 1
            log(f"\n  ── Running avg after {n} samples ──────────────────────")
            for m in ("raw", "self", "val"):
                r = running[m]
                log(f"     llm_{m:4s}: score={r['score']/r['n']:.3f}  "
                    f"val={r['val']/r['n']:.3f}  gt={r['gt']/r['n']:.3f}")
            log(f"  ────────────────────────────────────────────────────────")

        # ── record ───────────────────────────────────────────────
        results.append({
            "id":       sid,
            "task":     task,
            "category": sample.get("goal_category", ""),
            "gt_steps": [s["step_description"] for s in gt_steps],
            "timing":   sample.get("timing", ""),
            "llm_raw": {
                "steps":    steps_raw,
                "val_ok":   raw_val_ok,
                "val_score": v_raw,
                "gt_score":  g_raw,
                "score":     sc_raw,
            },
            "llm_self": {
                "steps":    steps_self,
                "val_ok":   self_val_ok,
                "val_score": v_self,
                "gt_score":  g_self,
                "score":     sc_self,
            },
            "llm_val": {
                "steps":    steps_val,
                "val_ok":   val_val_ok,
                "val_score": v_val,
                "gt_score":  g_val,
                "score":     sc_val,
                "corrected": corrected,
                "repair_ops": ops_log,
            },
        })

        # ── incremental save ─────────────────────────────────────
        if (ep_i + 1) % SAVE_EVERY == 0:
            _save(results, running, samples)

    # ── final summary ────────────────────────────────────────────
    N = len(results)
    log("\n" + "=" * 70)
    log(f"FINAL SUMMARY — {N} samples")
    log("=" * 70)
    log(f"{'Method':<12} {'Score':>8} {'VAL':>8} {'GT-sim':>8}")
    log(f"{'-'*40}")
    for m, label in [("raw","llm_raw"), ("self","llm_self"), ("val","llm_val")]:
        r = running[m]
        log(f"{label:<12} {r['score']/r['n']:>8.3f} {r['val']/r['n']:>8.3f} {r['gt']/r['n']:>8.3f}")

    _save(results, running, samples)
    log(f"\nResults → {OUT_JSON}")
    log(f"Log     → {LOG_PATH}")


def _save(results, running, samples):
    N = len(results)
    summary = {}
    for m in ("raw", "self", "val"):
        r = running[m]
        n = r["n"]
        summary[f"llm_{m}"] = {
            "score":     round(r["score"] / n, 4),
            "val_rate":  round(r["val"]   / n, 4),
            "gt_sim":    round(r["gt"]    / n, 4),
            "n":         n,
        }
    out = {
        "description": "Ego4D GoalStep 3-way comparison (llm_raw/llm_self/llm_val). "
                       "Score = 0.5*val_validity + 0.5*gt_token_f1_recall",
        "num_samples":  N,
        "summary":      summary,
        "results":      results,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
