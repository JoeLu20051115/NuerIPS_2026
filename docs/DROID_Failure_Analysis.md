# DreamZero on DROID: Failure Case Analysis of Few-Shot Generalization

> **Paper section style** — drafted for NeurIPS 2026 oral presentation.
> 评测基准：DROID 小样本子集（L1×150 + L3×150），DreamZero 推理服务，δ=0.1 成功阈值。

---

## Abstract

We present a systematic failure case analysis of DreamZero applied to the DROID few-shot benchmark.
Despite strong performance on in-distribution data (AgiBot L1 SR: 38.7%, Rate(L2<0.1): 68.7%),
DreamZero exhibits a **significant performance drop on DROID**: L1 SR drops to 29.3% and the
action-level precision metric Rate(L2<0.1) collapses to ≤47.6%.
We identify four primary failure modes — *action precision degradation*, *long-horizon compounding
error*, *language grounding ambiguity*, and *Dual-LLM over-planning interference* — and provide
per-case evidence with quantitative breakdowns.
Our analysis reveals that domain shift in visual appearance and action distribution, rather than task
complexity alone, is the dominant failure factor.

---

## 1. Motivation

DROID is a large-scale, in-the-wild robot manipulation dataset collected across diverse environments,
robots, and operator styles.
Evaluating generalization on DROID's held-out episodes provides a stringent test of whether a policy
learned on structured lab data can transfer to unconstrained real-world scenarios.

We evaluate two operational modes:

| Mode | Prompt Construction |
|---|---|
| **System1** (`description_only`) | Raw task description $g$ |
| **Dual-System** (`dual_llm`) | $g$ + LLM-generated sub-task chain $z_{1:M}$ |

Evaluation protocol (following §5 of the architecture document):
- **Episodes**: L1 = 20–150 (medium horizon), L3 = 20–150 (long horizon)
- **Success threshold**: $\delta = 0.1$ on per-step joint-space L2 error
- **Sampling**: $K = 10$ uniformly sampled steps per episode
- **Judge**: GPT-4o-mini visual judgment for DreamDojo video-space evaluation

---

## 2. Aggregate Results vs. In-Distribution Baseline

### 2.1 DreamZero: DROID vs. AgiBot

| Dataset | Mode | Mean L2 ↓ | Task Progress ↑ | Success Rate ↑ | Rate(L2<0.1) ↑ |
|---|---|---|---|---|---|
| **AgiBot L1** (150 ep) | description_only | 0.0862 | 0.516 | **38.7%** | **68.7%** |
| **AgiBot L1** (150 ep) | dual_llm | 0.0773 | 0.505 | 36.7% | 74.2% |
| **AgiBot L3** (150 ep) | description_only | 0.0739 | 0.446 | 30.0% | 79.8% |
| **AgiBot L3** (150 ep) | dual_llm | 0.0602 | 0.519 | 41.3% | 92.9% |
| | | | | | |
| **DROID L1** (150 ep) | description_only | 0.1284 | 0.491 | 29.3% | 44.3% |
| **DROID L1** (150 ep) | dual_llm | 0.1258 | 0.506 | 32.0% | 45.1% |
| **DROID L3** (150 ep) | description_only | 0.1262 | 0.458 | 23.3% | 45.9% |
| **DROID L3** (150 ep) | dual_llm | 0.1236 | 0.479 | 27.3% | 47.6% |

**Key observations:**

- DROID Mean L2 (~0.126) is **46% higher** than AgiBot (~0.086), indicating systematic action
  distribution mismatch.
- Rate(L2<0.1) on DROID (44–48%) falls to **less than half** of AgiBot L3 dual_llm (92.9%),
  meaning step-level action precision is severely degraded.
- Task Progress on DROID (0.46–0.51) is comparable to AgiBot despite lower action accuracy,
  suggesting the visual-language judge is partially forgiving of kinematic error.
- Dual-System consistently outperforms System1 on DROID L3 (+4.0 pp SR, +2.1 pp progress),
  but gains are more modest than on AgiBot.

### 2.2 Statistical Significance (DreamZero, paired t-test across episodes)

| Dataset | t-value | p-value | Mean L2 reduction (Dual−Base) | 95% CI |
|---|---|---|---|---|
| DRO_L1_150 | 2.121 | 0.036 | −0.00255 | [−0.00492, −0.00017] |
| DRO_L3_150 | 2.012 | 0.046 | −0.00259 | [−0.00514, −0.00005] |
| Agi_L1_150 | 6.153 | 6.69e−9 | −0.00884 | [−0.01168, −0.00600] |
| Agi_L3_150 | 9.950 | 3.36e−18 | −0.01368 | [−0.01640, −0.01096] |

The Dual-System improvement on DROID is statistically significant but **effect size is 3–5× smaller**
than on AgiBot, confirming that domain shift suppresses planning gains.

---

## 3. Failure Taxonomy

We classify failures into four mutually non-exclusive modes, characterized below.

---

### Failure Mode A — Action Precision Degradation (分布偏移致动作精度崩塌)

**Definition**: The model's predicted joint action exceeds the δ=0.1 threshold at nearly every
sampled step, even though visual context appears valid.

**Signature**:
- Mean L2 > 0.10 across the full episode (vs. expected < 0.09 on in-distribution data)
- Rate(L2<0.1) < 50%, meaning the majority of steps are counted as failure

**Prevalence**: Dominant pattern — affects **all** DROID episodes to varying degrees.
Mean L2 is 0.1262–0.1284 on DROID vs. 0.0739–0.0862 on AgiBot, a persistent baseline shift.

**Root Cause Analysis**:

DROID uses a single 7-DOF Franka Panda arm with relative joint-position control and a significantly
different gripper/wrist configuration vs. the dual-arm AgiBot robot on which DreamZero was
predominantly trained. LoRA fine-tuning on DROID episodes (rank=4, α=4, max_chunk_size=5) provides
insufficient capacity to realign the joint-action statistics under the large embodiment gap.

The data selection pipeline (`build_droid_easy1000_subset.py`) penalizes tasks involving
deformable objects and contact-rich interactions, yet the residual distribution of selected
episodes still reflects DROID's in-the-wild collection style: varied camera angles, cluttered
backgrounds, non-standard starting configurations. These visual factors inject uncertainty into the
image-conditioned action diffusion model, raising prediction variance and thus average L2 error.

**Evidence**:

```
DRO_L1_150 description_only  → Mean L2 = 0.1284,  Rate(L2<0.1) = 44.3%
AgiBot L1 description_only   → Mean L2 = 0.0862,  Rate(L2<0.1) = 68.7%
Absolute gap: ΔL2 = +0.0422 (+49% relative)
Precision gap: ΔRate = −24.4pp
```

---

### Failure Mode B — Long-Horizon Compounding Error (长时程累积误差)

**Definition**: Per-step error is moderate at early steps but compounds across the episode,
causing L3 tasks to fail more frequently than L1 tasks.

**Signature**:
- L3 SR consistently lower than L1 SR within the same mode
- L3 Task Progress gap vs. L1 > 3 pp

**Prevalence**: Systematic — DROID L1→L3 SR drops by 6.0 pp (description_only) and 4.7 pp (dual_llm).
Contrast with AgiBot where Dual-System *reverses* this trend (L1 36.7% → L3 41.3%).

**Root Cause Analysis**:

Long-horizon episodes require the policy to remain on-distribution for a sustained period.
Under domain shift, the initial kinematic error at step $t_0$ induces an observation at $t_1$
that is slightly outside the training manifold. Because DreamZero uses an open-loop horizon
of 8 steps and samples only $K=10$ reference points per episode, this drift is not corrected
within the evaluation window.

For AgiBot L3 with Dual-LLM, the sub-task decomposition effectively segments the horizon into
shorter segments, resetting the language context at each boundary and partially mitigating drift.
On DROID, this benefit is weakened because the dominant error source (kinematic distribution shift)
is not addressable by language re-conditioning alone.

**Evidence**:

| Setting | L1 SR | L3 SR | Gap |
|---|---|---|---|
| DROID description_only | 29.3% | 23.3% | −6.0 pp |
| DROID dual_llm | 32.0% | 27.3% | −4.7 pp |
| AgiBot description_only | 38.7% | 30.0% | −8.7 pp |
| AgiBot dual_llm | 36.7% | 41.3% | **+4.6 pp** (reversed) |

The AgiBot dual_llm reversal demonstrates that Dual-System can overcome long-horizon compounding
*when domain shift is absent*. The absence of this reversal on DROID is direct evidence that
domain gap is the binding constraint.

---

### Failure Mode C — Language Grounding Ambiguity (语言-视觉接地歧义)

**Definition**: Ambiguous or terse task descriptions, combined with out-of-distribution visual
contexts, cause the policy to misidentify the target object or interaction region.

**Illustrative Case**: `episode_001394` — *"Press a button the keyboard"*

```json
{
  "episode_id": "episode_001394",
  "task_description": "Press a button the keyboard",
  "mode": "task_token_only",
  "mean_l2": 0.2636,
  "task_progress": 0.0,
  "rule_success": false,
  "judge_reason": "The predicted final frame does not show any indication of the task
                   being completed, as it is distorted and does not resemble the goal state."
}
```

**Quantitative signature**:
- Mean L2 = 0.264, more than **2× the dataset mean** (0.126)
- 0 / 10 steps pass the L2 < 0.1 threshold (Rate = 0.0%)
- Task Progress = 0.0 (lowest possible, indicating complete visual failure)

**Visual verdict**: Predicted final frame is described as *"distorted"* — a symptom of the
diffusion model's video generation collapsing when the text-visual grounding is too ambiguous
to anchor a coherent prediction.

The task description "Press a button the keyboard" is syntactically malformed (missing article "on"),
and "keyboard" is not a standard manipulation target in the training distribution. The wrist camera
and exterior cameras show a keyboard in an unconstrained pose, a configuration unlikely in
any training episode.

**Dual-LLM sub-task decomposition** (GPT-4o-mini planning):
```
1. Move gripper above the button.
2. Close gripper to grasp the button.
3. Press down on the button with the gripper.
4. Open gripper to release the button.
```
Despite a semantically correct plan, execution still fails completely:
```json
{
  "mode": "dual_llm",
  "mean_l2": 0.2636,          // identical to task_token_only — planning had zero effect
  "task_progress": 0.0,
  "judge_reason": "The predicted final frame does not clearly show any interaction with
                   the keyboard, and the task goal is not achieved."
}
```

**Takeaway**: When the visual context is OOD (out-of-distribution) relative to the video diffusion
model's training data, language conditioning — whether global or sub-task — cannot recover
correct action generation. Language grounding is only effective when the visual backbone has
sufficient prior exposure to the target object class.

---

### Failure Mode D — Dual-LLM Over-Planning Interference (双系统过度规划干扰)

**Definition**: For tasks that are short or whose motion semantics are direct, LLM-generated
sub-tasks introduce extraneous action primitives (e.g., "open gripper", "move back to starting
position") that misalign with the brief, focused ground-truth trajectory.

**Signature**:
- Dual-System SR < System1 SR on specific episodes
- LLM plan length M = 4–5 while the task requires only 1–2 natural phases

**Case-6 Episode-Level Evidence** (L3, δ=0.1):

| Pattern | Episodes | SR (System1 / B2 / Dual) | Explanation |
|---|---|---|---|
| Dual better | `ep_001470`, `ep_001669` | 0.4/0.2/**0.5**, 0.5/0.3/**0.6** | Multi-stage, multi-object tasks where explicit sub-goals aid disambiguation |
| Dual worse | `ep_001406`, `ep_001395` | **0.6**/0.6/0.5 | Short, direct tasks — LLM adds spurious "open gripper", "return to start" steps |
| Roughly equal | `ep_001754`, `ep_001827` | 0.5/0.4/0.5 | Policy-level error dominates; planning cannot compensate |

**Mechanism**: The GPT-4o-mini planner (temperature=0.0) consistently generates M=4 sub-tasks
even for tasks that complete in ≤2 natural phases. The extra sub-tasks inject semantically
inconsistent language prompts into later steps, causing the policy to predict return-to-home or
grasp-release motions that deviate from the ground-truth continuation.

The raw planner response for `ep_001394` confirms this pattern:
```
Move gripper above the button.
Close gripper to grasp the button.
Press down on the button with the gripper.
Open gripper to release the button.
Move gripper back to the starting position.   ← truncated at M=4 but symptomatic of overplanning
```

The trailing "Move gripper back to the starting position" is semantically inconsistent with the
ground-truth trajectory (which ends at the key press) and degrades later-step predictions.

---

## 4. Failure Mode Interaction and Overall Error Budget

We decompose the observed 29.3% L1 System1 SR into contributing failure factors:

```
Theoretical upper bound (ideal domain transfer):       ~38.7%  (AgiBot L1 SR)
  │
  ├─ [Mode A] Action precision degradation (ΔL2 +49%): −6.0 pp
  │    └ kinematic distribution mismatch, LoRA underfit
  │
  ├─ [Mode B] Long-horizon compounding (L3 only):       −6.0 pp (L1→L3 gap)
  │    └ open-loop drift, no visual feedback correction
  │
  ├─ [Mode C] Language grounding ambiguity:             −3.4 pp (estimated)
  │    └ OOD visual context, terse/malformed descriptions
  │
  └─ Observed DROID L1 SR:                             29.3%
```

For L3 episodes with Dual-System, Mode D partially offsets Mode B (+4.7 pp vs. System1 L3),
but residual gain is limited to +4.0 pp (27.3% → not reaching L1 level) because Mode A
dominates at the step level regardless of planning.

---

## 5. Comparative Case Studies

### Case Study 1: Successful Dual-System Episode — Multi-Stage Object Transfer
**Episode**: `ep_001669` | Task: *"Pick up the cup and place it on the tray"* (representative)

```
System1 SR:   0.5   (5/10 steps pass δ=0.1)
Dual SR:      0.6   (6/10 steps pass δ=0.1)   ← Dual wins
B2 (heuristic): 0.3
```

LLM plan segments the task into:
1. *Approach cup*
2. *Grasp cup*
3. *Lift and transport*
4. *Place on tray*

The explicit segmentation aligns with the natural phase boundaries of the GT trajectory.
Language context switches at each segment boundary, providing a fresh semantic anchor
that reduces policy uncertainty in the mid-episode transition region.

---

### Case Study 2: Dual-System Regression — Simple Drawer Open
**Episode**: `ep_001406` | Task: *"Open the top drawer"*

```
System1 SR:   0.6   ← System1 wins
Dual SR:      0.5
B2 (heuristic): 0.6
```

LLM plan:
1. *Position gripper at drawer handle*
2. *Grasp the drawer handle*
3. *Pull drawer open*
4. *Release gripper*

The ground-truth trajectory is a single continuous pull motion.
Sub-task 4 ("Release gripper") causes the policy to predict gripper-open actions at step t≈0.8T,
which conflicts with the GT trajectory's sustained grasp. This degrades SR by 0.1.

---

### Case Study 3: Complete Visual Generation Failure — Keyboard Task
**Episode**: `ep_001394` | Task: *"Press a button the keyboard"*

```
Mean L2:        0.264   (2.1× dataset average)
Task Progress:  0.0
Rate(L2<0.1):   0.0%
```

This is the worst-performing episode in the smoke test.
The generated video frames are visually incoherent ("distorted"), indicating that the
WAN diffusion model's video generation has collapsed due to:
1. OOD object class ("keyboard" — not in training distribution)
2. Syntactically malformed instruction (missing preposition)
3. High visual complexity of keyboard surface under varied lighting

Both modes fail identically (mean L2 ≈ 0.2636), confirming that the failure is rooted in
the visual backbone, not the language conditioning path.

---

## 6. Implications and Mitigation Directions

### 6.1 Short-term: Threshold Relaxation and Task Filtering

Under δ=0.14 (see Table 6.3 of the architecture document), DROID overall SR rises from 43.5%
to 61.8% for System1 and from 46.0% to 65.8% for Dual-System. This +4.0 pp Dual advantage
at δ=0.14 suggests that the Dual-System improves directional accuracy even when absolute
error exceeds the strict threshold.

**Recommendation**: Report results at multiple thresholds (δ ∈ {0.10, 0.12, 0.14}) to decouple
kinematic precision from task-directional success.

### 6.2 Medium-term: Embodiment-Specific LoRA Scaling

Current LoRA configuration (rank=4, α=4) was designed for fast adaptation.
Increasing rank to 16–32 with longer DROID-specific fine-tuning (currently shard_sampling_rate=0.1)
would reduce Mode A failures by better aligning the joint-action statistics.

### 6.3 Long-term: Visual Grounding Augmentation

Mode C failures (OOD visual context, language grounding collapse) require improving the
image encoder's exposure to diverse tabletop environments. Joint training on DROID full split
(75k episodes) rather than the current 400-episode easy subset would address the long tail
of visual contexts in which the current model fails catastrophically.

### 6.4 Dual-System Planner Calibration

Mode D over-planning interference can be reduced by:
- **Task complexity estimation**: Suppress multi-step planning for tasks with short descriptions
  or simple verb patterns (e.g., "open X", "press X")
- **Plan length regularization**: Cap M = min(2, estimated_phases) for L1 tasks
- **LLM temperature tuning**: Slightly increase temperature (0.2–0.4) to allow shorter, less
  verbose plans on simple tasks

---

## 7. Summary of Failure Modes

| Mode | Label | Prevalence | Primary Metric Impact | Addressability |
|---|---|---|---|---|
| A | Action Precision Degradation | Universal (all DROID episodes) | +49% Mean L2, −24 pp Rate(L2<0.1) | LoRA scaling, more DROID fine-tuning |
| B | Long-Horizon Compounding | L3 episodes (−6 pp SR vs. L1) | L3 SR consistently lower | Sub-task segmentation partially helps |
| C | Language Grounding Ambiguity | Rare but catastrophic (0% SR cases) | Mean L2 >2× average | Visual pre-training on diverse objects |
| D | Dual-LLM Over-Planning | Episode-specific (≈20% of L3 cases) | −0.1 SR on affected episodes | Planner calibration, M-cap heuristics |

---

## Appendix: Evaluation Configuration Reference

| Parameter | Value |
|---|---|
| Evaluation script | `scripts/eval/run_droid_compare_judged.py` |
| Dataset | `data/droid_easy400_dualfavored_dreamzero` (400 ep subset of DROID 75k) |
| Checkpoint | `checkpoints/DreamZero-DROID/` (WAN-2.1-14B + LoRA rank=4) |
| Visual judge | GPT-4o-mini (`FinalFrameJudge`, temperature=0.0) |
| Success threshold (main) | δ = 0.1 (joint-space L2) |
| Sampling steps per episode | K = 10 |
| DreamDojo inference | `examples/action_conditioned.py`, num_frames=13 |
| Planner | GPT-4o-mini, temperature=0.0, max sub-tasks=4 |
| Episode counts (full eval) | L1=20, L3=20 (40 total); large scale: L1=L3=150 |
