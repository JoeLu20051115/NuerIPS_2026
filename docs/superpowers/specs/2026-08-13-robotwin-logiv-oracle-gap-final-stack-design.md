# RoboTwin LOGIV Final-Stack Oracle Gap Sweep Design

**Status:** Approved on 2026-08-13. Execute without further configuration questions.

## Objective and Evidence Label

Run exactly one fresh attempt for each of the 40 unsuccessful `(task, seed)`
cells in the audited 60/100 RoboTwin development oracle. The experiment is
allowed to select a task- and seed-specific configuration and therefore remains
a **development oracle**. It is not a frozen 100-episode evaluation, a fair
generalization result, or an independent holdout.

The implementation checkout is fixed at `554d3118`; the TACO runtime is fixed
at `8de0ed9520989f9fd156904291d0895b9a361886`; the policy checkpoint remains
`rhodes-team-teleai/pi05_TACO_robotwin2_finetuned`. Each gap gets one attempt.
There is no retry, mid-run retuning, or replacement seed.

## Frozen 40-Cell Scope

- `handover_block`: 100002, 100003, 100006, 100008, 100009, 100010, 100011
- `open_microwave`: 100001, 100004, 100005, 100006, 100007, 100010, 100011, 100013
- `place_dual_shoes`: 100001, 100002, 100008, 100009, 100012, 100019, 100021
- `stamp_seal`: 100001, 100008
- `blocks_ranking_size`: 100005, 100007, 100009
- `move_can_pot`: 100009, 100010
- `turn_switch`: 100001, 100004, 100008, 100010
- `stack_blocks_three`: 100001, 100002, 100003, 100005, 100010
- `beat_block_hammer`: 100008, 100010

`stack_bowls_three` has no missing cell and is not run. The configuration audit
must prove that the union above is exactly the complement of the 60 selected
cells embedded in
`results/robotwin-logiv-strict-baseline-instruction-oracle-development-20260813.json`.

## Task-Directed Profiles

All profiles use `demo_clean`, `unseen`, 50-step base action chunks, low-detail
VLM images, DAG monitoring from the first observation, and the original frozen
episode instruction for each cell.

1. **Checkpoint CFN:** `handover_block`, `move_can_pot`, and
   `beat_block_hammer`. Use 50-step repair chunks, the checkpoint CFN for each
   task, no minimum base delay, and stage thresholds `[4,4,2]`, `[4]`, and
   `[4,2]`, respectively.
2. **Registered PDDL node prompt:** `turn_switch`. Use the registered node
   prompt, 50-step repair chunks, no minimum base delay, and threshold `[4]`.
3. **Original episode repair prompt:** `blocks_ranking_size`,
   `place_dual_shoes`, and `stamp_seal`. Preserve the original episode prompt,
   use 50-step repair chunks, minimum base steps 700/250/150, and thresholds
   `[4,3,2]`, `[3,2]`, and `[3,2]`.
4. **10-step node-internal replanning:** `open_microwave` and
   `stack_blocks_three`. Preserve the original episode prompt, replan every 10
   policy steps within a node, use 50-step repair chunks, minimum base steps
   400/650, and thresholds `[4,3]` and `[4,3,2]`.

## Execution Topology

Two independent queues run concurrently:

- GPU 0: `open_microwave` (8 cells), then `stack_blocks_three` (5 cells).
- GPU 1: `handover_block` (7), `move_can_pot` (2),
  `beat_block_hammer` (2), `turn_switch` (4), `place_dual_shoes` (7),
  `stamp_seal` (2), then `blocks_ranking_size` (3).

GPU 2 remains unused. Every task is a separate launcher process with a
12-hour hard timeout. A nonzero exit stops only that GPU's remaining queue and
preserves all completed artifacts; the other queue continues. Failed or timed
out tasks are not automatically retried.

The launcher receives a minimal backward-compatible `--gpu` option so an
explicit task whitelist can be scheduled independently of its historical
worker grouping. Without `--gpu` and without explicit tasks, its existing
three-worker behavior is unchanged.

## Data Flow and Provenance

The committed 60/100 JSON supplies both the frozen protocol and selected-cell
set. Four audited gap configs encode its exact complement and matching episode
instructions. The launcher invokes the fixed TACO runtime, which writes native
RoboTwin results and LOGIV event records under one unique run tag. After both
queues finish, the oracle reporter scans the historical plus new event tree,
requires baseline-instruction equality, embeds selected records, and freshly
revalidates every selected PDDL observation with VAL.

The final report must retain the development-oracle label and record the runtime
commit, checkpoint digest, source run tags, exact selected records, and VAL
certificates. The 60/100 report remains immutable; the sweep produces a new
dated report.

## Acceptance and Failure Handling

Before launch:

- focused launcher tests prove explicit GPU routing and backward compatibility;
- config tests prove exactly 40 unique gap cells, exact episode instructions,
  mutually exclusive profile membership, and every profile flag;
- the repository suite and 14 TACO runtime tests pass;
- checkpoint, tokenizer, VAL, API credential, RoboTwin assets, GPUs, and absence
  of duplicate evaluators pass preflight.

After launch, accept partial data if a task fails, but do not fill it through an
unplanned retry. For a fully completed sweep, require 40 distinct new episode
records and one native outcome per requested cell. Rebuild and audit the oracle
even when no new success is found. Report both the number of newly recovered
cells and the resulting oracle total out of 100, without relabeling it as a
generalization result.
