# LOGIV Origin Release Design

## Goal

Turn the current research workspace into a small, deployable LOGIV source tree
named **LOGIV Origin**. The release must contain the complete runtime method,
stable deployment configuration, integration entry points, documentation, and
tests. It must not expose historical experiments, benchmark claims, result
archives, run logs, seed pools, or seed-selection tooling in the current tree.

The cleanup changes the current tree only. Existing Git history remains intact
and recoverable; history rewriting and force-pushing are outside this release.

## Release boundary

### Keep

- The `pi05_libero_repro.logiv` implementation needed for typed grounding,
  PDDL planning, VAL certification, DAG execution, monitoring, bounded online
  repair, terminal recovery, and external-success adjudication.
- The protocol/runtime primitives imported by the LOGIV package.
- A single resolved `configs/logiv/origin/` profile with no inheritance from
  historical tuning files.
- Minimal LIBERO and RoboTwin deployment entry points.
- The OpenPI submodule pin and the RoboTwin integration patch required by those
  entry points.
- Unit/integration tests for the retained method and release hygiene.
- A concise README plus method and deployment documentation.

### Remove from the current tree

- `results/`, `evaluation_results/`, `evaluation_results_dualsystem/`, videos,
  screenshots, generated reports, JSONL event archives, and run logs.
- Seed scanning, candidate-pool discovery, accepted/frozen seed lists,
  20-to-10 selection, reachability auditing, paired-seed orchestration, and the
  associated tests/configuration.
- Experiment-only launchers, ablation/report builders, recovery-dataset
  builders, architecture snapshots, development plans, and old methodology
  notes containing results.
- Historical config sweeps and inherited tuning variants.
- Unrelated training/evaluation code and assets that are not part of LOGIV
  deployment.

An ordinary `seed` argument may remain where the underlying simulator requires
one to reproduce a single run. It must not choose, rank, reject, scan, freeze,
or substitute episodes.

## Runtime architecture

The released method preserves this control loop:

1. Convert observations into typed facts with an independent grounder.
2. Build or repair a symbolic plan for the remaining task state.
3. certify each proposed plan with VAL before dispatch.
4. Compile the plan into an executable DAG of policy prompts.
5. Dispatch bounded action chunks through the underlying policy.
6. Re-observe after each chunk and gate progress on fresh evidence.
7. Retry uncertainty within a finite evidence budget; locally repair confirmed
   failures without reopening confirmed persistent milestones.
8. Accept task completion only when the environment's native success signal
   agrees with the LOGIV terminal state.

GPT-4o is the deployment grounder/planner backend. Deterministic/oracle
implementations remain only where they are useful as explicit test doubles;
they are not presented as production evaluation modes.

## Configuration

The Origin configuration is copied as fully resolved JSON under
`configs/logiv/origin/`. Runtime defaults and launchers point only to those
files. No Origin config contains `extends`, accepted seed arrays, instruction
pools keyed by selected seed, result paths, or log paths.

Machine-local checkpoints, validators, credentials, and benchmark checkouts are
provided at runtime through arguments or environment variables. They are never
committed.

## Deployment surface

- LIBERO: a policy server plus `scripts/eval_logiv_libero.py` using the Origin
  profile and GPT-4o backend.
- RoboTwin: a generic single-run launcher and a clean integration patch. The
  launcher accepts tasks, episode count, and an optional simulator seed but has
  no accepted-seed or seed-pool interface and streams output instead of writing
  repository logs.

## Repository hygiene

`.gitignore` blocks results, evaluation outputs, logs, caches, videos, runtime
artifacts, credentials, checkpoints, and local benchmark checkouts. A release
hygiene test checks that forbidden directories and seed-selection terminology
are absent from tracked deployment files.

## Verification

The release is complete when:

- retained unit and integration tests pass;
- canonical Origin configs load without inheritance;
- all documented entry points provide `--help` or validate shell syntax;
- a tracked-file scan finds no result/log archive or seed-selection module;
- package build/import smoke tests pass;
- the branch is pushed, reviewed through a pull request, merged into `main`, and
  remote `main` is verified at the merge commit.
