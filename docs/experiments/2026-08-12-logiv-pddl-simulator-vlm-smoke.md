# LOGIV PDDL + Simulator-VLM Smoke Test

**Date:** 2026-08-12

**Purpose:** Development plumbing test for the PDF-aligned LOGIV control path.
This is simulator/oracle grounding evidence, not GPT-4o or real-VLM evidence.

## Configuration

- Method arm: `FULL_LOGIV`
- Checkpoint: frozen `full` pi0.5 LIBERO checkpoint
- GPU: 0 (`NVIDIA H200 NVL`)
- Task: LIBERO task 0, `put both the alphabet soup and the tomato sauce in the basket`
- Episode index: 0
- Master seed: 7
- Derived policy seed: 1356440968
- Perception backend: `scripted-oracle`
- OpenAI key forwarding: disabled
- Output: `artifacts/logiv-pddl-simulator-vlm-smoke/evaluation`

## Commands

```bash
scripts/run_policy_server.sh full 0 8012 \
  /mnt/data3/data_xingrui/.cache/openpi/openpi-assets/checkpoints/pi05_libero \
  artifacts/logiv-pddl-simulator-vlm-smoke/policy-server
```

```bash
scripts/run_logiv_eval.sh FULL_LOGIV 0 8012 \
  artifacts/logiv-pddl-simulator-vlm-smoke/evaluation \
  --run-id logiv-pddl-simulator-vlm-smoke-task0-seed7 \
  --checkpoint-name full \
  --goal-mode METADATA_ASSISTED \
  --deviation-mode NOMINAL \
  --perception-backend scripted-oracle \
  --oracle-grounding \
  --development-only \
  --task-ids 0 \
  --episode-indices 0 \
  --seed 7
```

## Result

- Terminal status: `EPISODE_SUCCESS`
- Native evaluator status: `EPISODE_SUCCESS`
- Valid record: yes
- Low-level actions: 280
- Policy requests: 57
- Physical PDDL-node attempts: 2
- Committed receipts: 2
- Failed/unknown receipts: 0/0
- Initial VAL calls: 1
- Repair rounds: 0
- Graph installs: 1
- Initial graph width: 2
- GPT-4o State Gate requests: 0
- GPT-4o repair requests: 0
- Uncaught exception: none
- Wall time: 68.58 seconds

The PDDL planner produced two `place-in` actions. The initial plan received VAL
certificate
`a87bccd8707124e74d49dbc8c1dc518867c4e8d5cf307d7e6e6a9d9bc46c30bc`.
The deterministic compiler produced graph
`f0bafa9513996c3b778d0729bbb00775b48059c3bce61d71495bd1c64dbf149c`
(`graph-f0bafa9513996c3b`). The event journal records `CERTIFICATE_ACTIVE`,
`GRAPH_ACTIVE`, and a `FACTS_AUTHORIZED` event before each of the two physical
dispatches, followed by the absorbing native-success receipt.

## Evidence

- Episode record: `artifacts/logiv-pddl-simulator-vlm-smoke/evaluation/episodes.jsonl`
- Run configuration: `artifacts/logiv-pddl-simulator-vlm-smoke/evaluation/run.json`
- VAL certificate: `artifacts/logiv-pddl-simulator-vlm-smoke/evaluation/artifacts/task_00/episode_000/certificate.json`
- Deterministic DAG: `artifacts/logiv-pddl-simulator-vlm-smoke/evaluation/artifacts/task_00/episode_000/graph.json`
- Gate/event journal: `artifacts/logiv-pddl-simulator-vlm-smoke/evaluation/artifacts/task_00/episode_000/events.jsonl`
- Policy-server log: `artifacts/logiv-pddl-simulator-vlm-smoke/policy-server/server.log`

## Conclusion

The PDF-aligned, zero-API path runs end to end: simulator facts enter the same
State Gate boundary intended for the VLM, the PDDL planner owns initial
planning, VAL certifies the plan, program code compiles the DAG, and the frozen
pi0.5 policy executes authorized nodes to native simulator success. This single
episode establishes plumbing viability only; it is not a performance estimate.
