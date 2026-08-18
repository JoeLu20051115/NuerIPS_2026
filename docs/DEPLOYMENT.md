# Deployment

## Requirements

- Linux with an NVIDIA GPU supported by the selected OpenPI checkpoint
- Docker with GPU support
- Git submodules
- `uv`
- `jq`

Clone and install:

```bash
git clone --recurse-submodules https://github.com/JoeLu20051115/NuerIPS_2026.git LOGIV_Origin
cd LOGIV_Origin
uv sync
```

The Python package includes the canonical PDDL domain, prompt contract,
coverage contract, monitor contract, and VAL build manifest.

## Build VAL

```bash
scripts/build_val_for_libero.sh
```

The builder uses `src/pi05_libero_repro/logiv/config/val-build.json` to pin the
source revision, container image, and binary digests. The resulting validator
is placed under `artifacts/tools/val/`.

## Start the policy service

```bash
scripts/run_policy_server.sh GPU PORT CHECKPOINT_DIR
```

Example:

```bash
scripts/run_policy_server.sh 0 8000 /models/pi05_libero
```

The launcher checks the checkpoint layout, normalization statistics, and the
pinned OpenPI revision before serving the policy over WebSocket.

## Integrate the controller

Use the packaged modules to assemble the environment loop:

- `gpt4o_grounding.py` converts fresh camera observations into typed facts.
- `initial_proposal.py` and `gpt4o_planning.py` build the initial symbolic
  proposal.
- `val.py` certifies the PDDL plan.
- `dag.py` compiles the certified actions into an execution graph.
- `controller.py` dispatches bounded action chunks and applies retry budgets.
- `shadow_monitor.py` confirms transitions from independent observations.
- `online_repair.py` performs bounded graph-local repair.
- `libero_adapter.py` binds LIBERO observations and actions.
- `robotwin.py` binds RoboTwin tasks and multi-camera state gates.

Load packaged assets without relying on the current directory:

```python
from pi05_libero_repro.logiv import origin_config_path

domain = origin_config_path("domain.pddl")
prompts = origin_config_path("prompts.json")
coverage = origin_config_path("coverage.json")
```

The environment owner supplies observations, native completion signals, policy
actions, and a VAL binary. LOGIV keeps execution bounded and rejects invalid or
stale certificates before action dispatch.

## Checks

```bash
uv run pytest -q
uv build
bash -n scripts/*.sh
uv run python -c 'from pi05_libero_repro.logiv import origin_config_path; print(origin_config_path("domain.pddl"))'
```
