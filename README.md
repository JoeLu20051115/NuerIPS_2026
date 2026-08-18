# LOGIV Origin

LOGIV Origin is a deployable closed-loop control module for long-horizon robot
tasks. It combines visual fact grounding, symbolic planning, formal plan
validation, causal DAG execution, monitored action dispatch, and bounded local
repair.

## Components

```text
src/pi05_libero_repro/logiv/   LOGIV runtime and packaged configuration
scripts/                       VAL builder and policy service launcher
docker/                        LIBERO runtime image
external_repos/openpi/         pinned OpenPI dependency
docs/                          method and deployment guides
tests/                         runtime tests
```

## Install

```bash
git clone --recurse-submodules https://github.com/JoeLu20051115/NuerIPS_2026.git LOGIV_Origin
cd LOGIV_Origin
uv sync
```

Build the pinned VAL validator:

```bash
scripts/build_val_for_libero.sh
```

Start the π0.5 policy service:

```bash
scripts/run_policy_server.sh 0 8000 /path/to/pi05_libero
```

## Runtime flow

1. Ground a fresh observation into typed three-valued facts.
2. Propose a plan for the remaining symbolic task state.
3. Validate the plan with VAL.
4. Compile the certified plan into a causal DAG.
5. Dispatch a bounded policy action chunk.
6. Re-observe and confirm the expected transition.
7. Retry uncertainty within a finite budget or repair a confirmed local
   failure.
8. Finish only when symbolic completion agrees with the environment signal.

Canonical configuration is available through
`pi05_libero_repro.logiv.origin_config_path`. LIBERO integration is provided by
`libero_adapter.py`; RoboTwin task grounding and execution support is provided
by `robotwin.py`.

See [docs/METHOD.md](docs/METHOD.md) for the control model and
[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) for environment setup.

## Verification

```bash
uv run pytest -q
uv build
bash -n scripts/*.sh
```
