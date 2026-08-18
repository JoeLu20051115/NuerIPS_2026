# Deployment guide

This guide covers the two supported integration surfaces: LIBERO through the
pinned OpenPI submodule, and RoboTwin through the TACO patch.

## Common prerequisites

Install the Python environment and initialize submodules:

```bash
uv sync
git submodule update --init --recursive
```

Export the GPT-4o credential only in the process environment:

```bash
export OPENAI_API_KEY='...'
```

Do not store credentials in configuration or shell scripts. The direct client
fails closed when the key is missing, the API response is malformed, the model
identity is unexpected, or the finite retry budget is exhausted.

## VAL

LIBERO uses the packaged reproducible build manifest in
`src/pi05_libero_repro/logiv/config/val-build.json`:

```bash
scripts/build_val_for_libero.sh
```

This produces:

```text
artifacts/tools/val-ubuntu22/Validate
artifacts/tools/val-ubuntu22/libVAL.so
```

The build script verifies both hashes. Runtime launchers also verify the
`Validate` hash before starting. RoboTwin may use another compatible VAL path
provided explicitly with `--val-binary`.

## LIBERO/OpenPI

### Build the evaluator image

The Dockerfile is tied to OpenPI revision
`650c5b0283a49c42784fb5055a0507da2c6d347d`:

```bash
docker build \
  -t pi05-libero-eval:650c5b0 \
  -f docker/Dockerfile.libero \
  external_repos/openpi
```

### Start the policy server

```bash
scripts/run_policy_server.sh GPU PORT CHECKPOINT_DIR
```

Example:

```bash
scripts/run_policy_server.sh 0 8000 /models/pi05_libero
```

The launcher checks:

- numeric GPU and port values;
- the checkpoint `params/` directory;
- LIBERO normalization statistics;
- the pinned OpenPI revision.

Server output is streamed to the terminal. The launcher does not create a log
directory.

### Run LOGIV

```bash
scripts/run_logiv_eval.sh GPU PORT OUTPUT_DIR [EVALUATOR_ARGS...]
```

Example:

```bash
scripts/run_logiv_eval.sh \
  0 8000 /tmp/logiv-origin-run \
  --task-ids 0,1 \
  --episode-indices 0:2 \
  --no-video
```

`OUTPUT_DIR` is an explicit runtime mount. Use a path outside the repository in
deployment. If a path inside the checkout is used, the generated records are
ignored by Git.

The wrapper fixes these public deployment choices:

```text
method arm:        FULL_LOGIV
perception:        gpt4o
goal mode:         METADATA_ASSISTED
deviation label:   ORIGIN
prompt state:      locked
```

Canonical defaults resolve from `src/pi05_libero_repro/logiv/config/` and ship
inside the wheel. Runtime limits and the task/episode range may be supplied as
extra evaluator arguments.

## RoboTwin/TACO

### Apply the integration

Start from a clean TACO checkout at commit `ee9e06d`:

```bash
git checkout ee9e06d
git apply /path/to/LOGIV_Origin/patches/robotwin/logiv-origin.patch
```

Validate the patched integration:

```bash
cd third_party/Robotwin
python3 -m unittest discover -s tests -p 'test_pi05*.py' -v
python3 -m unittest tests.test_logiv_origin_options -v
```

### Run episodes

```bash
uv run python scripts/run_robotwin_logiv.py \
  --task TASK \
  --episodes COUNT \
  --seed START \
  --taco /path/to/TACO \
  --checkpoint /path/to/checkpoint \
  --tokenizer /path/to/tokenizer.model \
  --val-binary /path/to/Validate
```

Optional flags configure GPU index, action chunk length, repair chunk length,
state-gate image detail, stall observations, and a checkpoint-bundled repair
CFN.

The requested seed is passed directly to the simulator. For more than one
episode, the evaluator uses consecutive values beginning at `START`. Setup
failure for a requested episode stops the run; the launcher does not scan for a
replacement.

## Runtime data policy

The source release does not track runtime output. `.gitignore` excludes:

- checkpoints, datasets, validators, and local benchmark checkouts;
- result and evaluation directories;
- logs and JSONL event streams;
- videos and screenshots;
- virtual environments and caches;
- local environment files.

RoboTwin gate evidence and LIBERO evaluator output are diagnostic runtime data.
Route them to caller-owned storage and apply the retention policy of the target
deployment.

## Troubleshooting

`OpenPI revision mismatch`
: Update the submodule with `git submodule update --init --recursive`; do not
  silently run against another revision.

`Validate hash mismatch`
: Rebuild VAL with `scripts/build_val_for_libero.sh` and confirm the Origin
  manifest is unchanged.

`OPENAI_API_KEY is required`
: Export the key in the process environment before starting the evaluator.

`RoboTwin could not build requested seed`
: The requested simulator state could not be constructed. The run stops rather
  than replacing the episode.

`repair_cfn_path must name an existing file`
: Remove `--repair-cfn` or pass the exact checkpoint-bundled CFN file.
