# RoboTwin integration

`logiv-origin.patch` connects LOGIV Origin to the π0.5 evaluator in TACO. It
contains the PDDL/DAG runtime, independent GPT-4o state grounding, VAL checks,
bounded local repair, optional CFN repair selection, and multi-camera gate
evidence.

The patch is based on TACO commit `ee9e06d`. Apply it to a clean checkout:

```bash
git checkout ee9e06d
git apply /path/to/LOGIV_Origin/patches/robotwin/logiv-origin.patch
```

Run the integration tests from `third_party/Robotwin`:

```bash
python3 -m unittest discover -s tests -p 'test_pi05*.py' -v
python3 -m unittest tests.test_logiv_origin_options -v
```

The patched evaluator consumes sequential simulator episodes beginning at the
requested runtime seed. It has no episode allowlist, pool discovery, ranking,
reachability filter, or episode substitution interface.
