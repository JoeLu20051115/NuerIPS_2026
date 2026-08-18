# RoboTwin Oracle Legacy Truth-Value Compatibility

## Context

Historical embedded oracle records may serialize unresolved facts as
`UNKNOWN`. Current RoboTwin records serialize the same state as `UNRESOLVED`.
`revalidate_embedded_report` currently constructs `TruthValue` directly from
the serialized string, so a historical `UNKNOWN` record raises `ValueError`.

## Decision

Normalize only the legacy serialized value `UNKNOWN` to `UNRESOLVED` at the
oracle report ingestion boundary before constructing `TruthValue`. Leave
`TRUE`, `FALSE`, `UNRESOLVED`, and invalid values unchanged so the enum remains
the validation authority.

This is preferred over changing the test because the oracle explicitly audits
historical records. It is also preferred over changing the enum's serialized
value because current VLM-facing output must remain `UNRESOLVED`.

## Scope

- Change only `scripts/report_robotwin_logiv_oracle.py`.
- Reuse the existing failing historical-record test; do not add abstractions or
  dependencies.
- Do not change experiment artifacts, reported accuracy, seed selections, or
  current truth-value serialization.

## Verification and Integration

1. Confirm the existing oracle test fails on `UNKNOWN`.
2. Apply the one-boundary normalization.
3. Run the oracle test file and the relevant RoboTwin report/launcher tests.
4. Commit and push the compatibility fix to PR #2.
5. Merge PR #2 into `main`, then verify the remote `main` contains the result
   commit and compatibility fix.
