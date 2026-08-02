# LOGIV LIBERO-10 Results

Allocated episode records: **10**; valid executions: **10**.

> Development-only records: **10/10**; oracle-grounded records: **10/10**. These records are not API-VLM or preregistered holdout evidence.

| Goal mode | Deviation | Method | Success | 95% Wilson interval |
| --- | --- | --- | ---: | ---: |
| METADATA_ASSISTED | NOMINAL | FULL_LOGIV | 8/10 | [0.490, 0.943] |
| ↳ task 0 |  |  | 1/1 | [0.207, 1.000] |
| ↳ task 1 |  |  | 1/1 | [0.207, 1.000] |
| ↳ task 2 |  |  | 1/1 | [0.207, 1.000] |
| ↳ task 3 |  |  | 1/1 | [0.207, 1.000] |
| ↳ task 4 |  |  | 1/1 | [0.207, 1.000] |
| ↳ task 5 |  |  | 0/1 | [0.000, 0.793] |
| ↳ task 6 |  |  | 1/1 | [0.207, 1.000] |
| ↳ task 7 |  |  | 1/1 | [0.207, 1.000] |
| ↳ task 8 |  |  | 1/1 | [0.207, 1.000] |
| ↳ task 9 |  |  | 0/1 | [0.000, 0.793] |

## Runtime and recovery metrics

| Method | Repaired episodes | Successful repaired episodes | Mean attempts | Mean repairs | Mean VAL calls | Mean steps | Mean wall time (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FULL_LOGIV | 1 | 0 | 2.20 | 0.20 | 1.30 | 288.80 | 58.26 |

| Method | Instrumented records | Committed receipts | Failed receipts | Unknown receipts | Precondition rejects | Effect rejects | Final-goal rejects |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FULL_LOGIV | 0 | — | — | — | — | — | — |

## Paired task-stratified comparisons

No complete paired comparison is available yet.
