# LOGIV LIBERO-10 Results

Allocated episode records: **5**; valid executions: **5**.

> Development-only records: **5/5**; oracle-grounded records: **5/5**. These records are not API-VLM or preregistered holdout evidence.

| Goal mode | Deviation | Method | Success | 95% Wilson interval |
| --- | --- | --- | ---: | ---: |
| METADATA_ASSISTED | NOMINAL | FULL_LOGIV | 1/5 | [0.036, 0.624] |
| ↳ task 8 |  |  | 1/5 | [0.036, 0.624] |

## Runtime and recovery metrics

| Method | Repaired episodes | Successful repaired episodes | Mean attempts | Mean repairs | Mean VAL calls | Mean steps | Mean wall time (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FULL_LOGIV | 5 | 1 | 3.40 | 2.40 | 5.00 | 498.80 | 122.15 |

| Method | Instrumented records | Committed receipts | Failed receipts | Unknown receipts | Precondition rejects | Effect rejects | Final-goal rejects |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FULL_LOGIV | 5 | 5 | 12 | 0 | 0 | 12 | 0 |

## Paired task-stratified comparisons

No complete paired comparison is available yet.
