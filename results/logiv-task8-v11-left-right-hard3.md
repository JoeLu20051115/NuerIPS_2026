# LOGIV LIBERO-10 Results

Allocated episode records: **3**; valid executions: **3**.

> Development-only records: **3/3**; oracle-grounded records: **3/3**. These records are not API-VLM or preregistered holdout evidence.

| Goal mode | Deviation | Method | Success | 95% Wilson interval |
| --- | --- | --- | ---: | ---: |
| METADATA_ASSISTED | NOMINAL | FULL_LOGIV | 0/3 | [0.000, 0.561] |
| ↳ task 8 |  |  | 0/3 | [0.000, 0.561] |

## Runtime and recovery metrics

| Method | Repaired episodes | Successful repaired episodes | Mean attempts | Mean repairs | Mean VAL calls | Mean steps | Mean wall time (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FULL_LOGIV | 3 | 0 | 3.67 | 2.67 | 5.00 | 520.00 | 131.93 |

| Method | Instrumented records | Committed receipts | Failed receipts | Unknown receipts | Precondition rejects | Effect rejects | Final-goal rejects |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FULL_LOGIV | 3 | 4 | 7 | 0 | 0 | 7 | 1 |

## Paired task-stratified comparisons

No complete paired comparison is available yet.
