# LOGIV LIBERO-10 Results

Allocated episode records: **40**; valid executions: **40**.

> Development-only records: **40/40**; oracle-grounded records: **20/40**. These records are not API-VLM or preregistered holdout evidence.

| Goal mode | Deviation | Method | Success | 95% Wilson interval |
| --- | --- | --- | ---: | ---: |
| METADATA_ASSISTED | NOMINAL | BASE | 12/20 | [0.387, 0.781] |
| ↳ task 8 |  |  | 12/20 | [0.387, 0.781] |
| METADATA_ASSISTED | NOMINAL | FULL_LOGIV | 10/20 | [0.299, 0.701] |
| ↳ task 8 |  |  | 10/20 | [0.299, 0.701] |

## Runtime and recovery metrics

| Method | Repaired episodes | Successful repaired episodes | Mean attempts | Mean repairs | Mean VAL calls | Mean steps | Mean wall time (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BASE | 0 | 0 | 1.00 | 0.00 | 0.00 | 448.10 | 101.07 |
| FULL_LOGIV | 18 | 8 | 3.85 | 2.05 | 4.70 | 458.30 | 112.75 |

| Method | Instrumented records | Committed receipts | Failed receipts | Unknown receipts | Precondition rejects | Effect rejects | Final-goal rejects |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BASE | 20 | 0 | 0 | 0 | 0 | 0 | 0 |
| FULL_LOGIV | 0 | — | — | — | — | — | — |

## Paired task-stratified comparisons

- Full − BASE: -0.100 (paired bootstrap 95% [-0.300, +0.100], N=20).
