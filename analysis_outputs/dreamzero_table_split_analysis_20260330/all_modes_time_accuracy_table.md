# Time-Accuracy Table (All Table Modes)

| Dataset | Mode | Split | Avg Duration (s) | Success Rate | 95% CI | Task Progress | Mean L2 |
| --- | --- | --- | ---: | ---: | --- | ---: | ---: |
| AgiBot | llm_dual | L1 | 22.0 | 52% | [38%, 66%] | 0.574 | 0.0685 |
| AgiBot | llm_dual | L2 | 42.0 | 42% | [28%, 56%] | 0.502 | 0.0745 |
| AgiBot | llm_dual | L3 | 126.0 | 48% | [34%, 62%] | 0.536 | 0.0614 |
| AgiBot | llm_val | L1 | 22.0 | 62% | [48%, 76%] | 0.624 | 0.0672 |
| AgiBot | llm_val | L2 | 42.0 | 48% | [34%, 62%] | 0.542 | 0.0704 |
| AgiBot | llm_val | L3 | 126.0 | 58% | [44%, 72%] | 0.584 | 0.0593 |
| AgiBot | original | L1 | 22.0 | 30% | [18%, 42%] | 0.460 | 0.0845 |
| AgiBot | original | L2 | 42.0 | 22% | [12%, 34%] | 0.416 | 0.0942 |
| AgiBot | original | L3 | 126.0 | 16% | [6%, 28%] | 0.406 | 0.0748 |
| DROID | llm_dual | L1 | 6.6 | 44% | [30%, 58%] | 0.520 | 0.1095 |
| DROID | llm_dual | L2 | 10.0 | 32% | [20%, 44%] | 0.480 | 0.1089 |
| DROID | llm_dual | L3 | 15.5 | 34% | [22%, 48%] | 0.464 | 0.1125 |
| DROID | llm_val | L1 | 6.6 | 50% | [36%, 64%] | 0.574 | 0.1075 |
| DROID | llm_val | L2 | 10.0 | 38% | [24%, 52%] | 0.524 | 0.1032 |
| DROID | llm_val | L3 | 15.5 | 48% | [34%, 62%] | 0.538 | 0.1071 |
| DROID | original | L1 | 6.6 | 30% | [18%, 42%] | 0.492 | 0.1291 |
| DROID | original | L2 | 10.0 | 18% | [8%, 30%] | 0.420 | 0.1255 |
| DROID | original | L3 | 15.5 | 22% | [12%, 34%] | 0.434 | 0.1389 |

- This table uses all 18 appendix cells: 2 datasets x 3 splits x 3 modes.
- Since each cell summarizes 50 episodes, the full table represents 900 episode evaluations.

