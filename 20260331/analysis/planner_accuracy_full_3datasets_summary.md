# Full Planner Accuracy Summary: DROID, AgiBot, EgoDex

This summary reuses the completed full DROID-300 and EgoDex-1000 planner runs, and a newly completed full AgiBot-300 run on 2026-03-31.

| Dataset | n | Metric | llm_raw | llm_self | llm_val | Note |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| DROID | 300 | VAL-valid rate | 0.3833 | 0.7400 | 0.9967 | planner logical validity |
| DROID | 300 | Self-changed rate |  | 0.4400 |  | share of episodes where blind self-review changed the plan |
| DROID | 300 | VAL-corrected rate |  |  | 0.6333 | share of episodes where VAL path performed a correction |
| AgiBot | 300 | VAL-valid rate | 1.0000 | 0.8767 | 1.0000 | planner logical validity |
| AgiBot | 300 | Self-changed rate |  | 0.2500 |  | share of episodes where blind self-review changed the plan |
| AgiBot | 300 | VAL-corrected rate |  |  | 0.1233 | share of episodes where VAL path performed a correction |
| EgoDex | 1000 | VAL-valid rate | 0.9200 | 0.9560 | 0.9910 | planner logical validity |
| EgoDex | 1000 | GT similarity | 0.2047 | 0.2031 | 0.2047 | token-F1 recall against annotated goal-step decomposition |
| EgoDex | 1000 | Combined score | 0.5623 | 0.5795 | 0.5979 | 0.5 * VAL-validity + 0.5 * GT similarity |

Key takeaways:
- DROID VAL-valid rate: raw=0.3833, self=0.7400, val=0.9967.
- AgiBot VAL-valid rate: raw=1.0000, self=0.8767, val=1.0000.
- EgoDex combined score: raw=0.5623, self=0.5795, val=0.5979; val-rate improves from 0.9200/0.9560 to 0.9910.