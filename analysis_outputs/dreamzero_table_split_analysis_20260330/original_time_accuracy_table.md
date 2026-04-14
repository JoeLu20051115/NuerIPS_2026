# Time-Accuracy Table (Original Mode)

| Point | Avg Duration (s) | Success Rate | 95% CI | Task Progress | Mean L2 |
| --- | ---: | ---: | --- | ---: | ---: |
| AgiBot-L1 | 22.0 | 30% | [18%, 42%] | 0.460 | 0.0845 |
| AgiBot-L2 | 42.0 | 22% | [12%, 34%] | 0.416 | 0.0942 |
| AgiBot-L3 | 126.0 | 16% | [6%, 28%] | 0.406 | 0.0748 |
| DROID-L1 | 6.6 | 30% | [18%, 42%] | 0.492 | 0.1291 |
| DROID-L2 | 10.0 | 18% | [8%, 30%] | 0.420 | 0.1255 |
| DROID-L3 | 15.5 | 22% | [12%, 34%] | 0.434 | 0.1389 |

- The six points are sorted by average episode duration, so the table can be read directly as a short-to-long horizon trend.
- The overall picture is downward: short DROID splits start around `30%`, while the longest AgiBot split ends at `16%`.

