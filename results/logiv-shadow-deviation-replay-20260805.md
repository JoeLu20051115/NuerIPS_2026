# Shadow confirmed-deviation offline replay

本报告只重放拓扑状态，不执行动作；拓扑候选不能单独升级为主动接管。

- 来源：100 cases
- 需要复核：20 cases
- Confirmed deviation：0

| Case | Task | Seed | Base | 首次 STALE | 剩余步数 | 后续拓扑进展 | 回溯标签 |
|---|---:|---:|---|---:|---:|---|---|
| t03-r02 | 3 | 4080838150 | 成功 | 171 | 64 | 是 | TRANSIENT_OR_SELF_RECOVERED |
| t03-r07 | 3 | 2172042871 | 成功 | 97 | 131 | 是 | TRANSIENT_OR_SELF_RECOVERED |
| t04-r04 | 4 | 2024286983 | 成功 | 106 | 125 | 是 | TRANSIENT_OR_SELF_RECOVERED |
| t05-r02 | 5 | 3095785511 | 失败 | None | None | 否 | STALL_WITHOUT_STALE |
| t05-r04 | 5 | 1785219942 | 失败 | 176 | 0 | 否 | PERSISTENT_FAILURE_CANDIDATE |
| t05-r05 | 5 | 958658519 | 失败 | 252 | 0 | 否 | PERSISTENT_FAILURE_CANDIDATE |
| t05-r07 | 5 | 1017983067 | 失败 | 177 | 0 | 否 | PERSISTENT_FAILURE_CANDIDATE |
| t05-r09 | 5 | 2219472439 | 失败 | 177 | 0 | 否 | PERSISTENT_FAILURE_CANDIDATE |
| t06-r02 | 6 | 2630018707 | 成功 | 121 | 87 | 是 | TRANSIENT_OR_SELF_RECOVERED |
| t06-r03 | 6 | 1568265727 | 成功 | 46 | 272 | 是 | TRANSIENT_OR_SELF_RECOVERED |
| t06-r05 | 6 | 398462197 | 成功 | 48 | 173 | 是 | TRANSIENT_OR_SELF_RECOVERED |
| t07-r08 | 7 | 1775485456 | 失败 | None | None | 否 | STALL_WITHOUT_STALE |
| t08-r00 | 8 | 1861017279 | 失败 | 217 | 303 | 否 | PERSISTENT_FAILURE_CANDIDATE |
| t08-r01 | 8 | 3353120377 | 失败 | 330 | 190 | 否 | PERSISTENT_FAILURE_CANDIDATE |
| t08-r02 | 8 | 2382481579 | 失败 | None | None | 否 | STALL_WITHOUT_STALE |
| t08-r04 | 8 | 3815632364 | 成功 | 116 | 388 | 是 | TRANSIENT_OR_SELF_RECOVERED |
| t08-r06 | 8 | 1992582348 | 成功 | 278 | 160 | 是 | TRANSIENT_OR_SELF_RECOVERED |
| t08-r07 | 8 | 3090199507 | 成功 | 151 | 294 | 是 | TRANSIENT_OR_SELF_RECOVERED |
| t09-r03 | 9 | 2948309517 | 失败 | 247 | 273 | 否 | PERSISTENT_FAILURE_CANDIDATE |
| t09-r09 | 9 | 2477098108 | 成功 | 115 | 139 | 是 | TRANSIENT_OR_SELF_RECOVERED |

## 接管约束

`CURRENT → STALE`、长期无拓扑进展以及 `PRECONDITION_UNKNOWN` 均只能产生候选。
真正的 `CONFIRMED_DEVIATION` 还必须具有动作归因后的强证据，例如 `ATTEMPTED_EFFECT_TIMEOUT`、`ABNORMAL_TRANSFER_AFTER_MANIPULATION` 或 `GOAL_REGRESSION`。
