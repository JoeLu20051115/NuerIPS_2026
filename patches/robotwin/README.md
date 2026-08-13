# RoboTwin LOGIV runtime patches

Apply these patches to the TACO checkout in the following order. The numeric
prefixes restart because the files were exported at successive experiment
checkpoints; the order below is authoritative.

1. `0001-fix-robotwin-parameterize-pi05-baseline-evaluation.patch`
2. `0001-feat-robotwin-add-LOGIV-PDDL-monitored-evaluation.patch`
3. `0002-fix-robotwin-monitor-unchanged-pi05-base-prefix.patch`
4. `0003-fix-robotwin-replay-frozen-LOGIV-scenes-directly.patch`
5. `0004-feat-robotwin-support-stage-specific-LOGIV-handoffs.patch`
6. `0001-feat-robotwin-protect-monitored-base-policy-window.patch`
7. `0005-feat-robotwin-complete-DAG-runtime.patch` (six commits)
8. `0001-fix-robotwin-resample-local-repair-actions-from-fres.patch`
9. `0002-fix-robotwin-collect-new-evidence-before-unknown-nod.patch`
10. `0003-fix-robotwin-advance-time-for-unknown-evidence-retries.patch`
11. `0004-feat-robotwin-add-independent-VLM-observer-evidence.patch`
12. `0001-feat-robotwin-route-PDDL-repair-through-bundled-CFN.patch`
13. `0002-feat-robotwin-verify-every-PDDL-node-with-bundled-CF.patch`
14. `0003-feat-robotwin-support-frozen-prompt-node-replanning.patch`
15. `0004-feat-robotwin-execute-registered-PDDL-node-prompts.patch`

Example:

```bash
git am /path/to/NuerIPS_2026/patches/robotwin/0001-fix-robotwin-parameterize-pi05-baseline-evaluation.patch
# Continue with items 2--15 in the order above.
```

The final four patches correspond to TACO commits `19b9f03`, `2d3fc45`,
`0743cc3`, and `8de0ed9`. The first three are optional for historical
configurations that did not enable CFN or node-internal replanning. The final
patch adds the explicit registered-node prompt switch used by the strict
`turn_switch` development cell.
