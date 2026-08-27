# RoboTwin LOGIV control-runtime patches

Apply these patches to TACO commit
`ee9e06dcf01d8b1a606b18841cb52b1ec88a423b` in the exact order below. The
numeric prefixes restart because the patches were exported at successive
development checkpoints; this list is authoritative. Applying the full stack
produces the control-runtime tree recorded at TACO commit
`8de0ed9520989f9fd156904291d0895b9a361886`.

1. `0001-fix-robotwin-parameterize-pi05-baseline-evaluation.patch`
2. `0001-feat-robotwin-add-LOGIV-PDDL-monitored-evaluation.patch`
3. `0002-fix-robotwin-monitor-unchanged-pi05-base-prefix.patch`
4. `0003-fix-robotwin-replay-frozen-LOGIV-scenes-directly.patch`
5. `0004-feat-robotwin-support-stage-specific-LOGIV-handoffs.patch`
6. `0001-feat-robotwin-protect-monitored-base-policy-window.patch`
7. `0005-feat-robotwin-complete-DAG-runtime.patch`
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
git clone https://github.com/breez3young/TACO.git
cd TACO
git checkout ee9e06dcf01d8b1a606b18841cb52b1ec88a423b
git am /path/to/NuerIPS_2026/patches/robotwin/0001-fix-robotwin-parameterize-pi05-baseline-evaluation.patch
# Continue with items 2 through 15 in the order above.
```

This clean patch stack ends at the live control path. Later development-only
patches that searched scene pools or persisted camera/API evidence are
intentionally excluded; they are not required to execute the frozen 100-cell
protocol.
