# VAL > DUAL > TOKEN Candidate Summary

Selection rule: lexicographic by `success > task_progress > lower mean_l2`, requiring `llm_val > dual_llm > task_token_only` on the same episode.

Matched episodes available now: 217
Qualified episodes now: 36

| Level | Qualified Now | Requested | Avg Steps | Arm Profile | VAL Trigger | Mean TP token/dual/val | Mean SR token/dual/val | Mean L2 token/dual/val | Dominant Tasks |
|---|---:|---:|---:|---|---|---|---|---|---|
| L1 | 15 | 50 | 130.3 | right_only:8, left_only:6, both:1 | ok:15 | 0.650 / 0.797 / 0.855 | 0.667 / 0.867 / 0.933 | 0.678 / 0.603 / 0.666 | press_stapler x2, place_fan x2, place_mouse_pad x1, beat_block_hammer x1, move_pillbottle_pad x1 |
| L2 | 11 | 50 | 174.5 | right_only:5, both:2, left_only:4 | ok:11 | 0.673 / 0.765 / 0.794 | 0.727 / 0.818 / 0.818 | 0.648 / 0.694 / 0.683 | place_a2b_right x2, place_burger_fries x2, place_fan x1, dump_bin_bigbin x1, place_mouse_pad x1 |
| L3 | 10 | 50 | 288.8 | both:8, left_only:2 | ok:8, corrected:1, corrected_valid:1 | 0.315 / 0.521 / 0.679 | 0.200 / 0.500 / 0.700 | 0.740 / 0.779 / 0.678 | handover_block x2, place_cans_plasticbox x2, stack_blocks_two x1, open_laptop x1, put_object_cabinet x1 |

## L1 Top Candidates (15 available)

| Episode | Task | Arms | Steps | VAL Result | token (SR/TP/L2) | dual (SR/TP/L2) | val (SR/TP/L2) |
|---:|---|---|---:|---|---|---|---|
| 74 | grab_roller | both | 94 | ok | 0/0.15/0.955 | 0/0.20/0.988 | 1/0.85/0.968 |
| 47 | adjust_bottle | left_only | 142 | ok | 1/0.82/0.596 | 1/0.82/0.570 | 1/0.88/0.611 |
| 37 | place_a2b_right | right_only | 144 | ok | 1/0.85/0.489 | 1/0.88/0.490 | 1/0.92/0.496 |
| 76 | move_can_pot | right_only | 144 | ok | 0/0.15/0.580 | 1/0.82/0.603 | 1/0.86/0.629 |
| 79 | move_stapler_pad | right_only | 146 | ok | 1/0.92/0.787 | 1/0.92/0.778 | 1/0.95/0.778 |
| 41 | place_object_stand | right_only | 141 | ok | 0/0.18/0.509 | 1/0.86/0.521 | 1/0.88/0.516 |
| 22 | stamp_seal | left_only | 142 | ok | 0/0.20/0.507 | 1/0.86/0.503 | 1/0.88/0.514 |
| 66 | press_stapler | right_only | 129 | ok | 1/0.90/1.547 | 1/0.92/0.404 | 1/0.93/1.363 |
| 72 | click_alarmclock | left_only | 91 | ok | 1/0.88/0.602 | 1/0.92/0.633 | 1/0.92/0.606 |
| 38 | place_fan | left_only | 134 | ok | 0/0.18/0.783 | 0/0.20/0.804 | 0/0.20/0.783 |

## L2 Top Candidates (11 available)

| Episode | Task | Arms | Steps | VAL Result | token (SR/TP/L2) | dual (SR/TP/L2) | val (SR/TP/L2) |
|---:|---|---|---:|---|---|---|---|
| 130 | adjust_bottle | left_only | 148 | ok | 1/0.82/0.502 | 1/0.86/0.508 | 1/0.92/0.493 |
| 126 | rotate_qrcode | right_only | 157 | ok | 1/0.82/0.938 | 1/0.82/0.914 | 1/0.88/0.927 |
| 113 | place_burger_fries | both | 241 | ok | 1/0.95/0.828 | 1/0.95/0.816 | 1/1.00/0.802 |
| 125 | press_stapler | right_only | 168 | ok | 0/0.18/0.725 | 1/0.90/1.316 | 1/0.95/1.340 |
| 136 | open_laptop | left_only | 195 | ok | 1/0.88/0.318 | 1/0.92/0.374 | 1/0.95/0.305 |
| 148 | place_mouse_pad | right_only | 150 | ok | 1/0.88/0.757 | 1/0.92/0.742 | 1/0.95/0.726 |
| 131 | dump_bin_bigbin | left_only | 162 | ok | 1/0.82/0.500 | 1/0.88/0.480 | 1/0.90/0.484 |
| 147 | place_fan | left_only | 154 | ok | 1/0.88/0.525 | 1/0.92/0.525 | 1/0.93/0.504 |
| 169 | place_burger_fries | both | 243 | ok | 1/0.92/0.489 | 1/0.95/0.486 | 1/0.95/0.476 |
| 110 | place_a2b_right | right_only | 150 | ok | 0/0.10/0.535 | 0/0.15/0.544 | 0/0.15/0.537 |

## L3 Top Candidates (10 available)

| Episode | Task | Arms | Steps | VAL Result | token (SR/TP/L2) | dual (SR/TP/L2) | val (SR/TP/L2) |
|---:|---|---|---:|---|---|---|---|
| 252 | place_cans_plasticbox | both | 290 | ok | 0/0.15/0.478 | 0/0.15/0.452 | 1/0.86/0.472 |
| 278 | shake_bottle | left_only | 255 | ok | 0/0.10/0.348 | 0/0.15/0.349 | 1/0.85/0.360 |
| 293 | place_can_basket | both | 268 | ok | 0/0.00/0.569 | 1/0.82/0.756 | 1/0.92/0.753 |
| 202 | dump_bin_bigbin | both | 324 | ok | 0/0.10/0.447 | 0/0.10/0.424 | 0/0.15/0.394 |
| 294 | place_cans_plasticbox | both | 287 | ok | 1/0.95/0.475 | 1/0.98/0.479 | 1/1.00/0.465 |
| 287 | handover_block | both | 278 | corrected_valid | 0/0.10/1.431 | 0/0.15/1.598 | 0/0.15/1.083 |
| 298 | put_object_cabinet | both | 275 | ok | 0/0.55/0.721 | 1/0.92/0.756 | 1/0.92/0.461 |
| 266 | handover_block | both | 275 | corrected | 0/0.15/1.257 | 0/0.20/1.298 | 0/0.20/1.115 |
| 247 | open_laptop | left_only | 314 | ok | 1/0.90/0.206 | 1/0.92/0.218 | 1/0.92/0.214 |
| 218 | stack_blocks_two | both | 322 | ok | 0/0.15/1.463 | 1/0.82/1.461 | 1/0.82/1.459 |

