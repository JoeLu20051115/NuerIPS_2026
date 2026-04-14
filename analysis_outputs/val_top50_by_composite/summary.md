# VAL Top-50 Per Level by Composite Score

Composite definition used in each level: `0.45 * success + 0.35 * task_progress + 0.20 * normalized_l2_quality`, where `normalized_l2_quality` is min-max normalized within the same level and lower `mean_l2` gets higher score.

| Level | Available Matched | Selected | Avg Steps | Arm Profile | VAL Trigger | `val>dual` Count | `val>token` Count | token (L2 / TP / SR) | dual (L2 / TP / SR) | val (L2 / TP / SR) | Dominant Tasks |
|---|---:|---:|---:|---|---|---:|---:|---|---|---|---|
| L1 | 73 | 50 | 125.3 | both:9, right_only:20, left_only:21 | corrected_valid:5, ok:45 | 25 | 32 | 0.562 / 0.816 / 0.880 | 0.558 / 0.878 / 0.960 | 0.566 / 0.895 / 1.000 | pick_diverse_bottles x4, turn_switch x4, pick_dual_bottles x4, click_bell x4, stamp_seal x3, move_pillbottle_pad x3, move_can_pot x3, beat_block_hammer x3 |
| L2 | 80 | 50 | 174.9 | left_only:23, right_only:12, both:15 | ok:48, corrected_valid:2 | 24 | 22 | 0.544 / 0.847 / 0.920 | 0.556 / 0.844 / 0.900 | 0.571 / 0.900 / 1.000 | move_pillbottle_pad x4, open_laptop x3, pick_diverse_bottles x3, place_container_plate x3, place_burger_fries x3, place_fan x3, adjust_bottle x3, move_can_pot x3 |
| L3 | 69 | 50 | 344.0 | left_only:11, right_only:5, both:34 | ok:45, corrected_valid:3, corrected:2 | 27 | 23 | 0.522 / 0.801 / 0.860 | 0.531 / 0.762 / 0.800 | 0.545 / 0.842 / 0.920 | open_laptop x5, place_cans_plasticbox x4, put_object_cabinet x4, blocks_ranking_size x4, place_can_basket x4, open_microwave x3, stack_bowls_three x3, shake_bottle x3 |

## L1 Top 10

| Rank | Episode | Task | Arms | Steps | VAL Result | token (SR/TP/L2) | dual (SR/TP/L2) | val (SR/TP/L2) | Score |
|---:|---:|---|---|---:|---|---|---|---|---:|
| 1 | 57 | pick_diverse_bottles | both | 138 | corrected_valid | 1/0.95/0.398 | 1/0.95/0.453 | 1/0.95/0.419 | 0.980 |
| 2 | 85 | place_mouse_pad | right_only | 137 | ok | 1/0.95/0.494 | 1/0.95/0.490 | 1/0.95/0.490 | 0.969 |
| 3 | 40 | place_object_scale | left_only | 147 | ok | 1/0.95/0.502 | 1/0.95/0.505 | 1/0.95/0.499 | 0.967 |
| 4 | 23 | turn_switch | left_only | 95 | ok | 1/0.92/0.397 | 1/0.92/0.402 | 1/0.90/0.402 | 0.965 |
| 5 | 45 | stamp_seal | left_only | 138 | ok | 1/0.95/0.511 | 1/0.95/0.519 | 1/0.95/0.515 | 0.965 |
| 6 | 10 | open_laptop | left_only | 147 | ok | 1/0.82/0.424 | 1/0.82/0.422 | 1/0.90/0.417 | 0.963 |
| 7 | 69 | turn_switch | right_only | 105 | ok | 1/0.82/0.445 | 1/0.92/0.442 | 1/0.92/0.463 | 0.962 |
| 8 | 18 | place_object_stand | right_only | 136 | ok | 1/0.95/0.540 | 1/0.95/0.536 | 1/0.95/0.537 | 0.961 |
| 9 | 54 | move_pillbottle_pad | left_only | 139 | ok | 1/0.95/0.569 | 1/0.95/0.545 | 1/0.95/0.540 | 0.961 |
| 10 | 77 | move_pillbottle_pad | left_only | 146 | ok | 1/0.95/0.441 | 1/0.95/0.559 | 1/0.96/0.563 | 0.961 |

## L2 Top 10

| Rank | Episode | Task | Arms | Steps | VAL Result | token (SR/TP/L2) | dual (SR/TP/L2) | val (SR/TP/L2) | Score |
|---:|---:|---|---|---:|---|---|---|---|---:|
| 1 | 188 | move_pillbottle_pad | left_only | 152 | ok | 1/0.98/0.400 | 1/0.86/0.407 | 1/1.00/0.410 | 0.977 |
| 2 | 136 | open_laptop | left_only | 195 | ok | 1/0.88/0.318 | 1/0.92/0.374 | 1/0.95/0.305 | 0.975 |
| 3 | 163 | open_laptop | left_only | 184 | ok | 1/0.92/0.321 | 1/0.95/0.322 | 1/0.95/0.331 | 0.971 |
| 4 | 173 | place_empty_cup | left_only | 174 | ok | 1/0.96/0.405 | 1/0.95/0.399 | 1/0.95/0.393 | 0.962 |
| 5 | 190 | open_laptop | right_only | 231 | ok | 1/0.95/0.260 | 1/0.90/0.260 | 1/0.88/0.258 | 0.958 |
| 6 | 191 | pick_diverse_bottles | both | 148 | corrected_valid | 1/0.96/0.441 | 1/0.96/0.473 | 1/0.95/0.421 | 0.958 |
| 7 | 144 | place_container_plate | left_only | 162 | ok | 1/1.00/0.489 | 1/1.00/0.479 | 1/1.00/0.551 | 0.956 |
| 8 | 178 | place_object_stand | right_only | 150 | ok | 1/0.95/0.513 | 1/1.00/0.430 | 1/0.95/0.469 | 0.951 |
| 9 | 169 | place_burger_fries | both | 243 | ok | 1/0.92/0.489 | 1/0.95/0.486 | 1/0.95/0.476 | 0.950 |
| 10 | 161 | move_pillbottle_pad | left_only | 149 | ok | 1/0.95/0.453 | 1/0.95/0.509 | 1/0.95/0.492 | 0.947 |

## L3 Top 10

| Rank | Episode | Task | Arms | Steps | VAL Result | token (SR/TP/L2) | dual (SR/TP/L2) | val (SR/TP/L2) | Score |
|---:|---:|---|---|---:|---|---|---|---|---:|
| 1 | 227 | open_microwave | left_only | 756 | ok | 1/0.96/0.096 | 1/0.95/0.104 | 1/0.95/0.099 | 0.982 |
| 2 | 206 | open_microwave | left_only | 843 | ok | 1/1.00/0.081 | 1/0.85/0.152 | 1/0.95/0.165 | 0.973 |
| 3 | 248 | open_microwave | left_only | 588 | ok | 1/0.95/0.111 | 1/0.90/0.134 | 1/0.90/0.114 | 0.963 |
| 4 | 268 | open_laptop | left_only | 398 | ok | 1/0.93/0.193 | 1/0.95/0.178 | 1/0.92/0.179 | 0.960 |
| 5 | 226 | open_laptop | right_only | 260 | ok | 1/0.95/0.271 | 1/0.93/0.268 | 1/0.95/0.266 | 0.958 |
| 6 | 247 | open_laptop | left_only | 314 | ok | 1/0.90/0.206 | 1/0.92/0.218 | 1/0.92/0.214 | 0.955 |
| 7 | 205 | open_laptop | left_only | 256 | ok | 1/0.95/0.279 | 1/0.95/0.262 | 1/0.92/0.252 | 0.950 |
| 8 | 294 | place_cans_plasticbox | both | 287 | ok | 1/0.95/0.475 | 1/0.98/0.479 | 1/1.00/0.465 | 0.946 |
| 9 | 289 | open_laptop | right_only | 261 | ok | 1/0.88/0.266 | 1/0.95/0.250 | 1/0.88/0.250 | 0.936 |
| 10 | 240 | stack_bowls_three | both | 490 | ok | 1/0.95/1.096 | 1/0.85/0.351 | 1/0.92/0.362 | 0.933 |

