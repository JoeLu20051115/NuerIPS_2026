# EgoDex Complexity-Aligned 50x3 Splits

This is a curated analysis split, not an unbiased random sample. The rule is explicit and reproducible.

| Split | N | Mean Time (s) | Mean Step Proxy | Mean Unique Ops | Stateful-op Rate | Task Mix |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| L1 | 50 | 2.56 | 2.00 | 2.00 | 0.00 | vertical_pick_place:34, basic_pick_place:13, pick_place_food:3 |
| L2 | 50 | 4.96 | 2.00 | 2.00 | 0.54 | insert_remove_furniture_bench_cabinet:6, stack_unstack_plates:15, add_remove_lid:3, insert_remove_usb:3, basic_pick_place:17, pick_place_food:6 |
| L3 | 50 | 17.27 | 3.00 | 3.00 | 1.00 | open_close_insert_remove_tupperware:25, insert_remove_furniture_bench_cabinet:9, add_remove_lid:12, insert_remove_usb:3, stack_unstack_plates:1 |
