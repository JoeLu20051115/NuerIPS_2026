# RoboTwin LingBot Benchmark

- Seed: `42`
- Quantile cutoffs: `Q33=147`, `Q66=243`
- Quota per level: `100`
- Total selected episodes: `300`

## Selected counts by level

- L1: `100` episodes across `24` tasks (min `1`, max `5` per task)
- L2: `100` episodes across `30` tasks (min `1`, max `4` per task)
- L3: `100` episodes across `21` tasks (min `4`, max `5` per task)

## Prompt sources

- lingbot_clean_prompt_bank: `300`

## Archive schema probes

- adjust_bottle / aloha-agilex_clean_50: `valid=True` `episode_length=140` `frame=320x240`
- adjust_bottle / aloha-agilex_randomized_500: `valid=True` `episode_length=142` `frame=320x240`
- beat_block_hammer / aloha-agilex_clean_50: `valid=True` `episode_length=126` `frame=320x240`
- beat_block_hammer / aloha-agilex_randomized_500: `valid=True` `episode_length=115` `frame=320x240`
- blocks_ranking_rgb / aloha-agilex_clean_50: `valid=True` `episode_length=479` `frame=320x240`
- blocks_ranking_rgb / aloha-agilex_randomized_500: `valid=True` `episode_length=448` `frame=320x240`
- blocks_ranking_size / aloha-agilex_clean_50: `valid=True` `episode_length=475` `frame=320x240`
- blocks_ranking_size / aloha-agilex_randomized_500: `valid=True` `episode_length=444` `frame=320x240`
- click_alarmclock / aloha-agilex_clean_50: `valid=True` `episode_length=77` `frame=320x240`
- click_alarmclock / aloha-agilex_randomized_500: `valid=True` `episode_length=80` `frame=320x240`
- click_bell / aloha-agilex_clean_50: `valid=True` `episode_length=81` `frame=320x240`
- click_bell / aloha-agilex_randomized_500: `valid=True` `episode_length=78` `frame=320x240`
