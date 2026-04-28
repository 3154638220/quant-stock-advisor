# R2B Tradable Upside Replacement v1

- 生成时间：`2026-04-28T06:54:52.699119+00:00`
- 配置快照：`/mnt/ssd/lh/config.yaml.backtest`
- `eval_contract_version`: `r0_eval_execution_contract_2026-04-28`
- `execution_contract_version`: `tplus1_open_buy_delta_limit_mask_2026-04-28`
- 组合表达：`S2 defensive Top-20 + capped replacement slots`，不再使用完整 upside Top-20 sleeve。
- 状态输入：上一已完成月份 `regime/breadth`；仅 `strong_up` 或 `wide` 允许 replacement gate。
- primary benchmark：`open_to_open`；promotion metric：`daily_bt_like_proxy_annualized_excess_vs_market`。

## 1. Leaderboard

| candidate_id                                     |   daily_proxy_annualized_excess_vs_market |   delta_vs_baseline_proxy | gate_decision   |   strong_up_median_excess |   delta_vs_baseline_strong_up_median_excess |   strong_up_positive_share |   delta_vs_baseline_strong_up_positive_share |   strong_down_median_excess |   delta_vs_baseline_strong_down_median_excess |   strong_up_switch_in_minus_out |   avg_turnover_half_l1 |   delta_vs_baseline_turnover |   n_rebalances |
|:-------------------------------------------------|------------------------------------------:|--------------------------:|:----------------|--------------------------:|--------------------------------------------:|---------------------------:|---------------------------------------------:|----------------------------:|----------------------------------------------:|--------------------------------:|-----------------------:|-----------------------------:|---------------:|
| BASELINE_S2_FIXED                                |                                   -0.0859 |                    0.0000 | reject          |                   -0.0612 |                                      0.0000 |                     0.1538 |                                       0.0000 |                      0.0629 |                                        0.0000 |                         -0.0237 |                 0.0844 |                       0.0000 |             64 |
| U2_C_s2_residual_elasticity__R2B_OVERLAY_10      |                                   -0.0887 |                   -0.0028 | reject          |                   -0.0612 |                                      0.0000 |                     0.1538 |                                       0.0000 |                      0.0639 |                                        0.0010 |                          0.0130 |                 0.1036 |                       0.0193 |             64 |
| U2_A_industry_breadth_strength__R2B_OVERLAY_10   |                                   -0.0892 |                   -0.0033 | reject          |                   -0.0598 |                                      0.0014 |                     0.1538 |                                       0.0000 |                      0.0595 |                                       -0.0034 |                          0.0162 |                 0.1068 |                       0.0225 |             64 |
| U2_A_industry_breadth_strength__R2B_REPLACE_3    |                                   -0.0907 |                   -0.0048 | reject          |                   -0.0596 |                                      0.0016 |                     0.1538 |                                       0.0000 |                      0.0589 |                                       -0.0040 |                          0.0313 |                 0.1410 |                       0.0567 |             64 |
| U2_A_industry_breadth_strength__R2B_REPLACE_5    |                                   -0.0918 |                   -0.0059 | reject          |                   -0.0596 |                                      0.0016 |                     0.1538 |                                       0.0000 |                      0.0542 |                                       -0.0088 |                          0.0154 |                 0.2013 |                       0.1169 |             64 |
| U2_C_s2_residual_elasticity__R2B_REPLACE_5       |                                   -0.0935 |                   -0.0075 | reject          |                   -0.0601 |                                      0.0011 |                     0.1538 |                                       0.0000 |                      0.0606 |                                       -0.0023 |                          0.0051 |                 0.2077 |                       0.1233 |             64 |
| U2_C_s2_residual_elasticity__R2B_REPLACE_3       |                                   -0.0968 |                   -0.0108 | reject          |                   -0.0626 |                                     -0.0014 |                     0.1538 |                                       0.0000 |                      0.0581 |                                       -0.0049 |                          0.0079 |                 0.1372 |                       0.0528 |             64 |
| U2_B_tradable_breakout_expansion__R2B_OVERLAY_10 |                                   -0.1012 |                   -0.0153 | reject          |                   -0.0612 |                                      0.0000 |                     0.1538 |                                       0.0000 |                      0.0559 |                                       -0.0070 |                          0.0049 |                 0.1036 |                       0.0193 |             64 |
| U2_B_tradable_breakout_expansion__R2B_REPLACE_3  |                                   -0.1089 |                   -0.0230 | reject          |                   -0.0606 |                                      0.0006 |                     0.1538 |                                       0.0000 |                      0.0410 |                                       -0.0219 |                         -0.0011 |                 0.1372 |                       0.0528 |             64 |
| U2_B_tradable_breakout_expansion__R2B_REPLACE_5  |                                   -0.1245 |                   -0.0386 | reject          |                   -0.0612 |                                      0.0000 |                     0.1538 |                                       0.0000 |                      0.0452 |                                       -0.0177 |                          0.0047 |                 0.2077 |                       0.1233 |             64 |

## 2. R2B 验收

| candidate_id                                     | status   | daily_proxy_not_below_baseline   | daily_proxy_>=_0   | strong_up_median_+2pct_or_positive_share_+10pct   | strong_down_not_worse_than_-2pct   | turnover_delta_<=_+0.10   | strong_up_switch_not_negative   |
|:-------------------------------------------------|:---------|:---------------------------------|:-------------------|:--------------------------------------------------|:-----------------------------------|:--------------------------|:--------------------------------|
| BASELINE_S2_FIXED                                | baseline | nan                              | nan                | nan                                               | nan                                | nan                       | nan                             |
| U2_C_s2_residual_elasticity__R2B_OVERLAY_10      | fail     | fail                             | fail               | fail                                              | pass                               | pass                      | pass                            |
| U2_A_industry_breadth_strength__R2B_OVERLAY_10   | fail     | fail                             | fail               | fail                                              | pass                               | pass                      | pass                            |
| U2_A_industry_breadth_strength__R2B_REPLACE_3    | fail     | fail                             | fail               | fail                                              | pass                               | pass                      | pass                            |
| U2_A_industry_breadth_strength__R2B_REPLACE_5    | fail     | fail                             | fail               | fail                                              | pass                               | fail                      | pass                            |
| U2_C_s2_residual_elasticity__R2B_REPLACE_5       | fail     | fail                             | fail               | fail                                              | pass                               | fail                      | pass                            |
| U2_C_s2_residual_elasticity__R2B_REPLACE_3       | fail     | fail                             | fail               | fail                                              | pass                               | pass                      | pass                            |
| U2_B_tradable_breakout_expansion__R2B_OVERLAY_10 | fail     | fail                             | fail               | fail                                              | pass                               | pass                      | pass                            |
| U2_B_tradable_breakout_expansion__R2B_REPLACE_3  | fail     | fail                             | fail               | fail                                              | fail                               | pass                      | fail                            |
| U2_B_tradable_breakout_expansion__R2B_REPLACE_5  | fail     | fail                             | fail               | fail                                              | pass                               | fail                      | pass                            |

## 3. Replacement 触发统计

| candidate_id                                     | replacement_allowed   |   rebalances |   avg_replacement_count |   max_replacement_count |   active_rebalance_share |
|:-------------------------------------------------|:----------------------|-------------:|------------------------:|------------------------:|-------------------------:|
| U2_A_industry_breadth_strength__R2B_OVERLAY_10   | False                 |           47 |                  0.0000 |                       0 |                   0.0000 |
| U2_A_industry_breadth_strength__R2B_OVERLAY_10   | True                  |           17 |                  4.5882 |                       5 |                   1.0000 |
| U2_A_industry_breadth_strength__R2B_REPLACE_3    | False                 |           47 |                  0.0000 |                       0 |                   0.0000 |
| U2_A_industry_breadth_strength__R2B_REPLACE_3    | True                  |           17 |                  3.0000 |                       3 |                   1.0000 |
| U2_A_industry_breadth_strength__R2B_REPLACE_5    | False                 |           47 |                  0.0000 |                       0 |                   0.0000 |
| U2_A_industry_breadth_strength__R2B_REPLACE_5    | True                  |           17 |                  4.5882 |                       5 |                   1.0000 |
| U2_B_tradable_breakout_expansion__R2B_OVERLAY_10 | False                 |           47 |                  0.0000 |                       0 |                   0.0000 |
| U2_B_tradable_breakout_expansion__R2B_OVERLAY_10 | True                  |           17 |                  5.0000 |                       5 |                   1.0000 |
| U2_B_tradable_breakout_expansion__R2B_REPLACE_3  | False                 |           47 |                  0.0000 |                       0 |                   0.0000 |
| U2_B_tradable_breakout_expansion__R2B_REPLACE_3  | True                  |           17 |                  3.0000 |                       3 |                   1.0000 |
| U2_B_tradable_breakout_expansion__R2B_REPLACE_5  | False                 |           47 |                  0.0000 |                       0 |                   0.0000 |
| U2_B_tradable_breakout_expansion__R2B_REPLACE_5  | True                  |           17 |                  5.0000 |                       5 |                   1.0000 |
| U2_C_s2_residual_elasticity__R2B_OVERLAY_10      | False                 |           47 |                  0.0000 |                       0 |                   0.0000 |
| U2_C_s2_residual_elasticity__R2B_OVERLAY_10      | True                  |           17 |                  5.0000 |                       5 |                   1.0000 |
| U2_C_s2_residual_elasticity__R2B_REPLACE_3       | False                 |           47 |                  0.0000 |                       0 |                   0.0000 |
| U2_C_s2_residual_elasticity__R2B_REPLACE_3       | True                  |           17 |                  3.0000 |                       3 |                   1.0000 |
| U2_C_s2_residual_elasticity__R2B_REPLACE_5       | False                 |           47 |                  0.0000 |                       0 |                   0.0000 |
| U2_C_s2_residual_elasticity__R2B_REPLACE_5       | True                  |           17 |                  5.0000 |                       5 |                   1.0000 |

## 4. Regime / Breadth 切片

| candidate_id                                     | regime      |   months |   median_benchmark_return |   median_strategy_return |   median_excess_return |   positive_excess_share |   benchmark_compound |   strategy_compound |   capture_ratio |
|:-------------------------------------------------|:------------|---------:|--------------------------:|-------------------------:|-----------------------:|------------------------:|---------------------:|--------------------:|----------------:|
| BASELINE_S2_FIXED                                | strong_down |       13 |                   -0.0604 |                  -0.0207 |                 0.0629 |                  0.7692 |              -0.6375 |             -0.2342 |          0.3674 |
| BASELINE_S2_FIXED                                | mild_down   |       13 |                   -0.0109 |                   0.0014 |                 0.0210 |                  0.7692 |              -0.1702 |              0.0771 |         -0.4527 |
| BASELINE_S2_FIXED                                | neutral     |       12 |                    0.0087 |                   0.0010 |                -0.0105 |                  0.4167 |               0.1358 |              0.0213 |          0.1567 |
| BASELINE_S2_FIXED                                | mild_up     |       13 |                    0.0378 |                   0.0121 |                -0.0258 |                  0.0769 |               0.6459 |              0.1512 |          0.2342 |
| BASELINE_S2_FIXED                                | strong_up   |       13 |                    0.0811 |                   0.0199 |                -0.0612 |                  0.1538 |               2.5282 |              0.6416 |          0.2538 |
| U2_A_industry_breadth_strength__R2B_REPLACE_3    | strong_down |       13 |                   -0.0604 |                  -0.0210 |                 0.0589 |                  0.7692 |              -0.6375 |             -0.2338 |          0.3668 |
| U2_A_industry_breadth_strength__R2B_REPLACE_3    | mild_down   |       13 |                   -0.0109 |                  -0.0040 |                 0.0063 |                  0.7692 |              -0.1702 |              0.0447 |         -0.2628 |
| U2_A_industry_breadth_strength__R2B_REPLACE_3    | neutral     |       12 |                    0.0087 |                   0.0010 |                -0.0106 |                  0.4167 |               0.1358 |             -0.0111 |         -0.0817 |
| U2_A_industry_breadth_strength__R2B_REPLACE_3    | mild_up     |       13 |                    0.0378 |                   0.0087 |                -0.0258 |                  0.0769 |               0.6459 |              0.1322 |          0.2047 |
| U2_A_industry_breadth_strength__R2B_REPLACE_3    | strong_up   |       13 |                    0.0811 |                   0.0402 |                -0.0596 |                  0.1538 |               2.5282 |              0.7109 |          0.2812 |
| U2_A_industry_breadth_strength__R2B_REPLACE_5    | strong_down |       13 |                   -0.0604 |                  -0.0212 |                 0.0542 |                  0.7692 |              -0.6375 |             -0.2567 |          0.4027 |
| U2_A_industry_breadth_strength__R2B_REPLACE_5    | mild_down   |       13 |                   -0.0109 |                  -0.0042 |                 0.0064 |                  0.7692 |              -0.1702 |              0.0868 |         -0.5098 |
| U2_A_industry_breadth_strength__R2B_REPLACE_5    | neutral     |       12 |                    0.0087 |                  -0.0021 |                -0.0174 |                  0.3333 |               0.1358 |             -0.0313 |         -0.2305 |
| U2_A_industry_breadth_strength__R2B_REPLACE_5    | mild_up     |       13 |                    0.0378 |                   0.0091 |                -0.0258 |                  0.0769 |               0.6459 |              0.1531 |          0.2370 |
| U2_A_industry_breadth_strength__R2B_REPLACE_5    | strong_up   |       13 |                    0.0811 |                   0.0263 |                -0.0596 |                  0.1538 |               2.5282 |              0.6808 |          0.2693 |
| U2_A_industry_breadth_strength__R2B_OVERLAY_10   | strong_down |       13 |                   -0.0604 |                  -0.0209 |                 0.0595 |                  0.7692 |              -0.6375 |             -0.2444 |          0.3834 |
| U2_A_industry_breadth_strength__R2B_OVERLAY_10   | mild_down   |       13 |                   -0.0109 |                  -0.0014 |                 0.0177 |                  0.7692 |              -0.1702 |              0.0795 |         -0.4673 |
| U2_A_industry_breadth_strength__R2B_OVERLAY_10   | neutral     |       12 |                    0.0087 |                   0.0010 |                -0.0106 |                  0.4167 |               0.1358 |             -0.0017 |         -0.0123 |
| U2_A_industry_breadth_strength__R2B_OVERLAY_10   | mild_up     |       13 |                    0.0378 |                   0.0088 |                -0.0258 |                  0.0769 |               0.6459 |              0.1522 |          0.2356 |
| U2_A_industry_breadth_strength__R2B_OVERLAY_10   | strong_up   |       13 |                    0.0811 |                   0.0290 |                -0.0598 |                  0.1538 |               2.5282 |              0.6556 |          0.2593 |
| U2_B_tradable_breakout_expansion__R2B_REPLACE_3  | strong_down |       13 |                   -0.0604 |                  -0.0255 |                 0.0410 |                  0.6923 |              -0.6375 |             -0.3006 |          0.4716 |
| U2_B_tradable_breakout_expansion__R2B_REPLACE_3  | mild_down   |       13 |                   -0.0109 |                  -0.0087 |                 0.0057 |                  0.7692 |              -0.1702 |              0.0416 |         -0.2443 |
| U2_B_tradable_breakout_expansion__R2B_REPLACE_3  | neutral     |       12 |                    0.0087 |                   0.0010 |                -0.0101 |                  0.4167 |               0.1358 |              0.0052 |          0.0385 |
| U2_B_tradable_breakout_expansion__R2B_REPLACE_3  | mild_up     |       13 |                    0.0378 |                   0.0121 |                -0.0258 |                  0.0769 |               0.6459 |              0.1614 |          0.2499 |
| U2_B_tradable_breakout_expansion__R2B_REPLACE_3  | strong_up   |       13 |                    0.0811 |                   0.0237 |                -0.0606 |                  0.1538 |               2.5282 |              0.6239 |          0.2468 |
| U2_B_tradable_breakout_expansion__R2B_REPLACE_5  | strong_down |       13 |                   -0.0604 |                  -0.0213 |                 0.0452 |                  0.6923 |              -0.6375 |             -0.3333 |          0.5228 |
| U2_B_tradable_breakout_expansion__R2B_REPLACE_5  | mild_down   |       13 |                   -0.0109 |                  -0.0135 |                 0.0057 |                  0.6154 |              -0.1702 |             -0.0304 |          0.1787 |
| U2_B_tradable_breakout_expansion__R2B_REPLACE_5  | neutral     |       12 |                    0.0087 |                   0.0028 |                -0.0049 |                  0.3333 |               0.1358 |              0.0016 |          0.0118 |
| U2_B_tradable_breakout_expansion__R2B_REPLACE_5  | mild_up     |       13 |                    0.0378 |                   0.0123 |                -0.0258 |                  0.0769 |               0.6459 |              0.1912 |          0.2960 |
| U2_B_tradable_breakout_expansion__R2B_REPLACE_5  | strong_up   |       13 |                    0.0811 |                   0.0199 |                -0.0612 |                  0.1538 |               2.5282 |              0.6280 |          0.2484 |
| U2_B_tradable_breakout_expansion__R2B_OVERLAY_10 | strong_down |       13 |                   -0.0604 |                  -0.0209 |                 0.0559 |                  0.6923 |              -0.6375 |             -0.2752 |          0.4317 |
| U2_B_tradable_breakout_expansion__R2B_OVERLAY_10 | mild_down   |       13 |                   -0.0109 |                  -0.0045 |                 0.0107 |                  0.7692 |              -0.1702 |              0.0336 |         -0.1971 |
| U2_B_tradable_breakout_expansion__R2B_OVERLAY_10 | neutral     |       12 |                    0.0087 |                   0.0010 |                -0.0079 |                  0.4167 |               0.1358 |              0.0136 |          0.0998 |
| U2_B_tradable_breakout_expansion__R2B_OVERLAY_10 | mild_up     |       13 |                    0.0378 |                   0.0121 |                -0.0258 |                  0.0769 |               0.6459 |              0.1674 |          0.2592 |
| U2_B_tradable_breakout_expansion__R2B_OVERLAY_10 | strong_up   |       13 |                    0.0811 |                   0.0199 |                -0.0612 |                  0.1538 |               2.5282 |              0.6365 |          0.2518 |
| U2_C_s2_residual_elasticity__R2B_REPLACE_3       | strong_down |       13 |                   -0.0604 |                  -0.0182 |                 0.0581 |                  0.7692 |              -0.6375 |             -0.2560 |          0.4016 |
| U2_C_s2_residual_elasticity__R2B_REPLACE_3       | mild_down   |       13 |                   -0.0109 |                   0.0000 |                 0.0195 |                  0.7692 |              -0.1702 |              0.0343 |         -0.2014 |
| U2_C_s2_residual_elasticity__R2B_REPLACE_3       | neutral     |       12 |                    0.0087 |                   0.0010 |                -0.0106 |                  0.4167 |               0.1358 |              0.0182 |          0.1337 |
| U2_C_s2_residual_elasticity__R2B_REPLACE_3       | mild_up     |       13 |                    0.0378 |                   0.0087 |                -0.0258 |                  0.0769 |               0.6459 |              0.1585 |          0.2454 |
| U2_C_s2_residual_elasticity__R2B_REPLACE_3       | strong_up   |       13 |                    0.0811 |                   0.0199 |                -0.0626 |                  0.1538 |               2.5282 |              0.6366 |          0.2518 |
| U2_C_s2_residual_elasticity__R2B_REPLACE_5       | strong_down |       13 |                   -0.0604 |                  -0.0211 |                 0.0606 |                  0.7692 |              -0.6375 |             -0.2507 |          0.3932 |
| U2_C_s2_residual_elasticity__R2B_REPLACE_5       | mild_down   |       13 |                   -0.0109 |                  -0.0074 |                 0.0098 |                  0.7692 |              -0.1702 |              0.0050 |         -0.0293 |
| U2_C_s2_residual_elasticity__R2B_REPLACE_5       | neutral     |       12 |                    0.0087 |                   0.0010 |                -0.0072 |                  0.4167 |               0.1358 |              0.0219 |          0.1612 |
| U2_C_s2_residual_elasticity__R2B_REPLACE_5       | mild_up     |       13 |                    0.0378 |                   0.0107 |                -0.0258 |                  0.0769 |               0.6459 |              0.1727 |          0.2674 |
| U2_C_s2_residual_elasticity__R2B_REPLACE_5       | strong_up   |       13 |                    0.0811 |                   0.0242 |                -0.0601 |                  0.1538 |               2.5282 |              0.6668 |          0.2637 |
| U2_C_s2_residual_elasticity__R2B_OVERLAY_10      | strong_down |       13 |                   -0.0604 |                  -0.0209 |                 0.0639 |                  0.7692 |              -0.6375 |             -0.2406 |          0.3775 |
| U2_C_s2_residual_elasticity__R2B_OVERLAY_10      | mild_down   |       13 |                   -0.0109 |                   0.0000 |                 0.0194 |                  0.7692 |              -0.1702 |              0.0481 |         -0.2824 |
| U2_C_s2_residual_elasticity__R2B_OVERLAY_10      | neutral     |       12 |                    0.0087 |                   0.0010 |                -0.0092 |                  0.4167 |               0.1358 |              0.0216 |          0.1590 |
| U2_C_s2_residual_elasticity__R2B_OVERLAY_10      | mild_up     |       13 |                    0.0378 |                   0.0121 |                -0.0258 |                  0.0769 |               0.6459 |              0.1600 |          0.2478 |
| U2_C_s2_residual_elasticity__R2B_OVERLAY_10      | strong_up   |       13 |                    0.0811 |                   0.0238 |                -0.0612 |                  0.1538 |               2.5282 |              0.6519 |          0.2579 |

| candidate_id                                     | breadth   |   months |   median_benchmark_return |   median_strategy_return |   median_excess_return |   positive_excess_share |
|:-------------------------------------------------|:----------|---------:|--------------------------:|-------------------------:|-----------------------:|------------------------:|
| BASELINE_S2_FIXED                                | narrow    |       19 |                   -0.0345 |                  -0.0137 |                 0.0249 |                  0.7895 |
| BASELINE_S2_FIXED                                | mid       |       26 |                    0.0150 |                   0.0064 |                -0.0043 |                  0.4231 |
| BASELINE_S2_FIXED                                | wide      |       19 |                    0.0678 |                   0.0121 |                -0.0470 |                  0.1053 |
| U2_A_industry_breadth_strength__R2B_REPLACE_3    | narrow    |       19 |                   -0.0345 |                  -0.0137 |                 0.0237 |                  0.7895 |
| U2_A_industry_breadth_strength__R2B_REPLACE_3    | mid       |       26 |                    0.0150 |                   0.0061 |                -0.0044 |                  0.4231 |
| U2_A_industry_breadth_strength__R2B_REPLACE_3    | wide      |       19 |                    0.0678 |                   0.0088 |                -0.0441 |                  0.1053 |
| U2_A_industry_breadth_strength__R2B_REPLACE_5    | narrow    |       19 |                   -0.0345 |                  -0.0137 |                 0.0210 |                  0.7895 |
| U2_A_industry_breadth_strength__R2B_REPLACE_5    | mid       |       26 |                    0.0150 |                   0.0060 |                -0.0108 |                  0.3846 |
| U2_A_industry_breadth_strength__R2B_REPLACE_5    | wide      |       19 |                    0.0678 |                   0.0087 |                -0.0580 |                  0.1053 |
| U2_A_industry_breadth_strength__R2B_OVERLAY_10   | narrow    |       19 |                   -0.0345 |                  -0.0137 |                 0.0221 |                  0.7895 |
| U2_A_industry_breadth_strength__R2B_OVERLAY_10   | mid       |       26 |                    0.0150 |                   0.0062 |                -0.0044 |                  0.4231 |
| U2_A_industry_breadth_strength__R2B_OVERLAY_10   | wide      |       19 |                    0.0678 |                   0.0089 |                -0.0525 |                  0.1053 |
| U2_B_tradable_breakout_expansion__R2B_REPLACE_3  | narrow    |       19 |                   -0.0345 |                  -0.0169 |                 0.0210 |                  0.7368 |
| U2_B_tradable_breakout_expansion__R2B_REPLACE_3  | mid       |       26 |                    0.0150 |                   0.0061 |                -0.0044 |                  0.4231 |
| U2_B_tradable_breakout_expansion__R2B_REPLACE_3  | wide      |       19 |                    0.0678 |                   0.0121 |                -0.0596 |                  0.1053 |
| U2_B_tradable_breakout_expansion__R2B_REPLACE_5  | narrow    |       19 |                   -0.0345 |                  -0.0169 |                 0.0210 |                  0.7368 |
| U2_B_tradable_breakout_expansion__R2B_REPLACE_5  | mid       |       26 |                    0.0150 |                   0.0089 |                -0.0080 |                  0.3462 |
| U2_B_tradable_breakout_expansion__R2B_REPLACE_5  | wide      |       19 |                    0.0678 |                   0.0121 |                -0.0589 |                  0.0526 |
| U2_B_tradable_breakout_expansion__R2B_OVERLAY_10 | narrow    |       19 |                   -0.0345 |                  -0.0137 |                 0.0210 |                  0.7368 |
| U2_B_tradable_breakout_expansion__R2B_OVERLAY_10 | mid       |       26 |                    0.0150 |                   0.0062 |                -0.0044 |                  0.4231 |
| U2_B_tradable_breakout_expansion__R2B_OVERLAY_10 | wide      |       19 |                    0.0678 |                   0.0121 |                -0.0517 |                  0.1053 |
| U2_C_s2_residual_elasticity__R2B_REPLACE_3       | narrow    |       19 |                   -0.0345 |                  -0.0137 |                 0.0267 |                  0.7895 |
| U2_C_s2_residual_elasticity__R2B_REPLACE_3       | mid       |       26 |                    0.0150 |                   0.0061 |                -0.0044 |                  0.4231 |
| U2_C_s2_residual_elasticity__R2B_REPLACE_3       | wide      |       19 |                    0.0678 |                   0.0088 |                -0.0596 |                  0.1053 |
| U2_C_s2_residual_elasticity__R2B_REPLACE_5       | narrow    |       19 |                   -0.0345 |                  -0.0137 |                 0.0239 |                  0.7895 |
| U2_C_s2_residual_elasticity__R2B_REPLACE_5       | mid       |       26 |                    0.0150 |                   0.0073 |                -0.0045 |                  0.4231 |
| U2_C_s2_residual_elasticity__R2B_REPLACE_5       | wide      |       19 |                    0.0678 |                   0.0086 |                -0.0536 |                  0.1053 |
| U2_C_s2_residual_elasticity__R2B_OVERLAY_10      | narrow    |       19 |                   -0.0345 |                  -0.0137 |                 0.0214 |                  0.7895 |
| U2_C_s2_residual_elasticity__R2B_OVERLAY_10      | mid       |       26 |                    0.0150 |                   0.0062 |                -0.0044 |                  0.4231 |
| U2_C_s2_residual_elasticity__R2B_OVERLAY_10      | wide      |       19 |                    0.0678 |                   0.0121 |                -0.0523 |                  0.1053 |

## 5. Switch quality

| candidate_id                                     | regime      |   rebalances |   mean_switch_in |   mean_switch_out |   mean_switch_in_minus_out |   median_switch_in_minus_out |   switch_in_winning_share |   mean_topk_minus_next |
|:-------------------------------------------------|:------------|-------------:|-----------------:|------------------:|---------------------------:|-----------------------------:|--------------------------:|-----------------------:|
| BASELINE_S2_FIXED                                | strong_down |            3 |          -0.0504 |            0.0139 |                    -0.0644 |                      -0.0176 |                    0.3333 |                 0.0034 |
| BASELINE_S2_FIXED                                | mild_down   |            1 |          -0.0137 |           -0.0452 |                     0.0315 |                       0.0315 |                    1.0000 |                 0.0022 |
| BASELINE_S2_FIXED                                | neutral     |            5 |           0.0266 |            0.0089 |                     0.0177 |                      -0.0081 |                    0.4000 |                 0.0090 |
| BASELINE_S2_FIXED                                | mild_up     |            4 |           0.0482 |            0.0531 |                    -0.0049 |                      -0.0298 |                    0.2500 |                -0.0060 |
| BASELINE_S2_FIXED                                | strong_up   |            1 |           0.1127 |            0.1364 |                    -0.0237 |                      -0.0237 |                    0.0000 |                -0.0895 |
| U2_A_industry_breadth_strength__R2B_REPLACE_3    | strong_down |           14 |          -0.0065 |           -0.0682 |                     0.0617 |                       0.0758 |                    0.6429 |                 0.0470 |
| U2_A_industry_breadth_strength__R2B_REPLACE_3    | mild_down   |           13 |           0.0255 |           -0.0096 |                     0.0351 |                      -0.0207 |                    0.3846 |                 0.0100 |
| U2_A_industry_breadth_strength__R2B_REPLACE_3    | neutral     |           12 |           0.0127 |           -0.0134 |                     0.0261 |                       0.0174 |                    0.5000 |                -0.0049 |
| U2_A_industry_breadth_strength__R2B_REPLACE_3    | mild_up     |            9 |           0.0253 |            0.0438 |                    -0.0185 |                      -0.0107 |                    0.4444 |                -0.0063 |
| U2_A_industry_breadth_strength__R2B_REPLACE_3    | strong_up   |           11 |           0.0804 |            0.0491 |                     0.0313 |                       0.0155 |                    0.6364 |                -0.0501 |
| U2_A_industry_breadth_strength__R2B_REPLACE_5    | strong_down |           14 |          -0.0116 |           -0.0654 |                     0.0539 |                       0.0758 |                    0.6429 |                 0.0433 |
| U2_A_industry_breadth_strength__R2B_REPLACE_5    | mild_down   |           13 |           0.0247 |           -0.0046 |                     0.0293 |                      -0.0292 |                    0.3846 |                 0.0125 |
| U2_A_industry_breadth_strength__R2B_REPLACE_5    | neutral     |           12 |           0.0129 |           -0.0132 |                     0.0261 |                       0.0174 |                    0.5000 |                -0.0049 |
| U2_A_industry_breadth_strength__R2B_REPLACE_5    | mild_up     |            9 |           0.0253 |            0.0446 |                    -0.0193 |                      -0.0107 |                    0.4444 |                -0.0063 |
| U2_A_industry_breadth_strength__R2B_REPLACE_5    | strong_up   |           11 |           0.0775 |            0.0621 |                     0.0154 |                      -0.0200 |                    0.4545 |                -0.0508 |
| U2_A_industry_breadth_strength__R2B_OVERLAY_10   | strong_down |           14 |          -0.0132 |           -0.0663 |                     0.0531 |                       0.0707 |                    0.6429 |                 0.0405 |
| U2_A_industry_breadth_strength__R2B_OVERLAY_10   | mild_down   |           13 |           0.0261 |           -0.0038 |                     0.0299 |                      -0.0292 |                    0.3846 |                 0.0130 |
| U2_A_industry_breadth_strength__R2B_OVERLAY_10   | neutral     |           12 |           0.0129 |           -0.0133 |                     0.0262 |                       0.0174 |                    0.5000 |                -0.0049 |
| U2_A_industry_breadth_strength__R2B_OVERLAY_10   | mild_up     |            9 |           0.0263 |            0.0465 |                    -0.0201 |                      -0.0107 |                    0.4444 |                -0.0063 |
| U2_A_industry_breadth_strength__R2B_OVERLAY_10   | strong_up   |           11 |           0.0779 |            0.0617 |                     0.0162 |                      -0.0200 |                    0.4545 |                -0.0506 |
| U2_B_tradable_breakout_expansion__R2B_REPLACE_3  | strong_down |           14 |          -0.0440 |           -0.0482 |                     0.0042 |                       0.0104 |                    0.5714 |                 0.0434 |
| U2_B_tradable_breakout_expansion__R2B_REPLACE_3  | mild_down   |           13 |           0.0102 |            0.0442 |                    -0.0340 |                      -0.0220 |                    0.3077 |                 0.0002 |
| U2_B_tradable_breakout_expansion__R2B_REPLACE_3  | neutral     |           12 |          -0.0156 |            0.0073 |                    -0.0229 |                      -0.0356 |                    0.4167 |                -0.0100 |
| U2_B_tradable_breakout_expansion__R2B_REPLACE_3  | mild_up     |           10 |           0.0226 |            0.0070 |                     0.0156 |                      -0.0050 |                    0.4000 |                -0.0172 |
| U2_B_tradable_breakout_expansion__R2B_REPLACE_3  | strong_up   |           12 |           0.0517 |            0.0528 |                    -0.0011 |                      -0.0087 |                    0.4167 |                -0.0370 |
| U2_B_tradable_breakout_expansion__R2B_REPLACE_5  | strong_down |           14 |          -0.0501 |           -0.0480 |                    -0.0021 |                       0.0104 |                    0.5714 |                 0.0392 |
| U2_B_tradable_breakout_expansion__R2B_REPLACE_5  | mild_down   |           13 |           0.0049 |            0.0384 |                    -0.0335 |                      -0.0249 |                    0.3846 |                -0.0050 |
| U2_B_tradable_breakout_expansion__R2B_REPLACE_5  | neutral     |           12 |          -0.0168 |            0.0060 |                    -0.0228 |                      -0.0267 |                    0.4167 |                -0.0091 |
| U2_B_tradable_breakout_expansion__R2B_REPLACE_5  | mild_up     |           10 |           0.0237 |            0.0049 |                     0.0188 |                       0.0046 |                    0.5000 |                -0.0189 |
| U2_B_tradable_breakout_expansion__R2B_REPLACE_5  | strong_up   |           12 |           0.0556 |            0.0509 |                     0.0047 |                      -0.0125 |                    0.4167 |                -0.0374 |
| U2_B_tradable_breakout_expansion__R2B_OVERLAY_10 | strong_down |           14 |          -0.0480 |           -0.0495 |                     0.0015 |                       0.0104 |                    0.5714 |                 0.0398 |
| U2_B_tradable_breakout_expansion__R2B_OVERLAY_10 | mild_down   |           13 |           0.0081 |            0.0390 |                    -0.0309 |                      -0.0142 |                    0.3846 |                -0.0012 |
| U2_B_tradable_breakout_expansion__R2B_OVERLAY_10 | neutral     |           12 |          -0.0190 |            0.0058 |                    -0.0247 |                      -0.0434 |                    0.4167 |                -0.0122 |
| U2_B_tradable_breakout_expansion__R2B_OVERLAY_10 | mild_up     |           10 |           0.0242 |            0.0007 |                     0.0235 |                       0.0215 |                    0.6000 |                -0.0189 |
| U2_B_tradable_breakout_expansion__R2B_OVERLAY_10 | strong_up   |           12 |           0.0536 |            0.0487 |                     0.0049 |                      -0.0010 |                    0.5000 |                -0.0360 |
| U2_C_s2_residual_elasticity__R2B_REPLACE_3       | strong_down |           14 |          -0.0325 |           -0.0608 |                     0.0283 |                       0.0192 |                    0.7143 |                 0.0456 |
| U2_C_s2_residual_elasticity__R2B_REPLACE_3       | mild_down   |           13 |          -0.0116 |            0.0171 |                    -0.0287 |                      -0.0263 |                    0.3846 |                -0.0230 |
| U2_C_s2_residual_elasticity__R2B_REPLACE_3       | neutral     |           12 |          -0.0030 |            0.0062 |                    -0.0092 |                      -0.0135 |                    0.4167 |                -0.0134 |
| U2_C_s2_residual_elasticity__R2B_REPLACE_3       | mild_up     |           11 |           0.0293 |            0.0392 |                    -0.0099 |                      -0.0072 |                    0.4545 |                -0.0179 |
| U2_C_s2_residual_elasticity__R2B_REPLACE_3       | strong_up   |           12 |           0.0842 |            0.0762 |                     0.0079 |                      -0.0030 |                    0.4167 |                -0.0351 |
| U2_C_s2_residual_elasticity__R2B_REPLACE_5       | strong_down |           14 |          -0.0334 |           -0.0617 |                     0.0282 |                       0.0232 |                    0.7857 |                 0.0456 |
| U2_C_s2_residual_elasticity__R2B_REPLACE_5       | mild_down   |           13 |          -0.0164 |            0.0104 |                    -0.0269 |                      -0.0129 |                    0.4615 |                -0.0266 |
| U2_C_s2_residual_elasticity__R2B_REPLACE_5       | neutral     |           12 |          -0.0029 |            0.0053 |                    -0.0082 |                      -0.0004 |                    0.5000 |                -0.0123 |
| U2_C_s2_residual_elasticity__R2B_REPLACE_5       | mild_up     |           11 |           0.0285 |            0.0405 |                    -0.0120 |                      -0.0014 |                    0.3636 |                -0.0182 |
| U2_C_s2_residual_elasticity__R2B_REPLACE_5       | strong_up   |           12 |           0.0820 |            0.0769 |                     0.0051 |                      -0.0030 |                    0.4167 |                -0.0355 |
| U2_C_s2_residual_elasticity__R2B_OVERLAY_10      | strong_down |           14 |          -0.0334 |           -0.0641 |                     0.0307 |                       0.0221 |                    0.7143 |                 0.0469 |
| U2_C_s2_residual_elasticity__R2B_OVERLAY_10      | mild_down   |           13 |          -0.0101 |            0.0162 |                    -0.0263 |                      -0.0030 |                    0.4615 |                -0.0217 |
| U2_C_s2_residual_elasticity__R2B_OVERLAY_10      | neutral     |           12 |          -0.0017 |            0.0068 |                    -0.0085 |                      -0.0098 |                    0.4167 |                -0.0123 |
| U2_C_s2_residual_elasticity__R2B_OVERLAY_10      | mild_up     |           11 |           0.0286 |            0.0403 |                    -0.0117 |                      -0.0128 |                    0.3636 |                -0.0177 |
| U2_C_s2_residual_elasticity__R2B_OVERLAY_10      | strong_up   |           12 |           0.0882 |            0.0752 |                     0.0130 |                      -0.0030 |                    0.4167 |                -0.0332 |

## 6. 行业暴露

| candidate_id                                     | trade_date          | industry          |   weight |
|:-------------------------------------------------|:--------------------|:------------------|---------:|
| BASELINE_S2_FIXED                                | 2026-02-27 00:00:00 | proxy_sh_main_60  |   0.9500 |
| U2_A_industry_breadth_strength__R2B_REPLACE_3    | 2026-02-27 00:00:00 | proxy_sh_main_60  |   0.8000 |
| U2_A_industry_breadth_strength__R2B_OVERLAY_10   | 2026-02-27 00:00:00 | proxy_sh_main_60  |   0.8500 |
| U2_A_industry_breadth_strength__R2B_REPLACE_5    | 2026-02-27 00:00:00 | proxy_sh_main_60  |   0.7000 |
| U2_C_s2_residual_elasticity__R2B_REPLACE_3       | 2026-03-31 00:00:00 | proxy_chinext_300 |   0.0500 |
| U2_A_industry_breadth_strength__R2B_OVERLAY_10   | 2026-03-31 00:00:00 | proxy_sh_main_60  |   0.8000 |
| U2_C_s2_residual_elasticity__R2B_REPLACE_3       | 2026-03-31 00:00:00 | proxy_prefix_30   |   0.1000 |
| U2_C_s2_residual_elasticity__R2B_REPLACE_3       | 2026-03-31 00:00:00 | proxy_sh_main_60  |   0.7500 |
| U2_C_s2_residual_elasticity__R2B_REPLACE_5       | 2026-03-31 00:00:00 | proxy_sz_main_00  |   0.1500 |
| U2_C_s2_residual_elasticity__R2B_REPLACE_5       | 2026-03-31 00:00:00 | proxy_sh_main_60  |   0.6500 |
| U2_A_industry_breadth_strength__R2B_OVERLAY_10   | 2026-03-31 00:00:00 | proxy_sz_main_00  |   0.2000 |
| U2_B_tradable_breakout_expansion__R2B_REPLACE_3  | 2026-03-31 00:00:00 | proxy_sz_main_00  |   0.1500 |
| U2_B_tradable_breakout_expansion__R2B_REPLACE_5  | 2026-03-31 00:00:00 | proxy_sz_main_00  |   0.1500 |
| U2_B_tradable_breakout_expansion__R2B_REPLACE_5  | 2026-03-31 00:00:00 | proxy_prefix_30   |   0.0500 |
| U2_C_s2_residual_elasticity__R2B_OVERLAY_10      | 2026-03-31 00:00:00 | proxy_chinext_300 |   0.0200 |
| U2_A_industry_breadth_strength__R2B_REPLACE_5    | 2026-03-31 00:00:00 | proxy_sz_main_00  |   0.2500 |
| U2_A_industry_breadth_strength__R2B_REPLACE_5    | 2026-03-31 00:00:00 | proxy_sh_main_60  |   0.7500 |
| U2_A_industry_breadth_strength__R2B_REPLACE_3    | 2026-03-31 00:00:00 | proxy_sh_main_60  |   0.7500 |
| U2_C_s2_residual_elasticity__R2B_OVERLAY_10      | 2026-03-31 00:00:00 | proxy_prefix_30   |   0.0600 |
| U2_A_industry_breadth_strength__R2B_REPLACE_3    | 2026-03-31 00:00:00 | proxy_sz_main_00  |   0.2500 |
| U2_B_tradable_breakout_expansion__R2B_OVERLAY_10 | 2026-03-31 00:00:00 | proxy_star_688    |   0.0600 |
| U2_B_tradable_breakout_expansion__R2B_OVERLAY_10 | 2026-03-31 00:00:00 | proxy_sz_main_00  |   0.1200 |
| BASELINE_S2_FIXED                                | 2026-03-31 00:00:00 | proxy_sz_main_00  |   0.1000 |
| BASELINE_S2_FIXED                                | 2026-03-31 00:00:00 | proxy_sh_main_60  |   0.9000 |
| U2_B_tradable_breakout_expansion__R2B_OVERLAY_10 | 2026-03-31 00:00:00 | proxy_sh_main_60  |   0.8000 |
| U2_C_s2_residual_elasticity__R2B_OVERLAY_10      | 2026-03-31 00:00:00 | proxy_sh_main_60  |   0.8000 |
| U2_B_tradable_breakout_expansion__R2B_REPLACE_5  | 2026-03-31 00:00:00 | proxy_sh_main_60  |   0.6500 |
| U2_B_tradable_breakout_expansion__R2B_REPLACE_3  | 2026-03-31 00:00:00 | proxy_star_688    |   0.1000 |
| U2_B_tradable_breakout_expansion__R2B_REPLACE_3  | 2026-03-31 00:00:00 | proxy_sh_main_60  |   0.7500 |
| U2_C_s2_residual_elasticity__R2B_REPLACE_5       | 2026-03-31 00:00:00 | proxy_prefix_30   |   0.1500 |
| U2_A_industry_breadth_strength__R2B_REPLACE_5    | 2026-04-13 00:00:00 | proxy_sh_main_60  |   0.9000 |
| U2_C_s2_residual_elasticity__R2B_REPLACE_5       | 2026-04-13 00:00:00 | proxy_sh_main_60  |   0.9000 |
| U2_A_industry_breadth_strength__R2B_OVERLAY_10   | 2026-04-13 00:00:00 | proxy_sh_main_60  |   0.9000 |
| U2_A_industry_breadth_strength__R2B_OVERLAY_10   | 2026-04-13 00:00:00 | proxy_sz_main_00  |   0.1000 |
| U2_B_tradable_breakout_expansion__R2B_REPLACE_3  | 2026-04-13 00:00:00 | proxy_sh_main_60  |   0.9000 |
| U2_B_tradable_breakout_expansion__R2B_REPLACE_3  | 2026-04-13 00:00:00 | proxy_sz_main_00  |   0.1000 |
| U2_A_industry_breadth_strength__R2B_REPLACE_3    | 2026-04-13 00:00:00 | proxy_sz_main_00  |   0.1000 |
| U2_A_industry_breadth_strength__R2B_REPLACE_3    | 2026-04-13 00:00:00 | proxy_sh_main_60  |   0.9000 |
| U2_C_s2_residual_elasticity__R2B_OVERLAY_10      | 2026-04-13 00:00:00 | proxy_sh_main_60  |   0.9000 |
| U2_C_s2_residual_elasticity__R2B_REPLACE_5       | 2026-04-13 00:00:00 | proxy_sz_main_00  |   0.1000 |
| U2_B_tradable_breakout_expansion__R2B_REPLACE_5  | 2026-04-13 00:00:00 | proxy_sh_main_60  |   0.9000 |
| U2_A_industry_breadth_strength__R2B_REPLACE_5    | 2026-04-13 00:00:00 | proxy_sz_main_00  |   0.1000 |
| U2_B_tradable_breakout_expansion__R2B_OVERLAY_10 | 2026-04-13 00:00:00 | proxy_sh_main_60  |   0.9000 |
| U2_B_tradable_breakout_expansion__R2B_OVERLAY_10 | 2026-04-13 00:00:00 | proxy_sz_main_00  |   0.1000 |
| BASELINE_S2_FIXED                                | 2026-04-13 00:00:00 | proxy_sz_main_00  |   0.1000 |
| BASELINE_S2_FIXED                                | 2026-04-13 00:00:00 | proxy_sh_main_60  |   0.9000 |
| U2_C_s2_residual_elasticity__R2B_REPLACE_3       | 2026-04-13 00:00:00 | proxy_sh_main_60  |   0.9000 |
| U2_C_s2_residual_elasticity__R2B_REPLACE_3       | 2026-04-13 00:00:00 | proxy_sz_main_00  |   0.1000 |
| U2_B_tradable_breakout_expansion__R2B_REPLACE_5  | 2026-04-13 00:00:00 | proxy_sz_main_00  |   0.1000 |
| U2_C_s2_residual_elasticity__R2B_OVERLAY_10      | 2026-04-13 00:00:00 | proxy_sz_main_00  |   0.1000 |

## 7. 结论

- R2B 第一轮无候选通过，也无 gray zone。规则版 R3 不应启动。
- 无候选达到 daily proxy `>= +3%`，不补正式 full backtest。

## 8. 产出文件

- `data/results/r2b_tradable_upside_replacement_v1_2026-04-28_leaderboard.csv`
- `data/results/r2b_tradable_upside_replacement_v1_2026-04-28_replacement_diag_long.csv`
- `data/results/r2b_tradable_upside_replacement_v1_2026-04-28_standalone_leaderboard.csv`
- `data/results/r2b_tradable_upside_replacement_v1_2026-04-28_overlap_long.csv`
- `data/results/r2b_tradable_upside_replacement_v1_2026-04-28_industry_exposure_long.csv`
- `data/results/r2b_tradable_upside_replacement_v1_2026-04-28_regime_long.csv`
- `data/results/r2b_tradable_upside_replacement_v1_2026-04-28_breadth_long.csv`
- `data/results/r2b_tradable_upside_replacement_v1_2026-04-28_year_long.csv`
- `data/results/r2b_tradable_upside_replacement_v1_2026-04-28_switch_long.csv`
- `data/results/r2b_tradable_upside_replacement_v1_2026-04-28_monthly_long.csv`
- `data/results/r2b_tradable_upside_replacement_v1_2026-04-28_summary.json`

## 9. 参数

- `start`: `2021-01-01`
- `end`: `2026-04-28`
- `top_k`: `20`
- `rebalance_rule`: `M`
- `portfolio_method`: `defensive_core_capped_replacement`
- `max_turnover`: `1.0`
- `execution_mode`: `tplus1_open`
- `state_lag`: `previous_completed_month`
- `state_threshold_mode`: `expanding`
- `replacement_state_gate`: `lagged_regime==strong_up or lagged_breadth==wide`
- `upside_pct`: `0.9`
- `score_margin`: `0.1`
- `max_industry_names`: `5`
- `max_limit_up_hits_20d`: `2.0`
- `max_expansion`: `1.5`
- `industry_map_source`: `symbol_prefix_proxy_missing_data_cache_industry_map`
- `prefilter`: `{'enabled': False, 'limit_move_max': 2, 'turnover_low_pct': 0.1, 'turnover_high_pct': 0.98, 'price_position_high_pct': 0.9}`
- `universe_filter`: `{'enabled': True, 'min_amount_20d': 50000000, 'require_roe_ttm_positive': True}`
- `benchmark_symbol`: `market_ew_proxy`
- `benchmark_min_history_days`: `551`
- `primary_benchmark_return_mode`: `open_to_open`
- `comparison_benchmark_return_mode`: `close_to_close`
- `config_source`: `/mnt/ssd/lh/config.yaml.backtest`
- `eval_contract_version`: `r0_eval_execution_contract_2026-04-28`
- `execution_contract_version`: `tplus1_open_buy_delta_limit_mask_2026-04-28`
- `p1_experiment_mode`: `daily_proxy_first`
- `legacy_proxy_decision_role`: `diagnostic_only`
- `primary_decision_metric`: `daily_bt_like_proxy_annualized_excess_vs_market`
- `gate_thresholds`: `{'reject': 0.0, 'full_backtest': 0.03}`
- `defensive_weight_diag_rows`: `64`
