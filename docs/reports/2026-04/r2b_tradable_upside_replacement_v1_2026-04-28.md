# R2B Tradable Upside Replacement v1

- 生成时间：`2026-04-28T08:44:22.905354+00:00`
- 配置快照：`/mnt/ssd/lh/config.yaml.backtest`
- `eval_contract_version`: `r0_eval_execution_contract_2026-04-28`
- `execution_contract_version`: `tplus1_open_buy_delta_limit_mask_2026-04-28`
- `industry_map_source`: `akshare.stock_industry_clf_hist_sw+stock_industry_category_cninfo`
- `industry_map_source_status`: `real_industry_map`
- 组合表达：`S2 defensive Top-20 + capped replacement slots`，不再使用完整 upside Top-20 sleeve。
- 状态输入：上一已完成月份 `regime/breadth`；仅 `strong_up` 或 `wide` 允许 replacement gate。
- primary benchmark：`open_to_open`；promotion metric：`daily_bt_like_proxy_annualized_excess_vs_market`。

## 1. Leaderboard

| candidate_id                                     |   daily_proxy_annualized_excess_vs_market |   delta_vs_baseline_proxy | gate_decision   |   strong_up_median_excess |   delta_vs_baseline_strong_up_median_excess |   strong_up_positive_share |   delta_vs_baseline_strong_up_positive_share |   strong_down_median_excess |   delta_vs_baseline_strong_down_median_excess |   strong_up_switch_in_minus_out |   avg_turnover_half_l1 |   delta_vs_baseline_turnover | industry_map_source_status   | industry_alpha_evidence_allowed   |   n_rebalances |
|:-------------------------------------------------|------------------------------------------:|--------------------------:|:----------------|--------------------------:|--------------------------------------------:|---------------------------:|---------------------------------------------:|----------------------------:|----------------------------------------------:|--------------------------------:|-----------------------:|-----------------------------:|:-----------------------------|:----------------------------------|---------------:|
| BASELINE_S2_FIXED                                |                                   -0.0859 |                    0.0000 | reject          |                   -0.0612 |                                      0.0000 |                     0.1538 |                                       0.0000 |                      0.0629 |                                        0.0000 |                         -0.0237 |                 0.0844 |                       0.0000 | real_industry_map            | True                              |             64 |
| U2_C_s2_residual_elasticity__R2B_OVERLAY_10      |                                   -0.0899 |                   -0.0039 | reject          |                   -0.0612 |                                      0.0000 |                     0.1538 |                                       0.0000 |                      0.0639 |                                        0.0010 |                          0.0114 |                 0.1036 |                       0.0193 | real_industry_map            | True                              |             64 |
| U2_A_industry_breadth_strength__R2B_OVERLAY_10   |                                   -0.0924 |                   -0.0065 | reject          |                   -0.0613 |                                     -0.0001 |                     0.1538 |                                       0.0000 |                      0.0613 |                                       -0.0017 |                          0.0262 |                 0.1047 |                       0.0203 | real_industry_map            | True                              |             64 |
| U2_C_s2_residual_elasticity__R2B_REPLACE_5       |                                   -0.0962 |                   -0.0103 | reject          |                   -0.0612 |                                      0.0000 |                     0.1538 |                                       0.0000 |                      0.0583 |                                       -0.0047 |                          0.0059 |                 0.2077 |                       0.1233 | real_industry_map            | True                              |             64 |
| U2_A_industry_breadth_strength__R2B_REPLACE_3    |                                   -0.0992 |                   -0.0133 | reject          |                   -0.0760 |                                     -0.0148 |                     0.1538 |                                       0.0000 |                      0.0600 |                                       -0.0029 |                          0.0250 |                 0.1385 |                       0.0541 | real_industry_map            | True                              |             64 |
| U2_C_s2_residual_elasticity__R2B_REPLACE_3       |                                   -0.1005 |                   -0.0146 | reject          |                   -0.0659 |                                     -0.0047 |                     0.1538 |                                       0.0000 |                      0.0581 |                                       -0.0049 |                          0.0075 |                 0.1372 |                       0.0528 | real_industry_map            | True                              |             64 |
| U2_B_tradable_breakout_expansion__R2B_OVERLAY_10 |                                   -0.1006 |                   -0.0147 | reject          |                   -0.0612 |                                      0.0000 |                     0.1538 |                                       0.0000 |                      0.0559 |                                       -0.0070 |                          0.0021 |                 0.1036 |                       0.0193 | real_industry_map            | True                              |             64 |
| U2_A_industry_breadth_strength__R2B_REPLACE_5    |                                   -0.1030 |                   -0.0171 | reject          |                   -0.0764 |                                     -0.0152 |                     0.1538 |                                       0.0000 |                      0.0585 |                                       -0.0044 |                          0.0270 |                 0.2103 |                       0.1259 | real_industry_map            | True                              |             64 |
| U2_B_tradable_breakout_expansion__R2B_REPLACE_3  |                                   -0.1088 |                   -0.0229 | reject          |                   -0.0612 |                                      0.0000 |                     0.1538 |                                       0.0000 |                      0.0410 |                                       -0.0219 |                         -0.0033 |                 0.1372 |                       0.0528 | real_industry_map            | True                              |             64 |
| U2_B_tradable_breakout_expansion__R2B_REPLACE_5  |                                   -0.1231 |                   -0.0371 | reject          |                   -0.0635 |                                     -0.0024 |                     0.1538 |                                       0.0000 |                      0.0452 |                                       -0.0177 |                          0.0067 |                 0.2077 |                       0.1233 | real_industry_map            | True                              |             64 |

## 2. R2B 验收

| candidate_id                                     | status   | daily_proxy_not_below_baseline   | daily_proxy_>=_0   | strong_up_median_+2pct_or_positive_share_+10pct   | strong_down_not_worse_than_-2pct   | turnover_delta_<=_+0.10   | strong_up_switch_not_negative   |
|:-------------------------------------------------|:---------|:---------------------------------|:-------------------|:--------------------------------------------------|:-----------------------------------|:--------------------------|:--------------------------------|
| BASELINE_S2_FIXED                                | baseline | nan                              | nan                | nan                                               | nan                                | nan                       | nan                             |
| U2_C_s2_residual_elasticity__R2B_OVERLAY_10      | fail     | fail                             | fail               | fail                                              | pass                               | pass                      | pass                            |
| U2_A_industry_breadth_strength__R2B_OVERLAY_10   | fail     | fail                             | fail               | fail                                              | pass                               | pass                      | pass                            |
| U2_C_s2_residual_elasticity__R2B_REPLACE_5       | fail     | fail                             | fail               | fail                                              | pass                               | fail                      | pass                            |
| U2_A_industry_breadth_strength__R2B_REPLACE_3    | fail     | fail                             | fail               | fail                                              | pass                               | pass                      | pass                            |
| U2_C_s2_residual_elasticity__R2B_REPLACE_3       | fail     | fail                             | fail               | fail                                              | pass                               | pass                      | pass                            |
| U2_B_tradable_breakout_expansion__R2B_OVERLAY_10 | fail     | fail                             | fail               | fail                                              | pass                               | pass                      | pass                            |
| U2_A_industry_breadth_strength__R2B_REPLACE_5    | fail     | fail                             | fail               | fail                                              | pass                               | fail                      | pass                            |
| U2_B_tradable_breakout_expansion__R2B_REPLACE_3  | fail     | fail                             | fail               | fail                                              | fail                               | pass                      | fail                            |
| U2_B_tradable_breakout_expansion__R2B_REPLACE_5  | fail     | fail                             | fail               | fail                                              | pass                               | fail                      | pass                            |

## 3. Replacement 触发统计

| candidate_id                                     | replacement_allowed   |   rebalances |   avg_replacement_count |   max_replacement_count |   active_rebalance_share |
|:-------------------------------------------------|:----------------------|-------------:|------------------------:|------------------------:|-------------------------:|
| U2_A_industry_breadth_strength__R2B_OVERLAY_10   | False                 |           47 |                  0.0000 |                       0 |                   0.0000 |
| U2_A_industry_breadth_strength__R2B_OVERLAY_10   | True                  |           17 |                  5.0000 |                       5 |                   1.0000 |
| U2_A_industry_breadth_strength__R2B_REPLACE_3    | False                 |           47 |                  0.0000 |                       0 |                   0.0000 |
| U2_A_industry_breadth_strength__R2B_REPLACE_3    | True                  |           17 |                  3.0000 |                       3 |                   1.0000 |
| U2_A_industry_breadth_strength__R2B_REPLACE_5    | False                 |           47 |                  0.0000 |                       0 |                   0.0000 |
| U2_A_industry_breadth_strength__R2B_REPLACE_5    | True                  |           17 |                  5.0000 |                       5 |                   1.0000 |
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
| U2_A_industry_breadth_strength__R2B_REPLACE_3    | strong_down |       13 |                   -0.0604 |                  -0.0203 |                 0.0600 |                  0.7692 |              -0.6375 |             -0.2554 |          0.4006 |
| U2_A_industry_breadth_strength__R2B_REPLACE_3    | mild_down   |       13 |                   -0.0109 |                   0.0000 |                 0.0057 |                  0.7692 |              -0.1702 |              0.0684 |         -0.4021 |
| U2_A_industry_breadth_strength__R2B_REPLACE_3    | neutral     |       12 |                    0.0087 |                   0.0010 |                -0.0087 |                  0.3333 |               0.1358 |             -0.0040 |         -0.0296 |
| U2_A_industry_breadth_strength__R2B_REPLACE_3    | mild_up     |       13 |                    0.0378 |                   0.0121 |                -0.0258 |                  0.0769 |               0.6459 |              0.1634 |          0.2530 |
| U2_A_industry_breadth_strength__R2B_REPLACE_3    | strong_up   |       13 |                    0.0811 |                   0.0113 |                -0.0760 |                  0.1538 |               2.5282 |              0.5855 |          0.2316 |
| U2_A_industry_breadth_strength__R2B_REPLACE_5    | strong_down |       13 |                   -0.0604 |                  -0.0175 |                 0.0585 |                  0.7692 |              -0.6375 |             -0.2671 |          0.4191 |
| U2_A_industry_breadth_strength__R2B_REPLACE_5    | mild_down   |       13 |                   -0.0109 |                   0.0000 |                 0.0070 |                  0.6923 |              -0.1702 |              0.0371 |         -0.2177 |
| U2_A_industry_breadth_strength__R2B_REPLACE_5    | neutral     |       12 |                    0.0087 |                   0.0027 |                -0.0048 |                  0.3333 |               0.1358 |              0.0018 |          0.0136 |
| U2_A_industry_breadth_strength__R2B_REPLACE_5    | mild_up     |       13 |                    0.0378 |                   0.0121 |                -0.0258 |                  0.0769 |               0.6459 |              0.1902 |          0.2945 |
| U2_A_industry_breadth_strength__R2B_REPLACE_5    | strong_up   |       13 |                    0.0811 |                   0.0086 |                -0.0764 |                  0.1538 |               2.5282 |              0.5664 |          0.2240 |
| U2_A_industry_breadth_strength__R2B_OVERLAY_10   | strong_down |       13 |                   -0.0604 |                  -0.0209 |                 0.0613 |                  0.7692 |              -0.6375 |             -0.2473 |          0.3879 |
| U2_A_industry_breadth_strength__R2B_OVERLAY_10   | mild_down   |       13 |                   -0.0109 |                   0.0003 |                 0.0146 |                  0.7692 |              -0.1702 |              0.0617 |         -0.3625 |
| U2_A_industry_breadth_strength__R2B_OVERLAY_10   | neutral     |       12 |                    0.0087 |                   0.0010 |                -0.0082 |                  0.4167 |               0.1358 |              0.0138 |          0.1016 |
| U2_A_industry_breadth_strength__R2B_OVERLAY_10   | mild_up     |       13 |                    0.0378 |                   0.0121 |                -0.0258 |                  0.0769 |               0.6459 |              0.1671 |          0.2588 |
| U2_A_industry_breadth_strength__R2B_OVERLAY_10   | strong_up   |       13 |                    0.0811 |                   0.0199 |                -0.0613 |                  0.1538 |               2.5282 |              0.6117 |          0.2420 |
| U2_B_tradable_breakout_expansion__R2B_REPLACE_3  | strong_down |       13 |                   -0.0604 |                  -0.0255 |                 0.0410 |                  0.6923 |              -0.6375 |             -0.3006 |          0.4716 |
| U2_B_tradable_breakout_expansion__R2B_REPLACE_3  | mild_down   |       13 |                   -0.0109 |                  -0.0087 |                 0.0057 |                  0.7692 |              -0.1702 |              0.0373 |         -0.2194 |
| U2_B_tradable_breakout_expansion__R2B_REPLACE_3  | neutral     |       12 |                    0.0087 |                   0.0010 |                -0.0106 |                  0.4167 |               0.1358 |              0.0028 |          0.0204 |
| U2_B_tradable_breakout_expansion__R2B_REPLACE_3  | mild_up     |       13 |                    0.0378 |                   0.0121 |                -0.0258 |                  0.0769 |               0.6459 |              0.1851 |          0.2866 |
| U2_B_tradable_breakout_expansion__R2B_REPLACE_3  | strong_up   |       13 |                    0.0811 |                   0.0199 |                -0.0612 |                  0.1538 |               2.5282 |              0.6021 |          0.2382 |
| U2_B_tradable_breakout_expansion__R2B_REPLACE_5  | strong_down |       13 |                   -0.0604 |                  -0.0213 |                 0.0452 |                  0.6923 |              -0.6375 |             -0.3409 |          0.5347 |
| U2_B_tradable_breakout_expansion__R2B_REPLACE_5  | mild_down   |       13 |                   -0.0109 |                  -0.0135 |                 0.0057 |                  0.6154 |              -0.1702 |             -0.0235 |          0.1379 |
| U2_B_tradable_breakout_expansion__R2B_REPLACE_5  | neutral     |       12 |                    0.0087 |                   0.0016 |                -0.0057 |                  0.4167 |               0.1358 |              0.0040 |          0.0296 |
| U2_B_tradable_breakout_expansion__R2B_REPLACE_5  | mild_up     |       13 |                    0.0378 |                   0.0123 |                -0.0258 |                  0.0769 |               0.6459 |              0.2034 |          0.3150 |
| U2_B_tradable_breakout_expansion__R2B_REPLACE_5  | strong_up   |       13 |                    0.0811 |                   0.0199 |                -0.0635 |                  0.1538 |               2.5282 |              0.6271 |          0.2480 |
| U2_B_tradable_breakout_expansion__R2B_OVERLAY_10 | strong_down |       13 |                   -0.0604 |                  -0.0209 |                 0.0559 |                  0.6923 |              -0.6375 |             -0.2784 |          0.4368 |
| U2_B_tradable_breakout_expansion__R2B_OVERLAY_10 | mild_down   |       13 |                   -0.0109 |                  -0.0045 |                 0.0107 |                  0.7692 |              -0.1702 |              0.0365 |         -0.2144 |
| U2_B_tradable_breakout_expansion__R2B_OVERLAY_10 | neutral     |       12 |                    0.0087 |                   0.0010 |                -0.0086 |                  0.4167 |               0.1358 |              0.0145 |          0.1070 |
| U2_B_tradable_breakout_expansion__R2B_OVERLAY_10 | mild_up     |       13 |                    0.0378 |                   0.0121 |                -0.0258 |                  0.0769 |               0.6459 |              0.1722 |          0.2666 |
| U2_B_tradable_breakout_expansion__R2B_OVERLAY_10 | strong_up   |       13 |                    0.0811 |                   0.0199 |                -0.0612 |                  0.1538 |               2.5282 |              0.6362 |          0.2516 |
| U2_C_s2_residual_elasticity__R2B_REPLACE_3       | strong_down |       13 |                   -0.0604 |                  -0.0182 |                 0.0581 |                  0.7692 |              -0.6375 |             -0.2465 |          0.3867 |
| U2_C_s2_residual_elasticity__R2B_REPLACE_3       | mild_down   |       13 |                   -0.0109 |                  -0.0040 |                 0.0195 |                  0.6923 |              -0.1702 |              0.0191 |         -0.1123 |
| U2_C_s2_residual_elasticity__R2B_REPLACE_3       | neutral     |       12 |                    0.0087 |                   0.0010 |                -0.0106 |                  0.4167 |               0.1358 |              0.0150 |          0.1102 |
| U2_C_s2_residual_elasticity__R2B_REPLACE_3       | mild_up     |       13 |                    0.0378 |                   0.0087 |                -0.0258 |                  0.0769 |               0.6459 |              0.1585 |          0.2454 |
| U2_C_s2_residual_elasticity__R2B_REPLACE_3       | strong_up   |       13 |                    0.0811 |                   0.0184 |                -0.0659 |                  0.1538 |               2.5282 |              0.6138 |          0.2428 |
| U2_C_s2_residual_elasticity__R2B_REPLACE_5       | strong_down |       13 |                   -0.0604 |                  -0.0212 |                 0.0583 |                  0.7692 |              -0.6375 |             -0.2487 |          0.3902 |
| U2_C_s2_residual_elasticity__R2B_REPLACE_5       | mild_down   |       13 |                   -0.0109 |                  -0.0124 |                 0.0111 |                  0.7692 |              -0.1702 |              0.0013 |         -0.0074 |
| U2_C_s2_residual_elasticity__R2B_REPLACE_5       | neutral     |       12 |                    0.0087 |                   0.0010 |                -0.0106 |                  0.4167 |               0.1358 |              0.0152 |          0.1121 |
| U2_C_s2_residual_elasticity__R2B_REPLACE_5       | mild_up     |       13 |                    0.0378 |                   0.0121 |                -0.0258 |                  0.0769 |               0.6459 |              0.1820 |          0.2818 |
| U2_C_s2_residual_elasticity__R2B_REPLACE_5       | strong_up   |       13 |                    0.0811 |                   0.0199 |                -0.0612 |                  0.1538 |               2.5282 |              0.6427 |          0.2542 |
| U2_C_s2_residual_elasticity__R2B_OVERLAY_10      | strong_down |       13 |                   -0.0604 |                  -0.0209 |                 0.0639 |                  0.7692 |              -0.6375 |             -0.2399 |          0.3763 |
| U2_C_s2_residual_elasticity__R2B_OVERLAY_10      | mild_down   |       13 |                   -0.0109 |                  -0.0022 |                 0.0194 |                  0.7692 |              -0.1702 |              0.0465 |         -0.2732 |
| U2_C_s2_residual_elasticity__R2B_OVERLAY_10      | neutral     |       12 |                    0.0087 |                   0.0010 |                -0.0106 |                  0.4167 |               0.1358 |              0.0189 |          0.1392 |
| U2_C_s2_residual_elasticity__R2B_OVERLAY_10      | mild_up     |       13 |                    0.0378 |                   0.0121 |                -0.0258 |                  0.0769 |               0.6459 |              0.1637 |          0.2534 |
| U2_C_s2_residual_elasticity__R2B_OVERLAY_10      | strong_up   |       13 |                    0.0811 |                   0.0238 |                -0.0612 |                  0.1538 |               2.5282 |              0.6423 |          0.2541 |

| candidate_id                                     | breadth   |   months |   median_benchmark_return |   median_strategy_return |   median_excess_return |   positive_excess_share |
|:-------------------------------------------------|:----------|---------:|--------------------------:|-------------------------:|-----------------------:|------------------------:|
| BASELINE_S2_FIXED                                | narrow    |       19 |                   -0.0345 |                  -0.0137 |                 0.0249 |                  0.7895 |
| BASELINE_S2_FIXED                                | mid       |       26 |                    0.0150 |                   0.0064 |                -0.0043 |                  0.4231 |
| BASELINE_S2_FIXED                                | wide      |       19 |                    0.0678 |                   0.0121 |                -0.0470 |                  0.1053 |
| U2_A_industry_breadth_strength__R2B_REPLACE_3    | narrow    |       19 |                   -0.0345 |                  -0.0137 |                 0.0313 |                  0.7895 |
| U2_A_industry_breadth_strength__R2B_REPLACE_3    | mid       |       26 |                    0.0150 |                   0.0061 |                -0.0044 |                  0.3846 |
| U2_A_industry_breadth_strength__R2B_REPLACE_3    | wide      |       19 |                    0.0678 |                   0.0088 |                -0.0596 |                  0.1053 |
| U2_A_industry_breadth_strength__R2B_REPLACE_5    | narrow    |       19 |                   -0.0345 |                  -0.0137 |                 0.0313 |                  0.7895 |
| U2_A_industry_breadth_strength__R2B_REPLACE_5    | mid       |       26 |                    0.0150 |                   0.0059 |                -0.0089 |                  0.3462 |
| U2_A_industry_breadth_strength__R2B_REPLACE_5    | wide      |       19 |                    0.0678 |                   0.0071 |                -0.0596 |                  0.1053 |
| U2_A_industry_breadth_strength__R2B_OVERLAY_10   | narrow    |       19 |                   -0.0345 |                  -0.0137 |                 0.0313 |                  0.7895 |
| U2_A_industry_breadth_strength__R2B_OVERLAY_10   | mid       |       26 |                    0.0150 |                   0.0062 |                -0.0044 |                  0.4231 |
| U2_A_industry_breadth_strength__R2B_OVERLAY_10   | wide      |       19 |                    0.0678 |                   0.0089 |                -0.0596 |                  0.1053 |
| U2_B_tradable_breakout_expansion__R2B_REPLACE_3  | narrow    |       19 |                   -0.0345 |                  -0.0169 |                 0.0210 |                  0.7368 |
| U2_B_tradable_breakout_expansion__R2B_REPLACE_3  | mid       |       26 |                    0.0150 |                   0.0061 |                -0.0044 |                  0.4231 |
| U2_B_tradable_breakout_expansion__R2B_REPLACE_3  | wide      |       19 |                    0.0678 |                   0.0100 |                -0.0596 |                  0.1053 |
| U2_B_tradable_breakout_expansion__R2B_REPLACE_5  | narrow    |       19 |                   -0.0345 |                  -0.0169 |                 0.0210 |                  0.7368 |
| U2_B_tradable_breakout_expansion__R2B_REPLACE_5  | mid       |       26 |                    0.0150 |                   0.0089 |                -0.0079 |                  0.3846 |
| U2_B_tradable_breakout_expansion__R2B_REPLACE_5  | wide      |       19 |                    0.0678 |                   0.0121 |                -0.0596 |                  0.0526 |
| U2_B_tradable_breakout_expansion__R2B_OVERLAY_10 | narrow    |       19 |                   -0.0345 |                  -0.0137 |                 0.0210 |                  0.7368 |
| U2_B_tradable_breakout_expansion__R2B_OVERLAY_10 | mid       |       26 |                    0.0150 |                   0.0062 |                -0.0044 |                  0.4231 |
| U2_B_tradable_breakout_expansion__R2B_OVERLAY_10 | wide      |       19 |                    0.0678 |                   0.0121 |                -0.0552 |                  0.1053 |
| U2_C_s2_residual_elasticity__R2B_REPLACE_3       | narrow    |       19 |                   -0.0345 |                  -0.0137 |                 0.0267 |                  0.7895 |
| U2_C_s2_residual_elasticity__R2B_REPLACE_3       | mid       |       26 |                    0.0150 |                   0.0061 |                -0.0044 |                  0.4231 |
| U2_C_s2_residual_elasticity__R2B_REPLACE_3       | wide      |       19 |                    0.0678 |                   0.0088 |                -0.0596 |                  0.0526 |
| U2_C_s2_residual_elasticity__R2B_REPLACE_5       | narrow    |       19 |                   -0.0345 |                  -0.0137 |                 0.0232 |                  0.7895 |
| U2_C_s2_residual_elasticity__R2B_REPLACE_5       | mid       |       26 |                    0.0150 |                   0.0096 |                -0.0045 |                  0.4231 |
| U2_C_s2_residual_elasticity__R2B_REPLACE_5       | wide      |       19 |                    0.0678 |                   0.0121 |                -0.0536 |                  0.1053 |
| U2_C_s2_residual_elasticity__R2B_OVERLAY_10      | narrow    |       19 |                   -0.0345 |                  -0.0137 |                 0.0211 |                  0.7895 |
| U2_C_s2_residual_elasticity__R2B_OVERLAY_10      | mid       |       26 |                    0.0150 |                   0.0062 |                -0.0044 |                  0.4231 |
| U2_C_s2_residual_elasticity__R2B_OVERLAY_10      | wide      |       19 |                    0.0678 |                   0.0121 |                -0.0570 |                  0.1053 |

## 5. Switch quality

| candidate_id                                     | regime      |   rebalances |   mean_switch_in |   mean_switch_out |   mean_switch_in_minus_out |   median_switch_in_minus_out |   switch_in_winning_share |   mean_topk_minus_next |
|:-------------------------------------------------|:------------|-------------:|-----------------:|------------------:|---------------------------:|-----------------------------:|--------------------------:|-----------------------:|
| BASELINE_S2_FIXED                                | strong_down |            3 |          -0.0504 |            0.0139 |                    -0.0644 |                      -0.0176 |                    0.3333 |                 0.0034 |
| BASELINE_S2_FIXED                                | mild_down   |            1 |          -0.0137 |           -0.0452 |                     0.0315 |                       0.0315 |                    1.0000 |                 0.0022 |
| BASELINE_S2_FIXED                                | neutral     |            5 |           0.0266 |            0.0089 |                     0.0177 |                      -0.0081 |                    0.4000 |                 0.0090 |
| BASELINE_S2_FIXED                                | mild_up     |            4 |           0.0482 |            0.0531 |                    -0.0049 |                      -0.0298 |                    0.2500 |                -0.0060 |
| BASELINE_S2_FIXED                                | strong_up   |            1 |           0.1127 |            0.1364 |                    -0.0237 |                      -0.0237 |                    0.0000 |                -0.0895 |
| U2_A_industry_breadth_strength__R2B_REPLACE_3    | strong_down |           14 |          -0.0265 |           -0.0670 |                     0.0404 |                       0.0288 |                    0.7857 |                 0.0355 |
| U2_A_industry_breadth_strength__R2B_REPLACE_3    | mild_down   |           13 |           0.0407 |            0.0093 |                     0.0313 |                       0.0255 |                    0.5385 |                 0.0260 |
| U2_A_industry_breadth_strength__R2B_REPLACE_3    | neutral     |           12 |          -0.0044 |           -0.0168 |                     0.0124 |                       0.0156 |                    0.5000 |                -0.0041 |
| U2_A_industry_breadth_strength__R2B_REPLACE_3    | mild_up     |           10 |           0.0207 |            0.0392 |                    -0.0185 |                      -0.0270 |                    0.5000 |                -0.0043 |
| U2_A_industry_breadth_strength__R2B_REPLACE_3    | strong_up   |           12 |           0.0702 |            0.0452 |                     0.0250 |                       0.0115 |                    0.5000 |                -0.0255 |
| U2_A_industry_breadth_strength__R2B_REPLACE_5    | strong_down |           14 |          -0.0270 |           -0.0659 |                     0.0389 |                       0.0312 |                    0.7857 |                 0.0352 |
| U2_A_industry_breadth_strength__R2B_REPLACE_5    | mild_down   |           13 |           0.0395 |            0.0074 |                     0.0321 |                       0.0255 |                    0.5385 |                 0.0250 |
| U2_A_industry_breadth_strength__R2B_REPLACE_5    | neutral     |           12 |          -0.0057 |           -0.0153 |                     0.0096 |                       0.0050 |                    0.5000 |                -0.0042 |
| U2_A_industry_breadth_strength__R2B_REPLACE_5    | mild_up     |           10 |           0.0198 |            0.0435 |                    -0.0237 |                      -0.0330 |                    0.4000 |                -0.0035 |
| U2_A_industry_breadth_strength__R2B_REPLACE_5    | strong_up   |           12 |           0.0738 |            0.0468 |                     0.0270 |                       0.0208 |                    0.5000 |                -0.0260 |
| U2_A_industry_breadth_strength__R2B_OVERLAY_10   | strong_down |           14 |          -0.0267 |           -0.0666 |                     0.0399 |                       0.0336 |                    0.7857 |                 0.0340 |
| U2_A_industry_breadth_strength__R2B_OVERLAY_10   | mild_down   |           13 |           0.0373 |            0.0065 |                     0.0308 |                       0.0273 |                    0.5385 |                 0.0213 |
| U2_A_industry_breadth_strength__R2B_OVERLAY_10   | neutral     |           12 |          -0.0051 |           -0.0158 |                     0.0107 |                       0.0092 |                    0.5000 |                -0.0042 |
| U2_A_industry_breadth_strength__R2B_OVERLAY_10   | mild_up     |           10 |           0.0200 |            0.0421 |                    -0.0221 |                      -0.0238 |                    0.5000 |                -0.0036 |
| U2_A_industry_breadth_strength__R2B_OVERLAY_10   | strong_up   |           12 |           0.0744 |            0.0482 |                     0.0262 |                       0.0208 |                    0.5000 |                -0.0242 |
| U2_B_tradable_breakout_expansion__R2B_REPLACE_3  | strong_down |           14 |          -0.0440 |           -0.0482 |                     0.0042 |                       0.0104 |                    0.5714 |                 0.0434 |
| U2_B_tradable_breakout_expansion__R2B_REPLACE_3  | mild_down   |           13 |           0.0102 |            0.0442 |                    -0.0340 |                      -0.0220 |                    0.3077 |                 0.0002 |
| U2_B_tradable_breakout_expansion__R2B_REPLACE_3  | neutral     |           12 |          -0.0156 |            0.0073 |                    -0.0229 |                      -0.0356 |                    0.4167 |                -0.0100 |
| U2_B_tradable_breakout_expansion__R2B_REPLACE_3  | mild_up     |           10 |           0.0212 |            0.0057 |                     0.0155 |                      -0.0050 |                    0.4000 |                -0.0172 |
| U2_B_tradable_breakout_expansion__R2B_REPLACE_3  | strong_up   |           12 |           0.0461 |            0.0494 |                    -0.0033 |                      -0.0087 |                    0.4167 |                -0.0369 |
| U2_B_tradable_breakout_expansion__R2B_REPLACE_5  | strong_down |           14 |          -0.0498 |           -0.0480 |                    -0.0018 |                       0.0104 |                    0.5714 |                 0.0393 |
| U2_B_tradable_breakout_expansion__R2B_REPLACE_5  | mild_down   |           13 |           0.0049 |            0.0384 |                    -0.0335 |                      -0.0249 |                    0.3846 |                -0.0050 |
| U2_B_tradable_breakout_expansion__R2B_REPLACE_5  | neutral     |           12 |          -0.0168 |            0.0060 |                    -0.0228 |                      -0.0267 |                    0.4167 |                -0.0091 |
| U2_B_tradable_breakout_expansion__R2B_REPLACE_5  | mild_up     |           10 |           0.0227 |            0.0050 |                     0.0177 |                       0.0046 |                    0.5000 |                -0.0189 |
| U2_B_tradable_breakout_expansion__R2B_REPLACE_5  | strong_up   |           12 |           0.0543 |            0.0477 |                     0.0067 |                      -0.0125 |                    0.4167 |                -0.0366 |
| U2_B_tradable_breakout_expansion__R2B_OVERLAY_10 | strong_down |           14 |          -0.0513 |           -0.0495 |                    -0.0018 |                       0.0104 |                    0.5714 |                 0.0386 |
| U2_B_tradable_breakout_expansion__R2B_OVERLAY_10 | mild_down   |           13 |           0.0081 |            0.0390 |                    -0.0309 |                      -0.0142 |                    0.3846 |                -0.0012 |
| U2_B_tradable_breakout_expansion__R2B_OVERLAY_10 | neutral     |           12 |          -0.0190 |            0.0058 |                    -0.0247 |                      -0.0434 |                    0.4167 |                -0.0122 |
| U2_B_tradable_breakout_expansion__R2B_OVERLAY_10 | mild_up     |           10 |           0.0242 |            0.0004 |                     0.0238 |                       0.0215 |                    0.6000 |                -0.0189 |
| U2_B_tradable_breakout_expansion__R2B_OVERLAY_10 | strong_up   |           12 |           0.0514 |            0.0493 |                     0.0021 |                      -0.0010 |                    0.5000 |                -0.0370 |
| U2_C_s2_residual_elasticity__R2B_REPLACE_3       | strong_down |           14 |          -0.0325 |           -0.0608 |                     0.0283 |                       0.0192 |                    0.7143 |                 0.0456 |
| U2_C_s2_residual_elasticity__R2B_REPLACE_3       | mild_down   |           13 |          -0.0131 |            0.0166 |                    -0.0297 |                      -0.0263 |                    0.3846 |                -0.0228 |
| U2_C_s2_residual_elasticity__R2B_REPLACE_3       | neutral     |           12 |          -0.0030 |            0.0062 |                    -0.0092 |                      -0.0135 |                    0.4167 |                -0.0134 |
| U2_C_s2_residual_elasticity__R2B_REPLACE_3       | mild_up     |           11 |           0.0293 |            0.0372 |                    -0.0079 |                      -0.0072 |                    0.4545 |                -0.0179 |
| U2_C_s2_residual_elasticity__R2B_REPLACE_3       | strong_up   |           12 |           0.0837 |            0.0762 |                     0.0075 |                      -0.0030 |                    0.4167 |                -0.0353 |
| U2_C_s2_residual_elasticity__R2B_REPLACE_5       | strong_down |           14 |          -0.0348 |           -0.0617 |                     0.0269 |                       0.0195 |                    0.7857 |                 0.0449 |
| U2_C_s2_residual_elasticity__R2B_REPLACE_5       | mild_down   |           13 |          -0.0165 |            0.0132 |                    -0.0297 |                      -0.0195 |                    0.4615 |                -0.0262 |
| U2_C_s2_residual_elasticity__R2B_REPLACE_5       | neutral     |           12 |          -0.0029 |            0.0053 |                    -0.0082 |                      -0.0004 |                    0.5000 |                -0.0123 |
| U2_C_s2_residual_elasticity__R2B_REPLACE_5       | mild_up     |           11 |           0.0285 |            0.0406 |                    -0.0121 |                      -0.0014 |                    0.3636 |                -0.0182 |
| U2_C_s2_residual_elasticity__R2B_REPLACE_5       | strong_up   |           12 |           0.0818 |            0.0759 |                     0.0059 |                      -0.0030 |                    0.4167 |                -0.0356 |
| U2_C_s2_residual_elasticity__R2B_OVERLAY_10      | strong_down |           14 |          -0.0328 |           -0.0641 |                     0.0313 |                       0.0200 |                    0.7857 |                 0.0472 |
| U2_C_s2_residual_elasticity__R2B_OVERLAY_10      | mild_down   |           13 |          -0.0092 |            0.0178 |                    -0.0270 |                      -0.0099 |                    0.4615 |                -0.0214 |
| U2_C_s2_residual_elasticity__R2B_OVERLAY_10      | neutral     |           12 |          -0.0017 |            0.0068 |                    -0.0085 |                      -0.0098 |                    0.4167 |                -0.0123 |
| U2_C_s2_residual_elasticity__R2B_OVERLAY_10      | mild_up     |           11 |           0.0286 |            0.0416 |                    -0.0130 |                      -0.0128 |                    0.3636 |                -0.0177 |
| U2_C_s2_residual_elasticity__R2B_OVERLAY_10      | strong_up   |           12 |           0.0850 |            0.0736 |                     0.0114 |                      -0.0030 |                    0.4167 |                -0.0351 |

## 6. 行业暴露

| candidate_id                                     | trade_date          | industry   |   weight |
|:-------------------------------------------------|:--------------------|:-----------|---------:|
| U2_C_s2_residual_elasticity__R2B_OVERLAY_10      | 2026-04-13 00:00:00 | 银行       |   0.4000 |
| U2_C_s2_residual_elasticity__R2B_OVERLAY_10      | 2026-04-13 00:00:00 | 通信       |   0.1000 |
| U2_C_s2_residual_elasticity__R2B_OVERLAY_10      | 2026-04-13 00:00:00 | 公用事业   |   0.0500 |
| U2_A_industry_breadth_strength__R2B_REPLACE_5    | 2026-04-13 00:00:00 | 通信       |   0.1000 |
| U2_A_industry_breadth_strength__R2B_OVERLAY_10   | 2026-04-13 00:00:00 | 公用事业   |   0.0500 |
| U2_A_industry_breadth_strength__R2B_OVERLAY_10   | 2026-04-13 00:00:00 | 电子       |   0.0500 |
| U2_A_industry_breadth_strength__R2B_OVERLAY_10   | 2026-04-13 00:00:00 | 钢铁       |   0.0500 |
| U2_A_industry_breadth_strength__R2B_OVERLAY_10   | 2026-04-13 00:00:00 | 非银金融   |   0.0500 |
| U2_A_industry_breadth_strength__R2B_OVERLAY_10   | 2026-04-13 00:00:00 | 通信       |   0.1000 |
| U2_A_industry_breadth_strength__R2B_REPLACE_5    | 2026-04-13 00:00:00 | 石油石化   |   0.1000 |
| U2_A_industry_breadth_strength__R2B_REPLACE_5    | 2026-04-13 00:00:00 | 建筑装饰   |   0.1000 |
| U2_A_industry_breadth_strength__R2B_REPLACE_5    | 2026-04-13 00:00:00 | 交通运输   |   0.1000 |
| BASELINE_S2_FIXED                                | 2026-04-13 00:00:00 | 石油石化   |   0.1000 |
| BASELINE_S2_FIXED                                | 2026-04-13 00:00:00 | 建筑装饰   |   0.1000 |
| BASELINE_S2_FIXED                                | 2026-04-13 00:00:00 | 交通运输   |   0.1000 |
| BASELINE_S2_FIXED                                | 2026-04-13 00:00:00 | 银行       |   0.4000 |
| U2_A_industry_breadth_strength__R2B_REPLACE_3    | 2026-04-13 00:00:00 | 通信       |   0.1000 |
| U2_A_industry_breadth_strength__R2B_REPLACE_3    | 2026-04-13 00:00:00 | 公用事业   |   0.0500 |
| U2_A_industry_breadth_strength__R2B_REPLACE_3    | 2026-04-13 00:00:00 | 电子       |   0.0500 |
| U2_A_industry_breadth_strength__R2B_REPLACE_3    | 2026-04-13 00:00:00 | 钢铁       |   0.0500 |
| U2_A_industry_breadth_strength__R2B_REPLACE_3    | 2026-04-13 00:00:00 | 非银金融   |   0.0500 |
| U2_A_industry_breadth_strength__R2B_REPLACE_5    | 2026-04-13 00:00:00 | 银行       |   0.4000 |
| U2_B_tradable_breakout_expansion__R2B_REPLACE_3  | 2026-04-13 00:00:00 | 通信       |   0.1000 |
| U2_B_tradable_breakout_expansion__R2B_REPLACE_3  | 2026-04-13 00:00:00 | 公用事业   |   0.0500 |
| U2_B_tradable_breakout_expansion__R2B_REPLACE_3  | 2026-04-13 00:00:00 | 电子       |   0.0500 |
| U2_C_s2_residual_elasticity__R2B_REPLACE_3       | 2026-04-13 00:00:00 | 通信       |   0.1000 |
| U2_C_s2_residual_elasticity__R2B_REPLACE_3       | 2026-04-13 00:00:00 | 公用事业   |   0.0500 |
| U2_C_s2_residual_elasticity__R2B_REPLACE_3       | 2026-04-13 00:00:00 | 电子       |   0.0500 |
| U2_C_s2_residual_elasticity__R2B_REPLACE_3       | 2026-04-13 00:00:00 | 钢铁       |   0.0500 |
| U2_C_s2_residual_elasticity__R2B_REPLACE_3       | 2026-04-13 00:00:00 | 非银金融   |   0.0500 |
| U2_C_s2_residual_elasticity__R2B_REPLACE_5       | 2026-04-13 00:00:00 | 石油石化   |   0.1000 |
| U2_C_s2_residual_elasticity__R2B_REPLACE_5       | 2026-04-13 00:00:00 | 通信       |   0.1000 |
| U2_C_s2_residual_elasticity__R2B_REPLACE_5       | 2026-04-13 00:00:00 | 公用事业   |   0.0500 |
| U2_C_s2_residual_elasticity__R2B_REPLACE_5       | 2026-04-13 00:00:00 | 电子       |   0.0500 |
| U2_C_s2_residual_elasticity__R2B_REPLACE_5       | 2026-04-13 00:00:00 | 非银金融   |   0.0500 |
| U2_B_tradable_breakout_expansion__R2B_REPLACE_3  | 2026-04-13 00:00:00 | 钢铁       |   0.0500 |
| U2_B_tradable_breakout_expansion__R2B_REPLACE_3  | 2026-04-13 00:00:00 | 非银金融   |   0.0500 |
| U2_C_s2_residual_elasticity__R2B_OVERLAY_10      | 2026-04-13 00:00:00 | 钢铁       |   0.0500 |
| BASELINE_S2_FIXED                                | 2026-04-13 00:00:00 | 非银金融   |   0.0500 |
| U2_B_tradable_breakout_expansion__R2B_REPLACE_5  | 2026-04-13 00:00:00 | 通信       |   0.1000 |
| U2_B_tradable_breakout_expansion__R2B_REPLACE_5  | 2026-04-13 00:00:00 | 公用事业   |   0.0500 |
| U2_B_tradable_breakout_expansion__R2B_REPLACE_5  | 2026-04-13 00:00:00 | 电子       |   0.0500 |
| U2_B_tradable_breakout_expansion__R2B_REPLACE_5  | 2026-04-13 00:00:00 | 钢铁       |   0.0500 |
| U2_B_tradable_breakout_expansion__R2B_REPLACE_5  | 2026-04-13 00:00:00 | 非银金融   |   0.0500 |
| U2_B_tradable_breakout_expansion__R2B_OVERLAY_10 | 2026-04-13 00:00:00 | 银行       |   0.4000 |
| U2_B_tradable_breakout_expansion__R2B_OVERLAY_10 | 2026-04-13 00:00:00 | 交通运输   |   0.1000 |
| U2_B_tradable_breakout_expansion__R2B_OVERLAY_10 | 2026-04-13 00:00:00 | 建筑装饰   |   0.1000 |
| U2_B_tradable_breakout_expansion__R2B_OVERLAY_10 | 2026-04-13 00:00:00 | 石油石化   |   0.1000 |
| U2_B_tradable_breakout_expansion__R2B_OVERLAY_10 | 2026-04-13 00:00:00 | 公用事业   |   0.0500 |
| U2_C_s2_residual_elasticity__R2B_OVERLAY_10      | 2026-04-13 00:00:00 | 非银金融   |   0.0500 |

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
- `industry_map_source`: `akshare.stock_industry_clf_hist_sw+stock_industry_category_cninfo`
- `industry_map_source_status`: `real_industry_map`
- `industry_alpha_evidence_allowed`: `True`
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
