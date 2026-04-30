# R2 Regime-Aware Dual Sleeve v1 (Day 6-7)

- 生成时间：`2026-04-28T04:21:31.183521+00:00`
- 配置快照：`/mnt/ssd/lh/config.yaml.backtest`
- defensive sleeve：`S2 = vol_to_turnover`
- upside sleeve：`UPSIDE_C = limit_up_hits_20d + tail_strength_20d`
- 状态输入：上一已完成月份的 `regime / breadth`，避免当月事后调权
- 固定口径：`top_k=20` / `M` / `equal_weight` / `max_turnover=1.0` / `tplus1_open`
- `primary_decision_metric`: `daily_bt_like_proxy_annualized_excess_vs_market`
- Gate：`<0%`→reject / `0%~+3%`→gray_zone / `>=+3%`→full_backtest_candidate

## 1. Leaderboard

| candidate_id                           | label                                                                                  |   daily_proxy_annualized_excess_vs_market |   delta_vs_baseline_proxy | gate_decision   |   strong_up_median_excess |   delta_vs_baseline_strong_up_median_excess |   strong_up_positive_share |   delta_vs_baseline_strong_up_positive_share |   strong_down_median_excess |   delta_vs_baseline_strong_down_median_excess |   strong_up_switch_in_minus_out |   avg_turnover_half_l1 |   delta_vs_baseline_turnover |   n_rebalances |
|:---------------------------------------|:---------------------------------------------------------------------------------------|------------------------------------------:|--------------------------:|:----------------|--------------------------:|--------------------------------------------:|---------------------------:|---------------------------------------------:|----------------------------:|----------------------------------------------:|--------------------------------:|-----------------------:|-----------------------------:|---------------:|
| BASELINE_S2                            | S2 vol_to_turnover (defensive baseline)                                                |                                   -0.1280 |                    0.0000 | reject          |                   -0.0652 |                                      0.0000 |                     0.1538 |                                       0.0000 |                      0.0672 |                                        0.0000 |                         -0.1020 |                 0.0571 |                       0.0000 |             64 |
| DUAL_V1_80_20_TRIGGER_ONLY             | lagged strong_up/wide: defensive 80% + upside 20%; otherwise defensive 100%            |                                   -0.2299 |                   -0.1019 | reject          |                   -0.0859 |                                     -0.0207 |                     0.1538 |                                       0.0000 |                      0.0207 |                                       -0.0465 |                          0.0052 |                 0.1720 |                       0.1149 |             64 |
| DUAL_V2_60_40_MILD_85_15               | lagged strong_up/wide: 60/40; mild_up/mid: 85/15; neutral/down/narrow defensive        |                                   -0.3913 |                   -0.2634 | reject          |                   -0.0874 |                                     -0.0221 |                     0.0769 |                                      -0.0769 |                     -0.0042 |                                       -0.0714 |                         -0.0061 |                 0.2761 |                       0.2189 |             64 |
| DUAL_V3_40_60_MILD_70_30_NEUTRAL_90_10 | lagged strong_up/wide: 40/60; mild_up/mid: 70/30; neutral 90/10; down/narrow defensive |                                   -0.5197 |                   -0.3918 | reject          |                   -0.0973 |                                     -0.0320 |                     0.0769 |                                      -0.0769 |                     -0.0392 |                                       -0.1064 |                         -0.0320 |                 0.4335 |                       0.3764 |             64 |

## 2. R2 验收（对 BASELINE 的相对改善）

| candidate_id                           | status   | daily_proxy_>=_0   | strong_up_median_excess_improves   | strong_up_positive_share_improves   | strong_down_not_materially_worse   |
|:---------------------------------------|:---------|:-------------------|:-----------------------------------|:------------------------------------|:-----------------------------------|
| BASELINE_S2                            | baseline | nan                | nan                                | nan                                 | nan                                |
| DUAL_V1_80_20_TRIGGER_ONLY             | fail     | fail               | fail                               | fail                                | fail                               |
| DUAL_V2_60_40_MILD_85_15               | fail     | fail               | fail                               | fail                                | fail                               |
| DUAL_V3_40_60_MILD_70_30_NEUTRAL_90_10 | fail     | fail               | fail                               | fail                                | fail                               |

## 3. 状态权重触发统计

| candidate_id                           |   upside_weight |   rebalances |   avg_combined_names |
|:---------------------------------------|----------------:|-------------:|---------------------:|
| BASELINE_S2                            |          0.0000 |           64 |              20.0000 |
| DUAL_V1_80_20_TRIGGER_ONLY             |          0.0000 |           46 |              20.0000 |
| DUAL_V1_80_20_TRIGGER_ONLY             |          0.2000 |           18 |              40.0000 |
| DUAL_V2_60_40_MILD_85_15               |          0.0000 |           26 |              20.0000 |
| DUAL_V2_60_40_MILD_85_15               |          0.1500 |           20 |              39.9500 |
| DUAL_V2_60_40_MILD_85_15               |          0.4000 |           18 |              40.0000 |
| DUAL_V3_40_60_MILD_70_30_NEUTRAL_90_10 |          0.0000 |           26 |              20.0000 |
| DUAL_V3_40_60_MILD_70_30_NEUTRAL_90_10 |          0.3000 |           20 |              39.9500 |
| DUAL_V3_40_60_MILD_70_30_NEUTRAL_90_10 |          0.6000 |           18 |              40.0000 |

## 4. Regime 切片（candidate × regime）

| candidate_id                           | regime      |   months |   median_benchmark_return |   median_strategy_return |   median_excess_return |   positive_excess_share |   benchmark_compound |   strategy_compound |   capture_ratio |
|:---------------------------------------|:------------|---------:|--------------------------:|-------------------------:|-----------------------:|------------------------:|---------------------:|--------------------:|----------------:|
| BASELINE_S2                            | strong_down |       13 |                   -0.0551 |                  -0.0169 |                 0.0672 |                  0.9231 |              -0.6393 |             -0.0418 |          0.0654 |
| BASELINE_S2                            | mild_down   |       13 |                   -0.0125 |                   0.0014 |                 0.0056 |                  0.6923 |              -0.1549 |             -0.0915 |          0.5904 |
| BASELINE_S2                            | neutral     |       12 |                    0.0145 |                  -0.0054 |                -0.0161 |                  0.2500 |               0.1764 |              0.0033 |          0.0189 |
| BASELINE_S2                            | mild_up     |       13 |                    0.0418 |                   0.0071 |                -0.0402 |                  0.0769 |               0.7073 |              0.0789 |          0.1116 |
| BASELINE_S2                            | strong_up   |       13 |                    0.0773 |                   0.0142 |                -0.0652 |                  0.1538 |               2.0764 |              0.2692 |          0.1296 |
| DUAL_V1_80_20_TRIGGER_ONLY             | strong_down |       13 |                   -0.0551 |                  -0.0266 |                 0.0207 |                  0.8462 |              -0.6393 |             -0.3036 |          0.4748 |
| DUAL_V1_80_20_TRIGGER_ONLY             | mild_down   |       13 |                   -0.0125 |                  -0.0137 |                 0.0000 |                  0.5385 |              -0.1549 |             -0.2464 |          1.5907 |
| DUAL_V1_80_20_TRIGGER_ONLY             | neutral     |       12 |                    0.0145 |                  -0.0054 |                -0.0163 |                  0.2500 |               0.1764 |             -0.0100 |         -0.0565 |
| DUAL_V1_80_20_TRIGGER_ONLY             | mild_up     |       13 |                    0.0418 |                   0.0009 |                -0.0413 |                  0.0769 |               0.7073 |              0.0176 |          0.0249 |
| DUAL_V1_80_20_TRIGGER_ONLY             | strong_up   |       13 |                    0.0773 |                  -0.0019 |                -0.0859 |                  0.1538 |               2.0764 |              0.1987 |          0.0957 |
| DUAL_V2_60_40_MILD_85_15               | strong_down |       13 |                   -0.0551 |                  -0.0622 |                -0.0042 |                  0.4615 |              -0.6393 |             -0.5852 |          0.9152 |
| DUAL_V2_60_40_MILD_85_15               | mild_down   |       13 |                   -0.0125 |                  -0.0350 |                -0.0233 |                  0.2308 |              -0.1549 |             -0.4069 |          2.6270 |
| DUAL_V2_60_40_MILD_85_15               | neutral     |       12 |                    0.0145 |                  -0.0107 |                -0.0196 |                  0.1667 |               0.1764 |             -0.1597 |         -0.9053 |
| DUAL_V2_60_40_MILD_85_15               | mild_up     |       13 |                    0.0418 |                  -0.0069 |                -0.0444 |                  0.0000 |               0.7073 |             -0.1355 |         -0.1916 |
| DUAL_V2_60_40_MILD_85_15               | strong_up   |       13 |                    0.0773 |                  -0.0054 |                -0.0874 |                  0.0769 |               2.0764 |              0.0741 |          0.0357 |
| DUAL_V3_40_60_MILD_70_30_NEUTRAL_90_10 | strong_down |       13 |                   -0.0551 |                  -0.0975 |                -0.0392 |                  0.3846 |              -0.6393 |             -0.7543 |          1.1798 |
| DUAL_V3_40_60_MILD_70_30_NEUTRAL_90_10 | mild_down   |       13 |                   -0.0125 |                  -0.0522 |                -0.0428 |                  0.2308 |              -0.1549 |             -0.5332 |          3.4425 |
| DUAL_V3_40_60_MILD_70_30_NEUTRAL_90_10 | neutral     |       12 |                    0.0145 |                  -0.0152 |                -0.0289 |                  0.0833 |               0.1764 |             -0.2877 |         -1.6307 |
| DUAL_V3_40_60_MILD_70_30_NEUTRAL_90_10 | mild_up     |       13 |                    0.0418 |                  -0.0118 |                -0.0524 |                  0.0000 |               0.7073 |             -0.2673 |         -0.3779 |
| DUAL_V3_40_60_MILD_70_30_NEUTRAL_90_10 | strong_up   |       13 |                    0.0773 |                  -0.0083 |                -0.0973 |                  0.0769 |               2.0764 |             -0.0374 |         -0.0180 |

## 5. Breadth 切片（candidate × breadth）

| candidate_id                           | breadth   |   months |   median_benchmark_return |   median_strategy_return |   median_excess_return |   positive_excess_share |
|:---------------------------------------|:----------|---------:|--------------------------:|-------------------------:|-----------------------:|------------------------:|
| BASELINE_S2                            | narrow    |       19 |                   -0.0353 |                  -0.0169 |                 0.0184 |                  0.7895 |
| BASELINE_S2                            | mid       |       26 |                    0.0172 |                   0.0011 |                -0.0153 |                  0.3846 |
| BASELINE_S2                            | wide      |       19 |                    0.0695 |                   0.0085 |                -0.0507 |                  0.1053 |
| DUAL_V1_80_20_TRIGGER_ONLY             | narrow    |       19 |                   -0.0353 |                  -0.0206 |                 0.0184 |                  0.7368 |
| DUAL_V1_80_20_TRIGGER_ONLY             | mid       |       26 |                    0.0172 |                  -0.0063 |                -0.0165 |                  0.3462 |
| DUAL_V1_80_20_TRIGGER_ONLY             | wide      |       19 |                    0.0695 |                  -0.0019 |                -0.0634 |                  0.0526 |
| DUAL_V2_60_40_MILD_85_15               | narrow    |       19 |                   -0.0353 |                  -0.0350 |                -0.0054 |                  0.4211 |
| DUAL_V2_60_40_MILD_85_15               | mid       |       26 |                    0.0172 |                  -0.0139 |                -0.0326 |                  0.1154 |
| DUAL_V2_60_40_MILD_85_15               | wide      |       19 |                    0.0695 |                  -0.0123 |                -0.0687 |                  0.0526 |
| DUAL_V3_40_60_MILD_70_30_NEUTRAL_90_10 | narrow    |       19 |                   -0.0353 |                  -0.0529 |                -0.0316 |                  0.3684 |
| DUAL_V3_40_60_MILD_70_30_NEUTRAL_90_10 | mid       |       26 |                    0.0172 |                  -0.0152 |                -0.0454 |                  0.0769 |
| DUAL_V3_40_60_MILD_70_30_NEUTRAL_90_10 | wide      |       19 |                    0.0695 |                  -0.0236 |                -0.0790 |                  0.0526 |

## 6. 关键年份 strong_up（2021/2025/2026）

| candidate_id                           |   year | regime    |   months |   median_excess_return |   positive_excess_share |
|:---------------------------------------|-------:|:----------|---------:|-----------------------:|------------------------:|
| BASELINE_S2                            |   2021 | strong_up |        3 |                -0.0652 |                  0.3333 |
| BASELINE_S2                            |   2025 | strong_up |        2 |                -0.0784 |                  0.0000 |
| BASELINE_S2                            |   2026 | strong_up |        1 |                -0.1152 |                  0.0000 |
| DUAL_V1_80_20_TRIGGER_ONLY             |   2021 | strong_up |        3 |                -0.0652 |                  0.3333 |
| DUAL_V1_80_20_TRIGGER_ONLY             |   2025 | strong_up |        2 |                -0.0893 |                  0.0000 |
| DUAL_V1_80_20_TRIGGER_ONLY             |   2026 | strong_up |        1 |                -0.1152 |                  0.0000 |
| DUAL_V2_60_40_MILD_85_15               |   2021 | strong_up |        3 |                -0.0687 |                  0.0000 |
| DUAL_V2_60_40_MILD_85_15               |   2025 | strong_up |        2 |                -0.0996 |                  0.0000 |
| DUAL_V2_60_40_MILD_85_15               |   2026 | strong_up |        1 |                -0.1154 |                  0.0000 |
| DUAL_V3_40_60_MILD_70_30_NEUTRAL_90_10 |   2021 | strong_up |        3 |                -0.0716 |                  0.0000 |
| DUAL_V3_40_60_MILD_70_30_NEUTRAL_90_10 |   2025 | strong_up |        2 |                -0.1093 |                  0.0000 |
| DUAL_V3_40_60_MILD_70_30_NEUTRAL_90_10 |   2026 | strong_up |        1 |                -0.1157 |                  0.0000 |

## 7. Switch quality（candidate × regime）

| candidate_id                           | regime      |   rebalances |   mean_switch_in |   mean_switch_out |   mean_switch_in_minus_out |   median_switch_in_minus_out |   switch_in_winning_share |   mean_topk_minus_next |
|:---------------------------------------|:------------|-------------:|-----------------:|------------------:|---------------------------:|-----------------------------:|--------------------------:|-----------------------:|
| BASELINE_S2                            | strong_down |            2 |          -0.0207 |           -0.0143 |                    -0.0064 |                      -0.0064 |                    0.5000 |                 0.0194 |
| BASELINE_S2                            | mild_down   |            1 |          -0.0137 |           -0.0452 |                     0.0315 |                       0.0315 |                    1.0000 |                 0.0022 |
| BASELINE_S2                            | neutral     |            5 |           0.0369 |            0.0269 |                     0.0100 |                      -0.0109 |                    0.4000 |                -0.0014 |
| BASELINE_S2                            | mild_up     |            4 |           0.0354 |            0.0306 |                     0.0048 |                      -0.0158 |                    0.2500 |                 0.0070 |
| BASELINE_S2                            | strong_up   |            2 |           0.0014 |            0.1034 |                    -0.1020 |                      -0.1020 |                    0.0000 |                -0.0590 |
| DUAL_V1_80_20_TRIGGER_ONLY             | strong_down |            9 |          -0.0400 |           -0.0147 |                    -0.0252 |                      -0.0348 |                    0.4444 |                -0.0036 |
| DUAL_V1_80_20_TRIGGER_ONLY             | mild_down   |           10 |          -0.0295 |           -0.0156 |                    -0.0139 |                      -0.0230 |                    0.3000 |                -0.0084 |
| DUAL_V1_80_20_TRIGGER_ONLY             | neutral     |            7 |           0.0195 |            0.0266 |                    -0.0072 |                      -0.0246 |                    0.2857 |                -0.0029 |
| DUAL_V1_80_20_TRIGGER_ONLY             | mild_up     |            6 |          -0.0160 |            0.0110 |                    -0.0270 |                      -0.0141 |                    0.5000 |                -0.0072 |
| DUAL_V1_80_20_TRIGGER_ONLY             | strong_up   |            7 |           0.0467 |            0.0414 |                     0.0052 |                       0.0311 |                    0.5714 |                 0.0011 |
| DUAL_V2_60_40_MILD_85_15               | strong_down |           13 |          -0.0705 |           -0.0380 |                    -0.0325 |                      -0.0280 |                    0.2308 |                -0.0232 |
| DUAL_V2_60_40_MILD_85_15               | mild_down   |           12 |          -0.0128 |           -0.0090 |                    -0.0038 |                       0.0040 |                    0.5833 |                 0.0012 |
| DUAL_V2_60_40_MILD_85_15               | neutral     |           11 |           0.0219 |            0.0123 |                     0.0096 |                       0.0109 |                    0.5455 |                -0.0026 |
| DUAL_V2_60_40_MILD_85_15               | mild_up     |           10 |          -0.0076 |            0.0103 |                    -0.0179 |                      -0.0081 |                    0.4000 |                -0.0085 |
| DUAL_V2_60_40_MILD_85_15               | strong_up   |           11 |           0.0733 |            0.0794 |                    -0.0061 |                       0.0272 |                    0.5455 |                -0.0148 |
| DUAL_V3_40_60_MILD_70_30_NEUTRAL_90_10 | strong_down |           13 |          -0.0943 |           -0.0379 |                    -0.0564 |                      -0.0584 |                    0.3077 |                -0.0403 |
| DUAL_V3_40_60_MILD_70_30_NEUTRAL_90_10 | mild_down   |           12 |          -0.0180 |           -0.0151 |                    -0.0029 |                      -0.0104 |                    0.5000 |                -0.0057 |
| DUAL_V3_40_60_MILD_70_30_NEUTRAL_90_10 | neutral     |           11 |           0.0008 |            0.0143 |                    -0.0135 |                      -0.0109 |                    0.4545 |                -0.0138 |
| DUAL_V3_40_60_MILD_70_30_NEUTRAL_90_10 | mild_up     |           10 |          -0.0185 |            0.0066 |                    -0.0251 |                      -0.0087 |                    0.5000 |                -0.0258 |
| DUAL_V3_40_60_MILD_70_30_NEUTRAL_90_10 | strong_up   |           11 |           0.0757 |            0.1077 |                    -0.0320 |                      -0.0237 |                    0.4545 |                -0.0068 |

## 8. 结论 / 下一步

- **没有双袖套候选同时满足 R2 验收四条**。
- 三组双袖套 daily proxy 仍低于 0%，按 R0 不补正式 full backtest。
- 若保守 80/20 仍无法改善 strong_up 且 proxy 为负，下一步应回到候选层重做 upside 输入，而不是继续加大 sleeve 权重。

## 9. 产出文件

- `data/results/p2_regime_aware_dual_sleeve_v1_2026-04-28_leaderboard.csv`
- `data/results/p2_regime_aware_dual_sleeve_v1_2026-04-28_regime_long.csv`
- `data/results/p2_regime_aware_dual_sleeve_v1_2026-04-28_breadth_long.csv`
- `data/results/p2_regime_aware_dual_sleeve_v1_2026-04-28_year_long.csv`
- `data/results/p2_regime_aware_dual_sleeve_v1_2026-04-28_switch_long.csv`
- `data/results/p2_regime_aware_dual_sleeve_v1_2026-04-28_state_diag_long.csv`
- `data/results/p2_regime_aware_dual_sleeve_v1_2026-04-28_monthly_long.csv`
- `data/results/p2_regime_aware_dual_sleeve_v1_2026-04-28_summary.json`

## 10. 配置参数

- `start`: `2021-01-01`
- `end`: `2026-04-28`
- `top_k`: `20`
- `rebalance_rule`: `M`
- `portfolio_method`: `dual_sleeve_blended_equal_weight`
- `max_turnover`: `1.0`
- `execution_mode`: `tplus1_open`
- `state_lag`: `previous_completed_month`
- `defensive_sleeve`: `vol_to_turnover`
- `upside_sleeve`: `limit_up_hits_20d + tail_strength_20d`
- `prefilter`: `{'enabled': False, 'limit_move_max': 2, 'turnover_low_pct': 0.1, 'turnover_high_pct': 0.98, 'price_position_high_pct': 0.9}`
- `universe_filter`: `{'enabled': True, 'min_amount_20d': 50000000, 'require_roe_ttm_positive': True}`
- `benchmark_symbol`: `market_ew_proxy`
- `benchmark_min_history_days`: `551`
- `config_source`: `/mnt/ssd/lh/config.yaml.backtest`
- `p1_experiment_mode`: `daily_proxy_first`
- `legacy_proxy_decision_role`: `diagnostic_only`
- `primary_decision_metric`: `daily_bt_like_proxy_annualized_excess_vs_market`
- `gate_thresholds`: `{'reject': 0.0, 'full_backtest': 0.03}`
- `defensive_weight_diag_rows`: `64`
- `upside_weight_diag_rows`: `64`
