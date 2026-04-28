# R1F Fixed Baseline 最小复跑

- 生成时间：`2026-04-28T06:35:05.547082+00:00`
- 配置快照：`/mnt/ssd/lh/config.yaml.backtest`
- `eval_contract_version`: `r0_eval_execution_contract_2026-04-28`
- `execution_contract_version`: `tplus1_open_buy_delta_limit_mask_2026-04-28`
- 固定候选：`BASELINE_S2_FIXED` / `UPSIDE_C_FIXED` / `DUAL_V1_FIXED`
- 固定口径：`top_k=20` / `M` / `equal_weight` / `max_turnover=1.0` / `tplus1_open`
- primary benchmark：`open_to_open`；comparison benchmark：`close_to_close`

## 1. Fixed Leaderboard

| candidate_id      |   daily_proxy_annualized_excess_vs_market |   delta_vs_baseline_proxy | gate_decision   |   strong_up_median_excess |   delta_vs_baseline_strong_up_median_excess |   strong_up_positive_share |   strong_down_median_excess |   delta_vs_baseline_strong_down_median_excess |   strong_up_switch_in_minus_out |   avg_turnover_half_l1 |   n_rebalances |
|:------------------|------------------------------------------:|--------------------------:|:----------------|--------------------------:|--------------------------------------------:|---------------------------:|----------------------------:|----------------------------------------------:|--------------------------------:|-----------------------:|---------------:|
| BASELINE_S2_FIXED |                                   -0.0859 |                    0.0000 | reject          |                   -0.0612 |                                      0.0000 |                     0.1538 |                      0.0629 |                                        0.0000 |                         -0.0237 |                 0.0844 |             64 |
| DUAL_V1_FIXED     |                                   -0.1151 |                   -0.0291 | reject          |                   -0.0624 |                                     -0.0012 |                     0.1538 |                      0.0484 |                                       -0.0145 |                          0.0446 |                 0.1512 |             64 |
| UPSIDE_C_FIXED    |                                   -0.4772 |                   -0.3913 | reject          |                   -0.0618 |                                     -0.0007 |                     0.1538 |                     -0.0813 |                                       -0.1442 |                         -0.0489 |                 0.7520 |             64 |

## 2. 修复前后对照

| candidate_id      |   old_daily_proxy_annualized_excess_vs_market |   fixed_daily_proxy_annualized_excess_vs_market |   delta_daily_proxy_annualized_excess_vs_market |   old_strong_up_median_excess |   fixed_strong_up_median_excess |   delta_strong_up_median_excess |   old_strong_down_median_excess |   fixed_strong_down_median_excess |   delta_strong_down_median_excess |   old_strong_up_switch_in_minus_out |   fixed_strong_up_switch_in_minus_out |   delta_strong_up_switch_in_minus_out |   fixed_buy_fail_total_weight | fixed_primary_benchmark_return_mode   |
|:------------------|----------------------------------------------:|------------------------------------------------:|------------------------------------------------:|------------------------------:|--------------------------------:|--------------------------------:|--------------------------------:|----------------------------------:|----------------------------------:|------------------------------------:|--------------------------------------:|--------------------------------------:|------------------------------:|:--------------------------------------|
| BASELINE_S2_FIXED |                                       -0.1280 |                                         -0.0859 |                                          0.0420 |                       -0.0652 |                         -0.0612 |                          0.0041 |                          0.0672 |                            0.0629 |                           -0.0043 |                             -0.1020 |                               -0.0237 |                                0.0783 |                        0.0500 | open_to_open                          |
| UPSIDE_C_FIXED    |                                       -0.8662 |                                         -0.4772 |                                          0.3890 |                       -0.1525 |                         -0.0618 |                          0.0906 |                         -0.1383 |                           -0.0813 |                            0.0570 |                              0.0029 |                               -0.0489 |                               -0.0518 |                        1.5500 | open_to_open                          |
| DUAL_V1_FIXED     |                                       -0.2299 |                                         -0.1151 |                                          0.1148 |                       -0.0859 |                         -0.0624 |                          0.0236 |                          0.0207 |                            0.0484 |                            0.0277 |                              0.0052 |                                0.0446 |                                0.0394 |                        0.1300 | open_to_open                          |

## 3. R1F 判定

| candidate_id      | status   | daily_proxy_>=_0   | strong_up_median_excess_improves   | strong_up_positive_share_improves   | strong_down_not_materially_worse   |
|:------------------|:---------|:-------------------|:-----------------------------------|:------------------------------------|:-----------------------------------|
| BASELINE_S2_FIXED | baseline | nan                | nan                                | nan                                 | nan                                |
| DUAL_V1_FIXED     | fail     | fail               | fail                               | fail                                | pass                               |
| UPSIDE_C_FIXED    | fail     | fail               | fail                               | fail                                | fail                               |

- `UPSIDE_C_FIXED` 与 `DUAL_V1_FIXED` 仍为 reject，旧 R2 失败结论成立。
- 无候选达到 `>= +3%`，不补正式 full backtest。
- 后续应进入 R2B：重做更可交易的 upside 输入，而不是扩大 dual sleeve 权重网格。

## 4. Regime 切片

| candidate_id      | regime      |   months |   median_benchmark_return |   median_strategy_return |   median_excess_return |   positive_excess_share |   benchmark_compound |   strategy_compound |   capture_ratio |
|:------------------|:------------|---------:|--------------------------:|-------------------------:|-----------------------:|------------------------:|---------------------:|--------------------:|----------------:|
| BASELINE_S2_FIXED | strong_down |       13 |                   -0.0604 |                  -0.0207 |                 0.0629 |                  0.7692 |              -0.6375 |             -0.2342 |          0.3674 |
| BASELINE_S2_FIXED | mild_down   |       13 |                   -0.0109 |                   0.0014 |                 0.0210 |                  0.7692 |              -0.1702 |              0.0771 |         -0.4527 |
| BASELINE_S2_FIXED | neutral     |       12 |                    0.0087 |                   0.0010 |                -0.0105 |                  0.4167 |               0.1358 |              0.0213 |          0.1567 |
| BASELINE_S2_FIXED | mild_up     |       13 |                    0.0378 |                   0.0121 |                -0.0258 |                  0.0769 |               0.6459 |              0.1512 |          0.2342 |
| BASELINE_S2_FIXED | strong_up   |       13 |                    0.0811 |                   0.0199 |                -0.0612 |                  0.1538 |               2.5282 |              0.6416 |          0.2538 |
| UPSIDE_C_FIXED    | strong_down |       13 |                   -0.0604 |                  -0.1539 |                -0.0813 |                  0.3077 |              -0.6375 |             -0.8603 |          1.3495 |
| UPSIDE_C_FIXED    | mild_down   |       13 |                   -0.0109 |                  -0.0343 |                -0.0260 |                  0.3846 |              -0.1702 |             -0.4957 |          2.9120 |
| UPSIDE_C_FIXED    | neutral     |       12 |                    0.0087 |                  -0.0091 |                -0.0236 |                  0.3333 |               0.1358 |             -0.2417 |         -1.7791 |
| UPSIDE_C_FIXED    | mild_up     |       13 |                    0.0378 |                  -0.0488 |                -0.0865 |                  0.2308 |               0.6459 |             -0.2946 |         -0.4561 |
| UPSIDE_C_FIXED    | strong_up   |       13 |                    0.0811 |                   0.0248 |                -0.0618 |                  0.1538 |               2.5282 |              0.9458 |          0.3741 |
| DUAL_V1_FIXED     | strong_down |       13 |                   -0.0604 |                  -0.0210 |                 0.0484 |                  0.6923 |              -0.6375 |             -0.3058 |          0.4796 |
| DUAL_V1_FIXED     | mild_down   |       13 |                   -0.0109 |                  -0.0002 |                 0.0109 |                  0.6923 |              -0.1702 |              0.0086 |         -0.0507 |
| DUAL_V1_FIXED     | neutral     |       12 |                    0.0087 |                   0.0010 |                -0.0106 |                  0.4167 |               0.1358 |              0.0300 |          0.2211 |
| DUAL_V1_FIXED     | mild_up     |       13 |                    0.0378 |                   0.0092 |                -0.0258 |                  0.0769 |               0.6459 |              0.1425 |          0.2206 |
| DUAL_V1_FIXED     | strong_up   |       13 |                    0.0811 |                   0.0199 |                -0.0624 |                  0.1538 |               2.5282 |              0.6155 |          0.2434 |

## 5. Switch quality

| candidate_id      | regime      |   rebalances |   mean_switch_in |   mean_switch_out |   mean_switch_in_minus_out |   median_switch_in_minus_out |   switch_in_winning_share |   mean_topk_minus_next |
|:------------------|:------------|-------------:|-----------------:|------------------:|---------------------------:|-----------------------------:|--------------------------:|-----------------------:|
| BASELINE_S2_FIXED | strong_down |            3 |          -0.0504 |            0.0139 |                    -0.0644 |                      -0.0176 |                    0.3333 |                 0.0034 |
| BASELINE_S2_FIXED | mild_down   |            1 |          -0.0137 |           -0.0452 |                     0.0315 |                       0.0315 |                    1.0000 |                 0.0022 |
| BASELINE_S2_FIXED | neutral     |            5 |           0.0266 |            0.0089 |                     0.0177 |                      -0.0081 |                    0.4000 |                 0.0090 |
| BASELINE_S2_FIXED | mild_up     |            4 |           0.0482 |            0.0531 |                    -0.0049 |                      -0.0298 |                    0.2500 |                -0.0060 |
| BASELINE_S2_FIXED | strong_up   |            1 |           0.1127 |            0.1364 |                    -0.0237 |                      -0.0237 |                    0.0000 |                -0.0895 |
| UPSIDE_C_FIXED    | strong_down |           14 |          -0.1276 |           -0.0682 |                    -0.0594 |                      -0.0526 |                    0.1429 |                -0.0358 |
| UPSIDE_C_FIXED    | mild_down   |           13 |          -0.0471 |           -0.0517 |                     0.0046 |                       0.0069 |                    0.6923 |                -0.0404 |
| UPSIDE_C_FIXED    | neutral     |           12 |          -0.0183 |           -0.0280 |                     0.0098 |                       0.0026 |                    0.5000 |                 0.0035 |
| UPSIDE_C_FIXED    | mild_up     |           11 |          -0.0261 |           -0.0237 |                    -0.0024 |                      -0.0014 |                    0.4545 |                -0.0271 |
| UPSIDE_C_FIXED    | strong_up   |           12 |           0.0549 |            0.1039 |                    -0.0489 |                      -0.0371 |                    0.1667 |                -0.0134 |
| DUAL_V1_FIXED     | strong_down |            8 |          -0.0529 |           -0.0163 |                    -0.0366 |                      -0.0422 |                    0.3750 |                -0.0060 |
| DUAL_V1_FIXED     | mild_down   |            9 |          -0.0167 |           -0.0009 |                    -0.0158 |                      -0.0296 |                    0.3333 |                -0.0096 |
| DUAL_V1_FIXED     | neutral     |            8 |           0.0093 |            0.0133 |                    -0.0040 |                      -0.0189 |                    0.3750 |                 0.0040 |
| DUAL_V1_FIXED     | mild_up     |            6 |          -0.0113 |            0.0154 |                    -0.0267 |                      -0.0125 |                    0.5000 |                -0.0208 |
| DUAL_V1_FIXED     | strong_up   |            4 |           0.0927 |            0.0481 |                     0.0446 |                       0.0105 |                    0.7500 |                 0.0098 |

## 6. 产出文件

- `data/results/r1f_fixed_baseline_rerun_2026-04-28_leaderboard.csv`
- `data/results/r1f_fixed_baseline_rerun_2026-04-28_comparison.csv`
- `data/results/r1f_fixed_baseline_rerun_2026-04-28_regime_long.csv`
- `data/results/r1f_fixed_baseline_rerun_2026-04-28_breadth_long.csv`
- `data/results/r1f_fixed_baseline_rerun_2026-04-28_year_long.csv`
- `data/results/r1f_fixed_baseline_rerun_2026-04-28_switch_long.csv`
- `data/results/r1f_fixed_baseline_rerun_2026-04-28_monthly_long.csv`
- `data/results/r1f_fixed_baseline_rerun_2026-04-28_state_diag_long.csv`
- `data/results/r1f_fixed_baseline_rerun_2026-04-28_lagged_state_by_rebalance.csv`
- `data/results/r1f_fixed_baseline_rerun_2026-04-28_summary.json`

## 7. 配置参数

- `start`: `2021-01-01`
- `end`: `2026-04-28`
- `top_k`: `20`
- `rebalance_rule`: `M`
- `portfolio_method`: `fixed_baseline_minimal_rerun`
- `max_turnover`: `1.0`
- `execution_mode`: `tplus1_open`
- `state_lag`: `previous_completed_month`
- `state_threshold_mode`: `expanding`
- `defensive_sleeve`: `vol_to_turnover`
- `upside_sleeve`: `limit_up_hits_20d + tail_strength_20d`
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
- `upside_weight_diag_rows`: `64`
