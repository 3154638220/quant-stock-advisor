# R2B v2 Edge-Gated Replacement

- 生成时间：`2026-04-28T10:01:52.848245+00:00`
- 配置快照：`/mnt/ssd/lh/config.yaml.backtest`
- `eval_contract_version`: `r0_eval_execution_contract_2026-04-28`
- `execution_contract_version`: `tplus1_open_buy_delta_limit_mask_2026-04-28`
- `industry_map_source`: `akshare.stock_industry_clf_hist_sw+stock_industry_category_cninfo`
- `industry_map_source_status`: `real_industry_map`
- 组合表达：`S2 defensive core + edge-gated replacement`；每月替换 `0/1/2/3`，不默认填满 slot。
- pair gate 只使用当期可观测 feature；`realized_pair_edge` 只用于事后诊断。

## 1. Leaderboard

| candidate_id                                    |   daily_proxy_annualized_excess_vs_market |   delta_vs_baseline_proxy | gate_decision   |   strong_up_median_excess |   delta_vs_baseline_strong_up_median_excess |   strong_up_positive_share |   delta_vs_baseline_strong_up_positive_share |   strong_down_median_excess |   delta_vs_baseline_strong_down_median_excess |   strong_up_switch_in_minus_out |   strong_up_topk_minus_next |   avg_turnover_half_l1 |   delta_vs_baseline_turnover | industry_map_source_status   |   n_rebalances |
|:------------------------------------------------|------------------------------------------:|--------------------------:|:----------------|--------------------------:|--------------------------------------------:|---------------------------:|---------------------------------------------:|----------------------------:|----------------------------------------------:|--------------------------------:|----------------------------:|-----------------------:|-----------------------------:|:-----------------------------|---------------:|
| BASELINE_S2_FIXED                               |                                   -0.0859 |                    0.0000 | reject          |                   -0.0612 |                                      0.0000 |                     0.1538 |                                       0.0000 |                      0.0629 |                                        0.0000 |                         -0.0237 |                     -0.0895 |                 0.0844 |                       0.0000 | real_industry_map            |             64 |
| U3_A_real_industry_leadership__EDGE_GATED       |                                   -0.0834 |                    0.0025 | reject          |                   -0.0616 |                                     -0.0004 |                     0.1538 |                                       0.0000 |                      0.0629 |                                       -0.0000 |                          0.0235 |                     -0.0157 |                 0.1318 |                       0.0474 | real_industry_map            |             64 |
| U3_C_pairwise_residual_edge__EDGE_GATED         |                                   -0.0925 |                   -0.0066 | reject          |                   -0.0612 |                                      0.0000 |                     0.2308 |                                       0.0769 |                      0.0608 |                                       -0.0022 |                          0.0690 |                     -0.0174 |                 0.1397 |                       0.0554 | real_industry_map            |             64 |
| U3_B_buyable_leadership_persistence__EDGE_GATED |                                   -0.0982 |                   -0.0123 | reject          |                   -0.0612 |                                      0.0000 |                     0.1538 |                                       0.0000 |                      0.0470 |                                       -0.0159 |                         -0.0110 |                     -0.0337 |                 0.1333 |                       0.0490 | real_industry_map            |             64 |

## 2. R2B v2 验收

| candidate_id                                    | status    | daily_proxy_not_below_baseline   | daily_proxy_>=_0   | strong_up_median_+2pct_or_positive_share_+10pct   | strong_down_not_worse_than_-2pct   | turnover_delta_<=_+0.10   | strong_up_switch_not_negative   |
|:------------------------------------------------|:----------|:---------------------------------|:-------------------|:--------------------------------------------------|:-----------------------------------|:--------------------------|:--------------------------------|
| BASELINE_S2_FIXED                               | baseline  | nan                              | nan                | nan                                               | nan                                | nan                       | nan                             |
| U3_A_real_industry_leadership__EDGE_GATED       | gray_zone | pass                             | fail               | fail                                              | pass                               | pass                      | pass                            |
| U3_C_pairwise_residual_edge__EDGE_GATED         | fail      | fail                             | fail               | fail                                              | pass                               | pass                      | pass                            |
| U3_B_buyable_leadership_persistence__EDGE_GATED | fail      | fail                             | fail               | fail                                              | pass                               | pass                      | fail                            |

## 3. Replacement Count Distribution

| candidate_id                                    |   replacement_count |   months |   month_share |
|:------------------------------------------------|--------------------:|---------:|--------------:|
| U3_A_real_industry_leadership__EDGE_GATED       |                   0 |       53 |        0.8281 |
| U3_A_real_industry_leadership__EDGE_GATED       |                   3 |       11 |        0.1719 |
| U3_B_buyable_leadership_persistence__EDGE_GATED |                   0 |       48 |        0.7500 |
| U3_B_buyable_leadership_persistence__EDGE_GATED |                   2 |        2 |        0.0312 |
| U3_B_buyable_leadership_persistence__EDGE_GATED |                   3 |       14 |        0.2188 |
| U3_C_pairwise_residual_edge__EDGE_GATED         |                   0 |       47 |        0.7344 |
| U3_C_pairwise_residual_edge__EDGE_GATED         |                   3 |       17 |        0.2656 |

## 4. Selected Pair Diagnostics

| candidate_id                                    | state_gate                       | old_pool    | candidate_pool       |   selected_pairs |   avg_pair_edge_score |   avg_expected_edge_after_cost |   avg_realized_pair_edge |   realized_win_rate |
|:------------------------------------------------|:---------------------------------|:------------|:---------------------|-----------------:|----------------------:|-------------------------------:|-------------------------:|--------------------:|
| U3_A_real_industry_leadership__EDGE_GATED       | state_strong_up_and_wide         | S2_bottom_3 | candidate_top_pct_95 |               33 |                0.8121 |                         0.1321 |                   0.0205 |              0.4848 |
| U3_B_buyable_leadership_persistence__EDGE_GATED | state_up_or_wide_not_strong_down | S2_bottom_3 | candidate_top_pct_90 |               46 |                0.8287 |                         0.1687 |                  -0.0246 |              0.3478 |
| U3_C_pairwise_residual_edge__EDGE_GATED         | state_strong_up_or_wide          | S2_bottom_5 | candidate_buyable    |               51 |                0.9044 |                         0.1844 |                  -0.0070 |              0.3922 |

## 5. Regime / Breadth

| candidate_id                                    | regime      |   months |   median_benchmark_return |   median_strategy_return |   median_excess_return |   positive_excess_share |   benchmark_compound |   strategy_compound |   capture_ratio |
|:------------------------------------------------|:------------|---------:|--------------------------:|-------------------------:|-----------------------:|------------------------:|---------------------:|--------------------:|----------------:|
| BASELINE_S2_FIXED                               | strong_down |       13 |                   -0.0604 |                  -0.0207 |                 0.0629 |                  0.7692 |              -0.6375 |             -0.2342 |          0.3674 |
| BASELINE_S2_FIXED                               | mild_down   |       13 |                   -0.0109 |                   0.0014 |                 0.0210 |                  0.7692 |              -0.1702 |              0.0771 |         -0.4527 |
| BASELINE_S2_FIXED                               | neutral     |       12 |                    0.0087 |                   0.0010 |                -0.0105 |                  0.4167 |               0.1358 |              0.0213 |          0.1567 |
| BASELINE_S2_FIXED                               | mild_up     |       13 |                    0.0378 |                   0.0121 |                -0.0258 |                  0.0769 |               0.6459 |              0.1512 |          0.2342 |
| BASELINE_S2_FIXED                               | strong_up   |       13 |                    0.0811 |                   0.0199 |                -0.0612 |                  0.1538 |               2.5282 |              0.6416 |          0.2538 |
| U3_A_real_industry_leadership__EDGE_GATED       | strong_down |       13 |                   -0.0604 |                  -0.0207 |                 0.0629 |                  0.7692 |              -0.6375 |             -0.2347 |          0.3681 |
| U3_A_real_industry_leadership__EDGE_GATED       | mild_down   |       13 |                   -0.0109 |                   0.0005 |                 0.0071 |                  0.8462 |              -0.1702 |              0.0963 |         -0.5660 |
| U3_A_real_industry_leadership__EDGE_GATED       | neutral     |       12 |                    0.0087 |                   0.0010 |                -0.0105 |                  0.4167 |               0.1358 |              0.0135 |          0.0992 |
| U3_A_real_industry_leadership__EDGE_GATED       | mild_up     |       13 |                    0.0378 |                   0.0087 |                -0.0258 |                  0.0769 |               0.6459 |              0.1641 |          0.2541 |
| U3_A_real_industry_leadership__EDGE_GATED       | strong_up   |       13 |                    0.0811 |                   0.0199 |                -0.0616 |                  0.1538 |               2.5282 |              0.6157 |          0.2435 |
| U3_B_buyable_leadership_persistence__EDGE_GATED | strong_down |       13 |                   -0.0604 |                  -0.0210 |                 0.0470 |                  0.6923 |              -0.6375 |             -0.2834 |          0.4445 |
| U3_B_buyable_leadership_persistence__EDGE_GATED | mild_down   |       13 |                   -0.0109 |                   0.0000 |                 0.0098 |                  0.7692 |              -0.1702 |              0.0305 |         -0.1794 |
| U3_B_buyable_leadership_persistence__EDGE_GATED | neutral     |       12 |                    0.0087 |                   0.0010 |                -0.0106 |                  0.4167 |               0.1358 |             -0.0016 |         -0.0117 |
| U3_B_buyable_leadership_persistence__EDGE_GATED | mild_up     |       13 |                    0.0378 |                   0.0121 |                -0.0258 |                  0.0769 |               0.6459 |              0.1760 |          0.2726 |
| U3_B_buyable_leadership_persistence__EDGE_GATED | strong_up   |       13 |                    0.0811 |                   0.0199 |                -0.0612 |                  0.1538 |               2.5282 |              0.6943 |          0.2746 |
| U3_C_pairwise_residual_edge__EDGE_GATED         | strong_down |       13 |                   -0.0604 |                  -0.0198 |                 0.0608 |                  0.6923 |              -0.6375 |             -0.2589 |          0.4061 |
| U3_C_pairwise_residual_edge__EDGE_GATED         | mild_down   |       13 |                   -0.0109 |                   0.0000 |                 0.0087 |                  0.7692 |              -0.1702 |              0.0147 |         -0.0863 |
| U3_C_pairwise_residual_edge__EDGE_GATED         | neutral     |       12 |                    0.0087 |                   0.0010 |                -0.0088 |                  0.3333 |               0.1358 |             -0.0058 |         -0.0425 |
| U3_C_pairwise_residual_edge__EDGE_GATED         | mild_up     |       13 |                    0.0378 |                   0.0087 |                -0.0258 |                  0.0769 |               0.6459 |              0.1633 |          0.2529 |
| U3_C_pairwise_residual_edge__EDGE_GATED         | strong_up   |       13 |                    0.0811 |                   0.0228 |                -0.0612 |                  0.2308 |               2.5282 |              0.7404 |          0.2929 |

| candidate_id                                    | breadth   |   months |   median_benchmark_return |   median_strategy_return |   median_excess_return |   positive_excess_share |
|:------------------------------------------------|:----------|---------:|--------------------------:|-------------------------:|-----------------------:|------------------------:|
| BASELINE_S2_FIXED                               | narrow    |       19 |                   -0.0345 |                  -0.0137 |                 0.0249 |                  0.7895 |
| BASELINE_S2_FIXED                               | mid       |       26 |                    0.0150 |                   0.0064 |                -0.0043 |                  0.4231 |
| BASELINE_S2_FIXED                               | wide      |       19 |                    0.0678 |                   0.0121 |                -0.0470 |                  0.1053 |
| U3_A_real_industry_leadership__EDGE_GATED       | narrow    |       19 |                   -0.0345 |                  -0.0137 |                 0.0246 |                  0.7895 |
| U3_A_real_industry_leadership__EDGE_GATED       | mid       |       26 |                    0.0150 |                   0.0075 |                -0.0030 |                  0.4615 |
| U3_A_real_industry_leadership__EDGE_GATED       | wide      |       19 |                    0.0678 |                   0.0091 |                -0.0596 |                  0.1053 |
| U3_B_buyable_leadership_persistence__EDGE_GATED | narrow    |       19 |                   -0.0345 |                  -0.0169 |                 0.0210 |                  0.7368 |
| U3_B_buyable_leadership_persistence__EDGE_GATED | mid       |       26 |                    0.0150 |                   0.0061 |                -0.0044 |                  0.4231 |
| U3_B_buyable_leadership_persistence__EDGE_GATED | wide      |       19 |                    0.0678 |                   0.0121 |                -0.0416 |                  0.1053 |
| U3_C_pairwise_residual_edge__EDGE_GATED         | narrow    |       19 |                   -0.0345 |                  -0.0137 |                 0.0254 |                  0.7368 |
| U3_C_pairwise_residual_edge__EDGE_GATED         | mid       |       26 |                    0.0150 |                   0.0081 |                -0.0045 |                  0.3846 |
| U3_C_pairwise_residual_edge__EDGE_GATED         | wide      |       19 |                    0.0678 |                   0.0088 |                -0.0540 |                  0.1579 |

## 6. 2021 / 2025 / 2026 Strong-Up

| candidate_id                                    |   year |   months |   median_excess_return |   delta_strong_up_median_excess |   positive_excess_share |   delta_strong_up_positive_share |
|:------------------------------------------------|-------:|---------:|-----------------------:|--------------------------------:|------------------------:|---------------------------------:|
| U3_A_real_industry_leadership__EDGE_GATED       |   2021 |        3 |                -0.0596 |                          0.0000 |                  0.3333 |                           0.0000 |
| U3_A_real_industry_leadership__EDGE_GATED       |   2025 |        3 |                -0.0740 |                         -0.0014 |                  0.0000 |                           0.0000 |
| U3_A_real_industry_leadership__EDGE_GATED       |   2026 |        1 |                -0.1039 |                          0.0000 |                  0.0000 |                           0.0000 |
| U3_B_buyable_leadership_persistence__EDGE_GATED |   2021 |        3 |                -0.0596 |                          0.0000 |                  0.3333 |                           0.0000 |
| U3_B_buyable_leadership_persistence__EDGE_GATED |   2025 |        3 |                -0.0304 |                          0.0421 |                  0.0000 |                           0.0000 |
| U3_B_buyable_leadership_persistence__EDGE_GATED |   2026 |        1 |                -0.1039 |                          0.0000 |                  0.0000 |                           0.0000 |
| U3_C_pairwise_residual_edge__EDGE_GATED         |   2021 |        3 |                -0.0596 |                          0.0000 |                  0.3333 |                           0.0000 |
| U3_C_pairwise_residual_edge__EDGE_GATED         |   2025 |        3 |                -0.0287 |                          0.0439 |                  0.3333 |                           0.3333 |
| U3_C_pairwise_residual_edge__EDGE_GATED         |   2026 |        1 |                -0.1039 |                          0.0000 |                  0.0000 |                           0.0000 |

## 7. Switch Quality

| candidate_id                                    | regime      |   rebalances |   mean_switch_in |   mean_switch_out |   mean_switch_in_minus_out |   median_switch_in_minus_out |   switch_in_winning_share |   mean_topk_minus_next |
|:------------------------------------------------|:------------|-------------:|-----------------:|------------------:|---------------------------:|-----------------------------:|--------------------------:|-----------------------:|
| BASELINE_S2_FIXED                               | strong_down |            3 |          -0.0504 |            0.0139 |                    -0.0644 |                      -0.0176 |                    0.3333 |                 0.0034 |
| BASELINE_S2_FIXED                               | mild_down   |            1 |          -0.0137 |           -0.0452 |                     0.0315 |                       0.0315 |                    1.0000 |                 0.0022 |
| BASELINE_S2_FIXED                               | neutral     |            5 |           0.0266 |            0.0089 |                     0.0177 |                      -0.0081 |                    0.4000 |                 0.0090 |
| BASELINE_S2_FIXED                               | mild_up     |            4 |           0.0482 |            0.0531 |                    -0.0049 |                      -0.0298 |                    0.2500 |                -0.0060 |
| BASELINE_S2_FIXED                               | strong_up   |            1 |           0.1127 |            0.1364 |                    -0.0237 |                      -0.0237 |                    0.0000 |                -0.0895 |
| U3_A_real_industry_leadership__EDGE_GATED       | strong_down |           14 |          -0.0271 |           -0.0622 |                     0.0351 |                       0.0340 |                    0.7857 |                 0.0374 |
| U3_A_real_industry_leadership__EDGE_GATED       | mild_down   |           13 |           0.0171 |           -0.0069 |                     0.0239 |                      -0.0009 |                    0.4615 |                 0.0001 |
| U3_A_real_industry_leadership__EDGE_GATED       | neutral     |           12 |          -0.0104 |            0.0012 |                    -0.0116 |                      -0.0163 |                    0.4167 |                -0.0141 |
| U3_A_real_industry_leadership__EDGE_GATED       | mild_up     |           11 |           0.0156 |            0.0177 |                    -0.0021 |                      -0.0246 |                    0.3636 |                -0.0043 |
| U3_A_real_industry_leadership__EDGE_GATED       | strong_up   |           12 |           0.0690 |            0.0455 |                     0.0235 |                       0.0114 |                    0.5833 |                -0.0157 |
| U3_B_buyable_leadership_persistence__EDGE_GATED | strong_down |           14 |          -0.0562 |           -0.0586 |                     0.0024 |                       0.0292 |                    0.6429 |                 0.0418 |
| U3_B_buyable_leadership_persistence__EDGE_GATED | mild_down   |           13 |           0.0072 |           -0.0202 |                     0.0273 |                       0.0268 |                    0.5385 |                 0.0140 |
| U3_B_buyable_leadership_persistence__EDGE_GATED | neutral     |           12 |          -0.0061 |            0.0108 |                    -0.0169 |                      -0.0459 |                    0.4167 |                -0.0149 |
| U3_B_buyable_leadership_persistence__EDGE_GATED | mild_up     |           10 |           0.0114 |            0.0144 |                    -0.0030 |                       0.0017 |                    0.6000 |                -0.0189 |
| U3_B_buyable_leadership_persistence__EDGE_GATED | strong_up   |           12 |           0.0707 |            0.0817 |                    -0.0110 |                       0.0112 |                    0.5000 |                -0.0337 |
| U3_C_pairwise_residual_edge__EDGE_GATED         | strong_down |           14 |          -0.0240 |           -0.0648 |                     0.0408 |                       0.0501 |                    0.6429 |                 0.0382 |
| U3_C_pairwise_residual_edge__EDGE_GATED         | mild_down   |           13 |          -0.0047 |           -0.0102 |                     0.0055 |                       0.0072 |                    0.5385 |                -0.0131 |
| U3_C_pairwise_residual_edge__EDGE_GATED         | neutral     |           12 |          -0.0073 |           -0.0014 |                    -0.0059 |                       0.0012 |                    0.5000 |                -0.0164 |
| U3_C_pairwise_residual_edge__EDGE_GATED         | mild_up     |           11 |           0.0312 |            0.0066 |                     0.0246 |                       0.0132 |                    0.6364 |                -0.0180 |
| U3_C_pairwise_residual_edge__EDGE_GATED         | strong_up   |           12 |           0.1256 |            0.0566 |                     0.0690 |                       0.0349 |                    0.6667 |                -0.0174 |

## 8. 行业暴露

| candidate_id                                    | trade_date          | industry   |   weight |
|:------------------------------------------------|:--------------------|:-----------|---------:|
| U3_C_pairwise_residual_edge__EDGE_GATED         | 2026-04-13 00:00:00 | 建筑装饰   |   0.1000 |
| U3_C_pairwise_residual_edge__EDGE_GATED         | 2026-04-13 00:00:00 | 交通运输   |   0.1000 |
| BASELINE_S2_FIXED                               | 2026-04-13 00:00:00 | 通信       |   0.1000 |
| BASELINE_S2_FIXED                               | 2026-04-13 00:00:00 | 石油石化   |   0.1000 |
| BASELINE_S2_FIXED                               | 2026-04-13 00:00:00 | 建筑装饰   |   0.1000 |
| U3_C_pairwise_residual_edge__EDGE_GATED         | 2026-04-13 00:00:00 | 银行       |   0.4000 |
| BASELINE_S2_FIXED                               | 2026-04-13 00:00:00 | 钢铁       |   0.0500 |
| U3_B_buyable_leadership_persistence__EDGE_GATED | 2026-04-13 00:00:00 | 通信       |   0.1000 |
| U3_B_buyable_leadership_persistence__EDGE_GATED | 2026-04-13 00:00:00 | 石油石化   |   0.1000 |
| U3_B_buyable_leadership_persistence__EDGE_GATED | 2026-04-13 00:00:00 | 建筑装饰   |   0.1000 |
| U3_B_buyable_leadership_persistence__EDGE_GATED | 2026-04-13 00:00:00 | 交通运输   |   0.1000 |
| U3_B_buyable_leadership_persistence__EDGE_GATED | 2026-04-13 00:00:00 | 银行       |   0.4000 |
| U3_C_pairwise_residual_edge__EDGE_GATED         | 2026-04-13 00:00:00 | 钢铁       |   0.0500 |
| U3_A_real_industry_leadership__EDGE_GATED       | 2026-04-13 00:00:00 | 通信       |   0.1000 |
| U3_A_real_industry_leadership__EDGE_GATED       | 2026-04-13 00:00:00 | 石油石化   |   0.1000 |
| U3_A_real_industry_leadership__EDGE_GATED       | 2026-04-13 00:00:00 | 建筑装饰   |   0.1000 |
| U3_A_real_industry_leadership__EDGE_GATED       | 2026-04-13 00:00:00 | 交通运输   |   0.1000 |
| U3_A_real_industry_leadership__EDGE_GATED       | 2026-04-13 00:00:00 | 银行       |   0.4000 |
| BASELINE_S2_FIXED                               | 2026-04-13 00:00:00 | 非银金融   |   0.0500 |
| U3_C_pairwise_residual_edge__EDGE_GATED         | 2026-04-13 00:00:00 | 非银金融   |   0.0500 |

## 9. 结论

- R2B v2 进入 gray zone 候选：U3_A_real_industry_leadership__EDGE_GATED；可补更细 slice，但仍非 production candidate。
- 无候选达到 daily proxy `>= +3%`，不补正式 full backtest。

## 10. 产出文件

- `data/results/r2b_edge_gated_replacement_v2_2026-04-28_leaderboard.csv`
- `data/results/r2b_edge_gated_replacement_v2_2026-04-28_replacement_diag_long.csv`
- `data/results/r2b_edge_gated_replacement_v2_2026-04-28_replacement_count_distribution.csv`
- `data/results/r2b_edge_gated_replacement_v2_2026-04-28_selected_pairs.csv`
- `data/results/r2b_edge_gated_replacement_v2_2026-04-28_overlap_long.csv`
- `data/results/r2b_edge_gated_replacement_v2_2026-04-28_industry_exposure_long.csv`
- `data/results/r2b_edge_gated_replacement_v2_2026-04-28_regime_long.csv`
- `data/results/r2b_edge_gated_replacement_v2_2026-04-28_breadth_long.csv`
- `data/results/r2b_edge_gated_replacement_v2_2026-04-28_year_long.csv`
- `data/results/r2b_edge_gated_replacement_v2_2026-04-28_switch_long.csv`
- `data/results/r2b_edge_gated_replacement_v2_2026-04-28_monthly_long.csv`
- `data/results/r2b_edge_gated_replacement_v2_2026-04-28_summary.json`
