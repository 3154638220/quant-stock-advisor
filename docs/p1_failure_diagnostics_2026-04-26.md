# P1 G0/G1 Failure Diagnostics

- 生成时间：`2026-04-26T12:27:54.607740+00:00`
- 固定口径：`sort_by=xgboost` / `top_k=20` / `M` / `equal_weight` / `tplus1_open`
- 回测区间：`2021-01-01` ~ `2026-04-20`
- 结果类型：`signal_diagnostic`，仅解释 P1 失效机制，不 promotion

## 结论摘要

1. `G0/G1` 的关键落后不是单个特征组偶然失效，`2021` 两组都大幅跑输：`G0=-34.98%`，`G1=-43.15%`。
2. `G1` 在 `2022` 相对 `G0` 的月度超额合计变化为 `-26.51%`；这解释了 full backtest 中 weekly KDJ 增量没有兑现。
3. 换股诊断显示 `G1-only` 相对 `G0-only` 的前向收益按年并不稳定，当前更像改变排序边界，而不是稳定补足上涨参与。
4. 本轮不改变主计划：继续先解释标签/目标/市场状态，不扩 `weekly_kdj` interaction 网格。

## 年份失效

### G0

| year | is_key_year | months | strategy_compound_return | benchmark_compound_return | excess_sum | median_monthly_excess | benchmark_up_months | benchmark_up_positive_excess_share | worst_months | worst_months_excess_sum |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2021 | 1 | 12 | -0.0802499 | 0.307839 | -0.349795 | -0.0248123 | 8 | 0.125 | 2021-07,2021-04,2021-11 | -0.196038 |
| 2022 | 0 | 12 | -0.0703975 | -0.111008 | 0.0179565 | 0.0158944 | 6 | 0.166667 | 2022-05,2022-06,2022-07 | -0.178579 |
| 2023 | 0 | 12 | 0.143294 | 0.0717897 | 0.0642589 | 0.0120884 | 6 | 0.333333 | 2023-01,2023-11,2023-02 | -0.158453 |
| 2024 | 0 | 12 | 0.127105 | 0.033892 | 0.0556082 | -0.00565231 | 6 | 0.166667 | 2024-10,2024-09,2024-03 | -0.294368 |
| 2025 | 1 | 12 | 0.0506737 | 0.385677 | -0.284548 | -0.00987733 | 9 | 0.222222 | 2025-02,2025-08,2025-07 | -0.219172 |
| 2026 | 1 | 4 | -0.0410186 | 0.0520674 | -0.0964332 | -0.0173793 | 3 | 0.333333 | 2026-01,2026-04,2026-03 | -0.11031 |

### G1

| year | is_key_year | months | strategy_compound_return | benchmark_compound_return | excess_sum | median_monthly_excess | benchmark_up_months | benchmark_up_positive_excess_share | worst_months | worst_months_excess_sum |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2021 | 1 | 12 | -0.152712 | 0.307839 | -0.431546 | -0.0409157 | 8 | 0.125 | 2021-09,2021-08,2021-12 | -0.231607 |
| 2022 | 0 | 12 | -0.30362 | -0.111008 | -0.247097 | -0.027427 | 6 | 0.166667 | 2022-06,2022-07,2022-03 | -0.149097 |
| 2023 | 0 | 12 | 0.143534 | 0.0717897 | 0.0655115 | -0.00198849 | 6 | 0.333333 | 2023-01,2023-11,2023-02 | -0.109668 |
| 2024 | 0 | 12 | 0.164856 | 0.033892 | 0.0947267 | -0.00419224 | 6 | 0.166667 | 2024-10,2024-02,2024-11 | -0.244816 |
| 2025 | 1 | 12 | 0.0968576 | 0.385677 | -0.239419 | -0.0173148 | 9 | 0.111111 | 2025-08,2025-12,2025-07 | -0.164237 |
| 2026 | 1 | 4 | -0.0398502 | 0.0520674 | -0.0961828 | -0.0277818 | 3 | 0.333333 | 2026-01,2026-04,2026-02 | -0.132868 |

## 上涨月捕获率

### G0

| regime | months | benchmark_compound_return | strategy_compound_return | capture_ratio | median_benchmark_return | median_strategy_return | median_excess_return | positive_strategy_share | positive_excess_share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| benchmark_up | 38 | 5.17047 | 0.726778 | 0.140563 | 0.044658 | 0.0104661 | -0.038013 | 0.631579 | 0.210526 |
| benchmark_down | 26 | -0.695614 | -0.357119 | 0.513386 | -0.0244604 | -0.0213616 | 0.0203745 | 0.307692 | 0.730769 |

### G1

| regime | months | benchmark_compound_return | strategy_compound_return | capture_ratio | median_benchmark_return | median_strategy_return | median_excess_return | positive_strategy_share | positive_excess_share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| benchmark_up | 38 | 5.17047 | 0.834245 | 0.161348 | 0.044658 | 0.0138715 | -0.0338007 | 0.684211 | 0.184211 |
| benchmark_down | 26 | -0.695614 | -0.548736 | 0.788851 | -0.0244604 | -0.0289188 | 0.0032486 | 0.192308 | 0.538462 |

## G1 vs G0 月度退化

| month_end | g0_excess_return | g1_excess_return | g1_minus_g0_excess_return |
| --- | --- | --- | --- |
| 2022-04-30 | 0.106631 | 0.0148104 | -0.0918205 |
| 2024-12-31 | 0.089341 | 0.0186555 | -0.0706855 |
| 2022-03-31 | 0.0235508 | -0.0459413 | -0.0694921 |
| 2021-08-31 | -0.0275628 | -0.0767288 | -0.049166 |
| 2022-08-31 | 0.0246771 | -0.0214002 | -0.0460773 |
| 2022-09-30 | 0.0668804 | 0.0242772 | -0.0426032 |
| 2021-05-31 | -0.00773034 | -0.0476838 | -0.0399535 |
| 2022-01-31 | 0.0224515 | -0.0166436 | -0.0390951 |

## G1/G0 换股收益

| year | rebalances | mean_overlap_ratio | mean_g1_only_minus_g0_only_forward_return |
| --- | --- | --- | --- |
| 2021 | 12 | 0.282016 | 0.0171199 |
| 2022 | 12 | 0.316567 | -0.00753375 |
| 2023 | 12 | 0.465744 | 0.014537 |
| 2024 | 12 | 0.287454 | 0.0242675 |
| 2025 | 12 | 0.327577 | 0.0230551 |
| 2026 | 2 | 0.177489 | 0.0312179 |

## 持仓暴露 vs 基准

### G0

| feature | observations | mean_active_diff | median_active_diff | mean_active_zscore | median_active_zscore |
| --- | --- | --- | --- | --- | --- |
| amount_20d | 63 | 1.94687e+08 | 2.5367e+08 | 0.59462 | 0.596996 |
| log_market_cap | 63 | 1.38645 | 2.33808 | 0.423344 | 0.69376 |
| momentum_12_1 | 63 | 0.0208482 | 0.0270023 | 0.144272 | 0.0497503 |
| price_position | 63 | 0.0495484 | 0.0407362 | 0.248686 | 0.187214 |
| realized_vol | 63 | -0.13334 | -0.140677 | -0.660342 | -0.744444 |
| recent_return | 63 | -0.0126263 | -0.0122224 | -0.241657 | -0.260533 |
| turnover_roll_mean | 63 | 0.221121 | -0.139196 | 0.122245 | -0.100651 |
| vol_to_turnover | 63 | 2.02726 | 3.02092 | 0.602388 | 0.857855 |

### G1

| feature | observations | mean_active_diff | median_active_diff | mean_active_zscore | median_active_zscore |
| --- | --- | --- | --- | --- | --- |
| amount_20d | 63 | 1.30377e+08 | 1.68917e+08 | 0.43414 | 0.360916 |
| log_market_cap | 63 | 0.376468 | 0.602902 | 0.132171 | 0.177217 |
| momentum_12_1 | 63 | 0.0293514 | 0.0318744 | 0.124583 | 0.0883734 |
| price_position | 63 | 0.0447334 | 0.0237242 | 0.214359 | 0.113108 |
| realized_vol | 63 | -0.0658981 | -0.0706265 | -0.334682 | -0.349475 |
| recent_return | 63 | -0.0106775 | -0.00894491 | -0.206334 | -0.204148 |
| turnover_roll_mean | 63 | 0.358113 | 0.0410891 | 0.210064 | 0.0264312 |
| vol_to_turnover | 63 | 0.905483 | 1.13901 | 0.291604 | 0.380873 |

## Top-K 与 21-40 桶

### G0

| bucket | rebalances | mean_forward_return | median_forward_return | mean_positive_share |
| --- | --- | --- | --- | --- |
| 01_20 | 62 | 0.0297332 | 0.0120415 | 0.566817 |
| 21_40 | 62 | 0.0202436 | 0.0121961 | 0.557189 |
| 41_60 | 62 | 0.0178447 | -0.00397043 | 0.491683 |
| 61_100 | 62 | 0.0174264 | 0.00229202 | 0.528686 |
| 101_plus | 62 | 0.00246435 | -0.0126776 | 0.454297 |

### G1

| bucket | rebalances | mean_forward_return | median_forward_return | mean_positive_share |
| --- | --- | --- | --- | --- |
| 01_20 | 62 | 0.0380586 | 0.0174241 | 0.580964 |
| 21_40 | 62 | 0.0137211 | 0.00291876 | 0.543481 |
| 41_60 | 62 | 0.00835344 | -0.00364817 | 0.486235 |
| 61_100 | 62 | 0.0144684 | 0.000614693 | 0.522775 |
| 101_plus | 62 | 0.00246663 | -0.0116193 | 0.455295 |

## 本轮产物

- `data/results/p1_failure_diagnostics_2026-04-26_summary.json`
- `data/results/p1_failure_diagnostics_2026-04-26_monthly.csv`
- `data/results/p1_failure_diagnostics_2026-04-26_yearly.csv`
- `data/results/p1_failure_diagnostics_2026-04-26_capture.csv`
- `data/results/p1_failure_diagnostics_2026-04-26_exposure_summary.csv`
- `data/results/p1_failure_diagnostics_2026-04-26_rank_bucket_summary.csv`
- `data/results/p1_failure_diagnostics_2026-04-26_selection_overlap.csv`
