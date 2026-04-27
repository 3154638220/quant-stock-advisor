# P1 Market-Relative G1 Gap Diagnostics

- 生成时间：`2026-04-27T04:35:31.611476+00:00`
- 固定口径：`sort_by=xgboost` / `top_k=20` / `M` / `equal_weight` / `tplus1_open`
- 回测区间：`2021-01-01` ~ `2026-04-26`
- 结果类型：`signal_diagnostic`，仅解释 P1 失效机制，不 promotion
- 诊断对象：`rank + market_relative + G1`，对照 `rank + market_relative + G0`

## 结论摘要

1. 本轮明确是**标签口径失效 + light proxy 断层共同作用**，执行成本只是次要放大项。`G1` light proxy 年化超额为 `+17.98%`，但正式 full backtest 对 `market_ew` 年化超额为 `-33.97%`。
2. `market_relative` 标签把 `weekly_kdj_*` 推向了更小市值、低价格位置、更高波动/换手的组合边界：`G1` 相对基准的 `log_market_cap` 均值 z-score 为 `-0.28`，`realized_vol` 为 `+0.31`，`price_position` 为 `-0.18`；`G0` 则仍偏大市值、低波动。
3. Top-20 排名没有形成稳定优势：`G1` 的 Top-20 平均前向收益 `2.76%`，但中位数为 `-0.20%`；`41-60` 桶中位数反而为 `+0.56%`，说明 light proxy 的头部选择和正式月频 Top-20 边界存在错位。
4. 年份退化集中在 `2022/2024/2025`：`G1-G0` 月度超额合计分别为 `-40.41% / -62.85% / -5.89%`。正式年度超额中，`G1` 在 `2022/2024/2025` 分别为 `-34.57% / -46.15% / -42.46%`。
5. 执行不是主因：`G1` turnover mean `34.68%`，成本拖累年化约 `-0.62%`，远小于 `-33.97%` 的年化超额缺口。主要断层来自训练标签与 light proxy 没有刻画正式月频、Top-20、全样本市场状态下的排序风险。
6. 本轮不改变默认研究基线，不 promotion；暂停 `market_relative + weekly_kdj` 方向的扩网格，只保留为失败样本和后续标签/proxy 设计的反例。

## 年份失效

### G0

| year | is_key_year | months | strategy_compound_return | benchmark_compound_return | excess_sum | median_monthly_excess | benchmark_up_months | benchmark_up_positive_excess_share | worst_months | worst_months_excess_sum |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2021 | 1 | 12 | -0.0648014 | 0.307839 | -0.336688 | -0.0331715 | 8 | 0.125 | 2021-11,2021-07,2021-09 | -0.191231 |
| 2022 | 0 | 12 | -0.140532 | -0.111008 | -0.0597195 | 0.00804607 | 6 | 0.166667 | 2022-05,2022-06,2022-10 | -0.238264 |
| 2023 | 0 | 12 | 0.144211 | 0.0717897 | 0.0648507 | 0.00934226 | 6 | 0.333333 | 2023-01,2023-11,2023-02 | -0.12643 |
| 2024 | 0 | 12 | 0.096056 | 0.033892 | 0.0294731 | 0.00469831 | 6 | 0.333333 | 2024-10,2024-09,2024-11 | -0.268537 |
| 2025 | 1 | 12 | 0.0193927 | 0.385677 | -0.303776 | -0.0196302 | 9 | 0.333333 | 2025-12,2025-08,2025-02 | -0.259826 |
| 2026 | 1 | 4 | -0.114835 | 0.0520674 | -0.175069 | -0.0290656 | 3 | 0.333333 | 2026-01,2026-04,2026-02 | -0.197347 |

### G1

| year | is_key_year | months | strategy_compound_return | benchmark_compound_return | excess_sum | median_monthly_excess | benchmark_up_months | benchmark_up_positive_excess_share | worst_months | worst_months_excess_sum |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2021 | 1 | 12 | -0.114144 | 0.307839 | -0.389707 | -0.0414708 | 8 | 0 | 2021-08,2021-09,2021-05 | -0.179363 |
| 2022 | 0 | 12 | -0.447515 | -0.111008 | -0.463781 | -0.0398123 | 6 | 0 | 2022-08,2022-02,2022-11 | -0.182345 |
| 2023 | 0 | 12 | -0.069551 | 0.0717897 | -0.131641 | -0.0236045 | 6 | 0.5 | 2023-07,2023-04,2023-09 | -0.141115 |
| 2024 | 0 | 12 | -0.427636 | 0.033892 | -0.599015 | -0.0444793 | 6 | 0.166667 | 2024-10,2024-09,2024-02 | -0.409607 |
| 2025 | 1 | 12 | -0.0389279 | 0.385677 | -0.362661 | -0.0278398 | 9 | 0 | 2025-04,2025-12,2025-07 | -0.229408 |
| 2026 | 1 | 4 | -0.00190518 | 0.0520674 | -0.0562805 | -0.0114595 | 3 | 0.333333 | 2026-04,2026-01,2026-02 | -0.0921412 |

## 上涨月捕获率

### G0

| regime | months | benchmark_compound_return | strategy_compound_return | capture_ratio | median_benchmark_return | median_strategy_return | median_excess_return | positive_strategy_share | positive_excess_share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| benchmark_up | 38 | 5.17047 | 0.580965 | 0.112362 | 0.044658 | 0.01508 | -0.0331715 | 0.684211 | 0.263158 |
| benchmark_down | 26 | -0.695614 | -0.424672 | 0.610499 | -0.0244604 | -0.0299707 | 0.0140096 | 0.230769 | 0.730769 |

### G1

| regime | months | benchmark_compound_return | strategy_compound_return | capture_ratio | median_benchmark_return | median_strategy_return | median_excess_return | positive_strategy_share | positive_excess_share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| benchmark_up | 38 | 5.17047 | 0.538692 | 0.104186 | 0.044658 | 0.00532392 | -0.0393659 | 0.578947 | 0.131579 |
| benchmark_down | 26 | -0.695614 | -0.837511 | 1.20399 | -0.0244604 | -0.0672503 | -0.0252926 | 0.0769231 | 0.269231 |

## G1 vs G0 月度退化

| month_end | g0_excess_return | g1_excess_return | g1_minus_g0_excess_return |
| --- | --- | --- | --- |
| 2024-01-31 | 0.218436 | 0.0355215 | -0.182915 |
| 2022-04-30 | 0.112228 | -0.0422241 | -0.154452 |
| 2023-04-30 | 0.0864687 | -0.0449599 | -0.131429 |
| 2022-11-30 | 0.0343201 | -0.0585159 | -0.092836 |
| 2025-04-30 | -0.0202811 | -0.109733 | -0.0894519 |
| 2023-07-31 | 0.0180155 | -0.0645528 | -0.0825683 |
| 2024-12-31 | 0.0280746 | -0.0528535 | -0.0809281 |
| 2022-09-30 | 0.0514629 | -0.0257145 | -0.0771774 |

## G1/G0 换股收益

| year | rebalances | mean_overlap_ratio | mean_g1_only_minus_g0_only_forward_return |
| --- | --- | --- | --- |
| 2021 | 12 | 0.310556 | 0.00734829 |
| 2022 | 12 | 0.103652 | 0.00266889 |
| 2023 | 12 | 0.0488636 | -0.0252016 |
| 2024 | 12 | 0.0741615 | 0.0103207 |
| 2025 | 12 | 0.126804 | 0.0156492 |
| 2026 | 2 | 0.177489 | 0.0266781 |

## 持仓暴露 vs 基准

### G0

| feature | observations | mean_active_diff | median_active_diff | mean_active_zscore | median_active_zscore |
| --- | --- | --- | --- | --- | --- |
| amount_20d | 63 | 2.22198e+08 | 2.89762e+08 | 0.668871 | 0.681937 |
| log_market_cap | 63 | 2.05206 | 2.29276 | 0.617225 | 0.656901 |
| momentum_12_1 | 63 | 0.0336943 | 0.0527322 | 0.153974 | 0.168279 |
| price_position | 63 | 0.0623493 | 0.0463071 | 0.312763 | 0.190625 |
| realized_vol | 63 | -0.0999723 | -0.112049 | -0.499794 | -0.574643 |
| recent_return | 63 | -0.0105413 | -0.00885975 | -0.199447 | -0.180379 |
| turnover_roll_mean | 63 | -0.219161 | -0.307399 | -0.116389 | -0.180177 |
| vol_to_turnover | 63 | 2.77253 | 3.11194 | 0.819775 | 0.885634 |

### G1

| feature | observations | mean_active_diff | median_active_diff | mean_active_zscore | median_active_zscore |
| --- | --- | --- | --- | --- | --- |
| amount_20d | 63 | -8.6369e+07 | -7.13817e+07 | -0.121268 | -0.174383 |
| log_market_cap | 63 | -0.916204 | -0.900728 | -0.281016 | -0.275438 |
| momentum_12_1 | 63 | 0.0508026 | 0.0475106 | 0.11549 | 0.0788482 |
| price_position | 63 | -0.0400896 | -0.03841 | -0.181957 | -0.160101 |
| realized_vol | 63 | 0.0629193 | 0.0615116 | 0.313878 | 0.333909 |
| recent_return | 63 | -0.00981935 | -0.00925311 | -0.19874 | -0.20753 |
| turnover_roll_mean | 63 | 0.217511 | 0.207277 | 0.160408 | 0.121783 |
| vol_to_turnover | 63 | -0.592851 | -0.547284 | -0.171567 | -0.16068 |

## Top-K 与 21-40 桶

### G0

| bucket | rebalances | mean_forward_return | median_forward_return | mean_positive_share |
| --- | --- | --- | --- | --- |
| 01_20 | 62 | 0.0266626 | 0.00926847 | 0.5627 |
| 21_40 | 62 | 0.0220253 | 0.0068298 | 0.515762 |
| 41_60 | 62 | 0.0178857 | 5.55112e-17 | 0.528187 |
| 61_100 | 62 | 0.00989883 | -0.00183876 | 0.514105 |
| 101_plus | 62 | 0.00238233 | -0.0107061 | 0.453999 |

### G1

| bucket | rebalances | mean_forward_return | median_forward_return | mean_positive_share |
| --- | --- | --- | --- | --- |
| 01_20 | 62 | 0.0275899 | -0.00199346 | 0.510489 |
| 21_40 | 62 | 0.0177359 | -0.000766358 | 0.539195 |
| 41_60 | 62 | 0.019879 | 0.00562389 | 0.530622 |
| 61_100 | 62 | 0.00800948 | 0.00426162 | 0.520255 |
| 101_plus | 62 | 0.00412899 | -0.00895186 | 0.463438 |

## 本轮产物

- `data/results/p1_marketrel_g1_gap_diagnostics_2026-04-26_summary.json`
- `data/results/p1_marketrel_g1_gap_diagnostics_2026-04-26_monthly.csv`
- `data/results/p1_marketrel_g1_gap_diagnostics_2026-04-26_yearly.csv`
- `data/results/p1_marketrel_g1_gap_diagnostics_2026-04-26_capture.csv`
- `data/results/p1_marketrel_g1_gap_diagnostics_2026-04-26_exposure_summary.csv`
- `data/results/p1_marketrel_g1_gap_diagnostics_2026-04-26_rank_bucket_summary.csv`
- `data/results/p1_marketrel_g1_gap_diagnostics_2026-04-26_selection_overlap.csv`
- `data/results/p1_marketrel_g1_gap_diagnostics_2026-04-26_monthly_delta.csv`
