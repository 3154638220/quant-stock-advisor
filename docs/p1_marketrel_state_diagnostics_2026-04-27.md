# P1 Market-Relative State Diagnostics

- 生成时间：`2026-04-27T06:25:07.625556+00:00`
- 结果类型：`signal_diagnostic`
- 诊断对象：`rank + market_relative + G0/G1`
- 关键年份：`2022,2024,2025`

## 结论摘要

1. `market_relative + G1` 在关键年份合计超额为 `-142.55%`，弱于 `G0` 的 `-33.40%`；问题不是单一年份噪声。
2. 按市场月收益分层，`G1` 的强上涨月中位超额为 `-5.28%`，强下跌月中位超额为 `-1.39%`。这说明 `market_relative` 标签没有修复上涨参与，反而把下跌月防守也削弱。
3. 本轮仅归档市场状态诊断，不改变默认研究基线，不 promotion。后续候选必须先通过 `daily_bt_like_proxy_annualized_excess_vs_market` 准入，再看这些市场状态分层是否改善。

## 市场状态分层

| group | state_axis | state | months | benchmark_compound_return | strategy_compound_return | excess_sum | median_excess_return | positive_excess_share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| G0 | return_state | mild_down | 18 | -0.309605 | -0.235738 | 0.109913 | 0.00593929 | 0.611111 |
| G0 | return_state | mild_up | 23 | 0.799014 | 0.165173 | -0.423109 | -0.0191169 | 0.391304 |
| G0 | return_state | strong_down | 8 | -0.559114 | -0.247211 | 0.492286 | 0.0315626 | 1 |
| G0 | return_state | strong_up | 15 | 2.42992 | 0.35685 | -0.960019 | -0.0623513 | 0.0666667 |
| G1 | return_state | mild_down | 18 | -0.309605 | -0.609548 | -0.537311 | -0.0306236 | 0.222222 |
| G1 | return_state | mild_up | 23 | 0.799014 | 0.0197062 | -0.560748 | -0.0315886 | 0.173913 |
| G1 | return_state | strong_down | 8 | -0.559114 | -0.583843 | -0.0497395 | -0.013938 | 0.375 |
| G1 | return_state | strong_up | 15 | 2.42992 | 0.508956 | -0.855287 | -0.0528294 | 0.0666667 |
| G0 | vol_state | high_vol | 22 | -0.220586 | -0.243816 | -0.0877417 | -0.00119129 | 0.5 |
| G0 | vol_state | low_vol | 22 | 1.00052 | 0.161874 | -0.548372 | -0.0224922 | 0.363636 |
| G0 | vol_state | mid_vol | 20 | 0.204567 | 0.0352643 | -0.144815 | -0.00246392 | 0.5 |
| G1 | vol_state | high_vol | 22 | -0.220586 | -0.69098 | -0.915935 | -0.0395158 | 0.181818 |
| G1 | vol_state | low_vol | 22 | 1.00052 | 0.080258 | -0.618169 | -0.0341904 | 0.181818 |
| G1 | vol_state | mid_vol | 20 | 0.204567 | -0.251035 | -0.468982 | -0.0182868 | 0.2 |
| G0 | breadth_state | broad_breadth | 22 | 2.19374 | 0.379926 | -0.858121 | -0.0231276 | 0.227273 |
| G0 | breadth_state | mid_breadth | 20 | 0.171045 | -0.0560714 | -0.217398 | 0.00687663 | 0.55 |
| G0 | breadth_state | narrow_breadth | 22 | -0.497809 | -0.301698 | 0.294591 | 0.00960561 | 0.590909 |
| G1 | breadth_state | broad_breadth | 22 | 2.19374 | 0.328403 | -0.892507 | -0.0377579 | 0.181818 |
| G1 | breadth_state | mid_breadth | 20 | 0.171045 | -0.255582 | -0.445591 | -0.0306169 | 0.25 |
| G1 | breadth_state | narrow_breadth | 22 | -0.497809 | -0.747169 | -0.664987 | -0.0380948 | 0.136364 |

## G1 相对 G0 的状态分层

| state_axis | state | months | g1_minus_g0_excess_sum | median_g1_minus_g0_excess | g1_beats_g0_share |
| --- | --- | --- | --- | --- | --- |
| return_state | mild_down | 18 | -0.647223 | -0.0415413 | 0.111111 |
| return_state | mild_up | 23 | -0.13764 | -0.00735503 | 0.434783 |
| return_state | strong_down | 8 | -0.542026 | -0.0486057 | 0.125 |
| return_state | strong_up | 15 | 0.104732 | 0.00188305 | 0.6 |
| vol_state | high_vol | 22 | -0.828193 | -0.0428882 | 0.318182 |
| vol_state | low_vol | 22 | -0.0697965 | -0.00589911 | 0.454545 |
| vol_state | mid_vol | 20 | -0.324167 | -0.0120342 | 0.25 |
| breadth_state | broad_breadth | 22 | -0.0343859 | 0.00396477 | 0.590909 |
| breadth_state | mid_breadth | 20 | -0.228193 | -0.0137024 | 0.35 |
| breadth_state | narrow_breadth | 22 | -0.959578 | -0.0282567 | 0.0909091 |

## 关键年份分层

| group | year | return_state | months | benchmark_compound_return | strategy_compound_return | excess_sum | median_excess_return |
| --- | --- | --- | --- | --- | --- | --- | --- |
| G0 | 2022 | mild_down | 2 | -0.0685779 | -0.0530207 | 0.0160921 | 0.00804607 |
| G0 | 2022 | mild_up | 3 | 0.11801 | -0.0489761 | -0.162852 | -0.0606489 |
| G0 | 2022 | strong_down | 4 | -0.334845 | -0.152025 | 0.223553 | 0.046155 |
| G0 | 2022 | strong_up | 3 | 0.28346 | 0.125421 | -0.136512 | -0.0765527 |
| G0 | 2024 | mild_down | 4 | -0.0985076 | -0.0324089 | 0.070417 | 0.016115 |
| G0 | 2024 | mild_up | 3 | 0.105849 | 0.100825 | -0.00492816 | 0.00524121 |
| G0 | 2024 | strong_down | 2 | -0.258762 | -0.0447056 | 0.232352 | 0.116176 |
| G0 | 2024 | strong_up | 3 | 0.399136 | 0.0771729 | -0.268367 | -0.0653177 |
| G0 | 2025 | mild_down | 3 | -0.0345643 | -0.0470434 | -0.0117559 | -0.0158244 |
| G0 | 2025 | mild_up | 5 | 0.102306 | -0.0046038 | -0.0917766 | 0.00959786 |
| G0 | 2025 | strong_up | 4 | 0.302077 | 0.0746633 | -0.200244 | -0.0544682 |
| G1 | 2022 | mild_down | 2 | -0.0685779 | -0.162245 | -0.0995358 | -0.0497679 |
| G1 | 2022 | mild_up | 3 | 0.11801 | -0.0303581 | -0.144418 | -0.045946 |
| G1 | 2022 | strong_down | 4 | -0.334845 | -0.403854 | -0.0958147 | -0.0252926 |
| G1 | 2022 | strong_up | 3 | 0.28346 | 0.140879 | -0.124012 | -0.0542404 |
| G1 | 2024 | mild_down | 4 | -0.0985076 | -0.238939 | -0.159933 | -0.0512007 |
| G1 | 2024 | mild_up | 3 | 0.105849 | 0.0772581 | -0.0255866 | -0.0175811 |
| G1 | 2024 | strong_down | 2 | -0.258762 | -0.259629 | -0.00388929 | -0.00194465 |
| G1 | 2024 | strong_up | 3 | 0.399136 | -0.0570598 | -0.409607 | -0.134555 |
| G1 | 2025 | mild_down | 3 | -0.0345643 | -0.113988 | -0.076885 | 0.00880975 |
| G1 | 2025 | mild_up | 5 | 0.102306 | -0.0354853 | -0.132349 | -0.0135941 |
| G1 | 2025 | strong_up | 4 | 0.302077 | 0.124625 | -0.153427 | -0.0393659 |

## G1 相对 G0 最差月份

| month_end | year | month | g0_strategy_return | benchmark_return | g0_excess_return | g1_strategy_return | g1_excess_return | g1_minus_g0_strategy_return | g1_minus_g0_excess_return | group | return_state | vol_state | breadth_state | benchmark_daily_vol | breadth_positive_share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2024-01-31 | 2024 | 1 | 0.0290695 | -0.189367 | 0.218436 | -0.153845 | 0.0355215 | -0.182915 | -0.182915 | G1_minus_G0 | strong_down | high_vol | narrow_breadth | 0.0209085 | 0.334931 |
| 2022-04-30 | 2022 | 4 | -0.0377718 | -0.15 | 0.112228 | -0.192224 | -0.0422241 | -0.154452 | -0.154452 | G1_minus_G0 | strong_down | high_vol | narrow_breadth | 0.0270077 | 0.34764 |
| 2023-04-30 | 2023 | 4 | 0.058913 | -0.0275557 | 0.0864687 | -0.0725156 | -0.0449599 | -0.131429 | -0.131429 | G1_minus_G0 | mild_down | mid_vol | narrow_breadth | 0.0111802 | 0.406783 |
| 2022-11-30 | 2022 | 11 | 0.11115 | 0.0768298 | 0.0343201 | 0.0183139 | -0.0585159 | -0.092836 | -0.092836 | G1_minus_G0 | strong_up | low_vol | broad_breadth | 0.0100727 | 0.480977 |
| 2025-04-30 | 2025 | 4 | -0.0400315 | -0.0197504 | -0.0202811 | -0.129483 | -0.109733 | -0.0894519 | -0.0894519 | G1_minus_G0 | mild_down | high_vol | broad_breadth | 0.0290477 | 0.480909 |
| 2023-07-31 | 2023 | 7 | 0.0252207 | 0.00720522 | 0.0180155 | -0.0573476 | -0.0645528 | -0.0825683 | -0.0825683 | G1_minus_G0 | mild_up | low_vol | mid_breadth | 0.00718283 | 0.44851 |
| 2024-12-31 | 2024 | 12 | -0.0134792 | -0.0415538 | 0.0280746 | -0.0944073 | -0.0528535 | -0.0809281 | -0.0809281 | G1_minus_G0 | mild_down | high_vol | mid_breadth | 0.0168073 | 0.446289 |
| 2022-09-30 | 2022 | 9 | -0.0363286 | -0.0877915 | 0.0514629 | -0.113506 | -0.0257145 | -0.0771774 | -0.0771774 | G1_minus_G0 | strong_down | high_vol | narrow_breadth | 0.0140871 | 0.364844 |
| 2023-03-31 | 2023 | 3 | 0.0448318 | -0.0116135 | 0.0564453 | -0.0305653 | -0.0189518 | -0.0753971 | -0.0753971 | G1_minus_G0 | mild_down | low_vol | narrow_breadth | 0.00894382 | 0.403469 |
| 2022-08-31 | 2022 | 8 | -0.0198071 | -0.0315195 | 0.0117124 | -0.0942766 | -0.0627571 | -0.0744695 | -0.0744695 | G1_minus_G0 | mild_down | high_vol | narrow_breadth | 0.0141111 | 0.408605 |
| 2024-09-30 | 2024 | 9 | 0.165526 | 0.230844 | -0.0653177 | 0.0962893 | -0.134555 | -0.069237 | -0.069237 | G1_minus_G0 | strong_up | high_vol | broad_breadth | 0.0301123 | 0.543127 |
| 2024-02-29 | 2024 | 2 | 0.0444189 | 0.0633659 | -0.0189469 | -0.0236426 | -0.0870084 | -0.0680615 | -0.0680615 | G1_minus_G0 | strong_up | high_vol | broad_breadth | 0.0406989 | 0.584222 |

_仅展示前 12 行，共 20 行。_

## 产物

- `data/results/p1_marketrel_state_diagnostics_2026-04-27_monthly_state.csv`
- `data/results/p1_marketrel_state_diagnostics_2026-04-27_state_summary.csv`
- `data/results/p1_marketrel_state_diagnostics_2026-04-27_delta_state_summary.csv`
- `data/results/p1_marketrel_state_diagnostics_2026-04-27_key_year_state.csv`
- `data/results/p1_marketrel_state_diagnostics_2026-04-27_worst_months.csv`
- `data/results/p1_marketrel_state_diagnostics_2026-04-27.json`
- `docs/p1_marketrel_state_diagnostics_2026-04-27.md`
