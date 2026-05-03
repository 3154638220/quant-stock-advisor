# Monthly Selection Benchmark Suite

- 生成时间：`2026-05-03T16:22:09.582965+00:00`
- monthly_long：`data/results/monthly_selection_m8_concentration_regime_pitfix_2026_05_03_2026_05_03_monthly_long.csv`
- 模型：`M8_regime_aware_fixed_policy__indcap3`
- 候选池：`U1_liquid_tradable`
- Top-K：`20`
- 基准体系：主基准中证1000（若提供指数行情）、辅助中证2000、内部 alpha 基准 U1 候选池等权、宽基参照全A等权。

## Return Summary

| benchmark | role | primary | months | total_return | annualized_return | mean_monthly_return | median_monthly_return | monthly_positive_rate | max_drawdown | sharpe |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| model_top20_net | strategy | False | 39 | 113.18% | 26.23% | 2.13% | 2.22% | 66.67% | -16.14% | 1.23924 |
| model_top20_gross | strategy_gross | False | 39 | 120.87% | 27.61% | 2.22% | 2.31% | 66.67% | -15.28% | 1.2929 |
| u1_candidate_pool_ew | alpha_internal | False | 39 | -3.84% | -1.20% | 0.06% | -0.33% | 43.59% | -39.95% | 0.0368505 |
| all_a_market_ew | broad_equal_weight | False | 39 | 5.91% | 1.78% | 0.29% | 0.27% | 51.28% | -35.48% | 0.185682 |
| csi1000 | primary_index | True | 39 | -12.50% | -4.02% | -0.16% | -0.61% | 41.03% | -42.39% | -0.0904587 |
| csi2000 | secondary_index | False | 39 | -1.39% | -0.43% | 0.15% | -0.52% | 46.15% | -41.21% | 0.0830723 |

## Excess vs Benchmarks

| strategy | benchmark | months | excess_total_return | annualized_excess_arithmetic | mean_monthly_excess | median_monthly_excess | monthly_hit_rate | win_months | information_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| model_top20_net | u1_candidate_pool_ew | 39 | 117.02% | 27.42% | 2.07% | 1.68% | 66.67% | 26 | 1.91696 |
| model_top20_net | all_a_market_ew | 39 | 107.28% | 24.45% | 1.84% | 1.01% | 66.67% | 26 | 1.89604 |
| model_top20_net | csi1000 | 39 | 125.68% | 30.25% | 2.29% | 1.95% | 66.67% | 26 | 1.81907 |
| model_top20_net | csi2000 | 39 | 114.57% | 26.66% | 1.98% | 1.20% | 69.23% | 27 | 1.70873 |

## Cost Sensitivity (M10)

- 方法: `linear_scale (⚠️ 不应用于 promotion gate — 低估高成本影响)`
- 30bps gate: after_cost_excess_mean > 0 是 promotion 前提条件。
| label | cost_bps | after_cost_excess_mean | after_cost_excess_total | positive_months_ratio | breakeven |
| --- | --- | --- | --- | --- | --- |
| baseline_10bps | 10 | 1.8396% | 99.45% | 66.7% | True |
| stress_30bps | 30 | 1.6545% | 85.78% | 66.7% | True |
| stress_50bps | 50 | 1.4693% | 73.02% | 66.7% | True |

- 估算 Breakeven 成本: `75.0 bps`
- 解释: `after_cost_excess = topk_return - cost_drag - market_ew_return`



## Index Inputs

| benchmark | source | symbol_filter | status | covered_months | first_date | last_date |
| --- | --- | --- | --- | --- | --- | --- |
| csi1000 | data/cache/index_benchmarks.csv | 000852 | ok | 39 | 2021-01-04 | 2026-04-30 |
| csi2000 | data/cache/index_benchmarks.csv | 932000 | ok | 39 | 2021-01-04 | 2026-04-30 |

## 口径

- `model_top20_net` = `topk_return - cost_drag`，首月无上一期持仓时成本按 0 处理。
- `u1_candidate_pool_ew` = 同一 `U1_liquid_tradable` 候选池内股票月度收益等权平均，是内部 alpha 基准。
- `all_a_market_ew` = dataset 标签层全市场 open-to-open 等权基准，不是中证指数。
- 指数 CSV 若提供，按 `buy_trade_date` 开盘到 `sell_trade_date` 开盘计算 open-to-open 月收益，和模型执行时钟对齐。

## 本轮产物

- `data/results/monthly_selection_benchmark_suite_2026-05-04_2026-05-04_summary.csv`
- `data/results/monthly_selection_benchmark_suite_2026-05-04_2026-05-04_relative.csv`
- `data/results/monthly_selection_benchmark_suite_2026-05-04_2026-05-04_monthly_series.csv`
- `data/results/monthly_selection_benchmark_suite_2026-05-04_2026-05-04_cost_sensitivity.csv`
- `data/results/monthly_selection_benchmark_suite_2026-05-04_2026-05-04_capacity.csv`
- `data/results/monthly_selection_benchmark_suite_2026-05-04_2026-05-04_limit_up_stress.csv`
- `data/results/monthly_selection_benchmark_suite_2026-05-04_2026-05-04_manifest.json`
- `docs/reports/2026-05/monthly_selection_benchmark_suite_2026-05-04_2026-05-04.md`
