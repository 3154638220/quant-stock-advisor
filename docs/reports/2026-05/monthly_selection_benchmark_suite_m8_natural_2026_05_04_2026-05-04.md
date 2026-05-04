# Monthly Selection Benchmark Suite

- 生成时间：`2026-05-04T06:01:19.964424+00:00`
- monthly_long：`data/results/monthly_selection_m8_natural_industry_constraints_30bps_2026_05_04_monthly_long_augmented.csv`
- 模型：`M5_plus_industry_breadth_plus_fund_flow_plus_fundamental_elasticnet_excess__label_industry_neutral_excess`
- 候选池：`U1_liquid_tradable`
- Top-K：`20`
- 基准体系：主基准中证1000（若提供指数行情）、辅助中证2000、内部 alpha 基准 U1 候选池等权、宽基参照全A等权。

## Return Summary

| benchmark | role | primary | months | total_return | annualized_return | mean_monthly_return | median_monthly_return | monthly_positive_rate | max_drawdown | sharpe |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| model_top20_net | strategy | False | 38 | 51.32% | 13.97% | 1.29% | 0.94% | 57.89% | -25.72% | 0.712145 |
| model_top20_gross | strategy_gross | False | 38 | 67.48% | 17.69% | 1.56% | 1.24% | 60.53% | -23.46% | 0.860627 |
| u1_candidate_pool_ew | alpha_internal | False | 38 | -8.10% | -2.63% | -0.06% | -0.48% | 42.11% | -39.95% | -0.0353497 |
| all_a_market_ew | broad_equal_weight | False | 38 | 1.51% | 0.48% | 0.18% | 0.07% | 50.00% | -35.48% | 0.11712 |
| csi1000 | primary_index | True | 38 | 13.98% | 4.22% | 0.67% | -0.67% | 47.37% | -33.47% | 0.270269 |
| csi2000 | secondary_index | False | 38 | 34.04% | 9.69% | 1.10% | 0.03% | 50.00% | -30.43% | 0.448387 |

## Excess vs Benchmarks

| strategy | benchmark | months | excess_total_return | annualized_excess_arithmetic | mean_monthly_excess | median_monthly_excess | monthly_hit_rate | win_months | information_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| model_top20_net | u1_candidate_pool_ew | 38 | 59.42% | 16.61% | 1.35% | 1.23% | 71.05% | 27 | 1.18203 |
| model_top20_net | all_a_market_ew | 38 | 49.80% | 13.50% | 1.10% | 0.62% | 57.89% | 22 | 1.03143 |
| model_top20_net | csi1000 | 38 | 37.34% | 9.76% | 0.62% | 0.87% | 68.42% | 26 | 0.296954 |
| model_top20_net | csi2000 | 38 | 17.28% | 4.28% | 0.19% | 0.61% | 65.79% | 25 | 0.100844 |

## Cost Sensitivity (M10)

- 方法: `real_backtest (✅ 真实回测 cost_drag)`
- 30bps gate: after_cost_excess_mean > 0 是 promotion 前提条件。
| label | cost_bps | after_cost_excess_mean | after_cost_excess_total | positive_months_ratio | breakeven |
| --- | --- | --- | --- | --- | --- |
| real_30bps | 30 | 1.1031% | 48.00% | 57.9% | True |
| real_50bps | 50 | 0.9228% | 38.29% | 55.3% | True |

- 估算 Breakeven 成本: `75.0 bps`
- 解释: `after_cost_excess = topk_return - cost_drag - market_ew_return`



## Index Inputs

| benchmark | source | symbol_filter | status | covered_months | first_date | last_date |
| --- | --- | --- | --- | --- | --- | --- |
| csi1000 | data/cache/index_benchmarks.csv | 000852 | ok | 38 | 2021-01-04 | 2026-04-30 |
| csi2000 | data/cache/index_benchmarks.csv | 932000 | ok | 38 | 2021-01-04 | 2026-04-30 |

## 口径

- `model_top20_net` = `topk_return - cost_drag`，首月无上一期持仓时成本按 0 处理。
- `u1_candidate_pool_ew` = 同一 `U1_liquid_tradable` 候选池内股票月度收益等权平均，是内部 alpha 基准。
- `all_a_market_ew` = dataset 标签层全市场 open-to-open 等权基准，不是中证指数。
- 指数 CSV 若提供，按 `buy_trade_date` 开盘到 `sell_trade_date` 开盘计算 open-to-open 月收益，和模型执行时钟对齐。

## 本轮产物

- `data/results/monthly_selection_benchmark_suite_m8_natural_2026_05_04_2026-05-04_summary.csv`
- `data/results/monthly_selection_benchmark_suite_m8_natural_2026_05_04_2026-05-04_relative.csv`
- `data/results/monthly_selection_benchmark_suite_m8_natural_2026_05_04_2026-05-04_monthly_series.csv`
- `data/results/monthly_selection_benchmark_suite_m8_natural_2026_05_04_2026-05-04_cost_sensitivity.csv`
- `data/results/monthly_selection_benchmark_suite_m8_natural_2026_05_04_2026-05-04_capacity.csv`
- `data/results/monthly_selection_benchmark_suite_m8_natural_2026_05_04_2026-05-04_limit_up_stress.csv`
- `data/results/monthly_selection_benchmark_suite_m8_natural_2026_05_04_2026-05-04_manifest.json`
- `docs/reports/2026-05/monthly_selection_benchmark_suite_m8_natural_2026_05_04_2026-05-04.md`
