# Monthly Selection Benchmark Suite

- 生成时间：`2026-05-01T07:06:17.088733+00:00`
- monthly_long：`data/results/monthly_selection_m8_concentration_regime_2026-05-01_monthly_long.csv`
- 模型：`M8_regime_aware_fixed_policy__indcap3`
- 候选池：`U1_liquid_tradable`
- Top-K：`20`
- 基准体系：主基准中证1000（若提供指数行情）、辅助中证2000、内部 alpha 基准 U1 候选池等权、宽基参照全A等权。

## Return Summary

| benchmark | role | primary | months | total_return | annualized_return | mean_monthly_return | median_monthly_return | monthly_positive_rate | max_drawdown | sharpe |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| model_top20_net | strategy | False | 39 | 123.50% | 28.08% | 2.23% | 1.64% | 64.10% | -18.15% | 1.41477 |
| model_top20_gross | strategy_gross | False | 39 | 131.56% | 29.48% | 2.32% | 1.74% | 64.10% | -17.28% | 1.47356 |
| u1_candidate_pool_ew | alpha_internal | False | 39 | -3.84% | -1.20% | 0.06% | -0.33% | 43.59% | -39.95% | 0.0368505 |
| all_a_market_ew | broad_equal_weight | False | 39 | 5.91% | 1.78% | 0.29% | 0.27% | 51.28% | -35.48% | 0.185682 |

## Excess vs Benchmarks

| strategy | benchmark | months | excess_total_return | annualized_excess_arithmetic | mean_monthly_excess | median_monthly_excess | monthly_hit_rate | win_months | information_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| model_top20_net | u1_candidate_pool_ew | 39 | 127.34% | 29.27% | 2.16% | 1.76% | 74.36% | 29 | 2.02471 |
| model_top20_net | all_a_market_ew | 39 | 117.59% | 26.30% | 1.93% | 1.41% | 76.92% | 30 | 2.06874 |

## Index Inputs

| benchmark | status | note |
| --- | --- | --- |
| csi1000/csi2000 | not_supplied | 提供 --index-csv 后自动纳入主基准。 |

## 口径

- `model_top20_net` = `topk_return - cost_drag`，首月无上一期持仓时成本按 0 处理。
- `u1_candidate_pool_ew` = 同一 `U1_liquid_tradable` 候选池内股票月度收益等权平均，是内部 alpha 基准。
- `all_a_market_ew` = dataset 标签层全市场 open-to-open 等权基准，不是中证指数。
- 指数 CSV 若提供，按 `buy_trade_date` 开盘到 `sell_trade_date` 开盘计算 open-to-open 月收益，和模型执行时钟对齐。

## 本轮产物

- `data/results/monthly_selection_benchmark_suite_2026-05-01_summary.csv`
- `data/results/monthly_selection_benchmark_suite_2026-05-01_relative.csv`
- `data/results/monthly_selection_benchmark_suite_2026-05-01_monthly_series.csv`
- `data/results/monthly_selection_benchmark_suite_2026-05-01_manifest.json`
- `docs/reports/2026-05/monthly_selection_benchmark_suite_2026-05-01.md`
