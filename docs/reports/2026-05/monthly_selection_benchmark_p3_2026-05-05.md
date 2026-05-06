# Monthly Selection Benchmark Suite

- 生成时间：`2026-05-05T10:52:50.693048+00:00`
- monthly_long：`data/results/monthly_selection_m8_concentration_regime_2026-05-01_monthly_long.csv`
- 模型：`M8_regime_aware_fixed_policy__indcap3` · 候选池：`U1_liquid_tradable` · Top-K：`20`

## Return Summary

| benchmark | role | primary | months | total_return | annualized_return | mean_monthly_return | median_monthly_return | monthly_positive_rate | max_drawdown | sharpe |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| model_top20_net | strategy | False | 39 | 106.79% | 25.05% | 2.05% | 1.40% | 66.67% | -11.71% | 1.20552 |
| model_top20_gross | strategy_gross | False | 39 | 114.25% | 26.42% | 2.14% | 1.49% | 66.67% | -11.53% | 1.25986 |
| u1_candidate_pool_ew | alpha_internal | False | 39 | -3.84% | -1.20% | 0.06% | -0.33% | 43.59% | -39.95% | 0.0368505 |
| all_a_market_ew | broad_equal_weight | False | 39 | 5.91% | 1.78% | 0.29% | 0.27% | 51.28% | -35.48% | 0.185682 |

## Excess vs Benchmarks

| strategy | benchmark | months | excess_total_return | annualized_excess_arithmetic | mean_monthly_excess | median_monthly_excess | monthly_hit_rate | win_months | information_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| model_top20_net | u1_candidate_pool_ew | 39 | 110.63% | 26.25% | 1.98% | 1.76% | 71.79% | 28 | 1.80422 |
| model_top20_net | all_a_market_ew | 39 | 100.89% | 23.27% | 1.75% | 0.96% | 66.67% | 26 | 1.77178 |

## Cost Sensitivity (M10)

- 方法: `linear_scale (⚠️ 不应用于 promotion gate)`
- 30bps gate: after_cost_excess_mean > 0 是 promotion 前提条件。
| label | cost_bps | after_cost_excess_mean | after_cost_excess_total | positive_months_ratio | breakeven |
| --- | --- | --- | --- | --- | --- |
| baseline_10bps | 10 | 1.7547% | 92.90% | 66.7% | True |
| stress_30bps | 30 | 1.5698% | 79.68% | 66.7% | True |
| stress_50bps | 50 | 1.3849% | 67.35% | 66.7% | True |

- 估算 Breakeven 成本: `75.0 bps`



## Index Inputs

| benchmark | status | note |
| --- | --- | --- |
| csi1000/csi2000 | not_supplied | 提供 --index-csv 后自动纳入主基准。 |

## 口径

- `model_top20_net` = `topk_return - cost_drag`。
- `u1_candidate_pool_ew` = 同一候选池内股票月度收益等权平均。
- `all_a_market_ew` = dataset 标签层全市场 open-to-open 等权基准。

## 本轮产物

- `data/results/monthly_selection_benchmark_p3_2026-05-05_summary.csv`
- `data/results/monthly_selection_benchmark_p3_2026-05-05_relative.csv`
- `data/results/monthly_selection_benchmark_p3_2026-05-05_monthly_series.csv`
- `data/results/monthly_selection_benchmark_p3_2026-05-05_cost_sensitivity.csv`
- `data/results/monthly_selection_benchmark_p3_2026-05-05_capacity.csv`
- `data/results/monthly_selection_benchmark_p3_2026-05-05_limit_up_stress.csv`
- `data/results/monthly_selection_benchmark_p3_2026-05-05_statistical_tests.csv`
- `data/results/monthly_selection_benchmark_p3_2026-05-05_manifest.json`
- `docs/reports/2026-05/monthly_selection_benchmark_p3_2026-05-05.md`

## Statistical Significance (P3)

### Newey-West Adjusted t-Test (Monthly Excess)

| 指标 | 值 |
|---|---|
| 月均超额 (mean) | 1.7547% |
| NW HAC 标准误 | 0.4807% |
| NW-adjusted t | 3.65 |
| 单侧 p-value | 0.0001 |
| 观测月数 | 39 |
| 零假设 | 月均超额 ≤ 0 |

✅ NW-t > 2.0，超额在时序自相关调整后显著非零。

### Bootstrap 95% CI (Block Bootstrap, Block Size=3)

| 指标 | 值 |
|---|---|
| 均值超额 | 1.7547% |
| 95% CI 下界 | 0.7512% |
| 95% CI 上界 | 2.7285% |
| 观测月数 | 39 |

✅ Bootstrap 95% CI 下界 > 0，超额在保持时序结构下显著为正。

### Information Ratio (Monthly)

| 指标 | 值 |
|---|---|
| 月均超额 | 1.7547% |
| 超额标准差 | 3.4306% |
| IR (月频) | 0.51 |
| IR (年化) | 1.77 |

✅ IR > 0.5，策略信息比健康。
