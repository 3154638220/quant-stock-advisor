# Monthly Selection Benchmark Suite

- 生成时间：`2026-05-05T11:08:58.405043+00:00`
- monthly_long：`data/results/monthly_selection_m8_natural_industry_constraints_30bps_2026_05_04_monthly_long_fixed.csv`
- 模型：`M5_plus_industry_breadth_plus_fund_flow_plus_fundamental_elasticnet_excess__label_blended_excess_50_50__soft_sector_penalty_gamma0_2` · 候选池：`U1_liquid_tradable` · Top-K：`20`

## Return Summary

| benchmark | role | primary | months | total_return | annualized_return | mean_monthly_return | median_monthly_return | monthly_positive_rate | max_drawdown | sharpe |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| model_top20_net | strategy | False | 39 | 30.51% | 8.54% | 0.96% | 1.28% | 56.41% | -35.20% | 0.448642 |
| model_top20_gross | strategy_gross | False | 39 | 44.79% | 12.06% | 1.22% | 1.58% | 66.67% | -32.77% | 0.573933 |
| u1_candidate_pool_ew | alpha_internal | False | 39 | -3.84% | -1.20% | 0.06% | -0.33% | 43.59% | -39.95% | 0.0368505 |
| all_a_market_ew | broad_equal_weight | False | 39 | 5.91% | 1.78% | 0.29% | 0.27% | 51.28% | -35.48% | 0.185682 |

## Excess vs Benchmarks

| strategy | benchmark | months | excess_total_return | annualized_excess_arithmetic | mean_monthly_excess | median_monthly_excess | monthly_hit_rate | win_months | information_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| model_top20_net | u1_candidate_pool_ew | 39 | 34.35% | 9.74% | 0.89% | 0.68% | 58.97% | 23 | 0.815886 |
| model_top20_net | all_a_market_ew | 39 | 24.61% | 6.76% | 0.67% | 0.58% | 53.85% | 21 | 0.594165 |

## Cost Sensitivity (M10)

- 方法: `linear_scale (⚠️ 不应用于 promotion gate)`
- 30bps gate: after_cost_excess_mean > 0 是 promotion 前提条件。
| label | cost_bps | after_cost_excess_mean | after_cost_excess_total | positive_months_ratio | breakeven |
| --- | --- | --- | --- | --- | --- |
| baseline_10bps | 10 | 0.6657% | 25.94% | 53.8% | True |
| stress_30bps | 30 | 0.1295% | 2.24% | 51.3% | True |
| stress_50bps | 50 | -0.4066% | -17.09% | 46.2% | False |

- 估算 Breakeven 成本: `34.8 bps`



## Index Inputs

| benchmark | status | note |
| --- | --- | --- |
| csi1000/csi2000 | not_supplied | 提供 --index-csv 后自动纳入主基准。 |

## 口径

- `model_top20_net` = `topk_return - cost_drag`。
- `u1_candidate_pool_ew` = 同一候选池内股票月度收益等权平均。
- `all_a_market_ew` = dataset 标签层全市场 open-to-open 等权基准。

## 本轮产物

- `data/results/monthly_selection_benchmark_p3_natural_soft_2026-05-05_summary.csv`
- `data/results/monthly_selection_benchmark_p3_natural_soft_2026-05-05_relative.csv`
- `data/results/monthly_selection_benchmark_p3_natural_soft_2026-05-05_monthly_series.csv`
- `data/results/monthly_selection_benchmark_p3_natural_soft_2026-05-05_cost_sensitivity.csv`
- `data/results/monthly_selection_benchmark_p3_natural_soft_2026-05-05_capacity.csv`
- `data/results/monthly_selection_benchmark_p3_natural_soft_2026-05-05_limit_up_stress.csv`
- `data/results/monthly_selection_benchmark_p3_natural_soft_2026-05-05_statistical_tests.csv`
- `data/results/monthly_selection_benchmark_p3_natural_soft_2026-05-05_manifest.json`
- `docs/reports/2026-05/monthly_selection_benchmark_p3_natural_soft_2026-05-05.md`

## Statistical Significance (P3)

### Newey-West Adjusted t-Test (Monthly Excess)

| 指标 | 值 |
|---|---|
| 月均超额 (mean) | 0.6657% |
| NW HAC 标准误 | 0.4487% |
| NW-adjusted t | 1.48 |
| 单侧 p-value | 0.0690 |
| 观测月数 | 39 |
| 零假设 | 月均超额 ≤ 0 |

⚠️ NW-t ≤ 2.0，超额未达显著水平，需更多 OOS 验证。

### Bootstrap 95% CI (Block Bootstrap, Block Size=3)

| 指标 | 值 |
|---|---|
| 均值超额 | 0.6657% |
| 95% CI 下界 | -0.5068% |
| 95% CI 上界 | 1.7611% |
| 观测月数 | 39 |

⚠️ Bootstrap 95% CI 下界 ≤ 0，超额可能不稳健。

### Information Ratio (Monthly)

| 指标 | 值 |
|---|---|
| 月均超额 | 0.6657% |
| 超额标准差 | 3.8810% |
| IR (月频) | 0.17 |
| IR (年化) | 0.59 |

⚠️ IR ≤ 0.5，超额波动较大。
