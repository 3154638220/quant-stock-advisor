# W1 因子族边际贡献分析

- 生成日期：2026-05-12
- 对比口径：U1 liquid tradable / Top20 / `M8_regime_aware_fixed_policy__indcap3`
- 成本口径：16 bps，使用各 T1 重跑产物的 `topk_excess_after_cost_mean`
- 证据来源：2026-05-10 I9 修复后四组 M8 T1 重跑

## 结论

W1 新增因子族中，只有 `quality` 对当前 M8 生产框架有正向边际贡献。`reversal_volume` 在加入 `quality` 后造成明显负贡献，`liquidity_position` 虽能部分修复 `reversal_volume` 的损害，但组合仍低于生产 baseline，因此不应进入当前 promoted 配置。

## 同基对比

| 配置 | 新增族 | after-cost 超额/月 | delta vs baseline | delta vs 上一组合 | Rank IC | Hit rate | TopK-NextK |
|------|--------|-------------------:|------------------:|------------------:|--------:|---------:|-----------:|
| M8 baseline | - | 1.99% | - | - | 0.0973 | 69.2% | 0.91% |
| M8 + quality | quality | 2.15% | +0.16% | +0.16% | 0.1023 | 66.7% | 1.24% |
| M8 + quality + reversal_volume | reversal_volume | 1.34% | -0.65% | -0.81% | 0.1011 | 53.8% | 0.97% |
| M8 + all W1 | liquidity_position | 1.82% | -0.17% | +0.48% | 0.0983 | 56.4% | 1.47% |

## 边际贡献判断

| 因子族 | 生产边际贡献 | 判断 | 处理 |
|--------|-------------:|------|------|
| `quality` | +0.16%/月 vs baseline | 唯一稳定正贡献；Rank IC 同步提升 | 保留在 `monthly_selection_m8_indcap3_plus_quality` |
| `reversal_volume` | -0.81%/月 vs `+quality` | 明显破坏 M8 组合收益；hit rate 降低 12.8 pp | 不进入当前 M8 生产配置 |
| `liquidity_position` | +0.48%/月 vs `+quality+rv`，但 -0.33%/月 vs `+quality` | 只能修复部分 rv 损害，无法超过 baseline 或 quality-only | 不进入当前 M8 生产配置 |

## 解释

`reversal_volume` 的 Rank IC 与 `quality` 接近，但 after-cost 超额大幅下降，说明问题不在单纯排序相关性，而在 M8 regime-aware selection 的组合层面：新增信号改变了入选股票分布，压低 hit rate，且无法被 indcap3 行业分散约束修复。

`liquidity_position` 在 all W1 中带来相对 `quality+rv` 的修复，但 all W1 仍低于 baseline 0.17%/月、低于 quality-only 0.33%/月。该族可保留在因子库中等待未来模型架构重评，不适合当前 M8 promoted 主线。

## 证据文件

- Baseline: `data/results/monthly_selection_m8_baseline_rerun_2026_05_10_2026_05_10_leaderboard.csv`
- Quality: `data/results/monthly_selection_m8_plus_quality_2026_05_10_2026_05_10_leaderboard.csv`
- Quality + reversal_volume: `data/results/monthly_selection_m8_plus_quality_rv_2026_05_10_2026_05_10_leaderboard.csv`
- All W1: `data/results/monthly_selection_m8_plus_w1_all_2026_05_10_2026_05_10_leaderboard.csv`

