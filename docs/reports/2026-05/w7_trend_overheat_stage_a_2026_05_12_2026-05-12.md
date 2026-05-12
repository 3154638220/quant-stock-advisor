# W7 Trend Overheat Reversal Stage A

- 生成时间：`2026-05-12`
- 目的：将 W7 原趋势持续性失败后的反向诊断拆成独立 `trend_overheat_reversal` 因子族，重新验证 Stage A。
- 命令：`python scripts/run_monthly_selection_multisource.py --families price_volume,trend_overheat_reversal --output-prefix w7_trend_overheat_stage_a_2026_05_12 --ml-models elasticnet,extratrees --model-n-jobs 0`

## Conclusion

`trend_overheat_reversal` Stage A **未通过**。

新因子覆盖率充足，候选池通过样本覆盖率约 `99.997%`，但加入 `price_volume` 后，所有 U1/U2、ElasticNet/ExtraTrees 组合的 Rank IC delta 均为负。少数组合的 top-k after-cost 超额为正，不能抵消 IC Gate 的失败，不进入 M8 Baseline Gate。

## Key Metrics

| pool | model | top_k | after-cost delta | rank IC delta | quantile spread delta |
| --- | --- | --- | --- | --- | --- |
| U1 | elasticnet | 20 | -0.097pct | -0.003315 | -0.000999 |
| U1 | extratrees | 20 | -0.148pct | -0.003385 | -0.000735 |
| U2 | extratrees | 20 | +0.417pct | -0.005125 | -0.000813 |
| U2 | elasticnet | 20 | -0.031pct | -0.004216 | -0.001487 |
| U1 | extratrees | 30 | +0.304pct | -0.003385 | -0.000735 |
| U2 | extratrees | 30 | +0.265pct | -0.005125 | -0.000813 |

## Coverage

| feature | coverage | candidate-pool-pass coverage |
| --- | --- | --- |
| `feature_trend_overheat_bear_state` | 98.106% | 99.997% |
| `feature_trend_overheat_cooling_streak_days` | 98.106% | 99.997% |
| `feature_trend_overheat_ema_spread_reversal` | 98.106% | 99.997% |

## Artifacts

- `data/results/w7_trend_overheat_stage_a_2026_05_12_2026-05-12_leaderboard.csv`
- `data/results/w7_trend_overheat_stage_a_2026_05_12_2026-05-12_incremental_delta.csv`
- `data/results/w7_trend_overheat_stage_a_2026_05_12_2026-05-12_rank_ic.csv`
- `data/results/w7_trend_overheat_stage_a_2026_05_12_2026-05-12_feature_coverage.csv`
- `data/results/w7_trend_overheat_stage_a_2026_05_12_2026-05-12_manifest.json`

## Disposition

- 保留 `trend_overheat_reversal` 代码与测试，作为已验证失败的研究分支，便于复现。
- 不修改原 `trend_persistence` 方向，不进入生产配置。
- 不执行 W7 T4 M8 Baseline Gate。
