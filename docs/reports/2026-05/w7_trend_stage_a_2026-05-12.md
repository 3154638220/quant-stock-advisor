# W7 Trend Persistence Stage A IC Gate

- 日期：2026-05-12
- 命令：`python scripts/run_monthly_selection_multisource.py --families price_volume,trend_persistence --output-prefix w7_trend_stage_a_2026_05_12 --ml-models elasticnet,extratrees --model-n-jobs 0`
- 数据：`data/cache/monthly_selection_features.parquet` + `data/market.duckdb`
- 结果：Stage A 未通过，不进入 Stage B

## 产物

- `data/results/w7_trend_stage_a_2026_05_12_2026-05-12_leaderboard.csv`
- `data/results/w7_trend_stage_a_2026_05_12_2026-05-12_incremental_delta.csv`
- `data/results/w7_trend_stage_a_2026_05_12_2026-05-12_rank_ic.csv`
- `data/results/w7_trend_stage_a_2026_05_12_2026-05-12_individual_factor_ic.csv`
- `data/results/w7_trend_stage_a_2026_05_12_2026-05-12_feature_coverage.csv`
- `data/results/w7_trend_stage_a_2026_05_12_2026-05-12_manifest.json`

## 单因子 IC

| pool | factor | Rank IC mean | IC IR | 正向期数占比 |
| --- | --- | ---: | ---: | ---: |
| U1 | `feature_trend_flip_days_ago` | 0.0103 | 0.1502 | 52.4% |
| U1 | `feature_trend_bull_ratio_60d` | -0.0438 | -0.3122 | 36.5% |
| U1 | `feature_trend_bull_ratio_20d` | -0.0540 | -0.3873 | 31.7% |
| U1 | `feature_trend_bull_state` | -0.0583 | -0.4882 | 34.9% |
| U1 | `feature_trend_streak_days` | -0.0607 | -0.4434 | 31.7% |
| U1 | `feature_trend_ema_spread` | -0.0834 | -0.5823 | 28.6% |
| U2 | `feature_trend_flip_days_ago` | 0.0196 | 0.2848 | 61.9% |
| U2 | `feature_trend_bull_ratio_60d` | -0.0318 | -0.2275 | 39.7% |
| U2 | `feature_trend_bull_ratio_20d` | -0.0385 | -0.2848 | 34.9% |
| U2 | `feature_trend_streak_days` | -0.0432 | -0.3233 | 34.9% |
| U2 | `feature_trend_bull_state` | -0.0434 | -0.3747 | 39.7% |
| U2 | `feature_trend_ema_spread` | -0.0601 | -0.4306 | 31.7% |

`feature_trend_flip_days_ago` 是唯一正 IC 子因子，但 U1/U2 的 IC IR 均低于 0.3；其余方向类趋势因子呈负 IC，说明“月末处于多头/强趋势”在截面选股上更像拥挤或短期过热信号，而不是正向延续。

## 模型级增量

| pool | model | top_k | after-cost delta | Rank IC delta |
| --- | --- | ---: | ---: | ---: |
| U1 | elasticnet | 20 | +0.0061 | -0.0071 |
| U1 | extratrees | 20 | -0.0039 | -0.0051 |
| U2 | elasticnet | 20 | +0.0044 | -0.0086 |
| U2 | extratrees | 20 | +0.0027 | -0.0042 |

Top-20 after-cost delta 在 elasticnet 上为正，但 Stage A 的核心门槛是单因子 IC 与 `price_volume + trend_persistence` 相对 `price_volume only` 的 Rank IC delta。模型级 Rank IC delta 全部为负，不能支撑进入 M8 Baseline Gate。

## 结论

Stage A 未通过。当前 EMA12/EMA26 方案 A 不应作为正向 `trend_persistence` 因子族进入 M8 主线。

## 过滤支线复查

已补充运行候选池过滤诊断：

```bash
python scripts/run_trend_filter_audit.py \
  --output-prefix w7_trend_filter_audit_2026_05_12
```

产物：

- `data/results/w7_trend_filter_audit_2026_05_12_2026-05-12_summary.csv`
- `data/results/w7_trend_filter_audit_2026_05_12_2026-05-12_monthly.csv`
- `docs/reports/2026-05/w7_trend_filter_audit_2026_05_12_2026-05-12.md`

过滤复查结论：

| pool | filter | 月均保留数 | vs 同池全集月均超额增量 | t |
| --- | --- | ---: | ---: | ---: |
| U1 | `bull_state` | 1644 | -0.0052 | -2.48 |
| U2 | `bull_state` | 1359 | -0.0040 | -1.95 |
| U1 | `stable_q80` | 688 | +0.0014 | 0.84 |
| U2 | `stable_q70` | 929 | +0.0017 | 1.41 |
| U2 | `stable_q80` | 618 | +0.0021 | 1.37 |

`bull_state` 作为候选池过滤为负向；`stable_q70/q80` 虽为弱正且月均样本量充足，但统计强度不足，不支持进入 T4 M8 Baseline Gate。

后续若继续探索，仅保留离线反向假设研究：

1. 对方向类因子测试反向方向，验证“多头状态=过热”假设。
2. 若未来重启候选池过滤，只使用 `feature_trend_flip_days_ago` 的稳定性路线，避免把负 IC 的多头状态类因子混入。
