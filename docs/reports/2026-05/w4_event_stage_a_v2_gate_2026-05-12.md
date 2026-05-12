# W4 Event Stage A v2 Gate

- 生成时间：`2026-05-12`
- 目标：重跑 Event v2 稀疏窗口特征后的 Stage A IC Gate
- 口径：M5 `price_volume_only` vs `price_volume,event`，Top20/30/50，U1/U2，10bps cost
- 命令：

```bash
python scripts/run_monthly_selection_multisource.py \
  --families event \
  --output-prefix w4_event_stage_a_v2_2026_05_12 \
  --ml-models elasticnet,logistic,extratrees \
  --model-n-jobs 0
```

## Gate 结论

**Stage A v2 不通过。** Event v2 的长窗口特征修复了 buyback / reduction / unlock 的覆盖问题，但加入 `event` 后 Rank IC delta 仍接近 0，未形成稳定增量。

| pool | model | top_k | after-cost | baseline after-cost | delta after-cost | rank IC | baseline rank IC | delta rank IC |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| U1 | ExtraTrees | 20 | 0.78% | 1.03% | -0.24% | 0.1085 | 0.1084 | +0.0000 |
| U1 | ElasticNet | 20 | 0.75% | 0.73% | +0.02% | 0.1028 | 0.1025 | +0.0003 |
| U2 | ExtraTrees | 20 | 0.54% | 0.53% | +0.01% | 0.0886 | 0.0885 | +0.0001 |
| U2 | ElasticNet | 20 | 0.46% | 0.50% | -0.04% | 0.0828 | 0.0826 | +0.0002 |

Stage A 的核心判断看 Rank IC 增量。当前 delta 仅 `+0.0000 ~ +0.0003`，低于有效新因子族所需的可解释增量，且收益侧在 U1/ExtraTrees 上明显走弱。

## 覆盖率变化

| feature | coverage | candidate-pool coverage |
| --- | ---: | ---: |
| earnings guidance direction | 71.01% | 71.89% |
| earnings guidance magnitude | 70.35% | 71.29% |
| earnings surprise ttm | 49.58% | 50.85% |
| buyback recent 180d | 4.25% | 4.55% |
| reduction plan 180d | 24.47% | 26.50% |
| reduction ratio 180d | 24.47% | 26.50% |
| unlock ratio 90d | 8.46% | 7.36% |

v2 已解决原始 30d 稀疏窗口导致 reduction / unlock 基本不可见的问题。覆盖率改善没有转化成 Rank IC 改善，说明事件族当前主要问题从数据可用性转为信号强度不足。

## 特征重要性观察

U1 ExtraTrees 中事件特征仍有非零使用：`earnings_guidance_direction_z` importance 0.0124，`unlock_ratio_30d_z` 0.0081，`buyback_recent_30d_z` 0.0076，`unlock_ratio_90d_z` 0.0057。但这些使用没有带来 Rank IC 增量，且 Top20 after-cost 下降。

ElasticNet 的事件权重整体很小，最大为 `unlock_ratio_30d_z` 0.0011，说明线性模型几乎没有从事件族获得稳定边际信息。

## 处理建议

1. `event` 全族不进入 M8 Baseline Gate，不作为 promotion 候选。
2. 事件因子保留在 registry 中，供 M16 结构化事件专项继续探索。
3. 下一轮不再扩大窗口，而应做子类拆分：优先单独复核 earnings guidance；buyback / reduction / unlock 需要先改善字段质量和事件语义，再重跑 Stage A。

## 产物

- `data/results/w4_event_stage_a_v2_2026_05_12_2026-05-12_summary.json`
- `data/results/w4_event_stage_a_v2_2026_05_12_2026-05-12_leaderboard.csv`
- `data/results/w4_event_stage_a_v2_2026_05_12_2026-05-12_incremental_delta.csv`
- `data/results/w4_event_stage_a_v2_2026_05_12_2026-05-12_feature_coverage.csv`
- `data/results/w4_event_stage_a_v2_2026_05_12_2026-05-12_feature_importance.csv`
- `data/results/w4_event_stage_a_v2_2026_05_12_2026-05-12_manifest.json`
- `docs/w4_event_stage_a_v2_2026_05_12_2026-05-12.md`
