# P1 Rank Direction Rerun

- 生成时间：`2026-04-26`
- 目标：在修正 `XGBRanker` relevance 标签方向后，重跑 `G0/G1/G5/G6`，确认树模型 raw Rank IC 是否仍依赖推理端自动翻转。
- 入口：`scripts/run_p1_tree_groups.py`
- 口径：`history_start=2020-01-01`，`sample_start=2021-01-01`，`Top-20`，`M`，`label_horizons=5,10,20`，`proxy_horizon=5`，`val_frac=0.2`
- 结果文件：`data/results/p1_tree_rank_direction_rerun_20260426_rb_m_top20_lh_5-10-20_px_5_val20_20260426_085827_summary.csv`
- 方向诊断：`data/results/p1_tree_rank_direction_rerun_20260426_rb_m_top20_lh_5-10-20_px_5_val20_20260426_085827_direction_diagnostic.csv`

## 分组

| group | 定义 |
| --- | --- |
| `G0` | baseline technical |
| `G1` | `G0 + weekly_kdj_*` |
| `G5` | `G0 + weekly_kdj_* + weekly_kdj_interaction_*` |
| `G6` | `G0 + weekly_kdj_interaction_*` |

## Light Proxy 结果

| group | raw val Rank IC | auto flip | proxy excess | strategy ann return | Sharpe | MaxDD | beat rate | periods |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `G0` | `0.1036` | `False` | `10.88%` | `12.95%` | `1.094` | `4.12%` | `69.23%` | `13` |
| `G1` | `0.1084` | `False` | `18.43%` | `21.00%` | `2.131` | `2.44%` | `84.62%` | `13` |
| `G5` | `0.1058` | `False` | `13.55%` | `16.03%` | `1.725` | `3.25%` | `76.92%` | `13` |
| `G6` | `0.1144` | `False` | `12.49%` | `14.88%` | `1.553` | `3.07%` | `69.23%` | `13` |

## 方向诊断

1. 四组 raw `val_rank_ic` 全部转正，`tree_score_auto_flipped=False`。
2. 标签诊断正常：`target_proxy_rank_corr_mean=0.8220`，`target_proxy_rank_corr_negative_rate=0.0`。
3. 这支持当前判断：上一轮 raw IC 全负主要来自旧 rank relevance 标签方向，而不是 `weekly_kdj` 信号天然反向。

## 结论

1. `G1` 是本轮 light proxy 最优组：相对 `G0`，proxy excess 增加 `7.55pct`，Sharpe 和 beat rate 也同步改善。
2. `G6` 的 raw Rank IC 最高，但 proxy excess 不如 `G1`，说明“只保留 weekly interaction 门控表达”尚不能替代原始 `weekly_kdj_*`。
3. `G5` 没有超过 `G1`，说明把原始 weekly 列和 interaction 列简单合并会稀释效果。
4. 下一步应把 `G1` 作为优先 full backtest 候选重跑；`G5/G6` 暂不 promotion，只保留观察。

## G1 Full Backtest 跟进

- 生成时间：`2026-04-26`
- 结果文件：`data/results/p1_full_backtest_g1_rank_direction_20260426.json`
- 模型包：`data/models/xgboost_panel_0f728a7dbdf6`
- 因子 cache：`data/cache/prepared_factors_p1_tree_2021_20260420.parquet`
- 口径：`2021-01-01 ~ 2026-04-20`，`Top-20`，`M`，`max_turnover=0.3`，`equal_weight`，`tplus1_open`，`market_ew_proxy`

| 指标 | `G0` old full | `G1` rank-direction full | 变化 |
| --- | ---: | ---: | ---: |
| with_cost annualized_return | `-19.21%` | `-1.22%` | `+17.99pct` |
| with_cost Sharpe | `-0.718` | `0.030` | `+0.748` |
| with_cost MaxDD | `75.19%` | `40.97%` | `-34.22pct` |
| excess_vs_market annualized_return | `-29.56%` | `-15.34%` | `+14.22pct` |
| rolling OOS annualized_return_agg | `-10.58%` | `9.28%` | `+19.86pct` |
| slice OOS annualized_return_agg | `-19.73%` | `10.25%` | `+29.98pct` |

### 年度表现

| year | strategy | market_ew | excess |
| --- | ---: | ---: | ---: |
| `2021` | `-6.30%` | `30.78%` | `-37.09%` |
| `2022` | `-28.53%` | `-11.10%` | `-17.43%` |
| `2023` | `14.49%` | `7.18%` | `7.32%` |
| `2024` | `16.49%` | `3.39%` | `13.10%` |
| `2025` | `9.69%` | `38.57%` | `-28.88%` |
| `2026` | `-3.99%` | `5.21%` | `-9.19%` |

### Full Backtest 结论

1. rank 标签方向修正后的 `G1` 相比旧 `G0` full backtest 有明显改善，且 rolling/slice OOS 聚合指标转正。
2. 但全样本 benchmark-first 仍未过线：`excess_vs_market annualized_return = -15.34%`。
3. 关键落后年份仍明显存在，尤其是 `2021`、`2025`、`2026`；因此 `G1` 仍不能 promotion 到默认主线。
4. 下一步不应继续简单扩 `weekly_kdj` 特征，而应诊断 `G1` 的年份失效来源：优先拆解持仓与市场风格暴露、调仓月贡献，以及 `2021/2025` 的 top-k 选择错误。
