# P1 Weekly KDJ Interaction A/B

- 生成时间：`2026-04-26`
- 目标：验证 `weekly_kdj` gated / interaction 特征是否优于直接加入原始 `weekly_kdj_*`
- 入口：`scripts/run_p1_tree_groups.py`
- 口径：`history_start=2020-01-01`，`sample_start=2021-01-01`，`Top-20`，`M`，`label_horizons=5,10,20`，`proxy_horizon=5`
- 结果文件：`data/results/p1_tree_weekly_interaction_ab_2021_rb_m_top20_lh_5-10-20_px_5_val20_20260426_074205_summary.csv`

## 分组

| group | 定义 |
| --- | --- |
| `G0` | baseline technical |
| `G1` | `G0 + weekly_kdj_*` |
| `G5` | `G0 + weekly_kdj_* + weekly_kdj_interaction_*` |
| `G6` | `G0 + weekly_kdj_interaction_*` |

## Light Proxy 结果

| group | raw val Rank IC | proxy excess | strategy ann return | Sharpe | MaxDD | beat rate | periods |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `G0` | `-0.1270` | `18.33%` | `20.39%` | `1.621` | `1.08%` | `76.92%` | `13` |
| `G1` | `-0.1277` | `17.87%` | `19.93%` | `1.647` | `3.30%` | `76.92%` | `13` |
| `G5` | `-0.1316` | `14.28%` | `16.36%` | `1.423` | `3.08%` | `76.92%` | `13` |
| `G6` | `-0.1332` | `17.65%` | `19.77%` | `1.699` | `2.38%` | `84.62%` | `13` |

说明：本轮四组 raw `val_rank_ic` 均为负，推理端按既有保护逻辑自动翻转 `tree_score`。后续 summary 已补 `tree_score_auto_flipped` 和 `effective_val_rank_ic` 字段，避免 raw IC 与实际排序方向混淆。

## 结论

1. `G5/G6` 没有在 light proxy 上超过 `G0`，不满足进入 full backtest 的条件。
2. `G6` 的 beat rate 与 Sharpe 略好于 `G0/G1`，但 proxy excess 仍低于 `G0`，只能作为观察项，不能 promotion。
3. `G1` 相对 `G0` 几乎持平但略差，支持此前判断：原始 `weekly_kdj_*` 直接进入树模型也没有形成稳定增量。
4. P1 下一步不应继续扩大 weekly_kdj interaction 网格；更值得做的是先解决树模型方向稳定性与标签/目标函数口径问题。

