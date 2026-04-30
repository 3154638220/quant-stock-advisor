# P1 Daily Proxy First Policy

- 日期：`2026-04-27`
- 结果类型：`research_policy`
- 适用范围：`scripts/run_p1_tree_groups.py` 及 P1 树模型候选

## 决策

P1 从本轮起采用 daily-proxy-first。旧 `light_strategy_proxy` 和 `full_like_proxy` 不再触发正式回测，不再参与 promotion 判断，只保留为 legacy diagnostic。

新的主准入指标是：

`daily_bt_like_proxy_annualized_excess_vs_market`

它直接复用日频 open-to-open 收益、交易成本、换手约束和 `market_ew_proxy` 口径，比旧 proxy 更接近正式 full backtest。

## 默认 Gate

| daily proxy 年化超额 | 状态 | 处理 |
| ---: | --- | --- |
| `< 0%` | `reject` | 停止，不补正式 full backtest |
| `0% ~ +3%` | `gray_zone` | 只归档诊断，默认不补正式 full backtest |
| `>= +3%` | `full_backtest_candidate` | 允许补正式 full backtest |

参数：

```bash
--daily-proxy-admission-threshold 0.0
--daily-proxy-full-backtest-threshold 0.03
```

`+3%` 是 safety margin，不是收益目标，也不是 promotion 线。它来自当前校准：daily proxy 相对正式 full backtest 最大绝对偏差约 `2.44pct`。

## Runner 输出

`scripts/run_p1_tree_groups.py` 当前默认输出：

1. `*_summary.csv`
2. `*_detail.csv`
3. `*_daily_proxy_leaderboard.csv`
4. `*_daily_proxy_monthly_state.csv`
5. `*_daily_proxy_state_summary.csv`
6. `*_topk_boundary.csv`
7. `*_bundle_manifest.csv`
8. `*_direction_diagnostic.csv`
9. JSON payload

summary 中新增或固化的关键字段：

| 字段 | 含义 |
| --- | --- |
| `result_type` | P1 主结果为 `daily_bt_like_proxy` |
| `primary_result_type` | `daily_bt_like_proxy` |
| `primary_decision_metric` | `daily_bt_like_proxy_annualized_excess_vs_market` |
| `legacy_proxy_decision_role` | `diagnostic_only` |
| `daily_proxy_first_status` | `reject` / `gray_zone` / `full_backtest_candidate` / `no_daily_proxy` |
| `pass_p1_daily_proxy_admission_gate` | 是否过硬停止线 |
| `pass_p1_daily_proxy_full_backtest_gate` | 是否达到正式回测触发线 |
| `daily_proxy_safety_margin_to_full_backtest` | daily proxy 距正式回测触发线的距离 |

旧 proxy 字段仍保留，但只能解释，不能决策：

1. `annualized_excess_vs_market`
2. `legacy_unconstrained_proxy_annualized_excess_vs_market`
3. `full_like_proxy_annualized_excess_vs_market`
4. `legacy_full_like_proxy_annualized_excess_vs_market`
5. `proxy_gap_full_like_minus_unconstrained`
6. `proxy_gap_daily_bt_like_minus_unconstrained`

## Full Backtest 触发

`--run-full-backtest` 打开时，runner 仍会先跑 daily proxy。若 daily proxy 低于硬停止线或低于 `+3%` 触发线，正式 full backtest 会被跳过，并在 summary 写入 `full_backtest_skipped_reason`。

只有在以下条件同时满足时，才允许补正式 full backtest：

1. `daily_proxy_first_status=full_backtest_candidate`
2. `pass_p1_daily_proxy_full_backtest_gate=True`
3. 状态切片没有明显恶化。
4. Top-K 边界没有明显反向。
5. 不依赖 `tree_score_auto_flipped=True` 才工作。

## Promotion 边界

daily proxy 不是 promotion 终点。promotion 仍必须经过正式 full backtest：

1. full backtest 年化超额不为负。
2. 年度超额中位数不为负。
3. rolling OOS 和 slice OOS 的超额中位数不为负。
4. 强上涨状态改善。
5. 下跌防守不明显劣化。
6. Top-K 边界和换手有收益补偿。

## 不再做的事

1. 不再为了旧 proxy 做单独断层对齐项目。
2. 不把旧 proxy 正收益当作补正式回测的理由。
3. 不把 daily proxy 当作最终收益或 promotion 结论。
4. 不为了 daily proxy 变好而打开大网格。
