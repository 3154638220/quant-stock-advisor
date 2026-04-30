# P1 Rank-Fix Same-Window Full Backtest

- 生成时间：`2026-04-26`
- 目标：补齐 rank 标签方向修正后的 `G0` full backtest，并与同一 rerun 中的 `G1` 做同窗正式对照。
- 入口：`scripts/run_backtest_eval.py`
- 训练来源：`scripts/run_p1_tree_groups.py` 的 `p1_tree_rank_direction_rerun_20260426` 批次
- 口径：`2021-01-01 ~ 2026-04-20`，`Top-20`，`M`，`max_turnover=0.3`，`equal_weight`，`tplus1_open`，`prefilter=false`，`universe_filter=true`，`market_ew_proxy`
- 研究身份：`result_type=full_backtest`，`research_topic=p1_tree_groups`，`research_config_id=rb_m_top20_lh_5_10_20_px_5_val20`，`canonical_config=p1_tree_full_backtest`

## 结果文件

- `G0`: `data/results/p1_full_backtest_g0_rank_direction_20260426.json`
- `G1`: `data/results/p1_full_backtest_g1_rankfix_same_window_20260426.json`
- 训练 summary：`data/results/p1_tree_rank_direction_rerun_20260426_rb_m_top20_lh_5-10-20_px_5_val20_20260426_085827_summary.csv`
- 因子 cache：`data/cache/prepared_factors_p1_tree_2021_20260420.parquet`
- cache schema：`prepared_factors_schema_version=20260424`，`cache_format_version=2`，两组均为 cache hit

## 分组与模型追踪

| group | 定义 | bundle | raw val Rank IC | auto flip |
| --- | --- | --- | ---: | --- |
| `G0` | baseline technical | `data/models/xgboost_panel_f0a634cbbc97` | `0.1036` | `False` |
| `G1` | `G0 + weekly_kdj_*` | `data/models/xgboost_panel_0f728a7dbdf6` | `0.1084` | `False` |

两组标签口径一致：`label_horizons=5,10,20`，等权融合，`target_column=forward_ret_fused`，`proxy_horizon=5`，`scope=cross_section_relative`。本轮不依赖推理端自动翻转。

## Full Backtest 对照

| 指标 | `G0` rank-fix full | `G1` rank-fix full | `G1 - G0` |
| --- | ---: | ---: | ---: |
| with_cost annualized_return | `+3.02%` | `-1.22%` | `-4.24pct` |
| with_cost Sharpe | `0.260` | `0.030` | `-0.230` |
| with_cost MaxDD | `25.12%` | `40.97%` | `+15.85pct` |
| excess_vs_market annualized_return | `-12.35%` | `-15.34%` | `-2.99pct` |
| rolling OOS median excess vs market | `+1.39%` | `-12.12%` | `-13.51pct` |
| slice OOS median excess vs market | `-22.10%` | `-17.42%` | `+4.68pct` |
| turnover_mean | `30.97%` | `31.61%` | `+0.64pct` |

## 年度表现

| year | `G0` strategy | `G0` excess | `G1` strategy | `G1` excess | `G1 - G0` excess |
| --- | ---: | ---: | ---: | ---: | ---: |
| `2021` | `-4.53%` | `-35.32%` | `-6.30%` | `-37.09%` | `-1.77pct` |
| `2022` | `-6.43%` | `+4.67%` | `-28.53%` | `-17.43%` | `-22.10pct` |
| `2023` | `+14.33%` | `+7.15%` | `+14.49%` | `+7.32%` | `+0.17pct` |
| `2024` | `+12.71%` | `+9.32%` | `+16.49%` | `+13.10%` | `+3.78pct` |
| `2025` | `+5.07%` | `-33.50%` | `+9.69%` | `-28.88%` | `+4.62pct` |
| `2026` | `-4.10%` | `-9.31%` | `-3.99%` | `-9.19%` | `+0.12pct` |

## 结论

1. `G1` 在 light proxy 上仍是最优，但这个优势没有迁移到正式 full backtest；同窗 full backtest 下，`G1` 的全样本收益、MaxDD、年度超额和 rolling OOS 中位超额都弱于 `G0`。
2. `weekly_kdj_*` 对 `2024 / 2025 / 2026` 有局部改善，尤其 `2025` 年超额改善 `+4.62pct`，但它在 `2022` 年造成明显退化，抵消了后续收益。
3. `G0` 自身也没有通过 benchmark-first gate：`annualized_excess_vs_market=-12.35%`，`slice OOS median excess=-22.10%`，关键强基准年份 `2021 / 2025 / 2026` 仍显著落后。
4. 因此当前 P1 结论从“`G1` 明显优于旧 `G0`”修正为“rank-fix 同窗下 `G1` 不优于 `G0`，二者都未过 market_ew gate”。

## 是否改变主计划

改变 P1 的下一步优先级，但不改变默认研究基线，也不允许 promotion。

- 不再把 `G1` 视为已证明的 full backtest 增量。
- 下一步诊断应同时拆 `G0/G1`，重点解释为什么 light proxy 的 `G1` 优势在正式回测中消失。
- 优先检查 `2022` 的 `G1` 退化、`2021/2025` 的共同上涨期参与不足，以及 full backtest 和 light proxy 在调仓日、持有期、执行收益上的口径差异。
