# P1 Label And Objective Experiment

- 日期：`2026-04-26`
- 结果类型：`light_strategy_proxy` + 代表候选 `full_backtest`
- 研究主题：`p1_tree_groups`
- 固定口径：`G0/G1`、`label_horizons=5,10,20`、等权、`proxy_horizon=5`、`rebalance_rule=M`、`top_k=20`、`val_frac=0.2`
- 代码入口：`scripts/run_p1_tree_groups.py`

## 改动

本轮只改训练目标和标签口径，不扩特征网格：

1. 目标函数增加 `--xgboost-objective rank/regression`，`regression` 也输出可比较的 `val_rank_ic`。
2. 标签口径增加 `--label-mode rank_fusion/raw_fusion/market_relative/benchmark_relative`。
3. `research_config_id` 显式编码 `label_mode` 与 `xgboost_objective`，避免跨标签或跨目标混比。

`market_relative` / `benchmark_relative` 当前使用同日截面等权前瞻收益作为训练端 market proxy；正式比较仍以 full backtest 的 `market_ew` 为准。

## Light Proxy

| 口径 | 组 | Val Rank IC | Proxy 年化超额 | Beat Rate | G1-G0 Proxy |
| --- | --- | ---: | ---: | ---: | ---: |
| `rank + rank_fusion` | G0 | 0.1021 | 9.72% | 69.23% | - |
| `rank + rank_fusion` | G1 | 0.1071 | 16.09% | 84.62% | +6.37% |
| `regression + rank_fusion` | G0 | 0.1410 | 16.09% | 84.62% | - |
| `regression + rank_fusion` | G1 | 0.1386 | 13.07% | 76.92% | -3.02% |
| `rank + market_relative` | G0 | 0.0693 | 13.73% | 76.92% | - |
| `rank + market_relative` | G1 | 0.0749 | 17.98% | 92.31% | +4.26% |

轻量代理层的直观结论：

1. `regression + rank_fusion` 提高了 G0 的验证 Rank IC 和 proxy 表现，但加入 weekly KDJ 后 G1 反而低于 G0。
2. `rank + market_relative` 的 G1 proxy 表现最好，但验证 Rank IC 明显低于 `rank_fusion`，说明它更像改变组合边界，而不是提高全截面排序质量。
3. 轻量层仍不能 promotion，必须看 full backtest。

## Full Backtest

对 light proxy 最好的 `rank + market_relative` 补同窗正式回测：

| 口径 | 组 | 含成本年化 | MaxDD | vs market_ew 年化超额 | Rolling OOS 中位超额 | 年度超额中位数 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `rank + market_relative` | G0 | -2.16% | 30.48% | -16.74% | -0.30% | -9.83% |
| `rank + market_relative` | G1 | -24.56% | 79.59% | -33.97% | -25.79% | -38.51% |
| `regression + rank_fusion` | G0 | -7.61% | 45.51% | -20.87% | -10.97% | -17.21% |

关键年度超额：

| 组 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `rank + market_relative` G0 | -38.57% | -2.96% | +7.24% | +6.22% | -36.63% | -16.69% |
| `rank + market_relative` G1 | -42.45% | -34.57% | -14.13% | -46.15% | -42.46% | -5.40% |
| `regression + rank_fusion` G0 | -27.36% | -24.21% | +0.97% | -10.01% | -33.60% | -10.20% |

## 结论

本轮不改变默认研究基线，也不允许 promotion。

`market_relative` 标签在 light proxy 上看似改善 G1，但正式回测完全不成立，尤其 `2022/2024/2025` 明显恶化。`regression + rank_fusion + G0` 也已经补正式回测：light proxy 为正，但 full backtest 年化超额 `-20.87%`，rolling / slice OOS 超额中位数均为负，按主计划 Promotion Gate 不通过。

P1 的问题不只是“标签是否相对市场”或“目标函数是否用 regression”，而是训练期 proxy Top-K 与正式月频换手/执行成本/全样本时段之间仍有断层。

后续状态：

1. `rank + market_relative + G1` 的 rank bucket 与持仓暴露拆解已补，入口：`docs/p1_marketrel_g1_gap_diagnostics_2026-04-26.md`。
2. `pass_p1_*_gate=True` 属于相对 baseline delta 字段；主计划已明确按绝对 Promotion Gate 判读，本轮不 promotion。

## 产物

- `data/results/p1_label_objective_rank_rankfusion_20260426_rb_m_top20_lh_5-10-20_px_5_val20_lbl_rank_fusion_obj_rank_20260426_124814_summary.csv`
- `data/results/p1_label_objective_reg_rankfusion_20260426_rb_m_top20_lh_5-10-20_px_5_val20_lbl_rank_fusion_obj_regression_20260426_130000_summary.csv`
- `data/results/p1_label_objective_rank_marketrel_20260426_rb_m_top20_lh_5-10-20_px_5_val20_lbl_market_relative_obj_rank_20260426_131137_summary.csv`
- `data/results/p1_label_objective_reg_rankfusion_rb_m_top20_lh_5-10-20_px_5_val20_lbl_rank_fusion_obj_regression_20260426_141830_summary.csv`
- `data/results/p1_full_backtest_g0_marketrel_rank_20260426.json`
- `data/results/p1_full_backtest_g1_marketrel_rank_20260426.json`
- `data/results/p1_full_backtest_g0.json`
- `docs/p1_marketrel_g1_gap_diagnostics_2026-04-26.md`
