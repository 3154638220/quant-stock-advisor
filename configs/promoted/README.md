# Promoted 配置注册表

本目录是生产配置的准入边界。研究配置只有在通过完整 promotion gate 后，才允许在
`promoted_registry.json` 中新增一条 `promoted_configs` 记录；未登记的研究结果不得写入
`config.yaml.example` 或本地生产 `config.yaml` 的默认主线。

## 当前状态

截至 `2026-04-30`，月度选股已有一个 active promoted 默认方法：

- `active_promoted_config_id` 为 `monthly_selection_u1_top20_indcap3_hardcap_baseline`。
- 默认方法为 `U1_liquid_tradable + Top20 + M8_regime_aware_fixed_policy__indcap3`。
- 选择层执行 hard cap：`selection_policy=industry_names_cap`，`max_industry_names=3`。
- `U3_A_real_industry_leadership__EDGE_GATED` 仅曾进入 gray zone，已被 Day 8 weight audit 判定不稳，不能作为 production candidate。
- `U3_B_buyable_leadership_persistence__EDGE_GATED`、`U3_C_pairwise_residual_edge__EDGE_GATED`、旧 R2 sleeve、R2B v1、P1/G0 标签近邻方向均未满足 promotion。

## 使用规则

1. 月度选股的生产配置只能来自 `promoted_registry.json` 的 `promoted_configs`。
2. `gray zone` 不是 production candidate。
3. `config.yaml.backtest` 与 `configs/backtests/` 只服务研究回测和证据归档。
4. `config.yaml.example` 是生产模板，不得承载未 promotion 的研究候选主线。

## 新增 promoted 记录的最低字段

每条 `promoted_configs` 记录至少包含：

| 字段 | 说明 |
| --- | --- |
| `config_id` | promoted 配置 ID |
| `config_path` | promoted 配置快照路径 |
| `promotion_date` | promotion 日期 |
| `full_backtest_report` | 正式回测报告 |
| `oos_report` | rolling/slice OOS 报告 |
| `state_slice_report` | strong-up/strong-down 等状态切片 |
| `boundary_report` | switch/topk boundary 诊断 |
| `owner_decision` | 人工确认结论 |

新增记录前必须确认正式 full backtest、OOS、状态切片、执行口径、成本、universe、换手与
boundary diagnostic 均可追溯。

## Promotion Gate 量化通过标准

候选策略须**同时满足**以下所有量化条件，方可通过 Promotion Gate 进入生产：

| 指标 | 阈值 | 说明 |
|------|------|------|
| 年化超额收益（after-cost） | > 5 bps | 扣除交易成本（30–50 bps）后的样本外年化超额，取 3 年 sliding window 最小值 |
| Sharpe Ratio（after-cost） | > 0.3 | 基于 T+1 开盘执行价计算的 O2O 日收益序列年化 Sharpe |
| 最大回撤（excess） | < 8% | 超额累计收益的最大回撤幅度，含成本 |
| 月度胜率 | > 55% | 月度超额 > 0 的比例，基于样本外回测窗口 |
| Calmar Ratio（excess） | > 0.4 | 年化超额 / 最大回撤 |
| 样本外 Rank IC 均值 | > 0.03 | 样本外各月截面 Rank IC（Spearman）均值 |
| M10 成本压力测试 | 通过 | 30/50 bps 两档 after-cost excess 均为正；成本上升 20 bps 后超额下降 < 50% |
| 最大单票权重 | ≤ 12% | 回测期间任意换仓日的单票权重上限 |
| 换手率（单边） | < 40% | 月均单向换手率 |

**Gate 评估流程**：

1. **全量回测**：至少覆盖 36 个月样本外窗口（最近 12 个月为纯 OOS，不参与训练）。
2. **M10 成本复核**：30 bps / 50 bps 两档 cost 下 after-cost excess 均须保持正。
3. **边界诊断**：Top20 边界替换测试（replace-1 / replace-3）稳定性通过。
4. **状态切片**：strong-up / choppy / strong-down 三种市场状态下超额均为正（或 at least not worse than baseline）。
5. **Owner 人工确认**：以上全部通过后，由 Owner 在 `promoted_registry.json` 中写入 ``owner_decision`` 字段确认 promotion。

**降级触发条件（OOS 自动监控）**：

- 连续 3 个月实现超额低于回测均值的 50%
- 单月超额 < -2%
- 月换手率 > 50%

触发任一条件后，自动将策略标记为 ``oos_degraded``，需重新走 Gate 评估流程。
