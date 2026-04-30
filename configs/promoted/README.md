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

1. 日更推荐和月度选股的生产配置只能来自 `promoted_registry.json` 的 `promoted_configs`。
2. `daily proxy` 不是 promotion 终点。
3. `gray zone` 不是 production candidate。
4. `config.yaml.backtest` 与 `configs/backtests/` 只服务研究回测和证据归档。
5. `config.yaml.example` 是生产模板，不得承载未 promotion 的研究候选主线。

## 新增 promoted 记录的最低字段

每条 `promoted_configs` 记录至少包含：

| 字段 | 说明 |
| --- | --- |
| `config_id` | promoted 配置 ID |
| `config_path` | promoted 配置快照路径 |
| `promotion_date` | promotion 日期 |
| `full_backtest_report` | 正式回测报告 |
| `daily_proxy_report` | daily proxy 报告 |
| `oos_report` | rolling/slice OOS 报告 |
| `state_slice_report` | strong-up/strong-down 等状态切片 |
| `boundary_report` | switch/topk boundary 诊断 |
| `owner_decision` | 人工确认结论 |

新增记录前必须确认正式 full backtest、OOS、状态切片、执行口径、成本、universe、换手与
boundary diagnostic 均可追溯。
