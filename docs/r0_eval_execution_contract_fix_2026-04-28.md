# R0 评估/执行契约修复

- 日期：`2026-04-28`
- 范围：对应 `docs/plan.md` 的「## 2. 必须先处理的评估/执行口径风险」
- 结论：R0 的核心执行/评估口径风险已完成代码修复与单元测试。后续 R1F/R2B 可以引用 `eval_contract_version=r0_eval_execution_contract_2026-04-28`。

## 1. 已修复项目

### 1.1 `limit_up_hits_20d` 语义修复

`scripts/run_p1_strong_up_attribution.py` 的 R1/R2 路径特征已拆成：

| 字段 | 语义 |
| --- | --- |
| `limit_up_hits_20d` | 仅统计 `pct_chg >= threshold` |
| `limit_down_hits_20d` | 仅统计 `pct_chg <= -threshold` |
| `limit_move_hits_20d` | 统计涨停或跌停任一方向 |

因此 `UPSIDE_C = limit_up_hits_20d + tail_strength_20d` 后续引用的是纯涨停弹性，不再混入跌停路径。

### 1.2 `tplus1_open` 买入失败处理修复

`src/backtest/engine.py` 新增真实入场日 mask：

```text
build_limit_up_open_mask(daily_long)
```

该 mask 优先使用入场日行情中的真实 `pre_close`；若输入数据没有 `pre_close` 列，才回退到同一标的上一交易日 `close.shift(1)`。涨停判定复用 `src.market.tradability.is_open_limit_up_unbuyable` 的 10% / 20% / 30% 板块限制。

`BacktestConfig` 新增：

```text
limit_up_open_mask
```

回测引擎现在只冻结新增/增持部分：

```text
failed_delta = max(target_weight - prior_weight, 0) if limit_up_open else 0
```

其中 `prior_weight` 使用上一段 open-to-open 区间的真实成交后持仓，而不是计划权重。若首次买入连续多日一字涨停，未成交权重会继续保持闲置或按配置重分配，不会在下一交易日被自动当作已持仓；若标的已经真实持有，当日一字涨停不会触发新增买入失败，已有部分继续获得 open-to-open 收益。

`limit_up_mode="redistribute"` 只在存在可买且已有有效目标权重的标的时记录实际重分配；若全部新增买入失败且没有可重分配对象，则失败权重记录为闲置，不再虚报为 redistributed。

`BacktestResult.meta` 也会输出：

| 字段 | 含义 |
| --- | --- |
| `limit_up_detection` | `open_preclose_mask` 或 `disabled_no_mask` |
| `buy_fail_diagnostic` | 失败日期、标的、目标权重、旧权重、失败权重、重分配权重、闲置权重、有效权重 |
| `buy_fail_event_count` | 买入失败事件数 |
| `buy_fail_total_weight` | 全样本失败新增权重合计 |
| `buy_fail_redistributed_weight` | 全样本重分配权重合计 |
| `buy_fail_idle_weight` | 全样本闲置权重合计 |

这些字段已向 full backtest JSON、P1 daily-proxy-like meta 和 P2 dual-sleeve summary 透传，确保后续研究产物可直接审计买入失败处理。

### 1.3 benchmark 口径对齐

新增 open-to-open 全市场等权 benchmark：

| 入口 | 新增/变更 |
| --- | --- |
| `scripts/run_backtest_eval.py` | `build_market_ew_open_to_open_benchmark` |
| `src/models/xtree/p1_workflow.py` | `build_market_ew_open_to_open_benchmark` |
| P1 daily backtest-like proxy | `tplus1_open` 默认使用 open-to-open benchmark |
| full backtest | `tplus1_open` 下 `market_ew` primary benchmark 改为 open-to-open，同时保留 `market_ew_close_to_close` 对照 |

JSON/report 参数中新增：

```text
primary_benchmark_return_mode=open_to_open
comparison_benchmark_return_mode=close_to_close
```

### 1.4 Regime 阈值可交易性

`classify_regimes` 新增：

```text
threshold_mode=diagnostic_only | expanding
```

R1 事后归因默认仍可使用 `diagnostic_only`，但 R2 dual sleeve 的状态决策改为 `expanding`，并在 `lagged_state_by_rebalance` 中输出：

| 字段 | 当前值 |
| --- | --- |
| `state_threshold_mode` | `expanding` |
| `state_threshold_source` | `expanding_through_lagged_state_month` |
| `state_threshold_observations` | 截至 lagged state month 的历史样本数 |
| `state_lag` | `previous_completed_month` |
| `state_feature_available_date` | 上一个完成月月末 |
| `lookahead_check` | `pass` |

同时输出 `regime_p20/p40/p60/p80` 与 `breadth_p30/p70`，使每个 rebalance 使用的状态阈值可追溯。

### 1.5 Universe gate 与 dual sleeve 表达

本轮没有直接放宽 universe，也没有把 dual sleeve 继续作为主表达。

当前处理是：

1. universe ablation 仍保留为后续 R2B 诊断，不进入生产股票池。
2. R2 dual sleeve 脚本只作为失败对照与 R1F sanity check；后续主线应转向 capped replacement slots。

### 1.6 生产入口与 lint

`scripts/daily_run.py` 已修复 `log` 初始化顺序，`log` 在 P1 因子过滤分支首次使用前完成初始化。

同时清理了现有 F 类 lint 错误，包括未用 import、未用局部变量、无占位 f-string 等。

## 2. 验证

已执行：

```bash
pytest -q
ruff check src scripts tests --select F
```

结果：

| 命令 | 结果 |
| --- | --- |
| `pytest -q` | pass，`1` 个既有 warning |
| `ruff check src scripts tests --select F` | pass |
| `ruff check src scripts tests` | 仍有非 F 类债务：`I001` import 排序与 `E741` 模糊变量名 |

完整 `ruff check` 未在本轮强行全量格式化，因为第 2 节验收要求是至少清零 F 类真实错误；剩余项属于风格/排序债，不影响 R0 执行契约。

## 3. 后续引用口径

后续实验应记录：

```text
eval_contract_version=r0_eval_execution_contract_2026-04-28
execution_contract_version=tplus1_open_buy_delta_limit_mask_2026-04-28
primary_benchmark_return_mode=open_to_open
state_threshold_mode=expanding  # 仅状态决策；R1 事后归因可为 diagnostic_only
```

R1F 复跑时需要特别比较：

1. `UPSIDE_C` 修复前后 `limit_up_hits_20d` 暴露差异。
2. buy-fail diagnostic 中新增/增持失败权重。
3. open-to-open primary benchmark 与 close-to-close comparison benchmark 的差异。
4. R2 dual sleeve 在 expanding threshold 下是否仍 reject。
