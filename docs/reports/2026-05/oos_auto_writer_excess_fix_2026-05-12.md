# OOS Auto Writer Excess Fix

- 日期：2026-05-12
- 范围：`src/monitoring/oos_auto_writer.py`
- 验证：`pytest -q -o addopts='' tests/test_oos_auto_writer.py tests/test_oos_tracker.py` → 29 passed

## 背景

`record_oos_from_m7_report()` 会在本月推荐生成后写入预测 OOS，并尝试回填上一期预测的实现表现。函数和字段语义均为 `realized_excess_monthly`，但原实现只计算 Top-K 持仓 T+1 open 到次月 open 的平均收益，没有扣除同期市场等权收益。

这会让 OOS #2 复核时的 realized excess 偏高或偏低，取决于当月市场方向。

## 修复

新增内部明细结构 `RealizedExcessDetail`，回填时同时计算：

| 字段 | 口径 |
|------|------|
| `portfolio_return` | Top-K 持仓 T+1 open 到次月 open 的平均收益 |
| `benchmark_return` | `a_share_daily` 全市场同窗口 open-to-open 等权收益 |
| `realized_excess` | `portfolio_return - benchmark_return` |

`_compute_realized_excess_from_holdings()` 保持原有返回 float 的兼容接口；`_try_backfill_previous()` 改为使用明细接口，并在更新 OOS 记录时同步写入 `benchmark_return`。

`OOSWriteResult` 同步暴露 `backfilled_realized_excess`、`backfilled_portfolio_return`、`backfilled_benchmark_return`，月报脚本的回填日志改为打印 realized excess 和 benchmark，不再误打印上一期预测值。

## 测试覆盖

新增两个场景：

1. 私有计算函数：3 只持仓上涨 10%，全市场 5 只等权收益 6%，返回 realized excess = 4%。
2. 自动回填流程：上一期预测记录带 holdings，新一期报告触发 backfill 后，`oos_tracking.realized_excess_monthly = 4%` 且 `benchmark_return = 6%`。

## 影响

2026-05-29 OOS #2 录入时，自动回填将按真正超额收益写入，和计划中的 OOS gate / 降级复核口径一致。
