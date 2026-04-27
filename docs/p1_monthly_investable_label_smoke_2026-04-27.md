# P1 Monthly Investable Label Smoke

- 日期：`2026-04-27`
- 结果类型：`daily_bt_like_proxy` / `legacy_light_strategy_proxy` / `topk_boundary_diagnostic`
- 目标：验证 H1 的两个最小标签口径能否先在 `G0` 通过 daily-proxy-first gate。
- 固定口径：`G0` / `regression` / `M` / `Top-20` / `equal_weight` / `max_turnover=0.3` / `tplus1_open`
- 训练窗口：`--history-start 2020-01-01`，`--sample-start 2021-01-01`

## 命令

```bash
python scripts/run_p1_tree_groups.py \
  --config config.yaml.backtest \
  --groups G0 \
  --history-start 2020-01-01 \
  --sample-start 2021-01-01 \
  --label-horizons 5,10,20 \
  --label-mode monthly_investable \
  --xgboost-objective regression \
  --proxy-horizon 5 \
  --rebalance-rule M \
  --top-k 20 \
  --val-frac 0.2 \
  --daily-proxy-admission-threshold 0.0 \
  --daily-proxy-full-backtest-threshold 0.03 \
  --run-full-backtest \
  --backtest-config config.yaml.backtest \
  --backtest-start 2021-01-01 \
  --backtest-top-k 20 \
  --backtest-max-turnover 0.3 \
  --backtest-portfolio-method equal_weight \
  --out-tag p1_next_label_proxy_smoke
```

第二轮仅把 `--label-mode` 改为 `monthly_investable_market_relative`。

## 输出

| 标签模式 | val Rank IC | 旧 light proxy 年化超额 | full-like proxy 年化超额 | daily backtest-like proxy 年化超额 | daily gate | 正式回测 |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| `monthly_investable` | `0.0631` | `+10.60%` | `+4.53%` | `-31.18%` | fail | skipped |
| `monthly_investable_market_relative` | `0.0944` | `+20.02%` | `+18.12%` | `-43.42%` | fail | skipped |

核心产物：

| 标签模式 | JSON / CSV stem |
| --- | --- |
| `monthly_investable` | `data/results/p1_next_label_proxy_smoke_rb_m_top20_lh_5-10-20_px_5_val20_lbl_monthly_investable_obj_regression_20260427_083555` |
| `monthly_investable_market_relative` | `data/results/p1_next_label_proxy_smoke_rb_m_top20_lh_5-10-20_px_5_val20_lbl_monthly_investable_market_relative_obj_regression_20260427_084426` |

## 状态切片

| 标签模式 | strong up 中位超额 | strong up 跑赢比例 | strong down 中位超额 | high vol 中位超额 |
| --- | ---: | ---: | ---: | ---: |
| `monthly_investable` | `-4.95%` | `0.0%` | `+1.87%` | `-0.79%` |
| `monthly_investable_market_relative` | `-5.07%` | `16.7%` | `-0.79%` | `-0.79%` |

## 边界诊断

| 标签模式 | Top-K minus next bucket | Switch-in minus switch-out | 平均边界换手 |
| --- | ---: | ---: | ---: |
| `monthly_investable` | `+0.20pct` | `+0.74pct` | `97.9%` |
| `monthly_investable_market_relative` | `-1.03pct` | `-0.26pct` | `96.2%` |

## Daily Path 断层

| 标签模式 | daily proxy 月数 | 月度跑赢比例 | 月度超额合计 | 主要拖累月份 |
| --- | ---: | ---: | ---: | --- |
| `monthly_investable` | `14` | `21.4%` | `-40.05pct` | `2025-08`、`2026-04`、`2026-01`、`2026-02`、`2025-07` |
| `monthly_investable_market_relative` | `14` | `28.6%` | `-59.54pct` | `2025-03`、`2025-07`、`2025-12`、`2025-08`、`2026-01` |

这两个结果都不是“单月踩雷”。`monthly_investable` 在 `2025-07`、`2025-08`、`2026-01`、`2026-02` 等上涨或偏上涨月份持续捕获不足；`monthly_investable_market_relative` 的失败更分散，且 Top-K 边界显示下一档 Top-K 的持有期收益反而更高。

## 结论

两个标签模式都出现了同一个结论：Rank IC、旧 light proxy 和 full-like proxy 均为正，但 daily backtest-like proxy 显著为负。`monthly_investable_market_relative` 的旧 proxy 最好看，却在 daily proxy 和 Top-K 边界诊断上最差，说明旧 proxy 仍会高估这类月频标签候选。

按当前 daily-proxy-first stop rule，H1 的这两个最小 G0 候选均为 `reject`：不进入 G1，不补正式 full backtest，也不进入 promotion 候选。后续不再为了旧 proxy 单独做断层对齐；若继续 H1，只能作为新的 daily proxy 候选进入 leaderboard，并且必须先让 `daily_bt_like_proxy_annualized_excess_vs_market` 至少回到非负，最好达到 `+3%` 的正式回测触发线。
