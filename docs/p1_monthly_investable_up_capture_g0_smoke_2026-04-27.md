# P1 Monthly Investable Up-Capture G0 Smoke

- 日期：`2026-04-27`
- 结果类型：`daily_bt_like_proxy`
- 研究主题：`p1_tree_groups`
- config id：`rb_m_top20_lh_5-10-20_px_5_val20_lbl_monthly_investable_up_capture_market_relative_obj_regression`
- output stem：`p1_monthly_investable_up_capture_g0_smoke_rb_m_top20_lh_5-10-20_px_5_val20_lbl_monthly_investable_up_capture_market_relative_obj_regression_20260427_114335`

## 只改了什么

新增一个最小组合标签：`monthly_investable_up_capture_market_relative`。

它把前两轮失败线索合在一起，但仍只跑 `G0 + regression`：

1. 标签收益窗口使用月频正式执行一致的 `tplus1_open` 持有期收益。
2. 标签先扣除同一调仓日的截面等权收益。
3. 当该调仓期截面等权收益为正时，对相对收益使用 `label_up_capture_multiplier=2.0`。

这个候选只检验一个问题：月频可投资标签对齐后，若额外放大上涨期截面相对收益，能否修复 strong up 参与不足。

## 命令

```bash
python scripts/run_p1_tree_groups.py \
  --config config.yaml.backtest \
  --groups G0 \
  --history-start 2020-01-01 \
  --sample-start 2021-01-01 \
  --label-horizons 5,10,20 \
  --label-mode monthly_investable_up_capture_market_relative \
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
  --backtest-max-turnover 1.0 \
  --backtest-portfolio-method equal_weight \
  --out-tag p1_monthly_investable_up_capture_g0_smoke
```

## 结论

本候选 **daily reject**，不进入正式 full backtest。

关键指标：

| 字段 | 值 |
| --- | ---: |
| `daily_proxy_first_status` | `reject` |
| `daily_bt_like_proxy_annualized_excess_vs_market` | `-47.48%` |
| `daily_bt_like_proxy_strategy_annualized_return` | `-29.09%` |
| `daily_bt_like_proxy_benchmark_annualized_return` | `+33.83%` |
| `daily_bt_like_proxy_strategy_max_drawdown` | `36.64%` |
| `daily_bt_like_proxy_period_beat_rate` | `40.96%` |
| `daily_bt_like_proxy_avg_turnover_half_l1` | `95.77%` |
| `legacy_unconstrained_proxy_annualized_excess_vs_market` | `+18.72%` |
| `legacy_full_like_proxy_annualized_excess_vs_market` | `+18.72%` |
| `proxy_gap_daily_bt_like_minus_unconstrained` | `-66.20pct` |
| `val_rank_ic` | `0.0982` |
| `tree_score_auto_flipped` | `False` |

旧 proxy 仍然为正，但 daily backtest-like proxy 明显为负，继续支持当前纪律：旧 proxy 只能解释，不能触发正式回测。

## 状态切片

上涨参与没有修复，反而比上一轮 `up_capture_market_relative` 更差：

| 状态 | 月数 | 中位超额 | 跑赢率 |
| --- | ---: | ---: | ---: |
| `strong_up` | `6` | `-5.18%` | `0.00%` |
| `strong_down` | `2` | `-1.32%` | `50.00%` |
| `high_vol` | `4` | `-4.38%` | `25.00%` |
| `wide_breadth` | `11` | `-4.88%` | `0.00%` |

这说明“月频可投资对齐 + 上涨期标签放大”不是当前 G0 的有效修复。

## Top-K 边界

Top-K 边界也偏负：

| 字段 | 值 |
| --- | ---: |
| `topk_boundary_topk_minus_next_mean_return` | `-1.74%` |
| `topk_boundary_switch_in_minus_out_mean_return` | `-0.33%` |
| `topk_boundary_avg_turnover_half_l1` | `95.42%` |

Top-K 平均不如下一桶，换入相对换出也略负；这不是值得扩 Top-K、缓冲或换手参数的候选。

## 是否改变主计划

不改变主计划。

1. `monthly_investable_up_capture_market_relative + regression + G0` 归档为失败样本。
2. 不补正式 full backtest。
3. 不扩到 `G1/G2/G3/G4`。
4. P1 继续 daily-proxy-first，每轮只允许少量 G0 机制候选。
5. 下一轮应优先从边界反向和高换手问题出发，而不是继续叠加上涨标签权重。

## 产物

- `data/results/p1_monthly_investable_up_capture_g0_smoke_rb_m_top20_lh_5-10-20_px_5_val20_lbl_monthly_investable_up_capture_market_relative_obj_regression_20260427_114335_summary.csv`
- `data/results/p1_monthly_investable_up_capture_g0_smoke_rb_m_top20_lh_5-10-20_px_5_val20_lbl_monthly_investable_up_capture_market_relative_obj_regression_20260427_114335_daily_proxy_leaderboard.csv`
- `data/results/p1_monthly_investable_up_capture_g0_smoke_rb_m_top20_lh_5-10-20_px_5_val20_lbl_monthly_investable_up_capture_market_relative_obj_regression_20260427_114335_daily_proxy_monthly_state.csv`
- `data/results/p1_monthly_investable_up_capture_g0_smoke_rb_m_top20_lh_5-10-20_px_5_val20_lbl_monthly_investable_up_capture_market_relative_obj_regression_20260427_114335_daily_proxy_state_summary.csv`
- `data/results/p1_monthly_investable_up_capture_g0_smoke_rb_m_top20_lh_5-10-20_px_5_val20_lbl_monthly_investable_up_capture_market_relative_obj_regression_20260427_114335_topk_boundary.csv`
- `data/results/p1_monthly_investable_up_capture_g0_smoke_rb_m_top20_lh_5-10-20_px_5_val20_lbl_monthly_investable_up_capture_market_relative_obj_regression_20260427_114335_bundle_manifest.csv`
- `data/results/p1_monthly_investable_up_capture_g0_smoke_rb_m_top20_lh_5-10-20_px_5_val20_lbl_monthly_investable_up_capture_market_relative_obj_regression_20260427_114335.json`
