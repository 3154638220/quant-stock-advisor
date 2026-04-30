# P1 Up-Capture Market-Relative G0 Smoke

- 日期：`2026-04-27`
- 结果类型：`daily_bt_like_proxy`
- 研究主题：`p1_tree_groups`
- config id：`rb_m_top20_lh_5-10-20_px_5_val20_lbl_up_capture_market_relative_obj_regression`
- output stem：`p1_up_capture_market_relative_g0_smoke_rb_m_top20_lh_5-10-20_px_5_val20_lbl_up_capture_market_relative_obj_regression_20260427_111458`

## 只改了什么

新增一个最小上涨参与标签：`up_capture_market_relative`。

它仍然只使用 G0 baseline technical 特征，不新增 `weekly_kdj`、fund flow 或 shareholder 特征；训练目标为 regression。标签先扣除同日截面等权前向收益，再对市场前向收益为正的截面使用 `label_up_capture_multiplier=2.0` 放大标签幅度，用来测试“上涨参与不足”是否能通过更重视上涨环境来改善。

## 命令

```bash
python scripts/run_p1_tree_groups.py \
  --config config.yaml.backtest \
  --groups G0 \
  --history-start 2020-01-01 \
  --sample-start 2021-01-01 \
  --label-horizons 5,10,20 \
  --label-mode up_capture_market_relative \
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
  --out-tag p1_up_capture_market_relative_g0_smoke
```

## 结论

本候选 **daily reject**，不进入正式 full backtest。

关键指标：

| 字段 | 值 |
| --- | ---: |
| `daily_proxy_first_status` | `reject` |
| `daily_bt_like_proxy_annualized_excess_vs_market` | `-39.21%` |
| `daily_bt_like_proxy_strategy_annualized_return` | `-16.23%` |
| `daily_bt_like_proxy_benchmark_annualized_return` | `+36.62%` |
| `daily_bt_like_proxy_strategy_max_drawdown` | `20.06%` |
| `daily_bt_like_proxy_period_beat_rate` | `37.60%` |
| `daily_bt_like_proxy_avg_turnover_half_l1` | `86.15%` |
| `legacy_unconstrained_proxy_annualized_excess_vs_market` | `+19.67%` |
| `legacy_full_like_proxy_annualized_excess_vs_market` | `+19.67%` |
| `proxy_gap_daily_bt_like_minus_unconstrained` | `-58.88pct` |
| `val_rank_ic` | `0.1050` |
| `tree_score_auto_flipped` | `False` |

旧 proxy 再次给出正超额，但 daily backtest-like proxy 明显为负，符合当前 daily-proxy-first 纪律：旧 proxy 只能作为 legacy diagnostic，不能触发正式回测。

## 状态切片

daily proxy 的状态切片没有显示上涨参与修复：

| 状态 | 月数 | 中位超额 | 跑赢率 |
| --- | ---: | ---: | ---: |
| `strong_up` | `6` | `-3.56%` | `0.00%` |
| `strong_down` | `2` | `-3.54%` | `50.00%` |
| `high_vol` | `4` | `-2.57%` | `50.00%` |
| `wide_breadth` | `11` | `-3.66%` | `18.18%` |

这说明“上涨截面加权”没有修复原始问题，至少当前这个最小标签形式不能进入下一层。

## Top-K 边界

Top-K 边界本身不是主要矛盾：

| 字段 | 值 |
| --- | ---: |
| `topk_boundary_topk_minus_next_mean_return` | `+0.74%` |
| `topk_boundary_switch_in_minus_out_mean_return` | `-0.36%` |
| `topk_boundary_avg_turnover_half_l1` | `89.17%` |

Top-K 对 next bucket 的平均收益差为正，但换入相对换出略负，且 daily proxy 已经 reject，因此不应从这个候选继续扩 Top-K 或缓冲参数。

## 是否改变主计划

不改变主计划。

1. P1 继续使用 daily-proxy-first。
2. `up_capture_market_relative + regression + G0` 归档为失败样本。
3. 不补正式 full backtest。
4. 不扩到 `G1/G2/G3/G4`。
5. 下一轮仍应只选一个新的 G0 机制候选，优先来自 daily proxy 状态切片、Top-K 边界或换手诊断，而不是扩大网格。

## 产物

- `data/results/p1_up_capture_market_relative_g0_smoke_rb_m_top20_lh_5-10-20_px_5_val20_lbl_up_capture_market_relative_obj_regression_20260427_111458_summary.csv`
- `data/results/p1_up_capture_market_relative_g0_smoke_rb_m_top20_lh_5-10-20_px_5_val20_lbl_up_capture_market_relative_obj_regression_20260427_111458_daily_proxy_leaderboard.csv`
- `data/results/p1_up_capture_market_relative_g0_smoke_rb_m_top20_lh_5-10-20_px_5_val20_lbl_up_capture_market_relative_obj_regression_20260427_111458_daily_proxy_monthly_state.csv`
- `data/results/p1_up_capture_market_relative_g0_smoke_rb_m_top20_lh_5-10-20_px_5_val20_lbl_up_capture_market_relative_obj_regression_20260427_111458_daily_proxy_state_summary.csv`
- `data/results/p1_up_capture_market_relative_g0_smoke_rb_m_top20_lh_5-10-20_px_5_val20_lbl_up_capture_market_relative_obj_regression_20260427_111458_topk_boundary.csv`
- `data/results/p1_up_capture_market_relative_g0_smoke_rb_m_top20_lh_5-10-20_px_5_val20_lbl_up_capture_market_relative_obj_regression_20260427_111458_bundle_manifest.csv`
- `data/results/p1_up_capture_market_relative_g0_smoke_rb_m_top20_lh_5-10-20_px_5_val20_lbl_up_capture_market_relative_obj_regression_20260427_111458.json`
