# P1 Rank-Fusion Long-Horizon G0 Smoke

- 日期：`2026-04-27`
- 结果类型：`daily_bt_like_proxy`
- 研究主题：`p1_tree_groups`
- config id：`rb_m_top20_lh_5-10-20_px_5_val20_lw_10-20-70_lbl_rank_fusion_obj_regression`
- output stem：`p1_rank_fusion_long_horizon_g0_smoke_rb_m_top20_lh_5-10-20_px_5_val20_lw_10-20-70_lbl_rank_fusion_obj_regression_20260427_120451`

## 只改了什么

新增一个从边界和换手问题出发的最小 G0 候选：`rank_fusion + regression + G0`，但把标签权重从默认等权改成 `5d/10d/20d = 0.1/0.2/0.7`。

这个候选不新增特征、不打开 G1/G2/G3/G4，也不改执行口径。它只测试一个问题：如果训练目标更偏向较慢的 20 日截面收益，能否降低 Top-K 边界噪声和换手，并让 daily proxy 转正。

本轮同时修正了 P1 结果身份：非默认 `label_weights` 会写入 `research_config_id` 的 `lw_...` 片段，避免不同权重共用同一个 config id。

## 命令

```bash
python scripts/run_p1_tree_groups.py \
  --config config.yaml.backtest \
  --groups G0 \
  --history-start 2020-01-01 \
  --sample-start 2021-01-01 \
  --label-horizons 5,10,20 \
  --label-weights 0.1,0.2,0.7 \
  --label-mode rank_fusion \
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
  --out-tag p1_rank_fusion_long_horizon_g0_smoke
```

## 结论

本候选 **daily reject**，不进入正式 full backtest。

关键指标：

| 字段 | 值 |
| --- | ---: |
| `daily_proxy_first_status` | `reject` |
| `daily_bt_like_proxy_annualized_excess_vs_market` | `-29.52%` |
| `daily_bt_like_proxy_strategy_annualized_return` | `-2.81%` |
| `daily_bt_like_proxy_benchmark_annualized_return` | `+36.62%` |
| `daily_bt_like_proxy_strategy_max_drawdown` | `15.03%` |
| `daily_bt_like_proxy_period_beat_rate` | `43.20%` |
| `daily_bt_like_proxy_avg_turnover_half_l1` | `77.69%` |
| `legacy_unconstrained_proxy_annualized_excess_vs_market` | `+16.83%` |
| `legacy_full_like_proxy_annualized_excess_vs_market` | `+16.83%` |
| `proxy_gap_daily_bt_like_minus_unconstrained` | `-46.34pct` |
| `val_rank_ic` | `0.1494` |
| `tree_score_auto_flipped` | `False` |

旧 proxy 继续显著为正，但 daily backtest-like proxy 仍为负，说明“慢标签”没有解决旧 proxy 高估的问题。

## 状态切片

上涨参与仍然没有修复：

| 状态 | 月数 | 中位超额 | 跑赢率 |
| --- | ---: | ---: | ---: |
| `strong_up` | `6` | `-3.78%` | `16.67%` |
| `strong_down` | `2` | `+0.00%` | `50.00%` |
| `high_vol` | `4` | `-0.03%` | `50.00%` |
| `wide_breadth` | `11` | `-4.44%` | `27.27%` |

相对前几轮，上涨月跑赢率略高于 0，但中位超额仍明显为负，不满足 P1-B 触发条件。

## Top-K 边界

边界和换手没有改善：

| 字段 | 值 |
| --- | ---: |
| `topk_boundary_topk_minus_next_mean_return` | `-1.02%` |
| `topk_boundary_switch_in_minus_out_mean_return` | `-0.95%` |
| `topk_boundary_avg_turnover_half_l1` | `77.92%` |

Top-K 平均收益低于下一桶，换入仍低于换出，换手也没有降到足以解释为稳定化改善。

## 是否改变主计划

不改变主计划。

1. `rank_fusion long-horizon weighted + regression + G0` 归档为失败样本。
2. 不补正式 full backtest。
3. 不扩到 `G1/G2/G3/G4`。
4. 旧 proxy 继续只作为 legacy diagnostic。
5. 下一轮 P1 不应继续围绕简单标签加权做小网格；如果继续 P1-A，应换成更明确的机制，优先考虑直接约束换入质量或降低边界反向，而不是只调 horizon 权重。

## 产物

- `data/results/p1_rank_fusion_long_horizon_g0_smoke_rb_m_top20_lh_5-10-20_px_5_val20_lw_10-20-70_lbl_rank_fusion_obj_regression_20260427_120451_summary.csv`
- `data/results/p1_rank_fusion_long_horizon_g0_smoke_rb_m_top20_lh_5-10-20_px_5_val20_lw_10-20-70_lbl_rank_fusion_obj_regression_20260427_120451_daily_proxy_leaderboard.csv`
- `data/results/p1_rank_fusion_long_horizon_g0_smoke_rb_m_top20_lh_5-10-20_px_5_val20_lw_10-20-70_lbl_rank_fusion_obj_regression_20260427_120451_daily_proxy_monthly_state.csv`
- `data/results/p1_rank_fusion_long_horizon_g0_smoke_rb_m_top20_lh_5-10-20_px_5_val20_lw_10-20-70_lbl_rank_fusion_obj_regression_20260427_120451_daily_proxy_state_summary.csv`
- `data/results/p1_rank_fusion_long_horizon_g0_smoke_rb_m_top20_lh_5-10-20_px_5_val20_lw_10-20-70_lbl_rank_fusion_obj_regression_20260427_120451_topk_boundary.csv`
- `data/results/p1_rank_fusion_long_horizon_g0_smoke_rb_m_top20_lh_5-10-20_px_5_val20_lw_10-20-70_lbl_rank_fusion_obj_regression_20260427_120451_bundle_manifest.csv`
- `data/results/p1_rank_fusion_long_horizon_g0_smoke_rb_m_top20_lh_5-10-20_px_5_val20_lw_10-20-70_lbl_rank_fusion_obj_regression_20260427_120451.json`

