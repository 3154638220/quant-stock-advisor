# P1 Top Bucket Rank Fusion G0 Smoke

- 日期：`2026-04-27`
- 结果类型：`daily_bt_like_proxy`
- config id：`rb_m_top20_lh_5-10-20_px_5_val20_lbl_top_bucket_rank_fusion_obj_regression`
- output stem：`p1_top_bucket_rank_fusion_g0_smoke_rb_m_top20_lh_5-10-20_px_5_val20_lbl_top_bucket_rank_fusion_obj_regression_20260427_122226`
- 结论：`daily_proxy_first_status=reject`，不进入正式 full backtest

## 只改了什么

新增一个最小边界机制标签：`top_bucket_rank_fusion`。

它仍然只使用 `G0` baseline technical 特征，不新增 `weekly_kdj`、fund flow 或 shareholder 特征；训练目标为 regression。标签继续融合 `forward_ret_5d / 10d / 20d`，但每个截面只保留顶部 20% 和底部 20% 的 rank 信号，中间 60% 置零。目的不是继续做 horizon 权重小网格，而是直接测试“Top-K 与 next bucket 边界噪声过高”是否能通过更稀疏的 top/bottom 监督缓解。

命令：

```bash
python scripts/run_p1_tree_groups.py \
  --config config.yaml.backtest \
  --groups G0 \
  --history-start 2020-01-01 \
  --sample-start 2021-01-01 \
  --label-horizons 5,10,20 \
  --label-mode top_bucket_rank_fusion \
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
  --out-tag p1_top_bucket_rank_fusion_g0_smoke
```

## 结果

| 指标 | G0 |
| --- | ---: |
| `val_rank_ic` | `15.86%` |
| `tree_score_auto_flipped` | `False` |
| legacy light proxy 年化超额 | `+15.76%` |
| full-like proxy 年化超额 | `+15.76%` |
| daily backtest-like proxy 年化超额 | `-28.09%` |
| `daily_proxy_first_status` | `reject` |
| 是否补正式 full backtest | 否 |

daily proxy 触发硬停止：

```text
daily_bt_like_proxy_below_admission_threshold:-0.280934<0
```

## 状态切片

上涨参与没有修复：

| 状态 | 月数 | 中位超额 | 跑赢率 |
| --- | ---: | ---: | ---: |
| `strong_up` | `6` | `-2.44%` | `16.67%` |
| `strong_down` | `2` | `-0.77%` | `50.00%` |
| `neutral_return` | `5` | `-1.91%` | `40.00%` |

## Top-K 边界和换手

| 指标 | G0 |
| --- | ---: |
| `topk_boundary_topk_minus_next_mean_return` | `+0.29%` |
| `topk_boundary_switch_in_minus_out_mean_return` | `-0.94%` |
| `topk_boundary_avg_turnover_half_l1` | `78.33%` |
| daily proxy 平均 half-L1 换手 | `78.46%` |

Top-K 和 next bucket 的平均收益差转正，但非常薄；更关键的是换入减换出仍显著为负，说明稀疏 top/bottom 标签没有修复实际调仓边界。

## 对主计划的影响

1. `top_bucket_rank_fusion + regression + G0` 归档为失败样本。
2. 不进入 `G1/G2/G3/G4` 扩组。
3. 不补正式 full backtest。
4. 旧 proxy 继续仅作为 legacy diagnostic，不改变任何准入结论。
5. 下一轮 P1 不应继续做“只改标签形状”的近邻变体；更应转向解释为什么 daily proxy 的日频持有收益和月频/旧 proxy 持续断裂，例如持仓日内路径、换入时点、退出损失和风格暴露。

## 产物

- `data/results/p1_top_bucket_rank_fusion_g0_smoke_rb_m_top20_lh_5-10-20_px_5_val20_lbl_top_bucket_rank_fusion_obj_regression_20260427_122226_summary.csv`
- `data/results/p1_top_bucket_rank_fusion_g0_smoke_rb_m_top20_lh_5-10-20_px_5_val20_lbl_top_bucket_rank_fusion_obj_regression_20260427_122226_daily_proxy_leaderboard.csv`
- `data/results/p1_top_bucket_rank_fusion_g0_smoke_rb_m_top20_lh_5-10-20_px_5_val20_lbl_top_bucket_rank_fusion_obj_regression_20260427_122226_daily_proxy_monthly_state.csv`
- `data/results/p1_top_bucket_rank_fusion_g0_smoke_rb_m_top20_lh_5-10-20_px_5_val20_lbl_top_bucket_rank_fusion_obj_regression_20260427_122226_daily_proxy_state_summary.csv`
- `data/results/p1_top_bucket_rank_fusion_g0_smoke_rb_m_top20_lh_5-10-20_px_5_val20_lbl_top_bucket_rank_fusion_obj_regression_20260427_122226_topk_boundary.csv`
- `data/results/p1_top_bucket_rank_fusion_g0_smoke_rb_m_top20_lh_5-10-20_px_5_val20_lbl_top_bucket_rank_fusion_obj_regression_20260427_122226_bundle_manifest.csv`
- `data/results/p1_top_bucket_rank_fusion_g0_smoke_rb_m_top20_lh_5-10-20_px_5_val20_lbl_top_bucket_rank_fusion_obj_regression_20260427_122226.json`
