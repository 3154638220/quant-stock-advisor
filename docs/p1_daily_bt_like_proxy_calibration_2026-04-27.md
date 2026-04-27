# P1 Proxy Calibration History

- 生成时间：`2026-04-27T05:57:17.475653+00:00`
- 结果类型：`light_strategy_proxy_calibration`
- 样本数：`5`

## Summary

| sample | group | label_mode | xgboost_objective | unconstrained_proxy_excess | full_like_proxy_excess | daily_bt_like_proxy_excess | full_backtest_excess | full_like_minus_unconstrained | full_like_minus_full_backtest | daily_bt_like_minus_unconstrained | daily_bt_like_minus_full_backtest | n_periods | daily_bt_like_n_periods |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| market_relative_rank_G0 | G0 | market_relative | rank | 0.12829 | 0.140945 | -0.142921 | -0.167354 | 0.0126556 | 0.308299 | -0.27121 | 0.0244333 | 64 | 1257 |
| market_relative_rank_G1 | G1 | market_relative | rank | 0.0467671 | 0.0403535 | -0.331068 | -0.339726 | -0.00641358 | 0.380079 | -0.377836 | 0.00865704 | 64 | 1257 |
| rank_fusion_regression_G0 | G0 | rank_fusion | regression | 0.070007 | 0.0794654 | -0.195746 | -0.208735 | 0.00945838 | 0.2882 | -0.265753 | 0.0129886 | 64 | 1257 |
| unknown_label_unknown_obj_G0 | G0 |  |  | 0.0981695 | 0.0853031 | -0.122117 | -0.123466 | -0.0128664 | 0.208769 | -0.220287 | 0.00134871 | 64 | 1257 |
| unknown_label_unknown_obj_G1 | G1 |  |  | 0.088538 | 0.0983859 | -0.170844 | -0.153384 | 0.00984789 | 0.25177 | -0.259382 | -0.0174602 | 64 | 1257 |

## 判读

本轮复用历史 full backtest 的 tree bundle 与 prepared factors，不重训模型。`full_like_proxy_excess` 比旧版 `unconstrained_proxy_excess` 更贴近正式口径，因为它加入了月频 Top-K 的持仓延续和 `max_turnover` 限制。`daily_bt_like_proxy_excess` 进一步直接复用日频 open-to-open 收益、交易成本和 `market_ew` 对齐口径。

本轮结论：daily full-backtest-like proxy 显著降低了旧 proxy 对正式 full backtest 的高估。

1. `full_like_minus_full_backtest` 平均为 `+28.74%`。
2. `daily_bt_like_minus_full_backtest` 平均为 `+0.60%`，最大绝对偏差为 `2.44%`。
3. `daily_bt_like_proxy_excess` 为正的样本数：`0/5`。

因此，daily proxy 可以作为 P1 下一轮准入 gate：若该层已经为负，就不应再补正式 full backtest；若该层接近或超过 0，再进入年度/市场状态诊断和正式回测。

## 产物

- `data/results/p1_daily_bt_like_proxy_calibration_2026-04-27_summary.csv`
- `data/results/p1_daily_bt_like_proxy_calibration_2026-04-27_detail.csv`
- `data/results/p1_daily_bt_like_proxy_calibration_2026-04-27.json`
- `docs/p1_daily_bt_like_proxy_calibration_2026-04-27.md`
