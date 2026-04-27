# P1 Proxy Calibration History

- 生成时间：`2026-04-27T05:06:36.721628+00:00`
- 结果类型：`light_strategy_proxy_calibration`
- 样本数：`5`

## Summary

| sample | group | label_mode | xgboost_objective | unconstrained_proxy_excess | full_like_proxy_excess | full_backtest_excess | full_like_minus_unconstrained | full_like_minus_full_backtest | n_periods |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| market_relative_rank_G0 | G0 | market_relative | rank | 0.196764 | 0.167104 | -0.167354 | -0.0296595 | 0.334459 | 63 |
| market_relative_rank_G1 | G1 | market_relative | rank | 0.260699 | 0.199827 | -0.339726 | -0.0608726 | 0.539552 | 63 |
| rank_fusion_regression_G0 | G0 | rank_fusion | regression | 0.15459 | 0.114172 | -0.208735 | -0.0404176 | 0.322907 | 63 |
| unknown_label_unknown_obj_G0 | G0 |  |  | 0.26685 | 0.164325 | -0.123466 | -0.102525 | 0.287791 | 63 |
| unknown_label_unknown_obj_G1 | G1 |  |  | 0.398535 | 0.267133 | -0.153384 | -0.131402 | 0.420517 | 63 |

## 判读

本轮复用历史 full backtest 的 tree bundle 与 prepared factors，不重训模型。`full_like_proxy_excess` 比旧版 `unconstrained_proxy_excess` 更贴近正式口径，因为它加入了月频 Top-K 的持仓延续和 `max_turnover` 限制。

若 `full_like_minus_full_backtest` 仍大幅为正，说明 light proxy 仍高估正式回测，需要继续加入更接近 full backtest 的元素，例如更长持有期、执行口径、年度/市场状态分层。

## 产物

- `data/results/p1_proxy_calibration_history_h20_2026-04-27_summary.csv`
- `data/results/p1_proxy_calibration_history_h20_2026-04-27_detail.csv`
- `data/results/p1_proxy_calibration_history_h20_2026-04-27.json`
- `docs/p1_proxy_calibration_history_h20_2026-04-27.md`
