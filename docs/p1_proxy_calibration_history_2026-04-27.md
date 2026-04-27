# P1 Proxy Calibration History

- 生成时间：`2026-04-27T04:58:56.241926+00:00`
- 结果类型：`light_strategy_proxy_calibration`
- 样本数：`5`

## Summary

| sample | group | label_mode | xgboost_objective | unconstrained_proxy_excess | full_like_proxy_excess | full_backtest_excess | full_like_minus_unconstrained | full_like_minus_full_backtest | n_periods |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| market_relative_rank_G0 | G0 | market_relative | rank | 0.127077 | 0.131811 | -0.167354 | 0.00473401 | 0.299165 | 64 |
| market_relative_rank_G1 | G1 | market_relative | rank | 0.050031 | 0.0358697 | -0.339726 | -0.0141612 | 0.375595 | 64 |
| rank_fusion_regression_G0 | G0 | rank_fusion | regression | 0.0725867 | 0.0794697 | -0.208735 | 0.00688305 | 0.288205 | 64 |
| unknown_label_unknown_obj_G0 | G0 |  |  | 0.102691 | 0.0869648 | -0.123466 | -0.0157259 | 0.210431 | 64 |
| unknown_label_unknown_obj_G1 | G1 |  |  | 0.0948394 | 0.10237 | -0.153384 | 0.00753049 | 0.255754 | 64 |

## 判读

本轮复用历史 full backtest 的 tree bundle 与 prepared factors，不重训模型。`full_like_proxy_excess` 比旧版 `unconstrained_proxy_excess` 更贴近正式口径，因为它加入了月频 Top-K 的持仓延续和 `max_turnover` 限制。

结论很明确：只加入 turnover cap 仍然不够。`h=5` 的 full-like proxy 对 5 个历史失败/对照样本全部仍为正，且相对正式 full backtest 平均高估 `+28.58pct`。

补跑 `proxy_horizon=20` 后，背离没有修复，反而更乐观：`h=20` 的 full-like proxy 相对正式 full backtest 平均高估 `+38.10pct`。这基本排除“只是 5d horizon 太短”的解释。

下一步不应继续微调 `proxy_horizon` 或 turnover cap，而应把 proxy 直接对齐到 full backtest 的收益生成方式：用日频 open-to-open 资产收益、月频权重前向持有、成本和 benchmark_ew 同口径，形成一个轻量但真正 full-backtest-like 的准入层。

## H20 对照

| sample | h20 full-like proxy excess | full backtest excess | h20 gap |
| --- | ---: | ---: | ---: |
| market_relative_rank_G0 | +16.71% | -16.74% | +33.45% |
| market_relative_rank_G1 | +19.98% | -33.97% | +53.96% |
| rank_fusion_regression_G0 | +11.42% | -20.87% | +32.29% |
| unknown_label_unknown_obj_G0 | +16.43% | -12.35% | +28.78% |
| unknown_label_unknown_obj_G1 | +26.71% | -15.34% | +42.05% |

## 产物

- `data/results/p1_proxy_calibration_history_2026-04-27_summary.csv`
- `data/results/p1_proxy_calibration_history_2026-04-27_detail.csv`
- `data/results/p1_proxy_calibration_history_2026-04-27.json`
- `docs/p1_proxy_calibration_history_2026-04-27.md`
- `data/results/p1_proxy_calibration_history_h20_2026-04-27_summary.csv`
- `data/results/p1_proxy_calibration_history_h20_2026-04-27_detail.csv`
- `data/results/p1_proxy_calibration_history_h20_2026-04-27.json`
- `docs/p1_proxy_calibration_history_h20_2026-04-27.md`
