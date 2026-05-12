# W4 Event V2 Sparse Window Features

- 日期：2026-05-12
- 目标：修复 W4 Stage A 后暴露的 buyback / reduction / unlock 稀疏窗口问题
- 代码：`src/features/event_factors.py`、`src/features/registry.py`
- 验证：`pytest -q -o addopts='' tests/test_event_factors.py` → 3 passed

## 背景

W4 Stage A IC Gate 不通过。主要可用信号集中在 earnings guidance，buyback / reduction / unlock 的原始 30d 窗口过稀疏，导致模型中 feature importance 为 0 或接近 0。

## 本次新增特征

| 新特征 | 窗口 | 方向 | 目的 |
|--------|------|------|------|
| `feature_event_buyback_amount_ratio_180d` | 过去 180d | 正向 | 捕捉低频回购计划的持续影响 |
| `feature_event_buyback_recent_180d` | 过去 180d | 正向 | 将回购事件从 30d 扩展为半年内是否发生 |
| `feature_event_reduction_plan_flag_180d` | 过去 180d | 负向 | 捕捉减持计划的持续风险 |
| `feature_event_reduction_ratio_180d` | 过去 180d | 负向 | 用减持比例合计替代纯 flag |
| `feature_event_unlock_ratio_90d` | 未来 90d | 负向 | 将限售解禁压力从 30d 扩展至季度窗口 |

原 30d 特征保留不变。为避免 30d flag 因过窄窗口完全缺失，v2 在更宽事件观察集内同时输出 30d 与长窗口特征：没有 30d 事件但有 180d/90d 观察记录时，30d flag/ratio 可为 0 而不是 NaN。

## 覆盖率快查

基于 `data/cache/monthly_selection_features.parquet` + `data/market.duckdb` 直接附加事件特征后的非空覆盖率：

| 特征 | 覆盖率 |
|------|-------:|
| `feature_event_earnings_guidance_direction` | 71.01% |
| `feature_event_earnings_guidance_magnitude` | 70.35% |
| `feature_event_earnings_surprise_ttm` | 49.58% |
| `feature_event_buyback_recent_30d` | 4.25% |
| `feature_event_buyback_amount_ratio_180d` | 4.25% |
| `feature_event_buyback_recent_180d` | 4.25% |
| `feature_event_reduction_plan_flag` | 24.47% |
| `feature_event_reduction_plan_flag_180d` | 24.47% |
| `feature_event_reduction_ratio_180d` | 24.47% |
| `feature_event_unlock_ratio_30d` | 8.46% |
| `feature_event_unlock_ratio_90d` | 8.46% |

`feature_event_buyback_amount_ratio` 仍为 0% 覆盖，原因是当前 buyback 数据中 30d 金额/市值比例仍不可用或窗口过窄；保留该列用于兼容旧报告和后续数据源补全。

注意：30d flag/ratio 的覆盖率不是“30d 真实事件发生率”。v2 为了避免在 180d/90d 观察集中把“无 30d 事件”误记为缺失，会在存在长窗口观察记录时将 30d flag/ratio 补为 0。因此覆盖率表衡量的是该字段可被模型读取的非空比例，事件发生强度仍需看取值分布或 feature importance。

## 结论

W4 v2 已解决 reduction / unlock 的极端稀疏问题，并让 buyback 至少以 180d 事件 flag/ratio 进入候选特征。当前仍不直接 promote；下一步应重跑 W4 Stage A（PV-only vs PV+event_v2），若 Rank IC delta 仍接近 0，则只保留 earnings guidance 子类供 M16 后续重评。
