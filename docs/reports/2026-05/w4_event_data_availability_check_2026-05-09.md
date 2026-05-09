# W4 Event Data Availability Check (2026-05-09)

## 结论

- 当前 `data/market.duckdb` 中事件表不存在：
  - `a_share_event_earnings_guidance`
  - `a_share_event_buyback`
  - `a_share_event_reduction`
  - `a_share_event_unlock`
- 因此 W4 事件因子在主数据集上的实际覆盖率为 `0.0%`（含候选池通过样本）。
- W4 第二阶段 gate 已执行（见 `docs/monthly_selection_w4_event_gate_2026_05_09_2026-05-09.md`），但因事件因子覆盖率为 0，结果未通过。

## 口径与样本

- 数据集：`data/cache/monthly_selection_features.parquet`
- 数据库：`data/market.duckdb`
- 样本范围：`2021-01-29` 到 `2026-04-30`
- 总行数：`307350`
- 候选池通过行数：`209148`

## 事件因子覆盖率

| feature | coverage_all | coverage_candidate_pool_pass |
| --- | --- | --- |
| feature_event_earnings_guidance_direction | 0.000000 | 0.000000 |
| feature_event_earnings_guidance_magnitude | 0.000000 | 0.000000 |
| feature_event_earnings_surprise_ttm | 0.000000 | 0.000000 |
| feature_event_buyback_amount_ratio | 0.000000 | 0.000000 |
| feature_event_buyback_recent_30d | 0.000000 | 0.000000 |
| feature_event_reduction_plan_flag | 0.000000 | 0.000000 |
| feature_event_unlock_ratio_30d | 0.000000 | 0.000000 |

## 后续动作

1. 先落地事件表数据抓取与入库（至少覆盖业绩预告/回购/减持/解禁四表）。
2. 事件表有数据后，重跑 W4 第二阶段 gate（coverage -> IC -> M5 delta）并对比本次失败基线。
