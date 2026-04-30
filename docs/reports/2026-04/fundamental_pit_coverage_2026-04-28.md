# Fundamental PIT Monthly Coverage

- 生成时间：`2026-04-28T13:53:50.069642+00:00`
- 结果类型：`fundamental_pit_coverage_summary`
- PIT 规则：对每个 `(signal_date, symbol)` 仅使用 `announcement_date <= signal_date` 的最新 `a_share_fundamental` 快照；未用 `report_period` 代替公告可用日。
- 输入月度表：`data/cache/monthly_selection_features.parquet`
- 输出明细：`data/results/fundamental_pit_coverage_2026-04-28_monthly.csv`

## Summary

| metric | value |
| --- | --- |
| fundamental_rows | 333201 |
| fundamental_symbols | 5196 |
| report_period_range | 1900-01-01 ~ 2026-04-16 |
| announcement_date_range | 1900-01-01 ~ 2026-04-17 |
| monthly_rows | 922173 |
| monthly_signal_dates | 64 |
| last_signal_date | 2026-04-13 |
| median_monthly_pit_coverage | 0.999732 |

## Latest Signal Coverage

| signal_date   | candidate_pool_version   |   symbols |   pit_fundamental_symbols |   pit_fundamental_coverage | latest_announcement_date   |
|:--------------|:-------------------------|----------:|--------------------------:|---------------------------:|:---------------------------|
| 2026-04-13    | U0_all_tradable          |      5184 |                      5183 |                   0.999807 | 2026-04-13                 |
| 2026-04-13    | U1_liquid_tradable       |      5184 |                      5183 |                   0.999807 | 2026-04-13                 |
| 2026-04-13    | U2_risk_sane             |      5184 |                      5183 |                   0.999807 | 2026-04-13                 |

## Source Distribution

| source                                |   rows |
|:--------------------------------------|-------:|
| stock_financial_analysis_indicator_em | 306312 |
| stock_value_em                        |  26097 |
| stock_financial_analysis_indicator    |    792 |

## M1 结论

- `fundamental` 已具备 PIT 可用日追溯：月度接入使用 `announcement_date`，不是 `report_period`。
- 覆盖率按月、按 `candidate_pool_version` 输出到 CSV；后续进入 M5/M4 扩展前，应按目标池读取该明细做 data gate。
- 当前摘要只证明 PIT 与覆盖可追溯，不构成基本面 alpha 有效性结论。
