# Newdata Quality Checks

- 生成时间：`2026-04-29T02:16:08.857693+00:00`
- 结果类型：`newdata_quality_summary`
- 研究主题：`newdata_quality_checks`
- 研究配置：`families_fund_flow-shareholder_flow_a_share_fund_flow_holder_a_share_shareholder_daily_a_share_daily_lag_30_width_100`
- 输出 stem：`newdata_quality_families_fund_flow-shareholder_flow_a_share_fund_flow_holder_a_share_shareholder_daily_a_share_daily_lag_30_width_100`
- 配置来源：`/mnt/ssd/lh/config.yaml.backtest`
- 数据家族：`fund_flow, shareholder`

## Summary

| result_type | research_topic | research_config_id | output_stem | config_source | family | ok | table_exists | total_rows | distinct_symbols | max_date | daily_max_date | duplicate_pk_rows | coverage_ratio_vs_daily | rows_without_daily_match | rows_after_daily_max_date | rows_without_daily_match_within_daily_span | rows_without_daily_match_absent_symbols | rows_without_daily_match_known_symbols | absent_symbol_count | all_zero_rows | notes | notice_date_coverage_ratio | fallback_lag_usage_ratio | negative_notice_lag_rows | median_notice_lag_days | p90_notice_lag_days | median_symbols_per_end_date | effective_factor_dates_ge_min_width |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| newdata_quality_summary | newdata_quality_checks | families_fund_flow-shareholder_flow_a_share_fund_flow_holder_a_share_shareholder_daily_a_share_daily_lag_30_width_100 | newdata_quality_families_fund_flow-shareholder_flow_a_share_fund_flow_holder_a_share_shareholder_daily_a_share_daily_lag_30_width_100 | /mnt/ssd/lh/config.yaml.backtest | fund_flow | True | True | 581433 | 4866 | 2026-04-24 | 2026-04-24 | 0 | 0.7732500458816864 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |  |  |  |  |  |  |  |  |
| newdata_quality_summary | newdata_quality_checks | families_fund_flow-shareholder_flow_a_share_fund_flow_holder_a_share_shareholder_daily_a_share_daily_lag_30_width_100 | newdata_quality_families_fund_flow-shareholder_flow_a_share_fund_flow_holder_a_share_shareholder_daily_a_share_daily_lag_30_width_100 | /mnt/ssd/lh/config.yaml.backtest | shareholder | True | True | 14075 | 5354 | 2025-12-31 |  | 0 |  |  |  |  |  |  |  |  | 存在 notice_date 早于 end_date 的记录，PIT 可用日已按 end_date + fallback lag 保守处理。 | 1.0 | 0.0 | 1.0 | 54.0 | 106.0 | 5290.0 | 32.0 |

## Research Budget Decision

- `fund_flow`：保留低优先级研究预算，只允许在明确机制假设下继续。原因：基础覆盖、重复、全零与日线对齐检查通过；但既有 G2/G4 回测未给出主线增量。
- `shareholder`：保留极低优先级研究预算，仅在 PIT/覆盖带来新证据时重启。原因：基础质量检查通过；但既有单因子与 G3 结论仍偏弱。

## Alignment Breakdown

- `fund_flow`：资金流最新 `2026-04-24`，日线最新 `2026-04-24`；未匹配 `0` 行，其中 `0` 行晚于日线最新日期，`0` 行位于日线覆盖区间内。覆盖区间内未匹配里，`0` 行来自日线表完全没有的 `0` 个标的，`0` 行是日线已覆盖标的但具体日期未匹配。
- `shareholder`：当前主要断点仍是 `存在 notice_date 早于 end_date 的记录，PIT 可用日已按 end_date + fallback lag 保守处理。`。

## 说明

- 该产物用于在 scout / tree 实验之前，先确认新数据链路是否满足基本可解释性。
- `fund_flow` 重点检查覆盖率、主键重复、时间戳对齐、关键列空值率与可疑全零行。
- `shareholder` 重点检查 `notice_date` 覆盖率、fallback lag 使用比例、公告滞后异常与截面宽度。

## 本轮产物

- `data/results/newdata_quality_families_fund_flow-shareholder_flow_a_share_fund_flow_holder_a_share_shareholder_daily_a_share_daily_lag_30_width_100_summary.csv`
- `data/results/newdata_quality_families_fund_flow-shareholder_flow_a_share_fund_flow_holder_a_share_shareholder_daily_a_share_daily_lag_30_width_100.json`
- `data/results/newdata_quality_families_fund_flow-shareholder_flow_a_share_fund_flow_holder_a_share_shareholder_daily_a_share_daily_lag_30_width_100_manifest.json`
- `docs/newdata_quality_families_fund_flow-shareholder_flow_a_share_fund_flow_holder_a_share_shareholder_daily_a_share_daily_lag_30_width_100.md`
