# Newdata Quality Checks

- 生成时间：`2026-04-27T06:42:04.106623+00:00`
- 结果类型：`newdata_quality_summary`
- 研究主题：`newdata_quality_checks`
- 研究配置：`families_fund_flow-shareholder_flow_a_share_fund_flow_holder_a_share_shareholder_daily_a_share_daily_lag_30_width_100`
- 输出 stem：`newdata_quality_current_families_fund_flow-shareholder_flow_a_share_fund_flow_holder_a_share_shareholder_daily_a_share_daily_lag_30_width_100`
- 数据家族：`fund_flow, shareholder`

## Summary

| result_type | research_topic | research_config_id | output_stem | family | ok | table_exists | total_rows | distinct_symbols | max_date | daily_max_date | duplicate_pk_rows | coverage_ratio_vs_daily | rows_without_daily_match | rows_after_daily_max_date | rows_without_daily_match_within_daily_span | rows_without_daily_match_absent_symbols | rows_without_daily_match_known_symbols | absent_symbol_count | all_zero_rows | notes | notice_date_coverage_ratio | fallback_lag_usage_ratio | median_symbols_per_end_date | effective_factor_dates_ge_min_width |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| newdata_quality_summary | newdata_quality_checks | families_fund_flow-shareholder_flow_a_share_fund_flow_holder_a_share_shareholder_daily_a_share_daily_lag_30_width_100 | newdata_quality_current_families_fund_flow-shareholder_flow_a_share_fund_flow_holder_a_share_shareholder_daily_a_share_daily_lag_30_width_100 | fund_flow | False | True | 581433 | 4866 | 2026-04-24 | 2026-04-13 | 0 | 0.7502553618599508 | 73892.0 | 43734.0 | 30158.0 | 30151.0 | 7.0 | 281.0 | 0.0 | 存在日线已覆盖标的的资金流 trade_date 找不到对应日线行，可能有日期错位。 \| 部分资金流标的完全不在日线表中，通常是数据源市场范围宽于当前日线 universe。 \| 资金流日期晚于日线表最新日期；请先补齐日线表再判断剩余对齐质量。 |  |  |  |  |
| newdata_quality_summary | newdata_quality_checks | families_fund_flow-shareholder_flow_a_share_fund_flow_holder_a_share_shareholder_daily_a_share_daily_lag_30_width_100 | newdata_quality_current_families_fund_flow-shareholder_flow_a_share_fund_flow_holder_a_share_shareholder_daily_a_share_daily_lag_30_width_100 | shareholder | False | True | 14075 | 5354 | 2025-12-31 |  | 0 |  |  |  |  |  |  |  |  | 存在 notice_date 早于 end_date 的记录。 | 1.0 | 0.0 | 5290.0 | 32.0 |

## Research Budget Decision

- `fund_flow`：暂停新增研究预算，仅保留数据链路维护。原因：存在日线已覆盖标的的资金流 trade_date 找不到对应日线行，可能有日期错位。 | 部分资金流标的完全不在日线表中，通常是数据源市场范围宽于当前日线 universe。 | 资金流日期晚于日线表最新日期；请先补齐日线表再判断剩余对齐质量。
- `shareholder`：暂停新增研究预算，仅保留数据链路维护。原因：存在 notice_date 早于 end_date 的记录。

## Alignment Breakdown

- `fund_flow`：资金流最新 `2026-04-24`，日线最新 `2026-04-13`；未匹配 `73892` 行，其中 `43734` 行晚于日线最新日期，`30158` 行位于日线覆盖区间内。覆盖区间内未匹配里，`30151` 行来自日线表完全没有的 `281` 个标的，`7` 行是日线已覆盖标的但具体日期未匹配。
- `shareholder`：当前主要断点仍是 `存在 notice_date 早于 end_date 的记录。`。

## 说明

- 该产物用于在 scout / tree 实验之前，先确认新数据链路是否满足基本可解释性。
- `fund_flow` 重点检查覆盖率、主键重复、时间戳对齐、关键列空值率与可疑全零行。
- `shareholder` 重点检查 `notice_date` 覆盖率、fallback lag 使用比例、公告滞后异常与截面宽度。

## 本轮产物

- `data/results/newdata_quality_current_families_fund_flow-shareholder_flow_a_share_fund_flow_holder_a_share_shareholder_daily_a_share_daily_lag_30_width_100_summary.csv`
- `data/results/newdata_quality_current_families_fund_flow-shareholder_flow_a_share_fund_flow_holder_a_share_shareholder_daily_a_share_daily_lag_30_width_100.json`
- `data/results/newdata_quality_current_families_fund_flow-shareholder_flow_a_share_fund_flow_holder_a_share_shareholder_daily_a_share_daily_lag_30_width_100_manifest.json`
- `docs/newdata_quality_current_families_fund_flow-shareholder_flow_a_share_fund_flow_holder_a_share_shareholder_daily_a_share_daily_lag_30_width_100.md`
