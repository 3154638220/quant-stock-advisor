# Newdata Quality Checks

- 生成时间：`2026-05-06T02:03:19.262715+00:00`
- 结果类型：`newdata_quality_summary`
- 研究主题：`newdata_quality_checks`
- 研究配置：`families_margin_trading_flow_a_share_fund_flow_holder_a_share_shareholder_daily_a_share_daily_lag_30_width_100`
- 输出 stem：`margin_trading_m1_families_margin_trading_flow_a_share_fund_flow_holder_a_share_shareholder_daily_a_share_daily_lag_30_width_100`
- 配置来源：`/home/x12dpg/hjx/lh/config.yaml.backtest`
- 数据家族：`margin_trading`

## Summary

| result_type | research_topic | research_config_id | output_stem | config_source | family | ok | table_exists | total_rows | distinct_symbols | max_date | duplicate_pk_rows | coverage_ratio_vs_daily | daily_max_date | rows_without_daily_match | median_symbols_per_day | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| newdata_quality_summary | newdata_quality_checks | families_margin_trading_flow_a_share_fund_flow_holder_a_share_shareholder_daily_a_share_daily_lag_30_width_100 | margin_trading_m1_families_margin_trading_flow_a_share_fund_flow_holder_a_share_shareholder_daily_a_share_daily_lag_30_width_100 | /home/x12dpg/hjx/lh/config.yaml.backtest | margin_trading | True | True | 646724 | 3768 | 2024-09-26 | 0 | 0.9115666033733092 | 2026-04-30 | 57192 | 3615.0 |  |

## Research Budget Decision

- `margin_trading`：保留高优先级研究预算，推进 M5 增量 delta 测试。原因：融资融券历史长、覆盖广、与现有因子相关性低，增量潜力大；M1 质量诊断通过，进入 M5 增量。

## Alignment Breakdown

- `margin_trading`：最新 `2024-09-26`，覆盖 `3768` 只标，日线覆盖率 `91.2%`。

## 说明

- 该产物用于在 scout / tree 实验之前，先确认新数据链路是否满足基本可解释性。
- `fund_flow` 重点检查覆盖率、主键重复、时间戳对齐、关键列空值率与可疑全零行。
- `shareholder` 重点检查 `notice_date` 覆盖率、fallback lag 使用比例、公告滞后异常与截面宽度。

## 本轮产物

- `data/results/margin_trading_m1_families_margin_trading_flow_a_share_fund_flow_holder_a_share_shareholder_daily_a_share_daily_lag_30_width_100_summary.csv`
- `data/results/margin_trading_m1_families_margin_trading_flow_a_share_fund_flow_holder_a_share_shareholder_daily_a_share_daily_lag_30_width_100.json`
- `data/results/margin_trading_m1_families_margin_trading_flow_a_share_fund_flow_holder_a_share_shareholder_daily_a_share_daily_lag_30_width_100_manifest.json`
- `docs/margin_trading_m1_families_margin_trading_flow_a_share_fund_flow_holder_a_share_shareholder_daily_a_share_daily_lag_30_width_100.md`
