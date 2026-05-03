# Monthly Selection Dataset

- 生成时间：`2026-05-03T16:17:00.749994+00:00`
- 结果类型：`monthly_selection_dataset`
- 研究主题：`monthly_selection_dataset`
- 研究配置：`rb_m_exec_tplus1_open_sell_mend_open_label_o2o_start_2021_01_01_end_latest_hist_120_amt20m_50_lmmax_3_daily_a_share_daily`
- 输出 stem：`monthly_selection_dataset_2026_05_04`
- 配置来源：`config.yaml.backtest`

## Quality

| result_type | research_topic | research_config_id | output_stem | config_source | dataset_version | rebalance_rule | execution_mode | benchmark_return_mode | sell_timing | candidate_pool_versions | industry_map_source_status | rows | symbols | signal_months | min_signal_date | max_signal_date | label_valid_rows |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| monthly_selection_dataset_quality | monthly_selection_dataset | rb_m_exec_tplus1_open_sell_mend_open_label_o2o_start_2021_01_01_end_latest_hist_120_amt20m_50_lmmax_3_daily_a_share_daily | monthly_selection_dataset_2026_05_04 | config.yaml.backtest | monthly_selection_features_v1 | M | tplus1_open | market_ew_open_to_open | holding_month_last_trading_day_open | U0_all_tradable,U1_liquid_tradable,U2_risk_sane | real_industry_map | 922050 | 5197 | 64 | 2021-01-29 | 2026-04-30 | 302207 |

## Candidate Pool Width

| candidate_pool_version | months | median_width | min_width | max_width | median_pass_ratio |
| --- | --- | --- | --- | --- | --- |
| U0_all_tradable | 64 | 4919.5 | 0 | 5172 | 0.998411 |
| U1_liquid_tradable | 64 | 3111.5 | 0 | 4843 | 0.649114 |
| U2_risk_sane | 64 | 2797 | 0 | 4209 | 0.589798 |

## Feature Coverage

| feature | rows | non_null | coverage_ratio | candidate_pool_pass_coverage_ratio |
| --- | --- | --- | --- | --- |
| feature_ret_5d | 307350 | 307017 | 0.998917 | 1 |
| feature_ret_20d | 307350 | 306083 | 0.995878 | 1 |
| feature_ret_60d | 307350 | 303506 | 0.987493 | 1 |
| feature_realized_vol_20d | 307350 | 306673 | 0.997797 | 1 |
| feature_amount_20d_log | 307350 | 306745 | 0.998032 | 1 |
| feature_turnover_20d | 307350 | 306745 | 0.998032 | 1 |
| feature_price_position_250d | 307350 | 303573 | 0.987711 | 1 |
| feature_limit_move_hits_20d | 307350 | 307350 | 1 | 1 |

## Label Distribution

| signal_date | candidate_pool_version | n | mean | median | p10 | p90 |
| --- | --- | --- | --- | --- | --- | --- |
| 2025-05-30 00:00:00 | U1_liquid_tradable | 4142 | 0.0487444 | 0.030545 | -0.0473166 | 0.160141 |
| 2025-06-30 00:00:00 | U1_liquid_tradable | 4241 | 0.0558415 | 0.033237 | -0.0555556 | 0.177789 |
| 2025-07-31 00:00:00 | U1_liquid_tradable | 4579 | 0.0919673 | 0.057072 | -0.033748 | 0.256448 |
| 2025-08-29 00:00:00 | U1_liquid_tradable | 4843 | 0.00579669 | -0.0240964 | -0.107709 | 0.150665 |
| 2025-09-30 00:00:00 | U1_liquid_tradable | 4572 | -0.00130256 | -0.0101752 | -0.10802 | 0.104938 |
| 2025-10-31 00:00:00 | U1_liquid_tradable | 4486 | -0.0197881 | -0.0380669 | -0.118555 | 0.0865751 |
| 2025-11-28 00:00:00 | U1_liquid_tradable | 4541 | 0.0211524 | -0.0033557 | -0.096391 | 0.151869 |
| 2025-12-31 00:00:00 | U1_liquid_tradable | 4235 | 0.0706312 | 0.0410959 | -0.0666982 | 0.236878 |
| 2026-01-30 00:00:00 | U1_liquid_tradable | 4806 | 0.0367706 | 0.0205575 | -0.0474753 | 0.136634 |
| 2026-02-27 00:00:00 | U1_liquid_tradable | 4678 | -0.0719742 | -0.0922709 | -0.186731 | 0.0645589 |
| 2026-03-31 00:00:00 | U1_liquid_tradable | 4500 | 0.046386 | 0.0149158 | -0.0888937 | 0.224083 |
| 2026-04-30 00:00:00 | U1_liquid_tradable | 0 |  |  |  |  |

## Reject Reasons

| candidate_pool_version | reject_reason | count |
| --- | --- | --- |
| U0_all_tradable | no_next_trade_date | 5143 |
| U0_all_tradable | open_limit_up_unbuyable | 2067 |
| U0_all_tradable | missing_next_day_bar | 155 |
| U0_all_tradable | suspended_like_next_open | 1 |
| U1_liquid_tradable | low_liquidity | 86398 |
| U1_liquid_tradable | insufficient_history | 7936 |
| U1_liquid_tradable | no_next_trade_date | 5143 |
| U1_liquid_tradable | open_limit_up_unbuyable | 2067 |
| U1_liquid_tradable | missing_next_day_bar | 155 |
| U1_liquid_tradable | suspended_like_next_open | 1 |
| U2_risk_sane | low_liquidity | 86398 |
| U2_risk_sane | absolute_high | 10952 |
| U2_risk_sane | insufficient_history | 7936 |
| U2_risk_sane | limit_move_path | 6363 |
| U2_risk_sane | extreme_turnover | 6166 |
| U2_risk_sane | extreme_volatility | 6166 |
| U2_risk_sane | no_next_trade_date | 5143 |
| U2_risk_sane | open_limit_up_unbuyable | 2067 |
| U2_risk_sane | missing_next_day_bar | 155 |
| U2_risk_sane | suspended_like_next_open | 1 |

## 口径

- 信号日：每月最后一个交易日。
- 执行口径：`tplus1_open`。
- 标签：从信号日后第一个 open-to-open 日收益开始，复利持有到下一次月末信号日开盘；不再包含下一信号日到下一月首个交易日开盘的隔夜区间。
- 候选池：`U0/U1/U2` 只做可交易、数据有效和极端风险过滤，不做 alpha 判断。
- 特征处理：按 `signal_date` 截面 winsorize 1%/99% 后 z-score，并保留缺失标记。

## 本轮产物

- `data/cache/monthly_selection_features.parquet`
- `data/results/monthly_selection_dataset_2026_05_04_quality.csv`
- `data/results/monthly_selection_dataset_2026_05_04_candidate_pool_width.csv`
- `data/results/monthly_selection_dataset_2026_05_04_candidate_pool_reject_reason.csv`
- `data/results/monthly_selection_dataset_2026_05_04_feature_coverage.csv`
- `data/results/monthly_selection_dataset_2026_05_04_label_distribution.csv`
- `docs/monthly_selection_dataset_2026_05_04.md`
- `data/results/monthly_selection_dataset_2026_05_04_manifest.json`
