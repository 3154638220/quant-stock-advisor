# Monthly Selection Dataset

- 生成时间：`2026-04-28T14:25:20.091718+00:00`
- 结果类型：`monthly_selection_dataset`
- 研究主题：`monthly_selection_dataset`
- 研究配置：`rb_m_exec_tplus1_open_label_o2o_start_2021_01_01_end_latest_hist_120_amt20m_50_lmmax_3_daily_a_share_daily`
- 输出 stem：`monthly_selection_dataset_2026-04-28`
- 配置来源：`/home/x12dpg/hjx/lh/config.yaml.backtest`

## Quality

| result_type | research_topic | research_config_id | output_stem | config_source | dataset_version | rebalance_rule | execution_mode | benchmark_return_mode | candidate_pool_versions | industry_map_source_status | rows | symbols | signal_months | min_signal_date | max_signal_date | label_valid_rows |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| monthly_selection_dataset_quality | monthly_selection_dataset | rb_m_exec_tplus1_open_label_o2o_start_2021_01_01_end_latest_hist_120_amt20m_50_lmmax_3_daily_a_share_daily | monthly_selection_dataset_2026-04-28 | /home/x12dpg/hjx/lh/config.yaml.backtest | monthly_selection_features_v1 | M | tplus1_open | market_ew_open_to_open | U0_all_tradable,U1_liquid_tradable,U2_risk_sane | real_industry_map | 922173 | 5197 | 64 | 2021-01-29 | 2026-04-13 | 297028 |

## Candidate Pool Width

| candidate_pool_version | months | median_width | min_width | max_width | median_pass_ratio |
| --- | --- | --- | --- | --- | --- |
| U0_all_tradable | 64 | 4919.5 | 0 | 5172 | 0.9984105345244694 |
| U1_liquid_tradable | 64 | 3111.5 | 0 | 4843 | 0.6491136311374425 |
| U2_risk_sane | 64 | 2797.0 | 0 | 4209 | 0.589798263699241 |

## Feature Coverage

| feature | rows | non_null | coverage_ratio | candidate_pool_pass_coverage_ratio |
| --- | --- | --- | --- | --- |
| feature_ret_5d | 307391 | 307058 | 0.9989166891678677 | 1.0 |
| feature_ret_20d | 307391 | 306119 | 0.9958619478123953 | 1.0 |
| feature_ret_60d | 307391 | 303546 | 0.9874915010524056 | 1.0 |
| feature_realized_vol_20d | 307391 | 306711 | 0.9977878337361862 | 1.0 |
| feature_amount_20d_log | 307391 | 306783 | 0.9980220631052958 | 1.0 |
| feature_turnover_20d | 307391 | 306783 | 0.9980220631052958 | 1.0 |
| feature_price_position_250d | 307391 | 303612 | 0.9877062113074228 | 1.0 |
| feature_limit_move_hits_20d | 307391 | 307391 | 1.0 | 1.0 |

## Label Distribution

| signal_date | candidate_pool_version | n | mean | median | p10 | p90 |
| --- | --- | --- | --- | --- | --- | --- |
| 2025-05-30 00:00:00 | U1_liquid_tradable | 4142 | 0.061677368768928195 | 0.04207509051905056 | -0.040424165750141614 | 0.18067135518553246 |
| 2025-06-30 00:00:00 | U1_liquid_tradable | 4241 | 0.04831629308135242 | 0.02162162162162118 | -0.06302916274694303 | 0.17857142857142794 |
| 2025-07-31 00:00:00 | U1_liquid_tradable | 4579 | 0.09221987895434425 | 0.05485232067510526 | -0.03826630090122957 | 0.2624100022435696 |
| 2025-08-29 00:00:00 | U1_liquid_tradable | 4843 | 0.013414689141958949 | -0.021243842364532473 | -0.10298619185807142 | 0.17057134935887616 |
| 2025-09-30 00:00:00 | U1_liquid_tradable | 4572 | 0.008153229591680519 | 0.0010169701730089464 | -0.09928130950047465 | 0.11371738530551397 |
| 2025-10-31 00:00:00 | U1_liquid_tradable | 4486 | -0.00516795232557371 | -0.02611342783300402 | -0.1069896181811898 | 0.10715553077157047 |
| 2025-11-28 00:00:00 | U1_liquid_tradable | 4541 | 0.025147133925535352 | -0.0035817605730817936 | -0.09598349714840404 | 0.16397228637413375 |
| 2025-12-31 00:00:00 | U1_liquid_tradable | 4235 | 0.06329359609254256 | 0.03367003367003374 | -0.07508735057581761 | 0.2309117369213879 |
| 2026-01-30 00:00:00 | U1_liquid_tradable | 4806 | 0.037570955443087334 | 0.01854497354497353 | -0.0559246131803362 | 0.1499301141193552 |
| 2026-02-27 00:00:00 | U1_liquid_tradable | 4678 | -0.0705169619672167 | -0.09090909090909083 | -0.18557591956217379 | 0.06249999999999976 |
| 2026-03-31 00:00:00 | U1_liquid_tradable | 0 |  |  |  |  |
| 2026-04-13 00:00:00 | U1_liquid_tradable | 0 |  |  |  |  |

## Reject Reasons

| candidate_pool_version | reject_reason | count |
| --- | --- | --- |
| U0_all_tradable | no_next_trade_date | 5184 |
| U0_all_tradable | open_limit_up_unbuyable | 2067 |
| U0_all_tradable | missing_next_day_bar | 155 |
| U0_all_tradable | suspended_like_next_open | 1 |
| U1_liquid_tradable | low_liquidity | 86463 |
| U1_liquid_tradable | insufficient_history | 7943 |
| U1_liquid_tradable | no_next_trade_date | 5184 |
| U1_liquid_tradable | open_limit_up_unbuyable | 2067 |
| U1_liquid_tradable | missing_next_day_bar | 155 |
| U1_liquid_tradable | suspended_like_next_open | 1 |
| U2_risk_sane | low_liquidity | 86463 |
| U2_risk_sane | absolute_high | 11050 |
| U2_risk_sane | insufficient_history | 7943 |
| U2_risk_sane | limit_move_path | 6635 |
| U2_risk_sane | extreme_turnover | 6167 |
| U2_risk_sane | extreme_volatility | 6167 |
| U2_risk_sane | no_next_trade_date | 5184 |
| U2_risk_sane | open_limit_up_unbuyable | 2067 |
| U2_risk_sane | missing_next_day_bar | 155 |
| U2_risk_sane | suspended_like_next_open | 1 |

## 口径

- 信号日：每月最后一个交易日。
- 执行口径：`tplus1_open`。
- 标签：从信号日后第一个 open-to-open 日收益开始，复利持有到下一月信号日生效前后，对齐 `build_open_to_open_returns`。
- 候选池：`U0/U1/U2` 只做可交易、数据有效和极端风险过滤，不做 alpha 判断。
- 特征处理：按 `signal_date` 截面 winsorize 1%/99% 后 z-score，并保留缺失标记。

## 本轮产物

- `data/cache/monthly_selection_features.parquet`
- `data/results/monthly_selection_dataset_2026-04-28_quality.csv`
- `data/results/monthly_selection_dataset_2026-04-28_candidate_pool_width.csv`
- `data/results/monthly_selection_dataset_2026-04-28_candidate_pool_reject_reason.csv`
- `data/results/monthly_selection_dataset_2026-04-28_feature_coverage.csv`
- `data/results/monthly_selection_dataset_2026-04-28_label_distribution.csv`
- `docs/monthly_selection_dataset_2026-04-28.md`
- `data/results/monthly_selection_dataset_2026-04-28_manifest.json`
