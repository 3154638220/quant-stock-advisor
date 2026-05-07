# Monthly Selection M5 Multisource

- 生成时间：`2026-05-07T13:17:27.945320+00:00` · 输出 stem：`m5_m13c_northbound_regime_delta_2026-05-07`
- 数据集：`data/cache/monthly_selection_features.parquet` · 数据库：`data/market.duckdb`
- 有效标签月份：`63` · 单窗训练行上限：`0`

## Leaderboard

| candidate_pool_version | model | model_type | top_k | months | mean_topk_return | median_topk_return | topk_excess_mean | topk_excess_median | topk_hit_rate | topk_minus_nextk_mean | industry_neutral_topk_excess_mean | turnover_mean | cost_drag_mean | topk_excess_after_cost_mean | topk_excess_annualized | rank_ic_mean | rank_ic_std | rank_ic_months | rank_ic_ir | quantile_top_minus_bottom_mean | neutral_median_excess | strong_down_median_excess | strong_up_median_excess |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| U1_liquid_tradable | M5_plus_northbound_regime_extratrees_excess | tree_sanity | 20 | 39 | 0.0236952 | 0.0108753 | 0.0207906 | 0.0113086 | 0.615385 | 0.00831147 | 0.019263 | 0.944737 | 0.00151158 | 0.0188869 | 0.280088 | 0.104233 | 0.130643 | 39 | 0.797848 | 0.0151036 | 0.00838631 | -0.00388243 | 0.0375236 |
| U1_liquid_tradable | M5_plus_northbound_regime_elasticnet_excess | elasticnet | 20 | 39 | 0.0207932 | 0.0169045 | 0.0178885 | 0.0155878 | 0.666667 | 0.0137585 | 0.0174349 | 0.963158 | 0.00154105 | 0.0163664 | 0.237094 | 0.103751 | 0.117364 | 39 | 0.88401 | 0.0154547 | 0.0199595 | 6.73254e-06 | 0.0172371 |
| U1_liquid_tradable | M5_price_volume_only_elasticnet_excess | elasticnet | 20 | 39 | 0.0207932 | 0.0169045 | 0.0178885 | 0.0155878 | 0.666667 | 0.0137585 | 0.0174349 | 0.963158 | 0.00154105 | 0.0163664 | 0.237094 | 0.103751 | 0.117364 | 39 | 0.88401 | 0.0154547 | 0.0199595 | 6.73254e-06 | 0.0172371 |
| U1_liquid_tradable | M5_price_volume_only_extratrees_excess | tree_sanity | 20 | 39 | 0.0171938 | 0.00802678 | 0.0142891 | 0.0102683 | 0.666667 | 0.00138728 | 0.0126803 | 0.951316 | 0.00152211 | 0.0124026 | 0.185608 | 0.103323 | 0.129463 | 39 | 0.79809 | 0.0155649 | 0.0102683 | 0.0175189 | 0.00618123 |
| U1_liquid_tradable | B0_market_ew | benchmark | 20 | 39 | 0.00290466 | 0.00265337 | 0 | 0 | 0 |  |  |  | 0 | 0 | 0 |  |  |  |  |  | 0 | 0 | 0 |

## Incremental Delta vs Price-Volume

| candidate_pool_version | model | model_type | top_k | months | mean_topk_return | median_topk_return | topk_excess_mean | topk_excess_median | topk_hit_rate | topk_minus_nextk_mean | industry_neutral_topk_excess_mean | turnover_mean | cost_drag_mean | topk_excess_after_cost_mean | topk_excess_annualized | rank_ic_mean | rank_ic_std | rank_ic_months | rank_ic_ir | quantile_top_minus_bottom_mean | neutral_median_excess | strong_down_median_excess | strong_up_median_excess | feature_spec | base_model | baseline_topk_excess_mean | baseline_topk_excess_after_cost_mean | baseline_rank_ic_mean | baseline_quantile_top_minus_bottom_mean | delta_topk_excess_mean | delta_topk_excess_after_cost_mean | delta_rank_ic_mean | delta_quantile_top_minus_bottom_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| U1_liquid_tradable | M5_plus_northbound_regime_extratrees_excess | tree_sanity | 20 | 39 | 0.0236952 | 0.0108753 | 0.0207906 | 0.0113086 | 0.615385 | 0.00831147 | 0.019263 | 0.944737 | 0.00151158 | 0.0188869 | 0.280088 | 0.104233 | 0.130643 | 39 | 0.797848 | 0.0151036 | 0.00838631 | -0.00388243 | 0.0375236 | plus_northbound_regime | extratrees_excess | 0.0142891 | 0.0124026 | 0.103323 | 0.0155649 | 0.00650147 | 0.00648428 | 0.000909523 | -0.000461391 |
| U1_liquid_tradable | M5_plus_northbound_regime_elasticnet_excess | elasticnet | 20 | 39 | 0.0207932 | 0.0169045 | 0.0178885 | 0.0155878 | 0.666667 | 0.0137585 | 0.0174349 | 0.963158 | 0.00154105 | 0.0163664 | 0.237094 | 0.103751 | 0.117364 | 39 | 0.88401 | 0.0154547 | 0.0199595 | 6.73254e-06 | 0.0172371 | plus_northbound_regime | elasticnet_excess | 0.0178885 | 0.0163664 | 0.103751 | 0.0154547 | 0 | 0 | 0 | 0 |

## Feature Coverage

| feature_spec | families | feature | raw_feature | rows | non_null | coverage_ratio | candidate_pool_pass_coverage_ratio | first_signal_date | last_signal_date |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| plus_northbound_regime | price_volume,northbound_regime | feature_ret_5d | feature_ret_5d | 307350 | 307017 | 0.998917 | 1 | 2021-01-29 | 2026-04-30 |
| plus_northbound_regime | price_volume,northbound_regime | feature_ret_20d | feature_ret_20d | 307350 | 306083 | 0.995878 | 1 | 2021-01-29 | 2026-04-30 |
| plus_northbound_regime | price_volume,northbound_regime | feature_ret_60d | feature_ret_60d | 307350 | 303506 | 0.987493 | 1 | 2021-01-29 | 2026-04-30 |
| plus_northbound_regime | price_volume,northbound_regime | feature_realized_vol_20d | feature_realized_vol_20d | 307350 | 306673 | 0.997797 | 1 | 2021-01-29 | 2026-04-30 |
| plus_northbound_regime | price_volume,northbound_regime | feature_amount_20d_log | feature_amount_20d_log | 307350 | 306745 | 0.998032 | 1 | 2021-01-29 | 2026-04-30 |
| plus_northbound_regime | price_volume,northbound_regime | feature_turnover_20d | feature_turnover_20d | 307350 | 306745 | 0.998032 | 1 | 2021-01-29 | 2026-04-30 |
| plus_northbound_regime | price_volume,northbound_regime | feature_price_position_250d | feature_price_position_250d | 307350 | 303573 | 0.987711 | 1 | 2021-01-29 | 2026-04-30 |
| plus_northbound_regime | price_volume,northbound_regime | feature_limit_move_hits_20d | feature_limit_move_hits_20d | 307350 | 307350 | 1 | 1 | 2021-01-29 | 2026-04-30 |
| plus_northbound_regime | price_volume,northbound_regime | feature_north_net_inflow_1m_z | feature_north_net_inflow_1m | 307350 | 307350 | 1 | 1 | 2021-01-29 | 2026-04-30 |
| plus_northbound_regime | price_volume,northbound_regime | feature_north_net_inflow_3m_z | feature_north_net_inflow_3m | 307350 | 307350 | 1 | 1 | 2021-01-29 | 2026-04-30 |
| plus_northbound_regime | price_volume,northbound_regime | feature_north_inflow_zscore_6m_z | feature_north_inflow_zscore_6m | 307350 | 307350 | 1 | 1 | 2021-01-29 | 2026-04-30 |
| plus_northbound_regime | price_volume,northbound_regime | feature_north_consecutive_outflow_days_z | feature_north_consecutive_outflow_days | 307350 | 307350 | 1 | 1 | 2021-01-29 | 2026-04-30 |

## Year Slice

| candidate_pool_version | model | top_k | year | months | mean_topk_return | mean_topk_excess | median_topk_excess | hit_rate | mean_topk_minus_nextk |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| U1_liquid_tradable | B0_market_ew | 20 | 2023 | 12 | -0.0185141 | 0 | 0 | 0 |  |
| U1_liquid_tradable | B0_market_ew | 20 | 2024 | 12 | -0.00586991 | 0 | 0 | 0 |  |
| U1_liquid_tradable | B0_market_ew | 20 | 2025 | 12 | 0.0329754 | 0 | 0 | 0 |  |
| U1_liquid_tradable | B0_market_ew | 20 | 2026 | 3 | 0.00339507 | 0 | 0 | 0 |  |
| U1_liquid_tradable | M5_plus_northbound_regime_elasticnet_excess | 20 | 2023 | 12 | -0.00410076 | 0.0144133 | 0.0118594 | 0.666667 | -0.00312713 |
| U1_liquid_tradable | M5_plus_northbound_regime_elasticnet_excess | 20 | 2024 | 12 | 0.00511674 | 0.0109866 | 0.00471615 | 0.666667 | 0.0208991 |
| U1_liquid_tradable | M5_plus_northbound_regime_elasticnet_excess | 20 | 2025 | 12 | 0.0667633 | 0.033788 | 0.0202027 | 0.75 | 0.029919 |
| U1_liquid_tradable | M5_plus_northbound_regime_elasticnet_excess | 20 | 2026 | 3 | -0.000806072 | -0.00420114 | -0.00251698 | 0.333333 | -0.0119042 |
| U1_liquid_tradable | M5_plus_northbound_regime_extratrees_excess | 20 | 2023 | 12 | -0.00214063 | 0.0163735 | 0.00669433 | 0.583333 | 0.00674977 |
| U1_liquid_tradable | M5_plus_northbound_regime_extratrees_excess | 20 | 2024 | 12 | 0.0219798 | 0.0278497 | 0.0154313 | 0.666667 | 0.0160671 |
| U1_liquid_tradable | M5_plus_northbound_regime_extratrees_excess | 20 | 2025 | 12 | 0.0588928 | 0.0259175 | 0.0334407 | 0.666667 | 0.00662768 |
| U1_liquid_tradable | M5_plus_northbound_regime_extratrees_excess | 20 | 2026 | 3 | -0.00688991 | -0.010285 | -0.00388243 | 0.333333 | -0.0097289 |
| U1_liquid_tradable | M5_price_volume_only_elasticnet_excess | 20 | 2023 | 12 | -0.00410076 | 0.0144133 | 0.0118594 | 0.666667 | -0.00312713 |
| U1_liquid_tradable | M5_price_volume_only_elasticnet_excess | 20 | 2024 | 12 | 0.00511674 | 0.0109866 | 0.00471615 | 0.666667 | 0.0208991 |
| U1_liquid_tradable | M5_price_volume_only_elasticnet_excess | 20 | 2025 | 12 | 0.0667633 | 0.033788 | 0.0202027 | 0.75 | 0.029919 |
| U1_liquid_tradable | M5_price_volume_only_elasticnet_excess | 20 | 2026 | 3 | -0.000806072 | -0.00420114 | -0.00251698 | 0.333333 | -0.0119042 |
| U1_liquid_tradable | M5_price_volume_only_extratrees_excess | 20 | 2023 | 12 | 0.00167838 | 0.0201925 | 0.0168347 | 0.75 | 0.0139437 |
| U1_liquid_tradable | M5_price_volume_only_extratrees_excess | 20 | 2024 | 12 | 0.00922111 | 0.015091 | 0.0122284 | 0.583333 | 0.000336514 |
| U1_liquid_tradable | M5_price_volume_only_extratrees_excess | 20 | 2025 | 12 | 0.0455205 | 0.0125452 | 0.00822474 | 0.666667 | -0.0124899 |
| U1_liquid_tradable | M5_price_volume_only_extratrees_excess | 20 | 2026 | 3 | -0.00216103 | -0.0055561 | 0.00104224 | 0.666667 | 0.0108733 |

## Realized Market State Slice

| candidate_pool_version | model | top_k | realized_market_state | months | mean_topk_return | mean_topk_excess | median_topk_excess | hit_rate | mean_topk_minus_nextk |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| U1_liquid_tradable | B0_market_ew | 20 | neutral | 25 | 0.00196727 | 0 | 0 | 0 |  |
| U1_liquid_tradable | B0_market_ew | 20 | strong_down | 7 | -0.0722071 | 0 | 0 | 0 |  |
| U1_liquid_tradable | B0_market_ew | 20 | strong_up | 7 | 0.0813642 | 0 | 0 | 0 |  |
| U1_liquid_tradable | M5_plus_northbound_regime_elasticnet_excess | 20 | neutral | 25 | 0.022316 | 0.0203488 | 0.0199595 | 0.72 | 0.0169158 |
| U1_liquid_tradable | M5_plus_northbound_regime_elasticnet_excess | 20 | strong_down | 7 | -0.071877 | 0.000330136 | 6.73254e-06 | 0.571429 | -0.0106594 |
| U1_liquid_tradable | M5_plus_northbound_regime_elasticnet_excess | 20 | strong_up | 7 | 0.108024 | 0.0266603 | 0.0172371 | 0.571429 | 0.0269 |
| U1_liquid_tradable | M5_plus_northbound_regime_extratrees_excess | 20 | neutral | 25 | 0.020176 | 0.0182087 | 0.00838631 | 0.64 | 0.00426322 |
| U1_liquid_tradable | M5_plus_northbound_regime_extratrees_excess | 20 | strong_down | 7 | -0.0657209 | 0.00648617 | -0.00388243 | 0.428571 | 0.00115667 |
| U1_liquid_tradable | M5_plus_northbound_regime_extratrees_excess | 20 | strong_up | 7 | 0.12568 | 0.0443159 | 0.0375236 | 0.714286 | 0.0299243 |
| U1_liquid_tradable | M5_price_volume_only_elasticnet_excess | 20 | neutral | 25 | 0.022316 | 0.0203488 | 0.0199595 | 0.72 | 0.0169158 |
| U1_liquid_tradable | M5_price_volume_only_elasticnet_excess | 20 | strong_down | 7 | -0.071877 | 0.000330136 | 6.73254e-06 | 0.571429 | -0.0106594 |
| U1_liquid_tradable | M5_price_volume_only_elasticnet_excess | 20 | strong_up | 7 | 0.108024 | 0.0266603 | 0.0172371 | 0.571429 | 0.0269 |
| U1_liquid_tradable | M5_price_volume_only_extratrees_excess | 20 | neutral | 25 | 0.00749222 | 0.00552495 | 0.0102683 | 0.64 | -0.0109263 |
| U1_liquid_tradable | M5_price_volume_only_extratrees_excess | 20 | strong_down | 7 | -0.0503783 | 0.0218288 | 0.0175189 | 0.714286 | 0.0239189 |
| U1_liquid_tradable | M5_price_volume_only_extratrees_excess | 20 | strong_up | 7 | 0.119414 | 0.0380501 | 0.00618123 | 0.714286 | 0.0228328 |

## 口径

- `price_volume_only` 是 M5 内部 baseline；后续 spec 按 industry_breadth -> fund_flow -> fundamental -> shareholder 累积加入。
- `topk_excess_after_cost` 使用半 L1 月度换手乘以 cost_bps 的简化成本敏感性。

## 本轮产物

- `data/results/m5_m13c_northbound_regime_delta_2026-05-07_summary.json`
- `data/results/m5_m13c_northbound_regime_delta_2026-05-07_leaderboard.csv`
- `data/results/m5_m13c_northbound_regime_delta_2026-05-07_incremental_delta.csv`
- `data/results/m5_m13c_northbound_regime_delta_2026-05-07_monthly_long.csv`
- `data/results/m5_m13c_northbound_regime_delta_2026-05-07_rank_ic.csv`
- `data/results/m5_m13c_northbound_regime_delta_2026-05-07_quantile_spread.csv`
- `data/results/m5_m13c_northbound_regime_delta_2026-05-07_feature_coverage.csv`
- `data/results/m5_m13c_northbound_regime_delta_2026-05-07_feature_importance.csv`
- `data/results/m5_m13c_northbound_regime_delta_2026-05-07_topk_holdings.csv`
- `data/results/m5_m13c_northbound_regime_delta_2026-05-07_industry_exposure.csv`
- `data/results/m5_m13c_northbound_regime_delta_2026-05-07_candidate_pool_width.csv`
- `data/results/m5_m13c_northbound_regime_delta_2026-05-07_candidate_pool_reject_reason.csv`
- `data/results/m5_m13c_northbound_regime_delta_2026-05-07_year_slice.csv`
- `data/results/m5_m13c_northbound_regime_delta_2026-05-07_regime_slice.csv`
- `data/results/m5_m13c_northbound_regime_delta_2026-05-07_market_states.csv`
- `data/results/m5_m13c_northbound_regime_delta_2026-05-07_manifest.json`
