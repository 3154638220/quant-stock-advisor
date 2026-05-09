# Monthly Selection M5 Multisource

- 生成时间：`2026-05-09T02:04:34.895630+00:00` · 输出 stem：`monthly_selection_w1_phase23_ic_gate_2026_05_09_2026-05-09`
- 数据集：`data/cache/monthly_selection_features.parquet` · 数据库：`data/market.duckdb`
- 有效标签月份：`63` · 单窗训练行上限：`0`

## Leaderboard

| candidate_pool_version | model | model_type | top_k | months | mean_topk_return | median_topk_return | topk_excess_mean | topk_excess_median | topk_hit_rate | topk_minus_nextk_mean | industry_neutral_topk_excess_mean | turnover_mean | cost_drag_mean | topk_excess_after_cost_mean | topk_excess_annualized | rank_ic_mean | rank_ic_std | rank_ic_months | rank_ic_ir | quantile_top_minus_bottom_mean | neutral_median_excess | strong_down_median_excess | strong_up_median_excess |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| U1_liquid_tradable | M5_price_volume_only_elasticnet_excess | elasticnet | 20 | 39 | 0.0207932 | 0.0169045 | 0.0178885 | 0.0155878 | 0.666667 | 0.0137585 | 0.0174349 | 0.963158 | 0.00154105 | 0.0163664 | 0.237094 | 0.103751 | 0.117364 | 39 | 0.88401 | 0.0154547 | 0.0199595 | 6.73254e-06 | 0.0172371 |
| U1_liquid_tradable | M5_plus_quality_elasticnet_excess | elasticnet | 20 | 39 | 0.0187057 | 0.0202365 | 0.015801 | 0.0119192 | 0.589744 | 0.00444835 | 0.014202 | 0.932895 | 0.00149263 | 0.0139255 | 0.20699 | 0.106594 | 0.110356 | 39 | 0.965912 | 0.0169411 | 0.00460119 | -0.000648907 | 0.0227131 |
| U1_liquid_tradable | M5_price_volume_only_extratrees_excess | tree_sanity | 20 | 39 | 0.0171938 | 0.00802678 | 0.0142891 | 0.0102683 | 0.666667 | 0.00138728 | 0.0126803 | 0.951316 | 0.00152211 | 0.0124026 | 0.185608 | 0.103323 | 0.129463 | 39 | 0.79809 | 0.0155649 | 0.0102683 | 0.0175189 | 0.00618123 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_extratrees_excess | tree_sanity | 20 | 39 | 0.0165684 | 0.00130114 | 0.0136638 | 0.00788888 | 0.615385 | 0.00847313 | 0.0119516 | 0.956579 | 0.00153053 | 0.0123235 | 0.176866 | 0.110659 | 0.123711 | 39 | 0.894494 | 0.0169134 | 0.00643327 | 0.00547531 | 0.0385294 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_plus_liquidity_position_extratrees_excess | tree_sanity | 20 | 39 | 0.01591 | 0.00310045 | 0.0130054 | 0.00747668 | 0.641026 | 0.00120547 | 0.0122521 | 0.959211 | 0.00153474 | 0.0114235 | 0.167726 | 0.109047 | 0.125074 | 39 | 0.871863 | 0.0156847 | 0.00670007 | 0.00747668 | 0.019582 |
| U1_liquid_tradable | M5_plus_quality_extratrees_excess | tree_sanity | 20 | 39 | 0.0139232 | 0.00346714 | 0.0110185 | 0.00897006 | 0.666667 | 0.000491264 | 0.0102292 | 0.956579 | 0.00153053 | 0.00920232 | 0.140537 | 0.107424 | 0.130592 | 39 | 0.822591 | 0.0162877 | 0.00587885 | 0.0175698 | 0.0188908 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_elasticnet_excess | elasticnet | 20 | 39 | 0.0122606 | -0.00211547 | 0.00935594 | -0.00188053 | 0.487179 | -0.00278118 | 0.00855263 | 0.925 | 0.00148 | 0.0078695 | 0.118233 | 0.103697 | 0.108327 | 39 | 0.957255 | 0.0159359 | -0.00188053 | -0.00856173 | 0.0179782 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_plus_liquidity_position_elasticnet_excess | elasticnet | 20 | 39 | 0.00830526 | -0.000951138 | 0.0054006 | -0.0042991 | 0.487179 | 0.00314457 | 0.00527771 | 0.928947 | 0.00148632 | 0.00371262 | 0.0667673 | 0.10355 | 0.11767 | 39 | 0.880005 | 0.0137344 | 0.00349149 | -0.00800792 | 0.0134291 |
| U1_liquid_tradable | B0_market_ew | benchmark | 20 | 39 | 0.00290466 | 0.00265337 | 0 | 0 | 0 |  |  |  | 0 | 0 | 0 |  |  |  |  |  | 0 | 0 | 0 |

## Incremental Delta vs Price-Volume

| candidate_pool_version | model | model_type | top_k | months | mean_topk_return | median_topk_return | topk_excess_mean | topk_excess_median | topk_hit_rate | topk_minus_nextk_mean | industry_neutral_topk_excess_mean | turnover_mean | cost_drag_mean | topk_excess_after_cost_mean | topk_excess_annualized | rank_ic_mean | rank_ic_std | rank_ic_months | rank_ic_ir | quantile_top_minus_bottom_mean | neutral_median_excess | strong_down_median_excess | strong_up_median_excess | feature_spec | base_model | baseline_topk_excess_mean | baseline_topk_excess_after_cost_mean | baseline_rank_ic_mean | baseline_quantile_top_minus_bottom_mean | delta_topk_excess_mean | delta_topk_excess_after_cost_mean | delta_rank_ic_mean | delta_quantile_top_minus_bottom_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_extratrees_excess | tree_sanity | 20 | 39 | 0.0165684 | 0.00130114 | 0.0136638 | 0.00788888 | 0.615385 | 0.00847313 | 0.0119516 | 0.956579 | 0.00153053 | 0.0123235 | 0.176866 | 0.110659 | 0.123711 | 39 | 0.894494 | 0.0169134 | 0.00643327 | 0.00547531 | 0.0385294 | plus_quality_plus_reversal_volume | extratrees_excess | 0.0142891 | 0.0124026 | 0.103323 | 0.0155649 | -0.000625364 | -7.90722e-05 | 0.00733525 | 0.0013485 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_plus_liquidity_position_extratrees_excess | tree_sanity | 20 | 39 | 0.01591 | 0.00310045 | 0.0130054 | 0.00747668 | 0.641026 | 0.00120547 | 0.0122521 | 0.959211 | 0.00153474 | 0.0114235 | 0.167726 | 0.109047 | 0.125074 | 39 | 0.871863 | 0.0156847 | 0.00670007 | 0.00747668 | 0.019582 | plus_quality_plus_reversal_volume_plus_liquidity_position | extratrees_excess | 0.0142891 | 0.0124026 | 0.103323 | 0.0155649 | -0.00128375 | -0.00097906 | 0.00572379 | 0.000119795 |
| U1_liquid_tradable | M5_plus_quality_elasticnet_excess | elasticnet | 20 | 39 | 0.0187057 | 0.0202365 | 0.015801 | 0.0119192 | 0.589744 | 0.00444835 | 0.014202 | 0.932895 | 0.00149263 | 0.0139255 | 0.20699 | 0.106594 | 0.110356 | 39 | 0.965912 | 0.0169411 | 0.00460119 | -0.000648907 | 0.0227131 | plus_quality | elasticnet_excess | 0.0178885 | 0.0163664 | 0.103751 | 0.0154547 | -0.00208748 | -0.00244081 | 0.00284294 | 0.00148633 |
| U1_liquid_tradable | M5_plus_quality_extratrees_excess | tree_sanity | 20 | 39 | 0.0139232 | 0.00346714 | 0.0110185 | 0.00897006 | 0.666667 | 0.000491264 | 0.0102292 | 0.956579 | 0.00153053 | 0.00920232 | 0.140537 | 0.107424 | 0.130592 | 39 | 0.822591 | 0.0162877 | 0.00587885 | 0.0175698 | 0.0188908 | plus_quality | extratrees_excess | 0.0142891 | 0.0124026 | 0.103323 | 0.0155649 | -0.00327058 | -0.00320026 | 0.00410011 | 0.000722747 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_elasticnet_excess | elasticnet | 20 | 39 | 0.0122606 | -0.00211547 | 0.00935594 | -0.00188053 | 0.487179 | -0.00278118 | 0.00855263 | 0.925 | 0.00148 | 0.0078695 | 0.118233 | 0.103697 | 0.108327 | 39 | 0.957255 | 0.0159359 | -0.00188053 | -0.00856173 | 0.0179782 | plus_quality_plus_reversal_volume | elasticnet_excess | 0.0178885 | 0.0163664 | 0.103751 | 0.0154547 | -0.00853257 | -0.00849686 | -5.40601e-05 | 0.000481146 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_plus_liquidity_position_elasticnet_excess | elasticnet | 20 | 39 | 0.00830526 | -0.000951138 | 0.0054006 | -0.0042991 | 0.487179 | 0.00314457 | 0.00527771 | 0.928947 | 0.00148632 | 0.00371262 | 0.0667673 | 0.10355 | 0.11767 | 39 | 0.880005 | 0.0137344 | 0.00349149 | -0.00800792 | 0.0134291 | plus_quality_plus_reversal_volume_plus_liquidity_position | elasticnet_excess | 0.0178885 | 0.0163664 | 0.103751 | 0.0154547 | -0.0124879 | -0.0126537 | -0.000200551 | -0.00172035 |

## Feature Coverage

| feature_spec | families | feature | raw_feature | rows | non_null | coverage_ratio | candidate_pool_pass_coverage_ratio | first_signal_date | last_signal_date |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| plus_quality | price_volume,quality | feature_ret_5d | feature_ret_5d | 307350 | 307017 | 0.998917 | 1 | 2021-01-29 | 2026-04-30 |
| plus_quality | price_volume,quality | feature_ret_20d | feature_ret_20d | 307350 | 306083 | 0.995878 | 1 | 2021-01-29 | 2026-04-30 |
| plus_quality | price_volume,quality | feature_ret_60d | feature_ret_60d | 307350 | 303506 | 0.987493 | 1 | 2021-01-29 | 2026-04-30 |
| plus_quality | price_volume,quality | feature_realized_vol_20d | feature_realized_vol_20d | 307350 | 306673 | 0.997797 | 1 | 2021-01-29 | 2026-04-30 |
| plus_quality | price_volume,quality | feature_amount_20d_log | feature_amount_20d_log | 307350 | 306745 | 0.998032 | 1 | 2021-01-29 | 2026-04-30 |
| plus_quality | price_volume,quality | feature_turnover_20d | feature_turnover_20d | 307350 | 306745 | 0.998032 | 1 | 2021-01-29 | 2026-04-30 |
| plus_quality | price_volume,quality | feature_price_position_250d | feature_price_position_250d | 307350 | 303573 | 0.987711 | 1 | 2021-01-29 | 2026-04-30 |
| plus_quality | price_volume,quality | feature_limit_move_hits_20d | feature_limit_move_hits_20d | 307350 | 307350 | 1 | 1 | 2021-01-29 | 2026-04-30 |
| plus_quality | price_volume,quality | feature_quality_roe_stability_z | feature_quality_roe_stability | 307350 | 307339 | 0.999964 | 1 | 2021-01-29 | 2026-04-30 |
| plus_quality | price_volume,quality | feature_quality_accruals_ratio_z | feature_quality_accruals_ratio | 307350 | 196215 | 0.638409 | 0.651806 | 2021-01-29 | 2026-04-30 |
| plus_quality | price_volume,quality | feature_quality_asset_growth_rate_z | feature_quality_asset_growth_rate | 307350 | 307347 | 0.99999 | 1 | 2021-01-29 | 2026-04-30 |
| plus_quality | price_volume,quality | feature_quality_earnings_surprise_z | feature_quality_earnings_surprise | 307350 | 307233 | 0.999619 | 1 | 2021-01-29 | 2026-04-30 |
| plus_quality_plus_reversal_volume | price_volume,quality,reversal_volume | feature_ret_5d | feature_ret_5d | 307350 | 307017 | 0.998917 | 1 | 2021-01-29 | 2026-04-30 |
| plus_quality_plus_reversal_volume | price_volume,quality,reversal_volume | feature_ret_20d | feature_ret_20d | 307350 | 306083 | 0.995878 | 1 | 2021-01-29 | 2026-04-30 |
| plus_quality_plus_reversal_volume | price_volume,quality,reversal_volume | feature_ret_60d | feature_ret_60d | 307350 | 303506 | 0.987493 | 1 | 2021-01-29 | 2026-04-30 |
| plus_quality_plus_reversal_volume | price_volume,quality,reversal_volume | feature_realized_vol_20d | feature_realized_vol_20d | 307350 | 306673 | 0.997797 | 1 | 2021-01-29 | 2026-04-30 |
| plus_quality_plus_reversal_volume | price_volume,quality,reversal_volume | feature_amount_20d_log | feature_amount_20d_log | 307350 | 306745 | 0.998032 | 1 | 2021-01-29 | 2026-04-30 |
| plus_quality_plus_reversal_volume | price_volume,quality,reversal_volume | feature_turnover_20d | feature_turnover_20d | 307350 | 306745 | 0.998032 | 1 | 2021-01-29 | 2026-04-30 |
| plus_quality_plus_reversal_volume | price_volume,quality,reversal_volume | feature_price_position_250d | feature_price_position_250d | 307350 | 303573 | 0.987711 | 1 | 2021-01-29 | 2026-04-30 |
| plus_quality_plus_reversal_volume | price_volume,quality,reversal_volume | feature_limit_move_hits_20d | feature_limit_move_hits_20d | 307350 | 307350 | 1 | 1 | 2021-01-29 | 2026-04-30 |
| plus_quality_plus_reversal_volume | price_volume,quality,reversal_volume | feature_quality_roe_stability_z | feature_quality_roe_stability | 307350 | 307339 | 0.999964 | 1 | 2021-01-29 | 2026-04-30 |
| plus_quality_plus_reversal_volume | price_volume,quality,reversal_volume | feature_quality_accruals_ratio_z | feature_quality_accruals_ratio | 307350 | 196215 | 0.638409 | 0.651806 | 2021-01-29 | 2026-04-30 |
| plus_quality_plus_reversal_volume | price_volume,quality,reversal_volume | feature_quality_asset_growth_rate_z | feature_quality_asset_growth_rate | 307350 | 307347 | 0.99999 | 1 | 2021-01-29 | 2026-04-30 |
| plus_quality_plus_reversal_volume | price_volume,quality,reversal_volume | feature_quality_earnings_surprise_z | feature_quality_earnings_surprise | 307350 | 307233 | 0.999619 | 1 | 2021-01-29 | 2026-04-30 |
| plus_quality_plus_reversal_volume | price_volume,quality,reversal_volume | feature_reversal_st_reversal_1m_z | feature_reversal_st_reversal_1m | 307350 | 299404 | 0.974147 | 0.999952 | 2021-01-29 | 2026-04-30 |
| plus_quality_plus_reversal_volume | price_volume,quality,reversal_volume | feature_reversal_st_reversal_1w_z | feature_reversal_st_reversal_1w | 307350 | 299404 | 0.974147 | 0.999952 | 2021-01-29 | 2026-04-30 |
| plus_quality_plus_reversal_volume | price_volume,quality,reversal_volume | feature_reversal_volume_spike_z | feature_reversal_volume_spike | 307350 | 299404 | 0.974147 | 0.999952 | 2021-01-29 | 2026-04-30 |
| plus_quality_plus_reversal_volume | price_volume,quality,reversal_volume | feature_reversal_turnover_anomaly_z | feature_reversal_turnover_anomaly | 307350 | 299404 | 0.974147 | 0.999952 | 2021-01-29 | 2026-04-30 |
| plus_quality_plus_reversal_volume | price_volume,quality,reversal_volume | feature_reversal_pv_divergence_z | feature_reversal_pv_divergence | 307350 | 299404 | 0.974147 | 0.999952 | 2021-01-29 | 2026-04-30 |
| plus_quality_plus_reversal_volume_plus_liquidity_position | price_volume,quality,reversal_volume,liquidity_position | feature_ret_5d | feature_ret_5d | 307350 | 307017 | 0.998917 | 1 | 2021-01-29 | 2026-04-30 |
| plus_quality_plus_reversal_volume_plus_liquidity_position | price_volume,quality,reversal_volume,liquidity_position | feature_ret_20d | feature_ret_20d | 307350 | 306083 | 0.995878 | 1 | 2021-01-29 | 2026-04-30 |
| plus_quality_plus_reversal_volume_plus_liquidity_position | price_volume,quality,reversal_volume,liquidity_position | feature_ret_60d | feature_ret_60d | 307350 | 303506 | 0.987493 | 1 | 2021-01-29 | 2026-04-30 |
| plus_quality_plus_reversal_volume_plus_liquidity_position | price_volume,quality,reversal_volume,liquidity_position | feature_realized_vol_20d | feature_realized_vol_20d | 307350 | 306673 | 0.997797 | 1 | 2021-01-29 | 2026-04-30 |
| plus_quality_plus_reversal_volume_plus_liquidity_position | price_volume,quality,reversal_volume,liquidity_position | feature_amount_20d_log | feature_amount_20d_log | 307350 | 306745 | 0.998032 | 1 | 2021-01-29 | 2026-04-30 |
| plus_quality_plus_reversal_volume_plus_liquidity_position | price_volume,quality,reversal_volume,liquidity_position | feature_turnover_20d | feature_turnover_20d | 307350 | 306745 | 0.998032 | 1 | 2021-01-29 | 2026-04-30 |
| plus_quality_plus_reversal_volume_plus_liquidity_position | price_volume,quality,reversal_volume,liquidity_position | feature_price_position_250d | feature_price_position_250d | 307350 | 303573 | 0.987711 | 1 | 2021-01-29 | 2026-04-30 |
| plus_quality_plus_reversal_volume_plus_liquidity_position | price_volume,quality,reversal_volume,liquidity_position | feature_limit_move_hits_20d | feature_limit_move_hits_20d | 307350 | 307350 | 1 | 1 | 2021-01-29 | 2026-04-30 |
| plus_quality_plus_reversal_volume_plus_liquidity_position | price_volume,quality,reversal_volume,liquidity_position | feature_quality_roe_stability_z | feature_quality_roe_stability | 307350 | 307339 | 0.999964 | 1 | 2021-01-29 | 2026-04-30 |
| plus_quality_plus_reversal_volume_plus_liquidity_position | price_volume,quality,reversal_volume,liquidity_position | feature_quality_accruals_ratio_z | feature_quality_accruals_ratio | 307350 | 196215 | 0.638409 | 0.651806 | 2021-01-29 | 2026-04-30 |
| plus_quality_plus_reversal_volume_plus_liquidity_position | price_volume,quality,reversal_volume,liquidity_position | feature_quality_asset_growth_rate_z | feature_quality_asset_growth_rate | 307350 | 307347 | 0.99999 | 1 | 2021-01-29 | 2026-04-30 |
| plus_quality_plus_reversal_volume_plus_liquidity_position | price_volume,quality,reversal_volume,liquidity_position | feature_quality_earnings_surprise_z | feature_quality_earnings_surprise | 307350 | 307233 | 0.999619 | 1 | 2021-01-29 | 2026-04-30 |
| plus_quality_plus_reversal_volume_plus_liquidity_position | price_volume,quality,reversal_volume,liquidity_position | feature_reversal_st_reversal_1m_z | feature_reversal_st_reversal_1m | 307350 | 299404 | 0.974147 | 0.999952 | 2021-01-29 | 2026-04-30 |
| plus_quality_plus_reversal_volume_plus_liquidity_position | price_volume,quality,reversal_volume,liquidity_position | feature_reversal_st_reversal_1w_z | feature_reversal_st_reversal_1w | 307350 | 299404 | 0.974147 | 0.999952 | 2021-01-29 | 2026-04-30 |
| plus_quality_plus_reversal_volume_plus_liquidity_position | price_volume,quality,reversal_volume,liquidity_position | feature_reversal_volume_spike_z | feature_reversal_volume_spike | 307350 | 299404 | 0.974147 | 0.999952 | 2021-01-29 | 2026-04-30 |
| plus_quality_plus_reversal_volume_plus_liquidity_position | price_volume,quality,reversal_volume,liquidity_position | feature_reversal_turnover_anomaly_z | feature_reversal_turnover_anomaly | 307350 | 299404 | 0.974147 | 0.999952 | 2021-01-29 | 2026-04-30 |
| plus_quality_plus_reversal_volume_plus_liquidity_position | price_volume,quality,reversal_volume,liquidity_position | feature_reversal_pv_divergence_z | feature_reversal_pv_divergence | 307350 | 299404 | 0.974147 | 0.999952 | 2021-01-29 | 2026-04-30 |
| plus_quality_plus_reversal_volume_plus_liquidity_position | price_volume,quality,reversal_volume,liquidity_position | feature_liquidity_amihud_z | feature_liquidity_amihud | 307350 | 299404 | 0.974147 | 0.999952 | 2021-01-29 | 2026-04-30 |
| plus_quality_plus_reversal_volume_plus_liquidity_position | price_volume,quality,reversal_volume,liquidity_position | feature_liquidity_high52w_ratio_z | feature_liquidity_high52w_ratio | 307350 | 299404 | 0.974147 | 0.999952 | 2021-01-29 | 2026-04-30 |
| plus_quality_plus_reversal_volume_plus_liquidity_position | price_volume,quality,reversal_volume,liquidity_position | feature_liquidity_low52w_ratio_z | feature_liquidity_low52w_ratio | 307350 | 299368 | 0.97403 | 0.999785 | 2021-01-29 | 2026-04-30 |
| plus_quality_plus_reversal_volume_plus_liquidity_position | price_volume,quality,reversal_volume,liquidity_position | feature_liquidity_price_range_width_z | feature_liquidity_price_range_width | 307350 | 299404 | 0.974147 | 0.999952 | 2021-01-29 | 2026-04-30 |

## Year Slice

| candidate_pool_version | model | top_k | year | months | mean_topk_return | mean_topk_excess | median_topk_excess | hit_rate | mean_topk_minus_nextk |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| U1_liquid_tradable | B0_market_ew | 20 | 2023 | 12 | -0.0185141 | 0 | 0 | 0 |  |
| U1_liquid_tradable | B0_market_ew | 20 | 2024 | 12 | -0.00586991 | 0 | 0 | 0 |  |
| U1_liquid_tradable | B0_market_ew | 20 | 2025 | 12 | 0.0329754 | 0 | 0 | 0 |  |
| U1_liquid_tradable | B0_market_ew | 20 | 2026 | 3 | 0.00339507 | 0 | 0 | 0 |  |
| U1_liquid_tradable | M5_plus_quality_elasticnet_excess | 20 | 2023 | 12 | 0.00593151 | 0.0244456 | 0.01974 | 0.583333 | 0.0106307 |
| U1_liquid_tradable | M5_plus_quality_elasticnet_excess | 20 | 2024 | 12 | -0.00293394 | 0.00293596 | -0.00102508 | 0.5 | 0.000622797 |
| U1_liquid_tradable | M5_plus_quality_elasticnet_excess | 20 | 2025 | 12 | 0.058057 | 0.0250816 | 0.0188136 | 0.75 | 0.00952777 |
| U1_liquid_tradable | M5_plus_quality_elasticnet_excess | 20 | 2026 | 3 | -0.00104425 | -0.00443932 | -0.012565 | 0.333333 | -0.0252964 |
| U1_liquid_tradable | M5_plus_quality_extratrees_excess | 20 | 2023 | 12 | -0.00422774 | 0.0142863 | 0.0158834 | 0.75 | 0.000441058 |
| U1_liquid_tradable | M5_plus_quality_extratrees_excess | 20 | 2024 | 12 | 0.0107727 | 0.0166426 | 0.0195861 | 0.666667 | 0.0126257 |
| U1_liquid_tradable | M5_plus_quality_extratrees_excess | 20 | 2025 | 12 | 0.0431891 | 0.0102137 | 0.0116563 | 0.666667 | -0.00952537 |
| U1_liquid_tradable | M5_plus_quality_extratrees_excess | 20 | 2026 | 3 | -0.0179346 | -0.0213297 | -0.0067868 | 0.333333 | -0.00777914 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_elasticnet_excess | 20 | 2023 | 12 | -0.00786059 | 0.0106535 | 0.00173571 | 0.5 | 0.00286896 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_elasticnet_excess | 20 | 2024 | 12 | -0.00839541 | -0.0025255 | -0.00579648 | 0.416667 | -0.00640433 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_elasticnet_excess | 20 | 2025 | 12 | 0.0555118 | 0.0225364 | 0.0110943 | 0.583333 | -9.43739e-05 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_elasticnet_excess | 20 | 2026 | 3 | 0.00236462 | -0.00103045 | -0.00881351 | 0.333333 | -0.0216363 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_extratrees_excess | 20 | 2023 | 12 | -0.00903068 | 0.0094834 | 0.00932026 | 0.583333 | 0.00111336 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_extratrees_excess | 20 | 2024 | 12 | 0.00808802 | 0.0139579 | -0.0031826 | 0.5 | 0.0113397 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_extratrees_excess | 20 | 2025 | 12 | 0.0557696 | 0.0227942 | 0.0265021 | 0.75 | 0.0124005 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_extratrees_excess | 20 | 2026 | 3 | -0.00391825 | -0.00731333 | 0.00467037 | 0.666667 | 0.0107363 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_plus_liquidity_position_elasticnet_excess | 20 | 2023 | 12 | 0.00121097 | 0.0197251 | 0.0123229 | 0.583333 | 0.0183751 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_plus_liquidity_position_elasticnet_excess | 20 | 2024 | 12 | -0.0177166 | -0.0118467 | -0.0083684 | 0.25 | -0.00591119 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_plus_liquidity_position_elasticnet_excess | 20 | 2025 | 12 | 0.0439813 | 0.0110059 | 0.014566 | 0.666667 | -0.000393833 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_plus_liquidity_position_elasticnet_excess | 20 | 2026 | 3 | -0.00193425 | -0.00532932 | -0.00692667 | 0.333333 | -0.00740086 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_plus_liquidity_position_extratrees_excess | 20 | 2023 | 12 | -0.00350306 | 0.015011 | 0.00549833 | 0.583333 | 0.00392355 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_plus_liquidity_position_extratrees_excess | 20 | 2024 | 12 | 0.0101735 | 0.0160434 | 0.0144531 | 0.666667 | 0.0111426 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_plus_liquidity_position_extratrees_excess | 20 | 2025 | 12 | 0.0431049 | 0.0101296 | 0.00717653 | 0.666667 | -0.0208352 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_plus_liquidity_position_extratrees_excess | 20 | 2026 | 3 | 0.00772887 | 0.0043338 | 0.00154428 | 0.666667 | 0.0387473 |
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
| U1_liquid_tradable | M5_plus_quality_elasticnet_excess | 20 | neutral | 25 | 0.0130604 | 0.0110932 | 0.00460119 | 0.56 | 5.46087e-05 |
| U1_liquid_tradable | M5_plus_quality_elasticnet_excess | 20 | strong_down | 7 | -0.0586878 | 0.0135193 | -0.000648907 | 0.428571 | -0.00255474 |
| U1_liquid_tradable | M5_plus_quality_elasticnet_excess | 20 | strong_up | 7 | 0.116261 | 0.0348966 | 0.0227131 | 0.857143 | 0.0271434 |
| U1_liquid_tradable | M5_plus_quality_extratrees_excess | 20 | neutral | 25 | 0.00830035 | 0.00633308 | 0.00587885 | 0.64 | -0.0022633 |
| U1_liquid_tradable | M5_plus_quality_extratrees_excess | 20 | strong_down | 7 | -0.0590032 | 0.0132038 | 0.0175698 | 0.714286 | 0.0104127 |
| U1_liquid_tradable | M5_plus_quality_extratrees_excess | 20 | strong_up | 7 | 0.106931 | 0.025567 | 0.0188908 | 0.714286 | 0.000407529 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_elasticnet_excess | 20 | neutral | 25 | 0.00539044 | 0.00342316 | -0.00188053 | 0.48 | -0.00554845 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_elasticnet_excess | 20 | strong_down | 7 | -0.0659445 | 0.00626256 | -0.00856173 | 0.285714 | -0.0113251 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_elasticnet_excess | 20 | strong_up | 7 | 0.115002 | 0.0336378 | 0.0179782 | 0.714286 | 0.0156458 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_extratrees_excess | 20 | neutral | 25 | 0.00307829 | 0.00111102 | 0.00643327 | 0.52 | -0.00617264 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_extratrees_excess | 20 | strong_down | 7 | -0.0485047 | 0.0237024 | 0.00547531 | 0.714286 | 0.0338669 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_extratrees_excess | 20 | strong_up | 7 | 0.129821 | 0.0484563 | 0.0385294 | 0.857143 | 0.0353856 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_plus_liquidity_position_elasticnet_excess | 20 | neutral | 25 | 0.00936204 | 0.00739476 | 0.00349149 | 0.52 | 0.00578896 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_plus_liquidity_position_elasticnet_excess | 20 | strong_down | 7 | -0.0744471 | -0.00224003 | -0.00800792 | 0.142857 | 0.00232011 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_plus_liquidity_position_elasticnet_excess | 20 | strong_up | 7 | 0.0872834 | 0.00591922 | 0.0134291 | 0.714286 | -0.00547524 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_plus_liquidity_position_extratrees_excess | 20 | neutral | 25 | 0.0107871 | 0.00881985 | 0.00670007 | 0.6 | -0.000705119 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_plus_liquidity_position_extratrees_excess | 20 | strong_down | 7 | -0.062257 | 0.00995006 | 0.00747668 | 0.714286 | 0.00594831 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_plus_liquidity_position_extratrees_excess | 20 | strong_up | 7 | 0.112373 | 0.0310089 | 0.019582 | 0.714286 | 0.00328615 |
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

- `data/results/monthly_selection_w1_phase23_ic_gate_2026_05_09_2026-05-09_summary.json`
- `data/results/monthly_selection_w1_phase23_ic_gate_2026_05_09_2026-05-09_leaderboard.csv`
- `data/results/monthly_selection_w1_phase23_ic_gate_2026_05_09_2026-05-09_incremental_delta.csv`
- `data/results/monthly_selection_w1_phase23_ic_gate_2026_05_09_2026-05-09_monthly_long.csv`
- `data/results/monthly_selection_w1_phase23_ic_gate_2026_05_09_2026-05-09_rank_ic.csv`
- `data/results/monthly_selection_w1_phase23_ic_gate_2026_05_09_2026-05-09_quantile_spread.csv`
- `data/results/monthly_selection_w1_phase23_ic_gate_2026_05_09_2026-05-09_feature_coverage.csv`
- `data/results/monthly_selection_w1_phase23_ic_gate_2026_05_09_2026-05-09_feature_importance.csv`
- `data/results/monthly_selection_w1_phase23_ic_gate_2026_05_09_2026-05-09_topk_holdings.csv`
- `data/results/monthly_selection_w1_phase23_ic_gate_2026_05_09_2026-05-09_industry_exposure.csv`
- `data/results/monthly_selection_w1_phase23_ic_gate_2026_05_09_2026-05-09_candidate_pool_width.csv`
- `data/results/monthly_selection_w1_phase23_ic_gate_2026_05_09_2026-05-09_candidate_pool_reject_reason.csv`
- `data/results/monthly_selection_w1_phase23_ic_gate_2026_05_09_2026-05-09_year_slice.csv`
- `data/results/monthly_selection_w1_phase23_ic_gate_2026_05_09_2026-05-09_regime_slice.csv`
- `data/results/monthly_selection_w1_phase23_ic_gate_2026_05_09_2026-05-09_market_states.csv`
- `data/results/monthly_selection_w1_phase23_ic_gate_2026_05_09_2026-05-09_manifest.json`
