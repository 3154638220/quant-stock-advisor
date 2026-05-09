# Monthly Selection M5 Multisource

- 生成时间：`2026-05-09T02:12:12.200656+00:00` · 输出 stem：`monthly_selection_w1_phase23_ic_gate_fast_2026_05_09_2026-05-09`
- 数据集：`data/cache/monthly_selection_features.parquet` · 数据库：`data/market.duckdb`
- 有效标签月份：`63` · 单窗训练行上限：`30000`

## Leaderboard

| candidate_pool_version | model | model_type | top_k | months | mean_topk_return | median_topk_return | topk_excess_mean | topk_excess_median | topk_hit_rate | topk_minus_nextk_mean | industry_neutral_topk_excess_mean | turnover_mean | cost_drag_mean | topk_excess_after_cost_mean | topk_excess_annualized | rank_ic_mean | rank_ic_std | rank_ic_months | rank_ic_ir | quantile_top_minus_bottom_mean | neutral_median_excess | strong_down_median_excess | strong_up_median_excess |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| U1_liquid_tradable | M5_plus_quality_extratrees_excess | tree_sanity | 20 | 39 | 0.0227815 | 0.0160672 | 0.0198768 | 0.00385778 | 0.564103 | 0.0107654 | 0.0173945 | 0.959211 | 0.00153474 | 0.0184179 | 0.266405 | 0.104546 | 0.129413 | 39 | 0.807849 | 0.0149832 | 0.0143881 | -0.00878075 | 0.00140145 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_extratrees_excess | tree_sanity | 20 | 39 | 0.021514 | 0.0099821 | 0.0186094 | 0.0110738 | 0.641026 | 0.0110605 | 0.0176622 | 0.972368 | 0.00155579 | 0.017446 | 0.247648 | 0.110809 | 0.123321 | 39 | 0.898541 | 0.0165317 | 0.0156997 | -0.0011228 | 0.0128637 |
| U1_liquid_tradable | M5_price_volume_only_extratrees_excess | tree_sanity | 20 | 39 | 0.0220276 | 0.0116705 | 0.0191229 | 0.00659546 | 0.589744 | 0.00883271 | 0.0180376 | 0.967105 | 0.00154737 | 0.0174299 | 0.255217 | 0.101473 | 0.129769 | 39 | 0.781953 | 0.0142109 | 0.0127356 | -0.0066743 | 0.0204708 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_elasticnet_excess | elasticnet | 20 | 39 | 0.0174918 | 0.0334816 | 0.0145871 | 0.00721511 | 0.615385 | 0.0140416 | 0.0135424 | 0.930263 | 0.00148842 | 0.0131408 | 0.189795 | 0.100687 | 0.107032 | 39 | 0.940715 | 0.0147411 | 0.00680369 | 0.00564807 | 0.0570906 |
| U1_liquid_tradable | M5_plus_quality_elasticnet_excess | elasticnet | 20 | 39 | 0.0158572 | 0.00945934 | 0.0129525 | 0.0105929 | 0.589744 | 0.000634988 | 0.0126725 | 0.926316 | 0.00148211 | 0.0113092 | 0.166995 | 0.102617 | 0.109807 | 39 | 0.934518 | 0.0154777 | 0.00611138 | -0.00393781 | 0.0247474 |
| U1_liquid_tradable | M5_price_volume_only_elasticnet_excess | elasticnet | 20 | 39 | 0.0158312 | 0.0143374 | 0.0129266 | 0.012903 | 0.589744 | 0.00160222 | 0.0133434 | 0.964474 | 0.00154316 | 0.0111976 | 0.166637 | 0.100298 | 0.116821 | 39 | 0.858562 | 0.0145334 | 0.0187152 | 0.00923413 | 0.0143565 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_plus_liquidity_position_extratrees_excess | tree_sanity | 20 | 39 | 0.0151329 | 0.00619279 | 0.0122282 | 0.0120884 | 0.589744 | -0.00231136 | 0.011211 | 0.956579 | 0.00153053 | 0.0104216 | 0.157021 | 0.108288 | 0.129079 | 39 | 0.83893 | 0.0147124 | 0.0199485 | -0.00856126 | 0.01116 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_plus_liquidity_position_elasticnet_excess | elasticnet | 20 | 39 | 0.0082922 | -0.00230006 | 0.00538754 | -0.00107502 | 0.435897 | 0.00679987 | 0.00636509 | 0.921053 | 0.00147368 | 0.0040629 | 0.066601 | 0.100496 | 0.115373 | 39 | 0.871046 | 0.0133906 | -0.000275949 | -0.00988719 | 0.0250981 |
| U1_liquid_tradable | B0_market_ew | benchmark | 20 | 39 | 0.00290466 | 0.00265337 | 0 | 0 | 0 |  |  |  | 0 | 0 | 0 |  |  |  |  |  | 0 | 0 | 0 |

## Incremental Delta vs Price-Volume

| candidate_pool_version | model | model_type | top_k | months | mean_topk_return | median_topk_return | topk_excess_mean | topk_excess_median | topk_hit_rate | topk_minus_nextk_mean | industry_neutral_topk_excess_mean | turnover_mean | cost_drag_mean | topk_excess_after_cost_mean | topk_excess_annualized | rank_ic_mean | rank_ic_std | rank_ic_months | rank_ic_ir | quantile_top_minus_bottom_mean | neutral_median_excess | strong_down_median_excess | strong_up_median_excess | feature_spec | base_model | baseline_topk_excess_mean | baseline_topk_excess_after_cost_mean | baseline_rank_ic_mean | baseline_quantile_top_minus_bottom_mean | delta_topk_excess_mean | delta_topk_excess_after_cost_mean | delta_rank_ic_mean | delta_quantile_top_minus_bottom_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_elasticnet_excess | elasticnet | 20 | 39 | 0.0174918 | 0.0334816 | 0.0145871 | 0.00721511 | 0.615385 | 0.0140416 | 0.0135424 | 0.930263 | 0.00148842 | 0.0131408 | 0.189795 | 0.100687 | 0.107032 | 39 | 0.940715 | 0.0147411 | 0.00680369 | 0.00564807 | 0.0570906 | plus_quality_plus_reversal_volume | elasticnet_excess | 0.0129266 | 0.0111976 | 0.100298 | 0.0145334 | 0.00166054 | 0.00194322 | 0.000388694 | 0.00020769 |
| U1_liquid_tradable | M5_plus_quality_extratrees_excess | tree_sanity | 20 | 39 | 0.0227815 | 0.0160672 | 0.0198768 | 0.00385778 | 0.564103 | 0.0107654 | 0.0173945 | 0.959211 | 0.00153474 | 0.0184179 | 0.266405 | 0.104546 | 0.129413 | 39 | 0.807849 | 0.0149832 | 0.0143881 | -0.00878075 | 0.00140145 | plus_quality | extratrees_excess | 0.0191229 | 0.0174299 | 0.101473 | 0.0142109 | 0.000753914 | 0.000988058 | 0.00307278 | 0.000772297 |
| U1_liquid_tradable | M5_plus_quality_elasticnet_excess | elasticnet | 20 | 39 | 0.0158572 | 0.00945934 | 0.0129525 | 0.0105929 | 0.589744 | 0.000634988 | 0.0126725 | 0.926316 | 0.00148211 | 0.0113092 | 0.166995 | 0.102617 | 0.109807 | 39 | 0.934518 | 0.0154777 | 0.00611138 | -0.00393781 | 0.0247474 | plus_quality | elasticnet_excess | 0.0129266 | 0.0111976 | 0.100298 | 0.0145334 | 2.59643e-05 | 0.000111564 | 0.00231901 | 0.000944347 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_extratrees_excess | tree_sanity | 20 | 39 | 0.021514 | 0.0099821 | 0.0186094 | 0.0110738 | 0.641026 | 0.0110605 | 0.0176622 | 0.972368 | 0.00155579 | 0.017446 | 0.247648 | 0.110809 | 0.123321 | 39 | 0.898541 | 0.0165317 | 0.0156997 | -0.0011228 | 0.0128637 | plus_quality_plus_reversal_volume | extratrees_excess | 0.0191229 | 0.0174299 | 0.101473 | 0.0142109 | -0.000513569 | 1.61002e-05 | 0.00933534 | 0.00232079 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_plus_liquidity_position_extratrees_excess | tree_sanity | 20 | 39 | 0.0151329 | 0.00619279 | 0.0122282 | 0.0120884 | 0.589744 | -0.00231136 | 0.011211 | 0.956579 | 0.00153053 | 0.0104216 | 0.157021 | 0.108288 | 0.129079 | 39 | 0.83893 | 0.0147124 | 0.0199485 | -0.00856126 | 0.01116 | plus_quality_plus_reversal_volume_plus_liquidity_position | extratrees_excess | 0.0191229 | 0.0174299 | 0.101473 | 0.0142109 | -0.0068947 | -0.00700827 | 0.00681476 | 0.000501463 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_plus_liquidity_position_elasticnet_excess | elasticnet | 20 | 39 | 0.0082922 | -0.00230006 | 0.00538754 | -0.00107502 | 0.435897 | 0.00679987 | 0.00636509 | 0.921053 | 0.00147368 | 0.0040629 | 0.066601 | 0.100496 | 0.115373 | 39 | 0.871046 | 0.0133906 | -0.000275949 | -0.00988719 | 0.0250981 | plus_quality_plus_reversal_volume_plus_liquidity_position | elasticnet_excess | 0.0129266 | 0.0111976 | 0.100298 | 0.0145334 | -0.00753903 | -0.00713469 | 0.000197747 | -0.00114282 |

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
| U1_liquid_tradable | M5_plus_quality_elasticnet_excess | 20 | 2023 | 12 | -0.0041412 | 0.0143729 | 0.0106937 | 0.583333 | 0.000726437 |
| U1_liquid_tradable | M5_plus_quality_elasticnet_excess | 20 | 2024 | 12 | -0.000326705 | 0.0055432 | 0.00831672 | 0.583333 | 0.00684688 |
| U1_liquid_tradable | M5_plus_quality_elasticnet_excess | 20 | 2025 | 12 | 0.0536185 | 0.0206431 | 0.0160708 | 0.666667 | -0.00537727 |
| U1_liquid_tradable | M5_plus_quality_elasticnet_excess | 20 | 2026 | 3 | 0.00954128 | 0.0061462 | -0.00401014 | 0.333333 | -0.000529338 |
| U1_liquid_tradable | M5_plus_quality_extratrees_excess | 20 | 2023 | 12 | -0.00156934 | 0.0169447 | 0.0127437 | 0.666667 | 0.0142388 |
| U1_liquid_tradable | M5_plus_quality_extratrees_excess | 20 | 2024 | 12 | 0.00733316 | 0.0132031 | 0.00115443 | 0.5 | -0.00630772 |
| U1_liquid_tradable | M5_plus_quality_extratrees_excess | 20 | 2025 | 12 | 0.069458 | 0.0364827 | 0.0078948 | 0.583333 | 0.029349 |
| U1_liquid_tradable | M5_plus_quality_extratrees_excess | 20 | 2026 | 3 | -0.00472787 | -0.00812294 | -0.0140925 | 0.333333 | -0.00917021 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_elasticnet_excess | 20 | 2023 | 12 | -0.016855 | 0.00165909 | -0.00596163 | 0.416667 | -0.00313702 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_elasticnet_excess | 20 | 2024 | 12 | -0.00359332 | 0.00227659 | 0.00396768 | 0.583333 | 0.00230933 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_elasticnet_excess | 20 | 2025 | 12 | 0.0704824 | 0.0375071 | 0.0401704 | 0.75 | 0.038181 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_elasticnet_excess | 20 | 2026 | 3 | 0.0272565 | 0.0238615 | 0.00740791 | 1 | 0.0331278 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_extratrees_excess | 20 | 2023 | 12 | 0.00274547 | 0.0212596 | 0.0157775 | 0.75 | 0.0149252 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_extratrees_excess | 20 | 2024 | 12 | 0.00367272 | 0.00954262 | 0.00116763 | 0.5 | -0.000402461 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_extratrees_excess | 20 | 2025 | 12 | 0.0612125 | 0.0282371 | 0.0161728 | 0.666667 | 0.0222483 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_extratrees_excess | 20 | 2026 | 3 | 0.00915954 | 0.00576447 | 0.00421346 | 0.666667 | -0.00329772 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_plus_liquidity_position_elasticnet_excess | 20 | 2023 | 12 | -0.0172501 | 0.00126396 | -0.000675485 | 0.416667 | 0.00240488 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_plus_liquidity_position_elasticnet_excess | 20 | 2024 | 12 | -0.0109185 | -0.00504855 | -0.00690679 | 0.25 | -0.00272014 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_plus_liquidity_position_elasticnet_excess | 20 | 2025 | 12 | 0.0508914 | 0.017916 | 0.0289015 | 0.666667 | 0.0156006 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_plus_liquidity_position_elasticnet_excess | 20 | 2026 | 3 | 0.0169074 | 0.0135123 | -0.00476783 | 0.333333 | 0.0272571 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_plus_liquidity_position_extratrees_excess | 20 | 2023 | 12 | -0.00636283 | 0.0121512 | 0.0158985 | 0.666667 | 0.00364797 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_plus_liquidity_position_extratrees_excess | 20 | 2024 | 12 | -0.00356568 | 0.00230422 | -0.00764326 | 0.5 | -0.0180551 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_plus_liquidity_position_extratrees_excess | 20 | 2025 | 12 | 0.0565794 | 0.0236041 | 0.00754187 | 0.666667 | 0.00536901 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_plus_liquidity_position_extratrees_excess | 20 | 2026 | 3 | 0.0101239 | 0.0067288 | -0.000921109 | 0.333333 | 0.00610475 |
| U1_liquid_tradable | M5_price_volume_only_elasticnet_excess | 20 | 2023 | 12 | -0.0083815 | 0.0101326 | 0.0163857 | 0.583333 | -0.00547957 |
| U1_liquid_tradable | M5_price_volume_only_elasticnet_excess | 20 | 2024 | 12 | 0.0018733 | 0.0077432 | 0.00712117 | 0.583333 | -0.00546174 |
| U1_liquid_tradable | M5_price_volume_only_elasticnet_excess | 20 | 2025 | 12 | 0.0591987 | 0.0262233 | 0.036773 | 0.666667 | 0.0197753 |
| U1_liquid_tradable | M5_price_volume_only_elasticnet_excess | 20 | 2026 | 3 | -0.00495595 | -0.00835102 | -0.000410202 | 0.333333 | -0.0145071 |
| U1_liquid_tradable | M5_price_volume_only_extratrees_excess | 20 | 2023 | 12 | 0.00246262 | 0.0209767 | 0.00966552 | 0.75 | 0.0118161 |
| U1_liquid_tradable | M5_price_volume_only_extratrees_excess | 20 | 2024 | 12 | 0.00446377 | 0.0103337 | -0.00107669 | 0.416667 | -0.00286002 |
| U1_liquid_tradable | M5_price_volume_only_extratrees_excess | 20 | 2025 | 12 | 0.0622922 | 0.0293169 | 0.0121342 | 0.583333 | 0.0178303 |
| U1_liquid_tradable | M5_price_volume_only_extratrees_excess | 20 | 2026 | 3 | 0.00948417 | 0.00608909 | 0.00844014 | 0.666667 | 0.0076796 |

## Realized Market State Slice

| candidate_pool_version | model | top_k | realized_market_state | months | mean_topk_return | mean_topk_excess | median_topk_excess | hit_rate | mean_topk_minus_nextk |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| U1_liquid_tradable | B0_market_ew | 20 | neutral | 25 | 0.00196727 | 0 | 0 | 0 |  |
| U1_liquid_tradable | B0_market_ew | 20 | strong_down | 7 | -0.0722071 | 0 | 0 | 0 |  |
| U1_liquid_tradable | B0_market_ew | 20 | strong_up | 7 | 0.0813642 | 0 | 0 | 0 |  |
| U1_liquid_tradable | M5_plus_quality_elasticnet_excess | 20 | neutral | 25 | 0.0122053 | 0.010238 | 0.00611138 | 0.6 | -0.00503843 |
| U1_liquid_tradable | M5_plus_quality_elasticnet_excess | 20 | strong_down | 7 | -0.0674001 | 0.00480699 | -0.00393781 | 0.428571 | 0.00396844 |
| U1_liquid_tradable | M5_plus_quality_elasticnet_excess | 20 | strong_up | 7 | 0.112157 | 0.0307928 | 0.0247474 | 0.714286 | 0.0175637 |
| U1_liquid_tradable | M5_plus_quality_extratrees_excess | 20 | neutral | 25 | 0.0221688 | 0.0202015 | 0.0143881 | 0.64 | 0.00778897 |
| U1_liquid_tradable | M5_plus_quality_extratrees_excess | 20 | strong_down | 7 | -0.0718065 | 0.000400633 | -0.00878075 | 0.285714 | 0.0031737 |
| U1_liquid_tradable | M5_plus_quality_extratrees_excess | 20 | strong_up | 7 | 0.119558 | 0.0381935 | 0.00140145 | 0.571429 | 0.0289871 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_elasticnet_excess | 20 | neutral | 25 | 0.00983286 | 0.00786559 | 0.00680369 | 0.6 | 0.00401162 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_elasticnet_excess | 20 | strong_down | 7 | -0.0639218 | 0.00828531 | 0.00564807 | 0.571429 | 0.0103296 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_elasticnet_excess | 20 | strong_up | 7 | 0.126259 | 0.0448944 | 0.0570906 | 0.714286 | 0.053575 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_extratrees_excess | 20 | neutral | 25 | 0.0187566 | 0.0167893 | 0.0156997 | 0.68 | 0.0114747 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_extratrees_excess | 20 | strong_down | 7 | -0.0683283 | 0.0038788 | -0.0011228 | 0.428571 | 0.0034417 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_extratrees_excess | 20 | strong_up | 7 | 0.121204 | 0.0398401 | 0.0128637 | 0.714286 | 0.0172002 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_plus_liquidity_position_elasticnet_excess | 20 | neutral | 25 | 0.00936078 | 0.00739351 | -0.000275949 | 0.48 | 0.00623465 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_plus_liquidity_position_elasticnet_excess | 20 | strong_down | 7 | -0.0812322 | -0.00902509 | -0.00988719 | 0.142857 | -0.00631787 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_plus_liquidity_position_elasticnet_excess | 20 | strong_up | 7 | 0.0940002 | 0.012636 | 0.0250981 | 0.571429 | 0.0219362 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_plus_liquidity_position_extratrees_excess | 20 | neutral | 25 | 0.013261 | 0.0112937 | 0.0199485 | 0.68 | -0.0064366 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_plus_liquidity_position_extratrees_excess | 20 | strong_down | 7 | -0.085036 | -0.0128289 | -0.00856126 | 0.142857 | -0.0105746 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_plus_liquidity_position_extratrees_excess | 20 | strong_up | 7 | 0.121987 | 0.0406229 | 0.01116 | 0.714286 | 0.0206849 |
| U1_liquid_tradable | M5_price_volume_only_elasticnet_excess | 20 | neutral | 25 | 0.0155658 | 0.0135985 | 0.0187152 | 0.56 | -0.003927 |
| U1_liquid_tradable | M5_price_volume_only_elasticnet_excess | 20 | strong_down | 7 | -0.0681647 | 0.00404236 | 0.00923413 | 0.571429 | 0.00422297 |
| U1_liquid_tradable | M5_price_volume_only_elasticnet_excess | 20 | strong_up | 7 | 0.100775 | 0.019411 | 0.0143565 | 0.714286 | 0.0187287 |
| U1_liquid_tradable | M5_price_volume_only_extratrees_excess | 20 | neutral | 25 | 0.0199887 | 0.0180214 | 0.0127356 | 0.68 | 0.00526469 |
| U1_liquid_tradable | M5_price_volume_only_extratrees_excess | 20 | strong_down | 7 | -0.0707642 | 0.00144292 | -0.0066743 | 0.285714 | -0.00128998 |
| U1_liquid_tradable | M5_price_volume_only_extratrees_excess | 20 | strong_up | 7 | 0.122101 | 0.0407368 | 0.0204708 | 0.571429 | 0.0316983 |

## 口径

- `price_volume_only` 是 M5 内部 baseline；后续 spec 按 industry_breadth -> fund_flow -> fundamental -> shareholder 累积加入。
- `topk_excess_after_cost` 使用半 L1 月度换手乘以 cost_bps 的简化成本敏感性。

## 本轮产物

- `data/results/monthly_selection_w1_phase23_ic_gate_fast_2026_05_09_2026-05-09_summary.json`
- `data/results/monthly_selection_w1_phase23_ic_gate_fast_2026_05_09_2026-05-09_leaderboard.csv`
- `data/results/monthly_selection_w1_phase23_ic_gate_fast_2026_05_09_2026-05-09_incremental_delta.csv`
- `data/results/monthly_selection_w1_phase23_ic_gate_fast_2026_05_09_2026-05-09_monthly_long.csv`
- `data/results/monthly_selection_w1_phase23_ic_gate_fast_2026_05_09_2026-05-09_rank_ic.csv`
- `data/results/monthly_selection_w1_phase23_ic_gate_fast_2026_05_09_2026-05-09_quantile_spread.csv`
- `data/results/monthly_selection_w1_phase23_ic_gate_fast_2026_05_09_2026-05-09_feature_coverage.csv`
- `data/results/monthly_selection_w1_phase23_ic_gate_fast_2026_05_09_2026-05-09_feature_importance.csv`
- `data/results/monthly_selection_w1_phase23_ic_gate_fast_2026_05_09_2026-05-09_topk_holdings.csv`
- `data/results/monthly_selection_w1_phase23_ic_gate_fast_2026_05_09_2026-05-09_industry_exposure.csv`
- `data/results/monthly_selection_w1_phase23_ic_gate_fast_2026_05_09_2026-05-09_candidate_pool_width.csv`
- `data/results/monthly_selection_w1_phase23_ic_gate_fast_2026_05_09_2026-05-09_candidate_pool_reject_reason.csv`
- `data/results/monthly_selection_w1_phase23_ic_gate_fast_2026_05_09_2026-05-09_year_slice.csv`
- `data/results/monthly_selection_w1_phase23_ic_gate_fast_2026_05_09_2026-05-09_regime_slice.csv`
- `data/results/monthly_selection_w1_phase23_ic_gate_fast_2026_05_09_2026-05-09_market_states.csv`
- `data/results/monthly_selection_w1_phase23_ic_gate_fast_2026_05_09_2026-05-09_manifest.json`
