# Monthly Selection M5 Multisource

- 生成时间：`2026-05-09T08:12:39.166958+00:00` · 输出 stem：`monthly_selection_w4_event_gate_2026_05_09_2026-05-09`
- 数据集：`data/cache/monthly_selection_features.parquet` · 数据库：`data/market.duckdb`
- 有效标签月份：`63` · 单窗训练行上限：`10000`

## Leaderboard

| candidate_pool_version | model | model_type | top_k | months | mean_topk_return | median_topk_return | topk_excess_mean | topk_excess_median | topk_hit_rate | topk_minus_nextk_mean | industry_neutral_topk_excess_mean | turnover_mean | cost_drag_mean | topk_excess_after_cost_mean | topk_excess_annualized | rank_ic_mean | rank_ic_std | rank_ic_months | rank_ic_ir | quantile_top_minus_bottom_mean | neutral_median_excess | strong_down_median_excess | strong_up_median_excess |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| U1_liquid_tradable | M5_plus_quality_extratrees_excess | tree_sanity | 20 | 39 | 0.0150048 | -0.000549226 | 0.0121001 | 0.00465069 | 0.538462 | 0.00132727 | 0.0114251 | 0.846053 | 0.000846053 | 0.0114012 | 0.155265 | 0.0970397 | 0.125628 | 39 | 0.772439 | 0.0128412 | 0.00465069 | -0.0112511 | 0.0201965 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_plus_liquidity_position_extratrees_excess | tree_sanity | 20 | 39 | 0.0133272 | 0.007138 | 0.0104226 | 0.0119308 | 0.564103 | 0.01435 | 0.0114831 | 0.846053 | 0.000846053 | 0.00983112 | 0.132495 | 0.104545 | 0.123022 | 39 | 0.84981 | 0.0135656 | 0.0119308 | -0.00199712 | 0.0260985 |
| U1_liquid_tradable | M5_price_volume_only_elasticnet_excess | elasticnet | 20 | 39 | 0.0134673 | -0.000514616 | 0.0105627 | 0.00915009 | 0.589744 | 0.00279089 | 0.0104044 | 0.960526 | 0.000960526 | 0.00958211 | 0.134381 | 0.0967171 | 0.109938 | 39 | 0.879743 | 0.014246 | 0.0116234 | -0.00533436 | 0.00153157 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_elasticnet_excess | elasticnet | 20 | 39 | 0.0122684 | 0.000404323 | 0.00936376 | 0.0022993 | 0.538462 | 0.00200943 | 0.00856979 | 0.880263 | 0.000880263 | 0.00836341 | 0.118336 | 0.0927181 | 0.102462 | 39 | 0.904904 | 0.0142374 | 0.0039073 | -0.00676287 | 0.000898245 |
| U1_liquid_tradable | M5_price_volume_only_extratrees_excess | tree_sanity | 20 | 39 | 0.0122182 | 0.00683248 | 0.00931352 | 0.00262677 | 0.512821 | 0.0116416 | 0.0102832 | 0.957895 | 0.000957895 | 0.00816316 | 0.117669 | 0.0991264 | 0.120637 | 39 | 0.821694 | 0.0126718 | 0.0166274 | -0.0135842 | 0.0465041 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_extratrees_excess | tree_sanity | 20 | 39 | 0.0119415 | -0.00640616 | 0.00903684 | -0.00469777 | 0.461538 | 0.00237683 | 0.00778568 | 0.860526 | 0.000860526 | 0.00806914 | 0.113998 | 0.105332 | 0.12126 | 39 | 0.868647 | 0.0154748 | -0.00847597 | -0.0040732 | 0.0460024 |
| U1_liquid_tradable | M5_plus_quality_elasticnet_excess | elasticnet | 20 | 39 | 0.0107519 | 0.0119359 | 0.00784724 | 0.00549763 | 0.538462 | 0.00135474 | 0.00817186 | 0.885526 | 0.000885526 | 0.0068502 | 0.0983393 | 0.095564 | 0.10579 | 39 | 0.903339 | 0.0148768 | 0.0110448 | -0.00336347 | 0.00549763 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_plus_liquidity_position_plus_event_extratrees_excess | tree_sanity | 20 | 39 | 0.00765348 | 0.00544767 | 0.00474882 | -7.77154e-05 | 0.487179 | -8.65871e-06 | 0.00461551 | 0.861842 | 0.000861842 | 0.00360506 | 0.058498 | 0.105287 | 0.122875 | 39 | 0.856858 | 0.0146505 | 0.00209971 | -0.0101572 | 0.0256349 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_plus_liquidity_position_elasticnet_excess | elasticnet | 20 | 39 | 0.00608396 | -0.0143998 | 0.0031793 | -0.00555555 | 0.410256 | 0.00237458 | 0.00455479 | 0.882895 | 0.000882895 | 0.00206198 | 0.0388258 | 0.0931755 | 0.112297 | 39 | 0.829721 | 0.0124421 | -0.00348657 | -0.00822579 | 0.025138 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_plus_liquidity_position_plus_event_elasticnet_excess | elasticnet | 20 | 39 | 0.00608396 | -0.0143998 | 0.0031793 | -0.00555555 | 0.410256 | 0.00237458 | 0.00455479 | 0.882895 | 0.000882895 | 0.00206198 | 0.0388258 | 0.0931755 | 0.112297 | 39 | 0.829721 | 0.0124421 | -0.00348657 | -0.00822579 | 0.025138 |
| U1_liquid_tradable | B0_market_ew | benchmark | 20 | 39 | 0.00290466 | 0.00265337 | 0 | 0 | 0 |  |  |  | 0 | 0 | 0 |  |  |  |  |  | 0 | 0 | 0 |

## Incremental Delta vs Price-Volume

| candidate_pool_version | model | model_type | top_k | months | mean_topk_return | median_topk_return | topk_excess_mean | topk_excess_median | topk_hit_rate | topk_minus_nextk_mean | industry_neutral_topk_excess_mean | turnover_mean | cost_drag_mean | topk_excess_after_cost_mean | topk_excess_annualized | rank_ic_mean | rank_ic_std | rank_ic_months | rank_ic_ir | quantile_top_minus_bottom_mean | neutral_median_excess | strong_down_median_excess | strong_up_median_excess | feature_spec | base_model | baseline_topk_excess_mean | baseline_topk_excess_after_cost_mean | baseline_rank_ic_mean | baseline_quantile_top_minus_bottom_mean | delta_topk_excess_mean | delta_topk_excess_after_cost_mean | delta_rank_ic_mean | delta_quantile_top_minus_bottom_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| U1_liquid_tradable | M5_plus_quality_extratrees_excess | tree_sanity | 20 | 39 | 0.0150048 | -0.000549226 | 0.0121001 | 0.00465069 | 0.538462 | 0.00132727 | 0.0114251 | 0.846053 | 0.000846053 | 0.0114012 | 0.155265 | 0.0970397 | 0.125628 | 39 | 0.772439 | 0.0128412 | 0.00465069 | -0.0112511 | 0.0201965 | plus_quality | extratrees_excess | 0.00931352 | 0.00816316 | 0.0991264 | 0.0126718 | 0.00278658 | 0.00323801 | -0.00208664 | 0.000169418 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_plus_liquidity_position_extratrees_excess | tree_sanity | 20 | 39 | 0.0133272 | 0.007138 | 0.0104226 | 0.0119308 | 0.564103 | 0.01435 | 0.0114831 | 0.846053 | 0.000846053 | 0.00983112 | 0.132495 | 0.104545 | 0.123022 | 39 | 0.84981 | 0.0135656 | 0.0119308 | -0.00199712 | 0.0260985 | plus_quality_plus_reversal_volume_plus_liquidity_position | extratrees_excess | 0.00931352 | 0.00816316 | 0.0991264 | 0.0126718 | 0.00110904 | 0.00166797 | 0.00541883 | 0.000893804 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_extratrees_excess | tree_sanity | 20 | 39 | 0.0119415 | -0.00640616 | 0.00903684 | -0.00469777 | 0.461538 | 0.00237683 | 0.00778568 | 0.860526 | 0.000860526 | 0.00806914 | 0.113998 | 0.105332 | 0.12126 | 39 | 0.868647 | 0.0154748 | -0.00847597 | -0.0040732 | 0.0460024 | plus_quality_plus_reversal_volume | extratrees_excess | 0.00931352 | 0.00816316 | 0.0991264 | 0.0126718 | -0.000276683 | -9.40152e-05 | 0.00620577 | 0.00280304 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_elasticnet_excess | elasticnet | 20 | 39 | 0.0122684 | 0.000404323 | 0.00936376 | 0.0022993 | 0.538462 | 0.00200943 | 0.00856979 | 0.880263 | 0.000880263 | 0.00836341 | 0.118336 | 0.0927181 | 0.102462 | 39 | 0.904904 | 0.0142374 | 0.0039073 | -0.00676287 | 0.000898245 | plus_quality_plus_reversal_volume | elasticnet_excess | 0.0105627 | 0.00958211 | 0.0967171 | 0.014246 | -0.00119891 | -0.0012187 | -0.00399899 | -8.65836e-06 |
| U1_liquid_tradable | M5_plus_quality_elasticnet_excess | elasticnet | 20 | 39 | 0.0107519 | 0.0119359 | 0.00784724 | 0.00549763 | 0.538462 | 0.00135474 | 0.00817186 | 0.885526 | 0.000885526 | 0.0068502 | 0.0983393 | 0.095564 | 0.10579 | 39 | 0.903339 | 0.0148768 | 0.0110448 | -0.00336347 | 0.00549763 | plus_quality | elasticnet_excess | 0.0105627 | 0.00958211 | 0.0967171 | 0.014246 | -0.00271543 | -0.00273191 | -0.00115303 | 0.000630768 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_plus_liquidity_position_plus_event_extratrees_excess | tree_sanity | 20 | 39 | 0.00765348 | 0.00544767 | 0.00474882 | -7.77154e-05 | 0.487179 | -8.65871e-06 | 0.00461551 | 0.861842 | 0.000861842 | 0.00360506 | 0.058498 | 0.105287 | 0.122875 | 39 | 0.856858 | 0.0146505 | 0.00209971 | -0.0101572 | 0.0256349 | plus_quality_plus_reversal_volume_plus_liquidity_position_plus_event | extratrees_excess | 0.00931352 | 0.00816316 | 0.0991264 | 0.0126718 | -0.0045647 | -0.0045581 | 0.00616027 | 0.00197877 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_plus_liquidity_position_elasticnet_excess | elasticnet | 20 | 39 | 0.00608396 | -0.0143998 | 0.0031793 | -0.00555555 | 0.410256 | 0.00237458 | 0.00455479 | 0.882895 | 0.000882895 | 0.00206198 | 0.0388258 | 0.0931755 | 0.112297 | 39 | 0.829721 | 0.0124421 | -0.00348657 | -0.00822579 | 0.025138 | plus_quality_plus_reversal_volume_plus_liquidity_position | elasticnet_excess | 0.0105627 | 0.00958211 | 0.0967171 | 0.014246 | -0.00738337 | -0.00752013 | -0.00354154 | -0.00180392 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_plus_liquidity_position_plus_event_elasticnet_excess | elasticnet | 20 | 39 | 0.00608396 | -0.0143998 | 0.0031793 | -0.00555555 | 0.410256 | 0.00237458 | 0.00455479 | 0.882895 | 0.000882895 | 0.00206198 | 0.0388258 | 0.0931755 | 0.112297 | 39 | 0.829721 | 0.0124421 | -0.00348657 | -0.00822579 | 0.025138 | plus_quality_plus_reversal_volume_plus_liquidity_position_plus_event | elasticnet_excess | 0.0105627 | 0.00958211 | 0.0967171 | 0.014246 | -0.00738337 | -0.00752013 | -0.00354154 | -0.00180392 |

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
| plus_quality_plus_reversal_volume_plus_liquidity_position_plus_event | price_volume,quality,reversal_volume,liquidity_position,event | feature_ret_5d | feature_ret_5d | 307350 | 307017 | 0.998917 | 1 | 2021-01-29 | 2026-04-30 |
| plus_quality_plus_reversal_volume_plus_liquidity_position_plus_event | price_volume,quality,reversal_volume,liquidity_position,event | feature_ret_20d | feature_ret_20d | 307350 | 306083 | 0.995878 | 1 | 2021-01-29 | 2026-04-30 |
| plus_quality_plus_reversal_volume_plus_liquidity_position_plus_event | price_volume,quality,reversal_volume,liquidity_position,event | feature_ret_60d | feature_ret_60d | 307350 | 303506 | 0.987493 | 1 | 2021-01-29 | 2026-04-30 |
| plus_quality_plus_reversal_volume_plus_liquidity_position_plus_event | price_volume,quality,reversal_volume,liquidity_position,event | feature_realized_vol_20d | feature_realized_vol_20d | 307350 | 306673 | 0.997797 | 1 | 2021-01-29 | 2026-04-30 |
| plus_quality_plus_reversal_volume_plus_liquidity_position_plus_event | price_volume,quality,reversal_volume,liquidity_position,event | feature_amount_20d_log | feature_amount_20d_log | 307350 | 306745 | 0.998032 | 1 | 2021-01-29 | 2026-04-30 |
| plus_quality_plus_reversal_volume_plus_liquidity_position_plus_event | price_volume,quality,reversal_volume,liquidity_position,event | feature_turnover_20d | feature_turnover_20d | 307350 | 306745 | 0.998032 | 1 | 2021-01-29 | 2026-04-30 |
| plus_quality_plus_reversal_volume_plus_liquidity_position_plus_event | price_volume,quality,reversal_volume,liquidity_position,event | feature_price_position_250d | feature_price_position_250d | 307350 | 303573 | 0.987711 | 1 | 2021-01-29 | 2026-04-30 |
| plus_quality_plus_reversal_volume_plus_liquidity_position_plus_event | price_volume,quality,reversal_volume,liquidity_position,event | feature_limit_move_hits_20d | feature_limit_move_hits_20d | 307350 | 307350 | 1 | 1 | 2021-01-29 | 2026-04-30 |
| plus_quality_plus_reversal_volume_plus_liquidity_position_plus_event | price_volume,quality,reversal_volume,liquidity_position,event | feature_quality_roe_stability_z | feature_quality_roe_stability | 307350 | 307339 | 0.999964 | 1 | 2021-01-29 | 2026-04-30 |
| plus_quality_plus_reversal_volume_plus_liquidity_position_plus_event | price_volume,quality,reversal_volume,liquidity_position,event | feature_quality_accruals_ratio_z | feature_quality_accruals_ratio | 307350 | 196215 | 0.638409 | 0.651806 | 2021-01-29 | 2026-04-30 |
| plus_quality_plus_reversal_volume_plus_liquidity_position_plus_event | price_volume,quality,reversal_volume,liquidity_position,event | feature_quality_asset_growth_rate_z | feature_quality_asset_growth_rate | 307350 | 307347 | 0.99999 | 1 | 2021-01-29 | 2026-04-30 |
| plus_quality_plus_reversal_volume_plus_liquidity_position_plus_event | price_volume,quality,reversal_volume,liquidity_position,event | feature_quality_earnings_surprise_z | feature_quality_earnings_surprise | 307350 | 307233 | 0.999619 | 1 | 2021-01-29 | 2026-04-30 |
| plus_quality_plus_reversal_volume_plus_liquidity_position_plus_event | price_volume,quality,reversal_volume,liquidity_position,event | feature_reversal_st_reversal_1m_z | feature_reversal_st_reversal_1m | 307350 | 299404 | 0.974147 | 0.999952 | 2021-01-29 | 2026-04-30 |
| plus_quality_plus_reversal_volume_plus_liquidity_position_plus_event | price_volume,quality,reversal_volume,liquidity_position,event | feature_reversal_st_reversal_1w_z | feature_reversal_st_reversal_1w | 307350 | 299404 | 0.974147 | 0.999952 | 2021-01-29 | 2026-04-30 |
| plus_quality_plus_reversal_volume_plus_liquidity_position_plus_event | price_volume,quality,reversal_volume,liquidity_position,event | feature_reversal_volume_spike_z | feature_reversal_volume_spike | 307350 | 299404 | 0.974147 | 0.999952 | 2021-01-29 | 2026-04-30 |
| plus_quality_plus_reversal_volume_plus_liquidity_position_plus_event | price_volume,quality,reversal_volume,liquidity_position,event | feature_reversal_turnover_anomaly_z | feature_reversal_turnover_anomaly | 307350 | 299404 | 0.974147 | 0.999952 | 2021-01-29 | 2026-04-30 |
| plus_quality_plus_reversal_volume_plus_liquidity_position_plus_event | price_volume,quality,reversal_volume,liquidity_position,event | feature_reversal_pv_divergence_z | feature_reversal_pv_divergence | 307350 | 299404 | 0.974147 | 0.999952 | 2021-01-29 | 2026-04-30 |
| plus_quality_plus_reversal_volume_plus_liquidity_position_plus_event | price_volume,quality,reversal_volume,liquidity_position,event | feature_liquidity_amihud_z | feature_liquidity_amihud | 307350 | 299404 | 0.974147 | 0.999952 | 2021-01-29 | 2026-04-30 |
| plus_quality_plus_reversal_volume_plus_liquidity_position_plus_event | price_volume,quality,reversal_volume,liquidity_position,event | feature_liquidity_high52w_ratio_z | feature_liquidity_high52w_ratio | 307350 | 299404 | 0.974147 | 0.999952 | 2021-01-29 | 2026-04-30 |
| plus_quality_plus_reversal_volume_plus_liquidity_position_plus_event | price_volume,quality,reversal_volume,liquidity_position,event | feature_liquidity_low52w_ratio_z | feature_liquidity_low52w_ratio | 307350 | 299368 | 0.97403 | 0.999785 | 2021-01-29 | 2026-04-30 |
| plus_quality_plus_reversal_volume_plus_liquidity_position_plus_event | price_volume,quality,reversal_volume,liquidity_position,event | feature_liquidity_price_range_width_z | feature_liquidity_price_range_width | 307350 | 299404 | 0.974147 | 0.999952 | 2021-01-29 | 2026-04-30 |
| plus_quality_plus_reversal_volume_plus_liquidity_position_plus_event | price_volume,quality,reversal_volume,liquidity_position,event | feature_event_earnings_guidance_direction_z | feature_event_earnings_guidance_direction | 307350 | 0 | 0 | 0 |  |  |
| plus_quality_plus_reversal_volume_plus_liquidity_position_plus_event | price_volume,quality,reversal_volume,liquidity_position,event | feature_event_earnings_guidance_magnitude_z | feature_event_earnings_guidance_magnitude | 307350 | 0 | 0 | 0 |  |  |
| plus_quality_plus_reversal_volume_plus_liquidity_position_plus_event | price_volume,quality,reversal_volume,liquidity_position,event | feature_event_earnings_surprise_ttm_z | feature_event_earnings_surprise_ttm | 307350 | 0 | 0 | 0 |  |  |
| plus_quality_plus_reversal_volume_plus_liquidity_position_plus_event | price_volume,quality,reversal_volume,liquidity_position,event | feature_event_buyback_amount_ratio_z | feature_event_buyback_amount_ratio | 307350 | 0 | 0 | 0 |  |  |
| plus_quality_plus_reversal_volume_plus_liquidity_position_plus_event | price_volume,quality,reversal_volume,liquidity_position,event | feature_event_buyback_recent_30d_z | feature_event_buyback_recent_30d | 307350 | 0 | 0 | 0 |  |  |
| plus_quality_plus_reversal_volume_plus_liquidity_position_plus_event | price_volume,quality,reversal_volume,liquidity_position,event | feature_event_reduction_plan_flag_z | feature_event_reduction_plan_flag | 307350 | 0 | 0 | 0 |  |  |
| plus_quality_plus_reversal_volume_plus_liquidity_position_plus_event | price_volume,quality,reversal_volume,liquidity_position,event | feature_event_unlock_ratio_30d_z | feature_event_unlock_ratio_30d | 307350 | 0 | 0 | 0 |  |  |

## Year Slice

| candidate_pool_version | model | top_k | year | months | mean_topk_return | mean_topk_excess | median_topk_excess | hit_rate | mean_topk_minus_nextk |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| U1_liquid_tradable | B0_market_ew | 20 | 2023 | 12 | -0.0185141 | 0 | 0 | 0 |  |
| U1_liquid_tradable | B0_market_ew | 20 | 2024 | 12 | -0.00586991 | 0 | 0 | 0 |  |
| U1_liquid_tradable | B0_market_ew | 20 | 2025 | 12 | 0.0329754 | 0 | 0 | 0 |  |
| U1_liquid_tradable | B0_market_ew | 20 | 2026 | 3 | 0.00339507 | 0 | 0 | 0 |  |
| U1_liquid_tradable | M5_plus_quality_elasticnet_excess | 20 | 2023 | 12 | -0.0156439 | 0.0028702 | -0.00207977 | 0.416667 | 0.00200871 |
| U1_liquid_tradable | M5_plus_quality_elasticnet_excess | 20 | 2024 | 12 | -0.003948 | 0.00192191 | -0.000887505 | 0.5 | 0.00147662 |
| U1_liquid_tradable | M5_plus_quality_elasticnet_excess | 20 | 2025 | 12 | 0.0571688 | 0.0241934 | 0.0114 | 0.75 | 0.00430257 |
| U1_liquid_tradable | M5_plus_quality_elasticnet_excess | 20 | 2026 | 3 | -0.0105329 | -0.013928 | -0.00136493 | 0.333333 | -0.0135401 |
| U1_liquid_tradable | M5_plus_quality_extratrees_excess | 20 | 2023 | 12 | -0.0172245 | 0.00128959 | -0.000612469 | 0.5 | -0.00457256 |
| U1_liquid_tradable | M5_plus_quality_extratrees_excess | 20 | 2024 | 12 | -0.00501391 | 0.000855998 | -0.00243898 | 0.5 | -0.00467782 |
| U1_liquid_tradable | M5_plus_quality_extratrees_excess | 20 | 2025 | 12 | 0.0658784 | 0.032903 | 0.0103952 | 0.583333 | 0.0191161 |
| U1_liquid_tradable | M5_plus_quality_extratrees_excess | 20 | 2026 | 3 | 0.0205021 | 0.017107 | 0.0399032 | 0.666667 | -0.0222086 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_elasticnet_excess | 20 | 2023 | 12 | -0.0117274 | 0.00678672 | 0.00140154 | 0.5 | 0.00462133 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_elasticnet_excess | 20 | 2024 | 12 | -0.0124842 | -0.00661426 | -0.00227127 | 0.5 | -0.00728961 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_elasticnet_excess | 20 | 2025 | 12 | 0.0645247 | 0.0315493 | 0.0213936 | 0.583333 | 0.0093821 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_elasticnet_excess | 20 | 2026 | 3 | -0.00176308 | -0.00515815 | 0.0122371 | 0.666667 | -0.000732747 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_extratrees_excess | 20 | 2023 | 12 | -0.0140177 | 0.00449641 | 0.00451808 | 0.5 | -0.00123327 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_extratrees_excess | 20 | 2024 | 12 | -0.00272148 | 0.00314843 | -0.00830421 | 0.416667 | 0.0134226 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_extratrees_excess | 20 | 2025 | 12 | 0.0488959 | 0.0159205 | -0.0106646 | 0.416667 | -0.00743478 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_extratrees_excess | 20 | 2026 | 3 | 0.0266125 | 0.0232174 | 0.043957 | 0.666667 | 0.0118806 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_plus_liquidity_position_elasticnet_excess | 20 | 2023 | 12 | -0.026717 | -0.00820296 | -0.0098776 | 0.333333 | -0.012886 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_plus_liquidity_position_elasticnet_excess | 20 | 2024 | 12 | -0.0218138 | -0.0159439 | -0.0076709 | 0.166667 | -0.00406135 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_plus_liquidity_position_elasticnet_excess | 20 | 2025 | 12 | 0.0678909 | 0.0349156 | 0.0400517 | 0.666667 | 0.0243576 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_plus_liquidity_position_elasticnet_excess | 20 | 2026 | 3 | 0.00165122 | -0.00174385 | 0.0158569 | 0.666667 | 0.00122848 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_plus_liquidity_position_extratrees_excess | 20 | 2023 | 12 | -0.00876748 | 0.0097466 | 0.0128597 | 0.666667 | 0.0146504 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_plus_liquidity_position_extratrees_excess | 20 | 2024 | 12 | -0.0105726 | -0.00470265 | -0.0097679 | 0.416667 | 0.0105005 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_plus_liquidity_position_extratrees_excess | 20 | 2025 | 12 | 0.058576 | 0.0256007 | 0.0262664 | 0.583333 | 0.0197602 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_plus_liquidity_position_extratrees_excess | 20 | 2026 | 3 | 0.0163098 | 0.0129147 | 0.0453983 | 0.666667 | 0.00690536 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_plus_liquidity_position_plus_event_elasticnet_excess | 20 | 2023 | 12 | -0.026717 | -0.00820296 | -0.0098776 | 0.333333 | -0.012886 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_plus_liquidity_position_plus_event_elasticnet_excess | 20 | 2024 | 12 | -0.0218138 | -0.0159439 | -0.0076709 | 0.166667 | -0.00406135 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_plus_liquidity_position_plus_event_elasticnet_excess | 20 | 2025 | 12 | 0.0678909 | 0.0349156 | 0.0400517 | 0.666667 | 0.0243576 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_plus_liquidity_position_plus_event_elasticnet_excess | 20 | 2026 | 3 | 0.00165122 | -0.00174385 | 0.0158569 | 0.666667 | 0.00122848 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_plus_liquidity_position_plus_event_extratrees_excess | 20 | 2023 | 12 | -0.0169757 | 0.00153835 | -0.00265803 | 0.416667 | 0.000625488 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_plus_liquidity_position_plus_event_extratrees_excess | 20 | 2024 | 12 | -0.00746736 | -0.00159746 | -0.00668554 | 0.416667 | -0.00373116 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_plus_liquidity_position_plus_event_extratrees_excess | 20 | 2025 | 12 | 0.0500166 | 0.0170413 | 0.00909594 | 0.583333 | 0.00792173 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_plus_liquidity_position_plus_event_extratrees_excess | 20 | 2026 | 3 | -0.00279886 | -0.00619393 | 0.0226865 | 0.666667 | -0.0193768 |
| U1_liquid_tradable | M5_price_volume_only_elasticnet_excess | 20 | 2023 | 12 | -0.010561 | 0.00795304 | 0.00486647 | 0.583333 | 0.000313835 |
| U1_liquid_tradable | M5_price_volume_only_elasticnet_excess | 20 | 2024 | 12 | 0.00512549 | 0.0109954 | 0.00882621 | 0.666667 | 0.00727168 |
| U1_liquid_tradable | M5_price_volume_only_elasticnet_excess | 20 | 2025 | 12 | 0.0474194 | 0.014444 | 0.0194329 | 0.5 | 0.00155914 |
| U1_liquid_tradable | M5_price_volume_only_elasticnet_excess | 20 | 2026 | 3 | 0.00713992 | 0.00374485 | 0.0135516 | 0.666667 | -0.000297052 |

## Realized Market State Slice

| candidate_pool_version | model | top_k | realized_market_state | months | mean_topk_return | mean_topk_excess | median_topk_excess | hit_rate | mean_topk_minus_nextk |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| U1_liquid_tradable | B0_market_ew | 20 | neutral | 25 | 0.00196727 | 0 | 0 | 0 |  |
| U1_liquid_tradable | B0_market_ew | 20 | strong_down | 7 | -0.0722071 | 0 | 0 | 0 |  |
| U1_liquid_tradable | B0_market_ew | 20 | strong_up | 7 | 0.0813642 | 0 | 0 | 0 |  |
| U1_liquid_tradable | M5_plus_quality_elasticnet_excess | 20 | neutral | 25 | 0.00832241 | 0.00635513 | 0.0110448 | 0.6 | 0.00104036 |
| U1_liquid_tradable | M5_plus_quality_elasticnet_excess | 20 | strong_down | 7 | -0.0735227 | -0.00131566 | -0.00336347 | 0.285714 | -0.0038316 |
| U1_liquid_tradable | M5_plus_quality_elasticnet_excess | 20 | strong_up | 7 | 0.103703 | 0.0223391 | 0.00549763 | 0.571429 | 0.00766383 |
| U1_liquid_tradable | M5_plus_quality_extratrees_excess | 20 | neutral | 25 | 0.00244809 | 0.000480816 | 0.00465069 | 0.52 | -0.0105699 |
| U1_liquid_tradable | M5_plus_quality_extratrees_excess | 20 | strong_down | 7 | -0.0549491 | 0.017258 | -0.0112511 | 0.428571 | 0.0126817 |
| U1_liquid_tradable | M5_plus_quality_extratrees_excess | 20 | strong_up | 7 | 0.129804 | 0.0484397 | 0.0201965 | 0.714286 | 0.0324625 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_elasticnet_excess | 20 | neutral | 25 | 0.00879565 | 0.00682837 | 0.0039073 | 0.56 | 0.000840461 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_elasticnet_excess | 20 | strong_down | 7 | -0.0759978 | -0.0037907 | -0.00676287 | 0.428571 | -0.0090945 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_elasticnet_excess | 20 | strong_up | 7 | 0.112937 | 0.0315732 | 0.000898245 | 0.571429 | 0.0172882 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_extratrees_excess | 20 | neutral | 25 | 0.0011818 | -0.000785475 | -0.00847597 | 0.44 | -0.000538292 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_extratrees_excess | 20 | strong_down | 7 | -0.0582288 | 0.0139783 | -0.0040732 | 0.428571 | -0.00139968 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_extratrees_excess | 20 | strong_up | 7 | 0.120539 | 0.0391751 | 0.0460024 | 0.571429 | 0.0165645 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_plus_liquidity_position_elasticnet_excess | 20 | neutral | 25 | 0.00289856 | 0.000931283 | -0.00348657 | 0.4 | 0.00128867 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_plus_liquidity_position_elasticnet_excess | 20 | strong_down | 7 | -0.0816333 | -0.00942625 | -0.00822579 | 0.285714 | -0.013571 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_plus_liquidity_position_elasticnet_excess | 20 | strong_up | 7 | 0.105178 | 0.0238135 | 0.025138 | 0.571429 | 0.0221985 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_plus_liquidity_position_extratrees_excess | 20 | neutral | 25 | 0.00386269 | 0.00189542 | 0.0119308 | 0.6 | 0.00319859 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_plus_liquidity_position_extratrees_excess | 20 | strong_down | 7 | -0.0635478 | 0.00865926 | -0.00199712 | 0.428571 | 0.0282076 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_plus_liquidity_position_extratrees_excess | 20 | strong_up | 7 | 0.124004 | 0.0426399 | 0.0260985 | 0.571429 | 0.0403187 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_plus_liquidity_position_plus_event_elasticnet_excess | 20 | neutral | 25 | 0.00289856 | 0.000931283 | -0.00348657 | 0.4 | 0.00128867 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_plus_liquidity_position_plus_event_elasticnet_excess | 20 | strong_down | 7 | -0.0816333 | -0.00942625 | -0.00822579 | 0.285714 | -0.013571 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_plus_liquidity_position_plus_event_elasticnet_excess | 20 | strong_up | 7 | 0.105178 | 0.0238135 | 0.025138 | 0.571429 | 0.0221985 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_plus_liquidity_position_plus_event_extratrees_excess | 20 | neutral | 25 | 3.74496e-05 | -0.00192982 | 0.00209971 | 0.52 | -0.00623786 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_plus_liquidity_position_plus_event_extratrees_excess | 20 | strong_down | 7 | -0.0745218 | -0.00231467 | -0.0101572 | 0.285714 | -0.00130059 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_plus_liquidity_position_plus_event_extratrees_excess | 20 | strong_up | 7 | 0.117029 | 0.0356646 | 0.0256349 | 0.571429 | 0.0235304 |
| U1_liquid_tradable | M5_price_volume_only_elasticnet_excess | 20 | neutral | 25 | 0.0172388 | 0.0152715 | 0.0116234 | 0.64 | 0.00488583 |
| U1_liquid_tradable | M5_price_volume_only_elasticnet_excess | 20 | strong_down | 7 | -0.0737232 | -0.00151611 | -0.00533436 | 0.428571 | -0.00394433 |
| U1_liquid_tradable | M5_price_volume_only_elasticnet_excess | 20 | strong_up | 7 | 0.0871884 | 0.0058242 | 0.00153157 | 0.571429 | 0.00204417 |
| U1_liquid_tradable | M5_price_volume_only_extratrees_excess | 20 | neutral | 25 | 0.00787262 | 0.00590535 | 0.0166274 | 0.52 | 0.0122526 |
| U1_liquid_tradable | M5_price_volume_only_extratrees_excess | 20 | strong_down | 7 | -0.0751753 | -0.00296822 | -0.0135842 | 0.428571 | 0.0163477 |
| U1_liquid_tradable | M5_price_volume_only_extratrees_excess | 20 | strong_up | 7 | 0.115132 | 0.0337673 | 0.0465041 | 0.571429 | 0.00475342 |

## 口径

- `price_volume_only` 是 M5 内部 baseline；后续 spec 按 industry_breadth -> fund_flow -> fundamental -> shareholder 累积加入。
- `topk_excess_after_cost` 使用半 L1 月度换手乘以 cost_bps 的简化成本敏感性。

## 本轮产物

- `data/results/monthly_selection_w4_event_gate_2026_05_09_2026-05-09_summary.json`
- `data/results/monthly_selection_w4_event_gate_2026_05_09_2026-05-09_leaderboard.csv`
- `data/results/monthly_selection_w4_event_gate_2026_05_09_2026-05-09_incremental_delta.csv`
- `data/results/monthly_selection_w4_event_gate_2026_05_09_2026-05-09_monthly_long.csv`
- `data/results/monthly_selection_w4_event_gate_2026_05_09_2026-05-09_rank_ic.csv`
- `data/results/monthly_selection_w4_event_gate_2026_05_09_2026-05-09_quantile_spread.csv`
- `data/results/monthly_selection_w4_event_gate_2026_05_09_2026-05-09_feature_coverage.csv`
- `data/results/monthly_selection_w4_event_gate_2026_05_09_2026-05-09_feature_importance.csv`
- `data/results/monthly_selection_w4_event_gate_2026_05_09_2026-05-09_topk_holdings.csv`
- `data/results/monthly_selection_w4_event_gate_2026_05_09_2026-05-09_industry_exposure.csv`
- `data/results/monthly_selection_w4_event_gate_2026_05_09_2026-05-09_candidate_pool_width.csv`
- `data/results/monthly_selection_w4_event_gate_2026_05_09_2026-05-09_candidate_pool_reject_reason.csv`
- `data/results/monthly_selection_w4_event_gate_2026_05_09_2026-05-09_year_slice.csv`
- `data/results/monthly_selection_w4_event_gate_2026_05_09_2026-05-09_regime_slice.csv`
- `data/results/monthly_selection_w4_event_gate_2026_05_09_2026-05-09_market_states.csv`
- `data/results/monthly_selection_w4_event_gate_2026_05_09_2026-05-09_manifest.json`
