# Monthly Selection M5 Multisource

- 生成时间：`2026-05-09T02:17:53.953005+00:00` · 输出 stem：`monthly_selection_w1_phase23_ic_gate_quick_2026_05_09_2026-05-09`
- 数据集：`data/cache/monthly_selection_features.parquet` · 数据库：`data/market.duckdb`
- 有效标签月份：`63` · 单窗训练行上限：`2000`

## Leaderboard

| candidate_pool_version | model | model_type | top_k | months | mean_topk_return | median_topk_return | topk_excess_mean | topk_excess_median | topk_hit_rate | topk_minus_nextk_mean | industry_neutral_topk_excess_mean | turnover_mean | cost_drag_mean | topk_excess_after_cost_mean | topk_excess_annualized | rank_ic_mean | rank_ic_std | rank_ic_months | rank_ic_ir | quantile_top_minus_bottom_mean | neutral_median_excess | strong_down_median_excess | strong_up_median_excess |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| U1_liquid_tradable | M5_plus_quality_extratrees_excess | tree_sanity | 20 | 57 | 0.00739498 | 0.010497 | 0.00459797 | -0.00415867 | 0.45614 | 0.00488067 | 0.00384443 | 0.850893 | 0.00136143 | 0.00339291 | 0.0565926 | 0.0709368 | 0.0888656 | 57 | 0.798248 | 0.0129646 | -0.00391721 | -0.0187091 | 0.031612 |
| U1_liquid_tradable | M5_price_volume_only_extratrees_excess | tree_sanity | 20 | 57 | 0.00895731 | 0.0116431 | 0.00616029 | -0.00173657 | 0.45614 | 0.00113179 | 0.00527373 | 0.970536 | 0.00155286 | 0.00308983 | 0.0764803 | 0.0741445 | 0.095544 | 57 | 0.776024 | 0.012613 | -0.00173657 | -0.0197753 | 0.0458196 |
| U1_liquid_tradable | M5_price_volume_only_elasticnet_excess | elasticnet | 20 | 57 | 0.00841414 | 0.0128541 | 0.00561712 | 0.00147413 | 0.508772 | 0.00396293 | 0.00708999 | 0.950893 | 0.00152143 | 0.00207139 | 0.0695274 | 0.0759086 | 0.086884 | 57 | 0.873677 | 0.0169295 | 0.012237 | -0.0120179 | -0.00289351 |
| U1_liquid_tradable | B0_market_ew | benchmark | 20 | 57 | 0.00279702 | 0.00334796 | 0 | 0 | 0 |  |  |  | 0 | 0 | 0 |  |  |  |  |  | 0 | 0 | 0 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_extratrees_excess | tree_sanity | 20 | 57 | 0.0067225 | 0.00597526 | 0.00392548 | -0.00403209 | 0.438596 | 0.00851842 | 0.00387387 | 0.914286 | 0.00146286 | -0.000403451 | 0.0481363 | 0.0687204 | 0.085419 | 57 | 0.80451 | 0.0128105 | -0.00272236 | -0.030117 | 0.0163297 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_plus_liquidity_position_extratrees_excess | tree_sanity | 20 | 57 | 0.00518123 | -0.00547386 | 0.00238421 | -0.00200775 | 0.491228 | -0.00158653 | 0.00448548 | 0.835714 | 0.00133714 | -0.00055393 | 0.0289887 | 0.0676513 | 0.0966692 | 57 | 0.699823 | 0.0111507 | -0.00497727 | -0.0218486 | 0.0295959 |
| U1_liquid_tradable | M5_plus_quality_elasticnet_excess | elasticnet | 20 | 57 | 0.00232313 | 0.000729249 | -0.000473888 | -0.00472346 | 0.403509 | -5.69859e-05 | 0.00240742 | 0.8 | 0.00128 | -0.00225118 | -0.00567185 | 0.0700796 | 0.0785304 | 57 | 0.892388 | 0.0160985 | -0.00472346 | -0.00805198 | 0.00794366 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_plus_liquidity_position_elasticnet_excess | elasticnet | 20 | 57 | 0.00136865 | 0.00153214 | -0.00142837 | -0.00135708 | 0.491228 | 0.00493269 | 0.00184981 | 0.808036 | 0.00129286 | -0.00234249 | -0.0170064 | 0.0538628 | 0.0845068 | 57 | 0.637378 | 0.0105393 | -0.00135708 | -0.0173938 | 0.0121239 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_elasticnet_excess | elasticnet | 20 | 57 | 0.00129875 | 0.00970565 | -0.00149827 | -0.00030646 | 0.45614 | -0.00560348 | 0.000147196 | 0.850893 | 0.00136143 | -0.00328168 | -0.0178318 | 0.0628528 | 0.0777934 | 57 | 0.807945 | 0.013894 | 0.000883374 | -0.0200532 | 0.0108472 |

## Incremental Delta vs Price-Volume

| candidate_pool_version | model | model_type | top_k | months | mean_topk_return | median_topk_return | topk_excess_mean | topk_excess_median | topk_hit_rate | topk_minus_nextk_mean | industry_neutral_topk_excess_mean | turnover_mean | cost_drag_mean | topk_excess_after_cost_mean | topk_excess_annualized | rank_ic_mean | rank_ic_std | rank_ic_months | rank_ic_ir | quantile_top_minus_bottom_mean | neutral_median_excess | strong_down_median_excess | strong_up_median_excess | feature_spec | base_model | baseline_topk_excess_mean | baseline_topk_excess_after_cost_mean | baseline_rank_ic_mean | baseline_quantile_top_minus_bottom_mean | delta_topk_excess_mean | delta_topk_excess_after_cost_mean | delta_rank_ic_mean | delta_quantile_top_minus_bottom_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| U1_liquid_tradable | M5_plus_quality_extratrees_excess | tree_sanity | 20 | 57 | 0.00739498 | 0.010497 | 0.00459797 | -0.00415867 | 0.45614 | 0.00488067 | 0.00384443 | 0.850893 | 0.00136143 | 0.00339291 | 0.0565926 | 0.0709368 | 0.0888656 | 57 | 0.798248 | 0.0129646 | -0.00391721 | -0.0187091 | 0.031612 | plus_quality | extratrees_excess | 0.00616029 | 0.00308983 | 0.0741445 | 0.012613 | -0.00156232 | 0.00030308 | -0.00320774 | 0.000351592 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_extratrees_excess | tree_sanity | 20 | 57 | 0.0067225 | 0.00597526 | 0.00392548 | -0.00403209 | 0.438596 | 0.00851842 | 0.00387387 | 0.914286 | 0.00146286 | -0.000403451 | 0.0481363 | 0.0687204 | 0.085419 | 57 | 0.80451 | 0.0128105 | -0.00272236 | -0.030117 | 0.0163297 | plus_quality_plus_reversal_volume | extratrees_excess | 0.00616029 | 0.00308983 | 0.0741445 | 0.012613 | -0.0022348 | -0.00349328 | -0.00542406 | 0.000197505 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_plus_liquidity_position_extratrees_excess | tree_sanity | 20 | 57 | 0.00518123 | -0.00547386 | 0.00238421 | -0.00200775 | 0.491228 | -0.00158653 | 0.00448548 | 0.835714 | 0.00133714 | -0.00055393 | 0.0289887 | 0.0676513 | 0.0966692 | 57 | 0.699823 | 0.0111507 | -0.00497727 | -0.0218486 | 0.0295959 | plus_quality_plus_reversal_volume_plus_liquidity_position | extratrees_excess | 0.00616029 | 0.00308983 | 0.0741445 | 0.012613 | -0.00377608 | -0.00364376 | -0.0064932 | -0.00146229 |
| U1_liquid_tradable | M5_plus_quality_elasticnet_excess | elasticnet | 20 | 57 | 0.00232313 | 0.000729249 | -0.000473888 | -0.00472346 | 0.403509 | -5.69859e-05 | 0.00240742 | 0.8 | 0.00128 | -0.00225118 | -0.00567185 | 0.0700796 | 0.0785304 | 57 | 0.892388 | 0.0160985 | -0.00472346 | -0.00805198 | 0.00794366 | plus_quality | elasticnet_excess | 0.00561712 | 0.00207139 | 0.0759086 | 0.0169295 | -0.00609101 | -0.00432256 | -0.00582893 | -0.000830976 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_plus_liquidity_position_elasticnet_excess | elasticnet | 20 | 57 | 0.00136865 | 0.00153214 | -0.00142837 | -0.00135708 | 0.491228 | 0.00493269 | 0.00184981 | 0.808036 | 0.00129286 | -0.00234249 | -0.0170064 | 0.0538628 | 0.0845068 | 57 | 0.637378 | 0.0105393 | -0.00135708 | -0.0173938 | 0.0121239 | plus_quality_plus_reversal_volume_plus_liquidity_position | elasticnet_excess | 0.00561712 | 0.00207139 | 0.0759086 | 0.0169295 | -0.00704549 | -0.00441387 | -0.0220458 | -0.00639018 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_elasticnet_excess | elasticnet | 20 | 57 | 0.00129875 | 0.00970565 | -0.00149827 | -0.00030646 | 0.45614 | -0.00560348 | 0.000147196 | 0.850893 | 0.00136143 | -0.00328168 | -0.0178318 | 0.0628528 | 0.0777934 | 57 | 0.807945 | 0.013894 | 0.000883374 | -0.0200532 | 0.0108472 | plus_quality_plus_reversal_volume | elasticnet_excess | 0.00561712 | 0.00207139 | 0.0759086 | 0.0169295 | -0.00711539 | -0.00535307 | -0.0130558 | -0.00303541 |

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
| U1_liquid_tradable | B0_market_ew | 20 | 2021 | 6 | 0.00166624 | 0 | 0 | 0 |  |
| U1_liquid_tradable | B0_market_ew | 20 | 2022 | 12 | 0.00301257 | 0 | 0 | 0 |  |
| U1_liquid_tradable | B0_market_ew | 20 | 2023 | 12 | -0.0185141 | 0 | 0 | 0 |  |
| U1_liquid_tradable | B0_market_ew | 20 | 2024 | 12 | -0.00586991 | 0 | 0 | 0 |  |
| U1_liquid_tradable | B0_market_ew | 20 | 2025 | 12 | 0.0329754 | 0 | 0 | 0 |  |
| U1_liquid_tradable | B0_market_ew | 20 | 2026 | 3 | 0.00339507 | 0 | 0 | 0 |  |
| U1_liquid_tradable | M5_plus_quality_elasticnet_excess | 20 | 2021 | 6 | 0.00993447 | 0.00826824 | 0.00329773 | 0.5 | 0.0053577 |
| U1_liquid_tradable | M5_plus_quality_elasticnet_excess | 20 | 2022 | 12 | -0.00425065 | -0.00726322 | -0.00551369 | 0.166667 | -0.000415646 |
| U1_liquid_tradable | M5_plus_quality_elasticnet_excess | 20 | 2023 | 12 | -0.0247652 | -0.00625108 | -0.00830294 | 0.25 | -0.00106742 |
| U1_liquid_tradable | M5_plus_quality_elasticnet_excess | 20 | 2024 | 12 | -0.0152706 | -0.00940065 | -0.0113197 | 0.416667 | 0.0015935 |
| U1_liquid_tradable | M5_plus_quality_elasticnet_excess | 20 | 2025 | 12 | 0.0465181 | 0.0135427 | 0.00779863 | 0.666667 | -0.00524358 |
| U1_liquid_tradable | M5_plus_quality_elasticnet_excess | 20 | 2026 | 3 | 0.0153436 | 0.0119486 | 0.00350689 | 0.666667 | 0.00873444 |
| U1_liquid_tradable | M5_plus_quality_extratrees_excess | 20 | 2021 | 6 | -0.000830932 | -0.00249717 | -0.0234 | 0.166667 | -0.0322459 |
| U1_liquid_tradable | M5_plus_quality_extratrees_excess | 20 | 2022 | 12 | 0.00858165 | 0.00556908 | 0.00336727 | 0.583333 | 0.0178766 |
| U1_liquid_tradable | M5_plus_quality_extratrees_excess | 20 | 2023 | 12 | -0.0213276 | -0.00281355 | -0.00796489 | 0.416667 | -0.0153055 |
| U1_liquid_tradable | M5_plus_quality_extratrees_excess | 20 | 2024 | 12 | -0.0105068 | -0.00463693 | -0.00900076 | 0.25 | 0.0113093 |
| U1_liquid_tradable | M5_plus_quality_extratrees_excess | 20 | 2025 | 12 | 0.049184 | 0.0162086 | 0.0306439 | 0.666667 | 0.0182739 |
| U1_liquid_tradable | M5_plus_quality_extratrees_excess | 20 | 2026 | 3 | 0.038442 | 0.035047 | 0.0292122 | 0.666667 | 0.0286075 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_elasticnet_excess | 20 | 2021 | 6 | -0.00234833 | -0.00401456 | -0.00469943 | 0.5 | -0.00884942 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_elasticnet_excess | 20 | 2022 | 12 | -0.00881991 | -0.0118325 | -0.00886104 | 0.333333 | -0.0126256 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_elasticnet_excess | 20 | 2023 | 12 | -0.0174039 | 0.00111014 | -0.0101323 | 0.416667 | -0.000802459 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_elasticnet_excess | 20 | 2024 | 12 | -0.0210058 | -0.0151359 | -0.0194281 | 0.333333 | -0.0197086 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_elasticnet_excess | 20 | 2025 | 12 | 0.0468237 | 0.0138483 | 0.00993782 | 0.583333 | 0.00507054 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_elasticnet_excess | 20 | 2026 | 3 | 0.0309968 | 0.0276017 | 0.0376527 | 1 | 0.023497 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_extratrees_excess | 20 | 2021 | 6 | 0.0147677 | 0.0131015 | -0.0261407 | 0.333333 | -0.00248979 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_extratrees_excess | 20 | 2022 | 12 | 0.00911375 | 0.00610118 | 0.00762994 | 0.5 | 0.00852664 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_extratrees_excess | 20 | 2023 | 12 | -0.0176335 | 0.000880594 | -0.00974043 | 0.333333 | 0.0105772 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_extratrees_excess | 20 | 2024 | 12 | -0.00794205 | -0.00207215 | -0.0203602 | 0.416667 | 0.00540425 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_extratrees_excess | 20 | 2025 | 12 | 0.0433801 | 0.0104047 | 0.0195697 | 0.666667 | 0.0174711 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_extratrees_excess | 20 | 2026 | 3 | -0.00948101 | -0.0128761 | -0.00403209 | 0 | -0.00108718 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_plus_liquidity_position_elasticnet_excess | 20 | 2021 | 6 | -0.00757199 | -0.00923823 | -0.00531415 | 0.5 | 0.00169 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_plus_liquidity_position_elasticnet_excess | 20 | 2022 | 12 | -0.00290548 | -0.00591805 | -0.0115922 | 0.416667 | 0.0141983 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_plus_liquidity_position_elasticnet_excess | 20 | 2023 | 12 | -0.0263316 | -0.00781753 | -0.00561892 | 0.25 | -0.00359182 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_plus_liquidity_position_elasticnet_excess | 20 | 2024 | 12 | -0.0152382 | -0.00936825 | -0.0107662 | 0.416667 | -0.00104718 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_plus_liquidity_position_elasticnet_excess | 20 | 2025 | 12 | 0.0500767 | 0.0171013 | 0.0180818 | 0.75 | 0.0123846 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_plus_liquidity_position_elasticnet_excess | 20 | 2026 | 3 | 0.0187425 | 0.0153475 | 0.0129763 | 1 | 0.00256568 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_plus_liquidity_position_extratrees_excess | 20 | 2021 | 6 | -0.0075505 | -0.00921674 | -0.0157519 | 0.5 | 0.00671598 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_plus_liquidity_position_extratrees_excess | 20 | 2022 | 12 | 0.0136197 | 0.0106071 | 0.010582 | 0.583333 | 0.00995753 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_plus_liquidity_position_extratrees_excess | 20 | 2023 | 12 | -0.021679 | -0.00316488 | -0.018444 | 0.333333 | 0.0031767 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_plus_liquidity_position_extratrees_excess | 20 | 2024 | 12 | -0.00881801 | -0.0029481 | -0.000985647 | 0.5 | -0.0127929 |

## Realized Market State Slice

| candidate_pool_version | model | top_k | realized_market_state | months | mean_topk_return | mean_topk_excess | median_topk_excess | hit_rate | mean_topk_minus_nextk |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| U1_liquid_tradable | B0_market_ew | 20 | neutral | 31 | 0.00357718 | 0 | 0 | 0 |  |
| U1_liquid_tradable | B0_market_ew | 20 | strong_down | 13 | -0.0729604 | 0 | 0 | 0 |  |
| U1_liquid_tradable | B0_market_ew | 20 | strong_up | 13 | 0.076694 | 0 | 0 | 0 |  |
| U1_liquid_tradable | M5_plus_quality_elasticnet_excess | 20 | neutral | 31 | -0.00116786 | -0.00474504 | -0.00472346 | 0.419355 | -0.00542818 |
| U1_liquid_tradable | M5_plus_quality_elasticnet_excess | 20 | strong_down | 13 | -0.0813041 | -0.0083437 | -0.00805198 | 0.153846 | 0.0067516 |
| U1_liquid_tradable | M5_plus_quality_elasticnet_excess | 20 | strong_up | 13 | 0.094275 | 0.017581 | 0.00794366 | 0.615385 | 0.00594265 |
| U1_liquid_tradable | M5_plus_quality_extratrees_excess | 20 | neutral | 31 | 0.00757743 | 0.00400025 | -0.00391721 | 0.483871 | 0.00534243 |
| U1_liquid_tradable | M5_plus_quality_extratrees_excess | 20 | strong_down | 13 | -0.0835813 | -0.0106209 | -0.0187091 | 0.307692 | 0.0078247 |
| U1_liquid_tradable | M5_plus_quality_extratrees_excess | 20 | strong_up | 13 | 0.0979362 | 0.0212422 | 0.031612 | 0.538462 | 0.000835491 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_elasticnet_excess | 20 | neutral | 31 | 0.00526625 | 0.00168907 | 0.000883374 | 0.516129 | -0.0035973 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_elasticnet_excess | 20 | strong_down | 13 | -0.0915942 | -0.0186338 | -0.0200532 | 0.230769 | -0.00572019 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_elasticnet_excess | 20 | strong_up | 13 | 0.0847307 | 0.00803666 | 0.0108472 | 0.538462 | -0.0102707 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_extratrees_excess | 20 | neutral | 31 | 0.0054215 | 0.00184432 | -0.00272236 | 0.419355 | 0.0106094 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_extratrees_excess | 20 | strong_down | 13 | -0.0970341 | -0.0240737 | -0.030117 | 0.153846 | -0.00135323 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_extratrees_excess | 20 | strong_up | 13 | 0.113581 | 0.0368874 | 0.0163297 | 0.769231 | 0.0134039 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_plus_liquidity_position_elasticnet_excess | 20 | neutral | 31 | 0.0017688 | -0.00180839 | -0.00135708 | 0.483871 | -0.000974659 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_plus_liquidity_position_elasticnet_excess | 20 | strong_down | 13 | -0.0889584 | -0.015998 | -0.0173938 | 0.230769 | 0.00830226 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_plus_liquidity_position_elasticnet_excess | 20 | strong_up | 13 | 0.0907415 | 0.0140475 | 0.0121239 | 0.769231 | 0.0156499 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_plus_liquidity_position_extratrees_excess | 20 | neutral | 31 | 0.00384229 | 0.000265104 | -0.00497727 | 0.483871 | -0.006146 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_plus_liquidity_position_extratrees_excess | 20 | strong_down | 13 | -0.091862 | -0.0189016 | -0.0218486 | 0.461538 | -0.00688759 |
| U1_liquid_tradable | M5_plus_quality_plus_reversal_volume_plus_liquidity_position_extratrees_excess | 20 | strong_up | 13 | 0.105417 | 0.0287232 | 0.0295959 | 0.538462 | 0.0145871 |
| U1_liquid_tradable | M5_price_volume_only_elasticnet_excess | 20 | neutral | 31 | 0.0170938 | 0.0135166 | 0.012237 | 0.677419 | 0.00838135 |
| U1_liquid_tradable | M5_price_volume_only_elasticnet_excess | 20 | strong_down | 13 | -0.0879218 | -0.0149614 | -0.0120179 | 0.153846 | 0.00847513 |
| U1_liquid_tradable | M5_price_volume_only_elasticnet_excess | 20 | strong_up | 13 | 0.0840525 | 0.00735844 | -0.00289351 | 0.461538 | -0.0110855 |
| U1_liquid_tradable | M5_price_volume_only_extratrees_excess | 20 | neutral | 31 | 0.00668226 | 0.00310507 | -0.00173657 | 0.419355 | -0.00455511 |
| U1_liquid_tradable | M5_price_volume_only_extratrees_excess | 20 | strong_down | 13 | -0.090784 | -0.0178236 | -0.0197753 | 0.384615 | 0.00107535 |
| U1_liquid_tradable | M5_price_volume_only_extratrees_excess | 20 | strong_up | 13 | 0.114124 | 0.0374297 | 0.0458196 | 0.615385 | 0.0147493 |

## 口径

- `price_volume_only` 是 M5 内部 baseline；后续 spec 按 industry_breadth -> fund_flow -> fundamental -> shareholder 累积加入。
- `topk_excess_after_cost` 使用半 L1 月度换手乘以 cost_bps 的简化成本敏感性。

## 本轮产物

- `data/results/monthly_selection_w1_phase23_ic_gate_quick_2026_05_09_2026-05-09_summary.json`
- `data/results/monthly_selection_w1_phase23_ic_gate_quick_2026_05_09_2026-05-09_leaderboard.csv`
- `data/results/monthly_selection_w1_phase23_ic_gate_quick_2026_05_09_2026-05-09_incremental_delta.csv`
- `data/results/monthly_selection_w1_phase23_ic_gate_quick_2026_05_09_2026-05-09_monthly_long.csv`
- `data/results/monthly_selection_w1_phase23_ic_gate_quick_2026_05_09_2026-05-09_rank_ic.csv`
- `data/results/monthly_selection_w1_phase23_ic_gate_quick_2026_05_09_2026-05-09_quantile_spread.csv`
- `data/results/monthly_selection_w1_phase23_ic_gate_quick_2026_05_09_2026-05-09_feature_coverage.csv`
- `data/results/monthly_selection_w1_phase23_ic_gate_quick_2026_05_09_2026-05-09_feature_importance.csv`
- `data/results/monthly_selection_w1_phase23_ic_gate_quick_2026_05_09_2026-05-09_topk_holdings.csv`
- `data/results/monthly_selection_w1_phase23_ic_gate_quick_2026_05_09_2026-05-09_industry_exposure.csv`
- `data/results/monthly_selection_w1_phase23_ic_gate_quick_2026_05_09_2026-05-09_candidate_pool_width.csv`
- `data/results/monthly_selection_w1_phase23_ic_gate_quick_2026_05_09_2026-05-09_candidate_pool_reject_reason.csv`
- `data/results/monthly_selection_w1_phase23_ic_gate_quick_2026_05_09_2026-05-09_year_slice.csv`
- `data/results/monthly_selection_w1_phase23_ic_gate_quick_2026_05_09_2026-05-09_regime_slice.csv`
- `data/results/monthly_selection_w1_phase23_ic_gate_quick_2026_05_09_2026-05-09_market_states.csv`
- `data/results/monthly_selection_w1_phase23_ic_gate_quick_2026_05_09_2026-05-09_manifest.json`
