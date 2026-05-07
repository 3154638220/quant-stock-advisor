# Monthly Selection M5 Multisource

- 生成时间：`2026-05-07T04:19:32.689814+00:00` · 输出 stem：`m14_lhb_delta_test_2026-05-07`
- 数据集：`data/cache/monthly_selection_features.parquet` · 数据库：`data/market.duckdb`
- 有效标签月份：`63` · 单窗训练行上限：`0`

## Leaderboard

| candidate_pool_version | model | model_type | top_k | months | mean_topk_return | median_topk_return | topk_excess_mean | topk_excess_median | topk_hit_rate | topk_minus_nextk_mean | industry_neutral_topk_excess_mean | turnover_mean | cost_drag_mean | topk_excess_after_cost_mean | topk_excess_annualized | rank_ic_mean | rank_ic_std | rank_ic_months | rank_ic_ir | quantile_top_minus_bottom_mean | neutral_median_excess | strong_down_median_excess | strong_up_median_excess |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| U1_liquid_tradable | M5_plus_lhb_extratrees_excess | tree_sanity | 20 | 39 | 0.0239563 | 0.0178151 | 0.0210517 | 0.0144672 | 0.74359 | 0.0176444 | 0.0182207 | 0.971053 | 0.00155368 | 0.01945 | 0.284022 | 0.10124 | 0.126024 | 39 | 0.803337 | 0.0143702 | 0.0129645 | 0.0206139 | 0.00771673 |
| U1_liquid_tradable | M5_price_volume_only_elasticnet_excess | elasticnet | 20 | 39 | 0.0207932 | 0.0169045 | 0.0178885 | 0.0155878 | 0.666667 | 0.0137585 | 0.0174349 | 0.963158 | 0.00154105 | 0.0163664 | 0.237094 | 0.103751 | 0.117364 | 39 | 0.88401 | 0.0154547 | 0.0199595 | 6.73254e-06 | 0.0172371 |
| U1_liquid_tradable | M5_price_volume_only_extratrees_excess | tree_sanity | 20 | 39 | 0.0171938 | 0.00802678 | 0.0142891 | 0.0102683 | 0.666667 | 0.00138728 | 0.0126803 | 0.951316 | 0.00152211 | 0.0124026 | 0.185608 | 0.103323 | 0.129463 | 39 | 0.79809 | 0.0155649 | 0.0102683 | 0.0175189 | 0.00618123 |
| U1_liquid_tradable | M5_plus_lhb_elasticnet_excess | elasticnet | 20 | 39 | 0.0150689 | 0.0111575 | 0.0121642 | 0.00609807 | 0.615385 | 0.00785414 | 0.0120641 | 0.969737 | 0.00155158 | 0.0104051 | 0.156144 | 0.102499 | 0.111199 | 39 | 0.921758 | 0.0157844 | 0.0111044 | 0.00114266 | -0.00204698 |
| U1_liquid_tradable | M5_price_volume_only_xgboost_excess | xgboost_regression | 20 | 39 | 0.0125281 | 0.00544917 | 0.00962345 | 0.00273244 | 0.564103 | 0.00646434 | 0.0084179 | 0.951316 | 0.00152211 | 0.007627 | 0.121794 | 0.0951941 | 0.109375 | 39 | 0.870343 | 0.0136185 | 0.00447 | 0.00206966 | -0.000653651 |
| U1_liquid_tradable | M5_price_volume_only_xgboost_top20 | xgboost_classifier | 20 | 39 | 0.00889472 | -0.00464138 | 0.00599006 | 0.00183483 | 0.512821 | -0.00111266 | 0.00699303 | 0.944737 | 0.00151158 | 0.00524344 | 0.0742968 | -0.0480456 | 0.156532 | 39 | -0.306939 | 0.000502087 | -0.0161431 | 0.00183483 | 0.0385171 |
| U1_liquid_tradable | M5_plus_lhb_xgboost_excess | xgboost_regression | 20 | 39 | 0.00823367 | -0.00242475 | 0.00532901 | 0.00275917 | 0.538462 | 0.00417036 | 0.0041282 | 0.968421 | 0.00154947 | 0.00438022 | 0.0658561 | 0.105872 | 0.132149 | 39 | 0.801154 | 0.0151434 | 0.00851846 | 0.00275917 | -0.00368618 |
| U1_liquid_tradable | B0_market_ew | benchmark | 20 | 39 | 0.00290466 | 0.00265337 | 0 | 0 | 0 |  |  |  | 0 | 0 | 0 |  |  |  |  |  | 0 | 0 | 0 |
| U1_liquid_tradable | M5_plus_lhb_logistic_top20 | logistic_classifier | 20 | 39 | -0.000876791 | 0.00935926 | -0.00378145 | -0.00307882 | 0.487179 | -0.00346769 | -0.00301873 | 0.960526 | 0.00153684 | -0.00551729 | -0.0444454 | -0.0378705 | 0.164729 | 39 | -0.229896 | 0.00104021 | 0.00331438 | -0.0183063 | 0.00489963 |
| U1_liquid_tradable | M5_plus_lhb_xgboost_top20 | xgboost_classifier | 20 | 39 | -0.00433084 | -0.00716314 | -0.00723549 | -0.0126085 | 0.435897 | -0.00361778 | -0.00319001 | 0.946053 | 0.00151368 | -0.00838641 | -0.0834527 | -0.0574845 | 0.144257 | 39 | -0.398487 | -0.00591335 | -0.0126085 | -0.0288322 | 0.0401159 |
| U1_liquid_tradable | M5_price_volume_only_logistic_top20 | logistic_classifier | 20 | 39 | -0.0032163 | -0.00174597 | -0.00612096 | -0.0150829 | 0.333333 | -0.0109733 | -0.00555021 | 0.957895 | 0.00153263 | -0.00867059 | -0.0710285 | -0.044809 | 0.175483 | 39 | -0.255347 | 0.000221346 | -0.00927874 | -0.0437419 | -0.0065472 |

## Incremental Delta vs Price-Volume

| candidate_pool_version | model | model_type | top_k | months | mean_topk_return | median_topk_return | topk_excess_mean | topk_excess_median | topk_hit_rate | topk_minus_nextk_mean | industry_neutral_topk_excess_mean | turnover_mean | cost_drag_mean | topk_excess_after_cost_mean | topk_excess_annualized | rank_ic_mean | rank_ic_std | rank_ic_months | rank_ic_ir | quantile_top_minus_bottom_mean | neutral_median_excess | strong_down_median_excess | strong_up_median_excess | feature_spec | base_model | baseline_topk_excess_mean | baseline_topk_excess_after_cost_mean | baseline_rank_ic_mean | baseline_quantile_top_minus_bottom_mean | delta_topk_excess_mean | delta_topk_excess_after_cost_mean | delta_rank_ic_mean | delta_quantile_top_minus_bottom_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| U1_liquid_tradable | M5_plus_lhb_extratrees_excess | tree_sanity | 20 | 39 | 0.0239563 | 0.0178151 | 0.0210517 | 0.0144672 | 0.74359 | 0.0176444 | 0.0182207 | 0.971053 | 0.00155368 | 0.01945 | 0.284022 | 0.10124 | 0.126024 | 39 | 0.803337 | 0.0143702 | 0.0129645 | 0.0206139 | 0.00771673 | plus_lhb | extratrees_excess | 0.0142891 | 0.0124026 | 0.103323 | 0.0155649 | 0.00676255 | 0.00704747 | -0.00208374 | -0.00119476 |
| U1_liquid_tradable | M5_plus_lhb_logistic_top20 | logistic_classifier | 20 | 39 | -0.000876791 | 0.00935926 | -0.00378145 | -0.00307882 | 0.487179 | -0.00346769 | -0.00301873 | 0.960526 | 0.00153684 | -0.00551729 | -0.0444454 | -0.0378705 | 0.164729 | 39 | -0.229896 | 0.00104021 | 0.00331438 | -0.0183063 | 0.00489963 | plus_lhb | logistic_top20 | -0.00612096 | -0.00867059 | -0.044809 | 0.000221346 | 0.00233951 | 0.0031533 | 0.00693847 | 0.000818865 |
| U1_liquid_tradable | M5_plus_lhb_xgboost_excess | xgboost_regression | 20 | 39 | 0.00823367 | -0.00242475 | 0.00532901 | 0.00275917 | 0.538462 | 0.00417036 | 0.0041282 | 0.968421 | 0.00154947 | 0.00438022 | 0.0658561 | 0.105872 | 0.132149 | 39 | 0.801154 | 0.0151434 | 0.00851846 | 0.00275917 | -0.00368618 | plus_lhb | xgboost_excess | 0.00962345 | 0.007627 | 0.0951941 | 0.0136185 | -0.00429443 | -0.00324677 | 0.0106776 | 0.00152486 |
| U1_liquid_tradable | M5_plus_lhb_elasticnet_excess | elasticnet | 20 | 39 | 0.0150689 | 0.0111575 | 0.0121642 | 0.00609807 | 0.615385 | 0.00785414 | 0.0120641 | 0.969737 | 0.00155158 | 0.0104051 | 0.156144 | 0.102499 | 0.111199 | 39 | 0.921758 | 0.0157844 | 0.0111044 | 0.00114266 | -0.00204698 | plus_lhb | elasticnet_excess | 0.0178885 | 0.0163664 | 0.103751 | 0.0154547 | -0.00572427 | -0.0059613 | -0.00125245 | 0.000329625 |
| U1_liquid_tradable | M5_plus_lhb_xgboost_top20 | xgboost_classifier | 20 | 39 | -0.00433084 | -0.00716314 | -0.00723549 | -0.0126085 | 0.435897 | -0.00361778 | -0.00319001 | 0.946053 | 0.00151368 | -0.00838641 | -0.0834527 | -0.0574845 | 0.144257 | 39 | -0.398487 | -0.00591335 | -0.0126085 | -0.0288322 | 0.0401159 | plus_lhb | xgboost_top20 | 0.00599006 | 0.00524344 | -0.0480456 | 0.000502087 | -0.0132256 | -0.0136299 | -0.00943883 | -0.00641543 |

## Feature Coverage

| feature_spec | families | feature | raw_feature | rows | non_null | coverage_ratio | candidate_pool_pass_coverage_ratio | first_signal_date | last_signal_date |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| plus_lhb | price_volume,lhb | feature_ret_5d | feature_ret_5d | 307350 | 307017 | 0.998917 | 1 | 2021-01-29 | 2026-04-30 |
| plus_lhb | price_volume,lhb | feature_ret_20d | feature_ret_20d | 307350 | 306083 | 0.995878 | 1 | 2021-01-29 | 2026-04-30 |
| plus_lhb | price_volume,lhb | feature_ret_60d | feature_ret_60d | 307350 | 303506 | 0.987493 | 1 | 2021-01-29 | 2026-04-30 |
| plus_lhb | price_volume,lhb | feature_realized_vol_20d | feature_realized_vol_20d | 307350 | 306673 | 0.997797 | 1 | 2021-01-29 | 2026-04-30 |
| plus_lhb | price_volume,lhb | feature_amount_20d_log | feature_amount_20d_log | 307350 | 306745 | 0.998032 | 1 | 2021-01-29 | 2026-04-30 |
| plus_lhb | price_volume,lhb | feature_turnover_20d | feature_turnover_20d | 307350 | 306745 | 0.998032 | 1 | 2021-01-29 | 2026-04-30 |
| plus_lhb | price_volume,lhb | feature_price_position_250d | feature_price_position_250d | 307350 | 303573 | 0.987711 | 1 | 2021-01-29 | 2026-04-30 |
| plus_lhb | price_volume,lhb | feature_limit_move_hits_20d | feature_limit_move_hits_20d | 307350 | 307350 | 1 | 1 | 2021-01-29 | 2026-04-30 |
| plus_lhb | price_volume,lhb | feature_lhb_appearance_count_1m_z | feature_lhb_appearance_count_1m | 307350 | 37958 | 0.123501 | 0.150295 | 2021-01-29 | 2026-04-30 |
| plus_lhb | price_volume,lhb | feature_lhb_appearance_count_3m_z | feature_lhb_appearance_count_3m | 307350 | 82274 | 0.267688 | 0.306161 | 2021-01-29 | 2026-04-30 |
| plus_lhb | price_volume,lhb | feature_lhb_recent_5d_z | feature_lhb_recent_5d | 307350 | 82274 | 0.267688 | 0.306161 | 2021-01-29 | 2026-04-30 |
| plus_lhb | price_volume,lhb | feature_lhb_avg_change_1m_z | feature_lhb_avg_change_1m | 307350 | 37958 | 0.123501 | 0.150295 | 2021-01-29 | 2026-04-30 |
| plus_lhb | price_volume,lhb | feature_lhb_avg_amount_1m_z | feature_lhb_avg_amount_1m | 307350 | 37958 | 0.123501 | 0.150295 | 2021-01-29 | 2026-04-30 |
| plus_lhb | price_volume,lhb | feature_lhb_is_bullish_1m_z | feature_lhb_is_bullish_1m | 307350 | 82274 | 0.267688 | 0.306161 | 2021-01-29 | 2026-04-30 |
| plus_lhb | price_volume,lhb | feature_lhb_is_bearish_1m_z | feature_lhb_is_bearish_1m | 307350 | 82274 | 0.267688 | 0.306161 | 2021-01-29 | 2026-04-30 |
| plus_lhb | price_volume,lhb | feature_lhb_is_high_turnover_1m_z | feature_lhb_is_high_turnover_1m | 307350 | 82274 | 0.267688 | 0.306161 | 2021-01-29 | 2026-04-30 |

## Year Slice

| candidate_pool_version | model | top_k | year | months | mean_topk_return | mean_topk_excess | median_topk_excess | hit_rate | mean_topk_minus_nextk |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| U1_liquid_tradable | B0_market_ew | 20 | 2023 | 12 | -0.0185141 | 0 | 0 | 0 |  |
| U1_liquid_tradable | B0_market_ew | 20 | 2024 | 12 | -0.00586991 | 0 | 0 | 0 |  |
| U1_liquid_tradable | B0_market_ew | 20 | 2025 | 12 | 0.0329754 | 0 | 0 | 0 |  |
| U1_liquid_tradable | B0_market_ew | 20 | 2026 | 3 | 0.00339507 | 0 | 0 | 0 |  |
| U1_liquid_tradable | M5_plus_lhb_elasticnet_excess | 20 | 2023 | 12 | -0.00143439 | 0.0170797 | 0.0155468 | 0.75 | 0.00209189 |
| U1_liquid_tradable | M5_plus_lhb_elasticnet_excess | 20 | 2024 | 12 | -0.00262879 | 0.00324112 | 0.000253757 | 0.5 | 0.0108425 |
| U1_liquid_tradable | M5_plus_lhb_elasticnet_excess | 20 | 2025 | 12 | 0.0528737 | 0.0198984 | 0.0103188 | 0.583333 | 0.0127386 |
| U1_liquid_tradable | M5_plus_lhb_elasticnet_excess | 20 | 2026 | 3 | 0.000653519 | -0.00274155 | 0.000178112 | 0.666667 | -0.000588273 |
| U1_liquid_tradable | M5_plus_lhb_extratrees_excess | 20 | 2023 | 12 | 0.00116509 | 0.0196792 | 0.0142354 | 0.75 | 0.0224344 |
| U1_liquid_tradable | M5_plus_lhb_extratrees_excess | 20 | 2024 | 12 | 0.00805008 | 0.01392 | 0.0190434 | 0.666667 | 0.018262 |
| U1_liquid_tradable | M5_plus_lhb_extratrees_excess | 20 | 2025 | 12 | 0.0642911 | 0.0313158 | 0.00733738 | 0.833333 | 0.0074497 |
| U1_liquid_tradable | M5_plus_lhb_extratrees_excess | 20 | 2026 | 3 | 0.017407 | 0.0140119 | 0.0179637 | 0.666667 | 0.0367933 |
| U1_liquid_tradable | M5_plus_lhb_logistic_top20 | 20 | 2023 | 12 | -0.0311237 | -0.0126096 | -0.000575182 | 0.5 | -0.0164477 |
| U1_liquid_tradable | M5_plus_lhb_logistic_top20 | 20 | 2024 | 12 | -0.0112358 | -0.00536586 | -0.00836582 | 0.5 | 0.016162 |
| U1_liquid_tradable | M5_plus_lhb_logistic_top20 | 20 | 2025 | 12 | 0.0356069 | 0.00263155 | 0.000910405 | 0.5 | -0.000574244 |
| U1_liquid_tradable | M5_plus_lhb_logistic_top20 | 20 | 2026 | 3 | 0.0156119 | 0.0122168 | -0.0175824 | 0.333333 | -0.04164 |
| U1_liquid_tradable | M5_plus_lhb_xgboost_excess | 20 | 2023 | 12 | -0.0131454 | 0.00536864 | 0.00199301 | 0.583333 | 0.00470162 |
| U1_liquid_tradable | M5_plus_lhb_xgboost_excess | 20 | 2024 | 12 | -0.00214123 | 0.00372868 | -0.00800411 | 0.416667 | 0.00350734 |
| U1_liquid_tradable | M5_plus_lhb_xgboost_excess | 20 | 2025 | 12 | 0.0422913 | 0.00931596 | 0.0121706 | 0.583333 | 0.00682593 |
| U1_liquid_tradable | M5_plus_lhb_xgboost_excess | 20 | 2026 | 3 | -0.00098087 | -0.00437594 | 0.00924339 | 0.666667 | -0.00592483 |
| U1_liquid_tradable | M5_plus_lhb_xgboost_top20 | 20 | 2023 | 12 | -0.0408563 | -0.0223423 | -0.0198233 | 0.25 | 0.00443098 |
| U1_liquid_tradable | M5_plus_lhb_xgboost_top20 | 20 | 2024 | 12 | -0.0173525 | -0.0114826 | -0.0198195 | 0.416667 | -0.0063776 |
| U1_liquid_tradable | M5_plus_lhb_xgboost_top20 | 20 | 2025 | 12 | 0.0325211 | -0.000454239 | -0.00537989 | 0.5 | -0.00949668 |
| U1_liquid_tradable | M5_plus_lhb_xgboost_top20 | 20 | 2026 | 3 | 0.0464501 | 0.043055 | 0.0389627 | 1 | -0.00125793 |
| U1_liquid_tradable | M5_price_volume_only_elasticnet_excess | 20 | 2023 | 12 | -0.00410076 | 0.0144133 | 0.0118594 | 0.666667 | -0.00312713 |
| U1_liquid_tradable | M5_price_volume_only_elasticnet_excess | 20 | 2024 | 12 | 0.00511674 | 0.0109866 | 0.00471615 | 0.666667 | 0.0208991 |
| U1_liquid_tradable | M5_price_volume_only_elasticnet_excess | 20 | 2025 | 12 | 0.0667633 | 0.033788 | 0.0202027 | 0.75 | 0.029919 |
| U1_liquid_tradable | M5_price_volume_only_elasticnet_excess | 20 | 2026 | 3 | -0.000806072 | -0.00420114 | -0.00251698 | 0.333333 | -0.0119042 |
| U1_liquid_tradable | M5_price_volume_only_extratrees_excess | 20 | 2023 | 12 | 0.00167838 | 0.0201925 | 0.0168347 | 0.75 | 0.0139437 |
| U1_liquid_tradable | M5_price_volume_only_extratrees_excess | 20 | 2024 | 12 | 0.00922111 | 0.015091 | 0.0122284 | 0.583333 | 0.000336514 |
| U1_liquid_tradable | M5_price_volume_only_extratrees_excess | 20 | 2025 | 12 | 0.0455205 | 0.0125452 | 0.00822474 | 0.666667 | -0.0124899 |
| U1_liquid_tradable | M5_price_volume_only_extratrees_excess | 20 | 2026 | 3 | -0.00216103 | -0.0055561 | 0.00104224 | 0.666667 | 0.0108733 |
| U1_liquid_tradable | M5_price_volume_only_logistic_top20 | 20 | 2023 | 12 | -0.0341508 | -0.0156367 | -0.0131089 | 0.333333 | -0.0163869 |
| U1_liquid_tradable | M5_price_volume_only_logistic_top20 | 20 | 2024 | 12 | -0.0196706 | -0.0138007 | -0.0281047 | 0.333333 | -0.00291331 |
| U1_liquid_tradable | M5_price_volume_only_logistic_top20 | 20 | 2025 | 12 | 0.029817 | -0.00315836 | -0.0101145 | 0.333333 | -0.01208 |
| U1_liquid_tradable | M5_price_volume_only_logistic_top20 | 20 | 2026 | 3 | 0.0542057 | 0.0508106 | -0.0150829 | 0.333333 | -0.0171315 |
| U1_liquid_tradable | M5_price_volume_only_xgboost_excess | 20 | 2023 | 12 | 0.00199017 | 0.0205042 | 0.0168457 | 0.75 | 0.0155425 |
| U1_liquid_tradable | M5_price_volume_only_xgboost_excess | 20 | 2024 | 12 | -0.0157829 | -0.00991304 | -0.0101641 | 0.333333 | -0.0146791 |
| U1_liquid_tradable | M5_price_volume_only_xgboost_excess | 20 | 2025 | 12 | 0.0519958 | 0.0190204 | 0.00174376 | 0.583333 | 0.018623 |
| U1_liquid_tradable | M5_price_volume_only_xgboost_excess | 20 | 2026 | 3 | 0.0100532 | 0.00665816 | 0.00289835 | 0.666667 | 0.00609086 |

## Realized Market State Slice

| candidate_pool_version | model | top_k | realized_market_state | months | mean_topk_return | mean_topk_excess | median_topk_excess | hit_rate | mean_topk_minus_nextk |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| U1_liquid_tradable | B0_market_ew | 20 | neutral | 25 | 0.00196727 | 0 | 0 | 0 |  |
| U1_liquid_tradable | B0_market_ew | 20 | strong_down | 7 | -0.0722071 | 0 | 0 | 0 |  |
| U1_liquid_tradable | B0_market_ew | 20 | strong_up | 7 | 0.0813642 | 0 | 0 | 0 |  |
| U1_liquid_tradable | M5_plus_lhb_elasticnet_excess | 20 | neutral | 25 | 0.0159146 | 0.0139473 | 0.0111044 | 0.68 | 0.0114555 |
| U1_liquid_tradable | M5_plus_lhb_elasticnet_excess | 20 | strong_down | 7 | -0.0693248 | 0.00288232 | 0.00114266 | 0.571429 | -0.00136805 |
| U1_liquid_tradable | M5_plus_lhb_elasticnet_excess | 20 | strong_up | 7 | 0.0964422 | 0.015078 | -0.00204698 | 0.428571 | 0.0042144 |
| U1_liquid_tradable | M5_plus_lhb_extratrees_excess | 20 | neutral | 25 | 0.021943 | 0.0199757 | 0.0129645 | 0.72 | 0.017809 |
| U1_liquid_tradable | M5_plus_lhb_extratrees_excess | 20 | strong_down | 7 | -0.0585708 | 0.0136363 | 0.0206139 | 0.714286 | 0.0235887 |
| U1_liquid_tradable | M5_plus_lhb_extratrees_excess | 20 | strong_up | 7 | 0.113674 | 0.0323096 | 0.00771673 | 0.857143 | 0.0111124 |
| U1_liquid_tradable | M5_plus_lhb_logistic_top20 | 20 | neutral | 25 | -0.000387225 | -0.0023545 | 0.00331438 | 0.52 | -0.00851319 |
| U1_liquid_tradable | M5_plus_lhb_logistic_top20 | 20 | strong_down | 7 | -0.0988723 | -0.0266652 | -0.0183063 | 0.285714 | 0.014971 |
| U1_liquid_tradable | M5_plus_lhb_logistic_top20 | 20 | strong_up | 7 | 0.0953702 | 0.014006 | 0.00489963 | 0.571429 | -0.0038867 |
| U1_liquid_tradable | M5_plus_lhb_xgboost_excess | 20 | neutral | 25 | 0.00980935 | 0.00784208 | 0.00851846 | 0.6 | 0.00572348 |
| U1_liquid_tradable | M5_plus_lhb_xgboost_excess | 20 | strong_down | 7 | -0.0701993 | 0.00200781 | 0.00275917 | 0.571429 | -0.00531969 |
| U1_liquid_tradable | M5_plus_lhb_xgboost_excess | 20 | strong_up | 7 | 0.0810392 | -0.000325007 | -0.00368618 | 0.285714 | 0.00811358 |
| U1_liquid_tradable | M5_plus_lhb_xgboost_top20 | 20 | neutral | 25 | -0.0077007 | -0.00966797 | -0.0126085 | 0.4 | -0.00734723 |
| U1_liquid_tradable | M5_plus_lhb_xgboost_top20 | 20 | strong_down | 7 | -0.0938888 | -0.0216817 | -0.0288322 | 0.285714 | -0.0041128 |
| U1_liquid_tradable | M5_plus_lhb_xgboost_top20 | 20 | strong_up | 7 | 0.0972623 | 0.0158981 | 0.0401159 | 0.714286 | 0.0101967 |
| U1_liquid_tradable | M5_price_volume_only_elasticnet_excess | 20 | neutral | 25 | 0.022316 | 0.0203488 | 0.0199595 | 0.72 | 0.0169158 |
| U1_liquid_tradable | M5_price_volume_only_elasticnet_excess | 20 | strong_down | 7 | -0.071877 | 0.000330136 | 6.73254e-06 | 0.571429 | -0.0106594 |
| U1_liquid_tradable | M5_price_volume_only_elasticnet_excess | 20 | strong_up | 7 | 0.108024 | 0.0266603 | 0.0172371 | 0.571429 | 0.0269 |
| U1_liquid_tradable | M5_price_volume_only_extratrees_excess | 20 | neutral | 25 | 0.00749222 | 0.00552495 | 0.0102683 | 0.64 | -0.0109263 |
| U1_liquid_tradable | M5_price_volume_only_extratrees_excess | 20 | strong_down | 7 | -0.0503783 | 0.0218288 | 0.0175189 | 0.714286 | 0.0239189 |
| U1_liquid_tradable | M5_price_volume_only_extratrees_excess | 20 | strong_up | 7 | 0.119414 | 0.0380501 | 0.00618123 | 0.714286 | 0.0228328 |
| U1_liquid_tradable | M5_price_volume_only_logistic_top20 | 20 | neutral | 25 | 0.00318464 | 0.00121737 | -0.00927874 | 0.4 | -0.00779209 |
| U1_liquid_tradable | M5_price_volume_only_logistic_top20 | 20 | strong_down | 7 | -0.121716 | -0.049509 | -0.0437419 | 0 | -0.00605287 |
| U1_liquid_tradable | M5_price_volume_only_logistic_top20 | 20 | strong_up | 7 | 0.092423 | 0.0110588 | -0.0065472 | 0.428571 | -0.027255 |
| U1_liquid_tradable | M5_price_volume_only_xgboost_excess | 20 | neutral | 25 | 0.00778426 | 0.00581698 | 0.00447 | 0.6 | 0.00135677 |
| U1_liquid_tradable | M5_price_volume_only_xgboost_excess | 20 | strong_down | 7 | -0.0672935 | 0.00491355 | 0.00206966 | 0.571429 | 0.00207186 |
| U1_liquid_tradable | M5_price_volume_only_xgboost_excess | 20 | strong_up | 7 | 0.109292 | 0.0279278 | -0.000653651 | 0.428571 | 0.0290982 |
| U1_liquid_tradable | M5_price_volume_only_xgboost_top20 | 20 | neutral | 25 | -0.00598118 | -0.00794846 | -0.0161431 | 0.4 | -0.0125192 |
| U1_liquid_tradable | M5_price_volume_only_xgboost_top20 | 20 | strong_down | 7 | -0.0606488 | 0.0115583 | 0.00183483 | 0.571429 | -0.000948609 |
| U1_liquid_tradable | M5_price_volume_only_xgboost_top20 | 20 | strong_up | 7 | 0.131566 | 0.0502023 | 0.0385171 | 0.857143 | 0.039461 |

## 口径

- `price_volume_only` 是 M5 内部 baseline；后续 spec 按 industry_breadth -> fund_flow -> fundamental -> shareholder 累积加入。
- `topk_excess_after_cost` 使用半 L1 月度换手乘以 cost_bps 的简化成本敏感性。

## 本轮产物

- `data/results/m14_lhb_delta_test_2026-05-07_summary.json`
- `data/results/m14_lhb_delta_test_2026-05-07_leaderboard.csv`
- `data/results/m14_lhb_delta_test_2026-05-07_incremental_delta.csv`
- `data/results/m14_lhb_delta_test_2026-05-07_monthly_long.csv`
- `data/results/m14_lhb_delta_test_2026-05-07_rank_ic.csv`
- `data/results/m14_lhb_delta_test_2026-05-07_quantile_spread.csv`
- `data/results/m14_lhb_delta_test_2026-05-07_feature_coverage.csv`
- `data/results/m14_lhb_delta_test_2026-05-07_feature_importance.csv`
- `data/results/m14_lhb_delta_test_2026-05-07_topk_holdings.csv`
- `data/results/m14_lhb_delta_test_2026-05-07_industry_exposure.csv`
- `data/results/m14_lhb_delta_test_2026-05-07_candidate_pool_width.csv`
- `data/results/m14_lhb_delta_test_2026-05-07_candidate_pool_reject_reason.csv`
- `data/results/m14_lhb_delta_test_2026-05-07_year_slice.csv`
- `data/results/m14_lhb_delta_test_2026-05-07_regime_slice.csv`
- `data/results/m14_lhb_delta_test_2026-05-07_market_states.csv`
- `data/results/m14_lhb_delta_test_2026-05-07_manifest.json`
