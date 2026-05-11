# Monthly Selection M5 Multisource

- 生成时间：`2026-05-11T09:53:52.086576+00:00` · 输出 stem：`w4_event_stage_a_baseline_2026-05-11`
- 数据集：`data/cache/monthly_selection_features.parquet` · 数据库：`data/market.duckdb`
- 有效标签月份：`63` · 单窗训练行上限：`0`

## Leaderboard

| candidate_pool_version | model | model_type | top_k | months | mean_topk_return | median_topk_return | topk_excess_mean | topk_excess_median | topk_hit_rate | topk_minus_nextk_mean | industry_neutral_topk_excess_mean | turnover_mean | cost_drag_mean | topk_excess_after_cost_mean | topk_excess_annualized | rank_ic_mean | rank_ic_std | rank_ic_months | rank_ic_ir | quantile_top_minus_bottom_mean | neutral_median_excess | strong_down_median_excess | strong_up_median_excess |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| U1_liquid_tradable | M5_price_volume_only_elasticnet_excess | elasticnet | 20 | 39 | 0.0207932 | 0.0169045 | 0.0178885 | 0.0155878 | 0.666667 | 0.0137585 | 0.0174349 | 0.963158 | 0.000963158 | 0.0169442 | 0.237094 | 0.103751 | 0.117364 | 39 | 0.88401 | 0.0154547 | 0.0199595 | 6.73254e-06 | 0.0172371 |
| U1_liquid_tradable | M5_price_volume_only_extratrees_excess | tree_sanity | 20 | 39 | 0.0171938 | 0.00802678 | 0.0142891 | 0.0102683 | 0.666667 | 0.00138728 | 0.0126803 | 0.951316 | 0.000951316 | 0.0129734 | 0.185608 | 0.103323 | 0.129463 | 39 | 0.79809 | 0.0155649 | 0.0102683 | 0.0175189 | 0.00618123 |
| U1_liquid_tradable | B0_market_ew | benchmark | 20 | 39 | 0.00290466 | 0.00265337 | 0 | 0 | 0 |  |  |  | 0 | 0 | 0 |  |  |  |  |  | 0 | 0 | 0 |
| U1_liquid_tradable | M5_price_volume_only_logistic_top20 | logistic_classifier | 20 | 39 | -0.0032163 | -0.00174597 | -0.00612096 | -0.0150829 | 0.333333 | -0.0108748 | -0.00555021 | 0.957895 | 0.000957895 | -0.00809585 | -0.0710285 | -0.0447992 | 0.175478 | 39 | -0.255298 | 0.000192016 | -0.00927874 | -0.0437419 | -0.0065472 |
| U2_risk_sane | M5_price_volume_only_extratrees_excess | tree_sanity | 20 | 39 | 0.0197052 | 0.0092702 | 0.0168005 | 0.0159295 | 0.564103 | 0.0165071 | 0.016898 | 0.971053 | 0.000971053 | 0.0152898 | 0.221319 | 0.0828294 | 0.129981 | 39 | 0.637241 | 0.0111553 | 0.00850468 | 0.0159295 | 0.0302217 |
| U2_risk_sane | M5_price_volume_only_elasticnet_excess | elasticnet | 20 | 39 | 0.017733 | 0.0170118 | 0.0148284 | 0.00686879 | 0.589744 | 0.0096347 | 0.0147005 | 0.968421 | 0.000968421 | 0.0138924 | 0.193194 | 0.082875 | 0.1224 | 39 | 0.677082 | 0.0116541 | 0.0225659 | -0.00533903 | -0.00214467 |
| U2_risk_sane | M5_price_volume_only_logistic_top20 | logistic_classifier | 20 | 39 | 0.00543749 | -0.00638424 | 0.00253284 | -0.00979723 | 0.435897 | 0.00916514 | 0.00237542 | 0.961842 | 0.000961842 | 0.000988202 | 0.030821 | -0.0390621 | 0.176906 | 39 | -0.220806 | 0.00199977 | -0.00334626 | -0.0299228 | 0.0489217 |
| U2_risk_sane | B0_market_ew | benchmark | 20 | 39 | 0.00290466 | 0.00265337 | 0 | 0 | 0 |  |  |  | 0 | 0 | 0 |  |  |  |  |  | 0 | 0 | 0 |
| U1_liquid_tradable | M5_price_volume_only_extratrees_excess | tree_sanity | 30 | 39 | 0.02023 | 0.00690797 | 0.0173253 | 0.0139519 | 0.641026 | 0.0123028 | 0.0149292 | 0.946491 | 0.000946491 | 0.0161592 | 0.228904 | 0.103323 | 0.129463 | 39 | 0.79809 | 0.0155649 | 0.0138135 | 0.0139519 | 0.0398128 |
| U1_liquid_tradable | M5_price_volume_only_elasticnet_excess | elasticnet | 30 | 39 | 0.0151418 | 0.0105646 | 0.0122372 | 0.00411101 | 0.538462 | 0.00642645 | 0.0116703 | 0.957895 | 0.000957895 | 0.0112963 | 0.157144 | 0.103751 | 0.117364 | 39 | 0.88401 | 0.0154547 | 0.0115903 | -0.00138867 | -0.000349478 |
| U1_liquid_tradable | B0_market_ew | benchmark | 30 | 39 | 0.00290466 | 0.00265337 | 0 | 0 | 0 |  |  |  | 0 | 0 | 0 |  |  |  |  |  | 0 | 0 | 0 |
| U1_liquid_tradable | M5_price_volume_only_logistic_top20 | logistic_classifier | 30 | 39 | 0.000934551 | 0.00501901 | -0.00197011 | -0.0147076 | 0.384615 | 0.00407112 | -0.00141536 | 0.932456 | 0.000932456 | -0.00397174 | -0.0233868 | -0.0447992 | 0.175478 | 39 | -0.255298 | 0.000192016 | -0.00146282 | -0.054074 | -0.0122098 |
| U2_risk_sane | M5_price_volume_only_extratrees_excess | tree_sanity | 30 | 39 | 0.0163384 | 0.013187 | 0.0134338 | 0.00573903 | 0.589744 | 0.0106851 | 0.0132876 | 0.964912 | 0.000964912 | 0.012008 | 0.173666 | 0.0828294 | 0.129981 | 39 | 0.637241 | 0.0111553 | 0.00573903 | 0.00386045 | 0.0253862 |
| U2_risk_sane | M5_price_volume_only_elasticnet_excess | elasticnet | 30 | 39 | 0.0150854 | 0.00364622 | 0.0121808 | 0.00454383 | 0.615385 | 0.00306652 | 0.011985 | 0.957895 | 0.000957895 | 0.0110583 | 0.15637 | 0.082875 | 0.1224 | 39 | 0.677082 | 0.0116541 | 0.00620218 | 0.0022594 | -0.0155237 |
| U2_risk_sane | B0_market_ew | benchmark | 30 | 39 | 0.00290466 | 0.00265337 | 0 | 0 | 0 |  |  |  | 0 | 0 | 0 |  |  |  |  |  | 0 | 0 | 0 |
| U2_risk_sane | M5_price_volume_only_logistic_top20 | logistic_classifier | 30 | 39 | 0.00304506 | -0.0165194 | 0.000140404 | -0.0125129 | 0.384615 | 0.00560429 | 0.00116413 | 0.948246 | 0.000948246 | -0.000747035 | 0.00168615 | -0.0390621 | 0.176906 | 39 | -0.220806 | 0.00199977 | -0.00217027 | -0.0384898 | 0.0377217 |
| U1_liquid_tradable | M5_price_volume_only_extratrees_excess | tree_sanity | 50 | 39 | 0.0157452 | 0.000443402 | 0.0128406 | 0.0116983 | 0.615385 | 0.00722615 | 0.01084 | 0.929474 | 0.000929474 | 0.0116322 | 0.165448 | 0.103323 | 0.129463 | 39 | 0.79809 | 0.0155649 | 0.0116983 | 0.00864781 | 0.0243339 |
| U1_liquid_tradable | M5_price_volume_only_elasticnet_excess | elasticnet | 50 | 39 | 0.0128839 | 0.0183645 | 0.00997923 | 0.00558392 | 0.564103 | 0.00386067 | 0.0096542 | 0.944211 | 0.000944211 | 0.00893755 | 0.126547 | 0.103751 | 0.117364 | 39 | 0.88401 | 0.0154547 | 0.00953003 | -0.00259528 | -0.0120033 |
| U1_liquid_tradable | B0_market_ew | benchmark | 50 | 39 | 0.00290466 | 0.00265337 | 0 | 0 | 0 |  |  |  | 0 | 0 | 0 |  |  |  |  |  | 0 | 0 | 0 |
| U1_liquid_tradable | M5_price_volume_only_logistic_top20 | logistic_classifier | 50 | 39 | -0.000916861 | -0.00244658 | -0.00382152 | -0.000923454 | 0.487179 | 0.00451387 | -0.00350416 | 0.905263 | 0.000905263 | -0.00501759 | -0.0449065 | -0.0447992 | 0.175478 | 39 | -0.255298 | 0.000192016 | 0.00310752 | -0.0384772 | 0.009859 |
| U2_risk_sane | M5_price_volume_only_elasticnet_excess | elasticnet | 50 | 39 | 0.0129873 | 0.00387014 | 0.0100826 | 0.00398661 | 0.589744 | 0.00712021 | 0.0102616 | 0.948947 | 0.000948947 | 0.00897414 | 0.127932 | 0.082875 | 0.1224 | 39 | 0.677082 | 0.0116541 | 0.0104184 | 0.00135757 | -0.0118722 |
| U2_risk_sane | M5_price_volume_only_extratrees_excess | tree_sanity | 50 | 39 | 0.0107611 | -0.00263385 | 0.00785649 | 0.00330677 | 0.589744 | 0.000793433 | 0.0078389 | 0.946316 | 0.000946316 | 0.0066642 | 0.0984603 | 0.0828294 | 0.129981 | 39 | 0.637241 | 0.0111553 | 0.00399711 | 0.000411902 | 0.0111217 |
| U2_risk_sane | B0_market_ew | benchmark | 50 | 39 | 0.00290466 | 0.00265337 | 0 | 0 | 0 |  |  |  | 0 | 0 | 0 |  |  |  |  |  | 0 | 0 | 0 |
| U2_risk_sane | M5_price_volume_only_logistic_top20 | logistic_classifier | 50 | 39 | -0.000795547 | 0.00120577 | -0.0037002 | -0.0050846 | 0.410256 | -0.00341196 | -0.00275655 | 0.92 | 0.00092 | -0.004203 | -0.0435099 | -0.0390621 | 0.176906 | 39 | -0.220806 | 0.00199977 | -0.0050846 | -0.0388024 | 0.00856399 |

## Incremental Delta vs Price-Volume

_无记录_

## Feature Coverage

_无记录_

## Year Slice

| candidate_pool_version | model | top_k | year | months | mean_topk_return | mean_topk_excess | median_topk_excess | hit_rate | mean_topk_minus_nextk |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| U1_liquid_tradable | B0_market_ew | 20 | 2023 | 12 | -0.0185141 | 0 | 0 | 0 |  |
| U1_liquid_tradable | B0_market_ew | 20 | 2024 | 12 | -0.00586991 | 0 | 0 | 0 |  |
| U1_liquid_tradable | B0_market_ew | 20 | 2025 | 12 | 0.0329754 | 0 | 0 | 0 |  |
| U1_liquid_tradable | B0_market_ew | 20 | 2026 | 3 | 0.00339507 | 0 | 0 | 0 |  |
| U1_liquid_tradable | B0_market_ew | 30 | 2023 | 12 | -0.0185141 | 0 | 0 | 0 |  |
| U1_liquid_tradable | B0_market_ew | 30 | 2024 | 12 | -0.00586991 | 0 | 0 | 0 |  |
| U1_liquid_tradable | B0_market_ew | 30 | 2025 | 12 | 0.0329754 | 0 | 0 | 0 |  |
| U1_liquid_tradable | B0_market_ew | 30 | 2026 | 3 | 0.00339507 | 0 | 0 | 0 |  |
| U1_liquid_tradable | B0_market_ew | 50 | 2023 | 12 | -0.0185141 | 0 | 0 | 0 |  |
| U1_liquid_tradable | B0_market_ew | 50 | 2024 | 12 | -0.00586991 | 0 | 0 | 0 |  |
| U1_liquid_tradable | B0_market_ew | 50 | 2025 | 12 | 0.0329754 | 0 | 0 | 0 |  |
| U1_liquid_tradable | B0_market_ew | 50 | 2026 | 3 | 0.00339507 | 0 | 0 | 0 |  |
| U1_liquid_tradable | M5_price_volume_only_elasticnet_excess | 20 | 2023 | 12 | -0.00410076 | 0.0144133 | 0.0118594 | 0.666667 | -0.00312713 |
| U1_liquid_tradable | M5_price_volume_only_elasticnet_excess | 20 | 2024 | 12 | 0.00511674 | 0.0109866 | 0.00471615 | 0.666667 | 0.0208991 |
| U1_liquid_tradable | M5_price_volume_only_elasticnet_excess | 20 | 2025 | 12 | 0.0667633 | 0.033788 | 0.0202027 | 0.75 | 0.029919 |
| U1_liquid_tradable | M5_price_volume_only_elasticnet_excess | 20 | 2026 | 3 | -0.000806072 | -0.00420114 | -0.00251698 | 0.333333 | -0.0119042 |
| U1_liquid_tradable | M5_price_volume_only_elasticnet_excess | 30 | 2023 | 12 | -0.0034308 | 0.0150833 | 0.00647271 | 0.583333 | 0.00212422 |
| U1_liquid_tradable | M5_price_volume_only_elasticnet_excess | 30 | 2024 | 12 | -0.00256799 | 0.00330192 | -0.00109149 | 0.416667 | 0.0117538 |
| U1_liquid_tradable | M5_price_volume_only_elasticnet_excess | 30 | 2025 | 12 | 0.0532711 | 0.0202958 | 0.018551 | 0.666667 | 0.00757913 |
| U1_liquid_tradable | M5_price_volume_only_elasticnet_excess | 30 | 2026 | 3 | 0.00775444 | 0.00435937 | -0.00386266 | 0.333333 | -0.00228475 |
| U1_liquid_tradable | M5_price_volume_only_elasticnet_excess | 50 | 2023 | 12 | -0.00271391 | 0.0158002 | 0.0102431 | 0.583333 | 0.0102634 |
| U1_liquid_tradable | M5_price_volume_only_elasticnet_excess | 50 | 2024 | 12 | -0.00835878 | -0.00248887 | -0.00765117 | 0.416667 | -0.00652327 |
| U1_liquid_tradable | M5_price_volume_only_elasticnet_excess | 50 | 2025 | 12 | 0.0509948 | 0.0180195 | 0.0131333 | 0.666667 | 0.0125019 |
| U1_liquid_tradable | M5_price_volume_only_elasticnet_excess | 50 | 2026 | 3 | 0.00780195 | 0.00440688 | 0.00953003 | 0.666667 | -0.0147795 |
| U1_liquid_tradable | M5_price_volume_only_extratrees_excess | 20 | 2023 | 12 | 0.00167838 | 0.0201925 | 0.0168347 | 0.75 | 0.0139437 |
| U1_liquid_tradable | M5_price_volume_only_extratrees_excess | 20 | 2024 | 12 | 0.00922111 | 0.015091 | 0.0122284 | 0.583333 | 0.000336514 |
| U1_liquid_tradable | M5_price_volume_only_extratrees_excess | 20 | 2025 | 12 | 0.0455205 | 0.0125452 | 0.00822474 | 0.666667 | -0.0124899 |
| U1_liquid_tradable | M5_price_volume_only_extratrees_excess | 20 | 2026 | 3 | -0.00216103 | -0.0055561 | 0.00104224 | 0.666667 | 0.0108733 |
| U1_liquid_tradable | M5_price_volume_only_extratrees_excess | 30 | 2023 | 12 | -0.00374319 | 0.0147709 | 0.00917672 | 0.666667 | 0.00594184 |
| U1_liquid_tradable | M5_price_volume_only_extratrees_excess | 30 | 2024 | 12 | 0.0145958 | 0.0204657 | 0.0198664 | 0.583333 | 0.0168008 |
| U1_liquid_tradable | M5_price_volume_only_extratrees_excess | 30 | 2025 | 12 | 0.0562309 | 0.0232555 | 0.0276364 | 0.666667 | 0.017682 |
| U1_liquid_tradable | M5_price_volume_only_extratrees_excess | 30 | 2026 | 3 | -0.0053444 | -0.00873947 | 0.00466217 | 0.666667 | -0.00176273 |
| U1_liquid_tradable | M5_price_volume_only_extratrees_excess | 50 | 2023 | 12 | -0.00587775 | 0.0126363 | 0.0106844 | 0.666667 | 0.00332351 |
| U1_liquid_tradable | M5_price_volume_only_extratrees_excess | 50 | 2024 | 12 | 0.00983461 | 0.0157045 | 0.0126471 | 0.583333 | 0.0163857 |
| U1_liquid_tradable | M5_price_volume_only_extratrees_excess | 50 | 2025 | 12 | 0.0489147 | 0.0159394 | 0.0197973 | 0.583333 | 0.00617373 |
| U1_liquid_tradable | M5_price_volume_only_extratrees_excess | 50 | 2026 | 3 | -0.00679855 | -0.0101936 | 0.00234571 | 0.666667 | -0.0095918 |
| U1_liquid_tradable | M5_price_volume_only_logistic_top20 | 20 | 2023 | 12 | -0.0341508 | -0.0156367 | -0.0131089 | 0.333333 | -0.0163869 |
| U1_liquid_tradable | M5_price_volume_only_logistic_top20 | 20 | 2024 | 12 | -0.0196706 | -0.0138007 | -0.0281047 | 0.333333 | -0.00291331 |
| U1_liquid_tradable | M5_price_volume_only_logistic_top20 | 20 | 2025 | 12 | 0.029817 | -0.00315836 | -0.0101145 | 0.333333 | -0.01176 |
| U1_liquid_tradable | M5_price_volume_only_logistic_top20 | 20 | 2026 | 3 | 0.0542057 | 0.0508106 | -0.0150829 | 0.333333 | -0.0171315 |

## Realized Market State Slice

| candidate_pool_version | model | top_k | realized_market_state | months | mean_topk_return | mean_topk_excess | median_topk_excess | hit_rate | mean_topk_minus_nextk |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| U1_liquid_tradable | B0_market_ew | 20 | neutral | 25 | 0.00196727 | 0 | 0 | 0 |  |
| U1_liquid_tradable | B0_market_ew | 20 | strong_down | 7 | -0.0722071 | 0 | 0 | 0 |  |
| U1_liquid_tradable | B0_market_ew | 20 | strong_up | 7 | 0.0813642 | 0 | 0 | 0 |  |
| U1_liquid_tradable | M5_price_volume_only_elasticnet_excess | 20 | neutral | 25 | 0.022316 | 0.0203488 | 0.0199595 | 0.72 | 0.0169158 |
| U1_liquid_tradable | M5_price_volume_only_elasticnet_excess | 20 | strong_down | 7 | -0.071877 | 0.000330136 | 6.73254e-06 | 0.571429 | -0.0106594 |
| U1_liquid_tradable | M5_price_volume_only_elasticnet_excess | 20 | strong_up | 7 | 0.108024 | 0.0266603 | 0.0172371 | 0.571429 | 0.0269 |
| U1_liquid_tradable | M5_price_volume_only_extratrees_excess | 20 | neutral | 25 | 0.00749222 | 0.00552495 | 0.0102683 | 0.64 | -0.0109263 |
| U1_liquid_tradable | M5_price_volume_only_extratrees_excess | 20 | strong_down | 7 | -0.0503783 | 0.0218288 | 0.0175189 | 0.714286 | 0.0239189 |
| U1_liquid_tradable | M5_price_volume_only_extratrees_excess | 20 | strong_up | 7 | 0.119414 | 0.0380501 | 0.00618123 | 0.714286 | 0.0228328 |
| U1_liquid_tradable | M5_price_volume_only_logistic_top20 | 20 | neutral | 25 | 0.00318464 | 0.00121737 | -0.00927874 | 0.4 | -0.00779209 |
| U1_liquid_tradable | M5_price_volume_only_logistic_top20 | 20 | strong_down | 7 | -0.121716 | -0.049509 | -0.0437419 | 0 | -0.00605287 |
| U1_liquid_tradable | M5_price_volume_only_logistic_top20 | 20 | strong_up | 7 | 0.092423 | 0.0110588 | -0.0065472 | 0.428571 | -0.0267064 |
| U2_risk_sane | B0_market_ew | 20 | neutral | 25 | 0.00196727 | 0 | 0 | 0 |  |
| U2_risk_sane | B0_market_ew | 20 | strong_down | 7 | -0.0722071 | 0 | 0 | 0 |  |
| U2_risk_sane | B0_market_ew | 20 | strong_up | 7 | 0.0813642 | 0 | 0 | 0 |  |
| U2_risk_sane | M5_price_volume_only_elasticnet_excess | 20 | neutral | 25 | 0.0201659 | 0.0181987 | 0.0225659 | 0.72 | 0.0110538 |
| U2_risk_sane | M5_price_volume_only_elasticnet_excess | 20 | strong_down | 7 | -0.0722781 | -7.10265e-05 | -0.00533903 | 0.285714 | -0.0127369 |
| U2_risk_sane | M5_price_volume_only_elasticnet_excess | 20 | strong_up | 7 | 0.0990552 | 0.017691 | -0.00214467 | 0.428571 | 0.026938 |
| U2_risk_sane | M5_price_volume_only_extratrees_excess | 20 | neutral | 25 | 0.01457 | 0.0126027 | 0.00850468 | 0.56 | 0.00775238 |
| U2_risk_sane | M5_price_volume_only_extratrees_excess | 20 | strong_down | 7 | -0.0520508 | 0.0201563 | 0.0159295 | 0.571429 | 0.0278794 |
| U2_risk_sane | M5_price_volume_only_extratrees_excess | 20 | strong_up | 7 | 0.109801 | 0.0284368 | 0.0302217 | 0.571429 | 0.0364016 |
| U2_risk_sane | M5_price_volume_only_logistic_top20 | 20 | neutral | 25 | 0.00939349 | 0.00742621 | -0.00334626 | 0.48 | 0.00935273 |
| U2_risk_sane | M5_price_volume_only_logistic_top20 | 20 | strong_down | 7 | -0.114475 | -0.0422683 | -0.0299228 | 0 | 0.00182321 |
| U2_risk_sane | M5_price_volume_only_logistic_top20 | 20 | strong_up | 7 | 0.111222 | 0.0298576 | 0.0489217 | 0.714286 | 0.0158371 |
| U1_liquid_tradable | B0_market_ew | 30 | neutral | 25 | 0.00196727 | 0 | 0 | 0 |  |
| U1_liquid_tradable | B0_market_ew | 30 | strong_down | 7 | -0.0722071 | 0 | 0 | 0 |  |
| U1_liquid_tradable | B0_market_ew | 30 | strong_up | 7 | 0.0813642 | 0 | 0 | 0 |  |
| U1_liquid_tradable | M5_price_volume_only_elasticnet_excess | 30 | neutral | 25 | 0.0153187 | 0.0133514 | 0.0115903 | 0.68 | 0.00698717 |
| U1_liquid_tradable | M5_price_volume_only_elasticnet_excess | 30 | strong_down | 7 | -0.0679811 | 0.00422603 | -0.00138867 | 0.142857 | -0.00518236 |
| U1_liquid_tradable | M5_price_volume_only_elasticnet_excess | 30 | strong_up | 7 | 0.0976331 | 0.0162689 | -0.000349478 | 0.428571 | 0.0160327 |
| U1_liquid_tradable | M5_price_volume_only_extratrees_excess | 30 | neutral | 25 | 0.0141924 | 0.0122251 | 0.0138135 | 0.6 | 0.00388619 |
| U1_liquid_tradable | M5_price_volume_only_extratrees_excess | 30 | strong_down | 7 | -0.0560983 | 0.0161088 | 0.0139519 | 0.714286 | 0.0306402 |
| U1_liquid_tradable | M5_price_volume_only_extratrees_excess | 30 | strong_up | 7 | 0.118121 | 0.0367568 | 0.0398128 | 0.714286 | 0.0240246 |
| U1_liquid_tradable | M5_price_volume_only_logistic_top20 | 30 | neutral | 25 | 0.0100182 | 0.00805092 | -0.00146282 | 0.48 | 0.0130869 |
| U1_liquid_tradable | M5_price_volume_only_logistic_top20 | 30 | strong_down | 7 | -0.123048 | -0.0508409 | -0.054074 | 0 | -0.0162154 |
| U1_liquid_tradable | M5_price_volume_only_logistic_top20 | 30 | strong_up | 7 | 0.0924755 | 0.0111113 | -0.0122098 | 0.428571 | -0.00784171 |
| U2_risk_sane | B0_market_ew | 30 | neutral | 25 | 0.00196727 | 0 | 0 | 0 |  |
| U2_risk_sane | B0_market_ew | 30 | strong_down | 7 | -0.0722071 | 0 | 0 | 0 |  |
| U2_risk_sane | B0_market_ew | 30 | strong_up | 7 | 0.0813642 | 0 | 0 | 0 |  |
| U2_risk_sane | M5_price_volume_only_elasticnet_excess | 30 | neutral | 25 | 0.0158999 | 0.0139326 | 0.00620218 | 0.68 | 0.00455832 |

## 口径

- `price_volume_only` 是 M5 内部 baseline；后续 spec 按 industry_breadth -> fund_flow -> fundamental -> shareholder 累积加入。
- `topk_excess_after_cost` 使用半 L1 月度换手乘以 cost_bps 的简化成本敏感性。

## 本轮产物

- `data/results/w4_event_stage_a_baseline_2026-05-11_summary.json`
- `data/results/w4_event_stage_a_baseline_2026-05-11_leaderboard.csv`
- `data/results/w4_event_stage_a_baseline_2026-05-11_incremental_delta.csv`
- `data/results/w4_event_stage_a_baseline_2026-05-11_monthly_long.csv`
- `data/results/w4_event_stage_a_baseline_2026-05-11_rank_ic.csv`
- `data/results/w4_event_stage_a_baseline_2026-05-11_quantile_spread.csv`
- `data/results/w4_event_stage_a_baseline_2026-05-11_feature_coverage.csv`
- `data/results/w4_event_stage_a_baseline_2026-05-11_feature_importance.csv`
- `data/results/w4_event_stage_a_baseline_2026-05-11_topk_holdings.csv`
- `data/results/w4_event_stage_a_baseline_2026-05-11_industry_exposure.csv`
- `data/results/w4_event_stage_a_baseline_2026-05-11_candidate_pool_width.csv`
- `data/results/w4_event_stage_a_baseline_2026-05-11_candidate_pool_reject_reason.csv`
- `data/results/w4_event_stage_a_baseline_2026-05-11_year_slice.csv`
- `data/results/w4_event_stage_a_baseline_2026-05-11_regime_slice.csv`
- `data/results/w4_event_stage_a_baseline_2026-05-11_market_states.csv`
- `data/results/w4_event_stage_a_baseline_2026-05-11_manifest.json`
