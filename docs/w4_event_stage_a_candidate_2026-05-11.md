# Monthly Selection M5 Multisource

- 生成时间：`2026-05-11T09:56:46.679768+00:00` · 输出 stem：`w4_event_stage_a_candidate_2026-05-11`
- 数据集：`data/cache/monthly_selection_features.parquet` · 数据库：`data/market.duckdb`
- 有效标签月份：`63` · 单窗训练行上限：`0`

## Leaderboard

| candidate_pool_version | model | model_type | top_k | months | mean_topk_return | median_topk_return | topk_excess_mean | topk_excess_median | topk_hit_rate | topk_minus_nextk_mean | industry_neutral_topk_excess_mean | turnover_mean | cost_drag_mean | topk_excess_after_cost_mean | topk_excess_annualized | rank_ic_mean | rank_ic_std | rank_ic_months | rank_ic_ir | quantile_top_minus_bottom_mean | neutral_median_excess | strong_down_median_excess | strong_up_median_excess |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| U1_liquid_tradable | M5_plus_event_elasticnet_excess | elasticnet | 20 | 39 | 0.0219599 | 0.0153766 | 0.0190552 | 0.0128466 | 0.615385 | 0.0165108 | 0.0183951 | 0.964474 | 0.000964474 | 0.0177235 | 0.254217 | 0.102751 | 0.116289 | 39 | 0.883583 | 0.014999 | 0.0209307 | -0.00667486 | 0.018265 |
| U1_liquid_tradable | M5_plus_event_extratrees_excess | tree_sanity | 20 | 39 | 0.0220059 | 0.0125723 | 0.0191013 | 0.0118364 | 0.589744 | 0.00982146 | 0.0173891 | 0.960526 | 0.000960526 | 0.017641 | 0.254897 | 0.103207 | 0.130283 | 39 | 0.792177 | 0.0145675 | 0.0146277 | 0.0118364 | 0.00600528 |
| U1_liquid_tradable | M5_price_volume_only_elasticnet_excess | elasticnet | 20 | 39 | 0.0207932 | 0.0169045 | 0.0178885 | 0.0155878 | 0.666667 | 0.0137585 | 0.0174349 | 0.963158 | 0.000963158 | 0.0169442 | 0.237094 | 0.103751 | 0.117364 | 39 | 0.88401 | 0.0154547 | 0.0199595 | 6.73254e-06 | 0.0172371 |
| U1_liquid_tradable | M5_price_volume_only_extratrees_excess | tree_sanity | 20 | 39 | 0.0171938 | 0.00802678 | 0.0142891 | 0.0102683 | 0.666667 | 0.00138728 | 0.0126803 | 0.951316 | 0.000951316 | 0.0129734 | 0.185608 | 0.103323 | 0.129463 | 39 | 0.79809 | 0.0155649 | 0.0102683 | 0.0175189 | 0.00618123 |
| U1_liquid_tradable | B0_market_ew | benchmark | 20 | 39 | 0.00290466 | 0.00265337 | 0 | 0 | 0 |  |  |  | 0 | 0 | 0 |  |  |  |  |  | 0 | 0 | 0 |
| U1_liquid_tradable | M5_price_volume_only_logistic_top20 | logistic_classifier | 20 | 39 | -0.0032163 | -0.00174597 | -0.00612096 | -0.0150829 | 0.333333 | -0.0108748 | -0.00555021 | 0.957895 | 0.000957895 | -0.00809585 | -0.0710285 | -0.0447992 | 0.175478 | 39 | -0.255298 | 0.000192016 | -0.00927874 | -0.0437419 | -0.0065472 |
| U1_liquid_tradable | M5_plus_event_logistic_top20 | logistic_classifier | 20 | 39 | -0.00416147 | -0.0113268 | -0.00706612 | -0.0169199 | 0.333333 | -0.00747294 | -0.00671013 | 0.953947 | 0.000953947 | -0.00922592 | -0.0815745 | -0.0454262 | 0.174637 | 39 | -0.260118 | -0.00044688 | -0.0143796 | -0.0424999 | -0.0106234 |
| U2_risk_sane | M5_plus_event_extratrees_excess | tree_sanity | 20 | 39 | 0.0199204 | 0.0231899 | 0.0170157 | 0.00538169 | 0.589744 | 0.011929 | 0.0149772 | 0.973684 | 0.000973684 | 0.015769 | 0.224424 | 0.0830812 | 0.129133 | 39 | 0.643376 | 0.0121838 | 0.00747342 | -0.00473187 | 0.00450937 |
| U2_risk_sane | M5_price_volume_only_extratrees_excess | tree_sanity | 20 | 39 | 0.0197052 | 0.0092702 | 0.0168005 | 0.0159295 | 0.564103 | 0.0165071 | 0.016898 | 0.971053 | 0.000971053 | 0.0152898 | 0.221319 | 0.0828294 | 0.129981 | 39 | 0.637241 | 0.0111553 | 0.00850468 | 0.0159295 | 0.0302217 |
| U2_risk_sane | M5_plus_event_elasticnet_excess | elasticnet | 20 | 39 | 0.0183762 | 0.0144659 | 0.0154716 | 0.00686879 | 0.564103 | 0.00915786 | 0.0146075 | 0.969737 | 0.000969737 | 0.014488 | 0.202301 | 0.0818031 | 0.121572 | 39 | 0.672878 | 0.0109067 | 0.0169727 | -0.00993887 | -0.00819423 |
| U2_risk_sane | M5_price_volume_only_elasticnet_excess | elasticnet | 20 | 39 | 0.017733 | 0.0170118 | 0.0148284 | 0.00686879 | 0.589744 | 0.0096347 | 0.0147005 | 0.968421 | 0.000968421 | 0.0138924 | 0.193194 | 0.082875 | 0.1224 | 39 | 0.677082 | 0.0116541 | 0.0225659 | -0.00533903 | -0.00214467 |
| U2_risk_sane | M5_price_volume_only_logistic_top20 | logistic_classifier | 20 | 39 | 0.00543749 | -0.00638424 | 0.00253284 | -0.00979723 | 0.435897 | 0.00916514 | 0.00237542 | 0.961842 | 0.000961842 | 0.000988202 | 0.030821 | -0.0390621 | 0.176906 | 39 | -0.220806 | 0.00199977 | -0.00334626 | -0.0299228 | 0.0489217 |
| U2_risk_sane | B0_market_ew | benchmark | 20 | 39 | 0.00290466 | 0.00265337 | 0 | 0 | 0 |  |  |  | 0 | 0 | 0 |  |  |  |  |  | 0 | 0 | 0 |
| U2_risk_sane | M5_plus_event_logistic_top20 | logistic_classifier | 20 | 39 | 0.003074 | -0.010458 | 0.000169338 | -0.0133383 | 0.410256 | 0.00991521 | 5.85795e-05 | 0.961842 | 0.000961842 | -0.00155787 | 0.00203395 | -0.0398034 | 0.176174 | 39 | -0.225932 | 0.00135367 | -0.00257002 | -0.0327271 | 0.0389258 |
| U1_liquid_tradable | M5_price_volume_only_extratrees_excess | tree_sanity | 30 | 39 | 0.02023 | 0.00690797 | 0.0173253 | 0.0139519 | 0.641026 | 0.0123028 | 0.0149292 | 0.946491 | 0.000946491 | 0.0161592 | 0.228904 | 0.103323 | 0.129463 | 39 | 0.79809 | 0.0155649 | 0.0138135 | 0.0139519 | 0.0398128 |
| U1_liquid_tradable | M5_plus_event_extratrees_excess | tree_sanity | 30 | 39 | 0.0169224 | 0.011587 | 0.0140178 | 0.00214822 | 0.538462 | 0.00296722 | 0.0119244 | 0.950877 | 0.000950877 | 0.0128878 | 0.181808 | 0.103207 | 0.130283 | 39 | 0.792177 | 0.0145675 | 0.000607475 | 0.00396242 | -0.00578323 |
| U1_liquid_tradable | M5_price_volume_only_elasticnet_excess | elasticnet | 30 | 39 | 0.0151418 | 0.0105646 | 0.0122372 | 0.00411101 | 0.538462 | 0.00642645 | 0.0116703 | 0.957895 | 0.000957895 | 0.0112963 | 0.157144 | 0.103751 | 0.117364 | 39 | 0.88401 | 0.0154547 | 0.0115903 | -0.00138867 | -0.000349478 |
| U1_liquid_tradable | M5_plus_event_elasticnet_excess | elasticnet | 30 | 39 | 0.0150786 | 0.010847 | 0.012174 | 0.00398578 | 0.589744 | 0.00582025 | 0.0112154 | 0.954386 | 0.000954386 | 0.0111291 | 0.156277 | 0.102751 | 0.116289 | 39 | 0.883583 | 0.014999 | 0.0101585 | -0.00178397 | -0.010108 |
| U1_liquid_tradable | B0_market_ew | benchmark | 30 | 39 | 0.00290466 | 0.00265337 | 0 | 0 | 0 |  |  |  | 0 | 0 | 0 |  |  |  |  |  | 0 | 0 | 0 |
| U1_liquid_tradable | M5_plus_event_logistic_top20 | logistic_classifier | 30 | 39 | 0.00126468 | 0.00527729 | -0.00163998 | -0.0191792 | 0.461538 | 0.00829152 | -0.000831872 | 0.932456 | 0.000932456 | -0.00353641 | -0.0195032 | -0.0454262 | 0.174637 | 39 | -0.260118 | -0.00044688 | 0.00398343 | -0.0420557 | 0.0137086 |
| U1_liquid_tradable | M5_price_volume_only_logistic_top20 | logistic_classifier | 30 | 39 | 0.000934551 | 0.00501901 | -0.00197011 | -0.0147076 | 0.384615 | 0.00407112 | -0.00141536 | 0.932456 | 0.000932456 | -0.00397174 | -0.0233868 | -0.0447992 | 0.175478 | 39 | -0.255298 | 0.000192016 | -0.00146282 | -0.054074 | -0.0122098 |
| U2_risk_sane | M5_price_volume_only_extratrees_excess | tree_sanity | 30 | 39 | 0.0163384 | 0.013187 | 0.0134338 | 0.00573903 | 0.589744 | 0.0106851 | 0.0132876 | 0.964912 | 0.000964912 | 0.012008 | 0.173666 | 0.0828294 | 0.129981 | 39 | 0.637241 | 0.0111553 | 0.00573903 | 0.00386045 | 0.0253862 |
| U2_risk_sane | M5_plus_event_extratrees_excess | tree_sanity | 30 | 39 | 0.0154 | 0.0194547 | 0.0124953 | 0.00337663 | 0.538462 | 0.00544189 | 0.0118709 | 0.965789 | 0.000965789 | 0.0111288 | 0.16069 | 0.0830812 | 0.129133 | 39 | 0.643376 | 0.0121838 | 0.0032521 | 0.00337663 | 0.00484669 |
| U2_risk_sane | M5_price_volume_only_elasticnet_excess | elasticnet | 30 | 39 | 0.0150854 | 0.00364622 | 0.0121808 | 0.00454383 | 0.615385 | 0.00306652 | 0.011985 | 0.957895 | 0.000957895 | 0.0110583 | 0.15637 | 0.082875 | 0.1224 | 39 | 0.677082 | 0.0116541 | 0.00620218 | 0.0022594 | -0.0155237 |
| U2_risk_sane | M5_plus_event_elasticnet_excess | elasticnet | 30 | 39 | 0.0141072 | 0.00671385 | 0.0112025 | 0.005339 | 0.615385 | 0.00168923 | 0.0111289 | 0.961404 | 0.000961404 | 0.00998269 | 0.14303 | 0.0818031 | 0.121572 | 39 | 0.672878 | 0.0109067 | 0.00707935 | 0.00116101 | -0.00973215 |
| U2_risk_sane | B0_market_ew | benchmark | 30 | 39 | 0.00290466 | 0.00265337 | 0 | 0 | 0 |  |  |  | 0 | 0 | 0 |  |  |  |  |  | 0 | 0 | 0 |
| U2_risk_sane | M5_price_volume_only_logistic_top20 | logistic_classifier | 30 | 39 | 0.00304506 | -0.0165194 | 0.000140404 | -0.0125129 | 0.384615 | 0.00560429 | 0.00116413 | 0.948246 | 0.000948246 | -0.000747035 | 0.00168615 | -0.0390621 | 0.176906 | 39 | -0.220806 | 0.00199977 | -0.00217027 | -0.0384898 | 0.0377217 |
| U2_risk_sane | M5_plus_event_logistic_top20 | logistic_classifier | 30 | 39 | -0.000158864 | -0.0140985 | -0.00306352 | -0.0159171 | 0.384615 | 0.00174992 | -0.00225718 | 0.941228 | 0.000941228 | -0.00386875 | -0.0361491 | -0.0398034 | 0.176174 | 39 | -0.225932 | 0.00135367 | -0.00823142 | -0.0430792 | 0.0303207 |
| U1_liquid_tradable | M5_plus_event_extratrees_excess | tree_sanity | 50 | 39 | 0.0161831 | 0.0214567 | 0.0132785 | 0.00510069 | 0.564103 | 0.010269 | 0.0114438 | 0.938947 | 0.000938947 | 0.0122539 | 0.17151 | 0.103207 | 0.130283 | 39 | 0.792177 | 0.0145675 | 0.0148696 | -0.00128869 | -0.0090922 |
| U1_liquid_tradable | M5_price_volume_only_extratrees_excess | tree_sanity | 50 | 39 | 0.0157452 | 0.000443402 | 0.0128406 | 0.0116983 | 0.615385 | 0.00722615 | 0.01084 | 0.929474 | 0.000929474 | 0.0116322 | 0.165448 | 0.103323 | 0.129463 | 39 | 0.79809 | 0.0155649 | 0.0116983 | 0.00864781 | 0.0243339 |
| U1_liquid_tradable | M5_plus_event_elasticnet_excess | elasticnet | 50 | 39 | 0.013844 | 0.0242847 | 0.0109393 | 0.00515381 | 0.615385 | 0.00792839 | 0.0103761 | 0.944737 | 0.000944737 | 0.0100491 | 0.139465 | 0.102751 | 0.116289 | 39 | 0.883583 | 0.014999 | 0.00886923 | 0.00282061 | -0.0134446 |
| U1_liquid_tradable | M5_price_volume_only_elasticnet_excess | elasticnet | 50 | 39 | 0.0128839 | 0.0183645 | 0.00997923 | 0.00558392 | 0.564103 | 0.00386067 | 0.0096542 | 0.944211 | 0.000944211 | 0.00893755 | 0.126547 | 0.103751 | 0.117364 | 39 | 0.88401 | 0.0154547 | 0.00953003 | -0.00259528 | -0.0120033 |
| U1_liquid_tradable | B0_market_ew | benchmark | 50 | 39 | 0.00290466 | 0.00265337 | 0 | 0 | 0 |  |  |  | 0 | 0 | 0 |  |  |  |  |  | 0 | 0 | 0 |
| U1_liquid_tradable | M5_price_volume_only_logistic_top20 | logistic_classifier | 50 | 39 | -0.000916861 | -0.00244658 | -0.00382152 | -0.000923454 | 0.487179 | 0.00451387 | -0.00350416 | 0.905263 | 0.000905263 | -0.00501759 | -0.0449065 | -0.0447992 | 0.175478 | 39 | -0.255298 | 0.000192016 | 0.00310752 | -0.0384772 | 0.009859 |
| U1_liquid_tradable | M5_plus_event_logistic_top20 | logistic_classifier | 50 | 39 | -0.00239029 | -0.0067016 | -0.00529494 | -0.00652093 | 0.461538 | 0.00145327 | -0.00440319 | 0.904211 | 0.000904211 | -0.00652874 | -0.0617212 | -0.0454262 | 0.174637 | 39 | -0.260118 | -0.00044688 | 0.00322668 | -0.0398228 | 0.00763611 |
| U2_risk_sane | M5_price_volume_only_elasticnet_excess | elasticnet | 50 | 39 | 0.0129873 | 0.00387014 | 0.0100826 | 0.00398661 | 0.589744 | 0.00712021 | 0.0102616 | 0.948947 | 0.000948947 | 0.00897414 | 0.127932 | 0.082875 | 0.1224 | 39 | 0.677082 | 0.0116541 | 0.0104184 | 0.00135757 | -0.0118722 |
| U2_risk_sane | M5_plus_event_elasticnet_excess | elasticnet | 50 | 39 | 0.012726 | 0.00380951 | 0.00982139 | 0.00494451 | 0.641026 | 0.00625944 | 0.00993886 | 0.946842 | 0.000946842 | 0.00892773 | 0.124436 | 0.0818031 | 0.121572 | 39 | 0.672878 | 0.0109067 | 0.00494451 | 0.00575904 | -0.0100188 |
| U2_risk_sane | M5_plus_event_extratrees_excess | tree_sanity | 50 | 39 | 0.0130075 | 0.0124005 | 0.0101029 | 0.01108 | 0.615385 | 0.00511575 | 0.00951064 | 0.950526 | 0.000950526 | 0.00888211 | 0.128203 | 0.0830812 | 0.129133 | 39 | 0.643376 | 0.0121838 | 0.0185582 | 0.00566645 | -0.00592637 |
| U2_risk_sane | M5_price_volume_only_extratrees_excess | tree_sanity | 50 | 39 | 0.0107611 | -0.00263385 | 0.00785649 | 0.00330677 | 0.589744 | 0.000793433 | 0.0078389 | 0.946316 | 0.000946316 | 0.0066642 | 0.0984603 | 0.0828294 | 0.129981 | 39 | 0.637241 | 0.0111553 | 0.00399711 | 0.000411902 | 0.0111217 |
| U2_risk_sane | B0_market_ew | benchmark | 50 | 39 | 0.00290466 | 0.00265337 | 0 | 0 | 0 |  |  |  | 0 | 0 | 0 |  |  |  |  |  | 0 | 0 | 0 |

_仅展示前 40 行，共 42 行。_

## Incremental Delta vs Price-Volume

| candidate_pool_version | model | model_type | top_k | months | mean_topk_return | median_topk_return | topk_excess_mean | topk_excess_median | topk_hit_rate | topk_minus_nextk_mean | industry_neutral_topk_excess_mean | turnover_mean | cost_drag_mean | topk_excess_after_cost_mean | topk_excess_annualized | rank_ic_mean | rank_ic_std | rank_ic_months | rank_ic_ir | quantile_top_minus_bottom_mean | neutral_median_excess | strong_down_median_excess | strong_up_median_excess | feature_spec | base_model | baseline_topk_excess_mean | baseline_topk_excess_after_cost_mean | baseline_rank_ic_mean | baseline_quantile_top_minus_bottom_mean | delta_topk_excess_mean | delta_topk_excess_after_cost_mean | delta_rank_ic_mean | delta_quantile_top_minus_bottom_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| U1_liquid_tradable | M5_plus_event_extratrees_excess | tree_sanity | 20 | 39 | 0.0220059 | 0.0125723 | 0.0191013 | 0.0118364 | 0.589744 | 0.00982146 | 0.0173891 | 0.960526 | 0.000960526 | 0.017641 | 0.254897 | 0.103207 | 0.130283 | 39 | 0.792177 | 0.0145675 | 0.0146277 | 0.0118364 | 0.00600528 | plus_event | extratrees_excess | 0.0142891 | 0.0129734 | 0.103323 | 0.0155649 | 0.00481215 | 0.00466766 | -0.000116515 | -0.000997442 |
| U1_liquid_tradable | M5_plus_event_elasticnet_excess | elasticnet | 20 | 39 | 0.0219599 | 0.0153766 | 0.0190552 | 0.0128466 | 0.615385 | 0.0165108 | 0.0183951 | 0.964474 | 0.000964474 | 0.0177235 | 0.254217 | 0.102751 | 0.116289 | 39 | 0.883583 | 0.014999 | 0.0209307 | -0.00667486 | 0.018265 | plus_event | elasticnet_excess | 0.0178885 | 0.0169442 | 0.103751 | 0.0154547 | 0.0011667 | 0.00077926 | -0.000999804 | -0.00045574 |
| U1_liquid_tradable | M5_plus_event_logistic_top20 | logistic_classifier | 20 | 39 | -0.00416147 | -0.0113268 | -0.00706612 | -0.0169199 | 0.333333 | -0.00747294 | -0.00671013 | 0.953947 | 0.000953947 | -0.00922592 | -0.0815745 | -0.0454262 | 0.174637 | 39 | -0.260118 | -0.00044688 | -0.0143796 | -0.0424999 | -0.0106234 | plus_event | logistic_top20 | -0.00612096 | -0.00809585 | -0.0447992 | 0.000192016 | -0.000945168 | -0.00113007 | -0.00062692 | -0.000638896 |
| U2_risk_sane | M5_plus_event_elasticnet_excess | elasticnet | 20 | 39 | 0.0183762 | 0.0144659 | 0.0154716 | 0.00686879 | 0.564103 | 0.00915786 | 0.0146075 | 0.969737 | 0.000969737 | 0.014488 | 0.202301 | 0.0818031 | 0.121572 | 39 | 0.672878 | 0.0109067 | 0.0169727 | -0.00993887 | -0.00819423 | plus_event | elasticnet_excess | 0.0148284 | 0.0138924 | 0.082875 | 0.0116541 | 0.000643195 | 0.000595667 | -0.00107191 | -0.000747375 |
| U2_risk_sane | M5_plus_event_extratrees_excess | tree_sanity | 20 | 39 | 0.0199204 | 0.0231899 | 0.0170157 | 0.00538169 | 0.589744 | 0.011929 | 0.0149772 | 0.973684 | 0.000973684 | 0.015769 | 0.224424 | 0.0830812 | 0.129133 | 39 | 0.643376 | 0.0121838 | 0.00747342 | -0.00473187 | 0.00450937 | plus_event | extratrees_excess | 0.0168005 | 0.0152898 | 0.0828294 | 0.0111553 | 0.000215215 | 0.000479241 | 0.000251778 | 0.00102847 |
| U2_risk_sane | M5_plus_event_logistic_top20 | logistic_classifier | 20 | 39 | 0.003074 | -0.010458 | 0.000169338 | -0.0133383 | 0.410256 | 0.00991521 | 5.85795e-05 | 0.961842 | 0.000961842 | -0.00155787 | 0.00203395 | -0.0398034 | 0.176174 | 39 | -0.225932 | 0.00135367 | -0.00257002 | -0.0327271 | 0.0389258 | plus_event | logistic_top20 | 0.00253284 | 0.000988202 | -0.0390621 | 0.00199977 | -0.0023635 | -0.00254607 | -0.000741299 | -0.000646097 |
| U1_liquid_tradable | M5_plus_event_logistic_top20 | logistic_classifier | 30 | 39 | 0.00126468 | 0.00527729 | -0.00163998 | -0.0191792 | 0.461538 | 0.00829152 | -0.000831872 | 0.932456 | 0.000932456 | -0.00353641 | -0.0195032 | -0.0454262 | 0.174637 | 39 | -0.260118 | -0.00044688 | 0.00398343 | -0.0420557 | 0.0137086 | plus_event | logistic_top20 | -0.00197011 | -0.00397174 | -0.0447992 | 0.000192016 | 0.00033013 | 0.000435331 | -0.00062692 | -0.000638896 |
| U1_liquid_tradable | M5_plus_event_elasticnet_excess | elasticnet | 30 | 39 | 0.0150786 | 0.010847 | 0.012174 | 0.00398578 | 0.589744 | 0.00582025 | 0.0112154 | 0.954386 | 0.000954386 | 0.0111291 | 0.156277 | 0.102751 | 0.116289 | 39 | 0.883583 | 0.014999 | 0.0101585 | -0.00178397 | -0.010108 | plus_event | elasticnet_excess | 0.0122372 | 0.0112963 | 0.103751 | 0.0154547 | -6.31962e-05 | -0.000167171 | -0.000999804 | -0.00045574 |
| U1_liquid_tradable | M5_plus_event_extratrees_excess | tree_sanity | 30 | 39 | 0.0169224 | 0.011587 | 0.0140178 | 0.00214822 | 0.538462 | 0.00296722 | 0.0119244 | 0.950877 | 0.000950877 | 0.0128878 | 0.181808 | 0.103207 | 0.130283 | 39 | 0.792177 | 0.0145675 | 0.000607475 | 0.00396242 | -0.00578323 | plus_event | extratrees_excess | 0.0173253 | 0.0161592 | 0.103323 | 0.0155649 | -0.00330752 | -0.00327142 | -0.000116515 | -0.000997442 |
| U2_risk_sane | M5_plus_event_extratrees_excess | tree_sanity | 30 | 39 | 0.0154 | 0.0194547 | 0.0124953 | 0.00337663 | 0.538462 | 0.00544189 | 0.0118709 | 0.965789 | 0.000965789 | 0.0111288 | 0.16069 | 0.0830812 | 0.129133 | 39 | 0.643376 | 0.0121838 | 0.0032521 | 0.00337663 | 0.00484669 | plus_event | extratrees_excess | 0.0134338 | 0.012008 | 0.0828294 | 0.0111553 | -0.000938447 | -0.000879153 | 0.000251778 | 0.00102847 |
| U2_risk_sane | M5_plus_event_elasticnet_excess | elasticnet | 30 | 39 | 0.0141072 | 0.00671385 | 0.0112025 | 0.005339 | 0.615385 | 0.00168923 | 0.0111289 | 0.961404 | 0.000961404 | 0.00998269 | 0.14303 | 0.0818031 | 0.121572 | 39 | 0.672878 | 0.0109067 | 0.00707935 | 0.00116101 | -0.00973215 | plus_event | elasticnet_excess | 0.0121808 | 0.0110583 | 0.082875 | 0.0116541 | -0.000978257 | -0.0010756 | -0.00107191 | -0.000747375 |
| U2_risk_sane | M5_plus_event_logistic_top20 | logistic_classifier | 30 | 39 | -0.000158864 | -0.0140985 | -0.00306352 | -0.0159171 | 0.384615 | 0.00174992 | -0.00225718 | 0.941228 | 0.000941228 | -0.00386875 | -0.0361491 | -0.0398034 | 0.176174 | 39 | -0.225932 | 0.00135367 | -0.00823142 | -0.0430792 | 0.0303207 | plus_event | logistic_top20 | 0.000140404 | -0.000747035 | -0.0390621 | 0.00199977 | -0.00320393 | -0.00312172 | -0.000741299 | -0.000646097 |
| U1_liquid_tradable | M5_plus_event_elasticnet_excess | elasticnet | 50 | 39 | 0.013844 | 0.0242847 | 0.0109393 | 0.00515381 | 0.615385 | 0.00792839 | 0.0103761 | 0.944737 | 0.000944737 | 0.0100491 | 0.139465 | 0.102751 | 0.116289 | 39 | 0.883583 | 0.014999 | 0.00886923 | 0.00282061 | -0.0134446 | plus_event | elasticnet_excess | 0.00997923 | 0.00893755 | 0.103751 | 0.0154547 | 0.00096008 | 0.00111151 | -0.000999804 | -0.00045574 |
| U1_liquid_tradable | M5_plus_event_extratrees_excess | tree_sanity | 50 | 39 | 0.0161831 | 0.0214567 | 0.0132785 | 0.00510069 | 0.564103 | 0.010269 | 0.0114438 | 0.938947 | 0.000938947 | 0.0122539 | 0.17151 | 0.103207 | 0.130283 | 39 | 0.792177 | 0.0145675 | 0.0148696 | -0.00128869 | -0.0090922 | plus_event | extratrees_excess | 0.0128406 | 0.0116322 | 0.103323 | 0.0155649 | 0.000437912 | 0.000621721 | -0.000116515 | -0.000997442 |
| U1_liquid_tradable | M5_plus_event_logistic_top20 | logistic_classifier | 50 | 39 | -0.00239029 | -0.0067016 | -0.00529494 | -0.00652093 | 0.461538 | 0.00145327 | -0.00440319 | 0.904211 | 0.000904211 | -0.00652874 | -0.0617212 | -0.0454262 | 0.174637 | 39 | -0.260118 | -0.00044688 | 0.00322668 | -0.0398228 | 0.00763611 | plus_event | logistic_top20 | -0.00382152 | -0.00501759 | -0.0447992 | 0.000192016 | -0.00147343 | -0.00151115 | -0.00062692 | -0.000638896 |
| U2_risk_sane | M5_plus_event_extratrees_excess | tree_sanity | 50 | 39 | 0.0130075 | 0.0124005 | 0.0101029 | 0.01108 | 0.615385 | 0.00511575 | 0.00951064 | 0.950526 | 0.000950526 | 0.00888211 | 0.128203 | 0.0830812 | 0.129133 | 39 | 0.643376 | 0.0121838 | 0.0185582 | 0.00566645 | -0.00592637 | plus_event | extratrees_excess | 0.00785649 | 0.0066642 | 0.0828294 | 0.0111553 | 0.00224638 | 0.00221791 | 0.000251778 | 0.00102847 |
| U2_risk_sane | M5_plus_event_elasticnet_excess | elasticnet | 50 | 39 | 0.012726 | 0.00380951 | 0.00982139 | 0.00494451 | 0.641026 | 0.00625944 | 0.00993886 | 0.946842 | 0.000946842 | 0.00892773 | 0.124436 | 0.0818031 | 0.121572 | 39 | 0.672878 | 0.0109067 | 0.00494451 | 0.00575904 | -0.0100188 | plus_event | elasticnet_excess | 0.0100826 | 0.00897414 | 0.082875 | 0.0116541 | -0.000261221 | -4.64093e-05 | -0.00107191 | -0.000747375 |
| U2_risk_sane | M5_plus_event_logistic_top20 | logistic_classifier | 50 | 39 | -0.00148444 | -0.000843668 | -0.0043891 | -0.0103511 | 0.384615 | -0.00159565 | -0.00309123 | 0.914737 | 0.000914737 | -0.00514695 | -0.0514162 | -0.0398034 | 0.176174 | 39 | -0.225932 | 0.00135367 | -0.00617007 | -0.0457214 | 0.0135983 | plus_event | logistic_top20 | -0.0037002 | -0.004203 | -0.0390621 | 0.00199977 | -0.000688898 | -0.000943946 | -0.000741299 | -0.000646097 |

## Feature Coverage

| feature_spec | families | feature | raw_feature | rows | non_null | coverage_ratio | candidate_pool_pass_coverage_ratio | first_signal_date | last_signal_date |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| plus_event | price_volume,event | feature_ret_5d | feature_ret_5d | 307350 | 307017 | 0.998917 | 1 | 2021-01-29 | 2026-04-30 |
| plus_event | price_volume,event | feature_ret_20d | feature_ret_20d | 307350 | 306083 | 0.995878 | 1 | 2021-01-29 | 2026-04-30 |
| plus_event | price_volume,event | feature_ret_60d | feature_ret_60d | 307350 | 303506 | 0.987493 | 1 | 2021-01-29 | 2026-04-30 |
| plus_event | price_volume,event | feature_realized_vol_20d | feature_realized_vol_20d | 307350 | 306673 | 0.997797 | 1 | 2021-01-29 | 2026-04-30 |
| plus_event | price_volume,event | feature_amount_20d_log | feature_amount_20d_log | 307350 | 306745 | 0.998032 | 1 | 2021-01-29 | 2026-04-30 |
| plus_event | price_volume,event | feature_turnover_20d | feature_turnover_20d | 307350 | 306745 | 0.998032 | 1 | 2021-01-29 | 2026-04-30 |
| plus_event | price_volume,event | feature_price_position_250d | feature_price_position_250d | 307350 | 303573 | 0.987711 | 1 | 2021-01-29 | 2026-04-30 |
| plus_event | price_volume,event | feature_limit_move_hits_20d | feature_limit_move_hits_20d | 307350 | 307350 | 1 | 1 | 2021-01-29 | 2026-04-30 |
| plus_event | price_volume,event | feature_event_earnings_guidance_direction_z | feature_event_earnings_guidance_direction | 307350 | 218240 | 0.71007 | 0.718874 | 2021-03-31 | 2026-04-30 |
| plus_event | price_volume,event | feature_event_earnings_guidance_magnitude_z | feature_event_earnings_guidance_magnitude | 307350 | 216218 | 0.703491 | 0.712921 | 2021-03-31 | 2026-04-30 |
| plus_event | price_volume,event | feature_event_earnings_surprise_ttm_z | feature_event_earnings_surprise_ttm | 307350 | 152397 | 0.495842 | 0.508516 | 2021-07-30 | 2026-04-30 |
| plus_event | price_volume,event | feature_event_buyback_amount_ratio_z | feature_event_buyback_amount_ratio | 307350 | 0 | 0 | 0 |  |  |
| plus_event | price_volume,event | feature_event_buyback_recent_30d_z | feature_event_buyback_recent_30d | 307350 | 0 | 0 | 0 |  |  |
| plus_event | price_volume,event | feature_event_reduction_plan_flag_z | feature_event_reduction_plan_flag | 307350 | 10721 | 0.0348821 | 0.0396418 | 2021-01-29 | 2026-01-30 |
| plus_event | price_volume,event | feature_event_unlock_ratio_30d_z | feature_event_unlock_ratio_30d | 307350 | 851 | 0.00276883 | 0.00240978 | 2021-01-29 | 2021-06-30 |

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
| U1_liquid_tradable | M5_plus_event_elasticnet_excess | 20 | 2023 | 12 | -0.00304509 | 0.015469 | 0.0168078 | 0.583333 | 0.00301224 |
| U1_liquid_tradable | M5_plus_event_elasticnet_excess | 20 | 2024 | 12 | 0.00869732 | 0.0145672 | 0.00938254 | 0.583333 | 0.0313282 |
| U1_liquid_tradable | M5_plus_event_elasticnet_excess | 20 | 2025 | 12 | 0.0665407 | 0.0335653 | 0.0203276 | 0.666667 | 0.0235844 |
| U1_liquid_tradable | M5_plus_event_elasticnet_excess | 20 | 2026 | 3 | -0.00329335 | -0.00668842 | 0.00339663 | 0.666667 | -0.0170591 |
| U1_liquid_tradable | M5_plus_event_elasticnet_excess | 30 | 2023 | 12 | -0.00331425 | 0.0151998 | 0.00649469 | 0.75 | 0.00314917 |
| U1_liquid_tradable | M5_plus_event_elasticnet_excess | 30 | 2024 | 12 | -0.00235637 | 0.00351353 | -0.00229291 | 0.416667 | 0.0105794 |
| U1_liquid_tradable | M5_plus_event_elasticnet_excess | 30 | 2025 | 12 | 0.0549226 | 0.0219473 | 0.0183294 | 0.666667 | 0.0107445 |
| U1_liquid_tradable | M5_plus_event_elasticnet_excess | 30 | 2026 | 3 | -0.000985895 | -0.00438097 | -0.010215 | 0.333333 | -0.0222292 |
| U1_liquid_tradable | M5_plus_event_elasticnet_excess | 50 | 2023 | 12 | -0.00252757 | 0.0159865 | 0.00900708 | 0.75 | 0.0141979 |
| U1_liquid_tradable | M5_plus_event_elasticnet_excess | 50 | 2024 | 12 | -0.00531639 | 0.000553516 | -0.00403968 | 0.333333 | 0.00308837 |
| U1_liquid_tradable | M5_plus_event_elasticnet_excess | 50 | 2025 | 12 | 0.0509514 | 0.017976 | 0.0174933 | 0.666667 | 0.0121778 |
| U1_liquid_tradable | M5_plus_event_elasticnet_excess | 50 | 2026 | 3 | 0.00754185 | 0.00414678 | 0.00507736 | 1 | -0.0147873 |
| U1_liquid_tradable | M5_plus_event_extratrees_excess | 20 | 2023 | 12 | 0.00333684 | 0.0218509 | 0.0152024 | 0.75 | 0.0182853 |
| U1_liquid_tradable | M5_plus_event_extratrees_excess | 20 | 2024 | 12 | 0.0145125 | 0.0203824 | 0.0140934 | 0.583333 | 0.00882682 |
| U1_liquid_tradable | M5_plus_event_extratrees_excess | 20 | 2025 | 12 | 0.056346 | 0.0233707 | 0.0248481 | 0.583333 | 0.00802389 |
| U1_liquid_tradable | M5_plus_event_extratrees_excess | 20 | 2026 | 3 | -0.0107043 | -0.0140993 | -0.00687275 | 0 | -0.0128653 |
| U1_liquid_tradable | M5_plus_event_extratrees_excess | 30 | 2023 | 12 | -0.00703241 | 0.0114817 | 0.00408921 | 0.666667 | -0.00568904 |
| U1_liquid_tradable | M5_plus_event_extratrees_excess | 30 | 2024 | 12 | 0.010635 | 0.0165049 | 0.00192407 | 0.5 | 0.00522542 |
| U1_liquid_tradable | M5_plus_event_extratrees_excess | 30 | 2025 | 12 | 0.0539578 | 0.0209824 | 0.0158518 | 0.5 | 0.0147458 |
| U1_liquid_tradable | M5_plus_event_extratrees_excess | 30 | 2026 | 3 | -0.0102499 | -0.013645 | -0.00909707 | 0.333333 | -0.018555 |
| U1_liquid_tradable | M5_plus_event_extratrees_excess | 50 | 2023 | 12 | -0.00393894 | 0.0145751 | 0.0105668 | 0.833333 | 0.00686168 |
| U1_liquid_tradable | M5_plus_event_extratrees_excess | 50 | 2024 | 12 | 0.00983415 | 0.0157041 | 0.00389746 | 0.5 | 0.0161082 |
| U1_liquid_tradable | M5_plus_event_extratrees_excess | 50 | 2025 | 12 | 0.0480129 | 0.0150375 | 0.0116636 | 0.5 | 0.0110127 |
| U1_liquid_tradable | M5_plus_event_extratrees_excess | 50 | 2026 | 3 | -0.00525158 | -0.00864665 | -0.00285176 | 0 | -0.00243401 |
| U1_liquid_tradable | M5_plus_event_logistic_top20 | 20 | 2023 | 12 | -0.038816 | -0.0203019 | -0.0182843 | 0.333333 | -0.0224672 |
| U1_liquid_tradable | M5_plus_event_logistic_top20 | 20 | 2024 | 12 | -0.0169182 | -0.0110483 | -0.0166377 | 0.333333 | 0.00656728 |
| U1_liquid_tradable | M5_plus_event_logistic_top20 | 20 | 2025 | 12 | 0.028921 | -0.00405439 | -0.0205345 | 0.333333 | -0.00224434 |
| U1_liquid_tradable | M5_plus_event_logistic_top20 | 20 | 2026 | 3 | 0.053154 | 0.0497589 | -0.0143796 | 0.333333 | -0.0245713 |

## Realized Market State Slice

| candidate_pool_version | model | top_k | realized_market_state | months | mean_topk_return | mean_topk_excess | median_topk_excess | hit_rate | mean_topk_minus_nextk |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| U1_liquid_tradable | B0_market_ew | 20 | neutral | 25 | 0.00196727 | 0 | 0 | 0 |  |
| U1_liquid_tradable | B0_market_ew | 20 | strong_down | 7 | -0.0722071 | 0 | 0 | 0 |  |
| U1_liquid_tradable | B0_market_ew | 20 | strong_up | 7 | 0.0813642 | 0 | 0 | 0 |  |
| U1_liquid_tradable | M5_plus_event_elasticnet_excess | 20 | neutral | 25 | 0.0237018 | 0.0217345 | 0.0209307 | 0.72 | 0.0219692 |
| U1_liquid_tradable | M5_plus_event_elasticnet_excess | 20 | strong_down | 7 | -0.0753843 | -0.0031772 | -0.00667486 | 0.285714 | -0.0212424 |
| U1_liquid_tradable | M5_plus_event_elasticnet_excess | 20 | strong_up | 7 | 0.113083 | 0.0317188 | 0.018265 | 0.571429 | 0.0347698 |
| U1_liquid_tradable | M5_plus_event_extratrees_excess | 20 | neutral | 25 | 0.0191344 | 0.0171672 | 0.0146277 | 0.6 | 0.00781279 |
| U1_liquid_tradable | M5_plus_event_extratrees_excess | 20 | strong_down | 7 | -0.0625676 | 0.00963946 | 0.0118364 | 0.571429 | 0.00965447 |
| U1_liquid_tradable | M5_plus_event_extratrees_excess | 20 | strong_up | 7 | 0.116835 | 0.0354706 | 0.00600528 | 0.571429 | 0.0171622 |
| U1_liquid_tradable | M5_plus_event_logistic_top20 | 20 | neutral | 25 | 0.00254925 | 0.000581979 | -0.0143796 | 0.4 | -0.00651581 |
| U1_liquid_tradable | M5_plus_event_logistic_top20 | 20 | strong_down | 7 | -0.122582 | -0.050375 | -0.0424999 | 0 | -0.00283894 |
| U1_liquid_tradable | M5_plus_event_logistic_top20 | 20 | strong_up | 7 | 0.0902923 | 0.0089281 | -0.0106234 | 0.428571 | -0.0155253 |
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
| U2_risk_sane | M5_plus_event_elasticnet_excess | 20 | neutral | 25 | 0.022572 | 0.0206047 | 0.0169727 | 0.64 | 0.0167353 |
| U2_risk_sane | M5_plus_event_elasticnet_excess | 20 | strong_down | 7 | -0.0722307 | -2.36591e-05 | -0.00993887 | 0.428571 | -0.0178528 |
| U2_risk_sane | M5_plus_event_elasticnet_excess | 20 | strong_up | 7 | 0.0939984 | 0.0126342 | -0.00819423 | 0.428571 | 0.00910619 |
| U2_risk_sane | M5_plus_event_extratrees_excess | 20 | neutral | 25 | 0.0157952 | 0.0138279 | 0.00747342 | 0.64 | 0.00612485 |
| U2_risk_sane | M5_plus_event_extratrees_excess | 20 | strong_down | 7 | -0.0502935 | 0.0219136 | -0.00473187 | 0.428571 | 0.0218455 |
| U2_risk_sane | M5_plus_event_extratrees_excess | 20 | strong_up | 7 | 0.104867 | 0.0235029 | 0.00450937 | 0.571429 | 0.0227414 |
| U2_risk_sane | M5_plus_event_logistic_top20 | 20 | neutral | 25 | 0.00923866 | 0.00727139 | -0.00257002 | 0.48 | 0.0135803 |
| U2_risk_sane | M5_plus_event_logistic_top20 | 20 | strong_down | 7 | -0.119507 | -0.0473 | -0.0327271 | 0 | -0.0126412 |
| U2_risk_sane | M5_plus_event_logistic_top20 | 20 | strong_up | 7 | 0.103638 | 0.0222742 | 0.0389258 | 0.571429 | 0.0193822 |
| U2_risk_sane | M5_price_volume_only_elasticnet_excess | 20 | neutral | 25 | 0.0201659 | 0.0181987 | 0.0225659 | 0.72 | 0.0110538 |
| U2_risk_sane | M5_price_volume_only_elasticnet_excess | 20 | strong_down | 7 | -0.0722781 | -7.10265e-05 | -0.00533903 | 0.285714 | -0.0127369 |
| U2_risk_sane | M5_price_volume_only_elasticnet_excess | 20 | strong_up | 7 | 0.0990552 | 0.017691 | -0.00214467 | 0.428571 | 0.026938 |
| U2_risk_sane | M5_price_volume_only_extratrees_excess | 20 | neutral | 25 | 0.01457 | 0.0126027 | 0.00850468 | 0.56 | 0.00775238 |
| U2_risk_sane | M5_price_volume_only_extratrees_excess | 20 | strong_down | 7 | -0.0520508 | 0.0201563 | 0.0159295 | 0.571429 | 0.0278794 |
| U2_risk_sane | M5_price_volume_only_extratrees_excess | 20 | strong_up | 7 | 0.109801 | 0.0284368 | 0.0302217 | 0.571429 | 0.0364016 |
| U2_risk_sane | M5_price_volume_only_logistic_top20 | 20 | neutral | 25 | 0.00939349 | 0.00742621 | -0.00334626 | 0.48 | 0.00935273 |

## 口径

- `price_volume_only` 是 M5 内部 baseline；后续 spec 按 industry_breadth -> fund_flow -> fundamental -> shareholder 累积加入。
- `topk_excess_after_cost` 使用半 L1 月度换手乘以 cost_bps 的简化成本敏感性。

## 本轮产物

- `data/results/w4_event_stage_a_candidate_2026-05-11_summary.json`
- `data/results/w4_event_stage_a_candidate_2026-05-11_leaderboard.csv`
- `data/results/w4_event_stage_a_candidate_2026-05-11_incremental_delta.csv`
- `data/results/w4_event_stage_a_candidate_2026-05-11_monthly_long.csv`
- `data/results/w4_event_stage_a_candidate_2026-05-11_rank_ic.csv`
- `data/results/w4_event_stage_a_candidate_2026-05-11_quantile_spread.csv`
- `data/results/w4_event_stage_a_candidate_2026-05-11_feature_coverage.csv`
- `data/results/w4_event_stage_a_candidate_2026-05-11_feature_importance.csv`
- `data/results/w4_event_stage_a_candidate_2026-05-11_topk_holdings.csv`
- `data/results/w4_event_stage_a_candidate_2026-05-11_industry_exposure.csv`
- `data/results/w4_event_stage_a_candidate_2026-05-11_candidate_pool_width.csv`
- `data/results/w4_event_stage_a_candidate_2026-05-11_candidate_pool_reject_reason.csv`
- `data/results/w4_event_stage_a_candidate_2026-05-11_year_slice.csv`
- `data/results/w4_event_stage_a_candidate_2026-05-11_regime_slice.csv`
- `data/results/w4_event_stage_a_candidate_2026-05-11_market_states.csv`
- `data/results/w4_event_stage_a_candidate_2026-05-11_manifest.json`
