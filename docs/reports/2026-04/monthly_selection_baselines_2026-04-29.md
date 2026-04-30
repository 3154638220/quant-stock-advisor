# Monthly Selection Baselines

- 生成时间：`2026-04-29T02:14:13.331444+00:00`
- 结果类型：`monthly_selection_baselines`
- 研究主题：`monthly_selection_baselines`
- 研究配置：`dataset_monthly_selection_features_pools_u1_liquid_tradable-u2_risk_sane_topk_20-30-50_buckets_5_wf_24m_costbps_10_0`
- 输出 stem：`monthly_selection_baselines_2026-04-29`
- 数据集：`data/cache/monthly_selection_features.parquet`
- 训练/评估：静态 baseline 全样本打分；ML baseline 使用 walk-forward，只用测试月之前的数据训练。
- 有效标签月份：`62`

## Leaderboard

| candidate_pool_version | model | model_type | top_k | months | mean_topk_return | median_topk_return | topk_excess_mean | topk_excess_median | topk_hit_rate | topk_minus_nextk_mean | industry_neutral_topk_excess_mean | turnover_mean | cost_drag_mean | topk_excess_after_cost_mean | topk_excess_annualized | rank_ic_mean | rank_ic_std | rank_ic_months | rank_ic_ir | quantile_top_minus_bottom_mean | neutral_median_excess | strong_down_median_excess | strong_up_median_excess |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| U1_liquid_tradable | M4_xgboost_top20 | xgboost_classifier | 20 | 38 | 0.0217819 | 0.0111941 | 0.00963304 | 0.00598313 | 0.526316 | -0.00115139 | 0.00954458 | 0.933784 | 0.000933784 | 0.00967835 | 0.121922 | -0.0446552 | 0.161515 | 38 | -0.276478 | 0.00201158 | -0.00341153 | -0.00929608 | 0.0768332 |
| U1_liquid_tradable | M4_elasticnet_excess | elasticnet | 20 | 38 | 0.0214962 | 0.00278989 | 0.00934733 | -0.000909284 | 0.5 | 0.00128806 | 0.00988822 | 0.956757 | 0.000956757 | 0.00807928 | 0.118118 | 0.106543 | 0.119264 | 38 | 0.89334 | 0.0154743 | 0.00518401 | -0.0132606 | -0.0147957 |
| U1_liquid_tradable | M4_extratrees_excess | tree_sanity | 20 | 38 | 0.0200638 | 0.0183132 | 0.0079149 | 0.00973237 | 0.710526 | 0.00169408 | 0.00985643 | 0.978378 | 0.000978378 | 0.00690162 | 0.0992244 | 0.112303 | 0.116132 | 38 | 0.967026 | 0.017342 | 0.0136723 | 0.00156394 | 0.000491866 |
| U1_liquid_tradable | M4_xgboost_excess | xgboost_regression | 20 | 38 | 0.0144737 | 0.0154004 | 0.00232486 | 0.00347546 | 0.526316 | -0.00266986 | 0.00194368 | 0.968919 | 0.000968919 | 0.0019296 | 0.0282578 | 0.0933833 | 0.0933463 | 38 | 1.0004 | 0.0147066 | -0.000769381 | -0.00533792 | 0.00703689 |
| U1_liquid_tradable | B0_market_ew | benchmark | 20 | 62 | 0.0117199 | 0.00904956 | 0 | 0 | 0 |  |  |  | 0 | 0 | 0 |  |  |  |  |  | 0 | 0 | 0 |
| U1_liquid_tradable | B3_low_vol_20d | single_factor | 20 | 62 | 0.0074114 | -0.00359064 | -0.00430849 | -0.00981251 | 0.451613 | -0.0022127 | -0.0057161 | 0.72541 | 0.00072541 | -0.00522657 | -0.0504941 | 0.0876728 | 0.167327 | 62 | 0.523962 | 0.00876962 | -0.00602327 | 0.0388101 | -0.0591243 |
| U1_liquid_tradable | M4_logistic_top20 | logistic_classifier | 20 | 38 | 0.00793932 | -0.00905649 | -0.00420953 | -0.0131646 | 0.421053 | -0.0131621 | -0.00592373 | 0.954054 | 0.000954054 | -0.00582115 | -0.0493611 | -0.0346394 | 0.160734 | 38 | -0.215508 | 0.00119184 | -0.0117848 | -0.0354131 | 0.041132 |
| U1_liquid_tradable | B3_low_vol_quality_proxy | linear_blend | 20 | 62 | 0.00501527 | -0.00192419 | -0.00670461 | -0.00461669 | 0.435484 | 0.00198709 | -0.00339838 | 0.443443 | 0.000443443 | -0.00748518 | -0.0775539 | 0.0692464 | 0.165527 | 62 | 0.41834 | 0.00555183 | -0.00461669 | 0.0489029 | -0.0436988 |
| U1_liquid_tradable | B3_low_limit_move_hits_20d | single_factor | 20 | 62 | 0.00435382 | -0.000640639 | -0.00736606 | -0.0101941 | 0.322581 | -0.00788796 | -0.00483534 | 0.365574 | 0.000365574 | -0.00812965 | -0.0848981 | 0.0632369 | 0.0955423 | 62 | 0.661874 | 0.00287212 | -0.0133822 | 0.000893479 | -0.0223422 |
| U1_liquid_tradable | B4_price_volume_equal_blend | linear_blend | 20 | 62 | -0.00636451 | -0.0158089 | -0.0180844 | -0.0228419 | 0.33871 | 0.000908425 | -0.0159574 | 0.782787 | 0.000782787 | -0.0178635 | -0.196677 | -0.0592661 | 0.128248 | 62 | -0.462121 | -0.0102299 | -0.0206213 | -0.0631393 | 0.00270794 |
| U1_liquid_tradable | B3_liquidity_amount_20d | single_factor | 20 | 62 | -0.00914168 | -0.0274223 | -0.0208616 | -0.0304018 | 0.290323 | -0.00442451 | -0.0204833 | 0.405738 | 0.000405738 | -0.0200088 | -0.223522 | -0.0970286 | 0.118284 | 62 | -0.820303 | -0.0200286 | -0.0317816 | -0.0146853 | -0.0246874 |
| U1_liquid_tradable | B4_turnover_20d | single_factor | 20 | 62 | -0.0114528 | -0.00912104 | -0.0231726 | -0.0256262 | 0.354839 | -0.00936558 | -0.0206535 | 0.537705 | 0.000537705 | -0.0239216 | -0.245232 | -0.0555362 | 0.179102 | 62 | -0.310082 | -0.00104758 | -0.0157926 | -0.0522819 | -0.0263955 |
| U1_liquid_tradable | B1_current_s2_proxy_vol_to_turnover | linear_blend | 20 | 62 | -0.0162041 | -0.0156827 | -0.027924 | -0.0286574 | 0.322581 | -0.00402815 | -0.0266891 | 0.92377 | 0.00092377 | -0.0288823 | -0.288127 | -0.0676398 | 0.102066 | 62 | -0.662705 | -0.0107565 | -0.0282058 | -0.0324245 | -0.0267197 |
| U1_liquid_tradable | B2_short_momentum_5d | single_factor | 20 | 62 | -0.0179774 | -0.0200078 | -0.0296973 | -0.0359719 | 0.322581 | 0.0166788 | -0.0250078 | 0.972131 | 0.000972131 | -0.0308079 | -0.303555 | -0.0511408 | 0.111606 | 62 | -0.458224 | -0.0124995 | -0.0185068 | -0.0401776 | -0.0861429 |
| U1_liquid_tradable | B2_momentum_60d | single_factor | 20 | 62 | -0.0204629 | -0.0163022 | -0.0321828 | -0.0337163 | 0.274194 | 0.00143578 | -0.0281445 | 0.654918 | 0.000654918 | -0.0312837 | -0.324664 | -0.085011 | 0.148777 | 62 | -0.571397 | -0.0139215 | -0.0346576 | -0.022719 | -0.0425083 |
| U1_liquid_tradable | B4_price_position_250d | single_factor | 20 | 62 | -0.0204731 | -0.0304067 | -0.032193 | -0.0419823 | 0.290323 | -0.0121155 | -0.0276502 | 0.937705 | 0.000937705 | -0.0329549 | -0.324749 | -0.0622466 | 0.139719 | 62 | -0.445514 | -0.0072319 | -0.0356056 | -0.0563108 | -0.0483332 |
| U1_liquid_tradable | B2_momentum_20d | single_factor | 20 | 62 | -0.0314747 | -0.0205463 | -0.0431945 | -0.058095 | 0.274194 | -0.00299928 | -0.0410408 | 0.95082 | 0.00095082 | -0.0435613 | -0.411314 | -0.0716427 | 0.134644 | 62 | -0.532089 | -0.0122691 | -0.0236814 | -0.0709588 | -0.0726182 |
| U1_liquid_tradable | B2_momentum_blend | linear_blend | 20 | 62 | -0.0417398 | -0.0423432 | -0.0534597 | -0.0463341 | 0.290323 | -0.0174821 | -0.047541 | 0.945902 | 0.000945902 | -0.0537581 | -0.482787 | -0.090774 | 0.138724 | 62 | -0.654351 | -0.0150235 | -0.0315916 | -0.0710214 | -0.0669403 |
| U2_risk_sane | M4_elasticnet_excess | elasticnet | 20 | 38 | 0.0217144 | 0.00447054 | 0.00956551 | 0.000720254 | 0.5 | 0.0123091 | 0.00890652 | 0.940541 | 0.000940541 | 0.00824262 | 0.121022 | 0.0901327 | 0.115604 | 38 | 0.779665 | 0.0134166 | 0.00770263 | -0.00851515 | -0.0135026 |
| U2_risk_sane | M4_extratrees_excess | tree_sanity | 20 | 38 | 0.018462 | 0.0199396 | 0.00631317 | 0.00467707 | 0.552632 | 0.00180535 | 0.00587936 | 0.960811 | 0.000960811 | 0.00494461 | 0.0784447 | 0.0948686 | 0.114949 | 38 | 0.825307 | 0.0150839 | 0.0214 | -0.00277409 | -0.0163332 |
| U2_risk_sane | M4_xgboost_top20 | xgboost_classifier | 20 | 38 | 0.0148138 | -0.00159345 | 0.00266497 | -0.00101654 | 0.5 | 0.00885697 | 0.00266127 | 0.922973 | 0.000922973 | 0.00322749 | 0.0324525 | -0.0417506 | 0.161159 | 38 | -0.259065 | 0.00221423 | -0.0204016 | -0.0197364 | 0.0709626 |
| U2_risk_sane | M4_xgboost_excess | xgboost_regression | 20 | 38 | 0.0142531 | 0.00618196 | 0.00210419 | -0.00285639 | 0.5 | -0.00179084 | 0.0040774 | 0.967568 | 0.000967568 | 0.00228653 | 0.0255446 | 0.0739489 | 0.0930656 | 38 | 0.794589 | 0.0119445 | 0.00315658 | -0.00842377 | -0.0201295 |
| U2_risk_sane | M4_logistic_top20 | logistic_classifier | 20 | 38 | 0.0137746 | -0.00965432 | 0.00162576 | -0.0120496 | 0.421053 | -0.00771206 | -0.00124808 | 0.952703 | 0.000952703 | 0.00194989 | 0.0196845 | -0.0288083 | 0.163771 | 38 | -0.175906 | 0.00159897 | -0.0115893 | -0.038868 | 0.0727649 |
| U2_risk_sane | B0_market_ew | benchmark | 20 | 62 | 0.0117199 | 0.00904956 | 0 | 0 | 0 |  |  |  | 0 | 0 | 0 |  |  |  |  |  | 0 | 0 | 0 |
| U2_risk_sane | B4_turnover_20d | single_factor | 20 | 62 | 0.0102787 | 0.00123892 | -0.00144121 | -0.0103168 | 0.419355 | 0.0077656 | -0.000894893 | 0.92541 | 0.00092541 | -0.00197856 | -0.0171581 | -0.0390334 | 0.177828 | 62 | -0.219501 | 0.00180553 | -0.0055927 | -0.0233197 | 0.0225871 |
| U2_risk_sane | B3_low_vol_20d | single_factor | 20 | 62 | 0.00721013 | -0.00293349 | -0.00450975 | -0.00829981 | 0.451613 | -0.00208821 | -0.00591809 | 0.729508 | 0.000729508 | -0.00543523 | -0.0527947 | 0.069313 | 0.165332 | 62 | 0.419234 | 0.0038195 | -0.00567184 | 0.0308134 | -0.0591243 |
| U2_risk_sane | B1_current_s2_proxy_vol_to_turnover | linear_blend | 20 | 62 | 0.00601639 | -0.00106728 | -0.00570349 | -0.00862063 | 0.419355 | 0.00493261 | -0.0037968 | 0.968033 | 0.000968033 | -0.00601073 | -0.0663352 | -0.0661147 | 0.112942 | 62 | -0.585388 | -0.00810425 | -0.00862063 | -0.0437586 | -0.000286406 |
| U2_risk_sane | B3_low_limit_move_hits_20d | single_factor | 20 | 62 | 0.00511444 | 0.00150545 | -0.00660544 | -0.0101636 | 0.33871 | -0.00774162 | -0.00431077 | 0.377049 | 0.000377049 | -0.00742574 | -0.0764481 | 0.0410429 | 0.0825861 | 62 | 0.496971 | 0.000830339 | -0.0114401 | 0.000893479 | -0.0223422 |
| U2_risk_sane | B3_low_vol_quality_proxy | linear_blend | 20 | 62 | 0.003517 | -0.00310712 | -0.00820288 | -0.00562045 | 0.435484 | 0.00107037 | -0.00539374 | 0.478689 | 0.000478689 | -0.00904325 | -0.0941128 | 0.0478635 | 0.164321 | 62 | 0.29128 | -0.0003026 | -0.00562045 | 0.042372 | -0.0386739 |
| U2_risk_sane | B3_liquidity_amount_20d | single_factor | 20 | 62 | 0.000629624 | -0.0173169 | -0.0110903 | -0.0163927 | 0.33871 | 0.0034148 | -0.0101199 | 0.434426 | 0.000434426 | -0.0104636 | -0.125258 | -0.0863164 | 0.120005 | 62 | -0.719275 | -0.0175306 | -0.0222595 | -0.00560961 | 0.00520969 |
| U2_risk_sane | B4_price_position_250d | single_factor | 20 | 62 | 0.00168524 | -0.00269237 | -0.0100346 | -0.0138929 | 0.370968 | 0.00229677 | -0.00654901 | 0.969672 | 0.000969672 | -0.0106528 | -0.113987 | -0.0413491 | 0.135241 | 62 | -0.305743 | -0.0016541 | -0.00951125 | 0.0184902 | -0.0431337 |
| U2_risk_sane | B4_price_volume_equal_blend | linear_blend | 20 | 62 | -0.00064768 | -0.00191884 | -0.0123676 | -0.0277693 | 0.322581 | 0.00098899 | -0.00987192 | 0.922131 | 0.000922131 | -0.0128892 | -0.13872 | -0.0465713 | 0.12517 | 62 | -0.372063 | -0.00735065 | -0.0223089 | -0.0368561 | -0.0438076 |
| U2_risk_sane | B2_momentum_20d | single_factor | 20 | 62 | -0.00225598 | 0.00120171 | -0.0139759 | -0.024032 | 0.370968 | -0.00110428 | -0.0129617 | 0.986885 | 0.000986885 | -0.0154263 | -0.155401 | -0.0482818 | 0.132628 | 62 | -0.36404 | -0.00689258 | -0.0111765 | -0.0454709 | -0.0370465 |
| U2_risk_sane | B2_momentum_60d | single_factor | 20 | 62 | -0.0104603 | -0.0167694 | -0.0221802 | -0.0248457 | 0.290323 | -0.00771091 | -0.0181556 | 0.871311 | 0.000871311 | -0.0220893 | -0.235978 | -0.0636978 | 0.142805 | 62 | -0.446047 | -0.00773664 | -0.028351 | 0.00793038 | -0.0214574 |
| U2_risk_sane | B2_momentum_blend | linear_blend | 20 | 62 | -0.0102924 | -0.00935088 | -0.0220123 | -0.0330675 | 0.322581 | 0.00878741 | -0.0212074 | 0.966393 | 0.000966393 | -0.0233635 | -0.234402 | -0.0678013 | 0.131171 | 62 | -0.516892 | -0.00956187 | -0.0248626 | -0.000836316 | -0.0516 |
| U2_risk_sane | B2_short_momentum_5d | single_factor | 20 | 62 | -0.0121104 | -0.0103504 | -0.0238302 | -0.0335738 | 0.322581 | -0.00448733 | -0.0232414 | 0.997541 | 0.000997541 | -0.0254734 | -0.251306 | -0.0390731 | 0.112883 | 62 | -0.346137 | -0.00884103 | -0.0192381 | -0.0768007 | -0.0373765 |
| U1_liquid_tradable | M4_xgboost_top20 | xgboost_classifier | 30 | 38 | 0.0222885 | 0.0154963 | 0.0101397 | 0.00366548 | 0.526316 | 0.0110849 | 0.0096183 | 0.926126 | 0.000926126 | 0.00970488 | 0.128697 | -0.0446552 | 0.161515 | 38 | -0.276478 | 0.00201158 | -0.0081971 | 0.00649844 | 0.0801591 |
| U1_liquid_tradable | M4_extratrees_excess | tree_sanity | 30 | 38 | 0.0209763 | 0.0147154 | 0.00882746 | 0.0133215 | 0.631579 | 0.00243786 | 0.00985466 | 0.975676 | 0.000975676 | 0.00757602 | 0.111227 | 0.112303 | 0.116132 | 38 | 0.967026 | 0.017342 | 0.0154375 | 0.00531292 | -0.0051787 |
| U1_liquid_tradable | M4_elasticnet_excess | elasticnet | 30 | 38 | 0.0203472 | 0.00491414 | 0.00819835 | -0.00257848 | 0.473684 | 0.00403338 | 0.00881127 | 0.947748 | 0.000947748 | 0.00672678 | 0.10294 | 0.106543 | 0.119264 | 38 | 0.89334 | 0.0154743 | 0.00138782 | -0.0108354 | -0.0201791 |
| U1_liquid_tradable | M4_xgboost_excess | xgboost_regression | 30 | 38 | 0.0155707 | 0.013912 | 0.00342188 | 0.00536971 | 0.578947 | -0.00298408 | 0.0037349 | 0.958559 | 0.000958559 | 0.00309909 | 0.0418442 | 0.0933833 | 0.0933463 | 38 | 1.0004 | 0.0147066 | 0.00119029 | 0.00988119 | 0.00448683 |

_仅展示前 40 行，共 108 行。_

## Year Slice

| candidate_pool_version | model | top_k | year | months | mean_topk_return | mean_topk_excess | median_topk_excess | hit_rate | mean_topk_minus_nextk |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| U1_liquid_tradable | B0_market_ew | 20 | 2021 | 12 | 0.0156584 | 0 | 0 | 0 |  |
| U1_liquid_tradable | B0_market_ew | 20 | 2022 | 12 | 0.00642296 | 0 | 0 | 0 |  |
| U1_liquid_tradable | B0_market_ew | 20 | 2023 | 12 | -0.0165124 | 0 | 0 | 0 |  |
| U1_liquid_tradable | B0_market_ew | 20 | 2024 | 12 | 0.0243624 | 0 | 0 | 0 |  |
| U1_liquid_tradable | B0_market_ew | 20 | 2025 | 12 | 0.0332516 | 0 | 0 | 0 |  |
| U1_liquid_tradable | B0_market_ew | 20 | 2026 | 2 | -0.0157816 | 0 | 0 | 0 |  |
| U1_liquid_tradable | B0_market_ew | 30 | 2021 | 12 | 0.0156584 | 0 | 0 | 0 |  |
| U1_liquid_tradable | B0_market_ew | 30 | 2022 | 12 | 0.00642296 | 0 | 0 | 0 |  |
| U1_liquid_tradable | B0_market_ew | 30 | 2023 | 12 | -0.0165124 | 0 | 0 | 0 |  |
| U1_liquid_tradable | B0_market_ew | 30 | 2024 | 12 | 0.0243624 | 0 | 0 | 0 |  |
| U1_liquid_tradable | B0_market_ew | 30 | 2025 | 12 | 0.0332516 | 0 | 0 | 0 |  |
| U1_liquid_tradable | B0_market_ew | 30 | 2026 | 2 | -0.0157816 | 0 | 0 | 0 |  |
| U1_liquid_tradable | B0_market_ew | 50 | 2021 | 12 | 0.0156584 | 0 | 0 | 0 |  |
| U1_liquid_tradable | B0_market_ew | 50 | 2022 | 12 | 0.00642296 | 0 | 0 | 0 |  |
| U1_liquid_tradable | B0_market_ew | 50 | 2023 | 12 | -0.0165124 | 0 | 0 | 0 |  |
| U1_liquid_tradable | B0_market_ew | 50 | 2024 | 12 | 0.0243624 | 0 | 0 | 0 |  |
| U1_liquid_tradable | B0_market_ew | 50 | 2025 | 12 | 0.0332516 | 0 | 0 | 0 |  |
| U1_liquid_tradable | B0_market_ew | 50 | 2026 | 2 | -0.0157816 | 0 | 0 | 0 |  |
| U1_liquid_tradable | B1_current_s2_proxy_vol_to_turnover | 20 | 2021 | 12 | -0.0245377 | -0.0401961 | -0.0327026 | 0.25 | -0.0131523 |
| U1_liquid_tradable | B1_current_s2_proxy_vol_to_turnover | 20 | 2022 | 12 | -0.0379182 | -0.0443411 | -0.0478481 | 0.25 | 0.0031906 |
| U1_liquid_tradable | B1_current_s2_proxy_vol_to_turnover | 20 | 2023 | 12 | -0.017172 | -0.000659538 | -0.00492235 | 0.416667 | -0.00321401 |
| U1_liquid_tradable | B1_current_s2_proxy_vol_to_turnover | 20 | 2024 | 12 | 0.00914526 | -0.0152172 | -0.017973 | 0.416667 | 0.00228138 |
| U1_liquid_tradable | B1_current_s2_proxy_vol_to_turnover | 20 | 2025 | 12 | 0.00538939 | -0.0278622 | -0.0293883 | 0.333333 | -0.000824793 |
| U1_liquid_tradable | B1_current_s2_proxy_vol_to_turnover | 20 | 2026 | 2 | -0.111769 | -0.0959875 | -0.0959875 | 0 | -0.0545579 |
| U1_liquid_tradable | B1_current_s2_proxy_vol_to_turnover | 30 | 2021 | 12 | -0.0164945 | -0.0321528 | -0.0361571 | 0.25 | -0.00899474 |
| U1_liquid_tradable | B1_current_s2_proxy_vol_to_turnover | 30 | 2022 | 12 | -0.0426217 | -0.0490446 | -0.0487811 | 0.25 | -0.0189305 |
| U1_liquid_tradable | B1_current_s2_proxy_vol_to_turnover | 30 | 2023 | 12 | -0.015126 | 0.0013864 | -0.00799659 | 0.5 | 0.00660454 |
| U1_liquid_tradable | B1_current_s2_proxy_vol_to_turnover | 30 | 2024 | 12 | 0.000385067 | -0.0239774 | -0.0177895 | 0.25 | -0.00202156 |
| U1_liquid_tradable | B1_current_s2_proxy_vol_to_turnover | 30 | 2025 | 12 | 2.418e-05 | -0.0332275 | -0.0339024 | 0.25 | -0.0325543 |
| U1_liquid_tradable | B1_current_s2_proxy_vol_to_turnover | 30 | 2026 | 2 | -0.105218 | -0.0894366 | -0.0894366 | 0 | -0.0648708 |
| U1_liquid_tradable | B1_current_s2_proxy_vol_to_turnover | 50 | 2021 | 12 | -0.0129536 | -0.0286119 | -0.025256 | 0.166667 | 0.0067622 |
| U1_liquid_tradable | B1_current_s2_proxy_vol_to_turnover | 50 | 2022 | 12 | -0.0381885 | -0.0446114 | -0.0555275 | 0.25 | -0.0307598 |
| U1_liquid_tradable | B1_current_s2_proxy_vol_to_turnover | 50 | 2023 | 12 | -0.0179492 | -0.00143675 | -0.002708 | 0.416667 | 0.00584394 |
| U1_liquid_tradable | B1_current_s2_proxy_vol_to_turnover | 50 | 2024 | 12 | 0.00608236 | -0.0182801 | -0.0132268 | 0.416667 | -0.00489213 |
| U1_liquid_tradable | B1_current_s2_proxy_vol_to_turnover | 50 | 2025 | 12 | 0.0169272 | -0.0163245 | -0.0269529 | 0.333333 | 0.001353 |
| U1_liquid_tradable | B1_current_s2_proxy_vol_to_turnover | 50 | 2026 | 2 | -0.078341 | -0.0625593 | -0.0625593 | 0 | -0.0291937 |
| U1_liquid_tradable | B2_momentum_20d | 20 | 2021 | 12 | -0.0147482 | -0.0304066 | -0.0464483 | 0.333333 | 0.000892108 |
| U1_liquid_tradable | B2_momentum_20d | 20 | 2022 | 12 | -0.0692983 | -0.0757212 | -0.0777889 | 0.166667 | -0.0389737 |
| U1_liquid_tradable | B2_momentum_20d | 20 | 2023 | 12 | -0.0660271 | -0.0495147 | -0.0729816 | 0.166667 | -0.0267609 |
| U1_liquid_tradable | B2_momentum_20d | 20 | 2024 | 12 | -0.0317398 | -0.0561022 | -0.0619977 | 0.333333 | 0.0347233 |

## Realized Market State Slice

| candidate_pool_version | model | top_k | realized_market_state | months | mean_topk_return | mean_topk_excess | median_topk_excess | hit_rate | mean_topk_minus_nextk |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| U1_liquid_tradable | B0_market_ew | 20 | neutral | 36 | 0.0107482 | 0 | 0 | 0 |  |
| U1_liquid_tradable | B0_market_ew | 20 | strong_down | 13 | -0.0700111 | 0 | 0 | 0 |  |
| U1_liquid_tradable | B0_market_ew | 20 | strong_up | 13 | 0.0961418 | 0 | 0 | 0 |  |
| U1_liquid_tradable | B1_current_s2_proxy_vol_to_turnover | 20 | neutral | 36 | -0.013253 | -0.0240012 | -0.0282058 | 0.277778 | -0.00529975 |
| U1_liquid_tradable | B1_current_s2_proxy_vol_to_turnover | 20 | strong_down | 13 | -0.101278 | -0.0312672 | -0.0324245 | 0.384615 | -0.0183048 |
| U1_liquid_tradable | B1_current_s2_proxy_vol_to_turnover | 20 | strong_up | 13 | 0.0606977 | -0.0354441 | -0.0267197 | 0.384615 | 0.0137698 |
| U1_liquid_tradable | B2_momentum_20d | 20 | neutral | 36 | -0.0194827 | -0.0302309 | -0.0236814 | 0.361111 | -0.00136794 |
| U1_liquid_tradable | B2_momentum_20d | 20 | strong_down | 13 | -0.122447 | -0.0524356 | -0.0709588 | 0.230769 | -0.00118708 |
| U1_liquid_tradable | B2_momentum_20d | 20 | strong_up | 13 | 0.0262888 | -0.0698529 | -0.0726182 | 0.0769231 | -0.00932904 |
| U1_liquid_tradable | B2_momentum_60d | 20 | neutral | 36 | -0.0221218 | -0.03287 | -0.0346576 | 0.277778 | -0.0044101 |
| U1_liquid_tradable | B2_momentum_60d | 20 | strong_down | 13 | -0.0946585 | -0.0246474 | -0.022719 | 0.307692 | -0.00121766 |
| U1_liquid_tradable | B2_momentum_60d | 20 | strong_up | 13 | 0.0583266 | -0.0378151 | -0.0425083 | 0.230769 | 0.0202778 |
| U1_liquid_tradable | B2_momentum_blend | 20 | neutral | 36 | -0.0274909 | -0.0382391 | -0.0315916 | 0.388889 | -0.0116667 |
| U1_liquid_tradable | B2_momentum_blend | 20 | strong_down | 13 | -0.130872 | -0.0608607 | -0.0710214 | 0.153846 | -0.0256157 |
| U1_liquid_tradable | B2_momentum_blend | 20 | strong_up | 13 | 0.00793382 | -0.0882079 | -0.0669403 | 0.153846 | -0.0254527 |
| U1_liquid_tradable | B2_short_momentum_5d | 20 | neutral | 36 | -0.00128787 | -0.012036 | -0.0185068 | 0.388889 | 0.0162228 |
| U1_liquid_tradable | B2_short_momentum_5d | 20 | strong_down | 13 | -0.121844 | -0.0518328 | -0.0401776 | 0.153846 | 0.0223337 |
| U1_liquid_tradable | B2_short_momentum_5d | 20 | strong_up | 13 | 0.0396721 | -0.0564697 | -0.0861429 | 0.307692 | 0.0122867 |
| U1_liquid_tradable | B3_liquidity_amount_20d | 20 | neutral | 36 | -0.0216031 | -0.0323512 | -0.0317816 | 0.277778 | -0.00607315 |
| U1_liquid_tradable | B3_liquidity_amount_20d | 20 | strong_down | 13 | -0.0816543 | -0.0116432 | -0.0146853 | 0.307692 | -0.0143245 |
| U1_liquid_tradable | B3_liquidity_amount_20d | 20 | strong_up | 13 | 0.0978794 | 0.00173768 | -0.0246874 | 0.307692 | 0.010041 |
| U1_liquid_tradable | B3_low_limit_move_hits_20d | 20 | neutral | 36 | -0.000380669 | -0.0111288 | -0.0133822 | 0.277778 | -0.00813018 |
| U1_liquid_tradable | B3_low_limit_move_hits_20d | 20 | strong_down | 13 | -0.0601235 | 0.00988761 | 0.000893479 | 0.538462 | -0.000775042 |
| U1_liquid_tradable | B3_low_limit_move_hits_20d | 20 | strong_up | 13 | 0.081942 | -0.0141997 | -0.0223422 | 0.230769 | -0.0143301 |
| U1_liquid_tradable | B3_low_vol_20d | 20 | neutral | 36 | 0.00849364 | -0.00225452 | -0.00602327 | 0.472222 | 0.00161674 |
| U1_liquid_tradable | B3_low_vol_20d | 20 | strong_down | 13 | -0.0300732 | 0.0399379 | 0.0388101 | 0.769231 | 0.000240374 |
| U1_liquid_tradable | B3_low_vol_20d | 20 | strong_up | 13 | 0.041899 | -0.0542428 | -0.0591243 | 0.0769231 | -0.0152704 |
| U1_liquid_tradable | B3_low_vol_quality_proxy | 20 | neutral | 36 | -0.00271738 | -0.0134655 | -0.00461669 | 0.416667 | 0.00156133 |
| U1_liquid_tradable | B3_low_vol_quality_proxy | 20 | strong_down | 13 | -0.0155968 | 0.0544143 | 0.0489029 | 0.769231 | 0.0190593 |
| U1_liquid_tradable | B3_low_vol_quality_proxy | 20 | strong_up | 13 | 0.0470408 | -0.0491009 | -0.0436988 | 0.153846 | -0.0139061 |
| U1_liquid_tradable | B4_price_position_250d | 20 | neutral | 36 | -0.00582019 | -0.0165684 | -0.0356056 | 0.361111 | -0.00169183 |
| U1_liquid_tradable | B4_price_position_250d | 20 | strong_down | 13 | -0.114658 | -0.0446469 | -0.0563108 | 0.230769 | -0.0392545 |
| U1_liquid_tradable | B4_price_position_250d | 20 | strong_up | 13 | 0.0331343 | -0.0630074 | -0.0483332 | 0.153846 | -0.0138423 |
| U1_liquid_tradable | B4_price_volume_equal_blend | 20 | neutral | 36 | 0.00148714 | -0.00926102 | -0.0206213 | 0.305556 | 0.00997366 |
| U1_liquid_tradable | B4_price_volume_equal_blend | 20 | strong_down | 13 | -0.11351 | -0.0434986 | -0.0631393 | 0.230769 | -0.0101482 |
| U1_liquid_tradable | B4_price_volume_equal_blend | 20 | strong_up | 13 | 0.0790376 | -0.0171042 | 0.00270794 | 0.538462 | -0.0131387 |
| U1_liquid_tradable | B4_turnover_20d | 20 | neutral | 36 | -0.0115383 | -0.0222865 | -0.0157926 | 0.361111 | -0.0124305 |
| U1_liquid_tradable | B4_turnover_20d | 20 | strong_down | 13 | -0.113536 | -0.0435253 | -0.0522819 | 0.230769 | -0.00321359 |
| U1_liquid_tradable | B4_turnover_20d | 20 | strong_up | 13 | 0.0908678 | -0.00527396 | -0.0263955 | 0.461538 | -0.00703 |
| U1_liquid_tradable | M4_elasticnet_excess | 20 | neutral | 23 | 0.0270246 | 0.0177984 | 0.00518401 | 0.565217 | 0.0115055 |

## Industry Exposure

| candidate_pool_version | model | top_k | industry_level1 | mean_share | months |
| --- | --- | --- | --- | --- | --- |
| U1_liquid_tradable | B1_current_s2_proxy_vol_to_turnover | 20 | 电子 | 0.149038 | 52 |
| U1_liquid_tradable | B1_current_s2_proxy_vol_to_turnover | 20 | 计算机 | 0.143137 | 51 |
| U1_liquid_tradable | B1_current_s2_proxy_vol_to_turnover | 20 | 医药生物 | 0.128261 | 46 |
| U1_liquid_tradable | B1_current_s2_proxy_vol_to_turnover | 20 | 机械设备 | 0.126786 | 56 |
| U1_liquid_tradable | B1_current_s2_proxy_vol_to_turnover | 20 | 电力设备 | 0.106522 | 46 |
| U1_liquid_tradable | B1_current_s2_proxy_vol_to_turnover | 20 | 汽车 | 0.0971429 | 35 |
| U1_liquid_tradable | B1_current_s2_proxy_vol_to_turnover | 20 | 基础化工 | 0.0953488 | 43 |
| U1_liquid_tradable | B1_current_s2_proxy_vol_to_turnover | 20 | 石油石化 | 0.0875 | 4 |
| U1_liquid_tradable | B1_current_s2_proxy_vol_to_turnover | 20 | 传媒 | 0.0803571 | 28 |
| U1_liquid_tradable | B1_current_s2_proxy_vol_to_turnover | 20 | 公用事业 | 0.0791667 | 12 |
| U1_liquid_tradable | B1_current_s2_proxy_vol_to_turnover | 20 | 商贸零售 | 0.075 | 18 |
| U1_liquid_tradable | B1_current_s2_proxy_vol_to_turnover | 20 | 建筑装饰 | 0.068 | 25 |
| U1_liquid_tradable | B1_current_s2_proxy_vol_to_turnover | 20 | 房地产 | 0.0675 | 20 |
| U1_liquid_tradable | B1_current_s2_proxy_vol_to_turnover | 20 | _UNKNOWN_ | 0.0666667 | 3 |
| U1_liquid_tradable | B1_current_s2_proxy_vol_to_turnover | 20 | 有色金属 | 0.065 | 20 |
| U1_liquid_tradable | B1_current_s2_proxy_vol_to_turnover | 20 | 通信 | 0.0642857 | 28 |
| U1_liquid_tradable | B1_current_s2_proxy_vol_to_turnover | 20 | 家用电器 | 0.0636364 | 11 |
| U1_liquid_tradable | B1_current_s2_proxy_vol_to_turnover | 20 | 交通运输 | 0.0625 | 8 |
| U1_liquid_tradable | B1_current_s2_proxy_vol_to_turnover | 20 | 社会服务 | 0.0611111 | 18 |
| U1_liquid_tradable | B1_current_s2_proxy_vol_to_turnover | 20 | 环保 | 0.0608696 | 23 |
| U1_liquid_tradable | B1_current_s2_proxy_vol_to_turnover | 20 | 煤炭 | 0.06 | 10 |
| U1_liquid_tradable | B1_current_s2_proxy_vol_to_turnover | 20 | 轻工制造 | 0.0578947 | 19 |
| U1_liquid_tradable | B1_current_s2_proxy_vol_to_turnover | 20 | 农林牧渔 | 0.05625 | 8 |
| U1_liquid_tradable | B1_current_s2_proxy_vol_to_turnover | 20 | 建筑材料 | 0.05625 | 16 |
| U1_liquid_tradable | B1_current_s2_proxy_vol_to_turnover | 20 | 非银金融 | 0.05625 | 8 |
| U1_liquid_tradable | B1_current_s2_proxy_vol_to_turnover | 20 | 国防军工 | 0.0555556 | 18 |
| U1_liquid_tradable | B1_current_s2_proxy_vol_to_turnover | 20 | 食品饮料 | 0.055 | 10 |
| U1_liquid_tradable | B1_current_s2_proxy_vol_to_turnover | 20 | 纺织服饰 | 0.0529412 | 17 |
| U1_liquid_tradable | B1_current_s2_proxy_vol_to_turnover | 20 | 钢铁 | 0.05 | 3 |
| U1_liquid_tradable | B1_current_s2_proxy_vol_to_turnover | 20 | 综合 | 0.05 | 2 |
| U1_liquid_tradable | B1_current_s2_proxy_vol_to_turnover | 20 | 美容护理 | 0.05 | 8 |
| U1_liquid_tradable | B2_momentum_20d | 20 | 机械设备 | 0.123636 | 55 |
| U1_liquid_tradable | B2_momentum_20d | 20 | 电力设备 | 0.121951 | 41 |
| U1_liquid_tradable | B2_momentum_20d | 20 | 电子 | 0.12 | 40 |
| U1_liquid_tradable | B2_momentum_20d | 20 | 计算机 | 0.118367 | 49 |
| U1_liquid_tradable | B2_momentum_20d | 20 | 医药生物 | 0.110256 | 39 |
| U1_liquid_tradable | B2_momentum_20d | 20 | 基础化工 | 0.109091 | 44 |
| U1_liquid_tradable | B2_momentum_20d | 20 | 传媒 | 0.106897 | 29 |
| U1_liquid_tradable | B2_momentum_20d | 20 | 汽车 | 0.0965517 | 29 |
| U1_liquid_tradable | B2_momentum_20d | 20 | 公用事业 | 0.095 | 20 |

## 口径

- 输入固定为 `data/cache/monthly_selection_features.parquet` 兼容的 M2 canonical dataset。
- 主训练池/主报告池为 `U1_liquid_tradable` 与 `U2_risk_sane`。
- 第一轮只使用 price-volume-only 特征：收益动量、低波、流动性、换手、价格位置和涨跌停路径特征。
- Top-K 并行报告 `20 / 30 / 50`；`B0_market_ew` 作为市场等权基准，非真实持仓模型。
- `realized_market_state` 使用同一持有期市场等权收益的全样本 20%/80% 分位切片，仅用于归因，不作为可交易信号。
- `topk_excess_after_cost` 使用半 L1 月度换手乘以 `cost_bps` 的简化成本敏感性。
- baseline overlap 不作为 M4 gate；M4 的核心是 Rank IC、Top-K 超额、Top-K vs next-K、分桶 spread、年度/状态稳定性、行业暴露和换手。

## 本轮结论

- 本轮新增：M4 price-volume-only baseline ranker，覆盖单因子、线性 blend、ElasticNet、Logistic top-bucket classifier、ExtraTrees sanity check、XGBoost regression/classifier。
- 数据质量：沿用 M2 canonical dataset 的 PIT 与候选池口径；本脚本只消费已落地特征，不引入新数据家族。
- `U1_liquid_tradable` Top20 当前领先模型：

| candidate_pool_version | model | model_type | top_k | months | mean_topk_return | median_topk_return | topk_excess_mean | topk_excess_median | topk_hit_rate | topk_minus_nextk_mean | industry_neutral_topk_excess_mean | turnover_mean | cost_drag_mean | topk_excess_after_cost_mean | topk_excess_annualized | rank_ic_mean | rank_ic_std | rank_ic_months | rank_ic_ir | quantile_top_minus_bottom_mean | neutral_median_excess | strong_down_median_excess | strong_up_median_excess |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| U1_liquid_tradable | M4_xgboost_top20 | xgboost_classifier | 20 | 38 | 0.0217819 | 0.0111941 | 0.00963304 | 0.00598313 | 0.526316 | -0.00115139 | 0.00954458 | 0.933784 | 0.000933784 | 0.00967835 | 0.121922 | -0.0446552 | 0.161515 | 38 | -0.276478 | 0.00201158 | -0.00341153 | -0.00929608 | 0.0768332 |
| U1_liquid_tradable | M4_elasticnet_excess | elasticnet | 20 | 38 | 0.0214962 | 0.00278989 | 0.00934733 | -0.000909284 | 0.5 | 0.00128806 | 0.00988822 | 0.956757 | 0.000956757 | 0.00807928 | 0.118118 | 0.106543 | 0.119264 | 38 | 0.89334 | 0.0154743 | 0.00518401 | -0.0132606 | -0.0147957 |
| U1_liquid_tradable | M4_extratrees_excess | tree_sanity | 20 | 38 | 0.0200638 | 0.0183132 | 0.0079149 | 0.00973237 | 0.710526 | 0.00169408 | 0.00985643 | 0.978378 | 0.000978378 | 0.00690162 | 0.0992244 | 0.112303 | 0.116132 | 38 | 0.967026 | 0.017342 | 0.0136723 | 0.00156394 | 0.000491866 |

- `U2_risk_sane` Top20 当前领先模型：

| candidate_pool_version | model | model_type | top_k | months | mean_topk_return | median_topk_return | topk_excess_mean | topk_excess_median | topk_hit_rate | topk_minus_nextk_mean | industry_neutral_topk_excess_mean | turnover_mean | cost_drag_mean | topk_excess_after_cost_mean | topk_excess_annualized | rank_ic_mean | rank_ic_std | rank_ic_months | rank_ic_ir | quantile_top_minus_bottom_mean | neutral_median_excess | strong_down_median_excess | strong_up_median_excess |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| U2_risk_sane | M4_elasticnet_excess | elasticnet | 20 | 38 | 0.0217144 | 0.00447054 | 0.00956551 | 0.000720254 | 0.5 | 0.0123091 | 0.00890652 | 0.940541 | 0.000940541 | 0.00824262 | 0.121022 | 0.0901327 | 0.115604 | 38 | 0.779665 | 0.0134166 | 0.00770263 | -0.00851515 | -0.0135026 |
| U2_risk_sane | M4_extratrees_excess | tree_sanity | 20 | 38 | 0.018462 | 0.0199396 | 0.00631317 | 0.00467707 | 0.552632 | 0.00180535 | 0.00587936 | 0.960811 | 0.000960811 | 0.00494461 | 0.0784447 | 0.0948686 | 0.114949 | 38 | 0.825307 | 0.0150839 | 0.0214 | -0.00277409 | -0.0163332 |
| U2_risk_sane | M4_xgboost_top20 | xgboost_classifier | 20 | 38 | 0.0148138 | -0.00159345 | 0.00266497 | -0.00101654 | 0.5 | 0.00885697 | 0.00266127 | 0.922973 | 0.000922973 | 0.00322749 | 0.0324525 | -0.0417506 | 0.161159 | 38 | -0.259065 | 0.00221423 | -0.0204016 | -0.0197364 | 0.0709626 |

- 静态单因子/线性 blend 多数无法稳定跑赢市场，应保留为低门槛对照，不作为推荐候选。
- walk-forward ML baseline 有弱正向起点，但 strong-up / strong-down 切片仍不稳；M4 不进入生产。
- 下一步进入 M5，逐个验证 industry breadth、fund flow、fundamental、shareholder 等增量是否能稳定改善 Rank IC、Top-K 超额、分桶 spread 和强市参与度。

## 本轮产物

- `data/results/monthly_selection_baselines_2026-04-29_summary.json`
- `data/results/monthly_selection_baselines_2026-04-29_leaderboard.csv`
- `data/results/monthly_selection_baselines_2026-04-29_monthly_long.csv`
- `data/results/monthly_selection_baselines_2026-04-29_rank_ic.csv`
- `data/results/monthly_selection_baselines_2026-04-29_quantile_spread.csv`
- `data/results/monthly_selection_baselines_2026-04-29_topk_holdings.csv`
- `data/results/monthly_selection_baselines_2026-04-29_industry_exposure.csv`
- `data/results/monthly_selection_baselines_2026-04-29_candidate_pool_width.csv`
- `data/results/monthly_selection_baselines_2026-04-29_candidate_pool_reject_reason.csv`
- `data/results/monthly_selection_baselines_2026-04-29_feature_importance.csv`
- `data/results/monthly_selection_baselines_2026-04-29_year_slice.csv`
- `data/results/monthly_selection_baselines_2026-04-29_regime_slice.csv`
- `data/results/monthly_selection_baselines_2026-04-29_market_states.csv`
- `data/results/monthly_selection_baselines_2026-04-29_manifest.json`
