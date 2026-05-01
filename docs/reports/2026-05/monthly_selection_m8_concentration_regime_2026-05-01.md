# Monthly Selection M8 Concentration + Regime

- 生成时间：`2026-05-01T06:24:27.041077+00:00`
- 结果类型：`monthly_selection_m8_concentration_regime`
- 研究配置：`dataset_monthly_selection_features_families_price_volume_only-industry_breadth-fund_flow-fundamental_pools_u1_liquid_tradable_topk_20_capgrid_20_3_maxfit_0_jobs_all_wf_24m_costbps_10_0`
- 输出 stem：`monthly_selection_m8_concentration_regime_2026-05-01`
- 数据集：`data/cache/monthly_selection_features.parquet`
- 训练/评估：M5/M6 walk-forward score；M8 在选择层做行业 cap 和 lagged-regime fixed policy。
- 有效标签月份：`63`
- 交易时点：`sell existing monthly Top-K on the holding month's last trading day open; buy the next monthly Top-K on the following trading day's open`
- 行业集中度策略：`greedy monthly Top-K selection with max names per industry; uncapped rows retained as control`
- Regime 策略：`strong_down=80% ElasticNet/20% ExtraTrees; strong_up or wide breadth=60% ExtraTrees/25% rank_ndcg/15% top20; neutral=50/50 stable M5`

## Leaderboard

| candidate_pool_version | base_model | model | base_model_type | model_type | selection_policy | max_industry_names | top_k | months | mean_topk_return | median_topk_return | topk_excess_mean | topk_excess_median | topk_hit_rate | topk_minus_nextk_mean | industry_neutral_topk_excess_mean | turnover_mean | cost_drag_mean | topk_excess_after_cost_mean | topk_excess_annualized | rank_ic_mean | rank_ic_std | rank_ic_months | rank_ic_ir | quantile_top_minus_bottom_mean | neutral_median_excess | strong_down_median_excess | strong_up_median_excess | max_industry_share_mean | max_industry_share_max | industry_count_mean | concentration_pass_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| U1_liquid_tradable | M8_regime_aware_fixed_policy | M8_regime_aware_fixed_policy__indcap3 | lagged_regime_fixed_score_blend | industry_cap_selection | industry_names_cap | 3 | 20 | 39 | 0.023179 | 0.0174104 | 0.0202743 | 0.0150201 | 0.769231 | 0.014287 | 0.018593 | 0.951316 | 0.000951316 | 0.0194528 | 0.272341 | 0.104108 | 0.126043 | 39 | 0.825973 | 0.0159338 | 0.0153415 | 0.0123861 | 0.00716442 | 0.15 | 0.15 | 10.5641 | 1 |
| U1_liquid_tradable | M5_plus_industry_breadth_plus_fund_flow_plus_fundamental_elasticnet_excess | M5_plus_industry_breadth_plus_fund_flow_plus_fundamental_elasticnet_excess__indcap3 | elasticnet | industry_cap_selection | industry_names_cap | 3 | 20 | 39 | 0.0181541 | 0.0182443 | 0.0152494 | 0.00462486 | 0.615385 | 0.00730048 | 0.0139608 | 0.914474 | 0.000914474 | 0.0146817 | 0.199148 | 0.104611 | 0.120622 | 39 | 0.867263 | 0.0177313 | 0.0020713 | 0.00462486 | 0.0183193 | 0.15 | 0.15 | 11.7179 | 1 |
| U1_liquid_tradable | M8_regime_aware_fixed_policy | M8_regime_aware_fixed_policy__uncapped | lagged_regime_fixed_score_blend | unconstrained_selection | uncapped | 0 | 20 | 39 | 0.0169349 | 0.011885 | 0.0140302 | 0.00813766 | 0.641026 | -0.00224436 | 0.0138483 | 0.95 | 0.00095 | 0.0132871 | 0.181982 | 0.104108 | 0.126043 | 39 | 0.825973 | 0.0159338 | 0.015072 | 0.00745117 | -0.0115867 | 0.29359 | 0.65 | 9 | 0.717949 |
| U1_liquid_tradable | M5_plus_industry_breadth_plus_fund_flow_plus_fundamental_elasticnet_excess | M5_plus_industry_breadth_plus_fund_flow_plus_fundamental_elasticnet_excess__uncapped | elasticnet | unconstrained_selection | uncapped | 0 | 20 | 39 | 0.0135035 | 0.0114559 | 0.0105989 | 0.00137869 | 0.512821 | 0.000204518 | 0.0103577 | 0.910526 | 0.000910526 | 0.009931 | 0.134869 | 0.104611 | 0.120622 | 39 | 0.867263 | 0.0177313 | 0.00137869 | 0.0107253 | -0.00814103 | 0.248718 | 0.55 | 10.6923 | 0.846154 |
| U1_liquid_tradable | M5_plus_industry_breadth_plus_fund_flow_plus_fundamental_extratrees_excess | M5_plus_industry_breadth_plus_fund_flow_plus_fundamental_extratrees_excess__indcap3 | tree_sanity | industry_cap_selection | industry_names_cap | 3 | 20 | 39 | 0.0108014 | -0.00813899 | 0.0078967 | 0.00492855 | 0.564103 | 0.00217267 | 0.00772268 | 0.946053 | 0.000946053 | 0.00650394 | 0.0989863 | 0.110605 | 0.138242 | 39 | 0.800083 | 0.0161751 | 0.00782732 | -0.00463758 | 0.0185264 | 0.148718 | 0.15 | 10.641 | 1 |
| U1_liquid_tradable | M5_plus_industry_breadth_plus_fund_flow_plus_fundamental_extratrees_excess | M5_plus_industry_breadth_plus_fund_flow_plus_fundamental_extratrees_excess__uncapped | tree_sanity | unconstrained_selection | uncapped | 0 | 20 | 39 | 0.0108111 | 0.00521471 | 0.0079064 | 0.0153959 | 0.564103 | 0.00151257 | 0.00735109 | 0.940789 | 0.000940789 | 0.0064371 | 0.0991133 | 0.110605 | 0.138242 | 39 | 0.800083 | 0.0161751 | 0.0169344 | -0.00735076 | 0.022467 | 0.30641 | 0.6 | 8.74359 | 0.641026 |
| U1_liquid_tradable | M6_top20_calibrated | M6_top20_calibrated__indcap3 | top_bucket_classifier_rank_calibrated | industry_cap_selection | industry_names_cap | 3 | 20 | 39 | 0.00123868 | -0.0124196 | -0.00166598 | -0.0108585 | 0.461538 | -0.00298357 | 0.000638714 | 0.955263 | 0.000955263 | -0.00187243 | -0.0198096 | -0.058348 | 0.139906 | 39 | -0.41705 | -0.00257895 | -0.0145653 | 0.0110587 | 0.00756909 | 0.15 | 0.15 | 9.17949 | 1 |
| U1_liquid_tradable | M6_top20_calibrated | M6_top20_calibrated__uncapped | top_bucket_classifier_rank_calibrated | unconstrained_selection | uncapped | 0 | 20 | 39 | 0.000169548 | 0.00447214 | -0.00273511 | -0.00561768 | 0.435897 | 0.0100743 | 0.00203217 | 0.964474 | 0.000964474 | -0.00280899 | -0.0323321 | -0.058348 | 0.139906 | 39 | -0.41705 | -0.00257895 | -0.00561768 | -0.0351222 | -0.00273452 | 0.65641 | 1 | 3.41026 | 0.0769231 |
| U1_liquid_tradable | M6_xgboost_rank_ndcg | M6_xgboost_rank_ndcg__uncapped | xgboost_ranker | unconstrained_selection | uncapped | 0 | 20 | 39 | -0.00151367 | -0.00540457 | -0.00441832 | -0.00939811 | 0.384615 | -0.00269697 | -0.00709359 | 0.911842 | 0.000911842 | -0.00519912 | -0.0517502 | 0.0229536 | 0.0863624 | 39 | 0.265783 | 0.00763245 | -0.0138396 | -0.00403328 | 0.00812581 | 0.476923 | 1 | 7.4359 | 0.410256 |
| U1_liquid_tradable | M6_xgboost_rank_ndcg | M6_xgboost_rank_ndcg__indcap3 | xgboost_ranker | industry_cap_selection | industry_names_cap | 3 | 20 | 39 | -0.00407061 | -0.0223953 | -0.00697527 | -0.0107454 | 0.410256 | -0.00370739 | -0.00435605 | 0.871053 | 0.000871053 | -0.00775603 | -0.0805655 | 0.0229536 | 0.0863624 | 39 | 0.265783 | 0.00763245 | -0.0252802 | -0.00706363 | 0.0239633 | 0.15 | 0.15 | 11.1282 | 1 |

## Gate

| candidate_pool_version | base_model | model | selection_policy | max_industry_names | top_k | topk_excess_after_cost_mean | m5_stable_after_cost_baseline | baseline_delta_after_cost | rank_ic_mean | topk_minus_nextk_mean | max_industry_share_mean | concentration_pass_rate | industry_count_mean | baseline_gate | rank_gate | spread_gate | year_regime_gate | concentration_gate | cost_gate_10bps | m8_gate_pass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| U1_liquid_tradable | M8_regime_aware_fixed_policy | M8_regime_aware_fixed_policy__indcap3 | industry_names_cap | 3 | 20 | 0.0194528 | 0.009931 | 0.00952184 | 0.104108 | 0.014287 | 0.15 | 1 | 10.5641 | True | True | True | True | True | True | True |
| U1_liquid_tradable | M5_plus_industry_breadth_plus_fund_flow_plus_fundamental_elasticnet_excess | M5_plus_industry_breadth_plus_fund_flow_plus_fundamental_elasticnet_excess__indcap3 | industry_names_cap | 3 | 20 | 0.0146817 | 0.009931 | 0.00475072 | 0.104611 | 0.00730048 | 0.15 | 1 | 11.7179 | True | True | True | True | True | True | True |
| U1_liquid_tradable | M8_regime_aware_fixed_policy | M8_regime_aware_fixed_policy__uncapped | uncapped | 0 | 20 | 0.0132871 | 0.009931 | 0.00335615 | 0.104108 | -0.00224436 | 0.29359 | 0.717949 | 9 | True | True | False | True | False | True | False |
| U1_liquid_tradable | M5_plus_industry_breadth_plus_fund_flow_plus_fundamental_elasticnet_excess | M5_plus_industry_breadth_plus_fund_flow_plus_fundamental_elasticnet_excess__uncapped | uncapped | 0 | 20 | 0.009931 | 0.009931 | 0 | 0.104611 | 0.000204518 | 0.248718 | 0.846154 | 10.6923 | True | True | True | True | False | True | False |
| U1_liquid_tradable | M5_plus_industry_breadth_plus_fund_flow_plus_fundamental_extratrees_excess | M5_plus_industry_breadth_plus_fund_flow_plus_fundamental_extratrees_excess__indcap3 | industry_names_cap | 3 | 20 | 0.00650394 | 0.009931 | -0.00342706 | 0.110605 | 0.00217267 | 0.148718 | 1 | 10.641 | False | True | True | True | True | True | False |
| U1_liquid_tradable | M5_plus_industry_breadth_plus_fund_flow_plus_fundamental_extratrees_excess | M5_plus_industry_breadth_plus_fund_flow_plus_fundamental_extratrees_excess__uncapped | uncapped | 0 | 20 | 0.0064371 | 0.009931 | -0.0034939 | 0.110605 | 0.00151257 | 0.30641 | 0.641026 | 8.74359 | False | True | True | True | False | True | False |
| U1_liquid_tradable | M6_top20_calibrated | M6_top20_calibrated__indcap3 | industry_names_cap | 3 | 20 | -0.00187243 | 0.009931 | -0.0118034 | -0.058348 | -0.00298357 | 0.15 | 1 | 9.17949 | False | False | False | True | True | False | False |
| U1_liquid_tradable | M6_top20_calibrated | M6_top20_calibrated__uncapped | uncapped | 0 | 20 | -0.00280899 | 0.009931 | -0.01274 | -0.058348 | 0.0100743 | 0.65641 | 0.0769231 | 3.41026 | False | False | True | False | False | False | False |
| U1_liquid_tradable | M6_xgboost_rank_ndcg | M6_xgboost_rank_ndcg__uncapped | uncapped | 0 | 20 | -0.00519912 | 0.009931 | -0.0151301 | 0.0229536 | -0.00269697 | 0.476923 | 0.410256 | 7.4359 | False | True | False | True | False | False | False |
| U1_liquid_tradable | M6_xgboost_rank_ndcg | M6_xgboost_rank_ndcg__indcap3 | industry_names_cap | 3 | 20 | -0.00775603 | 0.009931 | -0.017687 | 0.0229536 | -0.00370739 | 0.15 | 1 | 11.1282 | False | True | False | True | True | False | False |

## Industry Concentration

| candidate_pool_version | model | top_k | selection_policy | max_industry_names | mean_max_industry_share | max_max_industry_share | mean_industry_count | pass_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| U1_liquid_tradable | M5_plus_industry_breadth_plus_fund_flow_plus_fundamental_extratrees_excess__indcap3 | 20 | industry_names_cap | 3 | 0.148718 | 0.15 | 10.641 | 1 |
| U1_liquid_tradable | M5_plus_industry_breadth_plus_fund_flow_plus_fundamental_elasticnet_excess__indcap3 | 20 | industry_names_cap | 3 | 0.15 | 0.15 | 11.7179 | 1 |
| U1_liquid_tradable | M6_top20_calibrated__indcap3 | 20 | industry_names_cap | 3 | 0.15 | 0.15 | 9.17949 | 1 |
| U1_liquid_tradable | M6_xgboost_rank_ndcg__indcap3 | 20 | industry_names_cap | 3 | 0.15 | 0.15 | 11.1282 | 1 |
| U1_liquid_tradable | M8_regime_aware_fixed_policy__indcap3 | 20 | industry_names_cap | 3 | 0.15 | 0.15 | 10.5641 | 1 |
| U1_liquid_tradable | M5_plus_industry_breadth_plus_fund_flow_plus_fundamental_elasticnet_excess__uncapped | 20 | uncapped | 0 | 0.248718 | 0.55 | 10.6923 | 0.846154 |
| U1_liquid_tradable | M8_regime_aware_fixed_policy__uncapped | 20 | uncapped | 0 | 0.29359 | 0.65 | 9 | 0.717949 |
| U1_liquid_tradable | M5_plus_industry_breadth_plus_fund_flow_plus_fundamental_extratrees_excess__uncapped | 20 | uncapped | 0 | 0.30641 | 0.6 | 8.74359 | 0.641026 |
| U1_liquid_tradable | M6_xgboost_rank_ndcg__uncapped | 20 | uncapped | 0 | 0.476923 | 1 | 7.4359 | 0.410256 |
| U1_liquid_tradable | M6_top20_calibrated__uncapped | 20 | uncapped | 0 | 0.65641 | 1 | 3.41026 | 0.0769231 |

## Year Slice

| candidate_pool_version | model | top_k | year | months | mean_topk_return | mean_topk_excess | median_topk_excess | hit_rate | mean_topk_minus_nextk |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| U1_liquid_tradable | M5_plus_industry_breadth_plus_fund_flow_plus_fundamental_elasticnet_excess__indcap3 | 20 | 2023 | 12 | -0.0100852 | 0.00842884 | 0.00200011 | 0.583333 | -0.00422032 |
| U1_liquid_tradable | M5_plus_industry_breadth_plus_fund_flow_plus_fundamental_elasticnet_excess__indcap3 | 20 | 2024 | 12 | 0.00161785 | 0.00748776 | 0.00125656 | 0.5 | 0.0149795 |
| U1_liquid_tradable | M5_plus_industry_breadth_plus_fund_flow_plus_fundamental_elasticnet_excess__indcap3 | 20 | 2025 | 12 | 0.0633376 | 0.0303623 | 0.022008 | 0.75 | 0.0099294 |
| U1_liquid_tradable | M5_plus_industry_breadth_plus_fund_flow_plus_fundamental_elasticnet_excess__indcap3 | 20 | 2026 | 3 | 0.0165217 | 0.0131266 | 0.0104371 | 0.666667 | 0.012152 |
| U1_liquid_tradable | M5_plus_industry_breadth_plus_fund_flow_plus_fundamental_elasticnet_excess__uncapped | 20 | 2023 | 12 | -0.0132802 | 0.00523389 | -0.00759122 | 0.416667 | -0.00020524 |
| U1_liquid_tradable | M5_plus_industry_breadth_plus_fund_flow_plus_fundamental_elasticnet_excess__uncapped | 20 | 2024 | 12 | -0.00219705 | 0.00367286 | -0.00303365 | 0.416667 | 0.00239705 |
| U1_liquid_tradable | M5_plus_industry_breadth_plus_fund_flow_plus_fundamental_elasticnet_excess__uncapped | 20 | 2025 | 12 | 0.0552418 | 0.0222664 | 0.0178286 | 0.666667 | -0.00694354 |
| U1_liquid_tradable | M5_plus_industry_breadth_plus_fund_flow_plus_fundamental_elasticnet_excess__uncapped | 20 | 2026 | 3 | 0.0164879 | 0.0130928 | 0.01628 | 0.666667 | 0.0216656 |
| U1_liquid_tradable | M5_plus_industry_breadth_plus_fund_flow_plus_fundamental_extratrees_excess__indcap3 | 20 | 2023 | 12 | -0.00483777 | 0.0136763 | 0.00308143 | 0.583333 | 7.705e-05 |
| U1_liquid_tradable | M5_plus_industry_breadth_plus_fund_flow_plus_fundamental_extratrees_excess__indcap3 | 20 | 2024 | 12 | -0.00885857 | -0.00298867 | -0.0107178 | 0.416667 | -0.00608772 |
| U1_liquid_tradable | M5_plus_industry_breadth_plus_fund_flow_plus_fundamental_extratrees_excess__indcap3 | 20 | 2025 | 12 | 0.048652 | 0.0156766 | 0.020201 | 0.666667 | 0.0121549 |
| U1_liquid_tradable | M5_plus_industry_breadth_plus_fund_flow_plus_fundamental_extratrees_excess__indcap3 | 20 | 2026 | 3 | 0.000595218 | -0.00279985 | 0.0204922 | 0.666667 | 0.00366769 |
| U1_liquid_tradable | M5_plus_industry_breadth_plus_fund_flow_plus_fundamental_extratrees_excess__uncapped | 20 | 2023 | 12 | -0.00901444 | 0.00949964 | -0.00312425 | 0.5 | -0.0148203 |
| U1_liquid_tradable | M5_plus_industry_breadth_plus_fund_flow_plus_fundamental_extratrees_excess__uncapped | 20 | 2024 | 12 | -0.00495541 | 0.000914497 | 0.00402257 | 0.5 | -0.000127193 |
| U1_liquid_tradable | M5_plus_industry_breadth_plus_fund_flow_plus_fundamental_extratrees_excess__uncapped | 20 | 2025 | 12 | 0.0504697 | 0.0174943 | 0.0222581 | 0.666667 | 0.0258567 |
| U1_liquid_tradable | M5_plus_industry_breadth_plus_fund_flow_plus_fundamental_extratrees_excess__uncapped | 20 | 2026 | 3 | -0.00545549 | -0.00885056 | 0.00313919 | 0.666667 | -0.0239734 |
| U1_liquid_tradable | M6_top20_calibrated__indcap3 | 20 | 2023 | 12 | -0.0305617 | -0.0120477 | -0.0266465 | 0.333333 | -0.00263765 |
| U1_liquid_tradable | M6_top20_calibrated__indcap3 | 20 | 2024 | 12 | -0.0172106 | -0.0113407 | -0.0238142 | 0.333333 | 0.0101531 |
| U1_liquid_tradable | M6_top20_calibrated__indcap3 | 20 | 2025 | 12 | 0.035338 | 0.00236269 | 0.00753304 | 0.583333 | -0.0171506 |
| U1_liquid_tradable | M6_top20_calibrated__indcap3 | 20 | 2026 | 3 | 0.06584 | 0.0624449 | 0.0572579 | 1 | -0.000245972 |
| U1_liquid_tradable | M6_top20_calibrated__uncapped | 20 | 2023 | 12 | -0.037525 | -0.0190109 | -0.00791798 | 0.416667 | -0.00930174 |
| U1_liquid_tradable | M6_top20_calibrated__uncapped | 20 | 2024 | 12 | -0.0150039 | -0.00913399 | -0.0200352 | 0.333333 | 0.0166951 |
| U1_liquid_tradable | M6_top20_calibrated__uncapped | 20 | 2025 | 12 | 0.0382124 | 0.00523703 | -0.0041761 | 0.416667 | 0.0249368 |
| U1_liquid_tradable | M6_top20_calibrated__uncapped | 20 | 2026 | 3 | 0.05947 | 0.0560749 | 0.0741496 | 1 | 0.00164451 |
| U1_liquid_tradable | M6_xgboost_rank_ndcg__indcap3 | 20 | 2023 | 12 | -0.0319649 | -0.0134508 | -0.0208871 | 0.25 | -0.00728776 |
| U1_liquid_tradable | M6_xgboost_rank_ndcg__indcap3 | 20 | 2024 | 12 | -0.00389835 | 0.00197156 | 0.0163268 | 0.583333 | -0.000689703 |
| U1_liquid_tradable | M6_xgboost_rank_ndcg__indcap3 | 20 | 2025 | 12 | 0.0205009 | -0.0124744 | -0.0211953 | 0.416667 | -0.0130717 |
| U1_liquid_tradable | M6_xgboost_rank_ndcg__indcap3 | 20 | 2026 | 3 | 0.00853123 | 0.00513616 | -0.0252802 | 0.333333 | 0.0360006 |
| U1_liquid_tradable | M6_xgboost_rank_ndcg__uncapped | 20 | 2023 | 12 | -0.0145204 | 0.00399367 | -0.0116189 | 0.333333 | 0.00423623 |
| U1_liquid_tradable | M6_xgboost_rank_ndcg__uncapped | 20 | 2024 | 12 | -0.012529 | -0.0066591 | -0.00505562 | 0.416667 | -0.0147307 |
| U1_liquid_tradable | M6_xgboost_rank_ndcg__uncapped | 20 | 2025 | 12 | 0.0257232 | -0.00725211 | -0.0240532 | 0.416667 | 0.00336277 |
| U1_liquid_tradable | M6_xgboost_rank_ndcg__uncapped | 20 | 2026 | 3 | -0.014373 | -0.017768 | -0.0252802 | 0.333333 | -0.00653369 |
| U1_liquid_tradable | M8_regime_aware_fixed_policy__indcap3 | 20 | 2023 | 12 | 0.00660961 | 0.0251237 | 0.0112281 | 0.833333 | 0.016047 |
| U1_liquid_tradable | M8_regime_aware_fixed_policy__indcap3 | 20 | 2024 | 12 | 0.00448982 | 0.0103597 | 0.0101372 | 0.583333 | 0.0106823 |
| U1_liquid_tradable | M8_regime_aware_fixed_policy__indcap3 | 20 | 2025 | 12 | 0.0624215 | 0.0294461 | 0.0333449 | 0.916667 | 0.0203693 |
| U1_liquid_tradable | M8_regime_aware_fixed_policy__indcap3 | 20 | 2026 | 3 | 0.0072434 | 0.00384833 | 0.0028764 | 0.666667 | -0.00266374 |
| U1_liquid_tradable | M8_regime_aware_fixed_policy__uncapped | 20 | 2023 | 12 | -0.0009074 | 0.0176067 | 0.0106127 | 0.75 | 0.00224122 |
| U1_liquid_tradable | M8_regime_aware_fixed_policy__uncapped | 20 | 2024 | 12 | 0.00342402 | 0.00929393 | 0.00423929 | 0.583333 | 0.00690571 |
| U1_liquid_tradable | M8_regime_aware_fixed_policy__uncapped | 20 | 2025 | 12 | 0.0531454 | 0.0201701 | 0.031772 | 0.666667 | -0.0112321 |
| U1_liquid_tradable | M8_regime_aware_fixed_policy__uncapped | 20 | 2026 | 3 | -0.00249464 | -0.00588971 | -0.00643569 | 0.333333 | -0.0208359 |

## Realized Market State Slice

| candidate_pool_version | model | top_k | realized_market_state | months | mean_topk_return | mean_topk_excess | median_topk_excess | hit_rate | mean_topk_minus_nextk |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| U1_liquid_tradable | M5_plus_industry_breadth_plus_fund_flow_plus_fundamental_elasticnet_excess__indcap3 | 20 | neutral | 25 | 0.0210333 | 0.019066 | 0.0020713 | 0.56 | 0.00981298 |
| U1_liquid_tradable | M5_plus_industry_breadth_plus_fund_flow_plus_fundamental_elasticnet_excess__indcap3 | 20 | strong_down | 7 | -0.0613709 | 0.0108361 | 0.00462486 | 0.857143 | 0.00862549 |
| U1_liquid_tradable | M5_plus_industry_breadth_plus_fund_flow_plus_fundamental_elasticnet_excess__indcap3 | 20 | strong_up | 7 | 0.087396 | 0.00603179 | 0.0183193 | 0.571429 | -0.00299773 |
| U1_liquid_tradable | M5_plus_industry_breadth_plus_fund_flow_plus_fundamental_elasticnet_excess__uncapped | 20 | neutral | 25 | 0.0165223 | 0.0145551 | 0.00137869 | 0.52 | 0.00590356 |
| U1_liquid_tradable | M5_plus_industry_breadth_plus_fund_flow_plus_fundamental_elasticnet_excess__uncapped | 20 | strong_down | 7 | -0.0659479 | 0.00625914 | 0.0107253 | 0.571429 | -0.00451683 |
| U1_liquid_tradable | M5_plus_industry_breadth_plus_fund_flow_plus_fundamental_elasticnet_excess__uncapped | 20 | strong_up | 7 | 0.0821736 | 0.00080943 | -0.00814103 | 0.428571 | -0.0154279 |
| U1_liquid_tradable | M5_plus_industry_breadth_plus_fund_flow_plus_fundamental_extratrees_excess__indcap3 | 20 | neutral | 25 | 0.0092481 | 0.00728083 | 0.00782732 | 0.6 | 0.000180498 |
| U1_liquid_tradable | M5_plus_industry_breadth_plus_fund_flow_plus_fundamental_extratrees_excess__indcap3 | 20 | strong_down | 7 | -0.068671 | 0.00353606 | -0.00463758 | 0.428571 | -0.00272839 |
| U1_liquid_tradable | M5_plus_industry_breadth_plus_fund_flow_plus_fundamental_extratrees_excess__indcap3 | 20 | strong_up | 7 | 0.0958211 | 0.0144569 | 0.0185264 | 0.571429 | 0.0141886 |
| U1_liquid_tradable | M5_plus_industry_breadth_plus_fund_flow_plus_fundamental_extratrees_excess__uncapped | 20 | neutral | 25 | 0.00991544 | 0.00794816 | 0.0169344 | 0.6 | 0.000816379 |
| U1_liquid_tradable | M5_plus_industry_breadth_plus_fund_flow_plus_fundamental_extratrees_excess__uncapped | 20 | strong_down | 7 | -0.0750377 | -0.00283058 | -0.00735076 | 0.428571 | -0.0186443 |
| U1_liquid_tradable | M5_plus_industry_breadth_plus_fund_flow_plus_fundamental_extratrees_excess__uncapped | 20 | strong_up | 7 | 0.0998585 | 0.0184942 | 0.022467 | 0.571429 | 0.0241559 |
| U1_liquid_tradable | M6_top20_calibrated__indcap3 | 20 | neutral | 25 | -0.0016401 | -0.00360737 | -0.0145653 | 0.4 | -0.00241617 |
| U1_liquid_tradable | M6_top20_calibrated__indcap3 | 20 | strong_down | 7 | -0.0726217 | -0.000414602 | 0.0110587 | 0.571429 | 0.0123592 |
| U1_liquid_tradable | M6_top20_calibrated__indcap3 | 20 | strong_up | 7 | 0.0853804 | 0.00401618 | 0.00756909 | 0.571429 | -0.0203528 |
| U1_liquid_tradable | M6_top20_calibrated__uncapped | 20 | neutral | 25 | 0.000311256 | -0.00165602 | -0.00561768 | 0.44 | 0.00948166 |
| U1_liquid_tradable | M6_top20_calibrated__uncapped | 20 | strong_down | 7 | -0.0870547 | -0.0148476 | -0.0351222 | 0.428571 | -0.00900637 |
| U1_liquid_tradable | M6_top20_calibrated__uncapped | 20 | strong_up | 7 | 0.0868877 | 0.0055235 | -0.00273452 | 0.428571 | 0.0312713 |
| U1_liquid_tradable | M6_xgboost_rank_ndcg__indcap3 | 20 | neutral | 25 | -0.0155783 | -0.0175456 | -0.0252802 | 0.36 | -0.0104175 |
| U1_liquid_tradable | M6_xgboost_rank_ndcg__indcap3 | 20 | strong_down | 7 | -0.0515168 | 0.0206902 | -0.00706363 | 0.428571 | 0.0141887 |
| U1_liquid_tradable | M6_xgboost_rank_ndcg__indcap3 | 20 | strong_up | 7 | 0.0844746 | 0.00311036 | 0.0239633 | 0.571429 | 0.00236133 |
| U1_liquid_tradable | M6_xgboost_rank_ndcg__uncapped | 20 | neutral | 25 | -0.0132418 | -0.0152091 | -0.0138396 | 0.36 | -0.00213077 |
| U1_liquid_tradable | M6_xgboost_rank_ndcg__uncapped | 20 | strong_down | 7 | -0.0454906 | 0.0267165 | -0.00403328 | 0.285714 | -0.0104561 |
| U1_liquid_tradable | M6_xgboost_rank_ndcg__uncapped | 20 | strong_up | 7 | 0.0843495 | 0.00298529 | 0.00812581 | 0.571429 | 0.00304 |
| U1_liquid_tradable | M8_regime_aware_fixed_policy__indcap3 | 20 | neutral | 25 | 0.0222899 | 0.0203226 | 0.0153415 | 0.76 | 0.0124249 |
| U1_liquid_tradable | M8_regime_aware_fixed_policy__indcap3 | 20 | strong_down | 7 | -0.0490634 | 0.0231437 | 0.0123861 | 0.857143 | 0.0181155 |
| U1_liquid_tradable | M8_regime_aware_fixed_policy__indcap3 | 20 | strong_up | 7 | 0.0985967 | 0.0172325 | 0.00716442 | 0.714286 | 0.0171086 |
| U1_liquid_tradable | M8_regime_aware_fixed_policy__uncapped | 20 | neutral | 25 | 0.0202944 | 0.0183272 | 0.015072 | 0.68 | 0.00866245 |
| U1_liquid_tradable | M8_regime_aware_fixed_policy__uncapped | 20 | strong_down | 7 | -0.0595715 | 0.0126355 | 0.00745117 | 0.714286 | -0.00462323 |
| U1_liquid_tradable | M8_regime_aware_fixed_policy__uncapped | 20 | strong_up | 7 | 0.081443 | 7.8752e-05 | -0.0115867 | 0.428571 | -0.0388184 |

## 本轮结论

- 本轮新增的是选择层行业只数上限，而不是继续堆模型；unconstrained 行作为对照保留。
- `M8_regime_aware_fixed_policy` 只使用 lagged realized market state 和信号日可见 breadth，权重固定，不按测试月收益回填。
- `gate.csv` 是进入后续 M9/M10 的研究证据表；本脚本不写 promoted registry，也不改生产配置。
- 若候选通过集中度但收益、Top-K vs next-K 或 regime 切片失败，结论应是继续研究，不 promotion。

## 本轮产物

- `data/results/monthly_selection_m8_concentration_regime_2026-05-01_summary.json`
- `data/results/monthly_selection_m8_concentration_regime_2026-05-01_leaderboard.csv`
- `data/results/monthly_selection_m8_concentration_regime_2026-05-01_monthly_long.csv`
- `data/results/monthly_selection_m8_concentration_regime_2026-05-01_industry_concentration.csv`
- `data/results/monthly_selection_m8_concentration_regime_2026-05-01_industry_exposure.csv`
- `data/results/monthly_selection_m8_concentration_regime_2026-05-01_regime_slice.csv`
- `data/results/monthly_selection_m8_concentration_regime_2026-05-01_year_slice.csv`
- `data/results/monthly_selection_m8_concentration_regime_2026-05-01_topk_holdings.csv`
- `data/results/monthly_selection_m8_concentration_regime_2026-05-01_gate.csv`
- `data/results/monthly_selection_m8_concentration_regime_2026-05-01_rank_ic.csv`
- `data/results/monthly_selection_m8_concentration_regime_2026-05-01_quantile_spread.csv`
- `data/results/monthly_selection_m8_concentration_regime_2026-05-01_lagged_states.csv`
- `data/results/monthly_selection_m8_concentration_regime_2026-05-01_feature_coverage.csv`
- `data/results/monthly_selection_m8_concentration_regime_2026-05-01_feature_importance.csv`
- `data/results/monthly_selection_m8_concentration_regime_2026-05-01_candidate_pool_width.csv`
- `data/results/monthly_selection_m8_concentration_regime_2026-05-01_candidate_pool_reject_reason.csv`
- `data/results/monthly_selection_m8_concentration_regime_2026-05-01_manifest.json`
