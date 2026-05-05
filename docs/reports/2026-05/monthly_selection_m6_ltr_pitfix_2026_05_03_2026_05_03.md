# Monthly Selection M6 Learning-to-Rank

- 生成时间：`2026-05-03T05:13:12.567063+00:00`
- 结果类型：`monthly_selection_m6_ltr`
- 研究主题：`monthly_selection_m6_ltr`
- 研究配置：`dataset_monthly_selection_features_families_price_volume_only_industry_breadth_fund_flow_fundamental_pools_u1_liquid_tradable_u2_risk_sane_topk_20_30_50_models_xgboost_rank_ndcg_xgboost_rank_pairwise_top20_calibrated_ranker_top20_ensemble_grades_5_maxfit_0_jobs_all_wf_24m_costbps_10_0`
- 输出 stem：`monthly_selection_m6_ltr_pitfix_2026_05_03_2026_05_03`
- 数据集：`data/cache/monthly_selection_features.parquet`
- 数据库：`data/market.duckdb`
- 训练/评估：按 signal_date walk-forward；每个测试月只用历史月份训练。
- 有效标签月份：`63`
- 单窗训练行上限：`0`（`0` 表示不抽样）

## Leaderboard

| candidate_pool_version | model | model_type | top_k | months | mean_topk_return | median_topk_return | topk_excess_mean | topk_excess_median | topk_hit_rate | topk_minus_nextk_mean | industry_neutral_topk_excess_mean | turnover_mean | cost_drag_mean | topk_excess_after_cost_mean | topk_excess_annualized | rank_ic_mean | rank_ic_std | rank_ic_months | rank_ic_ir | quantile_top_minus_bottom_mean | neutral_median_excess | strong_down_median_excess | strong_up_median_excess |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| U1_liquid_tradable | M6_xgboost_rank_ndcg | xgboost_ranker | 20 | 39 | 0.0133695 | 0.00569568 | 0.0104648 | 0.00675653 | 0.564103 | 0.00624433 | 0.00821476 | 0.871053 | 0.000871053 | 0.00969885 | 0.133064 | 0.0181123 | 0.103873 | 39 | 0.174369 | 0.00570137 | 0.00665064 | 0.0317323 | -0.0119531 |
| U1_liquid_tradable | M6_xgboost_rank_pairwise | xgboost_ranker | 20 | 39 | 0.0119284 | 0.0194197 | 0.00902377 | 0.0140331 | 0.589744 | 0.00275812 | 0.0100994 | 0.911842 | 0.000911842 | 0.00801506 | 0.113824 | 0.0647914 | 0.142573 | 39 | 0.454445 | 0.00463435 | 0.0151138 | 0.0169571 | -0.00257178 |
| U1_liquid_tradable | B0_market_ew | benchmark | 20 | 39 | 0.00290466 | 0.00265337 | 0 | 0 | 0 |  |  |  | 0 | 0 | 0 |  |  |  |  |  | 0 | 0 | 0 |
| U1_liquid_tradable | M6_top20_calibrated | top_bucket_classifier_rank_calibrated | 20 | 39 | -0.00434218 | -0.00781339 | -0.00724684 | -0.0111613 | 0.435897 | 0.00281573 | -0.00287714 | 0.980263 | 0.000980263 | -0.00751005 | -0.0835784 | -0.0418853 | 0.141487 | 39 | -0.296036 | -0.000850439 | -0.0113852 | -0.0388671 | 0.0081708 |
| U1_liquid_tradable | M6_ranker_top20_ensemble | ranker_classifier_ensemble | 20 | 39 | -0.00736994 | -0.00524388 | -0.0102746 | -0.0187114 | 0.435897 | -0.0171359 | -0.0019208 | 0.957895 | 0.000957895 | -0.0107403 | -0.116561 | -0.00678755 | 0.0858886 | 39 | -0.0790274 | 0.00471686 | -0.021075 | -0.0166634 | -0.0196919 |
| U2_risk_sane | M6_xgboost_rank_ndcg | xgboost_ranker | 20 | 39 | 0.0137453 | 0.0145945 | 0.0108406 | -0.00285024 | 0.487179 | 0.00546731 | 0.000654089 | 0.842105 | 0.000842105 | 0.0104625 | 0.138131 | 0.0266043 | 0.0878344 | 39 | 0.302891 | 0.00855749 | -0.00285024 | 0.0447977 | -0.0152324 |
| U2_risk_sane | M6_xgboost_rank_pairwise | xgboost_ranker | 20 | 39 | 0.0120926 | 0.0223192 | 0.00918795 | 0.00736198 | 0.666667 | 0.00741581 | 0.00232671 | 0.944737 | 0.000944737 | 0.00960524 | 0.116001 | 0.0239268 | 0.136483 | 39 | 0.17531 | -0.000117369 | 0.00916589 | 0.00844243 | -0.0360816 |
| U2_risk_sane | M6_ranker_top20_ensemble | ranker_classifier_ensemble | 20 | 39 | 0.00341078 | 0.00947348 | 0.000506125 | 0.00096357 | 0.512821 | -0.000972688 | 0.00302105 | 0.952632 | 0.000952632 | 0.000250462 | 0.00609044 | 0.00411326 | 0.0921316 | 39 | 0.0446454 | 0.00647954 | 0.00096357 | -0.0412106 | 0.0216678 |
| U2_risk_sane | B0_market_ew | benchmark | 20 | 39 | 0.00290466 | 0.00265337 | 0 | 0 | 0 |  |  |  | 0 | 0 | 0 |  |  |  |  |  | 0 | 0 | 0 |
| U2_risk_sane | M6_top20_calibrated | top_bucket_classifier_rank_calibrated | 20 | 39 | -0.000877053 | -0.00473733 | -0.00378171 | -0.00577955 | 0.384615 | 0.0015229 | -0.00413982 | 0.977632 | 0.000977632 | -0.0039044 | -0.0444484 | -0.0333103 | 0.144956 | 39 | -0.229796 | 0.00269216 | -0.00443264 | -0.0159024 | 0.0308682 |
| U1_liquid_tradable | M6_xgboost_rank_pairwise | xgboost_ranker | 30 | 39 | 0.0121817 | 0.021728 | 0.00927706 | 0.0134802 | 0.641026 | 0.00720477 | 0.00851357 | 0.892105 | 0.000892105 | 0.00867941 | 0.117184 | 0.0647914 | 0.142573 | 39 | 0.454445 | 0.00463435 | 0.0139416 | 0.0117541 | 0.00928388 |
| U1_liquid_tradable | M6_xgboost_rank_ndcg | xgboost_ranker | 30 | 39 | 0.0109737 | 0.00343241 | 0.00806907 | 0.00563682 | 0.538462 | 0.00583841 | 0.00640471 | 0.845614 | 0.000845614 | 0.00728747 | 0.101244 | 0.0181123 | 0.103873 | 39 | 0.174369 | 0.00570137 | -0.000514132 | 0.0249867 | -0.0079371 |
| U1_liquid_tradable | B0_market_ew | benchmark | 30 | 39 | 0.00290466 | 0.00265337 | 0 | 0 | 0 |  |  |  | 0 | 0 | 0 |  |  |  |  |  | 0 | 0 | 0 |
| U1_liquid_tradable | M6_ranker_top20_ensemble | ranker_classifier_ensemble | 30 | 39 | -0.00234147 | -0.00533728 | -0.00524612 | -0.00534575 | 0.487179 | -0.00888847 | 0.00218239 | 0.95 | 0.00095 | -0.00562851 | -0.0611684 | -0.00678755 | 0.0858886 | 39 | -0.0790274 | 0.00471686 | -0.00703319 | 0.0089799 | 0.00664454 |
| U1_liquid_tradable | M6_top20_calibrated | top_bucket_classifier_rank_calibrated | 30 | 39 | -0.00471157 | -0.00634293 | -0.00761623 | -0.0132182 | 0.358974 | 0.00246713 | -0.00395547 | 0.972807 | 0.000972807 | -0.00782082 | -0.0876619 | -0.0418853 | 0.141487 | 39 | -0.296036 | -0.000850439 | -0.0141724 | -0.0229718 | -0.00450357 |
| U2_risk_sane | M6_xgboost_rank_ndcg | xgboost_ranker | 30 | 39 | 0.0132426 | 0.0149775 | 0.010338 | -0.00254531 | 0.487179 | 0.009867 | 0.000153828 | 0.796491 | 0.000796491 | 0.00988053 | 0.131358 | 0.0266043 | 0.0878344 | 39 | 0.302891 | 0.00855749 | -0.00254531 | 0.0423631 | -0.024917 |
| U2_risk_sane | M6_xgboost_rank_pairwise | xgboost_ranker | 30 | 39 | 0.0105394 | 0.00677668 | 0.0076347 | 0.00470012 | 0.564103 | 0.0110734 | 0.00295491 | 0.926316 | 0.000926316 | 0.00781923 | 0.0955631 | 0.0239268 | 0.136483 | 39 | 0.17531 | -0.000117369 | 0.0089842 | -0.00167576 | -0.0450017 |
| U2_risk_sane | M6_ranker_top20_ensemble | ranker_classifier_ensemble | 30 | 39 | 0.00368922 | 0.00333875 | 0.000784567 | 0.0067998 | 0.564103 | 0.000776865 | 0.00275073 | 0.936842 | 0.000936842 | 0.000402616 | 0.00945553 | 0.00411326 | 0.0921316 | 39 | 0.0446454 | 0.00647954 | 0.00966288 | -0.0425632 | 0.0067998 |
| U2_risk_sane | B0_market_ew | benchmark | 30 | 39 | 0.00290466 | 0.00265337 | 0 | 0 | 0 |  |  |  | 0 | 0 | 0 |  |  |  |  |  | 0 | 0 | 0 |
| U2_risk_sane | M6_top20_calibrated | top_bucket_classifier_rank_calibrated | 30 | 39 | -0.00188328 | -0.0104288 | -0.00478794 | -0.00509249 | 0.435897 | 0.00523414 | -0.0041707 | 0.97193 | 0.00097193 | -0.00485459 | -0.0559661 | -0.0333103 | 0.144956 | 39 | -0.229796 | 0.00269216 | -0.00509249 | -0.0220344 | 0.02302 |
| U1_liquid_tradable | M6_xgboost_rank_ndcg | xgboost_ranker | 50 | 39 | 0.0101782 | 0.00738239 | 0.00727359 | 0.00492635 | 0.538462 | 0.00674837 | 0.0056001 | 0.821053 | 0.000821053 | 0.00684366 | 0.0908609 | 0.0181123 | 0.103873 | 39 | 0.174369 | 0.00570137 | -0.00390479 | 0.0236896 | 0.00492635 |
| U1_liquid_tradable | M6_xgboost_rank_pairwise | xgboost_ranker | 50 | 39 | 0.00907428 | 0.0162721 | 0.00616962 | 0.0120343 | 0.615385 | 0.00939525 | 0.00510806 | 0.823684 | 0.000823684 | 0.00588619 | 0.0766 | 0.0647914 | 0.142573 | 39 | 0.454445 | 0.00463435 | 0.0120343 | 0.0116088 | 0.0121358 |
| U1_liquid_tradable | B0_market_ew | benchmark | 50 | 39 | 0.00290466 | 0.00265337 | 0 | 0 | 0 |  |  |  | 0 | 0 | 0 |  |  |  |  |  | 0 | 0 | 0 |
| U1_liquid_tradable | M6_ranker_top20_ensemble | ranker_classifier_ensemble | 50 | 39 | 0.00122954 | 0.00248381 | -0.00167512 | -0.00366508 | 0.487179 | -0.00206214 | 0.00422232 | 0.934211 | 0.000934211 | -0.00232557 | -0.0199173 | -0.00678755 | 0.0858886 | 39 | -0.0790274 | 0.00471686 | -0.00507195 | 0.0222701 | 0.0113865 |
| U1_liquid_tradable | M6_top20_calibrated | top_bucket_classifier_rank_calibrated | 50 | 39 | -0.00499703 | -0.0104636 | -0.00790169 | -0.00696853 | 0.358974 | -0.0109078 | -0.00417259 | 0.956842 | 0.000956842 | -0.00789923 | -0.0908061 | -0.0418853 | 0.141487 | 39 | -0.296036 | -0.000850439 | -0.00696853 | -0.0172294 | -0.00486639 |
| U2_risk_sane | M6_xgboost_rank_ndcg | xgboost_ranker | 50 | 39 | 0.0105602 | 0.00991304 | 0.00765555 | 0.00376367 | 0.538462 | 0.0106553 | 2.54104e-05 | 0.717895 | 0.000717895 | 0.00727751 | 0.0958351 | 0.0266043 | 0.0878344 | 39 | 0.302891 | 0.00855749 | 0.00371932 | 0.0243509 | -0.0281344 |
| U2_risk_sane | M6_xgboost_rank_pairwise | xgboost_ranker | 50 | 39 | 0.0056554 | 0.00902805 | 0.00275074 | 0.001031 | 0.512821 | 0.00343428 | 0.000915496 | 0.891053 | 0.000891053 | 0.00262795 | 0.0335129 | 0.0239268 | 0.136483 | 39 | 0.17531 | -0.000117369 | 0.00115426 | 0.001031 | -0.0391648 |
| U2_risk_sane | M6_ranker_top20_ensemble | ranker_classifier_ensemble | 50 | 39 | 0.00373781 | -0.00135652 | 0.000833149 | 0.00455281 | 0.564103 | -0.00417751 | 0.00240034 | 0.922105 | 0.000922105 | 0.000553752 | 0.0100437 | 0.00411326 | 0.0921316 | 39 | 0.0446454 | 0.00647954 | 0.00455281 | -0.0300937 | 0.0264826 |
| U2_risk_sane | B0_market_ew | benchmark | 50 | 39 | 0.00290466 | 0.00265337 | 0 | 0 | 0 |  |  |  | 0 | 0 | 0 |  |  |  |  |  | 0 | 0 | 0 |
| U2_risk_sane | M6_top20_calibrated | top_bucket_classifier_rank_calibrated | 50 | 39 | -0.00261952 | -0.00553948 | -0.00552418 | -0.00129989 | 0.435897 | -0.00392723 | -0.00360954 | 0.961053 | 0.000961053 | -0.0059988 | -0.0643127 | -0.0333103 | 0.144956 | 39 | -0.229796 | 0.00269216 | -0.000686254 | -0.0221893 | 0.00239916 |

## U1 Top20 Leading Models

| candidate_pool_version | model | model_type | top_k | months | mean_topk_return | median_topk_return | topk_excess_mean | topk_excess_median | topk_hit_rate | topk_minus_nextk_mean | industry_neutral_topk_excess_mean | turnover_mean | cost_drag_mean | topk_excess_after_cost_mean | topk_excess_annualized | rank_ic_mean | rank_ic_std | rank_ic_months | rank_ic_ir | quantile_top_minus_bottom_mean | neutral_median_excess | strong_down_median_excess | strong_up_median_excess |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| U1_liquid_tradable | M6_xgboost_rank_ndcg | xgboost_ranker | 20 | 39 | 0.0133695 | 0.00569568 | 0.0104648 | 0.00675653 | 0.564103 | 0.00624433 | 0.00821476 | 0.871053 | 0.000871053 | 0.00969885 | 0.133064 | 0.0181123 | 0.103873 | 39 | 0.174369 | 0.00570137 | 0.00665064 | 0.0317323 | -0.0119531 |
| U1_liquid_tradable | M6_xgboost_rank_pairwise | xgboost_ranker | 20 | 39 | 0.0119284 | 0.0194197 | 0.00902377 | 0.0140331 | 0.589744 | 0.00275812 | 0.0100994 | 0.911842 | 0.000911842 | 0.00801506 | 0.113824 | 0.0647914 | 0.142573 | 39 | 0.454445 | 0.00463435 | 0.0151138 | 0.0169571 | -0.00257178 |
| U1_liquid_tradable | B0_market_ew | benchmark | 20 | 39 | 0.00290466 | 0.00265337 | 0 | 0 | 0 |  |  |  | 0 | 0 | 0 |  |  |  |  |  | 0 | 0 | 0 |
| U1_liquid_tradable | M6_top20_calibrated | top_bucket_classifier_rank_calibrated | 20 | 39 | -0.00434218 | -0.00781339 | -0.00724684 | -0.0111613 | 0.435897 | 0.00281573 | -0.00287714 | 0.980263 | 0.000980263 | -0.00751005 | -0.0835784 | -0.0418853 | 0.141487 | 39 | -0.296036 | -0.000850439 | -0.0113852 | -0.0388671 | 0.0081708 |
| U1_liquid_tradable | M6_ranker_top20_ensemble | ranker_classifier_ensemble | 20 | 39 | -0.00736994 | -0.00524388 | -0.0102746 | -0.0187114 | 0.435897 | -0.0171359 | -0.0019208 | 0.957895 | 0.000957895 | -0.0107403 | -0.116561 | -0.00678755 | 0.0858886 | 39 | -0.0790274 | 0.00471686 | -0.021075 | -0.0166634 | -0.0196919 |

## Feature Coverage

| feature_spec | families | feature | raw_feature | rows | non_null | coverage_ratio | candidate_pool_pass_coverage_ratio | first_signal_date | last_signal_date |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_ret_20d_z | feature_ret_20d | 307350 | 306083 | 0.995878 | 1 | 2021-01-29 | 2026-04-30 |
| m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_ret_60d_z | feature_ret_60d | 307350 | 303506 | 0.987493 | 1 | 2021-01-29 | 2026-04-30 |
| m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_ret_5d_z | feature_ret_5d | 307350 | 307017 | 0.998917 | 1 | 2021-01-29 | 2026-04-30 |
| m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_realized_vol_20d_z | feature_realized_vol_20d | 307350 | 306673 | 0.997797 | 1 | 2021-01-29 | 2026-04-30 |
| m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_amount_20d_log_z | feature_amount_20d_log | 307350 | 306745 | 0.998032 | 1 | 2021-01-29 | 2026-04-30 |
| m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_turnover_20d_z | feature_turnover_20d | 307350 | 306745 | 0.998032 | 1 | 2021-01-29 | 2026-04-30 |
| m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_price_position_250d_z | feature_price_position_250d | 307350 | 303573 | 0.987711 | 1 | 2021-01-29 | 2026-04-30 |
| m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_limit_move_hits_20d_z | feature_limit_move_hits_20d | 307350 | 307350 | 1 | 1 | 2021-01-29 | 2026-04-30 |
| m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_industry_ret20_mean_z | feature_industry_ret20_mean | 307350 | 307350 | 1 | 1 | 2021-01-29 | 2026-04-30 |
| m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_industry_ret60_mean_z | feature_industry_ret60_mean | 307350 | 307350 | 1 | 1 | 2021-01-29 | 2026-04-30 |
| m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_industry_positive_ret20_ratio_z | feature_industry_positive_ret20_ratio | 307350 | 307350 | 1 | 1 | 2021-01-29 | 2026-04-30 |
| m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_industry_amount20_mean_z | feature_industry_amount20_mean | 307350 | 307350 | 1 | 1 | 2021-01-29 | 2026-04-30 |
| m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_industry_low_vol20_mean_z | feature_industry_low_vol20_mean | 307350 | 307350 | 1 | 1 | 2021-01-29 | 2026-04-30 |
| m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_fund_flow_main_inflow_5d_z | feature_fund_flow_main_inflow_5d | 307350 | 35590 | 0.115796 | 0.127948 | 2023-10-31 | 2026-04-30 |
| m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_fund_flow_main_inflow_10d_z | feature_fund_flow_main_inflow_10d | 307350 | 31196 | 0.1015 | 0.10964 | 2023-11-30 | 2026-04-30 |
| m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_fund_flow_main_inflow_20d_z | feature_fund_flow_main_inflow_20d | 307350 | 31048 | 0.101018 | 0.109138 | 2023-11-30 | 2026-04-30 |
| m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_fund_flow_super_inflow_10d_z | feature_fund_flow_super_inflow_10d | 307350 | 31196 | 0.1015 | 0.10964 | 2023-11-30 | 2026-04-30 |
| m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_fund_flow_divergence_20d_z | feature_fund_flow_divergence_20d | 307350 | 31048 | 0.101018 | 0.109138 | 2023-11-30 | 2026-04-30 |
| m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_fund_flow_main_inflow_streak_z | feature_fund_flow_main_inflow_streak | 307350 | 36162 | 0.117657 | 0.130348 | 2023-10-31 | 2026-04-30 |
| m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_fundamental_pe_ttm_z | feature_fundamental_pe_ttm | 307350 | 304469 | 0.990626 | 0.999766 | 2021-01-29 | 2026-04-30 |
| m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_fundamental_pb_z | feature_fundamental_pb | 307350 | 304469 | 0.990626 | 0.999766 | 2021-01-29 | 2026-04-30 |
| m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_fundamental_ev_ebitda_z | feature_fundamental_ev_ebitda | 307350 | 0 | 0 | 0 |  |  |
| m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_fundamental_roe_ttm_z | feature_fundamental_roe_ttm | 307350 | 304617 | 0.991108 | 0.992871 | 2021-01-29 | 2026-04-30 |
| m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_fundamental_net_profit_yoy_z | feature_fundamental_net_profit_yoy | 307350 | 306367 | 0.996802 | 0.997055 | 2021-01-29 | 2026-04-30 |
| m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_fundamental_gross_margin_change_z | feature_fundamental_gross_margin_change | 307350 | 299889 | 0.975725 | 0.96863 | 2021-01-29 | 2026-04-30 |
| m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_fundamental_gross_margin_delta_z | feature_fundamental_gross_margin_delta | 307350 | 299765 | 0.975321 | 0.968386 | 2021-01-29 | 2026-04-30 |
| m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_fundamental_debt_to_assets_change_z | feature_fundamental_debt_to_assets_change | 307350 | 306206 | 0.996278 | 0.997055 | 2021-01-29 | 2026-04-30 |
| m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_fundamental_ocf_to_net_profit_z | feature_fundamental_ocf_to_net_profit | 307350 | 245847 | 0.799893 | 0.811349 | 2021-01-29 | 2026-04-30 |
| m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_fundamental_ocf_to_asset_z | feature_fundamental_ocf_to_asset | 307350 | 306219 | 0.99632 | 0.996596 | 2021-01-29 | 2026-04-30 |
| m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_fundamental_asset_turnover_z | feature_fundamental_asset_turnover | 307350 | 306300 | 0.996584 | 0.996911 | 2021-01-29 | 2026-04-30 |
| m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_fundamental_net_margin_stability_z | feature_fundamental_net_margin_stability | 307350 | 306150 | 0.996096 | 0.996677 | 2021-01-29 | 2026-04-30 |

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
| U1_liquid_tradable | M6_ranker_top20_ensemble | 20 | 2023 | 12 | -0.0308347 | -0.0123206 | -0.0243447 | 0.333333 | -0.00635127 |
| U1_liquid_tradable | M6_ranker_top20_ensemble | 20 | 2024 | 12 | -0.0154052 | -0.00953527 | -0.0192017 | 0.416667 | -0.0187455 |
| U1_liquid_tradable | M6_ranker_top20_ensemble | 20 | 2025 | 12 | 0.023877 | -0.00909835 | 0.00153468 | 0.5 | -0.0139507 |
| U1_liquid_tradable | M6_ranker_top20_ensemble | 20 | 2026 | 3 | -0.00635765 | -0.00975272 | 0.00695692 | 0.666667 | -0.0665774 |
| U1_liquid_tradable | M6_ranker_top20_ensemble | 30 | 2023 | 12 | -0.0316059 | -0.0130918 | -0.0215813 | 0.333333 | -0.0068184 |
| U1_liquid_tradable | M6_ranker_top20_ensemble | 30 | 2024 | 12 | -0.00820648 | -0.00233658 | 0.00777646 | 0.583333 | -0.0137898 |
| U1_liquid_tradable | M6_ranker_top20_ensemble | 30 | 2025 | 12 | 0.0266153 | -0.00636004 | -0.0011857 | 0.5 | -0.00418227 |
| U1_liquid_tradable | M6_ranker_top20_ensemble | 30 | 2026 | 3 | 0.0223491 | 0.018954 | 0.0171251 | 0.666667 | -0.0163881 |
| U1_liquid_tradable | M6_ranker_top20_ensemble | 50 | 2023 | 12 | -0.0295374 | -0.0110233 | -0.0162398 | 0.25 | -0.00513378 |
| U1_liquid_tradable | M6_ranker_top20_ensemble | 50 | 2024 | 12 | -0.00347484 | 0.00239507 | 0.0129509 | 0.666667 | 0.00680633 |
| U1_liquid_tradable | M6_ranker_top20_ensemble | 50 | 2025 | 12 | 0.0291875 | -0.0037879 | 0.000592137 | 0.5 | -0.00673978 |
| U1_liquid_tradable | M6_ranker_top20_ensemble | 50 | 2026 | 3 | 0.031283 | 0.0278879 | 0.0141821 | 0.666667 | -0.00653885 |
| U1_liquid_tradable | M6_top20_calibrated | 20 | 2023 | 12 | -0.0277934 | -0.00927934 | -0.00790043 | 0.416667 | 0.0131335 |
| U1_liquid_tradable | M6_top20_calibrated | 20 | 2024 | 12 | -0.0204001 | -0.0145302 | -0.0112208 | 0.5 | -0.00925285 |
| U1_liquid_tradable | M6_top20_calibrated | 20 | 2025 | 12 | 0.0270396 | -0.00593578 | -0.0135417 | 0.333333 | 0.00833645 |
| U1_liquid_tradable | M6_top20_calibrated | 20 | 2026 | 3 | 0.0281674 | 0.0247723 | 0.0406443 | 0.666667 | -0.0122639 |
| U1_liquid_tradable | M6_top20_calibrated | 30 | 2023 | 12 | -0.0295162 | -0.0110021 | -0.00806412 | 0.5 | 0.00939533 |
| U1_liquid_tradable | M6_top20_calibrated | 30 | 2024 | 12 | -0.0185301 | -0.0126602 | -0.0133703 | 0.333333 | -0.00666275 |
| U1_liquid_tradable | M6_top20_calibrated | 30 | 2025 | 12 | 0.0240319 | -0.00894348 | -0.0137875 | 0.166667 | 0.00516561 |
| U1_liquid_tradable | M6_top20_calibrated | 30 | 2026 | 3 | 0.0348074 | 0.0314124 | 0.0404359 | 0.666667 | 0.000479966 |
| U1_liquid_tradable | M6_top20_calibrated | 50 | 2023 | 12 | -0.0332874 | -0.0147733 | -0.0179168 | 0.416667 | -0.000503575 |
| U1_liquid_tradable | M6_top20_calibrated | 50 | 2024 | 12 | -0.0166452 | -0.0107753 | -0.0091722 | 0.333333 | -0.0162199 |
| U1_liquid_tradable | M6_top20_calibrated | 50 | 2025 | 12 | 0.0247526 | -0.00822273 | -0.00591746 | 0.25 | -0.0113928 |
| U1_liquid_tradable | M6_top20_calibrated | 50 | 2026 | 3 | 0.0357585 | 0.0323635 | 0.0369126 | 0.666667 | -0.0293357 |
| U1_liquid_tradable | M6_xgboost_rank_ndcg | 20 | 2023 | 12 | -0.00150904 | 0.017005 | 0.00949783 | 0.583333 | 0.0100017 |
| U1_liquid_tradable | M6_xgboost_rank_ndcg | 20 | 2024 | 12 | -0.00575888 | 0.000111023 | 0.00830356 | 0.583333 | -0.00156184 |
| U1_liquid_tradable | M6_xgboost_rank_ndcg | 20 | 2025 | 12 | 0.0478646 | 0.0148893 | -0.0010666 | 0.5 | 0.0152408 |
| U1_liquid_tradable | M6_xgboost_rank_ndcg | 20 | 2026 | 3 | 0.0114166 | 0.00802149 | 0.0550527 | 0.666667 | -0.0135466 |

## Realized Market State Slice

| candidate_pool_version | model | top_k | realized_market_state | months | mean_topk_return | mean_topk_excess | median_topk_excess | hit_rate | mean_topk_minus_nextk |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| U1_liquid_tradable | B0_market_ew | 20 | neutral | 25 | 0.00196727 | 0 | 0 | 0 |  |
| U1_liquid_tradable | B0_market_ew | 20 | strong_down | 7 | -0.0722071 | 0 | 0 | 0 |  |
| U1_liquid_tradable | B0_market_ew | 20 | strong_up | 7 | 0.0813642 | 0 | 0 | 0 |  |
| U1_liquid_tradable | M6_ranker_top20_ensemble | 20 | neutral | 25 | -0.00749548 | -0.00946276 | -0.021075 | 0.44 | -0.0045132 |
| U1_liquid_tradable | M6_ranker_top20_ensemble | 20 | strong_down | 7 | -0.0855651 | -0.013358 | -0.0166634 | 0.428571 | -0.0517171 |
| U1_liquid_tradable | M6_ranker_top20_ensemble | 20 | strong_up | 7 | 0.0712736 | -0.0100906 | -0.0196919 | 0.428571 | -0.027636 |
| U1_liquid_tradable | M6_top20_calibrated | 20 | neutral | 25 | -0.00669738 | -0.00866465 | -0.0113852 | 0.4 | 0.00124915 |
| U1_liquid_tradable | M6_top20_calibrated | 20 | strong_down | 7 | -0.0835765 | -0.0113694 | -0.0388671 | 0.428571 | -0.00104625 |
| U1_liquid_tradable | M6_top20_calibrated | 20 | strong_up | 7 | 0.0833035 | 0.00193931 | 0.0081708 | 0.571429 | 0.0122726 |
| U1_liquid_tradable | M6_xgboost_rank_ndcg | 20 | neutral | 25 | 0.00503804 | 0.00307077 | 0.00665064 | 0.56 | 0.00972191 |
| U1_liquid_tradable | M6_xgboost_rank_ndcg | 20 | strong_down | 7 | -0.0238673 | 0.0483398 | 0.0317323 | 0.857143 | 0.00180246 |
| U1_liquid_tradable | M6_xgboost_rank_ndcg | 20 | strong_up | 7 | 0.0803615 | -0.00100274 | -0.0119531 | 0.285714 | -0.00173376 |
| U1_liquid_tradable | M6_xgboost_rank_pairwise | 20 | neutral | 25 | 0.0146028 | 0.0126355 | 0.0151138 | 0.68 | 0.00309535 |
| U1_liquid_tradable | M6_xgboost_rank_pairwise | 20 | strong_down | 7 | -0.0606871 | 0.0115199 | 0.0169571 | 0.571429 | 0.00105129 |
| U1_liquid_tradable | M6_xgboost_rank_pairwise | 20 | strong_up | 7 | 0.0749927 | -0.00637151 | -0.00257178 | 0.285714 | 0.00326053 |
| U2_risk_sane | B0_market_ew | 20 | neutral | 25 | 0.00196727 | 0 | 0 | 0 |  |
| U2_risk_sane | B0_market_ew | 20 | strong_down | 7 | -0.0722071 | 0 | 0 | 0 |  |
| U2_risk_sane | B0_market_ew | 20 | strong_up | 7 | 0.0813642 | 0 | 0 | 0 |  |
| U2_risk_sane | M6_ranker_top20_ensemble | 20 | neutral | 25 | 0.00369744 | 0.00173017 | 0.00096357 | 0.52 | 0.00368937 |
| U2_risk_sane | M6_ranker_top20_ensemble | 20 | strong_down | 7 | -0.0870747 | -0.0148676 | -0.0412106 | 0.428571 | 0.00331882 |
| U2_risk_sane | M6_ranker_top20_ensemble | 20 | strong_up | 7 | 0.0928725 | 0.0115083 | 0.0216678 | 0.571429 | -0.0219144 |
| U2_risk_sane | M6_top20_calibrated | 20 | neutral | 25 | -0.00029538 | -0.00226265 | -0.00443264 | 0.36 | -0.00288665 |
| U2_risk_sane | M6_top20_calibrated | 20 | strong_down | 7 | -0.0961689 | -0.0239618 | -0.0159024 | 0.142857 | 0.000326835 |
| U2_risk_sane | M6_top20_calibrated | 20 | strong_up | 7 | 0.0923374 | 0.0109732 | 0.0308682 | 0.714286 | 0.0184674 |
| U2_risk_sane | M6_xgboost_rank_ndcg | 20 | neutral | 25 | 0.00570826 | 0.00374099 | -0.00285024 | 0.48 | -0.00229274 |
| U2_risk_sane | M6_xgboost_rank_ndcg | 20 | strong_down | 7 | -0.014455 | 0.057752 | 0.0447977 | 0.714286 | 0.01675 |
| U2_risk_sane | M6_xgboost_rank_ndcg | 20 | strong_up | 7 | 0.0706492 | -0.010715 | -0.0152324 | 0.285714 | 0.0218991 |
| U2_risk_sane | M6_xgboost_rank_pairwise | 20 | neutral | 25 | 0.0140259 | 0.0120586 | 0.00916589 | 0.76 | 0.00941073 |
| U2_risk_sane | M6_xgboost_rank_pairwise | 20 | strong_down | 7 | -0.0493015 | 0.0229056 | 0.00844243 | 0.714286 | 0.0136003 |
| U2_risk_sane | M6_xgboost_rank_pairwise | 20 | strong_up | 7 | 0.0665821 | -0.0147821 | -0.0360816 | 0.285714 | -0.0058934 |
| U1_liquid_tradable | B0_market_ew | 30 | neutral | 25 | 0.00196727 | 0 | 0 | 0 |  |
| U1_liquid_tradable | B0_market_ew | 30 | strong_down | 7 | -0.0722071 | 0 | 0 | 0 |  |
| U1_liquid_tradable | B0_market_ew | 30 | strong_up | 7 | 0.0813642 | 0 | 0 | 0 |  |
| U1_liquid_tradable | M6_ranker_top20_ensemble | 30 | neutral | 25 | -0.00703881 | -0.00900609 | -0.00703319 | 0.44 | -0.0113593 |
| U1_liquid_tradable | M6_ranker_top20_ensemble | 30 | strong_down | 7 | -0.0680638 | 0.00414334 | 0.0089799 | 0.571429 | -0.00296017 |
| U1_liquid_tradable | M6_ranker_top20_ensemble | 30 | strong_up | 7 | 0.0801571 | -0.00120714 | 0.00664454 | 0.571429 | -0.00599256 |
| U1_liquid_tradable | M6_top20_calibrated | 30 | neutral | 25 | -0.00581753 | -0.00778481 | -0.0141724 | 0.36 | 0.000254147 |
| U1_liquid_tradable | M6_top20_calibrated | 30 | strong_down | 7 | -0.0827182 | -0.0105111 | -0.0229718 | 0.428571 | 0.0111869 |
| U1_liquid_tradable | M6_top20_calibrated | 30 | strong_up | 7 | 0.0772449 | -0.00411931 | -0.00450357 | 0.285714 | 0.00165088 |
| U1_liquid_tradable | M6_xgboost_rank_ndcg | 30 | neutral | 25 | 0.00156827 | -0.000399 | -0.000514132 | 0.48 | 0.00557621 |

## 口径

- M6 默认输入沿用 M5 收敛方向：`price_volume + industry_breadth + fund_flow + fundamental`，暂不把 shareholder 作为主输入。
- `M6_xgboost_rank_ndcg` 与 `M6_xgboost_rank_pairwise` 使用每个 signal_date 作为 query group，标签为同月未来 market-relative excess 的分级 relevance。
- `M6_top20_calibrated` 使用 future top20 bucket 分类概率，并在每个测试月内转换为截面分位 score。
- `M6_ranker_top20_ensemble` 固定使用 `0.60 * rank_ndcg_percentile + 0.40 * top20_percentile`，不使用未来月调权。
- `topk_excess_after_cost` 使用半 L1 月度换手乘以 `cost_bps` 的简化成本敏感性。
- 本脚本只生成研究候选与诊断产物，不写入 promoted registry，不生成交易指令。

## 本轮结论

- 本轮新增：M6 learning-to-rank runner，覆盖 XGBoost Lambda/NDCG 排序、pairwise 排序、top-bucket rank calibration 与固定 ensemble。
- Gate 仍看 Rank IC、Top-K after-cost 超额、Top-K vs next-K、分桶 spread、年度/状态稳定性和行业暴露；oracle overlap 不作为主评价。
- 若 strong-up 或关键年份切片仍不稳，下一步优先做 regime-aware calibration，而不是把模型直接提升为推荐候选。

## 本轮产物

- `data/results/monthly_selection_m6_ltr_pitfix_2026_05_03_2026_05_03_summary.json`
- `data/results/monthly_selection_m6_ltr_pitfix_2026_05_03_2026_05_03_leaderboard.csv`
- `data/results/monthly_selection_m6_ltr_pitfix_2026_05_03_2026_05_03_monthly_long.csv`
- `data/results/monthly_selection_m6_ltr_pitfix_2026_05_03_2026_05_03_rank_ic.csv`
- `data/results/monthly_selection_m6_ltr_pitfix_2026_05_03_2026_05_03_quantile_spread.csv`
- `data/results/monthly_selection_m6_ltr_pitfix_2026_05_03_2026_05_03_feature_coverage.csv`
- `data/results/monthly_selection_m6_ltr_pitfix_2026_05_03_2026_05_03_feature_importance.csv`
- `data/results/monthly_selection_m6_ltr_pitfix_2026_05_03_2026_05_03_topk_holdings.csv`
- `data/results/monthly_selection_m6_ltr_pitfix_2026_05_03_2026_05_03_industry_exposure.csv`
- `data/results/monthly_selection_m6_ltr_pitfix_2026_05_03_2026_05_03_candidate_pool_width.csv`
- `data/results/monthly_selection_m6_ltr_pitfix_2026_05_03_2026_05_03_candidate_pool_reject_reason.csv`
- `data/results/monthly_selection_m6_ltr_pitfix_2026_05_03_2026_05_03_year_slice.csv`
- `data/results/monthly_selection_m6_ltr_pitfix_2026_05_03_2026_05_03_regime_slice.csv`
- `data/results/monthly_selection_m6_ltr_pitfix_2026_05_03_2026_05_03_market_states.csv`
- `data/results/monthly_selection_m6_ltr_pitfix_2026_05_03_2026_05_03_manifest.json`
