# Monthly Selection M6 Learning-to-Rank

- 生成时间：`2026-04-29T12:31:53.538137+00:00`
- 结果类型：`monthly_selection_m6_ltr`
- 研究主题：`monthly_selection_m6_ltr`
- 研究配置：`dataset_monthly_selection_features_families_price_volume_only-industry_breadth-fund_flow-fundamental_pools_u1_liquid_tradable-u2_risk_sane_topk_20-30-50_models_xgboost_rank_ndcg-xgboost_rank_pairwise-top20_calibrated-ranker_top20_ensemble_grades_5_maxfit_0_wf_24m_costbps_10_0`
- 输出 stem：`monthly_selection_m6_ltr_2026-04-29`
- 数据集：`data/cache/monthly_selection_features.parquet`
- 数据库：`data/market.duckdb`
- 训练/评估：按 signal_date walk-forward；每个测试月只用历史月份训练。
- 有效标签月份：`62`
- 单窗训练行上限：`0`（`0` 表示不抽样）

## Leaderboard

| candidate_pool_version | model | model_type | top_k | months | mean_topk_return | median_topk_return | topk_excess_mean | topk_excess_median | topk_hit_rate | topk_minus_nextk_mean | industry_neutral_topk_excess_mean | turnover_mean | cost_drag_mean | topk_excess_after_cost_mean | topk_excess_annualized | rank_ic_mean | rank_ic_std | rank_ic_months | rank_ic_ir | quantile_top_minus_bottom_mean | neutral_median_excess | strong_down_median_excess | strong_up_median_excess |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| U1_liquid_tradable | M6_xgboost_rank_ndcg | xgboost_ranker | 20 | 38 | 0.0124968 | 0.00660345 | 0.00034796 | -0.00257952 | 0.421053 | 0.00523571 | 0.00116761 | 0.905405 | 0.000905405 | 0.000214234 | 0.00418352 | 0.00929723 | 0.078261 | 38 | 0.118798 | -0.0011085 | 0.00382698 | -0.0062846 | -0.0294861 |
| U1_liquid_tradable | B0_market_ew | benchmark | 20 | 38 | 0.0121489 | 0.0033285 | 0 | 0 | 0 |  |  |  | 0 | 0 | 0 |  |  |  |  |  | 0 | 0 | 0 |
| U1_liquid_tradable | M6_xgboost_rank_pairwise | xgboost_ranker | 20 | 38 | 0.0115777 | 0.00849837 | -0.000571186 | -0.00448697 | 0.421053 | -0.000418645 | 0.000650027 | 0.931081 | 0.000931081 | -0.00233952 | -0.00683274 | 0.0591345 | 0.118499 | 38 | 0.499028 | 0.00448232 | -0.00264133 | -0.0055457 | -0.0390196 |
| U1_liquid_tradable | M6_top20_calibrated | top_bucket_classifier_rank_calibrated | 20 | 38 | 0.00976192 | -0.00286389 | -0.00238694 | -0.00635963 | 0.473684 | 0.00243701 | 0.00276822 | 0.917568 | 0.000917568 | -0.00258294 | -0.0282703 | -0.0491818 | 0.125767 | 38 | -0.391055 | -0.00173037 | -0.0236359 | -0.0255 | 0.0493761 |
| U1_liquid_tradable | M6_ranker_top20_ensemble | ranker_classifier_ensemble | 20 | 38 | 0.00769451 | 0.0023152 | -0.00445435 | -0.010244 | 0.368421 | -0.00866226 | -0.00268808 | 0.932432 | 0.000932432 | -0.00510215 | -0.0521619 | -0.0164518 | 0.0699931 | 38 | -0.235048 | -0.00156817 | -0.0131879 | -0.0161142 | 0.031722 |
| U2_risk_sane | M6_xgboost_rank_ndcg | xgboost_ranker | 20 | 38 | 0.0247633 | 0.028524 | 0.0126144 | 0.0147394 | 0.657895 | 0.00762405 | 0.00384424 | 0.940541 | 0.000940541 | 0.0121064 | 0.16233 | 0.00666237 | 0.0777473 | 38 | 0.0856927 | 0.000673815 | 0.0131781 | -0.00184697 | 0.0181031 |
| U2_risk_sane | M6_ranker_top20_ensemble | ranker_classifier_ensemble | 20 | 38 | 0.0133615 | -0.000620145 | 0.0012126 | -0.00130647 | 0.5 | 0.00406081 | 0.0018428 | 0.97027 | 0.00097027 | 8.27375e-05 | 0.0146486 | -0.0140576 | 0.0833367 | 38 | -0.168685 | 0.000307451 | -0.00917621 | -0.0124117 | 0.0516311 |
| U2_risk_sane | B0_market_ew | benchmark | 20 | 38 | 0.0121489 | 0.0033285 | 0 | 0 | 0 |  |  |  | 0 | 0 | 0 |  |  |  |  |  | 0 | 0 | 0 |
| U2_risk_sane | M6_top20_calibrated | top_bucket_classifier_rank_calibrated | 20 | 38 | 0.00496333 | -0.0171211 | -0.00718553 | -0.0138333 | 0.394737 | -0.00760311 | -0.00522525 | 0.947297 | 0.000947297 | -0.00733761 | -0.0828989 | -0.0386747 | 0.125795 | 38 | -0.307442 | -6.08227e-05 | -0.01395 | -0.0341666 | 0.024924 |
| U2_risk_sane | M6_xgboost_rank_pairwise | xgboost_ranker | 20 | 38 | 0.00424489 | 0.00401736 | -0.00790397 | -0.0035956 | 0.368421 | -0.00413925 | -0.00231134 | 0.945946 | 0.000945946 | -0.00941381 | -0.0908312 | 0.0344716 | 0.13056 | 38 | 0.264028 | 0.000428869 | 0.000575254 | -0.00864988 | -0.0586849 |
| U1_liquid_tradable | B0_market_ew | benchmark | 30 | 38 | 0.0121489 | 0.0033285 | 0 | 0 | 0 |  |  |  | 0 | 0 | 0 |  |  |  |  |  | 0 | 0 | 0 |
| U1_liquid_tradable | M6_xgboost_rank_pairwise | xgboost_ranker | 30 | 38 | 0.0125554 | 0.0104375 | 0.000406495 | -0.0038539 | 0.447368 | 0.000781482 | 0.000732323 | 0.923423 | 0.000923423 | -0.00111934 | 0.00488886 | 0.0591345 | 0.118499 | 38 | 0.499028 | 0.00448232 | -0.00368579 | -0.00384821 | -0.0446435 |
| U1_liquid_tradable | M6_ranker_top20_ensemble | ranker_classifier_ensemble | 30 | 38 | 0.0110219 | 0.00270525 | -0.00112696 | -0.00723213 | 0.368421 | -0.00235236 | 0.000363952 | 0.916216 | 0.000916216 | -0.00168916 | -0.01344 | -0.0164518 | 0.0699931 | 38 | -0.235048 | -0.00156817 | -0.00372046 | -0.0184939 | 0.0281731 |
| U1_liquid_tradable | M6_xgboost_rank_ndcg | xgboost_ranker | 30 | 38 | 0.0103469 | -0.00255751 | -0.00180193 | -0.000850108 | 0.5 | 0.00462507 | -0.000737155 | 0.891892 | 0.000891892 | -0.00189086 | -0.0214102 | 0.00929723 | 0.078261 | 38 | 0.118798 | -0.0011085 | 0.00627043 | -0.00861102 | -0.0401017 |
| U1_liquid_tradable | M6_top20_calibrated | top_bucket_classifier_rank_calibrated | 30 | 38 | 0.00846805 | -0.00923586 | -0.00368081 | -0.00743368 | 0.447368 | -0.000257682 | 0.00104072 | 0.906306 | 0.000906306 | -0.00397151 | -0.0432864 | -0.0491818 | 0.125767 | 38 | -0.391055 | -0.00173037 | -0.0128166 | -0.0139406 | 0.0370881 |
| U2_risk_sane | M6_xgboost_rank_ndcg | xgboost_ranker | 30 | 38 | 0.0214441 | 0.0241337 | 0.0092952 | 0.00160447 | 0.552632 | 0.010155 | 0.00158023 | 0.931532 | 0.000931532 | 0.00862985 | 0.117425 | 0.00666237 | 0.0777473 | 38 | 0.0856927 | 0.000673815 | 0.00361303 | -0.00236666 | 0.000249559 |
| U2_risk_sane | B0_market_ew | benchmark | 30 | 38 | 0.0121489 | 0.0033285 | 0 | 0 | 0 |  |  |  | 0 | 0 | 0 |  |  |  |  |  | 0 | 0 | 0 |
| U2_risk_sane | M6_top20_calibrated | top_bucket_classifier_rank_calibrated | 30 | 38 | 0.00935468 | -0.0126151 | -0.00279418 | -0.0055373 | 0.394737 | -0.000413656 | -0.00157701 | 0.935135 | 0.000935135 | -0.00253128 | -0.0330196 | -0.0386747 | 0.125795 | 38 | -0.307442 | -6.08227e-05 | -0.0119795 | -0.00974848 | 0.0336553 |
| U2_risk_sane | M6_ranker_top20_ensemble | ranker_classifier_ensemble | 30 | 38 | 0.0100743 | -0.00651472 | -0.00207458 | -0.00335077 | 0.394737 | 0.000312345 | -0.00131097 | 0.958559 | 0.000958559 | -0.00316937 | -0.0246129 | -0.0140576 | 0.0833367 | 38 | -0.168685 | 0.000307451 | -0.00834417 | -0.0109986 | 0.0443096 |
| U2_risk_sane | M6_xgboost_rank_pairwise | xgboost_ranker | 30 | 38 | 0.00466236 | 0.00519354 | -0.0074865 | -0.00768208 | 0.421053 | -0.00184106 | -0.00238828 | 0.928829 | 0.000928829 | -0.00875827 | -0.0862296 | 0.0344716 | 0.13056 | 38 | 0.264028 | 0.000428869 | 0.00373672 | -0.0197748 | -0.0548007 |
| U1_liquid_tradable | B0_market_ew | benchmark | 50 | 38 | 0.0121489 | 0.0033285 | 0 | 0 | 0 |  |  |  | 0 | 0 | 0 |  |  |  |  |  | 0 | 0 | 0 |
| U1_liquid_tradable | M6_xgboost_rank_pairwise | xgboost_ranker | 50 | 38 | 0.0134786 | 0.0122028 | 0.00132975 | -0.00245577 | 0.473684 | 0.00246077 | 0.00190206 | 0.884865 | 0.000884865 | -0.00035946 | 0.0160742 | 0.0591345 | 0.118499 | 38 | 0.499028 | 0.00448232 | 0.00504266 | 0.00118342 | -0.021821 |
| U1_liquid_tradable | M6_ranker_top20_ensemble | ranker_classifier_ensemble | 50 | 38 | 0.0114877 | 0.0071758 | -0.000661136 | -0.000106914 | 0.5 | 0.00285401 | 0.000293514 | 0.907027 | 0.000907027 | -0.00117567 | -0.00790485 | -0.0164518 | 0.0699931 | 38 | -0.235048 | -0.00156817 | -0.0014471 | -0.0156412 | 0.0288807 |
| U1_liquid_tradable | M6_top20_calibrated | top_bucket_classifier_rank_calibrated | 50 | 38 | 0.00685798 | -0.015 | -0.00529088 | -0.0165066 | 0.368421 | -0.00504411 | -0.000639056 | 0.890811 | 0.000890811 | -0.00528581 | -0.0616752 | -0.0491818 | 0.125767 | 38 | -0.391055 | -0.00173037 | -0.0255368 | -0.0192877 | 0.0364656 |
| U1_liquid_tradable | M6_xgboost_rank_ndcg | xgboost_ranker | 50 | 38 | 0.00728231 | -0.00177501 | -0.00486655 | -0.00717756 | 0.394737 | 0.00160334 | -0.00335716 | 0.867027 | 0.000867027 | -0.00539744 | -0.0568606 | 0.00929723 | 0.078261 | 38 | 0.118798 | -0.0011085 | -0.00546122 | -0.00450495 | -0.0380812 |
| U2_risk_sane | M6_xgboost_rank_ndcg | xgboost_ranker | 50 | 38 | 0.0183055 | 0.0134829 | 0.00615666 | -0.000482766 | 0.5 | 0.00987503 | -0.000140534 | 0.905405 | 0.000905405 | 0.00568267 | 0.0764337 | 0.00666237 | 0.0777473 | 38 | 0.0856927 | 0.000673815 | 8.38721e-05 | -0.00959378 | 0.00797494 |
| U2_risk_sane | B0_market_ew | benchmark | 50 | 38 | 0.0121489 | 0.0033285 | 0 | 0 | 0 |  |  |  | 0 | 0 | 0 |  |  |  |  |  | 0 | 0 | 0 |
| U2_risk_sane | M6_ranker_top20_ensemble | ranker_classifier_ensemble | 50 | 38 | 0.0109799 | -0.00416091 | -0.00116894 | -0.0135941 | 0.394737 | 0.000467314 | -0.00109183 | 0.943784 | 0.000943784 | -0.00166446 | -0.0139375 | -0.0140576 | 0.0833367 | 38 | -0.168685 | 0.000307451 | -0.0143367 | -0.0172155 | 0.0350528 |
| U2_risk_sane | M6_top20_calibrated | top_bucket_classifier_rank_calibrated | 50 | 38 | 0.00967909 | -0.00699228 | -0.00246977 | -0.0032579 | 0.421053 | -0.000737575 | -0.000872991 | 0.923243 | 0.000923243 | -0.00249819 | -0.0292379 | -0.0386747 | 0.125795 | 38 | -0.307442 | -6.08227e-05 | -0.00564794 | -0.0147343 | 0.0341571 |
| U2_risk_sane | M6_xgboost_rank_pairwise | xgboost_ranker | 50 | 38 | 0.005064 | 0.00791871 | -0.00708486 | -0.00705833 | 0.394737 | -0.00888217 | -0.00302366 | 0.898378 | 0.000898378 | -0.00846579 | -0.0817825 | 0.0344716 | 0.13056 | 38 | 0.264028 | 0.000428869 | 0.00208019 | -0.0210516 | -0.0405878 |

## U1 Top20 Leading Models

| candidate_pool_version | model | model_type | top_k | months | mean_topk_return | median_topk_return | topk_excess_mean | topk_excess_median | topk_hit_rate | topk_minus_nextk_mean | industry_neutral_topk_excess_mean | turnover_mean | cost_drag_mean | topk_excess_after_cost_mean | topk_excess_annualized | rank_ic_mean | rank_ic_std | rank_ic_months | rank_ic_ir | quantile_top_minus_bottom_mean | neutral_median_excess | strong_down_median_excess | strong_up_median_excess |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| U1_liquid_tradable | M6_xgboost_rank_ndcg | xgboost_ranker | 20 | 38 | 0.0124968 | 0.00660345 | 0.00034796 | -0.00257952 | 0.421053 | 0.00523571 | 0.00116761 | 0.905405 | 0.000905405 | 0.000214234 | 0.00418352 | 0.00929723 | 0.078261 | 38 | 0.118798 | -0.0011085 | 0.00382698 | -0.0062846 | -0.0294861 |
| U1_liquid_tradable | B0_market_ew | benchmark | 20 | 38 | 0.0121489 | 0.0033285 | 0 | 0 | 0 |  |  |  | 0 | 0 | 0 |  |  |  |  |  | 0 | 0 | 0 |
| U1_liquid_tradable | M6_xgboost_rank_pairwise | xgboost_ranker | 20 | 38 | 0.0115777 | 0.00849837 | -0.000571186 | -0.00448697 | 0.421053 | -0.000418645 | 0.000650027 | 0.931081 | 0.000931081 | -0.00233952 | -0.00683274 | 0.0591345 | 0.118499 | 38 | 0.499028 | 0.00448232 | -0.00264133 | -0.0055457 | -0.0390196 |
| U1_liquid_tradable | M6_top20_calibrated | top_bucket_classifier_rank_calibrated | 20 | 38 | 0.00976192 | -0.00286389 | -0.00238694 | -0.00635963 | 0.473684 | 0.00243701 | 0.00276822 | 0.917568 | 0.000917568 | -0.00258294 | -0.0282703 | -0.0491818 | 0.125767 | 38 | -0.391055 | -0.00173037 | -0.0236359 | -0.0255 | 0.0493761 |
| U1_liquid_tradable | M6_ranker_top20_ensemble | ranker_classifier_ensemble | 20 | 38 | 0.00769451 | 0.0023152 | -0.00445435 | -0.010244 | 0.368421 | -0.00866226 | -0.00268808 | 0.932432 | 0.000932432 | -0.00510215 | -0.0521619 | -0.0164518 | 0.0699931 | 38 | -0.235048 | -0.00156817 | -0.0131879 | -0.0161142 | 0.031722 |

## Feature Coverage

| feature_spec | families | feature | raw_feature | rows | non_null | coverage_ratio | candidate_pool_pass_coverage_ratio | first_signal_date | last_signal_date |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_ret_20d_z | feature_ret_20d | 307391 | 306119 | 0.995862 | 1 | 2021-01-29 | 2026-04-13 |
| m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_ret_60d_z | feature_ret_60d | 307391 | 303546 | 0.987492 | 1 | 2021-01-29 | 2026-04-13 |
| m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_ret_5d_z | feature_ret_5d | 307391 | 307058 | 0.998917 | 1 | 2021-01-29 | 2026-04-13 |
| m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_realized_vol_20d_z | feature_realized_vol_20d | 307391 | 306711 | 0.997788 | 1 | 2021-01-29 | 2026-04-13 |
| m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_amount_20d_log_z | feature_amount_20d_log | 307391 | 306783 | 0.998022 | 1 | 2021-01-29 | 2026-04-13 |
| m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_turnover_20d_z | feature_turnover_20d | 307391 | 306783 | 0.998022 | 1 | 2021-01-29 | 2026-04-13 |
| m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_price_position_250d_z | feature_price_position_250d | 307391 | 303612 | 0.987706 | 1 | 2021-01-29 | 2026-04-13 |
| m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_limit_move_hits_20d_z | feature_limit_move_hits_20d | 307391 | 307391 | 1 | 1 | 2021-01-29 | 2026-04-13 |
| m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_industry_ret20_mean_z | feature_industry_ret20_mean | 307391 | 307391 | 1 | 1 | 2021-01-29 | 2026-04-13 |
| m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_industry_ret60_mean_z | feature_industry_ret60_mean | 307391 | 307391 | 1 | 1 | 2021-01-29 | 2026-04-13 |
| m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_industry_positive_ret20_ratio_z | feature_industry_positive_ret20_ratio | 307391 | 307391 | 1 | 1 | 2021-01-29 | 2026-04-13 |
| m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_industry_amount20_mean_z | feature_industry_amount20_mean | 307391 | 307391 | 1 | 1 | 2021-01-29 | 2026-04-13 |
| m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_industry_low_vol20_mean_z | feature_industry_low_vol20_mean | 307391 | 307391 | 1 | 1 | 2021-01-29 | 2026-04-13 |
| m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_fund_flow_main_inflow_5d_z | feature_fund_flow_main_inflow_5d | 307391 | 31959 | 0.103969 | 0.115043 | 2025-09-30 | 2026-04-13 |
| m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_fund_flow_main_inflow_10d_z | feature_fund_flow_main_inflow_10d | 307391 | 27575 | 0.0897066 | 0.0967688 | 2025-09-30 | 2026-04-13 |
| m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_fund_flow_main_inflow_20d_z | feature_fund_flow_main_inflow_20d | 307391 | 27445 | 0.0892837 | 0.096305 | 2025-10-31 | 2026-04-13 |
| m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_fund_flow_super_inflow_10d_z | feature_fund_flow_super_inflow_10d | 307391 | 27575 | 0.0897066 | 0.0967688 | 2025-09-30 | 2026-04-13 |
| m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_fund_flow_divergence_20d_z | feature_fund_flow_divergence_20d | 307391 | 27445 | 0.0892837 | 0.096305 | 2025-10-31 | 2026-04-13 |
| m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_fund_flow_main_inflow_streak_z | feature_fund_flow_main_inflow_streak | 307391 | 31964 | 0.103985 | 0.115043 | 2025-09-30 | 2026-04-13 |
| m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_fundamental_pe_ttm_z | feature_fundamental_pe_ttm | 307391 | 304163 | 0.989499 | 0.998475 | 2021-01-29 | 2026-04-13 |
| m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_fundamental_pb_z | feature_fundamental_pb | 307391 | 304163 | 0.989499 | 0.998475 | 2021-01-29 | 2026-04-13 |
| m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_fundamental_ev_ebitda_z | feature_fundamental_ev_ebitda | 307391 | 0 | 0 | 0 |  |  |
| m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_fundamental_roe_ttm_z | feature_fundamental_roe_ttm | 307391 | 304567 | 0.990813 | 0.992536 | 2021-01-29 | 2026-04-13 |
| m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_fundamental_net_profit_yoy_z | feature_fundamental_net_profit_yoy | 307391 | 306079 | 0.995732 | 0.995764 | 2021-01-29 | 2026-04-13 |
| m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_fundamental_gross_margin_change_z | feature_fundamental_gross_margin_change | 307391 | 299622 | 0.974726 | 0.967497 | 2021-01-29 | 2026-04-13 |
| m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_fundamental_gross_margin_delta_z | feature_fundamental_gross_margin_delta | 307391 | 115226 | 0.374852 | 0.371201 | 2021-01-29 | 2026-04-13 |
| m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_fundamental_debt_to_assets_change_z | feature_fundamental_debt_to_assets_change | 307391 | 306168 | 0.996021 | 0.996753 | 2021-01-29 | 2026-04-13 |
| m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_fundamental_ocf_to_net_profit_z | feature_fundamental_ocf_to_net_profit | 307391 | 245793 | 0.79961 | 0.810469 | 2021-01-29 | 2026-04-13 |
| m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_fundamental_ocf_to_asset_z | feature_fundamental_ocf_to_asset | 307391 | 117855 | 0.383404 | 0.382509 | 2021-01-29 | 2026-04-13 |
| m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_fundamental_asset_turnover_z | feature_fundamental_asset_turnover | 307391 | 117936 | 0.383668 | 0.382825 | 2021-01-29 | 2026-04-13 |
| m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_fundamental_net_margin_stability_z | feature_fundamental_net_margin_stability | 307391 | 117848 | 0.383381 | 0.382772 | 2021-01-29 | 2026-04-13 |

## Year Slice

| candidate_pool_version | model | top_k | year | months | mean_topk_return | mean_topk_excess | median_topk_excess | hit_rate | mean_topk_minus_nextk |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| U1_liquid_tradable | B0_market_ew | 20 | 2023 | 12 | -0.0165124 | 0 | 0 | 0 |  |
| U1_liquid_tradable | B0_market_ew | 20 | 2024 | 12 | 0.0243624 | 0 | 0 | 0 |  |
| U1_liquid_tradable | B0_market_ew | 20 | 2025 | 12 | 0.0332516 | 0 | 0 | 0 |  |
| U1_liquid_tradable | B0_market_ew | 20 | 2026 | 2 | -0.0157816 | 0 | 0 | 0 |  |
| U1_liquid_tradable | B0_market_ew | 30 | 2023 | 12 | -0.0165124 | 0 | 0 | 0 |  |
| U1_liquid_tradable | B0_market_ew | 30 | 2024 | 12 | 0.0243624 | 0 | 0 | 0 |  |
| U1_liquid_tradable | B0_market_ew | 30 | 2025 | 12 | 0.0332516 | 0 | 0 | 0 |  |
| U1_liquid_tradable | B0_market_ew | 30 | 2026 | 2 | -0.0157816 | 0 | 0 | 0 |  |
| U1_liquid_tradable | B0_market_ew | 50 | 2023 | 12 | -0.0165124 | 0 | 0 | 0 |  |
| U1_liquid_tradable | B0_market_ew | 50 | 2024 | 12 | 0.0243624 | 0 | 0 | 0 |  |
| U1_liquid_tradable | B0_market_ew | 50 | 2025 | 12 | 0.0332516 | 0 | 0 | 0 |  |
| U1_liquid_tradable | B0_market_ew | 50 | 2026 | 2 | -0.0157816 | 0 | 0 | 0 |  |
| U1_liquid_tradable | M6_ranker_top20_ensemble | 20 | 2023 | 12 | -0.0204783 | -0.00396584 | -0.0112738 | 0.416667 | -0.0203621 |
| U1_liquid_tradable | M6_ranker_top20_ensemble | 20 | 2024 | 12 | 0.0220523 | -0.00231012 | -0.0124794 | 0.333333 | 0.00727276 |
| U1_liquid_tradable | M6_ranker_top20_ensemble | 20 | 2025 | 12 | 0.0327526 | -0.000499024 | -0.0027497 | 0.416667 | -0.0103134 |
| U1_liquid_tradable | M6_ranker_top20_ensemble | 20 | 2026 | 2 | -0.0597644 | -0.0439827 | -0.0439827 | 0 | -0.0241669 |
| U1_liquid_tradable | M6_ranker_top20_ensemble | 30 | 2023 | 12 | -0.00992779 | 0.00658463 | -0.0030706 | 0.416667 | 0.0147487 |
| U1_liquid_tradable | M6_ranker_top20_ensemble | 30 | 2024 | 12 | 0.0175784 | -0.00678405 | -0.0161028 | 0.25 | -0.00586783 |
| U1_liquid_tradable | M6_ranker_top20_ensemble | 30 | 2025 | 12 | 0.0378834 | 0.00463175 | -0.00394164 | 0.5 | -0.00654613 |
| U1_liquid_tradable | M6_ranker_top20_ensemble | 30 | 2026 | 2 | -0.0637879 | -0.0480063 | -0.0480063 | 0 | -0.0587034 |
| U1_liquid_tradable | M6_ranker_top20_ensemble | 50 | 2023 | 12 | -0.0140969 | 0.0024155 | -0.00273275 | 0.5 | 0.016714 |
| U1_liquid_tradable | M6_ranker_top20_ensemble | 50 | 2024 | 12 | 0.0196033 | -0.00475917 | -0.0120501 | 0.333333 | -0.0035013 |
| U1_liquid_tradable | M6_ranker_top20_ensemble | 50 | 2025 | 12 | 0.0389981 | 0.00574649 | 0.00770525 | 0.666667 | 0.00571444 |
| U1_liquid_tradable | M6_ranker_top20_ensemble | 50 | 2026 | 2 | -0.0487601 | -0.0329785 | -0.0329785 | 0.5 | -0.0593369 |
| U1_liquid_tradable | M6_top20_calibrated | 20 | 2023 | 12 | -0.0318585 | -0.015346 | -0.0178354 | 0.333333 | -0.00934263 |
| U1_liquid_tradable | M6_top20_calibrated | 20 | 2024 | 12 | 0.0261231 | 0.00176065 | 0.00946395 | 0.583333 | 0.00523934 |
| U1_liquid_tradable | M6_top20_calibrated | 20 | 2025 | 12 | 0.04258 | 0.00932838 | 0.0253367 | 0.583333 | 0.00595129 |
| U1_liquid_tradable | M6_top20_calibrated | 20 | 2026 | 2 | -0.0355915 | -0.0198099 | -0.0198099 | 0 | 0.0352151 |
| U1_liquid_tradable | M6_top20_calibrated | 30 | 2023 | 12 | -0.0260382 | -0.00952579 | -0.00492686 | 0.416667 | 0.00556533 |
| U1_liquid_tradable | M6_top20_calibrated | 30 | 2024 | 12 | 0.0192638 | -0.00509861 | -0.0135773 | 0.416667 | -0.00810326 |
| U1_liquid_tradable | M6_top20_calibrated | 30 | 2025 | 12 | 0.0386386 | 0.00538699 | 0.00309211 | 0.5 | -0.0011283 |
| U1_liquid_tradable | M6_top20_calibrated | 30 | 2026 | 2 | -0.0302926 | -0.014511 | -0.014511 | 0.5 | 0.0171014 |
| U1_liquid_tradable | M6_top20_calibrated | 50 | 2023 | 12 | -0.0280402 | -0.0115278 | -0.00838899 | 0.333333 | 0.0049241 |
| U1_liquid_tradable | M6_top20_calibrated | 50 | 2024 | 12 | 0.019695 | -0.00466744 | -0.0165066 | 0.333333 | -0.00383796 |
| U1_liquid_tradable | M6_top20_calibrated | 50 | 2025 | 12 | 0.0369685 | 0.00371684 | -0.000134765 | 0.5 | -0.0140977 |
| U1_liquid_tradable | M6_top20_calibrated | 50 | 2026 | 2 | -0.0414381 | -0.0256564 | -0.0256564 | 0 | -0.0177687 |
| U1_liquid_tradable | M6_xgboost_rank_ndcg | 20 | 2023 | 12 | -0.0132956 | 0.00321683 | -0.00234923 | 0.5 | -0.0022628 |
| U1_liquid_tradable | M6_xgboost_rank_ndcg | 20 | 2024 | 12 | 0.0159405 | -0.00842191 | -0.0062846 | 0.333333 | 0.00669603 |
| U1_liquid_tradable | M6_xgboost_rank_ndcg | 20 | 2025 | 12 | 0.0414915 | 0.00823985 | -0.000478389 | 0.416667 | 0.00914186 |
| U1_liquid_tradable | M6_xgboost_rank_ndcg | 20 | 2026 | 2 | -0.027379 | -0.0115974 | -0.0115974 | 0.5 | 0.0180279 |

## Realized Market State Slice

| candidate_pool_version | model | top_k | realized_market_state | months | mean_topk_return | mean_topk_excess | median_topk_excess | hit_rate | mean_topk_minus_nextk |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| U1_liquid_tradable | B0_market_ew | 20 | neutral | 23 | 0.00922614 | 0 | 0 | 0 |  |
| U1_liquid_tradable | B0_market_ew | 20 | strong_down | 8 | -0.0671689 | 0 | 0 | 0 |  |
| U1_liquid_tradable | B0_market_ew | 20 | strong_up | 7 | 0.112401 | 0 | 0 | 0 |  |
| U1_liquid_tradable | M6_ranker_top20_ensemble | 20 | neutral | 23 | -0.00164329 | -0.0108694 | -0.0131879 | 0.304348 | -0.0184828 |
| U1_liquid_tradable | M6_ranker_top20_ensemble | 20 | strong_down | 8 | -0.0877366 | -0.0205677 | -0.0161142 | 0.25 | -0.00253849 |
| U1_liquid_tradable | M6_ranker_top20_ensemble | 20 | strong_up | 7 | 0.14744 | 0.0350391 | 0.031722 | 0.714286 | 0.0166067 |
| U1_liquid_tradable | M6_top20_calibrated | 20 | neutral | 23 | -0.00177109 | -0.0109972 | -0.0236359 | 0.434783 | 0.00206554 |
| U1_liquid_tradable | M6_top20_calibrated | 20 | strong_down | 8 | -0.0906464 | -0.0234775 | -0.0255 | 0.25 | -0.00315558 |
| U1_liquid_tradable | M6_top20_calibrated | 20 | strong_up | 7 | 0.162408 | 0.0500075 | 0.0493761 | 0.857143 | 0.0100491 |
| U1_liquid_tradable | M6_xgboost_rank_ndcg | 20 | neutral | 23 | 0.0123142 | 0.00308803 | 0.00382698 | 0.521739 | 0.00726123 |
| U1_liquid_tradable | M6_xgboost_rank_ndcg | 20 | strong_down | 8 | -0.0533861 | 0.0137828 | -0.0062846 | 0.375 | 0.0116161 |
| U1_liquid_tradable | M6_xgboost_rank_ndcg | 20 | strong_up | 7 | 0.0883917 | -0.0240093 | -0.0294861 | 0.142857 | -0.00871143 |
| U1_liquid_tradable | M6_xgboost_rank_pairwise | 20 | neutral | 23 | 0.0111994 | 0.00197325 | -0.00264133 | 0.434783 | -0.00196468 |
| U1_liquid_tradable | M6_xgboost_rank_pairwise | 20 | strong_down | 8 | -0.0716386 | -0.00446964 | -0.0055457 | 0.375 | -0.00457393 |
| U1_liquid_tradable | M6_xgboost_rank_pairwise | 20 | strong_up | 7 | 0.107925 | -0.00447611 | -0.0390196 | 0.428571 | 0.00941008 |
| U2_risk_sane | B0_market_ew | 20 | neutral | 23 | 0.00922614 | 0 | 0 | 0 |  |
| U2_risk_sane | B0_market_ew | 20 | strong_down | 8 | -0.0671689 | 0 | 0 | 0 |  |
| U2_risk_sane | B0_market_ew | 20 | strong_up | 7 | 0.112401 | 0 | 0 | 0 |  |
| U2_risk_sane | M6_ranker_top20_ensemble | 20 | neutral | 23 | 0.000426803 | -0.00879934 | -0.00917621 | 0.391304 | -0.004872 |
| U2_risk_sane | M6_ranker_top20_ensemble | 20 | strong_down | 8 | -0.0809243 | -0.0137553 | -0.0124117 | 0.375 | 0.00694527 |
| U2_risk_sane | M6_ranker_top20_ensemble | 20 | strong_up | 7 | 0.163616 | 0.0512152 | 0.0516311 | 1 | 0.0301149 |
| U2_risk_sane | M6_top20_calibrated | 20 | neutral | 23 | -0.00453904 | -0.0137652 | -0.01395 | 0.304348 | -0.00338878 |
| U2_risk_sane | M6_top20_calibrated | 20 | strong_down | 8 | -0.092884 | -0.0257151 | -0.0341666 | 0.25 | -0.0228811 |
| U2_risk_sane | M6_top20_calibrated | 20 | strong_up | 7 | 0.148011 | 0.03561 | 0.024924 | 0.857143 | -0.00398962 |
| U2_risk_sane | M6_xgboost_rank_ndcg | 20 | neutral | 23 | 0.0140185 | 0.00479231 | 0.0131781 | 0.608696 | 0.0102422 |
| U2_risk_sane | M6_xgboost_rank_ndcg | 20 | strong_down | 8 | -0.0393504 | 0.0278186 | -0.00184697 | 0.5 | 0.00160023 |
| U2_risk_sane | M6_xgboost_rank_ndcg | 20 | strong_up | 7 | 0.13334 | 0.0209395 | 0.0181031 | 1 | 0.00590582 |
| U2_risk_sane | M6_xgboost_rank_pairwise | 20 | neutral | 23 | 0.0105231 | 0.00129695 | 0.000575254 | 0.521739 | 0.00137386 |
| U2_risk_sane | M6_xgboost_rank_pairwise | 20 | strong_down | 8 | -0.0660286 | 0.00114036 | -0.00864988 | 0.25 | -0.00330708 |
| U2_risk_sane | M6_xgboost_rank_pairwise | 20 | strong_up | 7 | 0.063929 | -0.048472 | -0.0586849 | 0 | -0.0232048 |
| U1_liquid_tradable | B0_market_ew | 30 | neutral | 23 | 0.00922614 | 0 | 0 | 0 |  |
| U1_liquid_tradable | B0_market_ew | 30 | strong_down | 8 | -0.0671689 | 0 | 0 | 0 |  |
| U1_liquid_tradable | B0_market_ew | 30 | strong_up | 7 | 0.112401 | 0 | 0 | 0 |  |
| U1_liquid_tradable | M6_ranker_top20_ensemble | 30 | neutral | 23 | 0.00679103 | -0.00243512 | -0.00372046 | 0.391304 | 0.0044226 |
| U1_liquid_tradable | M6_ranker_top20_ensemble | 30 | strong_down | 8 | -0.0898093 | -0.0226403 | -0.0184939 | 0.125 | -0.00981296 |
| U1_liquid_tradable | M6_ranker_top20_ensemble | 30 | strong_up | 7 | 0.140159 | 0.027758 | 0.0281731 | 0.571429 | -0.0160865 |
| U1_liquid_tradable | M6_top20_calibrated | 30 | neutral | 23 | -0.000811075 | -0.0100372 | -0.0128166 | 0.391304 | 0.00253091 |
| U1_liquid_tradable | M6_top20_calibrated | 30 | strong_down | 8 | -0.0910058 | -0.0238368 | -0.0139406 | 0.25 | -0.00874014 |
| U1_liquid_tradable | M6_top20_calibrated | 30 | strong_up | 7 | 0.152641 | 0.04024 | 0.0370881 | 0.857143 | 0.000274056 |
| U1_liquid_tradable | M6_xgboost_rank_ndcg | 30 | neutral | 23 | 0.0105203 | 0.00129419 | 0.00627043 | 0.608696 | 0.00819731 |

## 口径

- M6 默认输入沿用 M5 收敛方向：`price_volume + industry_breadth + fund_flow + fundamental`，暂不把 shareholder 作为主输入。
- `M6_xgboost_rank_ndcg` 与 `M6_xgboost_rank_pairwise` 使用每个 signal_date 作为 query group，标签为同月未来 market-relative excess 的分级 relevance。
- `M6_top20_calibrated` 使用 future top20 bucket 分类概率，并在每个测试月内转换为截面分位 score。
- `M6_ranker_top20_ensemble` 固定使用 `0.60 * rank_ndcg_percentile + 0.40 * top20_percentile`，不使用未来月调权。
- `topk_excess_after_cost` 使用半 L1 月度换手乘以 `cost_bps` 的简化成本敏感性。
- 本脚本只生成研究候选与诊断产物，不写入 promoted registry，不生成交易指令。

## 本轮结论

- 本轮完成：M6 learning-to-rank full-fit 主口径，覆盖 XGBoost Lambda/NDCG 排序、pairwise 排序、top-bucket rank calibration 与固定 ensemble。
- Full-fit 使用 `max_fit_rows=0`，每个 walk-forward 训练窗使用全部历史训练行，不再采用 5000 行受限抽样。
- `U1_liquid_tradable` 下 M6 没有通过 M5 对照 gate：Top20 最强 M6 after-cost 月均超额约 `0.000214`，显著低于 M5 最强约 `0.011436`。
- `U2_risk_sane` 下 `M6_xgboost_rank_ndcg` 通过 Top20 / Top30 watchlist gate：Top20 after-cost 月均超额约 `0.012106`，高于 M5 最强约 `0.009373`；Top30 after-cost 月均超额约 `0.008630`，略高于 M5 最强约 `0.008317`。
- `U2` Top50 未通过 M5 对照；`top20_calibrated` 与固定 ensemble 没有稳定改善 Rank IC，不提升为主模型。
- M6 不进入生产；仅把 `U2_risk_sane + M6_xgboost_rank_ndcg + Top20/Top30` 标记为下一阶段 watchlist，进入 M7 研究报告和后续 regime-aware 复核。

## M6 vs M5 Gate

| candidate_pool_version | top_k | best_m6_model | best_m5_model | m6_after_cost_mean | m5_after_cost_mean | delta_after_cost_mean_vs_m5 | m6_rank_ic_mean | m5_rank_ic_mean | m6_quantile_spread_mean | m5_quantile_spread_mean | gate_pass_count | gate_total | m6_promotion_decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| U1_liquid_tradable | 20 | M6_xgboost_rank_ndcg | M5_plus_industry_breadth_plus_fund_flow_plus_fundamental_extratrees_excess | 0.000214 | 0.011436 | -0.011221 | 0.009297 | 0.107500 | -0.001108 | 0.014534 | 2 | 7 | fail |
| U2_risk_sane | 20 | M6_xgboost_rank_ndcg | M5_plus_industry_breadth_plus_fund_flow_plus_fundamental_extratrees_excess | 0.012106 | 0.009373 | 0.002733 | 0.006662 | 0.088744 | 0.000674 | 0.012681 | 7 | 7 | candidate_watchlist |
| U1_liquid_tradable | 30 | M6_xgboost_rank_pairwise | M5_plus_industry_breadth_plus_fund_flow_plus_fundamental_extratrees_excess | -0.001119 | 0.011175 | -0.012294 | 0.059134 | 0.107500 | 0.004482 | 0.014534 | 4 | 7 | fail |
| U2_risk_sane | 30 | M6_xgboost_rank_ndcg | M5_plus_industry_breadth_extratrees_excess | 0.008630 | 0.008317 | 0.000313 | 0.006662 | 0.079547 | 0.000674 | 0.010672 | 7 | 7 | candidate_watchlist |
| U1_liquid_tradable | 50 | M6_xgboost_rank_pairwise | M5_plus_industry_breadth_plus_fund_flow_plus_fundamental_plus_shareholder_elasticnet_excess | -0.000359 | 0.009747 | -0.010106 | 0.059134 | 0.104324 | 0.004482 | 0.017220 | 5 | 7 | fail |
| U2_risk_sane | 50 | M6_xgboost_rank_ndcg | M5_plus_industry_breadth_plus_fund_flow_plus_fundamental_plus_shareholder_elasticnet_excess | 0.005683 | 0.009074 | -0.003391 | 0.006662 | 0.085823 | 0.000674 | 0.014338 | 4 | 7 | fail |

## 本轮产物

- `data/results/monthly_selection_m6_ltr_2026-04-29_summary.json`
- `data/results/monthly_selection_m6_ltr_2026-04-29_leaderboard.csv`
- `data/results/monthly_selection_m6_ltr_2026-04-29_monthly_long.csv`
- `data/results/monthly_selection_m6_ltr_2026-04-29_rank_ic.csv`
- `data/results/monthly_selection_m6_ltr_2026-04-29_quantile_spread.csv`
- `data/results/monthly_selection_m6_ltr_2026-04-29_feature_coverage.csv`
- `data/results/monthly_selection_m6_ltr_2026-04-29_feature_importance.csv`
- `data/results/monthly_selection_m6_ltr_2026-04-29_topk_holdings.csv`
- `data/results/monthly_selection_m6_ltr_2026-04-29_industry_exposure.csv`
- `data/results/monthly_selection_m6_ltr_2026-04-29_candidate_pool_width.csv`
- `data/results/monthly_selection_m6_ltr_2026-04-29_candidate_pool_reject_reason.csv`
- `data/results/monthly_selection_m6_ltr_2026-04-29_year_slice.csv`
- `data/results/monthly_selection_m6_ltr_2026-04-29_regime_slice.csv`
- `data/results/monthly_selection_m6_ltr_2026-04-29_market_states.csv`
- `data/results/monthly_selection_m6_ltr_2026-04-29_vs_m5_gate.csv`
- `data/results/monthly_selection_m6_ltr_2026-04-29_vs_m5_monthly_delta.csv`
- `data/results/monthly_selection_m6_ltr_2026-04-29_vs_m5_gate_summary.json`
- `data/results/monthly_selection_m6_ltr_2026-04-29_manifest.json`
