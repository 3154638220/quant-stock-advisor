# Monthly Selection M7 Recommendation Report

- 生成时间：`2026-04-30T05:35:14.422072+00:00`
- 结果类型：`monthly_selection_m7_recommendation_report`
- 研究配置：`dataset_monthly_selection_features_20260429_predict_families_price_volume_only-industry_breadth-fund_flow-fundamental_pools_u2_risk_sane_topk_20-30_model_m6_xgboost_rank_ndcg_maxfit_0_wf_24m`
- 报告信号日：`2026-04-29`
- 下一交易日：``
- 候选池：`U2_risk_sane`
- 模型：`M6_xgboost_rank_ndcg`
- 生产状态：`research_only_not_promoted`

## 推荐名单

| signal_date | top_k | rank | symbol | name | score | score_percentile | industry | industry_level2 | feature_contrib | risk_flags | last_month_rank | last_month_selected | buyability | candidate_pool_version | candidate_pool_rule | model | model_type | feature_spec |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-04-29 | 20 | 1 | 688339 | UNKNOWN | 1 | 1 | 电力设备 | 电池 | feature_fundamental_asset_turnover_z=+4.63; feature_fundamental_net_margin_stability_z=-5; feature_fundamental_gross_margin_delta_z=-3.42 | no_next_trade_date |  | False | no_next_trade_date | U2_risk_sane | U1 + exclude extreme limit-move path, extreme volatility/turnover, and absolute-high names | M6_xgboost_rank_ndcg | xgboost_ranker | m6_core_price_volume_industry_breadth_fund_flow_fundamental |
| 2026-04-29 | 20 | 2 | 301129 | UNKNOWN | 0.999744 | 0.999744 | 机械设备 | 通用设备 | feature_fundamental_asset_turnover_z=+4.72; feature_fundamental_ocf_to_asset_z=-5; feature_fundamental_net_margin_stability_z=-4.48 | no_next_trade_date |  | False | no_next_trade_date | U2_risk_sane | U1 + exclude extreme limit-move path, extreme volatility/turnover, and absolute-high names | M6_xgboost_rank_ndcg | xgboost_ranker | m6_core_price_volume_industry_breadth_fund_flow_fundamental |
| 2026-04-29 | 20 | 3 | 000987 | UNKNOWN | 0.999489 | 0.999489 | 非银金融 | 多元金融 | feature_fundamental_asset_turnover_z=+3.88; feature_fundamental_gross_margin_change_z=+3.52; feature_fundamental_ocf_to_asset_z=+3.13 | no_next_trade_date |  | False | no_next_trade_date | U2_risk_sane | U1 + exclude extreme limit-move path, extreme volatility/turnover, and absolute-high names | M6_xgboost_rank_ndcg | xgboost_ranker | m6_core_price_volume_industry_breadth_fund_flow_fundamental |
| 2026-04-29 | 20 | 4 | 300426 | UNKNOWN | 0.999233 | 0.999233 | 传媒 | 影视院线 | feature_fundamental_asset_turnover_z=+5; feature_fundamental_pb_z=+4.7; feature_fundamental_net_margin_stability_z=-5 | no_next_trade_date |  | False | no_next_trade_date | U2_risk_sane | U1 + exclude extreme limit-move path, extreme volatility/turnover, and absolute-high names | M6_xgboost_rank_ndcg | xgboost_ranker | m6_core_price_volume_industry_breadth_fund_flow_fundamental |
| 2026-04-29 | 20 | 5 | 600120 | UNKNOWN | 0.998978 | 0.998978 | 非银金融 | 多元金融 | feature_fundamental_asset_turnover_z=+4.81; feature_fundamental_gross_margin_change_z=+3.8; feature_fundamental_net_margin_stability_z=-3.4 | no_next_trade_date |  | False | no_next_trade_date | U2_risk_sane | U1 + exclude extreme limit-move path, extreme volatility/turnover, and absolute-high names | M6_xgboost_rank_ndcg | xgboost_ranker | m6_core_price_volume_industry_breadth_fund_flow_fundamental |
| 2026-04-29 | 20 | 6 | 000617 | UNKNOWN | 0.998722 | 0.998722 | 非银金融 | 多元金融 | feature_fundamental_asset_turnover_z=+5; feature_fundamental_net_margin_stability_z=-5; feature_fundamental_ocf_to_asset_z=-5 | no_next_trade_date |  | False | no_next_trade_date | U2_risk_sane | U1 + exclude extreme limit-move path, extreme volatility/turnover, and absolute-high names | M6_xgboost_rank_ndcg | xgboost_ranker | m6_core_price_volume_industry_breadth_fund_flow_fundamental |
| 2026-04-29 | 20 | 7 | 600909 | UNKNOWN | 0.998466 | 0.998466 | 非银金融 | 证券Ⅱ | feature_fundamental_asset_turnover_z=+3.94; feature_industry_ret60_mean_z=-1.42; feature_industry_low_vol20_mean_z=+2.48 | no_next_trade_date |  | False | no_next_trade_date | U2_risk_sane | U1 + exclude extreme limit-move path, extreme volatility/turnover, and absolute-high names | M6_xgboost_rank_ndcg | xgboost_ranker | m6_core_price_volume_industry_breadth_fund_flow_fundamental |
| 2026-04-29 | 20 | 8 | 601059 | UNKNOWN | 0.998211 | 0.998211 | 非银金融 | 证券Ⅱ | feature_fundamental_asset_turnover_z=+4.15; feature_industry_ret60_mean_z=-1.42; feature_industry_low_vol20_mean_z=+2.48 | no_next_trade_date |  | False | no_next_trade_date | U2_risk_sane | U1 + exclude extreme limit-move path, extreme volatility/turnover, and absolute-high names | M6_xgboost_rank_ndcg | xgboost_ranker | m6_core_price_volume_industry_breadth_fund_flow_fundamental |
| 2026-04-29 | 20 | 9 | 600340 | UNKNOWN | 0.997955 | 0.997955 | 房地产 | 房地产开发 | feature_fundamental_asset_turnover_z=+5; feature_fundamental_net_margin_stability_z=-5; feature_fundamental_ocf_to_asset_z=-5 | no_next_trade_date | 7 | True | no_next_trade_date | U2_risk_sane | U1 + exclude extreme limit-move path, extreme volatility/turnover, and absolute-high names | M6_xgboost_rank_ndcg | xgboost_ranker | m6_core_price_volume_industry_breadth_fund_flow_fundamental |
| 2026-04-29 | 20 | 10 | 600999 | UNKNOWN | 0.997699 | 0.997699 | 非银金融 | 证券Ⅱ | feature_fundamental_asset_turnover_z=+4.19; feature_industry_ret60_mean_z=-1.42; feature_industry_low_vol20_mean_z=+2.48 | no_next_trade_date |  | False | no_next_trade_date | U2_risk_sane | U1 + exclude extreme limit-move path, extreme volatility/turnover, and absolute-high names | M6_xgboost_rank_ndcg | xgboost_ranker | m6_core_price_volume_industry_breadth_fund_flow_fundamental |
| 2026-04-29 | 20 | 11 | 600061 | UNKNOWN | 0.997444 | 0.997444 | 非银金融 | 证券Ⅱ | feature_fundamental_asset_turnover_z=+3.17; feature_fundamental_net_margin_stability_z=-5; feature_fundamental_gross_margin_delta_z=+3.81 | no_next_trade_date |  | False | no_next_trade_date | U2_risk_sane | U1 + exclude extreme limit-move path, extreme volatility/turnover, and absolute-high names | M6_xgboost_rank_ndcg | xgboost_ranker | m6_core_price_volume_industry_breadth_fund_flow_fundamental |
| 2026-04-29 | 20 | 12 | 600816 | UNKNOWN | 0.997188 | 0.997188 | 非银金融 | 多元金融 | feature_fundamental_asset_turnover_z=+5; feature_fundamental_net_margin_stability_z=-5; feature_fundamental_net_profit_yoy_z=+4.69 | no_next_trade_date |  | False | no_next_trade_date | U2_risk_sane | U1 + exclude extreme limit-move path, extreme volatility/turnover, and absolute-high names | M6_xgboost_rank_ndcg | xgboost_ranker | m6_core_price_volume_industry_breadth_fund_flow_fundamental |
| 2026-04-29 | 20 | 13 | 688089 | UNKNOWN | 0.996933 | 0.996933 | 基础化工 | 化学制品 | feature_fundamental_asset_turnover_z=+5; feature_fundamental_net_margin_stability_z=-5; feature_fundamental_ocf_to_asset_z=-5 | no_next_trade_date |  | False | no_next_trade_date | U2_risk_sane | U1 + exclude extreme limit-move path, extreme volatility/turnover, and absolute-high names | M6_xgboost_rank_ndcg | xgboost_ranker | m6_core_price_volume_industry_breadth_fund_flow_fundamental |
| 2026-04-29 | 20 | 14 | 600783 | UNKNOWN | 0.996677 | 0.996677 | 机械设备 | 通用设备 | feature_fundamental_asset_turnover_z=+5; feature_fundamental_net_margin_stability_z=-5; feature_fundamental_ocf_to_asset_z=+3.13 | no_next_trade_date |  | False | no_next_trade_date | U2_risk_sane | U1 + exclude extreme limit-move path, extreme volatility/turnover, and absolute-high names | M6_xgboost_rank_ndcg | xgboost_ranker | m6_core_price_volume_industry_breadth_fund_flow_fundamental |
| 2026-04-29 | 20 | 15 | 301155 | UNKNOWN | 0.996421 | 0.996421 | 电力设备 | 风电设备 | feature_fundamental_asset_turnover_z=+4.27; feature_fundamental_ocf_to_asset_z=-5; feature_industry_ret60_mean_z=+1.15 | no_next_trade_date |  | False | no_next_trade_date | U2_risk_sane | U1 + exclude extreme limit-move path, extreme volatility/turnover, and absolute-high names | M6_xgboost_rank_ndcg | xgboost_ranker | m6_core_price_volume_industry_breadth_fund_flow_fundamental |
| 2026-04-29 | 20 | 16 | 601162 | UNKNOWN | 0.996166 | 0.996166 | 非银金融 | 证券Ⅱ | feature_fundamental_asset_turnover_z=+4.9; feature_industry_ret60_mean_z=-1.42; feature_industry_low_vol20_mean_z=+2.48 | no_next_trade_date |  | False | no_next_trade_date | U2_risk_sane | U1 + exclude extreme limit-move path, extreme volatility/turnover, and absolute-high names | M6_xgboost_rank_ndcg | xgboost_ranker | m6_core_price_volume_industry_breadth_fund_flow_fundamental |
| 2026-04-29 | 20 | 17 | 600390 | UNKNOWN | 0.99591 | 0.99591 | 非银金融 | 多元金融 | feature_fundamental_asset_turnover_z=+3; feature_fundamental_ocf_to_asset_z=-5; feature_fundamental_gross_margin_delta_z=-3.05 | no_next_trade_date |  | False | no_next_trade_date | U2_risk_sane | U1 + exclude extreme limit-move path, extreme volatility/turnover, and absolute-high names | M6_xgboost_rank_ndcg | xgboost_ranker | m6_core_price_volume_industry_breadth_fund_flow_fundamental |
| 2026-04-29 | 20 | 18 | 601456 | UNKNOWN | 0.995654 | 0.995654 | 非银金融 | 证券Ⅱ | feature_fundamental_asset_turnover_z=+4.21; feature_industry_ret60_mean_z=-1.42; feature_industry_low_vol20_mean_z=+2.48 | no_next_trade_date |  | False | no_next_trade_date | U2_risk_sane | U1 + exclude extreme limit-move path, extreme volatility/turnover, and absolute-high names | M6_xgboost_rank_ndcg | xgboost_ranker | m6_core_price_volume_industry_breadth_fund_flow_fundamental |
| 2026-04-29 | 20 | 19 | 600927 | UNKNOWN | 0.995399 | 0.995399 | 非银金融 | 多元金融 | feature_fundamental_asset_turnover_z=+5; feature_fundamental_ocf_to_asset_z=+3.13; feature_industry_ret60_mean_z=-1.42 | no_next_trade_date |  | False | no_next_trade_date | U2_risk_sane | U1 + exclude extreme limit-move path, extreme volatility/turnover, and absolute-high names | M6_xgboost_rank_ndcg | xgboost_ranker | m6_core_price_volume_industry_breadth_fund_flow_fundamental |
| 2026-04-29 | 20 | 20 | 000728 | UNKNOWN | 0.995143 | 0.995143 | 非银金融 | 证券Ⅱ | feature_fundamental_asset_turnover_z=+5; feature_industry_ret60_mean_z=-1.42; feature_industry_low_vol20_mean_z=+2.48 | no_next_trade_date |  | False | no_next_trade_date | U2_risk_sane | U1 + exclude extreme limit-move path, extreme volatility/turnover, and absolute-high names | M6_xgboost_rank_ndcg | xgboost_ranker | m6_core_price_volume_industry_breadth_fund_flow_fundamental |

## M6 Historical Evidence

| candidate_pool_version | model | model_type | top_k | months | mean_topk_return | median_topk_return | topk_excess_mean | topk_excess_median | topk_hit_rate | topk_minus_nextk_mean | industry_neutral_topk_excess_mean | turnover_mean | cost_drag_mean | topk_excess_after_cost_mean | topk_excess_annualized | rank_ic_mean | rank_ic_std | rank_ic_months | rank_ic_ir | quantile_top_minus_bottom_mean | neutral_median_excess | strong_down_median_excess | strong_up_median_excess |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| U2_risk_sane | M6_xgboost_rank_ndcg | xgboost_ranker | 20 | 38 | 0.0247633 | 0.028524 | 0.0126144 | 0.0147394 | 0.657895 | 0.00762405 | 0.00384424 | 0.940541 | 0.000940541 | 0.0121064 | 0.16233 | 0.00666237 | 0.0777473 | 38 | 0.0856927 | 0.000673815 | 0.0131781 | -0.00184697 | 0.0181031 |
| U2_risk_sane | M6_xgboost_rank_ndcg | xgboost_ranker | 30 | 38 | 0.0214441 | 0.0241337 | 0.0092952 | 0.00160447 | 0.552632 | 0.010155 | 0.00158023 | 0.931532 | 0.000931532 | 0.00862985 | 0.117425 | 0.00666237 | 0.0777473 | 38 | 0.0856927 | 0.000673815 | 0.00361303 | -0.00236666 | 0.000249559 |

## Risk Summary

| signal_date | candidate_pool_version | model | top_k | selected_count | risk_flagged_count | not_buyable_count | last_month_selected_count |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-04-29 | U2_risk_sane | M6_xgboost_rank_ndcg | 20 | 20 | 20 | 20 | 1 |
| 2026-04-29 | U2_risk_sane | M6_xgboost_rank_ndcg | 30 | 30 | 30 | 30 | 1 |

## Industry Exposure

| signal_date | candidate_pool_version | model | top_k | industry | industry_count | topk_count | industry_share |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-04-29 | U2_risk_sane | M6_xgboost_rank_ndcg | 20 | 传媒 | 1 | 20 | 0.05 |
| 2026-04-29 | U2_risk_sane | M6_xgboost_rank_ndcg | 20 | 基础化工 | 1 | 20 | 0.05 |
| 2026-04-29 | U2_risk_sane | M6_xgboost_rank_ndcg | 20 | 房地产 | 1 | 20 | 0.05 |
| 2026-04-29 | U2_risk_sane | M6_xgboost_rank_ndcg | 20 | 机械设备 | 2 | 20 | 0.1 |
| 2026-04-29 | U2_risk_sane | M6_xgboost_rank_ndcg | 20 | 电力设备 | 2 | 20 | 0.1 |
| 2026-04-29 | U2_risk_sane | M6_xgboost_rank_ndcg | 20 | 非银金融 | 13 | 20 | 0.65 |
| 2026-04-29 | U2_risk_sane | M6_xgboost_rank_ndcg | 30 | _UNKNOWN_ | 1 | 30 | 0.0333333 |
| 2026-04-29 | U2_risk_sane | M6_xgboost_rank_ndcg | 30 | 传媒 | 1 | 30 | 0.0333333 |
| 2026-04-29 | U2_risk_sane | M6_xgboost_rank_ndcg | 30 | 医药生物 | 1 | 30 | 0.0333333 |
| 2026-04-29 | U2_risk_sane | M6_xgboost_rank_ndcg | 30 | 国防军工 | 1 | 30 | 0.0333333 |
| 2026-04-29 | U2_risk_sane | M6_xgboost_rank_ndcg | 30 | 基础化工 | 1 | 30 | 0.0333333 |
| 2026-04-29 | U2_risk_sane | M6_xgboost_rank_ndcg | 30 | 房地产 | 1 | 30 | 0.0333333 |
| 2026-04-29 | U2_risk_sane | M6_xgboost_rank_ndcg | 30 | 机械设备 | 2 | 30 | 0.0666667 |
| 2026-04-29 | U2_risk_sane | M6_xgboost_rank_ndcg | 30 | 电力设备 | 2 | 30 | 0.0666667 |
| 2026-04-29 | U2_risk_sane | M6_xgboost_rank_ndcg | 30 | 非银金融 | 20 | 30 | 0.666667 |

## Feature Coverage

| candidate_pool_version | feature_spec | families | feature | raw_feature | rows | candidate_pool_pass_rows | non_null | coverage_ratio | candidate_pool_pass_coverage_ratio | first_signal_date | last_signal_date |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| U2_risk_sane | m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_ret_20d_z | feature_ret_20d | 307350 | 191169 | 306082 | 0.995874 | 1 | 2021-01-29 | 2026-04-29 |
| U2_risk_sane | m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_ret_60d_z | feature_ret_60d | 307350 | 191169 | 303506 | 0.987493 | 1 | 2021-01-29 | 2026-04-29 |
| U2_risk_sane | m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_ret_5d_z | feature_ret_5d | 307350 | 191169 | 307017 | 0.998917 | 1 | 2021-01-29 | 2026-04-29 |
| U2_risk_sane | m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_realized_vol_20d_z | feature_realized_vol_20d | 307350 | 191169 | 306673 | 0.997797 | 1 | 2021-01-29 | 2026-04-29 |
| U2_risk_sane | m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_amount_20d_log_z | feature_amount_20d_log | 307350 | 191169 | 306745 | 0.998032 | 1 | 2021-01-29 | 2026-04-29 |
| U2_risk_sane | m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_turnover_20d_z | feature_turnover_20d | 307350 | 191169 | 306745 | 0.998032 | 1 | 2021-01-29 | 2026-04-29 |
| U2_risk_sane | m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_price_position_250d_z | feature_price_position_250d | 307350 | 191169 | 303572 | 0.987708 | 1 | 2021-01-29 | 2026-04-29 |
| U2_risk_sane | m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_limit_move_hits_20d_z | feature_limit_move_hits_20d | 307350 | 191169 | 307350 | 1 | 1 | 2021-01-29 | 2026-04-29 |
| U2_risk_sane | m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_industry_ret20_mean_z | feature_industry_ret20_mean | 307350 | 191169 | 307350 | 1 | 1 | 2021-01-29 | 2026-04-29 |
| U2_risk_sane | m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_industry_ret60_mean_z | feature_industry_ret60_mean | 307350 | 191169 | 307350 | 1 | 1 | 2021-01-29 | 2026-04-29 |
| U2_risk_sane | m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_industry_positive_ret20_ratio_z | feature_industry_positive_ret20_ratio | 307350 | 191169 | 307350 | 1 | 1 | 2021-01-29 | 2026-04-29 |
| U2_risk_sane | m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_industry_amount20_mean_z | feature_industry_amount20_mean | 307350 | 191169 | 307350 | 1 | 1 | 2021-01-29 | 2026-04-29 |
| U2_risk_sane | m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_industry_low_vol20_mean_z | feature_industry_low_vol20_mean | 307350 | 191169 | 307350 | 1 | 1 | 2021-01-29 | 2026-04-29 |
| U2_risk_sane | m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_fund_flow_main_inflow_5d_z | feature_fund_flow_main_inflow_5d | 307350 | 191169 | 35590 | 0.115796 | 0.144563 | 2023-10-31 | 2026-04-29 |
| U2_risk_sane | m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_fund_flow_main_inflow_10d_z | feature_fund_flow_main_inflow_10d | 307350 | 191169 | 31196 | 0.1015 | 0.126501 | 2023-11-30 | 2026-04-29 |
| U2_risk_sane | m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_fund_flow_main_inflow_20d_z | feature_fund_flow_main_inflow_20d | 307350 | 191169 | 31048 | 0.101018 | 0.12603 | 2023-11-30 | 2026-04-29 |
| U2_risk_sane | m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_fund_flow_super_inflow_10d_z | feature_fund_flow_super_inflow_10d | 307350 | 191169 | 31196 | 0.1015 | 0.126501 | 2023-11-30 | 2026-04-29 |
| U2_risk_sane | m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_fund_flow_divergence_20d_z | feature_fund_flow_divergence_20d | 307350 | 191169 | 31048 | 0.101018 | 0.12603 | 2023-11-30 | 2026-04-29 |
| U2_risk_sane | m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_fund_flow_main_inflow_streak_z | feature_fund_flow_main_inflow_streak | 307350 | 191169 | 36162 | 0.117657 | 0.146891 | 2023-10-31 | 2026-04-29 |
| U2_risk_sane | m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_fundamental_pe_ttm_z | feature_fundamental_pe_ttm | 307350 | 191169 | 304197 | 0.989741 | 0.998739 | 2021-01-29 | 2026-04-29 |
| U2_risk_sane | m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_fundamental_pb_z | feature_fundamental_pb | 307350 | 191169 | 304197 | 0.989741 | 0.998739 | 2021-01-29 | 2026-04-29 |
| U2_risk_sane | m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_fundamental_ev_ebitda_z | feature_fundamental_ev_ebitda | 307350 | 191169 | 0 | 0 | 0 |  |  |
| U2_risk_sane | m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_fundamental_roe_ttm_z | feature_fundamental_roe_ttm | 307350 | 191169 | 304622 | 0.991124 | 0.993226 | 2021-01-29 | 2026-04-29 |
| U2_risk_sane | m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_fundamental_net_profit_yoy_z | feature_fundamental_net_profit_yoy | 307350 | 191169 | 306093 | 0.99591 | 0.996004 | 2021-01-29 | 2026-04-29 |
| U2_risk_sane | m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_fundamental_gross_margin_change_z | feature_fundamental_gross_margin_change | 307350 | 191169 | 299649 | 0.974944 | 0.966088 | 2021-01-29 | 2026-04-29 |
| U2_risk_sane | m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_fundamental_gross_margin_delta_z | feature_fundamental_gross_margin_delta | 307350 | 191169 | 299525 | 0.97454 | 0.965842 | 2021-01-29 | 2026-04-29 |
| U2_risk_sane | m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_fundamental_debt_to_assets_change_z | feature_fundamental_debt_to_assets_change | 307350 | 191169 | 306204 | 0.996271 | 0.997071 | 2021-01-29 | 2026-04-29 |
| U2_risk_sane | m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_fundamental_ocf_to_net_profit_z | feature_fundamental_ocf_to_net_profit | 307350 | 191169 | 245641 | 0.799222 | 0.815205 | 2021-01-29 | 2026-04-29 |
| U2_risk_sane | m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_fundamental_ocf_to_asset_z | feature_fundamental_ocf_to_asset | 307350 | 191169 | 305945 | 0.995429 | 0.995543 | 2021-01-29 | 2026-04-29 |
| U2_risk_sane | m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_fundamental_asset_turnover_z | feature_fundamental_asset_turnover | 307350 | 191169 | 306026 | 0.995692 | 0.995873 | 2021-01-29 | 2026-04-29 |
| U2_risk_sane | m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_fundamental_net_margin_stability_z | feature_fundamental_net_margin_stability | 307350 | 191169 | 305876 | 0.995204 | 0.995637 | 2021-01-29 | 2026-04-29 |

## 口径与结论

- 本轮新增的是 M7 研究版推荐报告，不新增生产策略；推荐名单不自动生成交易指令。
- 数据质量沿用 M2/M5/M6 的 PIT 检查，本脚本只使用 `report_signal_date` 当日可观测特征，并只用该日期之前已完成标签的月份训练。
- 候选池使用 `U2_risk_sane` watchlist 口径，只做准入过滤；alpha 判断来自 `M6_xgboost_rank_ndcg` 排序分数。
- 历史 baseline 改善证据来自 M6 walk-forward 附件；M6 结论仍是 watchlist，不进入生产。
- 当前推荐月尚无未来收益标签，因此本报告不声称本月已跑赢市场；稳定性判断以历史 leaderboard / monthly_long / rank_ic / quantile_spread 为准。
- 分桶、年份和 regime 的历史失败月份保留在附件中；后续进入 regime-aware calibration 复核。

## 本轮产物

- `data/results/monthly_selection_m7_recommendation_report_cutoff_2026_04_29_2026-04-30_summary.json`
- `data/results/monthly_selection_m7_recommendation_report_cutoff_2026_04_29_2026-04-30_recommendations.csv`
- `data/results/monthly_selection_m7_recommendation_report_cutoff_2026_04_29_2026-04-30_leaderboard.csv`
- `data/results/monthly_selection_m7_recommendation_report_cutoff_2026_04_29_2026-04-30_monthly_long.csv`
- `data/results/monthly_selection_m7_recommendation_report_cutoff_2026_04_29_2026-04-30_rank_ic.csv`
- `data/results/monthly_selection_m7_recommendation_report_cutoff_2026_04_29_2026-04-30_quantile_spread.csv`
- `data/results/monthly_selection_m7_recommendation_report_cutoff_2026_04_29_2026-04-30_topk_holdings.csv`
- `data/results/monthly_selection_m7_recommendation_report_cutoff_2026_04_29_2026-04-30_industry_exposure.csv`
- `data/results/monthly_selection_m7_recommendation_report_cutoff_2026_04_29_2026-04-30_candidate_pool_width.csv`
- `data/results/monthly_selection_m7_recommendation_report_cutoff_2026_04_29_2026-04-30_candidate_pool_reject_reason.csv`
- `data/results/monthly_selection_m7_recommendation_report_cutoff_2026_04_29_2026-04-30_feature_importance.csv`
- `data/results/monthly_selection_m7_recommendation_report_cutoff_2026_04_29_2026-04-30_feature_contrib.csv`
- `data/results/monthly_selection_m7_recommendation_report_cutoff_2026_04_29_2026-04-30_feature_coverage.csv`
- `data/results/monthly_selection_m7_recommendation_report_cutoff_2026_04_29_2026-04-30_risk_summary.csv`
- `data/results/monthly_selection_m7_recommendation_report_cutoff_2026_04_29_2026-04-30_year_slice.csv`
- `data/results/monthly_selection_m7_recommendation_report_cutoff_2026_04_29_2026-04-30_regime_slice.csv`
- `data/results/monthly_selection_m7_recommendation_report_cutoff_2026_04_29_2026-04-30_manifest.json`
