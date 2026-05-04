# Monthly Selection M7 Recommendation Report

- 生成时间：`2026-05-04T03:43:39.551823+00:00`
- 结果类型：`monthly_selection_m7_recommendation_report`
- 研究配置：`dataset_monthly_selection_features_families_price_volume_only-industry_breadth-fund_flow-fundamental_pools_u2_risk_sane_topk_20-30_model_m6_xgboost_rank_ndcg_maxfit_0_jobs_all_wf_24m`
- 报告信号日：`2026-03-31`
- 买入交易日：`2026-04-01`
- 卖出交易日：`2026-04-30`
- 候选池：`U2_risk_sane`
- 模型：`M6_xgboost_rank_ndcg`
- 生产状态：`research_only_not_promoted`

## 推荐名单

| signal_date | top_k | rank | symbol | name | score | score_percentile | industry | industry_level2 | feature_contrib | risk_flags | last_month_rank | last_month_selected | buyability | next_trade_date | buy_trade_date | sell_trade_date | candidate_pool_version | candidate_pool_rule | model | model_type | feature_spec |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-03-31 | 20 | 1 | 600999 | 招商证券 | 1 | 1 | 非银金融 | 证券Ⅱ | feature_fundamental_asset_turnover_z=+4.74; feature_industry_low_vol20_mean_z=+2.69; feature_industry_ret60_mean_z=-2.05 |  |  | False | buyable_tplus1_open | 2026-04-01 | 2026-04-01 | 2026-04-30 | U2_risk_sane | U1 + exclude extreme limit-move path, extreme volatility/turnover, and absolute-high names | M6_xgboost_rank_ndcg | xgboost_ranker | m6_core_price_volume_industry_breadth_fund_flow_fundamental |
| 2026-03-31 | 20 | 2 | 000166 | 申万宏源 | 0.999754 | 0.999754 | 非银金融 | 证券Ⅱ | feature_fundamental_asset_turnover_z=+4.76; feature_industry_low_vol20_mean_z=+2.69; feature_industry_ret60_mean_z=-2.05 |  |  | False | buyable_tplus1_open | 2026-04-01 | 2026-04-01 | 2026-04-30 | U2_risk_sane | U1 + exclude extreme limit-move path, extreme volatility/turnover, and absolute-high names | M6_xgboost_rank_ndcg | xgboost_ranker | m6_core_price_volume_industry_breadth_fund_flow_fundamental |
| 2026-03-31 | 20 | 3 | 000728 | 国元证券 | 0.999508 | 0.999508 | 非银金融 | 证券Ⅱ | feature_fundamental_asset_turnover_z=+4.26; feature_industry_low_vol20_mean_z=+2.69; feature_industry_ret60_mean_z=-2.05 |  | 3 | True | buyable_tplus1_open | 2026-04-01 | 2026-04-01 | 2026-04-30 | U2_risk_sane | U1 + exclude extreme limit-move path, extreme volatility/turnover, and absolute-high names | M6_xgboost_rank_ndcg | xgboost_ranker | m6_core_price_volume_industry_breadth_fund_flow_fundamental |
| 2026-03-31 | 20 | 4 | 000776 | 广发证券 | 0.999262 | 0.999262 | 非银金融 | 证券Ⅱ | feature_fundamental_asset_turnover_z=+3.8; feature_industry_low_vol20_mean_z=+2.69; feature_industry_ret60_mean_z=-2.05 |  | 4 | True | buyable_tplus1_open | 2026-04-01 | 2026-04-01 | 2026-04-30 | U2_risk_sane | U1 + exclude extreme limit-move path, extreme volatility/turnover, and absolute-high names | M6_xgboost_rank_ndcg | xgboost_ranker | m6_core_price_volume_industry_breadth_fund_flow_fundamental |
| 2026-03-31 | 20 | 5 | 601211 | 国泰海通 | 0.999016 | 0.999016 | 非银金融 | 证券Ⅱ | feature_fundamental_asset_turnover_z=+3.92; feature_industry_low_vol20_mean_z=+2.69; feature_industry_ret60_mean_z=-2.05 |  |  | False | buyable_tplus1_open | 2026-04-01 | 2026-04-01 | 2026-04-30 | U2_risk_sane | U1 + exclude extreme limit-move path, extreme volatility/turnover, and absolute-high names | M6_xgboost_rank_ndcg | xgboost_ranker | m6_core_price_volume_industry_breadth_fund_flow_fundamental |
| 2026-03-31 | 20 | 6 | 601377 | 兴业证券 | 0.998771 | 0.998771 | 非银金融 | 证券Ⅱ | feature_fundamental_asset_turnover_z=+4.33; feature_industry_low_vol20_mean_z=+2.69; feature_industry_ret60_mean_z=-2.05 |  | 7 | True | buyable_tplus1_open | 2026-04-01 | 2026-04-01 | 2026-04-30 | U2_risk_sane | U1 + exclude extreme limit-move path, extreme volatility/turnover, and absolute-high names | M6_xgboost_rank_ndcg | xgboost_ranker | m6_core_price_volume_industry_breadth_fund_flow_fundamental |
| 2026-03-31 | 20 | 7 | 600340 | 华夏幸福 | 0.987214 | 0.987214 | _UNKNOWN_ |  | feature_fundamental_asset_turnover_z=+5; feature_fundamental_gross_margin_delta_z=-3.85; feature_fundamental_ocf_to_asset_z=-4.82 |  |  | False | buyable_tplus1_open | 2026-04-01 | 2026-04-01 | 2026-04-30 | U2_risk_sane | U1 + exclude extreme limit-move path, extreme volatility/turnover, and absolute-high names | M6_xgboost_rank_ndcg | xgboost_ranker | m6_core_price_volume_industry_breadth_fund_flow_fundamental |
| 2026-03-31 | 20 | 8 | 002640 | 跨境通 | 0.984264 | 0.984264 | 商贸零售 | 互联网电商 | feature_fundamental_roe_ttm_z=-5; feature_industry_ret60_mean_z=-2.05; feature_industry_low_vol20_mean_z=+1.6 |  |  | False | buyable_tplus1_open | 2026-04-01 | 2026-04-01 | 2026-04-30 | U2_risk_sane | U1 + exclude extreme limit-move path, extreme volatility/turnover, and absolute-high names | M6_xgboost_rank_ndcg | xgboost_ranker | m6_core_price_volume_industry_breadth_fund_flow_fundamental |
| 2026-03-31 | 20 | 9 | 300518 | 新迅达 | 0.981067 | 0.981067 | 商贸零售 | 互联网电商 | feature_industry_ret60_mean_z=-2.05; feature_fundamental_net_margin_stability_z=-5; feature_industry_low_vol20_mean_z=+1.6 |  |  | False | buyable_tplus1_open | 2026-04-01 | 2026-04-01 | 2026-04-30 | U2_risk_sane | U1 + exclude extreme limit-move path, extreme volatility/turnover, and absolute-high names | M6_xgboost_rank_ndcg | xgboost_ranker | m6_core_price_volume_industry_breadth_fund_flow_fundamental |
| 2026-03-31 | 20 | 10 | 000759 | 中百集团 | 0.980329 | 0.980329 | 商贸零售 | 一般零售 | feature_fundamental_roe_ttm_z=-3.87; feature_industry_ret60_mean_z=-2.05; feature_industry_low_vol20_mean_z=+1.6 |  |  | False | buyable_tplus1_open | 2026-04-01 | 2026-04-01 | 2026-04-30 | U2_risk_sane | U1 + exclude extreme limit-move path, extreme volatility/turnover, and absolute-high names | M6_xgboost_rank_ndcg | xgboost_ranker | m6_core_price_volume_industry_breadth_fund_flow_fundamental |
| 2026-03-31 | 20 | 11 | 000564 | 供销大集 | 0.980084 | 0.980084 | 商贸零售 | 一般零售 | feature_fundamental_asset_turnover_z=+3.36; feature_industry_ret60_mean_z=-2.05; feature_fundamental_ocf_to_asset_z=-4.82 |  |  | False | buyable_tplus1_open | 2026-04-01 | 2026-04-01 | 2026-04-30 | U2_risk_sane | U1 + exclude extreme limit-move path, extreme volatility/turnover, and absolute-high names | M6_xgboost_rank_ndcg | xgboost_ranker | m6_core_price_volume_industry_breadth_fund_flow_fundamental |
| 2026-03-31 | 20 | 12 | 600958 | 东方证券 | 0.979838 | 0.979838 | _UNKNOWN_ |  | feature_fundamental_asset_turnover_z=+4.72; feature_industry_ret60_mean_z=-1.18; feature_industry_amount20_mean_z=-1.36 |  |  | False | buyable_tplus1_open | 2026-04-01 | 2026-04-01 | 2026-04-30 | U2_risk_sane | U1 + exclude extreme limit-move path, extreme volatility/turnover, and absolute-high names | M6_xgboost_rank_ndcg | xgboost_ranker | m6_core_price_volume_industry_breadth_fund_flow_fundamental |
| 2026-03-31 | 20 | 13 | 600710 | 苏美达 | 0.979592 | 0.979592 | 商贸零售 | 贸易Ⅱ | feature_industry_ret60_mean_z=-2.05; feature_industry_low_vol20_mean_z=+1.6; feature_industry_amount20_mean_z=-1.01 |  |  | False | buyable_tplus1_open | 2026-04-01 | 2026-04-01 | 2026-04-30 | U2_risk_sane | U1 + exclude extreme limit-move path, extreme volatility/turnover, and absolute-high names | M6_xgboost_rank_ndcg | xgboost_ranker | m6_core_price_volume_industry_breadth_fund_flow_fundamental |
| 2026-03-31 | 20 | 14 | 600655 | 豫园股份 | 0.979346 | 0.979346 | 商贸零售 | 一般零售 | feature_fundamental_net_profit_yoy_z=-5; feature_industry_ret60_mean_z=-2.05; feature_industry_low_vol20_mean_z=+1.6 |  |  | False | buyable_tplus1_open | 2026-04-01 | 2026-04-01 | 2026-04-30 | U2_risk_sane | U1 + exclude extreme limit-move path, extreme volatility/turnover, and absolute-high names | M6_xgboost_rank_ndcg | xgboost_ranker | m6_core_price_volume_industry_breadth_fund_flow_fundamental |
| 2026-03-31 | 20 | 15 | 300076 | GQY视讯 | 0.978117 | 0.978117 | _UNKNOWN_ |  | feature_fundamental_asset_turnover_z=+3.57; feature_fundamental_ocf_to_asset_z=-4.82; feature_fundamental_gross_margin_change_z=+2.77 |  |  | False | buyable_tplus1_open | 2026-04-01 | 2026-04-01 | 2026-04-30 | U2_risk_sane | U1 + exclude extreme limit-move path, extreme volatility/turnover, and absolute-high names | M6_xgboost_rank_ndcg | xgboost_ranker | m6_core_price_volume_industry_breadth_fund_flow_fundamental |
| 2026-03-31 | 20 | 16 | 000001 | 平安银行 | 0.976641 | 0.976641 | 银行 | 股份制银行Ⅱ | feature_fundamental_asset_turnover_z=+5; feature_industry_low_vol20_mean_z=+2.69; feature_industry_positive_ret20_ratio_z=+2.37 |  |  | False | buyable_tplus1_open | 2026-04-01 | 2026-04-01 | 2026-04-30 | U2_risk_sane | U1 + exclude extreme limit-move path, extreme volatility/turnover, and absolute-high names | M6_xgboost_rank_ndcg | xgboost_ranker | m6_core_price_volume_industry_breadth_fund_flow_fundamental |
| 2026-03-31 | 20 | 17 | 601169 | 北京银行 | 0.975166 | 0.975166 | 银行 | 城商行Ⅱ | feature_fundamental_asset_turnover_z=+5; feature_industry_low_vol20_mean_z=+2.69; feature_industry_positive_ret20_ratio_z=+2.37 |  |  | False | buyable_tplus1_open | 2026-04-01 | 2026-04-01 | 2026-04-30 | U2_risk_sane | U1 + exclude extreme limit-move path, extreme volatility/turnover, and absolute-high names | M6_xgboost_rank_ndcg | xgboost_ranker | m6_core_price_volume_industry_breadth_fund_flow_fundamental |
| 2026-03-31 | 20 | 18 | 600015 | 华夏银行 | 0.974182 | 0.974182 | 银行 | 股份制银行Ⅱ | feature_fundamental_asset_turnover_z=+5; feature_industry_low_vol20_mean_z=+2.69; feature_industry_positive_ret20_ratio_z=+2.37 |  |  | False | buyable_tplus1_open | 2026-04-01 | 2026-04-01 | 2026-04-30 | U2_risk_sane | U1 + exclude extreme limit-move path, extreme volatility/turnover, and absolute-high names | M6_xgboost_rank_ndcg | xgboost_ranker | m6_core_price_volume_industry_breadth_fund_flow_fundamental |
| 2026-03-31 | 20 | 19 | 600073 | 光明肉业 | 0.973937 | 0.973937 | 食品饮料 | 食品加工 | feature_industry_low_vol20_mean_z=+2.07; feature_industry_ret60_mean_z=-1.88; feature_fundamental_debt_to_assets_change_z=+2.68 |  |  | False | buyable_tplus1_open | 2026-04-01 | 2026-04-01 | 2026-04-30 | U2_risk_sane | U1 + exclude extreme limit-move path, extreme volatility/turnover, and absolute-high names | M6_xgboost_rank_ndcg | xgboost_ranker | m6_core_price_volume_industry_breadth_fund_flow_fundamental |
| 2026-03-31 | 20 | 20 | 601288 | 农业银行 | 0.972707 | 0.972707 | 银行 | 国有大型银行Ⅱ | feature_fundamental_asset_turnover_z=+5; feature_industry_low_vol20_mean_z=+2.69; feature_industry_positive_ret20_ratio_z=+2.37 |  |  | False | buyable_tplus1_open | 2026-04-01 | 2026-04-01 | 2026-04-30 | U2_risk_sane | U1 + exclude extreme limit-move path, extreme volatility/turnover, and absolute-high names | M6_xgboost_rank_ndcg | xgboost_ranker | m6_core_price_volume_industry_breadth_fund_flow_fundamental |

## M6 Historical Evidence

| candidate_pool_version | model | model_type | top_k | months | mean_topk_return | median_topk_return | topk_excess_mean | topk_excess_median | topk_hit_rate | topk_minus_nextk_mean | industry_neutral_topk_excess_mean | turnover_mean | cost_drag_mean | topk_excess_after_cost_mean | topk_excess_annualized | rank_ic_mean | rank_ic_std | rank_ic_months | rank_ic_ir | quantile_top_minus_bottom_mean | neutral_median_excess | strong_down_median_excess | strong_up_median_excess |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| U2_risk_sane | M6_xgboost_rank_ndcg | xgboost_ranker | 20 | 39 | -0.00516272 | 0.000743067 | -0.00806737 | -0.0117587 | 0.358974 | -0.00830543 | -0.00549767 | 0.952632 | 0.000952632 | -0.00890087 | -0.0926265 | -0.00499501 | 0.105277 | 39 | -0.0474465 | -0.00220577 | -0.0102033 | -0.00971033 | -0.0444757 |

## Risk Summary

| signal_date | candidate_pool_version | model | top_k | selected_count | risk_flagged_count | not_buyable_count | last_month_selected_count |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-03-31 | U2_risk_sane | M6_xgboost_rank_ndcg | 20 | 20 | 0 | 0 | 3 |
| 2026-03-31 | U2_risk_sane | M6_xgboost_rank_ndcg | 30 | 30 | 0 | 0 | 0 |

## Industry Exposure

| signal_date | candidate_pool_version | model | top_k | industry | industry_count | topk_count | industry_share |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-03-31 | U2_risk_sane | M6_xgboost_rank_ndcg | 20 | _UNKNOWN_ | 3 | 20 | 0.15 |
| 2026-03-31 | U2_risk_sane | M6_xgboost_rank_ndcg | 20 | 商贸零售 | 6 | 20 | 0.3 |
| 2026-03-31 | U2_risk_sane | M6_xgboost_rank_ndcg | 20 | 银行 | 4 | 20 | 0.2 |
| 2026-03-31 | U2_risk_sane | M6_xgboost_rank_ndcg | 20 | 非银金融 | 6 | 20 | 0.3 |
| 2026-03-31 | U2_risk_sane | M6_xgboost_rank_ndcg | 20 | 食品饮料 | 1 | 20 | 0.05 |
| 2026-03-31 | U2_risk_sane | M6_xgboost_rank_ndcg | 30 | _UNKNOWN_ | 3 | 30 | 0.1 |
| 2026-03-31 | U2_risk_sane | M6_xgboost_rank_ndcg | 30 | 商贸零售 | 9 | 30 | 0.3 |
| 2026-03-31 | U2_risk_sane | M6_xgboost_rank_ndcg | 30 | 汽车 | 1 | 30 | 0.0333333 |
| 2026-03-31 | U2_risk_sane | M6_xgboost_rank_ndcg | 30 | 银行 | 7 | 30 | 0.233333 |
| 2026-03-31 | U2_risk_sane | M6_xgboost_rank_ndcg | 30 | 非银金融 | 9 | 30 | 0.3 |
| 2026-03-31 | U2_risk_sane | M6_xgboost_rank_ndcg | 30 | 食品饮料 | 1 | 30 | 0.0333333 |

## Feature Coverage

| candidate_pool_version | feature_spec | families | feature | raw_feature | rows | candidate_pool_pass_rows | non_null | coverage_ratio | candidate_pool_pass_coverage_ratio | first_signal_date | last_signal_date |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| U2_risk_sane | m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_ret_20d_z | feature_ret_20d | 307350 | 187495 | 306083 | 0.995878 | 1 | 2021-01-29 | 2026-04-30 |
| U2_risk_sane | m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_ret_60d_z | feature_ret_60d | 307350 | 187495 | 303506 | 0.987493 | 1 | 2021-01-29 | 2026-04-30 |
| U2_risk_sane | m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_ret_5d_z | feature_ret_5d | 307350 | 187495 | 307017 | 0.998917 | 1 | 2021-01-29 | 2026-04-30 |
| U2_risk_sane | m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_realized_vol_20d_z | feature_realized_vol_20d | 307350 | 187495 | 306673 | 0.997797 | 1 | 2021-01-29 | 2026-04-30 |
| U2_risk_sane | m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_amount_20d_log_z | feature_amount_20d_log | 307350 | 187495 | 306745 | 0.998032 | 1 | 2021-01-29 | 2026-04-30 |
| U2_risk_sane | m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_turnover_20d_z | feature_turnover_20d | 307350 | 187495 | 306745 | 0.998032 | 1 | 2021-01-29 | 2026-04-30 |
| U2_risk_sane | m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_price_position_250d_z | feature_price_position_250d | 307350 | 187495 | 303573 | 0.987711 | 1 | 2021-01-29 | 2026-04-30 |
| U2_risk_sane | m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_limit_move_hits_20d_z | feature_limit_move_hits_20d | 307350 | 187495 | 307350 | 1 | 1 | 2021-01-29 | 2026-04-30 |
| U2_risk_sane | m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_industry_ret20_mean_z | feature_industry_ret20_mean | 307350 | 187495 | 307350 | 1 | 1 | 2021-01-29 | 2026-04-30 |
| U2_risk_sane | m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_industry_ret60_mean_z | feature_industry_ret60_mean | 307350 | 187495 | 307350 | 1 | 1 | 2021-01-29 | 2026-04-30 |
| U2_risk_sane | m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_industry_positive_ret20_ratio_z | feature_industry_positive_ret20_ratio | 307350 | 187495 | 307350 | 1 | 1 | 2021-01-29 | 2026-04-30 |
| U2_risk_sane | m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_industry_amount20_mean_z | feature_industry_amount20_mean | 307350 | 187495 | 307350 | 1 | 1 | 2021-01-29 | 2026-04-30 |
| U2_risk_sane | m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_industry_low_vol20_mean_z | feature_industry_low_vol20_mean | 307350 | 187495 | 307350 | 1 | 1 | 2021-01-29 | 2026-04-30 |
| U2_risk_sane | m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_fund_flow_main_inflow_5d_z | feature_fund_flow_main_inflow_5d | 307350 | 187495 | 35590 | 0.115796 | 0.126531 | 2023-10-31 | 2026-04-30 |
| U2_risk_sane | m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_fund_flow_main_inflow_10d_z | feature_fund_flow_main_inflow_10d | 307350 | 187495 | 31196 | 0.1015 | 0.108115 | 2023-11-30 | 2026-04-30 |
| U2_risk_sane | m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_fund_flow_main_inflow_20d_z | feature_fund_flow_main_inflow_20d | 307350 | 187495 | 31048 | 0.101018 | 0.107635 | 2023-11-30 | 2026-04-30 |
| U2_risk_sane | m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_fund_flow_super_inflow_10d_z | feature_fund_flow_super_inflow_10d | 307350 | 187495 | 31196 | 0.1015 | 0.108115 | 2023-11-30 | 2026-04-30 |
| U2_risk_sane | m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_fund_flow_divergence_20d_z | feature_fund_flow_divergence_20d | 307350 | 187495 | 31048 | 0.101018 | 0.107635 | 2023-11-30 | 2026-04-30 |
| U2_risk_sane | m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_fund_flow_main_inflow_streak_z | feature_fund_flow_main_inflow_streak | 307350 | 187495 | 36162 | 0.117657 | 0.128905 | 2023-10-31 | 2026-04-30 |
| U2_risk_sane | m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_fundamental_pe_ttm_z | feature_fundamental_pe_ttm | 307350 | 187495 | 304469 | 0.990626 | 0.999803 | 2021-01-29 | 2026-04-30 |
| U2_risk_sane | m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_fundamental_pb_z | feature_fundamental_pb | 307350 | 187495 | 304469 | 0.990626 | 0.999803 | 2021-01-29 | 2026-04-30 |
| U2_risk_sane | m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_fundamental_ev_ebitda_z | feature_fundamental_ev_ebitda | 307350 | 187495 | 0 | 0 | 0 |  |  |
| U2_risk_sane | m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_fundamental_roe_ttm_z | feature_fundamental_roe_ttm | 307350 | 187495 | 304617 | 0.991108 | 0.993194 | 2021-01-29 | 2026-04-30 |
| U2_risk_sane | m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_fundamental_net_profit_yoy_z | feature_fundamental_net_profit_yoy | 307350 | 187495 | 306367 | 0.996802 | 0.997019 | 2021-01-29 | 2026-04-30 |
| U2_risk_sane | m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_fundamental_gross_margin_change_z | feature_fundamental_gross_margin_change | 307350 | 187495 | 299889 | 0.975725 | 0.96682 | 2021-01-29 | 2026-04-30 |
| U2_risk_sane | m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_fundamental_gross_margin_delta_z | feature_fundamental_gross_margin_delta | 307350 | 187495 | 299765 | 0.975321 | 0.966564 | 2021-01-29 | 2026-04-30 |
| U2_risk_sane | m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_fundamental_debt_to_assets_change_z | feature_fundamental_debt_to_assets_change | 307350 | 187495 | 306206 | 0.996278 | 0.997019 | 2021-01-29 | 2026-04-30 |
| U2_risk_sane | m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_fundamental_ocf_to_net_profit_z | feature_fundamental_ocf_to_net_profit | 307350 | 187495 | 245847 | 0.799893 | 0.817622 | 2021-01-29 | 2026-04-30 |
| U2_risk_sane | m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_fundamental_ocf_to_asset_z | feature_fundamental_ocf_to_asset | 307350 | 187495 | 306219 | 0.99632 | 0.99656 | 2021-01-29 | 2026-04-30 |
| U2_risk_sane | m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_fundamental_asset_turnover_z | feature_fundamental_asset_turnover | 307350 | 187495 | 306300 | 0.996584 | 0.996891 | 2021-01-29 | 2026-04-30 |
| U2_risk_sane | m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_fundamental_net_margin_stability_z | feature_fundamental_net_margin_stability | 307350 | 187495 | 306150 | 0.996096 | 0.996645 | 2021-01-29 | 2026-04-30 |

## M9 Integrity

| check | value | pass | detail |
| --- | --- | --- | --- |
| target_candidate_pool_pass_rows | 4196 | True | latest selected signal date must have buyable candidates |
| target_next_trade_date_present | 4196 | True | all candidate_pool_pass rows should carry next_trade_date |
| recommendation_buyable | 0 | True | recommendation rows should be buyable at t+1 open |
| recommendation_names_readable | 0 | True | name should not be UNKNOWN or blank |
| recommendation_excludes_st_names | 0 | True | name-aware report filter should exclude ST/*ST targets |
| zero_coverage_core_features | 0 | True | zero coverage fields must not be core model features |
| low_coverage_core_features_lt_30pct | 0 | True | low coverage fields should be missing-marker-only or ablation-only |
| max_single_industry_share_le_40pct | 0.3 | True | 单行业占比 30.0%，超 40% 视为集中度过高 |
| distinct_industry_count_ge_3 | 6 | True | Top-K 推荐覆盖 6 个行业，至少应覆盖 3 个 |

## M9 Feature Policy

| feature | raw_feature | candidate_pool_pass_coverage_ratio | m9_feature_policy | active_feature |
| --- | --- | --- | --- | --- |
| feature_ret_20d_z | feature_ret_20d | 1 | core_feature | feature_ret_20d_z |
| feature_ret_60d_z | feature_ret_60d | 1 | core_feature | feature_ret_60d_z |
| feature_ret_5d_z | feature_ret_5d | 1 | core_feature | feature_ret_5d_z |
| feature_realized_vol_20d_z | feature_realized_vol_20d | 1 | core_feature | feature_realized_vol_20d_z |
| feature_amount_20d_log_z | feature_amount_20d_log | 1 | core_feature | feature_amount_20d_log_z |
| feature_turnover_20d_z | feature_turnover_20d | 1 | core_feature | feature_turnover_20d_z |
| feature_price_position_250d_z | feature_price_position_250d | 1 | core_feature | feature_price_position_250d_z |
| feature_limit_move_hits_20d_z | feature_limit_move_hits_20d | 1 | core_feature | feature_limit_move_hits_20d_z |
| feature_industry_ret20_mean_z | feature_industry_ret20_mean | 1 | core_feature | feature_industry_ret20_mean_z |
| feature_industry_ret60_mean_z | feature_industry_ret60_mean | 1 | core_feature | feature_industry_ret60_mean_z |
| feature_industry_positive_ret20_ratio_z | feature_industry_positive_ret20_ratio | 1 | core_feature | feature_industry_positive_ret20_ratio_z |
| feature_industry_amount20_mean_z | feature_industry_amount20_mean | 1 | core_feature | feature_industry_amount20_mean_z |
| feature_industry_low_vol20_mean_z | feature_industry_low_vol20_mean | 1 | core_feature | feature_industry_low_vol20_mean_z |
| feature_fund_flow_main_inflow_5d_z | feature_fund_flow_main_inflow_5d | 0.126531 | missing_flag_only_low_coverage | is_missing_feature_fund_flow_main_inflow_5d |
| feature_fund_flow_main_inflow_10d_z | feature_fund_flow_main_inflow_10d | 0.108115 | missing_flag_only_low_coverage | is_missing_feature_fund_flow_main_inflow_10d |
| feature_fund_flow_main_inflow_20d_z | feature_fund_flow_main_inflow_20d | 0.107635 | missing_flag_only_low_coverage | is_missing_feature_fund_flow_main_inflow_20d |
| feature_fund_flow_super_inflow_10d_z | feature_fund_flow_super_inflow_10d | 0.108115 | missing_flag_only_low_coverage | is_missing_feature_fund_flow_super_inflow_10d |
| feature_fund_flow_divergence_20d_z | feature_fund_flow_divergence_20d | 0.107635 | missing_flag_only_low_coverage | is_missing_feature_fund_flow_divergence_20d |
| feature_fund_flow_main_inflow_streak_z | feature_fund_flow_main_inflow_streak | 0.128905 | missing_flag_only_low_coverage | is_missing_feature_fund_flow_main_inflow_streak |
| feature_fundamental_pe_ttm_z | feature_fundamental_pe_ttm | 0.999803 | core_feature | feature_fundamental_pe_ttm_z |
| feature_fundamental_pb_z | feature_fundamental_pb | 0.999803 | core_feature | feature_fundamental_pb_z |
| feature_fundamental_ev_ebitda_z | feature_fundamental_ev_ebitda | 0 | missing_flag_only_low_coverage | is_missing_feature_fundamental_ev_ebitda |
| feature_fundamental_roe_ttm_z | feature_fundamental_roe_ttm | 0.993194 | core_feature | feature_fundamental_roe_ttm_z |
| feature_fundamental_net_profit_yoy_z | feature_fundamental_net_profit_yoy | 0.997019 | core_feature | feature_fundamental_net_profit_yoy_z |
| feature_fundamental_gross_margin_change_z | feature_fundamental_gross_margin_change | 0.96682 | core_feature | feature_fundamental_gross_margin_change_z |
| feature_fundamental_gross_margin_delta_z | feature_fundamental_gross_margin_delta | 0.966564 | core_feature | feature_fundamental_gross_margin_delta_z |
| feature_fundamental_debt_to_assets_change_z | feature_fundamental_debt_to_assets_change | 0.997019 | core_feature | feature_fundamental_debt_to_assets_change_z |
| feature_fundamental_ocf_to_net_profit_z | feature_fundamental_ocf_to_net_profit | 0.817622 | core_feature | feature_fundamental_ocf_to_net_profit_z |
| feature_fundamental_ocf_to_asset_z | feature_fundamental_ocf_to_asset | 0.99656 | core_feature | feature_fundamental_ocf_to_asset_z |
| feature_fundamental_asset_turnover_z | feature_fundamental_asset_turnover | 0.996891 | core_feature | feature_fundamental_asset_turnover_z |
| feature_fundamental_net_margin_stability_z | feature_fundamental_net_margin_stability | 0.996645 | core_feature | feature_fundamental_net_margin_stability_z |

## 口径与结论

- 本轮新增的是 M7 研究版推荐报告，不新增生产策略；推荐名单不自动生成交易指令。
- 数据质量沿用 M2/M5/M6 的 PIT 检查，本脚本只使用 `report_signal_date` 当日可观测特征，并只用该日期之前已完成标签的月份训练。
- M9 数据完整性 gate 要求报告信号日存在 `next_trade_date`、推荐行全部 `buyable_tplus1_open`、名称非 UNKNOWN，且零覆盖/低覆盖字段不作为核心模型特征。
- 月度交易时点按“本持有月最后一个交易日卖出、下一持有月首个交易日买入”展示；`sell_trade_date` 为下一次月末信号日。
- 候选池使用 `U2_risk_sane` watchlist 口径，只做准入过滤；alpha 判断来自 `M6_xgboost_rank_ndcg` 排序分数。
- 历史 baseline 改善证据来自 M6 walk-forward 附件；M6 结论仍是 watchlist，不进入生产。
- 当前推荐月尚无未来收益标签，因此本报告不声称本月已跑赢市场；稳定性判断以历史 leaderboard / monthly_long / rank_ic / quantile_spread 为准。
- 分桶、年份和 regime 的历史失败月份保留在附件中；后续进入 regime-aware calibration 复核。

## 本轮产物

- `data/results/monthly_selection_m7_recommendation_report_m8_natural_2026-05-04_summary.json`
- `data/results/monthly_selection_m7_recommendation_report_m8_natural_2026-05-04_recommendations.csv`
- `data/results/monthly_selection_m7_recommendation_report_m8_natural_2026-05-04_leaderboard.csv`
- `data/results/monthly_selection_m7_recommendation_report_m8_natural_2026-05-04_monthly_long.csv`
- `data/results/monthly_selection_m7_recommendation_report_m8_natural_2026-05-04_rank_ic.csv`
- `data/results/monthly_selection_m7_recommendation_report_m8_natural_2026-05-04_quantile_spread.csv`
- `data/results/monthly_selection_m7_recommendation_report_m8_natural_2026-05-04_topk_holdings.csv`
- `data/results/monthly_selection_m7_recommendation_report_m8_natural_2026-05-04_industry_exposure.csv`
- `data/results/monthly_selection_m7_recommendation_report_m8_natural_2026-05-04_candidate_pool_width.csv`
- `data/results/monthly_selection_m7_recommendation_report_m8_natural_2026-05-04_candidate_pool_reject_reason.csv`
- `data/results/monthly_selection_m7_recommendation_report_m8_natural_2026-05-04_feature_importance.csv`
- `data/results/monthly_selection_m7_recommendation_report_m8_natural_2026-05-04_feature_contrib.csv`
- `data/results/monthly_selection_m7_recommendation_report_m8_natural_2026-05-04_feature_coverage.csv`
- `data/results/monthly_selection_m7_recommendation_report_m8_natural_2026-05-04_feature_policy.csv`
- `data/results/monthly_selection_m7_recommendation_report_m8_natural_2026-05-04_m9_integrity.csv`
- `data/results/monthly_selection_m7_recommendation_report_m8_natural_2026-05-04_risk_summary.csv`
- `data/results/monthly_selection_m7_recommendation_report_m8_natural_2026-05-04_year_slice.csv`
- `data/results/monthly_selection_m7_recommendation_report_m8_natural_2026-05-04_regime_slice.csv`
- `data/results/monthly_selection_m7_recommendation_report_m8_natural_2026-05-04_manifest.json`
