# Monthly Selection M7 Recommendation Report

- 生成时间：`2026-04-30T07:38:12.404237+00:00`
- 结果类型：`monthly_selection_m7_recommendation_report`
- 研究配置：`dataset_monthly_selection_features_families_price_volume_only-industry_breadth-fund_flow-fundamental_pools_u2_risk_sane_topk_20-30_model_m6_xgboost_rank_ndcg_maxfit_50000_wf_24m`
- 报告信号日：`2026-03-31`
- 下一交易日：`2026-04-01`
- 候选池：`U2_risk_sane`
- 模型：`M6_xgboost_rank_ndcg`
- 生产状态：`research_only_not_promoted`

## 推荐名单

| signal_date | top_k | rank | symbol | name | score | score_percentile | industry | industry_level2 | feature_contrib | risk_flags | last_month_rank | last_month_selected | buyability | next_trade_date | candidate_pool_version | candidate_pool_rule | model | model_type | feature_spec |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-03-31 | 20 | 1 | 002039 | 黔源电力 | 1 | 1 | 公用事业 | 电力 | feature_fundamental_gross_margin_change_z=+4.2; feature_industry_positive_ret20_ratio_z=+2.33; feature_industry_ret60_mean_z=+2.8 |  |  | False | buyable_tplus1_open | 2026-04-01 | U2_risk_sane | U1 + exclude extreme limit-move path, extreme volatility/turnover, and absolute-high names | M6_xgboost_rank_ndcg | xgboost_ranker | m6_core_price_volume_industry_breadth_fund_flow_fundamental |
| 2026-03-31 | 20 | 2 | 301068 | 大地海洋 | 0.999754 | 0.999754 | 环保 | 环境治理 | feature_fundamental_gross_margin_change_z=+4.2; feature_fundamental_gross_margin_delta_z=-2.5; feature_fundamental_debt_to_assets_change_z=+3.15 |  |  | False | buyable_tplus1_open | 2026-04-01 | U2_risk_sane | U1 + exclude extreme limit-move path, extreme volatility/turnover, and absolute-high names | M6_xgboost_rank_ndcg | xgboost_ranker | m6_core_price_volume_industry_breadth_fund_flow_fundamental |
| 2026-03-31 | 20 | 3 | 000809 | 和展能源 | 0.999508 | 0.999508 | 房地产 | 房地产开发 | feature_fundamental_gross_margin_change_z=+4.2; feature_fundamental_net_margin_stability_z=-5; feature_fundamental_ocf_to_asset_z=-4.82 |  |  | False | buyable_tplus1_open | 2026-04-01 | U2_risk_sane | U1 + exclude extreme limit-move path, extreme volatility/turnover, and absolute-high names | M6_xgboost_rank_ndcg | xgboost_ranker | m6_core_price_volume_industry_breadth_fund_flow_fundamental |
| 2026-03-31 | 20 | 4 | 600593 | 大连圣亚 | 0.999262 | 0.999262 | 社会服务 | 旅游及景区 | feature_fundamental_gross_margin_change_z=+4.2; feature_fundamental_pb_z=+4.91; feature_fundamental_roe_ttm_z=+2.34 |  |  | False | buyable_tplus1_open | 2026-04-01 | U2_risk_sane | U1 + exclude extreme limit-move path, extreme volatility/turnover, and absolute-high names | M6_xgboost_rank_ndcg | xgboost_ranker | m6_core_price_volume_industry_breadth_fund_flow_fundamental |
| 2026-03-31 | 20 | 5 | 002778 | 中晟高科 | 0.999016 | 0.999016 | 环保 | 环境治理 | feature_fundamental_net_margin_stability_z=-5; feature_fundamental_ocf_to_asset_z=+4.66; feature_fundamental_gross_margin_change_z=+1.9 |  |  | False | buyable_tplus1_open | 2026-04-01 | U2_risk_sane | U1 + exclude extreme limit-move path, extreme volatility/turnover, and absolute-high names | M6_xgboost_rank_ndcg | xgboost_ranker | m6_core_price_volume_industry_breadth_fund_flow_fundamental |
| 2026-03-31 | 20 | 6 | 300511 | 雪榕生物 | 0.998771 | 0.998771 | 农林牧渔 | 种植业 | feature_fundamental_gross_margin_change_z=+3.08; feature_turnover_20d_z=+2.35; feature_fundamental_debt_to_assets_change_z=-1.77 |  |  | False | buyable_tplus1_open | 2026-04-01 | U2_risk_sane | U1 + exclude extreme limit-move path, extreme volatility/turnover, and absolute-high names | M6_xgboost_rank_ndcg | xgboost_ranker | m6_core_price_volume_industry_breadth_fund_flow_fundamental |
| 2026-03-31 | 20 | 7 | 002379 | 宏桥控股 | 0.998525 | 0.998525 | 有色金属 | 工业金属 | feature_fundamental_gross_margin_change_z=+4.2; feature_fundamental_pe_ttm_z=-4.96; feature_fundamental_debt_to_assets_change_z=+3.15 |  |  | False | buyable_tplus1_open | 2026-04-01 | U2_risk_sane | U1 + exclude extreme limit-move path, extreme volatility/turnover, and absolute-high names | M6_xgboost_rank_ndcg | xgboost_ranker | m6_core_price_volume_industry_breadth_fund_flow_fundamental |
| 2026-03-31 | 20 | 8 | 000762 | 西藏矿业 | 0.998279 | 0.998279 | 有色金属 | 能源金属 | feature_fundamental_gross_margin_change_z=+3.22; feature_fundamental_ocf_to_asset_z=+5; feature_fundamental_gross_margin_delta_z=-3.25 |  |  | False | buyable_tplus1_open | 2026-04-01 | U2_risk_sane | U1 + exclude extreme limit-move path, extreme volatility/turnover, and absolute-high names | M6_xgboost_rank_ndcg | xgboost_ranker | m6_core_price_volume_industry_breadth_fund_flow_fundamental |
| 2026-03-31 | 20 | 9 | 688372 | 伟测科技 | 0.998033 | 0.998033 | 电子 | 半导体 | feature_fundamental_gross_margin_change_z=+2.1; feature_amount_20d_log_z=+1.52; feature_ret_60d_z=+0.972 |  |  | False | buyable_tplus1_open | 2026-04-01 | U2_risk_sane | U1 + exclude extreme limit-move path, extreme volatility/turnover, and absolute-high names | M6_xgboost_rank_ndcg | xgboost_ranker | m6_core_price_volume_industry_breadth_fund_flow_fundamental |
| 2026-03-31 | 20 | 10 | 300887 | 谱尼测试 | 0.997787 | 0.997787 | 社会服务 | 专业服务 | feature_fundamental_gross_margin_change_z=+2.62; feature_fundamental_roe_ttm_z=-0.88; feature_realized_vol_20d_z=+1.09 |  |  | False | buyable_tplus1_open | 2026-04-01 | U2_risk_sane | U1 + exclude extreme limit-move path, extreme volatility/turnover, and absolute-high names | M6_xgboost_rank_ndcg | xgboost_ranker | m6_core_price_volume_industry_breadth_fund_flow_fundamental |
| 2026-03-31 | 20 | 11 | 000888 | 峨眉山Ａ | 0.997541 | 0.997541 | 社会服务 | 旅游及景区 | feature_fundamental_gross_margin_change_z=+2.46; feature_price_position_250d_z=-1.12; feature_fundamental_debt_to_assets_change_z=-1.01 |  |  | False | buyable_tplus1_open | 2026-04-01 | U2_risk_sane | U1 + exclude extreme limit-move path, extreme volatility/turnover, and absolute-high names | M6_xgboost_rank_ndcg | xgboost_ranker | m6_core_price_volume_industry_breadth_fund_flow_fundamental |
| 2026-03-31 | 20 | 12 | 300435 | 中泰股份 | 0.997295 | 0.997295 | 公用事业 | 燃气Ⅱ | feature_fundamental_gross_margin_change_z=+3.5; feature_ret_20d_z=-2.05; feature_industry_positive_ret20_ratio_z=+2.33 |  |  | False | buyable_tplus1_open | 2026-04-01 | U2_risk_sane | U1 + exclude extreme limit-move path, extreme volatility/turnover, and absolute-high names | M6_xgboost_rank_ndcg | xgboost_ranker | m6_core_price_volume_industry_breadth_fund_flow_fundamental |
| 2026-03-31 | 20 | 13 | 688280 | 精进电动 | 0.997049 | 0.997049 | 汽车 | 汽车零部件 | feature_fundamental_gross_margin_change_z=+1.87; feature_fundamental_gross_margin_delta_z=+3.1; feature_fundamental_roe_ttm_z=+0.997 |  |  | False | buyable_tplus1_open | 2026-04-01 | U2_risk_sane | U1 + exclude extreme limit-move path, extreme volatility/turnover, and absolute-high names | M6_xgboost_rank_ndcg | xgboost_ranker | m6_core_price_volume_industry_breadth_fund_flow_fundamental |
| 2026-03-31 | 20 | 14 | 002414 | 高德红外 | 0.996804 | 0.996804 | 国防军工 | 军工电子Ⅱ | feature_fundamental_gross_margin_change_z=+3.43; feature_fundamental_net_profit_yoy_z=+4.3; feature_industry_ret20_mean_z=-1.9 |  |  | False | buyable_tplus1_open | 2026-04-01 | U2_risk_sane | U1 + exclude extreme limit-move path, extreme volatility/turnover, and absolute-high names | M6_xgboost_rank_ndcg | xgboost_ranker | m6_core_price_volume_industry_breadth_fund_flow_fundamental |
| 2026-03-31 | 20 | 15 | 000600 | 建投能源 | 0.996558 | 0.996558 | 公用事业 | 电力 | feature_fundamental_gross_margin_change_z=+1.98; feature_fundamental_gross_margin_delta_z=+2.44; feature_industry_positive_ret20_ratio_z=+2.33 |  |  | False | buyable_tplus1_open | 2026-04-01 | U2_risk_sane | U1 + exclude extreme limit-move path, extreme volatility/turnover, and absolute-high names | M6_xgboost_rank_ndcg | xgboost_ranker | m6_core_price_volume_industry_breadth_fund_flow_fundamental |
| 2026-03-31 | 20 | 16 | 688171 | 纬德信息 | 0.996312 | 0.996312 | 计算机 | 软件开发 | feature_fundamental_gross_margin_change_z=+4.03; feature_fundamental_gross_margin_delta_z=+2.12; feature_fundamental_pe_ttm_z=+1.79 |  |  | False | buyable_tplus1_open | 2026-04-01 | U2_risk_sane | U1 + exclude extreme limit-move path, extreme volatility/turnover, and absolute-high names | M6_xgboost_rank_ndcg | xgboost_ranker | m6_core_price_volume_industry_breadth_fund_flow_fundamental |
| 2026-03-31 | 20 | 17 | 600749 | 西藏旅游 | 0.996066 | 0.996066 | 社会服务 | 旅游及景区 | feature_fundamental_gross_margin_change_z=+3.49; feature_fundamental_net_margin_stability_z=-1.34; feature_fundamental_ocf_to_asset_z=+1.02 |  |  | False | buyable_tplus1_open | 2026-04-01 | U2_risk_sane | U1 + exclude extreme limit-move path, extreme volatility/turnover, and absolute-high names | M6_xgboost_rank_ndcg | xgboost_ranker | m6_core_price_volume_industry_breadth_fund_flow_fundamental |
| 2026-03-31 | 20 | 18 | 002240 | 盛新锂能 | 0.99582 | 0.99582 | 有色金属 | 能源金属 | feature_fundamental_gross_margin_change_z=+4.07; feature_fundamental_gross_margin_delta_z=+3.15; feature_industry_ret20_mean_z=-2.24 |  |  | False | buyable_tplus1_open | 2026-04-01 | U2_risk_sane | U1 + exclude extreme limit-move path, extreme volatility/turnover, and absolute-high names | M6_xgboost_rank_ndcg | xgboost_ranker | m6_core_price_volume_industry_breadth_fund_flow_fundamental |
| 2026-03-31 | 20 | 19 | 601021 | 春秋航空 | 0.995574 | 0.995574 | 交通运输 | 航空机场 | feature_fundamental_gross_margin_change_z=+1.92; feature_ret_60d_z=-0.94; feature_industry_low_vol20_mean_z=+1.54 |  |  | False | buyable_tplus1_open | 2026-04-01 | U2_risk_sane | U1 + exclude extreme limit-move path, extreme volatility/turnover, and absolute-high names | M6_xgboost_rank_ndcg | xgboost_ranker | m6_core_price_volume_industry_breadth_fund_flow_fundamental |
| 2026-03-31 | 20 | 20 | 300377 | 赢时胜 | 0.995328 | 0.995328 | 计算机 | 软件开发 | feature_fundamental_gross_margin_change_z=+4.17; feature_turnover_20d_z=+3.04; feature_ret_60d_z=-1.29 |  |  | False | buyable_tplus1_open | 2026-04-01 | U2_risk_sane | U1 + exclude extreme limit-move path, extreme volatility/turnover, and absolute-high names | M6_xgboost_rank_ndcg | xgboost_ranker | m6_core_price_volume_industry_breadth_fund_flow_fundamental |

## M6 Historical Evidence

| candidate_pool_version | model | model_type | top_k | months | mean_topk_return | median_topk_return | topk_excess_mean | topk_excess_median | topk_hit_rate | topk_minus_nextk_mean | industry_neutral_topk_excess_mean | turnover_mean | cost_drag_mean | topk_excess_after_cost_mean | topk_excess_annualized | rank_ic_mean | rank_ic_std | rank_ic_months | rank_ic_ir | quantile_top_minus_bottom_mean | neutral_median_excess | strong_down_median_excess | strong_up_median_excess |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| U2_risk_sane | M6_xgboost_rank_ndcg | xgboost_ranker | 20 | 38 | 0.0247633 | 0.028524 | 0.0126144 | 0.0147394 | 0.657895 | 0.00762405 | 0.00384424 | 0.940541 | 0.000940541 | 0.0121064 | 0.16233 | 0.00666237 | 0.0777473 | 38 | 0.0856927 | 0.000673815 | 0.0131781 | -0.00184697 | 0.0181031 |
| U2_risk_sane | M6_xgboost_rank_ndcg | xgboost_ranker | 30 | 38 | 0.0214441 | 0.0241337 | 0.0092952 | 0.00160447 | 0.552632 | 0.010155 | 0.00158023 | 0.931532 | 0.000931532 | 0.00862985 | 0.117425 | 0.00666237 | 0.0777473 | 38 | 0.0856927 | 0.000673815 | 0.00361303 | -0.00236666 | 0.000249559 |

## Risk Summary

| signal_date | candidate_pool_version | model | top_k | selected_count | risk_flagged_count | not_buyable_count | last_month_selected_count |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-03-31 | U2_risk_sane | M6_xgboost_rank_ndcg | 20 | 20 | 0 | 0 | 0 |
| 2026-03-31 | U2_risk_sane | M6_xgboost_rank_ndcg | 30 | 30 | 0 | 0 | 0 |

## Industry Exposure

| signal_date | candidate_pool_version | model | top_k | industry | industry_count | topk_count | industry_share |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-03-31 | U2_risk_sane | M6_xgboost_rank_ndcg | 20 | 交通运输 | 1 | 20 | 0.05 |
| 2026-03-31 | U2_risk_sane | M6_xgboost_rank_ndcg | 20 | 公用事业 | 3 | 20 | 0.15 |
| 2026-03-31 | U2_risk_sane | M6_xgboost_rank_ndcg | 20 | 农林牧渔 | 1 | 20 | 0.05 |
| 2026-03-31 | U2_risk_sane | M6_xgboost_rank_ndcg | 20 | 国防军工 | 1 | 20 | 0.05 |
| 2026-03-31 | U2_risk_sane | M6_xgboost_rank_ndcg | 20 | 房地产 | 1 | 20 | 0.05 |
| 2026-03-31 | U2_risk_sane | M6_xgboost_rank_ndcg | 20 | 有色金属 | 3 | 20 | 0.15 |
| 2026-03-31 | U2_risk_sane | M6_xgboost_rank_ndcg | 20 | 汽车 | 1 | 20 | 0.05 |
| 2026-03-31 | U2_risk_sane | M6_xgboost_rank_ndcg | 20 | 环保 | 2 | 20 | 0.1 |
| 2026-03-31 | U2_risk_sane | M6_xgboost_rank_ndcg | 20 | 电子 | 1 | 20 | 0.05 |
| 2026-03-31 | U2_risk_sane | M6_xgboost_rank_ndcg | 20 | 社会服务 | 4 | 20 | 0.2 |
| 2026-03-31 | U2_risk_sane | M6_xgboost_rank_ndcg | 20 | 计算机 | 2 | 20 | 0.1 |
| 2026-03-31 | U2_risk_sane | M6_xgboost_rank_ndcg | 30 | 交通运输 | 2 | 30 | 0.0666667 |
| 2026-03-31 | U2_risk_sane | M6_xgboost_rank_ndcg | 30 | 公用事业 | 4 | 30 | 0.133333 |
| 2026-03-31 | U2_risk_sane | M6_xgboost_rank_ndcg | 30 | 农林牧渔 | 1 | 30 | 0.0333333 |
| 2026-03-31 | U2_risk_sane | M6_xgboost_rank_ndcg | 30 | 商贸零售 | 1 | 30 | 0.0333333 |
| 2026-03-31 | U2_risk_sane | M6_xgboost_rank_ndcg | 30 | 国防军工 | 1 | 30 | 0.0333333 |
| 2026-03-31 | U2_risk_sane | M6_xgboost_rank_ndcg | 30 | 房地产 | 2 | 30 | 0.0666667 |
| 2026-03-31 | U2_risk_sane | M6_xgboost_rank_ndcg | 30 | 有色金属 | 3 | 30 | 0.1 |
| 2026-03-31 | U2_risk_sane | M6_xgboost_rank_ndcg | 30 | 汽车 | 1 | 30 | 0.0333333 |
| 2026-03-31 | U2_risk_sane | M6_xgboost_rank_ndcg | 30 | 环保 | 3 | 30 | 0.1 |
| 2026-03-31 | U2_risk_sane | M6_xgboost_rank_ndcg | 30 | 电子 | 2 | 30 | 0.0666667 |
| 2026-03-31 | U2_risk_sane | M6_xgboost_rank_ndcg | 30 | 社会服务 | 4 | 30 | 0.133333 |
| 2026-03-31 | U2_risk_sane | M6_xgboost_rank_ndcg | 30 | 计算机 | 5 | 30 | 0.166667 |
| 2026-03-31 | U2_risk_sane | M6_xgboost_rank_ndcg | 30 | 通信 | 1 | 30 | 0.0333333 |

## Feature Coverage

| candidate_pool_version | feature_spec | families | feature | raw_feature | rows | candidate_pool_pass_rows | non_null | coverage_ratio | candidate_pool_pass_coverage_ratio | first_signal_date | last_signal_date |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| U2_risk_sane | m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_ret_20d_z | feature_ret_20d | 307350 | 187257 | 306082 | 0.995874 | 1 | 2021-01-29 | 2026-04-29 |
| U2_risk_sane | m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_ret_60d_z | feature_ret_60d | 307350 | 187257 | 303506 | 0.987493 | 1 | 2021-01-29 | 2026-04-29 |
| U2_risk_sane | m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_ret_5d_z | feature_ret_5d | 307350 | 187257 | 307017 | 0.998917 | 1 | 2021-01-29 | 2026-04-29 |
| U2_risk_sane | m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_realized_vol_20d_z | feature_realized_vol_20d | 307350 | 187257 | 306673 | 0.997797 | 1 | 2021-01-29 | 2026-04-29 |
| U2_risk_sane | m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_amount_20d_log_z | feature_amount_20d_log | 307350 | 187257 | 306745 | 0.998032 | 1 | 2021-01-29 | 2026-04-29 |
| U2_risk_sane | m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_turnover_20d_z | feature_turnover_20d | 307350 | 187257 | 306745 | 0.998032 | 1 | 2021-01-29 | 2026-04-29 |
| U2_risk_sane | m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_price_position_250d_z | feature_price_position_250d | 307350 | 187257 | 303572 | 0.987708 | 1 | 2021-01-29 | 2026-04-29 |
| U2_risk_sane | m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_limit_move_hits_20d_z | feature_limit_move_hits_20d | 307350 | 187257 | 307350 | 1 | 1 | 2021-01-29 | 2026-04-29 |
| U2_risk_sane | m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_industry_ret20_mean_z | feature_industry_ret20_mean | 307350 | 187257 | 307350 | 1 | 1 | 2021-01-29 | 2026-04-29 |
| U2_risk_sane | m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_industry_ret60_mean_z | feature_industry_ret60_mean | 307350 | 187257 | 307350 | 1 | 1 | 2021-01-29 | 2026-04-29 |
| U2_risk_sane | m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_industry_positive_ret20_ratio_z | feature_industry_positive_ret20_ratio | 307350 | 187257 | 307350 | 1 | 1 | 2021-01-29 | 2026-04-29 |
| U2_risk_sane | m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_industry_amount20_mean_z | feature_industry_amount20_mean | 307350 | 187257 | 307350 | 1 | 1 | 2021-01-29 | 2026-04-29 |
| U2_risk_sane | m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_industry_low_vol20_mean_z | feature_industry_low_vol20_mean | 307350 | 187257 | 307350 | 1 | 1 | 2021-01-29 | 2026-04-29 |
| U2_risk_sane | m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_fund_flow_main_inflow_5d_z | feature_fund_flow_main_inflow_5d | 307350 | 187257 | 35590 | 0.115796 | 0.126692 | 2023-10-31 | 2026-04-29 |
| U2_risk_sane | m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_fund_flow_main_inflow_10d_z | feature_fund_flow_main_inflow_10d | 307350 | 187257 | 31196 | 0.1015 | 0.108252 | 2023-11-30 | 2026-04-29 |
| U2_risk_sane | m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_fund_flow_main_inflow_20d_z | feature_fund_flow_main_inflow_20d | 307350 | 187257 | 31048 | 0.101018 | 0.107772 | 2023-11-30 | 2026-04-29 |
| U2_risk_sane | m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_fund_flow_super_inflow_10d_z | feature_fund_flow_super_inflow_10d | 307350 | 187257 | 31196 | 0.1015 | 0.108252 | 2023-11-30 | 2026-04-29 |
| U2_risk_sane | m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_fund_flow_divergence_20d_z | feature_fund_flow_divergence_20d | 307350 | 187257 | 31048 | 0.101018 | 0.107772 | 2023-11-30 | 2026-04-29 |
| U2_risk_sane | m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_fund_flow_main_inflow_streak_z | feature_fund_flow_main_inflow_streak | 307350 | 187257 | 36162 | 0.117657 | 0.129069 | 2023-10-31 | 2026-04-29 |
| U2_risk_sane | m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_fundamental_pe_ttm_z | feature_fundamental_pe_ttm | 307350 | 187257 | 304197 | 0.989741 | 0.998713 | 2021-01-29 | 2026-04-29 |
| U2_risk_sane | m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_fundamental_pb_z | feature_fundamental_pb | 307350 | 187257 | 304197 | 0.989741 | 0.998713 | 2021-01-29 | 2026-04-29 |
| U2_risk_sane | m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_fundamental_ev_ebitda_z | feature_fundamental_ev_ebitda | 307350 | 187257 | 0 | 0 | 0 |  |  |
| U2_risk_sane | m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_fundamental_roe_ttm_z | feature_fundamental_roe_ttm | 307350 | 187257 | 304622 | 0.991124 | 0.993191 | 2021-01-29 | 2026-04-29 |
| U2_risk_sane | m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_fundamental_net_profit_yoy_z | feature_fundamental_net_profit_yoy | 307350 | 187257 | 306093 | 0.99591 | 0.995925 | 2021-01-29 | 2026-04-29 |
| U2_risk_sane | m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_fundamental_gross_margin_change_z | feature_fundamental_gross_margin_change | 307350 | 187257 | 299649 | 0.974944 | 0.965913 | 2021-01-29 | 2026-04-29 |
| U2_risk_sane | m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_fundamental_gross_margin_delta_z | feature_fundamental_gross_margin_delta | 307350 | 187257 | 299525 | 0.97454 | 0.965662 | 2021-01-29 | 2026-04-29 |
| U2_risk_sane | m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_fundamental_debt_to_assets_change_z | feature_fundamental_debt_to_assets_change | 307350 | 187257 | 306204 | 0.996271 | 0.997015 | 2021-01-29 | 2026-04-29 |
| U2_risk_sane | m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_fundamental_ocf_to_net_profit_z | feature_fundamental_ocf_to_net_profit | 307350 | 187257 | 245641 | 0.799222 | 0.816765 | 2021-01-29 | 2026-04-29 |
| U2_risk_sane | m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_fundamental_ocf_to_asset_z | feature_fundamental_ocf_to_asset | 307350 | 187257 | 305945 | 0.995429 | 0.995466 | 2021-01-29 | 2026-04-29 |
| U2_risk_sane | m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_fundamental_asset_turnover_z | feature_fundamental_asset_turnover | 307350 | 187257 | 306026 | 0.995692 | 0.995797 | 2021-01-29 | 2026-04-29 |
| U2_risk_sane | m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_fundamental_net_margin_stability_z | feature_fundamental_net_margin_stability | 307350 | 187257 | 305876 | 0.995204 | 0.995557 | 2021-01-29 | 2026-04-29 |

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
| feature_fund_flow_main_inflow_5d_z | feature_fund_flow_main_inflow_5d | 0.126692 | missing_flag_only_low_coverage | is_missing_feature_fund_flow_main_inflow_5d |
| feature_fund_flow_main_inflow_10d_z | feature_fund_flow_main_inflow_10d | 0.108252 | missing_flag_only_low_coverage | is_missing_feature_fund_flow_main_inflow_10d |
| feature_fund_flow_main_inflow_20d_z | feature_fund_flow_main_inflow_20d | 0.107772 | missing_flag_only_low_coverage | is_missing_feature_fund_flow_main_inflow_20d |
| feature_fund_flow_super_inflow_10d_z | feature_fund_flow_super_inflow_10d | 0.108252 | missing_flag_only_low_coverage | is_missing_feature_fund_flow_super_inflow_10d |
| feature_fund_flow_divergence_20d_z | feature_fund_flow_divergence_20d | 0.107772 | missing_flag_only_low_coverage | is_missing_feature_fund_flow_divergence_20d |
| feature_fund_flow_main_inflow_streak_z | feature_fund_flow_main_inflow_streak | 0.129069 | missing_flag_only_low_coverage | is_missing_feature_fund_flow_main_inflow_streak |
| feature_fundamental_pe_ttm_z | feature_fundamental_pe_ttm | 0.998713 | core_feature | feature_fundamental_pe_ttm_z |
| feature_fundamental_pb_z | feature_fundamental_pb | 0.998713 | core_feature | feature_fundamental_pb_z |
| feature_fundamental_ev_ebitda_z | feature_fundamental_ev_ebitda | 0 | missing_flag_only_low_coverage | is_missing_feature_fundamental_ev_ebitda |
| feature_fundamental_roe_ttm_z | feature_fundamental_roe_ttm | 0.993191 | core_feature | feature_fundamental_roe_ttm_z |
| feature_fundamental_net_profit_yoy_z | feature_fundamental_net_profit_yoy | 0.995925 | core_feature | feature_fundamental_net_profit_yoy_z |
| feature_fundamental_gross_margin_change_z | feature_fundamental_gross_margin_change | 0.965913 | core_feature | feature_fundamental_gross_margin_change_z |
| feature_fundamental_gross_margin_delta_z | feature_fundamental_gross_margin_delta | 0.965662 | core_feature | feature_fundamental_gross_margin_delta_z |
| feature_fundamental_debt_to_assets_change_z | feature_fundamental_debt_to_assets_change | 0.997015 | core_feature | feature_fundamental_debt_to_assets_change_z |
| feature_fundamental_ocf_to_net_profit_z | feature_fundamental_ocf_to_net_profit | 0.816765 | core_feature | feature_fundamental_ocf_to_net_profit_z |
| feature_fundamental_ocf_to_asset_z | feature_fundamental_ocf_to_asset | 0.995466 | core_feature | feature_fundamental_ocf_to_asset_z |
| feature_fundamental_asset_turnover_z | feature_fundamental_asset_turnover | 0.995797 | core_feature | feature_fundamental_asset_turnover_z |
| feature_fundamental_net_margin_stability_z | feature_fundamental_net_margin_stability | 0.995557 | core_feature | feature_fundamental_net_margin_stability_z |

## 口径与结论

- 本轮新增的是 M7 研究版推荐报告，不新增生产策略；推荐名单不自动生成交易指令。
- 数据质量沿用 M2/M5/M6 的 PIT 检查，本脚本只使用 `report_signal_date` 当日可观测特征，并只用该日期之前已完成标签的月份训练。
- M9 数据完整性 gate 要求报告信号日存在 `next_trade_date`、推荐行全部 `buyable_tplus1_open`、名称非 UNKNOWN，且零覆盖/低覆盖字段不作为核心模型特征。
- 候选池使用 `U2_risk_sane` watchlist 口径，只做准入过滤；alpha 判断来自 `M6_xgboost_rank_ndcg` 排序分数。
- 历史 baseline 改善证据来自 M6 walk-forward 附件；M6 结论仍是 watchlist，不进入生产。
- 当前推荐月尚无未来收益标签，因此本报告不声称本月已跑赢市场；稳定性判断以历史 leaderboard / monthly_long / rank_ic / quantile_spread 为准。
- 分桶、年份和 regime 的历史失败月份保留在附件中；后续进入 regime-aware calibration 复核。

## 本轮产物

- `data/results/monthly_selection_m9_data_integrity_report_2026-04-30_summary.json`
- `data/results/monthly_selection_m9_data_integrity_report_2026-04-30_recommendations.csv`
- `data/results/monthly_selection_m9_data_integrity_report_2026-04-30_leaderboard.csv`
- `data/results/monthly_selection_m9_data_integrity_report_2026-04-30_monthly_long.csv`
- `data/results/monthly_selection_m9_data_integrity_report_2026-04-30_rank_ic.csv`
- `data/results/monthly_selection_m9_data_integrity_report_2026-04-30_quantile_spread.csv`
- `data/results/monthly_selection_m9_data_integrity_report_2026-04-30_topk_holdings.csv`
- `data/results/monthly_selection_m9_data_integrity_report_2026-04-30_industry_exposure.csv`
- `data/results/monthly_selection_m9_data_integrity_report_2026-04-30_candidate_pool_width.csv`
- `data/results/monthly_selection_m9_data_integrity_report_2026-04-30_candidate_pool_reject_reason.csv`
- `data/results/monthly_selection_m9_data_integrity_report_2026-04-30_feature_importance.csv`
- `data/results/monthly_selection_m9_data_integrity_report_2026-04-30_feature_contrib.csv`
- `data/results/monthly_selection_m9_data_integrity_report_2026-04-30_feature_coverage.csv`
- `data/results/monthly_selection_m9_data_integrity_report_2026-04-30_feature_policy.csv`
- `data/results/monthly_selection_m9_data_integrity_report_2026-04-30_m9_integrity.csv`
- `data/results/monthly_selection_m9_data_integrity_report_2026-04-30_risk_summary.csv`
- `data/results/monthly_selection_m9_data_integrity_report_2026-04-30_year_slice.csv`
- `data/results/monthly_selection_m9_data_integrity_report_2026-04-30_regime_slice.csv`
- `data/results/monthly_selection_m9_data_integrity_report_2026-04-30_manifest.json`
