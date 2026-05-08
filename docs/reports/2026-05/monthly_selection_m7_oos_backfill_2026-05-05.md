# Monthly Selection M7 Recommendation Report

- 生成时间：`2026-05-05T08:51:01.484853+00:00`
- 结果类型：`monthly_selection_m7_recommendation_report`
- 研究配置：`dataset_monthly_selection_features_families_price_volume_only-industry_breadth-fund_flow-fundamental_pools_u1_liquid_tradable_topk_20_model_m6_xgboost_rank_ndcg_maxfit_0_jobs_all_wf_24m`
- 报告信号日：`2026-03-31`
- 买入交易日：`2026-04-01`
- 卖出交易日：`2026-04-30`
- 候选池：`U1_liquid_tradable`
- 模型：`M6_xgboost_rank_ndcg`
- 生产状态：`research_only_not_promoted`

## 推荐名单

| signal_date | top_k | rank | symbol | name | score | score_percentile | industry | industry_level2 | feature_contrib | risk_flags | last_month_rank | last_month_selected | buyability | next_trade_date | buy_trade_date | sell_trade_date | candidate_pool_version | candidate_pool_rule | model | model_type | feature_spec |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-03-31 | 20 | 1 | 300344 | UNKNOWN | 1 | 1 | _UNKNOWN_ |  | feature_amount_20d_log=+18.9; feature_turnover_20d=+8.17; feature_limit_move_hits_20d=+4 | limit_move_path;extreme_volatility;extreme_turnover |  | False | buyable_tplus1_open | 2026-04-01 | 2026-04-01 | 2026-04-30 | U1_liquid_tradable | U0 + minimum history length + 20d average amount threshold | M6_xgboost_rank_ndcg | xgboost_ranker | m6_core_price_volume_industry_breadth_fund_flow_fundamental |
| 2026-03-31 | 20 | 2 | 600340 | 华夏幸福 | 0.999778 | 0.999778 | _UNKNOWN_ |  | feature_amount_20d_log=+19; feature_fundamental_asset_turnover_z=+5; feature_fundamental_net_margin_stability_z=-5 |  |  | False | buyable_tplus1_open | 2026-04-01 | 2026-04-01 | 2026-04-30 | U1_liquid_tradable | U0 + minimum history length + 20d average amount threshold | M6_xgboost_rank_ndcg | xgboost_ranker | m6_core_price_volume_industry_breadth_fund_flow_fundamental |
| 2026-03-31 | 20 | 3 | 002512 | ST达华 | 0.999556 | 0.999556 | 计算机 | 计算机设备 | feature_amount_20d_log=+19.7; feature_fundamental_pb_z=+1.95; feature_fundamental_roe_ttm_z=-2.43 |  |  | False | buyable_tplus1_open | 2026-04-01 | 2026-04-01 | 2026-04-30 | U1_liquid_tradable | U0 + minimum history length + 20d average amount threshold | M6_xgboost_rank_ndcg | xgboost_ranker | m6_core_price_volume_industry_breadth_fund_flow_fundamental |
| 2026-03-31 | 20 | 4 | 300290 | 荣科科技 | 0.999333 | 0.999333 | 计算机 | IT服务Ⅱ | feature_amount_20d_log=+19.3; feature_fundamental_pb_z=+2.71; feature_fundamental_pe_ttm_z=-2.64 |  |  | False | buyable_tplus1_open | 2026-04-01 | 2026-04-01 | 2026-04-30 | U1_liquid_tradable | U0 + minimum history length + 20d average amount threshold | M6_xgboost_rank_ndcg | xgboost_ranker | m6_core_price_volume_industry_breadth_fund_flow_fundamental |
| 2026-03-31 | 20 | 5 | 002228 | 合兴包装 | 0.999111 | 0.999111 | 轻工制造 | 包装印刷 | feature_amount_20d_log=+18.3; feature_industry_amount20_mean_z=-1.35; feature_fundamental_pb_z=-0.571 |  |  | False | buyable_tplus1_open | 2026-04-01 | 2026-04-01 | 2026-04-30 | U1_liquid_tradable | U0 + minimum history length + 20d average amount threshold | M6_xgboost_rank_ndcg | xgboost_ranker | m6_core_price_volume_industry_breadth_fund_flow_fundamental |
| 2026-03-31 | 20 | 6 | 002731 | ST萃华 | 0.998889 | 0.998889 | 纺织服饰 | 饰品 | feature_amount_20d_log=+19; feature_industry_amount20_mean_z=-1.56; feature_fundamental_pb_z=-0.471 |  |  | False | buyable_tplus1_open | 2026-04-01 | 2026-04-01 | 2026-04-30 | U1_liquid_tradable | U0 + minimum history length + 20d average amount threshold | M6_xgboost_rank_ndcg | xgboost_ranker | m6_core_price_volume_industry_breadth_fund_flow_fundamental |
| 2026-03-31 | 20 | 7 | 688382 | 益方生物 | 0.998667 | 0.998667 | 医药生物 | 化学制药 | feature_amount_20d_log=+19.6; feature_fundamental_asset_turnover_z=+5; feature_fundamental_net_margin_stability_z=-5 |  |  | False | buyable_tplus1_open | 2026-04-01 | 2026-04-01 | 2026-04-30 | U1_liquid_tradable | U0 + minimum history length + 20d average amount threshold | M6_xgboost_rank_ndcg | xgboost_ranker | m6_core_price_volume_industry_breadth_fund_flow_fundamental |
| 2026-03-31 | 20 | 8 | 000735 | 罗 牛 山 | 0.998444 | 0.998444 | 农林牧渔 | 养殖业 | feature_amount_20d_log=+19.2; feature_fundamental_pe_ttm_z=-1.78; feature_fundamental_gross_margin_delta_z=-1.2 |  |  | False | buyable_tplus1_open | 2026-04-01 | 2026-04-01 | 2026-04-30 | U1_liquid_tradable | U0 + minimum history length + 20d average amount threshold | M6_xgboost_rank_ndcg | xgboost_ranker | m6_core_price_volume_industry_breadth_fund_flow_fundamental |
| 2026-03-31 | 20 | 9 | 000838 | *ST发展 | 0.998222 | 0.998222 | 房地产 | 房地产开发 | feature_amount_20d_log=+18.2; feature_fundamental_gross_margin_change_z=+4.19; feature_fundamental_gross_margin_delta_z=+2.67 |  |  | False | buyable_tplus1_open | 2026-04-01 | 2026-04-01 | 2026-04-30 | U1_liquid_tradable | U0 + minimum history length + 20d average amount threshold | M6_xgboost_rank_ndcg | xgboost_ranker | m6_core_price_volume_industry_breadth_fund_flow_fundamental |
| 2026-03-31 | 20 | 10 | 603758 | 秦安股份 | 0.998 | 0.998 | 汽车 | 汽车零部件 | feature_amount_20d_log=+17.8; feature_fundamental_debt_to_assets_change_z=+3.15; feature_industry_ret60_mean_z=-1.2 |  |  | False | buyable_tplus1_open | 2026-04-01 | 2026-04-01 | 2026-04-30 | U1_liquid_tradable | U0 + minimum history length + 20d average amount threshold | M6_xgboost_rank_ndcg | xgboost_ranker | m6_core_price_volume_industry_breadth_fund_flow_fundamental |
| 2026-03-31 | 20 | 11 | 300255 | 常山药业 | 0.997778 | 0.997778 | 医药生物 | 化学制药 | feature_amount_20d_log=+20.4; feature_fundamental_pb_z=+5; feature_fundamental_net_profit_yoy_z=-3.22 |  |  | False | buyable_tplus1_open | 2026-04-01 | 2026-04-01 | 2026-04-30 | U1_liquid_tradable | U0 + minimum history length + 20d average amount threshold | M6_xgboost_rank_ndcg | xgboost_ranker | m6_core_price_volume_industry_breadth_fund_flow_fundamental |
| 2026-03-31 | 20 | 12 | 002163 | 海南发展 | 0.997556 | 0.997556 | 建筑装饰 | 装修装饰Ⅱ | feature_amount_20d_log=+19.8; feature_fundamental_roe_ttm_z=-4.09; feature_fundamental_pb_z=+1.53 |  |  | False | buyable_tplus1_open | 2026-04-01 | 2026-04-01 | 2026-04-30 | U1_liquid_tradable | U0 + minimum history length + 20d average amount threshold | M6_xgboost_rank_ndcg | xgboost_ranker | m6_core_price_volume_industry_breadth_fund_flow_fundamental |
| 2026-03-31 | 20 | 13 | 603008 | ST喜临门 | 0.997333 | 0.997333 | 轻工制造 | 家居用品 | feature_amount_20d_log=+19.2; feature_industry_amount20_mean_z=-1.35; feature_fundamental_pb_z=-0.52 |  |  | False | buyable_tplus1_open | 2026-04-01 | 2026-04-01 | 2026-04-30 | U1_liquid_tradable | U0 + minimum history length + 20d average amount threshold | M6_xgboost_rank_ndcg | xgboost_ranker | m6_core_price_volume_industry_breadth_fund_flow_fundamental |
| 2026-03-31 | 20 | 14 | 600882 | 妙可蓝多 | 0.997111 | 0.997111 | 食品饮料 | 饮料乳品 | feature_amount_20d_log=+18.5; feature_industry_ret60_mean_z=-1.88; feature_industry_low_vol20_mean_z=+2.07 |  |  | False | buyable_tplus1_open | 2026-04-01 | 2026-04-01 | 2026-04-30 | U1_liquid_tradable | U0 + minimum history length + 20d average amount threshold | M6_xgboost_rank_ndcg | xgboost_ranker | m6_core_price_volume_industry_breadth_fund_flow_fundamental |
| 2026-03-31 | 20 | 15 | 300716 | ST泉为 | 0.996889 | 0.996889 | _UNKNOWN_ |  | feature_amount_20d_log=+18.2; feature_fundamental_gross_margin_delta_z=-3.85; feature_fundamental_ocf_to_asset_z=-4.82 |  |  | False | buyable_tplus1_open | 2026-04-01 | 2026-04-01 | 2026-04-30 | U1_liquid_tradable | U0 + minimum history length + 20d average amount threshold | M6_xgboost_rank_ndcg | xgboost_ranker | m6_core_price_volume_industry_breadth_fund_flow_fundamental |
| 2026-03-31 | 20 | 16 | 301557 | 常友科技 | 0.996667 | 0.996667 | 电力设备 | 风电设备 | feature_amount_20d_log=+18.9; feature_industry_ret60_mean_z=+1.32; feature_industry_amount20_mean_z=+1.46 |  |  | False | buyable_tplus1_open | 2026-04-01 | 2026-04-01 | 2026-04-30 | U1_liquid_tradable | U0 + minimum history length + 20d average amount threshold | M6_xgboost_rank_ndcg | xgboost_ranker | m6_core_price_volume_industry_breadth_fund_flow_fundamental |
| 2026-03-31 | 20 | 17 | 605365 | 立达信 | 0.996444 | 0.996444 | 家用电器 | 照明设备Ⅱ | feature_amount_20d_log=+18.5; feature_industry_amount20_mean_z=-0.884; feature_industry_ret60_mean_z=-0.664 |  |  | False | buyable_tplus1_open | 2026-04-01 | 2026-04-01 | 2026-04-30 | U1_liquid_tradable | U0 + minimum history length + 20d average amount threshold | M6_xgboost_rank_ndcg | xgboost_ranker | m6_core_price_volume_industry_breadth_fund_flow_fundamental |
| 2026-03-31 | 20 | 18 | 605255 | 天普股份 | 0.996222 | 0.996222 | 汽车 | 汽车零部件 | feature_amount_20d_log=+19.3; feature_fundamental_pb_z=+1.93; feature_industry_ret60_mean_z=-1.2 |  |  | False | buyable_tplus1_open | 2026-04-01 | 2026-04-01 | 2026-04-30 | U1_liquid_tradable | U0 + minimum history length + 20d average amount threshold | M6_xgboost_rank_ndcg | xgboost_ranker | m6_core_price_volume_industry_breadth_fund_flow_fundamental |
| 2026-03-31 | 20 | 19 | 000980 | 众泰汽车 | 0.996 | 0.996 | 汽车 | 汽车零部件 | feature_amount_20d_log=+19.5; feature_fundamental_pb_z=+5; feature_fundamental_roe_ttm_z=-5 |  |  | False | buyable_tplus1_open | 2026-04-01 | 2026-04-01 | 2026-04-30 | U1_liquid_tradable | U0 + minimum history length + 20d average amount threshold | M6_xgboost_rank_ndcg | xgboost_ranker | m6_core_price_volume_industry_breadth_fund_flow_fundamental |
| 2026-03-31 | 20 | 20 | 301123 | 奕东电子 | 0.995778 | 0.995778 | 电子 | 消费电子 | feature_amount_20d_log=+19.5; feature_fundamental_pe_ttm_z=-2.03; feature_industry_low_vol20_mean_z=-1.14 |  |  | False | buyable_tplus1_open | 2026-04-01 | 2026-04-01 | 2026-04-30 | U1_liquid_tradable | U0 + minimum history length + 20d average amount threshold | M6_xgboost_rank_ndcg | xgboost_ranker | m6_core_price_volume_industry_breadth_fund_flow_fundamental |

## M6 Historical Evidence

| candidate_pool_version | model | model_type | top_k | months | mean_topk_return | median_topk_return | topk_excess_mean | topk_excess_median | topk_hit_rate | topk_minus_nextk_mean | industry_neutral_topk_excess_mean | turnover_mean | cost_drag_mean | topk_excess_after_cost_mean | topk_excess_annualized | rank_ic_mean | rank_ic_std | rank_ic_months | rank_ic_ir | quantile_top_minus_bottom_mean | neutral_median_excess | strong_down_median_excess | strong_up_median_excess |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| U1_liquid_tradable | M6_xgboost_rank_ndcg | xgboost_ranker | 20 | 39 | 0.0133695 | 0.00569568 | 0.0104648 | 0.00675653 | 0.564103 | 0.00624433 | 0.00821476 | 0.871053 | 0.000871053 | 0.00969885 | 0.133064 | 0.0181123 | 0.103873 | 39 | 0.174369 | 0.00570137 | 0.00665064 | 0.0317323 | -0.0119531 |

## Risk Summary

| signal_date | candidate_pool_version | model | top_k | selected_count | risk_flagged_count | not_buyable_count | last_month_selected_count |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-03-31 | U1_liquid_tradable | M6_xgboost_rank_ndcg | 20 | 20 | 1 | 0 | 0 |

## Industry Exposure

| signal_date | candidate_pool_version | model | top_k | industry | industry_count | topk_count | industry_share |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-03-31 | U1_liquid_tradable | M6_xgboost_rank_ndcg | 20 | _UNKNOWN_ | 3 | 20 | 0.15 |
| 2026-03-31 | U1_liquid_tradable | M6_xgboost_rank_ndcg | 20 | 农林牧渔 | 1 | 20 | 0.05 |
| 2026-03-31 | U1_liquid_tradable | M6_xgboost_rank_ndcg | 20 | 医药生物 | 2 | 20 | 0.1 |
| 2026-03-31 | U1_liquid_tradable | M6_xgboost_rank_ndcg | 20 | 家用电器 | 1 | 20 | 0.05 |
| 2026-03-31 | U1_liquid_tradable | M6_xgboost_rank_ndcg | 20 | 建筑装饰 | 1 | 20 | 0.05 |
| 2026-03-31 | U1_liquid_tradable | M6_xgboost_rank_ndcg | 20 | 房地产 | 1 | 20 | 0.05 |
| 2026-03-31 | U1_liquid_tradable | M6_xgboost_rank_ndcg | 20 | 汽车 | 3 | 20 | 0.15 |
| 2026-03-31 | U1_liquid_tradable | M6_xgboost_rank_ndcg | 20 | 电力设备 | 1 | 20 | 0.05 |
| 2026-03-31 | U1_liquid_tradable | M6_xgboost_rank_ndcg | 20 | 电子 | 1 | 20 | 0.05 |
| 2026-03-31 | U1_liquid_tradable | M6_xgboost_rank_ndcg | 20 | 纺织服饰 | 1 | 20 | 0.05 |
| 2026-03-31 | U1_liquid_tradable | M6_xgboost_rank_ndcg | 20 | 计算机 | 2 | 20 | 0.1 |
| 2026-03-31 | U1_liquid_tradable | M6_xgboost_rank_ndcg | 20 | 轻工制造 | 2 | 20 | 0.1 |
| 2026-03-31 | U1_liquid_tradable | M6_xgboost_rank_ndcg | 20 | 食品饮料 | 1 | 20 | 0.05 |

## Feature Coverage

| candidate_pool_version | feature_spec | families | feature | raw_feature | rows | candidate_pool_pass_rows | non_null | coverage_ratio | candidate_pool_pass_coverage_ratio | first_signal_date | last_signal_date |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| U1_liquid_tradable | m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_ret_5d | feature_ret_5d | 307350 | 209148 | 307017 | 0.998917 | 1 | 2021-01-29 | 2026-04-30 |
| U1_liquid_tradable | m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_ret_20d | feature_ret_20d | 307350 | 209148 | 306083 | 0.995878 | 1 | 2021-01-29 | 2026-04-30 |
| U1_liquid_tradable | m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_ret_60d | feature_ret_60d | 307350 | 209148 | 303506 | 0.987493 | 1 | 2021-01-29 | 2026-04-30 |
| U1_liquid_tradable | m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_realized_vol_20d | feature_realized_vol_20d | 307350 | 209148 | 306673 | 0.997797 | 1 | 2021-01-29 | 2026-04-30 |
| U1_liquid_tradable | m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_amount_20d_log | feature_amount_20d_log | 307350 | 209148 | 306745 | 0.998032 | 1 | 2021-01-29 | 2026-04-30 |
| U1_liquid_tradable | m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_turnover_20d | feature_turnover_20d | 307350 | 209148 | 306745 | 0.998032 | 1 | 2021-01-29 | 2026-04-30 |
| U1_liquid_tradable | m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_price_position_250d | feature_price_position_250d | 307350 | 209148 | 303573 | 0.987711 | 1 | 2021-01-29 | 2026-04-30 |
| U1_liquid_tradable | m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_limit_move_hits_20d | feature_limit_move_hits_20d | 307350 | 209148 | 307350 | 1 | 1 | 2021-01-29 | 2026-04-30 |
| U1_liquid_tradable | m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_industry_ret20_mean_z | feature_industry_ret20_mean | 307350 | 209148 | 307350 | 1 | 1 | 2021-01-29 | 2026-04-30 |
| U1_liquid_tradable | m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_industry_ret60_mean_z | feature_industry_ret60_mean | 307350 | 209148 | 307350 | 1 | 1 | 2021-01-29 | 2026-04-30 |
| U1_liquid_tradable | m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_industry_positive_ret20_ratio_z | feature_industry_positive_ret20_ratio | 307350 | 209148 | 307350 | 1 | 1 | 2021-01-29 | 2026-04-30 |
| U1_liquid_tradable | m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_industry_amount20_mean_z | feature_industry_amount20_mean | 307350 | 209148 | 307350 | 1 | 1 | 2021-01-29 | 2026-04-30 |
| U1_liquid_tradable | m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_industry_low_vol20_mean_z | feature_industry_low_vol20_mean | 307350 | 209148 | 307350 | 1 | 1 | 2021-01-29 | 2026-04-30 |
| U1_liquid_tradable | m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_fund_flow_main_inflow_5d_z | feature_fund_flow_main_inflow_5d | 307350 | 209148 | 35590 | 0.115796 | 0.127948 | 2023-10-31 | 2026-04-30 |
| U1_liquid_tradable | m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_fund_flow_main_inflow_10d_z | feature_fund_flow_main_inflow_10d | 307350 | 209148 | 31196 | 0.1015 | 0.10964 | 2023-11-30 | 2026-04-30 |
| U1_liquid_tradable | m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_fund_flow_main_inflow_20d_z | feature_fund_flow_main_inflow_20d | 307350 | 209148 | 31048 | 0.101018 | 0.109138 | 2023-11-30 | 2026-04-30 |
| U1_liquid_tradable | m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_fund_flow_super_inflow_10d_z | feature_fund_flow_super_inflow_10d | 307350 | 209148 | 31196 | 0.1015 | 0.10964 | 2023-11-30 | 2026-04-30 |
| U1_liquid_tradable | m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_fund_flow_divergence_20d_z | feature_fund_flow_divergence_20d | 307350 | 209148 | 31048 | 0.101018 | 0.109138 | 2023-11-30 | 2026-04-30 |
| U1_liquid_tradable | m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_fund_flow_main_inflow_streak_z | feature_fund_flow_main_inflow_streak | 307350 | 209148 | 36162 | 0.117657 | 0.130348 | 2023-10-31 | 2026-04-30 |
| U1_liquid_tradable | m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_fundamental_pe_ttm_z | feature_fundamental_pe_ttm | 307350 | 209148 | 304469 | 0.990626 | 0.999766 | 2021-01-29 | 2026-04-30 |
| U1_liquid_tradable | m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_fundamental_pb_z | feature_fundamental_pb | 307350 | 209148 | 304469 | 0.990626 | 0.999766 | 2021-01-29 | 2026-04-30 |
| U1_liquid_tradable | m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_fundamental_ev_ebitda_z | feature_fundamental_ev_ebitda | 307350 | 209148 | 0 | 0 | 0 |  |  |
| U1_liquid_tradable | m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_fundamental_roe_ttm_z | feature_fundamental_roe_ttm | 307350 | 209148 | 304617 | 0.991108 | 0.992871 | 2021-01-29 | 2026-04-30 |
| U1_liquid_tradable | m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_fundamental_net_profit_yoy_z | feature_fundamental_net_profit_yoy | 307350 | 209148 | 306367 | 0.996802 | 0.997055 | 2021-01-29 | 2026-04-30 |
| U1_liquid_tradable | m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_fundamental_gross_margin_change_z | feature_fundamental_gross_margin_change | 307350 | 209148 | 299889 | 0.975725 | 0.96863 | 2021-01-29 | 2026-04-30 |
| U1_liquid_tradable | m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_fundamental_gross_margin_delta_z | feature_fundamental_gross_margin_delta | 307350 | 209148 | 299765 | 0.975321 | 0.968386 | 2021-01-29 | 2026-04-30 |
| U1_liquid_tradable | m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_fundamental_debt_to_assets_change_z | feature_fundamental_debt_to_assets_change | 307350 | 209148 | 306206 | 0.996278 | 0.997055 | 2021-01-29 | 2026-04-30 |
| U1_liquid_tradable | m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_fundamental_ocf_to_net_profit_z | feature_fundamental_ocf_to_net_profit | 307350 | 209148 | 245847 | 0.799893 | 0.811349 | 2021-01-29 | 2026-04-30 |
| U1_liquid_tradable | m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_fundamental_ocf_to_asset_z | feature_fundamental_ocf_to_asset | 307350 | 209148 | 306219 | 0.99632 | 0.996596 | 2021-01-29 | 2026-04-30 |
| U1_liquid_tradable | m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_fundamental_asset_turnover_z | feature_fundamental_asset_turnover | 307350 | 209148 | 306300 | 0.996584 | 0.996911 | 2021-01-29 | 2026-04-30 |
| U1_liquid_tradable | m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_fundamental_net_margin_stability_z | feature_fundamental_net_margin_stability | 307350 | 209148 | 306150 | 0.996096 | 0.996677 | 2021-01-29 | 2026-04-30 |

## M9 Integrity

| check | value | pass | detail |
| --- | --- | --- | --- |
| target_candidate_pool_pass_rows | 4500 | True | latest selected signal date must have buyable candidates |
| target_next_trade_date_present | 4500 | True | all candidate_pool_pass rows should carry next_trade_date |
| recommendation_buyable | 0 | True | recommendation rows should be buyable at t+1 open |
| recommendation_names_readable | 1 | False | name should not be UNKNOWN or blank |
| recommendation_excludes_st_names | 5 | False | name-aware report filter should exclude ST/*ST targets |
| zero_coverage_core_features | 0 | True | zero coverage fields must not be core model features |
| low_coverage_core_features_lt_30pct | 0 | True | low coverage fields should be missing-marker-only or ablation-only |
| max_single_industry_share_le_40pct | 0.15 | True | 单行业占比 15.0%，超 40% 视为集中度过高 |
| distinct_industry_count_ge_3 | 13 | True | Top-K 推荐覆盖 13 个行业，至少应覆盖 3 个 |

## M9 Feature Policy

| feature | raw_feature | candidate_pool_pass_coverage_ratio | m9_feature_policy | active_feature |
| --- | --- | --- | --- | --- |
| feature_ret_5d | feature_ret_5d | 1 | core_feature | feature_ret_5d |
| feature_ret_20d | feature_ret_20d | 1 | core_feature | feature_ret_20d |
| feature_ret_60d | feature_ret_60d | 1 | core_feature | feature_ret_60d |
| feature_realized_vol_20d | feature_realized_vol_20d | 1 | core_feature | feature_realized_vol_20d |
| feature_amount_20d_log | feature_amount_20d_log | 1 | core_feature | feature_amount_20d_log |
| feature_turnover_20d | feature_turnover_20d | 1 | core_feature | feature_turnover_20d |
| feature_price_position_250d | feature_price_position_250d | 1 | core_feature | feature_price_position_250d |
| feature_limit_move_hits_20d | feature_limit_move_hits_20d | 1 | core_feature | feature_limit_move_hits_20d |
| feature_industry_ret20_mean_z | feature_industry_ret20_mean | 1 | core_feature | feature_industry_ret20_mean_z |
| feature_industry_ret60_mean_z | feature_industry_ret60_mean | 1 | core_feature | feature_industry_ret60_mean_z |
| feature_industry_positive_ret20_ratio_z | feature_industry_positive_ret20_ratio | 1 | core_feature | feature_industry_positive_ret20_ratio_z |
| feature_industry_amount20_mean_z | feature_industry_amount20_mean | 1 | core_feature | feature_industry_amount20_mean_z |
| feature_industry_low_vol20_mean_z | feature_industry_low_vol20_mean | 1 | core_feature | feature_industry_low_vol20_mean_z |
| feature_fund_flow_main_inflow_5d_z | feature_fund_flow_main_inflow_5d | 0.127948 | missing_flag_only_low_coverage | is_missing_feature_fund_flow_main_inflow_5d |
| feature_fund_flow_main_inflow_10d_z | feature_fund_flow_main_inflow_10d | 0.10964 | missing_flag_only_low_coverage | is_missing_feature_fund_flow_main_inflow_10d |
| feature_fund_flow_main_inflow_20d_z | feature_fund_flow_main_inflow_20d | 0.109138 | missing_flag_only_low_coverage | is_missing_feature_fund_flow_main_inflow_20d |
| feature_fund_flow_super_inflow_10d_z | feature_fund_flow_super_inflow_10d | 0.10964 | missing_flag_only_low_coverage | is_missing_feature_fund_flow_super_inflow_10d |
| feature_fund_flow_divergence_20d_z | feature_fund_flow_divergence_20d | 0.109138 | missing_flag_only_low_coverage | is_missing_feature_fund_flow_divergence_20d |
| feature_fund_flow_main_inflow_streak_z | feature_fund_flow_main_inflow_streak | 0.130348 | missing_flag_only_low_coverage | is_missing_feature_fund_flow_main_inflow_streak |
| feature_fundamental_pe_ttm_z | feature_fundamental_pe_ttm | 0.999766 | core_feature | feature_fundamental_pe_ttm_z |
| feature_fundamental_pb_z | feature_fundamental_pb | 0.999766 | core_feature | feature_fundamental_pb_z |
| feature_fundamental_ev_ebitda_z |  |  | missing_from_dataset |  |
| feature_fundamental_roe_ttm_z | feature_fundamental_roe_ttm | 0.992871 | core_feature | feature_fundamental_roe_ttm_z |
| feature_fundamental_net_profit_yoy_z | feature_fundamental_net_profit_yoy | 0.997055 | core_feature | feature_fundamental_net_profit_yoy_z |
| feature_fundamental_gross_margin_change_z | feature_fundamental_gross_margin_change | 0.96863 | core_feature | feature_fundamental_gross_margin_change_z |
| feature_fundamental_gross_margin_delta_z | feature_fundamental_gross_margin_delta | 0.968386 | core_feature | feature_fundamental_gross_margin_delta_z |
| feature_fundamental_debt_to_assets_change_z | feature_fundamental_debt_to_assets_change | 0.997055 | core_feature | feature_fundamental_debt_to_assets_change_z |
| feature_fundamental_ocf_to_net_profit_z | feature_fundamental_ocf_to_net_profit | 0.811349 | core_feature | feature_fundamental_ocf_to_net_profit_z |
| feature_fundamental_ocf_to_asset_z | feature_fundamental_ocf_to_asset | 0.996596 | core_feature | feature_fundamental_ocf_to_asset_z |
| feature_fundamental_asset_turnover_z | feature_fundamental_asset_turnover | 0.996911 | core_feature | feature_fundamental_asset_turnover_z |
| feature_fundamental_net_margin_stability_z | feature_fundamental_net_margin_stability | 0.996677 | core_feature | feature_fundamental_net_margin_stability_z |

## 口径与结论

- 本轮新增的是 M7 研究版推荐报告，不新增生产策略；推荐名单不自动生成交易指令。
- 数据质量沿用 M2/M5/M6 的 PIT 检查，本脚本只使用 `report_signal_date` 当日可观测特征，并只用该日期之前已完成标签的月份训练。
- M9 数据完整性 gate 要求报告信号日存在 `next_trade_date`、推荐行全部 `buyable_tplus1_open`、名称非 UNKNOWN，且零覆盖/低覆盖字段不作为核心模型特征。
- 月度交易时点按"本持有月最后一个交易日卖出、下一持有月首个交易日买入"展示；`sell_trade_date` 为下一次月末信号日。
- 候选池使用 `U2_risk_sane` watchlist 口径，只做准入过滤；alpha 判断来自 `M6_xgboost_rank_ndcg` 排序分数。
- 历史 baseline 改善证据来自 M6 walk-forward 附件；M6 结论仍是 watchlist，不进入生产。
- 当前推荐月尚无未来收益标签，因此本报告不声称本月已跑赢市场；稳定性判断以历史 leaderboard / monthly_long / rank_ic / quantile_spread 为准。
- 分桶、年份和 regime 的历史失败月份保留在附件中；后续进入 regime-aware calibration 复核。

## 本轮产物

- `data/results/monthly_selection_m7_oos_backfill_2026-05-05_summary.json`
- `data/results/monthly_selection_m7_oos_backfill_2026-05-05_recommendations.csv`
- `data/results/monthly_selection_m7_oos_backfill_2026-05-05_leaderboard.csv`
- `data/results/monthly_selection_m7_oos_backfill_2026-05-05_monthly_long.csv`
- `data/results/monthly_selection_m7_oos_backfill_2026-05-05_rank_ic.csv`
- `data/results/monthly_selection_m7_oos_backfill_2026-05-05_quantile_spread.csv`
- `data/results/monthly_selection_m7_oos_backfill_2026-05-05_topk_holdings.csv`
- `data/results/monthly_selection_m7_oos_backfill_2026-05-05_industry_exposure.csv`
- `data/results/monthly_selection_m7_oos_backfill_2026-05-05_candidate_pool_width.csv`
- `data/results/monthly_selection_m7_oos_backfill_2026-05-05_candidate_pool_reject_reason.csv`
- `data/results/monthly_selection_m7_oos_backfill_2026-05-05_feature_importance.csv`
- `data/results/monthly_selection_m7_oos_backfill_2026-05-05_feature_contrib.csv`
- `data/results/monthly_selection_m7_oos_backfill_2026-05-05_feature_coverage.csv`
- `data/results/monthly_selection_m7_oos_backfill_2026-05-05_feature_policy.csv`
- `data/results/monthly_selection_m7_oos_backfill_2026-05-05_m9_integrity.csv`
- `data/results/monthly_selection_m7_oos_backfill_2026-05-05_risk_summary.csv`
- `data/results/monthly_selection_m7_oos_backfill_2026-05-05_year_slice.csv`
- `data/results/monthly_selection_m7_oos_backfill_2026-05-05_regime_slice.csv`
- `data/results/monthly_selection_m7_oos_backfill_2026-05-05_manifest.json`
