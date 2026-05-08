# Monthly Selection M5 Multisource

- 生成时间：`2026-05-06T04:27:48.595368+00:00` · 输出 stem：`m5_margin_delta_full_2026-05-06`
- 数据集：`data/cache/monthly_selection_features.parquet` · 数据库：`data/market.duckdb`
- 有效标签月份：`63` · 单窗训练行上限：`0`

## Leaderboard

| candidate_pool_version | model | model_type | top_k | months | mean_topk_return | median_topk_return | topk_excess_mean | topk_excess_median | topk_hit_rate | topk_minus_nextk_mean | industry_neutral_topk_excess_mean | turnover_mean | cost_drag_mean | topk_excess_after_cost_mean | topk_excess_annualized | rank_ic_mean | rank_ic_std | rank_ic_months | rank_ic_ir | quantile_top_minus_bottom_mean | neutral_median_excess | strong_down_median_excess | strong_up_median_excess |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| U1_liquid_tradable | M5_price_volume_only_elasticnet_excess | elasticnet | 20 | 51 | 0.0164146 | 0.0207779 | 0.0134845 | 0.00537175 | 0.607843 | 0.00984394 | 0.0146953 | 0.965 | 0.000965 | 0.0134428 | 0.174371 | 0.0993018 | 0.111158 | 51 | 0.89334 | 0.0164281 | 0.0171706 | -0.0092517 | 0.00142474 |
| U1_liquid_tradable | M5_plus_industry_breadth_plus_fund_flow_plus_fundamental_plus_shareholder_plus_margin_trading_elasticnet_excess | elasticnet | 20 | 51 | 0.017605 | 0.0224348 | 0.014675 | 0.0108484 | 0.627451 | 0.011941 | 0.0161978 | 0.867 | 0.000867 | 0.0132143 | 0.191032 | 0.0963135 | 0.11773 | 51 | 0.818091 | 0.0180761 | 0.0127822 | 0.0225001 | -0.0266754 |
| U1_liquid_tradable | M5_plus_industry_breadth_plus_fund_flow_plus_fundamental_plus_shareholder_elasticnet_excess | elasticnet | 20 | 51 | 0.0171737 | 0.0120354 | 0.0142436 | 0.0174908 | 0.607843 | 0.0101213 | 0.0161273 | 0.863 | 0.000863 | 0.0130181 | 0.18497 | 0.0978982 | 0.119136 | 51 | 0.821736 | 0.0183609 | 0.0174908 | 0.0242076 | -0.0353133 |
| U1_liquid_tradable | M5_plus_industry_breadth_plus_fund_flow_elasticnet_excess | elasticnet | 20 | 51 | 0.0148223 | 0.00492588 | 0.0118923 | 0.00862209 | 0.588235 | 0.0120226 | 0.0170551 | 0.964 | 0.000964 | 0.0107218 | 0.152421 | 0.0817885 | 0.108779 | 51 | 0.751878 | 0.0145607 | 0.0126042 | 0.00107566 | -0.00789027 |
| U1_liquid_tradable | M5_plus_industry_breadth_elasticnet_excess | elasticnet | 20 | 51 | 0.0114985 | 0.00458493 | 0.00856849 | 0.00315811 | 0.529412 | 0.00738746 | 0.0138756 | 0.965 | 0.000965 | 0.00733056 | 0.107809 | 0.084385 | 0.111178 | 51 | 0.75901 | 0.0139925 | 0.0110535 | -0.00343948 | -0.00789027 |
| U1_liquid_tradable | M5_plus_industry_breadth_plus_fund_flow_plus_fundamental_elasticnet_excess | elasticnet | 20 | 51 | 0.0115172 | 0.0113926 | 0.0085872 | 0.00359774 | 0.54902 | 0.00206531 | 0.012946 | 0.904 | 0.000904 | 0.00711739 | 0.108055 | 0.0900053 | 0.112447 | 51 | 0.800426 | 0.0173627 | 0.00644385 | 0.014084 | -0.0124275 |
| U1_liquid_tradable | B0_market_ew | benchmark | 20 | 51 | 0.00293005 | 0.00334796 | 0 | 0 | 0 |  |  |  | 0 | 0 | 0 |  |  |  |  |  | 0 | 0 | 0 |

## Incremental Delta vs Price-Volume

| candidate_pool_version | model | model_type | top_k | months | mean_topk_return | median_topk_return | topk_excess_mean | topk_excess_median | topk_hit_rate | topk_minus_nextk_mean | industry_neutral_topk_excess_mean | turnover_mean | cost_drag_mean | topk_excess_after_cost_mean | topk_excess_annualized | rank_ic_mean | rank_ic_std | rank_ic_months | rank_ic_ir | quantile_top_minus_bottom_mean | neutral_median_excess | strong_down_median_excess | strong_up_median_excess | feature_spec | base_model | baseline_topk_excess_mean | baseline_topk_excess_after_cost_mean | baseline_rank_ic_mean | baseline_quantile_top_minus_bottom_mean | delta_topk_excess_mean | delta_topk_excess_after_cost_mean | delta_rank_ic_mean | delta_quantile_top_minus_bottom_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| U1_liquid_tradable | M5_plus_industry_breadth_plus_fund_flow_plus_fundamental_plus_shareholder_plus_margin_trading_elasticnet_excess | elasticnet | 20 | 51 | 0.017605 | 0.0224348 | 0.014675 | 0.0108484 | 0.627451 | 0.011941 | 0.0161978 | 0.867 | 0.000867 | 0.0132143 | 0.191032 | 0.0963135 | 0.11773 | 51 | 0.818091 | 0.0180761 | 0.0127822 | 0.0225001 | -0.0266754 | plus_industry_breadth_plus_fund_flow_plus_fundamental_plus_shareholder_plus_margin_trading | elasticnet_excess | 0.0134845 | 0.0134428 | 0.0993018 | 0.0164281 | 0.00119045 | -0.000228444 | -0.00298832 | 0.00164797 |
| U1_liquid_tradable | M5_plus_industry_breadth_plus_fund_flow_plus_fundamental_plus_shareholder_elasticnet_excess | elasticnet | 20 | 51 | 0.0171737 | 0.0120354 | 0.0142436 | 0.0174908 | 0.607843 | 0.0101213 | 0.0161273 | 0.863 | 0.000863 | 0.0130181 | 0.18497 | 0.0978982 | 0.119136 | 51 | 0.821736 | 0.0183609 | 0.0174908 | 0.0242076 | -0.0353133 | plus_industry_breadth_plus_fund_flow_plus_fundamental_plus_shareholder | elasticnet_excess | 0.0134845 | 0.0134428 | 0.0993018 | 0.0164281 | 0.000759084 | -0.000424702 | -0.00140366 | 0.00193284 |
| U1_liquid_tradable | M5_plus_industry_breadth_plus_fund_flow_elasticnet_excess | elasticnet | 20 | 51 | 0.0148223 | 0.00492588 | 0.0118923 | 0.00862209 | 0.588235 | 0.0120226 | 0.0170551 | 0.964 | 0.000964 | 0.0107218 | 0.152421 | 0.0817885 | 0.108779 | 51 | 0.751878 | 0.0145607 | 0.0126042 | 0.00107566 | -0.00789027 | plus_industry_breadth_plus_fund_flow | elasticnet_excess | 0.0134845 | 0.0134428 | 0.0993018 | 0.0164281 | -0.00159226 | -0.00272099 | -0.0175134 | -0.00186741 |
| U1_liquid_tradable | M5_plus_industry_breadth_elasticnet_excess | elasticnet | 20 | 51 | 0.0114985 | 0.00458493 | 0.00856849 | 0.00315811 | 0.529412 | 0.00738746 | 0.0138756 | 0.965 | 0.000965 | 0.00733056 | 0.107809 | 0.084385 | 0.111178 | 51 | 0.75901 | 0.0139925 | 0.0110535 | -0.00343948 | -0.00789027 | plus_industry_breadth | elasticnet_excess | 0.0134845 | 0.0134428 | 0.0993018 | 0.0164281 | -0.00491602 | -0.00611223 | -0.0149168 | -0.00243557 |
| U1_liquid_tradable | M5_plus_industry_breadth_plus_fund_flow_plus_fundamental_elasticnet_excess | elasticnet | 20 | 51 | 0.0115172 | 0.0113926 | 0.0085872 | 0.00359774 | 0.54902 | 0.00206531 | 0.012946 | 0.904 | 0.000904 | 0.00711739 | 0.108055 | 0.0900053 | 0.112447 | 51 | 0.800426 | 0.0173627 | 0.00644385 | 0.014084 | -0.0124275 | plus_industry_breadth_plus_fund_flow_plus_fundamental | elasticnet_excess | 0.0134845 | 0.0134428 | 0.0993018 | 0.0164281 | -0.00489732 | -0.0063254 | -0.00929654 | 0.000934566 |

## Feature Coverage

| feature_spec | families | feature | raw_feature | rows | non_null | coverage_ratio | candidate_pool_pass_coverage_ratio | first_signal_date | last_signal_date |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| plus_industry_breadth | price_volume,industry_breadth | feature_ret_5d | feature_ret_5d | 307350 | 307017 | 0.998917 | 1 | 2021-01-29 | 2026-04-30 |
| plus_industry_breadth | price_volume,industry_breadth | feature_ret_20d | feature_ret_20d | 307350 | 306083 | 0.995878 | 1 | 2021-01-29 | 2026-04-30 |
| plus_industry_breadth | price_volume,industry_breadth | feature_ret_60d | feature_ret_60d | 307350 | 303506 | 0.987493 | 1 | 2021-01-29 | 2026-04-30 |
| plus_industry_breadth | price_volume,industry_breadth | feature_realized_vol_20d | feature_realized_vol_20d | 307350 | 306673 | 0.997797 | 1 | 2021-01-29 | 2026-04-30 |
| plus_industry_breadth | price_volume,industry_breadth | feature_amount_20d_log | feature_amount_20d_log | 307350 | 306745 | 0.998032 | 1 | 2021-01-29 | 2026-04-30 |
| plus_industry_breadth | price_volume,industry_breadth | feature_turnover_20d | feature_turnover_20d | 307350 | 306745 | 0.998032 | 1 | 2021-01-29 | 2026-04-30 |
| plus_industry_breadth | price_volume,industry_breadth | feature_price_position_250d | feature_price_position_250d | 307350 | 303573 | 0.987711 | 1 | 2021-01-29 | 2026-04-30 |
| plus_industry_breadth | price_volume,industry_breadth | feature_limit_move_hits_20d | feature_limit_move_hits_20d | 307350 | 307350 | 1 | 1 | 2021-01-29 | 2026-04-30 |
| plus_industry_breadth | price_volume,industry_breadth | feature_industry_ret20_mean_z | feature_industry_ret20_mean | 307350 | 307350 | 1 | 1 | 2021-01-29 | 2026-04-30 |
| plus_industry_breadth | price_volume,industry_breadth | feature_industry_ret60_mean_z | feature_industry_ret60_mean | 307350 | 307350 | 1 | 1 | 2021-01-29 | 2026-04-30 |
| plus_industry_breadth | price_volume,industry_breadth | feature_industry_positive_ret20_ratio_z | feature_industry_positive_ret20_ratio | 307350 | 307350 | 1 | 1 | 2021-01-29 | 2026-04-30 |
| plus_industry_breadth | price_volume,industry_breadth | feature_industry_amount20_mean_z | feature_industry_amount20_mean | 307350 | 307350 | 1 | 1 | 2021-01-29 | 2026-04-30 |
| plus_industry_breadth | price_volume,industry_breadth | feature_industry_low_vol20_mean_z | feature_industry_low_vol20_mean | 307350 | 307350 | 1 | 1 | 2021-01-29 | 2026-04-30 |
| plus_industry_breadth_plus_fund_flow | price_volume,industry_breadth,fund_flow | feature_ret_5d | feature_ret_5d | 307350 | 307017 | 0.998917 | 1 | 2021-01-29 | 2026-04-30 |
| plus_industry_breadth_plus_fund_flow | price_volume,industry_breadth,fund_flow | feature_ret_20d | feature_ret_20d | 307350 | 306083 | 0.995878 | 1 | 2021-01-29 | 2026-04-30 |
| plus_industry_breadth_plus_fund_flow | price_volume,industry_breadth,fund_flow | feature_ret_60d | feature_ret_60d | 307350 | 303506 | 0.987493 | 1 | 2021-01-29 | 2026-04-30 |
| plus_industry_breadth_plus_fund_flow | price_volume,industry_breadth,fund_flow | feature_realized_vol_20d | feature_realized_vol_20d | 307350 | 306673 | 0.997797 | 1 | 2021-01-29 | 2026-04-30 |
| plus_industry_breadth_plus_fund_flow | price_volume,industry_breadth,fund_flow | feature_amount_20d_log | feature_amount_20d_log | 307350 | 306745 | 0.998032 | 1 | 2021-01-29 | 2026-04-30 |
| plus_industry_breadth_plus_fund_flow | price_volume,industry_breadth,fund_flow | feature_turnover_20d | feature_turnover_20d | 307350 | 306745 | 0.998032 | 1 | 2021-01-29 | 2026-04-30 |
| plus_industry_breadth_plus_fund_flow | price_volume,industry_breadth,fund_flow | feature_price_position_250d | feature_price_position_250d | 307350 | 303573 | 0.987711 | 1 | 2021-01-29 | 2026-04-30 |
| plus_industry_breadth_plus_fund_flow | price_volume,industry_breadth,fund_flow | feature_limit_move_hits_20d | feature_limit_move_hits_20d | 307350 | 307350 | 1 | 1 | 2021-01-29 | 2026-04-30 |
| plus_industry_breadth_plus_fund_flow | price_volume,industry_breadth,fund_flow | feature_industry_ret20_mean_z | feature_industry_ret20_mean | 307350 | 307350 | 1 | 1 | 2021-01-29 | 2026-04-30 |
| plus_industry_breadth_plus_fund_flow | price_volume,industry_breadth,fund_flow | feature_industry_ret60_mean_z | feature_industry_ret60_mean | 307350 | 307350 | 1 | 1 | 2021-01-29 | 2026-04-30 |
| plus_industry_breadth_plus_fund_flow | price_volume,industry_breadth,fund_flow | feature_industry_positive_ret20_ratio_z | feature_industry_positive_ret20_ratio | 307350 | 307350 | 1 | 1 | 2021-01-29 | 2026-04-30 |
| plus_industry_breadth_plus_fund_flow | price_volume,industry_breadth,fund_flow | feature_industry_amount20_mean_z | feature_industry_amount20_mean | 307350 | 307350 | 1 | 1 | 2021-01-29 | 2026-04-30 |
| plus_industry_breadth_plus_fund_flow | price_volume,industry_breadth,fund_flow | feature_industry_low_vol20_mean_z | feature_industry_low_vol20_mean | 307350 | 307350 | 1 | 1 | 2021-01-29 | 2026-04-30 |
| plus_industry_breadth_plus_fund_flow | price_volume,industry_breadth,fund_flow | feature_fund_flow_main_inflow_5d_z | feature_fund_flow_main_inflow_5d | 307350 | 35590 | 0.115796 | 0.127948 | 2023-10-31 | 2026-04-30 |
| plus_industry_breadth_plus_fund_flow | price_volume,industry_breadth,fund_flow | feature_fund_flow_main_inflow_10d_z | feature_fund_flow_main_inflow_10d | 307350 | 31196 | 0.1015 | 0.10964 | 2023-11-30 | 2026-04-30 |
| plus_industry_breadth_plus_fund_flow | price_volume,industry_breadth,fund_flow | feature_fund_flow_main_inflow_20d_z | feature_fund_flow_main_inflow_20d | 307350 | 31048 | 0.101018 | 0.109138 | 2023-11-30 | 2026-04-30 |
| plus_industry_breadth_plus_fund_flow | price_volume,industry_breadth,fund_flow | feature_fund_flow_super_inflow_10d_z | feature_fund_flow_super_inflow_10d | 307350 | 31196 | 0.1015 | 0.10964 | 2023-11-30 | 2026-04-30 |
| plus_industry_breadth_plus_fund_flow | price_volume,industry_breadth,fund_flow | feature_fund_flow_divergence_20d_z | feature_fund_flow_divergence_20d | 307350 | 31048 | 0.101018 | 0.109138 | 2023-11-30 | 2026-04-30 |
| plus_industry_breadth_plus_fund_flow | price_volume,industry_breadth,fund_flow | feature_fund_flow_main_inflow_streak_z | feature_fund_flow_main_inflow_streak | 307350 | 36162 | 0.117657 | 0.130348 | 2023-10-31 | 2026-04-30 |
| plus_industry_breadth_plus_fund_flow_plus_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_ret_5d | feature_ret_5d | 307350 | 307017 | 0.998917 | 1 | 2021-01-29 | 2026-04-30 |
| plus_industry_breadth_plus_fund_flow_plus_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_ret_20d | feature_ret_20d | 307350 | 306083 | 0.995878 | 1 | 2021-01-29 | 2026-04-30 |
| plus_industry_breadth_plus_fund_flow_plus_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_ret_60d | feature_ret_60d | 307350 | 303506 | 0.987493 | 1 | 2021-01-29 | 2026-04-30 |
| plus_industry_breadth_plus_fund_flow_plus_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_realized_vol_20d | feature_realized_vol_20d | 307350 | 306673 | 0.997797 | 1 | 2021-01-29 | 2026-04-30 |
| plus_industry_breadth_plus_fund_flow_plus_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_amount_20d_log | feature_amount_20d_log | 307350 | 306745 | 0.998032 | 1 | 2021-01-29 | 2026-04-30 |
| plus_industry_breadth_plus_fund_flow_plus_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_turnover_20d | feature_turnover_20d | 307350 | 306745 | 0.998032 | 1 | 2021-01-29 | 2026-04-30 |
| plus_industry_breadth_plus_fund_flow_plus_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_price_position_250d | feature_price_position_250d | 307350 | 303573 | 0.987711 | 1 | 2021-01-29 | 2026-04-30 |
| plus_industry_breadth_plus_fund_flow_plus_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_limit_move_hits_20d | feature_limit_move_hits_20d | 307350 | 307350 | 1 | 1 | 2021-01-29 | 2026-04-30 |
| plus_industry_breadth_plus_fund_flow_plus_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_industry_ret20_mean_z | feature_industry_ret20_mean | 307350 | 307350 | 1 | 1 | 2021-01-29 | 2026-04-30 |
| plus_industry_breadth_plus_fund_flow_plus_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_industry_ret60_mean_z | feature_industry_ret60_mean | 307350 | 307350 | 1 | 1 | 2021-01-29 | 2026-04-30 |
| plus_industry_breadth_plus_fund_flow_plus_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_industry_positive_ret20_ratio_z | feature_industry_positive_ret20_ratio | 307350 | 307350 | 1 | 1 | 2021-01-29 | 2026-04-30 |
| plus_industry_breadth_plus_fund_flow_plus_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_industry_amount20_mean_z | feature_industry_amount20_mean | 307350 | 307350 | 1 | 1 | 2021-01-29 | 2026-04-30 |
| plus_industry_breadth_plus_fund_flow_plus_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_industry_low_vol20_mean_z | feature_industry_low_vol20_mean | 307350 | 307350 | 1 | 1 | 2021-01-29 | 2026-04-30 |
| plus_industry_breadth_plus_fund_flow_plus_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_fund_flow_main_inflow_5d_z | feature_fund_flow_main_inflow_5d | 307350 | 35590 | 0.115796 | 0.127948 | 2023-10-31 | 2026-04-30 |
| plus_industry_breadth_plus_fund_flow_plus_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_fund_flow_main_inflow_10d_z | feature_fund_flow_main_inflow_10d | 307350 | 31196 | 0.1015 | 0.10964 | 2023-11-30 | 2026-04-30 |
| plus_industry_breadth_plus_fund_flow_plus_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_fund_flow_main_inflow_20d_z | feature_fund_flow_main_inflow_20d | 307350 | 31048 | 0.101018 | 0.109138 | 2023-11-30 | 2026-04-30 |
| plus_industry_breadth_plus_fund_flow_plus_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_fund_flow_super_inflow_10d_z | feature_fund_flow_super_inflow_10d | 307350 | 31196 | 0.1015 | 0.10964 | 2023-11-30 | 2026-04-30 |
| plus_industry_breadth_plus_fund_flow_plus_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_fund_flow_divergence_20d_z | feature_fund_flow_divergence_20d | 307350 | 31048 | 0.101018 | 0.109138 | 2023-11-30 | 2026-04-30 |
| plus_industry_breadth_plus_fund_flow_plus_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_fund_flow_main_inflow_streak_z | feature_fund_flow_main_inflow_streak | 307350 | 36162 | 0.117657 | 0.130348 | 2023-10-31 | 2026-04-30 |
| plus_industry_breadth_plus_fund_flow_plus_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_fundamental_pe_ttm_z | feature_fundamental_pe_ttm | 307350 | 304469 | 0.990626 | 0.999766 | 2021-01-29 | 2026-04-30 |
| plus_industry_breadth_plus_fund_flow_plus_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_fundamental_pb_z | feature_fundamental_pb | 307350 | 304469 | 0.990626 | 0.999766 | 2021-01-29 | 2026-04-30 |
| plus_industry_breadth_plus_fund_flow_plus_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_fundamental_ev_ebitda_z | feature_fundamental_ev_ebitda | 307350 | 0 | 0 | 0 |  |  |
| plus_industry_breadth_plus_fund_flow_plus_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_fundamental_roe_ttm_z | feature_fundamental_roe_ttm | 307350 | 304617 | 0.991108 | 0.992871 | 2021-01-29 | 2026-04-30 |
| plus_industry_breadth_plus_fund_flow_plus_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_fundamental_net_profit_yoy_z | feature_fundamental_net_profit_yoy | 307350 | 306367 | 0.996802 | 0.997055 | 2021-01-29 | 2026-04-30 |
| plus_industry_breadth_plus_fund_flow_plus_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_fundamental_gross_margin_change_z | feature_fundamental_gross_margin_change | 307350 | 299889 | 0.975725 | 0.96863 | 2021-01-29 | 2026-04-30 |
| plus_industry_breadth_plus_fund_flow_plus_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_fundamental_gross_margin_delta_z | feature_fundamental_gross_margin_delta | 307350 | 299765 | 0.975321 | 0.968386 | 2021-01-29 | 2026-04-30 |
| plus_industry_breadth_plus_fund_flow_plus_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_fundamental_debt_to_assets_change_z | feature_fundamental_debt_to_assets_change | 307350 | 306206 | 0.996278 | 0.997055 | 2021-01-29 | 2026-04-30 |
| plus_industry_breadth_plus_fund_flow_plus_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_fundamental_ocf_to_net_profit_z | feature_fundamental_ocf_to_net_profit | 307350 | 245847 | 0.799893 | 0.811349 | 2021-01-29 | 2026-04-30 |
| plus_industry_breadth_plus_fund_flow_plus_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_fundamental_ocf_to_asset_z | feature_fundamental_ocf_to_asset | 307350 | 306219 | 0.99632 | 0.996596 | 2021-01-29 | 2026-04-30 |
| plus_industry_breadth_plus_fund_flow_plus_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_fundamental_asset_turnover_z | feature_fundamental_asset_turnover | 307350 | 306300 | 0.996584 | 0.996911 | 2021-01-29 | 2026-04-30 |
| plus_industry_breadth_plus_fund_flow_plus_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_fundamental_net_margin_stability_z | feature_fundamental_net_margin_stability | 307350 | 306150 | 0.996096 | 0.996677 | 2021-01-29 | 2026-04-30 |
| plus_industry_breadth_plus_fund_flow_plus_fundamental_plus_shareholder | price_volume,industry_breadth,fund_flow,fundamental,shareholder | feature_ret_5d | feature_ret_5d | 307350 | 307017 | 0.998917 | 1 | 2021-01-29 | 2026-04-30 |
| plus_industry_breadth_plus_fund_flow_plus_fundamental_plus_shareholder | price_volume,industry_breadth,fund_flow,fundamental,shareholder | feature_ret_20d | feature_ret_20d | 307350 | 306083 | 0.995878 | 1 | 2021-01-29 | 2026-04-30 |
| plus_industry_breadth_plus_fund_flow_plus_fundamental_plus_shareholder | price_volume,industry_breadth,fund_flow,fundamental,shareholder | feature_ret_60d | feature_ret_60d | 307350 | 303506 | 0.987493 | 1 | 2021-01-29 | 2026-04-30 |
| plus_industry_breadth_plus_fund_flow_plus_fundamental_plus_shareholder | price_volume,industry_breadth,fund_flow,fundamental,shareholder | feature_realized_vol_20d | feature_realized_vol_20d | 307350 | 306673 | 0.997797 | 1 | 2021-01-29 | 2026-04-30 |
| plus_industry_breadth_plus_fund_flow_plus_fundamental_plus_shareholder | price_volume,industry_breadth,fund_flow,fundamental,shareholder | feature_amount_20d_log | feature_amount_20d_log | 307350 | 306745 | 0.998032 | 1 | 2021-01-29 | 2026-04-30 |
| plus_industry_breadth_plus_fund_flow_plus_fundamental_plus_shareholder | price_volume,industry_breadth,fund_flow,fundamental,shareholder | feature_turnover_20d | feature_turnover_20d | 307350 | 306745 | 0.998032 | 1 | 2021-01-29 | 2026-04-30 |
| plus_industry_breadth_plus_fund_flow_plus_fundamental_plus_shareholder | price_volume,industry_breadth,fund_flow,fundamental,shareholder | feature_price_position_250d | feature_price_position_250d | 307350 | 303573 | 0.987711 | 1 | 2021-01-29 | 2026-04-30 |
| plus_industry_breadth_plus_fund_flow_plus_fundamental_plus_shareholder | price_volume,industry_breadth,fund_flow,fundamental,shareholder | feature_limit_move_hits_20d | feature_limit_move_hits_20d | 307350 | 307350 | 1 | 1 | 2021-01-29 | 2026-04-30 |
| plus_industry_breadth_plus_fund_flow_plus_fundamental_plus_shareholder | price_volume,industry_breadth,fund_flow,fundamental,shareholder | feature_industry_ret20_mean_z | feature_industry_ret20_mean | 307350 | 307350 | 1 | 1 | 2021-01-29 | 2026-04-30 |
| plus_industry_breadth_plus_fund_flow_plus_fundamental_plus_shareholder | price_volume,industry_breadth,fund_flow,fundamental,shareholder | feature_industry_ret60_mean_z | feature_industry_ret60_mean | 307350 | 307350 | 1 | 1 | 2021-01-29 | 2026-04-30 |
| plus_industry_breadth_plus_fund_flow_plus_fundamental_plus_shareholder | price_volume,industry_breadth,fund_flow,fundamental,shareholder | feature_industry_positive_ret20_ratio_z | feature_industry_positive_ret20_ratio | 307350 | 307350 | 1 | 1 | 2021-01-29 | 2026-04-30 |
| plus_industry_breadth_plus_fund_flow_plus_fundamental_plus_shareholder | price_volume,industry_breadth,fund_flow,fundamental,shareholder | feature_industry_amount20_mean_z | feature_industry_amount20_mean | 307350 | 307350 | 1 | 1 | 2021-01-29 | 2026-04-30 |
| plus_industry_breadth_plus_fund_flow_plus_fundamental_plus_shareholder | price_volume,industry_breadth,fund_flow,fundamental,shareholder | feature_industry_low_vol20_mean_z | feature_industry_low_vol20_mean | 307350 | 307350 | 1 | 1 | 2021-01-29 | 2026-04-30 |
| plus_industry_breadth_plus_fund_flow_plus_fundamental_plus_shareholder | price_volume,industry_breadth,fund_flow,fundamental,shareholder | feature_fund_flow_main_inflow_5d_z | feature_fund_flow_main_inflow_5d | 307350 | 35590 | 0.115796 | 0.127948 | 2023-10-31 | 2026-04-30 |
| plus_industry_breadth_plus_fund_flow_plus_fundamental_plus_shareholder | price_volume,industry_breadth,fund_flow,fundamental,shareholder | feature_fund_flow_main_inflow_10d_z | feature_fund_flow_main_inflow_10d | 307350 | 31196 | 0.1015 | 0.10964 | 2023-11-30 | 2026-04-30 |
| plus_industry_breadth_plus_fund_flow_plus_fundamental_plus_shareholder | price_volume,industry_breadth,fund_flow,fundamental,shareholder | feature_fund_flow_main_inflow_20d_z | feature_fund_flow_main_inflow_20d | 307350 | 31048 | 0.101018 | 0.109138 | 2023-11-30 | 2026-04-30 |
| plus_industry_breadth_plus_fund_flow_plus_fundamental_plus_shareholder | price_volume,industry_breadth,fund_flow,fundamental,shareholder | feature_fund_flow_super_inflow_10d_z | feature_fund_flow_super_inflow_10d | 307350 | 31196 | 0.1015 | 0.10964 | 2023-11-30 | 2026-04-30 |

## Year Slice

| candidate_pool_version | model | top_k | year | months | mean_topk_return | mean_topk_excess | median_topk_excess | hit_rate | mean_topk_minus_nextk |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| U1_liquid_tradable | B0_market_ew | 20 | 2022 | 12 | 0.00301257 | 0 | 0 | 0 |  |
| U1_liquid_tradable | B0_market_ew | 20 | 2023 | 12 | -0.0185141 | 0 | 0 | 0 |  |
| U1_liquid_tradable | B0_market_ew | 20 | 2024 | 12 | -0.00586991 | 0 | 0 | 0 |  |
| U1_liquid_tradable | B0_market_ew | 20 | 2025 | 12 | 0.0329754 | 0 | 0 | 0 |  |
| U1_liquid_tradable | B0_market_ew | 20 | 2026 | 3 | 0.00339507 | 0 | 0 | 0 |  |
| U1_liquid_tradable | M5_plus_industry_breadth_elasticnet_excess | 20 | 2022 | 12 | -0.00814876 | -0.0111613 | -0.00530433 | 0.333333 | -0.00368132 |
| U1_liquid_tradable | M5_plus_industry_breadth_elasticnet_excess | 20 | 2023 | 12 | 0.00495145 | 0.0234655 | 0.00340319 | 0.583333 | 0.0237735 |
| U1_liquid_tradable | M5_plus_industry_breadth_elasticnet_excess | 20 | 2024 | 12 | -0.00144167 | 0.00442824 | 0.00975158 | 0.666667 | 0.0207242 |
| U1_liquid_tradable | M5_plus_industry_breadth_elasticnet_excess | 20 | 2025 | 12 | 0.0568005 | 0.0238252 | 0.0282377 | 0.666667 | -0.0013508 |
| U1_liquid_tradable | M5_plus_industry_breadth_elasticnet_excess | 20 | 2026 | 3 | -0.013171 | -0.0165661 | -0.00343948 | 0 | -0.0322756 |
| U1_liquid_tradable | M5_plus_industry_breadth_plus_fund_flow_elasticnet_excess | 20 | 2022 | 12 | -0.00814876 | -0.0111613 | -0.00530433 | 0.333333 | -0.00368132 |
| U1_liquid_tradable | M5_plus_industry_breadth_plus_fund_flow_elasticnet_excess | 20 | 2023 | 12 | 0.00495145 | 0.0234655 | 0.00340319 | 0.583333 | 0.0237735 |
| U1_liquid_tradable | M5_plus_industry_breadth_plus_fund_flow_elasticnet_excess | 20 | 2024 | 12 | -0.00144167 | 0.00442824 | 0.00975158 | 0.666667 | 0.0207242 |
| U1_liquid_tradable | M5_plus_industry_breadth_plus_fund_flow_elasticnet_excess | 20 | 2025 | 12 | 0.0610986 | 0.0281233 | 0.0205282 | 0.666667 | 0.00600643 |
| U1_liquid_tradable | M5_plus_industry_breadth_plus_fund_flow_elasticnet_excess | 20 | 2026 | 3 | 0.0261406 | 0.0227455 | 0.0190268 | 1 | 0.017092 |
| U1_liquid_tradable | M5_plus_industry_breadth_plus_fund_flow_plus_fundamental_elasticnet_excess | 20 | 2022 | 12 | -0.00507304 | -0.00808561 | -0.0110066 | 0.416667 | -1.87699e-05 |
| U1_liquid_tradable | M5_plus_industry_breadth_plus_fund_flow_plus_fundamental_elasticnet_excess | 20 | 2023 | 12 | -0.0148358 | 0.00367831 | -0.00361558 | 0.416667 | -0.020692 |
| U1_liquid_tradable | M5_plus_industry_breadth_plus_fund_flow_plus_fundamental_elasticnet_excess | 20 | 2024 | 12 | -0.000963077 | 0.00490683 | 0.00319044 | 0.583333 | 0.0140955 |
| U1_liquid_tradable | M5_plus_industry_breadth_plus_fund_flow_plus_fundamental_elasticnet_excess | 20 | 2025 | 12 | 0.0654196 | 0.0324443 | 0.0186328 | 0.75 | 0.0151419 |
| U1_liquid_tradable | M5_plus_industry_breadth_plus_fund_flow_plus_fundamental_elasticnet_excess | 20 | 2026 | 3 | 0.0176021 | 0.0142071 | 0.00858816 | 0.666667 | 0.00100368 |
| U1_liquid_tradable | M5_plus_industry_breadth_plus_fund_flow_plus_fundamental_plus_shareholder_elasticnet_excess | 20 | 2022 | 12 | 0.00490471 | 0.00189214 | 0.0149363 | 0.583333 | 0.0120502 |
| U1_liquid_tradable | M5_plus_industry_breadth_plus_fund_flow_plus_fundamental_plus_shareholder_elasticnet_excess | 20 | 2023 | 12 | -0.00770399 | 0.0108101 | -0.00124526 | 0.5 | -0.00354508 |
| U1_liquid_tradable | M5_plus_industry_breadth_plus_fund_flow_plus_fundamental_plus_shareholder_elasticnet_excess | 20 | 2024 | 12 | 0.00254012 | 0.00841003 | 0.0112421 | 0.666667 | 0.00137776 |
| U1_liquid_tradable | M5_plus_industry_breadth_plus_fund_flow_plus_fundamental_plus_shareholder_elasticnet_excess | 20 | 2025 | 12 | 0.0667262 | 0.0337508 | 0.0330856 | 0.666667 | 0.0245293 |
| U1_liquid_tradable | M5_plus_industry_breadth_plus_fund_flow_plus_fundamental_plus_shareholder_elasticnet_excess | 20 | 2026 | 3 | 0.026084 | 0.0226889 | 0.0229303 | 0.666667 | 0.034414 |
| U1_liquid_tradable | M5_plus_industry_breadth_plus_fund_flow_plus_fundamental_plus_shareholder_plus_margin_trading_elasticnet_excess | 20 | 2022 | 12 | 0.00781013 | 0.00479756 | 0.0111419 | 0.666667 | 0.0133237 |
| U1_liquid_tradable | M5_plus_industry_breadth_plus_fund_flow_plus_fundamental_plus_shareholder_plus_margin_trading_elasticnet_excess | 20 | 2023 | 12 | -0.00540378 | 0.0131103 | 0.00309053 | 0.583333 | 0.000774585 |
| U1_liquid_tradable | M5_plus_industry_breadth_plus_fund_flow_plus_fundamental_plus_shareholder_plus_margin_trading_elasticnet_excess | 20 | 2024 | 12 | -0.000208994 | 0.00566091 | 0.00236705 | 0.583333 | -0.00174951 |
| U1_liquid_tradable | M5_plus_industry_breadth_plus_fund_flow_plus_fundamental_plus_shareholder_plus_margin_trading_elasticnet_excess | 20 | 2025 | 12 | 0.0660028 | 0.0330275 | 0.0299818 | 0.666667 | 0.0294311 |
| U1_liquid_tradable | M5_plus_industry_breadth_plus_fund_flow_plus_fundamental_plus_shareholder_plus_margin_trading_elasticnet_excess | 20 | 2026 | 3 | 0.0264846 | 0.0230896 | 0.0241321 | 0.666667 | 0.0358784 |
| U1_liquid_tradable | M5_price_volume_only_elasticnet_excess | 20 | 2022 | 12 | 0.00218411 | -0.000828465 | -0.0110782 | 0.416667 | -0.00287826 |
| U1_liquid_tradable | M5_price_volume_only_elasticnet_excess | 20 | 2023 | 12 | -0.00410076 | 0.0144133 | 0.0118594 | 0.666667 | -0.00312713 |
| U1_liquid_tradable | M5_price_volume_only_elasticnet_excess | 20 | 2024 | 12 | 0.00511674 | 0.0109866 | 0.00471615 | 0.666667 | 0.0208991 |
| U1_liquid_tradable | M5_price_volume_only_elasticnet_excess | 20 | 2025 | 12 | 0.0667633 | 0.033788 | 0.0202027 | 0.75 | 0.029919 |
| U1_liquid_tradable | M5_price_volume_only_elasticnet_excess | 20 | 2026 | 3 | -0.000806072 | -0.00420114 | -0.00251698 | 0.333333 | -0.0119042 |

## Realized Market State Slice

| candidate_pool_version | model | top_k | realized_market_state | months | mean_topk_return | mean_topk_excess | median_topk_excess | hit_rate | mean_topk_minus_nextk |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| U1_liquid_tradable | B0_market_ew | 20 | neutral | 29 | 0.00418018 | 0 | 0 | 0 |  |
| U1_liquid_tradable | B0_market_ew | 20 | strong_down | 11 | -0.0748173 | 0 | 0 | 0 |  |
| U1_liquid_tradable | B0_market_ew | 20 | strong_up | 11 | 0.0773816 | 0 | 0 | 0 |  |
| U1_liquid_tradable | M5_plus_industry_breadth_elasticnet_excess | 20 | neutral | 29 | 0.016213 | 0.0120329 | 0.0110535 | 0.655172 | 0.0107953 |
| U1_liquid_tradable | M5_plus_industry_breadth_elasticnet_excess | 20 | strong_down | 11 | -0.0673085 | 0.00750879 | -0.00343948 | 0.454545 | -0.000361365 |
| U1_liquid_tradable | M5_plus_industry_breadth_elasticnet_excess | 20 | strong_up | 11 | 0.0778764 | 0.000494843 | -0.00789027 | 0.272727 | 0.00615198 |
| U1_liquid_tradable | M5_plus_industry_breadth_plus_fund_flow_elasticnet_excess | 20 | neutral | 29 | 0.0199766 | 0.0157964 | 0.0126042 | 0.724138 | 0.0155981 |
| U1_liquid_tradable | M5_plus_industry_breadth_plus_fund_flow_elasticnet_excess | 20 | strong_down | 11 | -0.0652661 | 0.00955118 | 0.00107566 | 0.545455 | 0.00144425 |
| U1_liquid_tradable | M5_plus_industry_breadth_plus_fund_flow_elasticnet_excess | 20 | strong_up | 11 | 0.0813222 | 0.00394064 | -0.00789027 | 0.272727 | 0.0131744 |
| U1_liquid_tradable | M5_plus_industry_breadth_plus_fund_flow_plus_fundamental_elasticnet_excess | 20 | neutral | 29 | 0.0193762 | 0.0151961 | 0.00644385 | 0.62069 | 0.00710351 |
| U1_liquid_tradable | M5_plus_industry_breadth_plus_fund_flow_plus_fundamental_elasticnet_excess | 20 | strong_down | 11 | -0.0624749 | 0.0123423 | 0.014084 | 0.636364 | 0.00418987 |
| U1_liquid_tradable | M5_plus_industry_breadth_plus_fund_flow_plus_fundamental_elasticnet_excess | 20 | strong_up | 11 | 0.0647902 | -0.0125913 | -0.0124275 | 0.272727 | -0.0133418 |
| U1_liquid_tradable | M5_plus_industry_breadth_plus_fund_flow_plus_fundamental_plus_shareholder_elasticnet_excess | 20 | neutral | 29 | 0.0220395 | 0.0178593 | 0.0174908 | 0.655172 | 0.00838624 |
| U1_liquid_tradable | M5_plus_industry_breadth_plus_fund_flow_plus_fundamental_plus_shareholder_elasticnet_excess | 20 | strong_down | 11 | -0.0464969 | 0.0283204 | 0.0242076 | 0.818182 | 0.0209327 |
| U1_liquid_tradable | M5_plus_industry_breadth_plus_fund_flow_plus_fundamental_plus_shareholder_elasticnet_excess | 20 | strong_up | 11 | 0.0680161 | -0.0093655 | -0.0353133 | 0.272727 | 0.00388431 |
| U1_liquid_tradable | M5_plus_industry_breadth_plus_fund_flow_plus_fundamental_plus_shareholder_plus_margin_trading_elasticnet_excess | 20 | neutral | 29 | 0.0208253 | 0.0166451 | 0.0127822 | 0.655172 | 0.00643587 |
| U1_liquid_tradable | M5_plus_industry_breadth_plus_fund_flow_plus_fundamental_plus_shareholder_plus_margin_trading_elasticnet_excess | 20 | strong_down | 11 | -0.0454701 | 0.0293472 | 0.0225001 | 0.909091 | 0.0251158 |
| U1_liquid_tradable | M5_plus_industry_breadth_plus_fund_flow_plus_fundamental_plus_shareholder_plus_margin_trading_elasticnet_excess | 20 | strong_up | 11 | 0.0721902 | -0.00519134 | -0.0266754 | 0.272727 | 0.0132799 |
| U1_liquid_tradable | M5_price_volume_only_elasticnet_excess | 20 | neutral | 29 | 0.0230734 | 0.0188932 | 0.0171706 | 0.724138 | 0.0126708 |
| U1_liquid_tradable | M5_price_volume_only_elasticnet_excess | 20 | strong_down | 11 | -0.0808196 | -0.00600236 | -0.0092517 | 0.363636 | -0.00831511 |
| U1_liquid_tradable | M5_price_volume_only_elasticnet_excess | 20 | strong_up | 11 | 0.0960937 | 0.0187121 | 0.00142474 | 0.545455 | 0.0205504 |

## 口径

- `price_volume_only` 是 M5 内部 baseline；后续 spec 按 industry_breadth -> fund_flow -> fundamental -> shareholder 累积加入。
- `topk_excess_after_cost` 使用半 L1 月度换手乘以 cost_bps 的简化成本敏感性。

## 本轮产物

- `data/results/m5_margin_delta_full_2026-05-06_summary.json`
- `data/results/m5_margin_delta_full_2026-05-06_leaderboard.csv`
- `data/results/m5_margin_delta_full_2026-05-06_incremental_delta.csv`
- `data/results/m5_margin_delta_full_2026-05-06_monthly_long.csv`
- `data/results/m5_margin_delta_full_2026-05-06_rank_ic.csv`
- `data/results/m5_margin_delta_full_2026-05-06_quantile_spread.csv`
- `data/results/m5_margin_delta_full_2026-05-06_feature_coverage.csv`
- `data/results/m5_margin_delta_full_2026-05-06_feature_importance.csv`
- `data/results/m5_margin_delta_full_2026-05-06_topk_holdings.csv`
- `data/results/m5_margin_delta_full_2026-05-06_industry_exposure.csv`
- `data/results/m5_margin_delta_full_2026-05-06_candidate_pool_width.csv`
- `data/results/m5_margin_delta_full_2026-05-06_candidate_pool_reject_reason.csv`
- `data/results/m5_margin_delta_full_2026-05-06_year_slice.csv`
- `data/results/m5_margin_delta_full_2026-05-06_regime_slice.csv`
- `data/results/m5_margin_delta_full_2026-05-06_market_states.csv`
- `data/results/m5_margin_delta_full_2026-05-06_manifest.json`
