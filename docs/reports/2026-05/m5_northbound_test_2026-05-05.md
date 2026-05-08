# Monthly Selection M5 Multisource

- 生成时间：`2026-05-05T11:57:29.745126+00:00` · 输出 stem：`m5_northbound_test_2026-05-05`
- 数据集：`data/cache/monthly_selection_features.parquet` · 数据库：`data/market.duckdb`
- 有效标签月份：`63` · 单窗训练行上限：`0`

## Leaderboard

| candidate_pool_version | model | model_type | top_k | months | mean_topk_return | median_topk_return | topk_excess_mean | topk_excess_median | topk_hit_rate | topk_minus_nextk_mean | industry_neutral_topk_excess_mean | turnover_mean | cost_drag_mean | topk_excess_after_cost_mean | topk_excess_annualized | rank_ic_mean | rank_ic_std | rank_ic_months | rank_ic_ir | quantile_top_minus_bottom_mean | neutral_median_excess | strong_down_median_excess | strong_up_median_excess |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| U1_liquid_tradable | M5_price_volume_only_elasticnet_excess | elasticnet | 20 | 51 | 0.0164146 | 0.0207779 | 0.0134845 | 0.00537175 | 0.607843 | 0.00984394 | 0.0146953 | 0.965 | 0.000965 | 0.0134428 | 0.174371 | 0.0993018 | 0.111158 | 51 | 0.89334 | 0.0164281 | 0.0171706 | -0.0092517 | 0.00142474 |
| U1_liquid_tradable | M5_plus_industry_breadth_plus_fund_flow_plus_fundamental_plus_shareholder_elasticnet_excess | elasticnet | 20 | 51 | 0.0156364 | 0.0120354 | 0.0127063 | 0.0123819 | 0.607843 | 0.00648458 | 0.014669 | 0.864 | 0.000864 | 0.0114491 | 0.163596 | 0.0985103 | 0.118907 | 51 | 0.828466 | 0.0183683 | 0.0123819 | 0.0243762 | -0.0353133 |
| U1_liquid_tradable | M5_plus_industry_breadth_plus_fund_flow_plus_fundamental_plus_shareholder_plus_northbound_elasticnet_excess | elasticnet | 20 | 51 | 0.0151872 | 0.0120354 | 0.0122572 | 0.0123819 | 0.607843 | 0.0058041 | 0.0143298 | 0.864 | 0.000864 | 0.0110463 | 0.157418 | 0.0984508 | 0.118967 | 51 | 0.827545 | 0.0183141 | 0.0123819 | 0.0243762 | -0.0353133 |
| U1_liquid_tradable | M5_plus_industry_breadth_plus_fund_flow_elasticnet_excess | elasticnet | 20 | 51 | 0.0135271 | 0.00492588 | 0.010597 | 0.00573072 | 0.568627 | 0.0109257 | 0.0158474 | 0.963 | 0.000963 | 0.00940164 | 0.134844 | 0.0824021 | 0.108564 | 51 | 0.759019 | 0.0146831 | 0.0110535 | 0.00107566 | -0.00789027 |
| U1_liquid_tradable | M5_plus_industry_breadth_elasticnet_excess | elasticnet | 20 | 51 | 0.0114985 | 0.00458493 | 0.00856849 | 0.00315811 | 0.529412 | 0.00727017 | 0.0138756 | 0.965 | 0.000965 | 0.00733056 | 0.107809 | 0.0843895 | 0.111178 | 51 | 0.759051 | 0.0140006 | 0.0110535 | -0.00343948 | -0.00789027 |
| U1_liquid_tradable | M5_plus_industry_breadth_plus_fund_flow_plus_fundamental_elasticnet_excess | elasticnet | 20 | 51 | 0.010501 | 0.0140648 | 0.00757096 | 0.00340478 | 0.529412 | 0.000506987 | 0.0113233 | 0.903 | 0.000903 | 0.00608183 | 0.0947317 | 0.0905788 | 0.112185 | 51 | 0.807404 | 0.0171824 | 0.00359774 | 0.014084 | -0.0124275 |
| U1_liquid_tradable | B0_market_ew | benchmark | 20 | 51 | 0.00293005 | 0.00334796 | 0 | 0 | 0 |  |  |  | 0 | 0 | 0 |  |  |  |  |  | 0 | 0 | 0 |
| U2_risk_sane | M5_plus_industry_breadth_plus_fund_flow_plus_fundamental_plus_shareholder_elasticnet_excess | elasticnet | 20 | 51 | 0.0163821 | 0.0171179 | 0.0134521 | 0.0147682 | 0.647059 | 0.0120175 | 0.0167091 | 0.857 | 0.000857 | 0.0124576 | 0.17392 | 0.0786525 | 0.121918 | 51 | 0.645125 | 0.0145976 | 0.0147682 | 0.0271623 | -0.00358735 |
| U2_risk_sane | M5_plus_industry_breadth_plus_fund_flow_plus_fundamental_plus_shareholder_plus_northbound_elasticnet_excess | elasticnet | 20 | 51 | 0.0161506 | 0.0185667 | 0.0132206 | 0.0147682 | 0.627451 | 0.011976 | 0.0166221 | 0.859 | 0.000859 | 0.0121909 | 0.170707 | 0.0786356 | 0.121968 | 51 | 0.644723 | 0.014537 | 0.0147682 | 0.0271623 | -0.00358735 |
| U2_risk_sane | M5_price_volume_only_elasticnet_excess | elasticnet | 20 | 51 | 0.0140248 | 0.0170118 | 0.0110948 | 0.000384803 | 0.509804 | 0.0066605 | 0.0122597 | 0.963 | 0.000963 | 0.0108536 | 0.141569 | 0.0743727 | 0.120259 | 51 | 0.618437 | 0.0118992 | 0.0225659 | -0.00533903 | -0.00628677 |
| U2_risk_sane | M5_plus_industry_breadth_plus_fund_flow_elasticnet_excess | elasticnet | 20 | 51 | 0.0121518 | 0.014879 | 0.00922171 | 0.00614479 | 0.627451 | 0.00389615 | 0.0130765 | 0.953 | 0.000953 | 0.00784465 | 0.116449 | 0.0602128 | 0.11254 | 51 | 0.535035 | 0.00998558 | 0.010439 | 0.00709764 | -0.0132587 |
| U2_risk_sane | M5_plus_industry_breadth_plus_fund_flow_plus_fundamental_elasticnet_excess | elasticnet | 20 | 51 | 0.0118968 | 0.00677598 | 0.00896677 | 0.0118677 | 0.54902 | 0.00459319 | 0.0133625 | 0.89 | 0.00089 | 0.00777649 | 0.11307 | 0.0694903 | 0.112288 | 51 | 0.618858 | 0.0129372 | 0.0121136 | 0.0282173 | -0.0176399 |
| U2_risk_sane | M5_plus_industry_breadth_elasticnet_excess | elasticnet | 20 | 51 | 0.0097622 | 0.00518971 | 0.00683215 | 0.00520474 | 0.588235 | 0.00248781 | 0.0105585 | 0.957 | 0.000957 | 0.00540331 | 0.0851379 | 0.060577 | 0.116983 | 51 | 0.517828 | 0.00926042 | 0.010439 | 0.00709764 | -0.0202538 |
| U2_risk_sane | B0_market_ew | benchmark | 20 | 51 | 0.00293005 | 0.00334796 | 0 | 0 | 0 |  |  |  | 0 | 0 | 0 |  |  |  |  |  | 0 | 0 | 0 |

## Incremental Delta vs Price-Volume

| candidate_pool_version | model | model_type | top_k | months | mean_topk_return | median_topk_return | topk_excess_mean | topk_excess_median | topk_hit_rate | topk_minus_nextk_mean | industry_neutral_topk_excess_mean | turnover_mean | cost_drag_mean | topk_excess_after_cost_mean | topk_excess_annualized | rank_ic_mean | rank_ic_std | rank_ic_months | rank_ic_ir | quantile_top_minus_bottom_mean | neutral_median_excess | strong_down_median_excess | strong_up_median_excess | feature_spec | base_model | baseline_topk_excess_mean | baseline_topk_excess_after_cost_mean | baseline_rank_ic_mean | baseline_quantile_top_minus_bottom_mean | delta_topk_excess_mean | delta_topk_excess_after_cost_mean | delta_rank_ic_mean | delta_quantile_top_minus_bottom_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| U1_liquid_tradable | M5_plus_industry_breadth_plus_fund_flow_plus_fundamental_plus_shareholder_elasticnet_excess | elasticnet | 20 | 51 | 0.0156364 | 0.0120354 | 0.0127063 | 0.0123819 | 0.607843 | 0.00648458 | 0.014669 | 0.864 | 0.000864 | 0.0114491 | 0.163596 | 0.0985103 | 0.118907 | 51 | 0.828466 | 0.0183683 | 0.0123819 | 0.0243762 | -0.0353133 | plus_industry_breadth_plus_fund_flow_plus_fundamental_plus_shareholder | elasticnet_excess | 0.0134845 | 0.0134428 | 0.0993018 | 0.0164281 | -0.000778193 | -0.00199372 | -0.000791536 | 0.0019402 |
| U1_liquid_tradable | M5_plus_industry_breadth_plus_fund_flow_plus_fundamental_plus_shareholder_plus_northbound_elasticnet_excess | elasticnet | 20 | 51 | 0.0151872 | 0.0120354 | 0.0122572 | 0.0123819 | 0.607843 | 0.0058041 | 0.0143298 | 0.864 | 0.000864 | 0.0110463 | 0.157418 | 0.0984508 | 0.118967 | 51 | 0.827545 | 0.0183141 | 0.0123819 | 0.0243762 | -0.0353133 | plus_industry_breadth_plus_fund_flow_plus_fundamental_plus_shareholder_plus_northbound | elasticnet_excess | 0.0134845 | 0.0134428 | 0.0993018 | 0.0164281 | -0.00122736 | -0.00239644 | -0.000850981 | 0.00188598 |
| U1_liquid_tradable | M5_plus_industry_breadth_plus_fund_flow_elasticnet_excess | elasticnet | 20 | 51 | 0.0135271 | 0.00492588 | 0.010597 | 0.00573072 | 0.568627 | 0.0109257 | 0.0158474 | 0.963 | 0.000963 | 0.00940164 | 0.134844 | 0.0824021 | 0.108564 | 51 | 0.759019 | 0.0146831 | 0.0110535 | 0.00107566 | -0.00789027 | plus_industry_breadth_plus_fund_flow | elasticnet_excess | 0.0134845 | 0.0134428 | 0.0993018 | 0.0164281 | -0.00288752 | -0.00404115 | -0.0168997 | -0.001745 |
| U1_liquid_tradable | M5_plus_industry_breadth_elasticnet_excess | elasticnet | 20 | 51 | 0.0114985 | 0.00458493 | 0.00856849 | 0.00315811 | 0.529412 | 0.00727017 | 0.0138756 | 0.965 | 0.000965 | 0.00733056 | 0.107809 | 0.0843895 | 0.111178 | 51 | 0.759051 | 0.0140006 | 0.0110535 | -0.00343948 | -0.00789027 | plus_industry_breadth | elasticnet_excess | 0.0134845 | 0.0134428 | 0.0993018 | 0.0164281 | -0.00491602 | -0.00611223 | -0.0149124 | -0.00242751 |
| U1_liquid_tradable | M5_plus_industry_breadth_plus_fund_flow_plus_fundamental_elasticnet_excess | elasticnet | 20 | 51 | 0.010501 | 0.0140648 | 0.00757096 | 0.00340478 | 0.529412 | 0.000506987 | 0.0113233 | 0.903 | 0.000903 | 0.00608183 | 0.0947317 | 0.0905788 | 0.112185 | 51 | 0.807404 | 0.0171824 | 0.00359774 | 0.014084 | -0.0124275 | plus_industry_breadth_plus_fund_flow_plus_fundamental | elasticnet_excess | 0.0134845 | 0.0134428 | 0.0993018 | 0.0164281 | -0.00591356 | -0.00736096 | -0.008723 | 0.000754289 |
| U2_risk_sane | M5_plus_industry_breadth_plus_fund_flow_plus_fundamental_plus_shareholder_elasticnet_excess | elasticnet | 20 | 51 | 0.0163821 | 0.0171179 | 0.0134521 | 0.0147682 | 0.647059 | 0.0120175 | 0.0167091 | 0.857 | 0.000857 | 0.0124576 | 0.17392 | 0.0786525 | 0.121918 | 51 | 0.645125 | 0.0145976 | 0.0147682 | 0.0271623 | -0.00358735 | plus_industry_breadth_plus_fund_flow_plus_fundamental_plus_shareholder | elasticnet_excess | 0.0110948 | 0.0108536 | 0.0743727 | 0.0118992 | 0.00235731 | 0.00160404 | 0.00427976 | 0.00269842 |
| U2_risk_sane | M5_plus_industry_breadth_plus_fund_flow_plus_fundamental_plus_shareholder_plus_northbound_elasticnet_excess | elasticnet | 20 | 51 | 0.0161506 | 0.0185667 | 0.0132206 | 0.0147682 | 0.627451 | 0.011976 | 0.0166221 | 0.859 | 0.000859 | 0.0121909 | 0.170707 | 0.0786356 | 0.121968 | 51 | 0.644723 | 0.014537 | 0.0147682 | 0.0271623 | -0.00358735 | plus_industry_breadth_plus_fund_flow_plus_fundamental_plus_shareholder_plus_northbound | elasticnet_excess | 0.0110948 | 0.0108536 | 0.0743727 | 0.0118992 | 0.00212582 | 0.00133735 | 0.00426292 | 0.00263778 |
| U2_risk_sane | M5_plus_industry_breadth_plus_fund_flow_elasticnet_excess | elasticnet | 20 | 51 | 0.0121518 | 0.014879 | 0.00922171 | 0.00614479 | 0.627451 | 0.00389615 | 0.0130765 | 0.953 | 0.000953 | 0.00784465 | 0.116449 | 0.0602128 | 0.11254 | 51 | 0.535035 | 0.00998558 | 0.010439 | 0.00709764 | -0.0132587 | plus_industry_breadth_plus_fund_flow | elasticnet_excess | 0.0110948 | 0.0108536 | 0.0743727 | 0.0118992 | -0.00187306 | -0.00300893 | -0.0141599 | -0.0019136 |
| U2_risk_sane | M5_plus_industry_breadth_plus_fund_flow_plus_fundamental_elasticnet_excess | elasticnet | 20 | 51 | 0.0118968 | 0.00677598 | 0.00896677 | 0.0118677 | 0.54902 | 0.00459319 | 0.0133625 | 0.89 | 0.00089 | 0.00777649 | 0.11307 | 0.0694903 | 0.112288 | 51 | 0.618858 | 0.0129372 | 0.0121136 | 0.0282173 | -0.0176399 | plus_industry_breadth_plus_fund_flow_plus_fundamental | elasticnet_excess | 0.0110948 | 0.0108536 | 0.0743727 | 0.0118992 | -0.002128 | -0.00307709 | -0.00488239 | 0.00103807 |
| U2_risk_sane | M5_plus_industry_breadth_elasticnet_excess | elasticnet | 20 | 51 | 0.0097622 | 0.00518971 | 0.00683215 | 0.00520474 | 0.588235 | 0.00248781 | 0.0105585 | 0.957 | 0.000957 | 0.00540331 | 0.0851379 | 0.060577 | 0.116983 | 51 | 0.517828 | 0.00926042 | 0.010439 | 0.00709764 | -0.0202538 | plus_industry_breadth | elasticnet_excess | 0.0110948 | 0.0108536 | 0.0743727 | 0.0118992 | -0.00426261 | -0.00545027 | -0.0137957 | -0.00263876 |

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
| U1_liquid_tradable | M5_plus_industry_breadth_elasticnet_excess | 20 | 2026 | 3 | -0.013171 | -0.0165661 | -0.00343948 | 0 | -0.0342696 |
| U1_liquid_tradable | M5_plus_industry_breadth_plus_fund_flow_elasticnet_excess | 20 | 2022 | 12 | -0.00814876 | -0.0111613 | -0.00530433 | 0.333333 | -0.00368132 |
| U1_liquid_tradable | M5_plus_industry_breadth_plus_fund_flow_elasticnet_excess | 20 | 2023 | 12 | 0.00495145 | 0.0234655 | 0.00340319 | 0.583333 | 0.0237735 |
| U1_liquid_tradable | M5_plus_industry_breadth_plus_fund_flow_elasticnet_excess | 20 | 2024 | 12 | -0.00144167 | 0.00442824 | 0.00975158 | 0.666667 | 0.0207242 |
| U1_liquid_tradable | M5_plus_industry_breadth_plus_fund_flow_elasticnet_excess | 20 | 2025 | 12 | 0.0570864 | 0.0241111 | 0.0211234 | 0.666667 | 0.00404453 |
| U1_liquid_tradable | M5_plus_industry_breadth_plus_fund_flow_elasticnet_excess | 20 | 2026 | 3 | 0.0201701 | 0.0167751 | 0.012799 | 0.666667 | 0.00629312 |
| U1_liquid_tradable | M5_plus_industry_breadth_plus_fund_flow_plus_fundamental_elasticnet_excess | 20 | 2022 | 12 | -0.00440621 | -0.00741878 | -0.0110066 | 0.416667 | 0.0013149 |
| U1_liquid_tradable | M5_plus_industry_breadth_plus_fund_flow_plus_fundamental_elasticnet_excess | 20 | 2023 | 12 | -0.0148358 | 0.00367831 | -0.00361558 | 0.416667 | -0.0203587 |
| U1_liquid_tradable | M5_plus_industry_breadth_plus_fund_flow_plus_fundamental_elasticnet_excess | 20 | 2024 | 12 | 0.000514319 | 0.00638423 | 0.00319044 | 0.583333 | 0.0162819 |
| U1_liquid_tradable | M5_plus_industry_breadth_plus_fund_flow_plus_fundamental_elasticnet_excess | 20 | 2025 | 12 | 0.0590449 | 0.0260696 | 0.0186328 | 0.666667 | 0.00395288 |
| U1_liquid_tradable | M5_plus_industry_breadth_plus_fund_flow_plus_fundamental_elasticnet_excess | 20 | 2026 | 3 | 0.017248 | 0.013853 | 0.00858816 | 0.666667 | 0.00385482 |
| U1_liquid_tradable | M5_plus_industry_breadth_plus_fund_flow_plus_fundamental_plus_shareholder_elasticnet_excess | 20 | 2022 | 12 | 0.00449711 | 0.00148454 | 0.0149363 | 0.583333 | 0.0119671 |
| U1_liquid_tradable | M5_plus_industry_breadth_plus_fund_flow_plus_fundamental_plus_shareholder_elasticnet_excess | 20 | 2023 | 12 | -0.00770399 | 0.0108101 | -0.00124526 | 0.5 | -0.00354508 |
| U1_liquid_tradable | M5_plus_industry_breadth_plus_fund_flow_plus_fundamental_plus_shareholder_elasticnet_excess | 20 | 2024 | 12 | 0.00348238 | 0.00935228 | 0.0112421 | 0.666667 | 0.00162973 |
| U1_liquid_tradable | M5_plus_industry_breadth_plus_fund_flow_plus_fundamental_plus_shareholder_elasticnet_excess | 20 | 2025 | 12 | 0.058755 | 0.0257796 | 0.020464 | 0.666667 | 0.00627884 |
| U1_liquid_tradable | M5_plus_industry_breadth_plus_fund_flow_plus_fundamental_plus_shareholder_elasticnet_excess | 20 | 2026 | 3 | 0.0296963 | 0.0263013 | 0.0336801 | 0.666667 | 0.0449157 |
| U1_liquid_tradable | M5_plus_industry_breadth_plus_fund_flow_plus_fundamental_plus_shareholder_plus_northbound_elasticnet_excess | 20 | 2022 | 12 | 0.00300665 | -5.91887e-06 | 0.0149363 | 0.583333 | 0.00928699 |
| U1_liquid_tradable | M5_plus_industry_breadth_plus_fund_flow_plus_fundamental_plus_shareholder_plus_northbound_elasticnet_excess | 20 | 2023 | 12 | -0.00770399 | 0.0108101 | -0.00124526 | 0.5 | -0.0030706 |
| U1_liquid_tradable | M5_plus_industry_breadth_plus_fund_flow_plus_fundamental_plus_shareholder_plus_northbound_elasticnet_excess | 20 | 2024 | 12 | 0.00348238 | 0.00935228 | 0.0112421 | 0.666667 | 0.00162973 |
| U1_liquid_tradable | M5_plus_industry_breadth_plus_fund_flow_plus_fundamental_plus_shareholder_plus_northbound_elasticnet_excess | 20 | 2025 | 12 | 0.058755 | 0.0257796 | 0.020464 | 0.666667 | 0.00627884 |
| U1_liquid_tradable | M5_plus_industry_breadth_plus_fund_flow_plus_fundamental_plus_shareholder_plus_northbound_elasticnet_excess | 20 | 2026 | 3 | 0.0280224 | 0.0246273 | 0.0336801 | 0.666667 | 0.0421698 |
| U1_liquid_tradable | M5_price_volume_only_elasticnet_excess | 20 | 2022 | 12 | 0.00218411 | -0.000828465 | -0.0110782 | 0.416667 | -0.00287826 |
| U1_liquid_tradable | M5_price_volume_only_elasticnet_excess | 20 | 2023 | 12 | -0.00410076 | 0.0144133 | 0.0118594 | 0.666667 | -0.00312713 |
| U1_liquid_tradable | M5_price_volume_only_elasticnet_excess | 20 | 2024 | 12 | 0.00511674 | 0.0109866 | 0.00471615 | 0.666667 | 0.0208991 |
| U1_liquid_tradable | M5_price_volume_only_elasticnet_excess | 20 | 2025 | 12 | 0.0667633 | 0.033788 | 0.0202027 | 0.75 | 0.029919 |
| U1_liquid_tradable | M5_price_volume_only_elasticnet_excess | 20 | 2026 | 3 | -0.000806072 | -0.00420114 | -0.00251698 | 0.333333 | -0.0119042 |
| U2_risk_sane | B0_market_ew | 20 | 2022 | 12 | 0.00301257 | 0 | 0 | 0 |  |
| U2_risk_sane | B0_market_ew | 20 | 2023 | 12 | -0.0185141 | 0 | 0 | 0 |  |
| U2_risk_sane | B0_market_ew | 20 | 2024 | 12 | -0.00586991 | 0 | 0 | 0 |  |
| U2_risk_sane | B0_market_ew | 20 | 2025 | 12 | 0.0329754 | 0 | 0 | 0 |  |
| U2_risk_sane | B0_market_ew | 20 | 2026 | 3 | 0.00339507 | 0 | 0 | 0 |  |

## Realized Market State Slice

| candidate_pool_version | model | top_k | realized_market_state | months | mean_topk_return | mean_topk_excess | median_topk_excess | hit_rate | mean_topk_minus_nextk |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| U1_liquid_tradable | B0_market_ew | 20 | neutral | 29 | 0.00418018 | 0 | 0 | 0 |  |
| U1_liquid_tradable | B0_market_ew | 20 | strong_down | 11 | -0.0748173 | 0 | 0 | 0 |  |
| U1_liquid_tradable | B0_market_ew | 20 | strong_up | 11 | 0.0773816 | 0 | 0 | 0 |  |
| U1_liquid_tradable | M5_plus_industry_breadth_elasticnet_excess | 20 | neutral | 29 | 0.016213 | 0.0120329 | 0.0110535 | 0.655172 | 0.010589 |
| U1_liquid_tradable | M5_plus_industry_breadth_elasticnet_excess | 20 | strong_down | 11 | -0.0673085 | 0.00750879 | -0.00343948 | 0.454545 | -0.000361365 |
| U1_liquid_tradable | M5_plus_industry_breadth_elasticnet_excess | 20 | strong_up | 11 | 0.0778764 | 0.000494843 | -0.00789027 | 0.272727 | 0.00615198 |
| U1_liquid_tradable | M5_plus_industry_breadth_plus_fund_flow_elasticnet_excess | 20 | neutral | 29 | 0.0179135 | 0.0137333 | 0.0110535 | 0.689655 | 0.0136702 |
| U1_liquid_tradable | M5_plus_industry_breadth_plus_fund_flow_elasticnet_excess | 20 | strong_down | 11 | -0.0658323 | 0.00898502 | 0.00107566 | 0.545455 | 0.00019934 |
| U1_liquid_tradable | M5_plus_industry_breadth_plus_fund_flow_elasticnet_excess | 20 | strong_up | 11 | 0.0813222 | 0.00394064 | -0.00789027 | 0.272727 | 0.0144166 |
| U1_liquid_tradable | M5_plus_industry_breadth_plus_fund_flow_plus_fundamental_elasticnet_excess | 20 | neutral | 29 | 0.0173131 | 0.013133 | 0.00359774 | 0.586207 | 0.00357365 |
| U1_liquid_tradable | M5_plus_industry_breadth_plus_fund_flow_plus_fundamental_elasticnet_excess | 20 | strong_down | 11 | -0.0617475 | 0.0130698 | 0.014084 | 0.636364 | 0.00641016 |
| U1_liquid_tradable | M5_plus_industry_breadth_plus_fund_flow_plus_fundamental_elasticnet_excess | 20 | strong_up | 11 | 0.0647902 | -0.0125913 | -0.0124275 | 0.272727 | -0.013481 |
| U1_liquid_tradable | M5_plus_industry_breadth_plus_fund_flow_plus_fundamental_plus_shareholder_elasticnet_excess | 20 | neutral | 29 | 0.019087 | 0.0149068 | 0.0123819 | 0.655172 | 0.00230082 |
| U1_liquid_tradable | M5_plus_industry_breadth_plus_fund_flow_plus_fundamental_plus_shareholder_elasticnet_excess | 20 | strong_down | 11 | -0.0459643 | 0.028853 | 0.0243762 | 0.818182 | 0.021998 |
| U1_liquid_tradable | M5_plus_industry_breadth_plus_fund_flow_plus_fundamental_plus_shareholder_elasticnet_excess | 20 | strong_up | 11 | 0.0681399 | -0.00924169 | -0.0353133 | 0.272727 | 0.00200113 |
| U1_liquid_tradable | M5_plus_industry_breadth_plus_fund_flow_plus_fundamental_plus_shareholder_plus_northbound_elasticnet_excess | 20 | neutral | 29 | 0.0188183 | 0.0146381 | 0.0123819 | 0.655172 | 0.00196552 |
| U1_liquid_tradable | M5_plus_industry_breadth_plus_fund_flow_plus_fundamental_plus_shareholder_plus_northbound_elasticnet_excess | 20 | strong_down | 11 | -0.0473942 | 0.027423 | 0.0243762 | 0.818182 | 0.0196722 |
| U1_liquid_tradable | M5_plus_industry_breadth_plus_fund_flow_plus_fundamental_plus_shareholder_plus_northbound_elasticnet_excess | 20 | strong_up | 11 | 0.0681958 | -0.00918575 | -0.0353133 | 0.272727 | 0.00205592 |
| U1_liquid_tradable | M5_price_volume_only_elasticnet_excess | 20 | neutral | 29 | 0.0230734 | 0.0188932 | 0.0171706 | 0.724138 | 0.0126708 |
| U1_liquid_tradable | M5_price_volume_only_elasticnet_excess | 20 | strong_down | 11 | -0.0808196 | -0.00600236 | -0.0092517 | 0.363636 | -0.00831511 |
| U1_liquid_tradable | M5_price_volume_only_elasticnet_excess | 20 | strong_up | 11 | 0.0960937 | 0.0187121 | 0.00142474 | 0.545455 | 0.0205504 |
| U2_risk_sane | B0_market_ew | 20 | neutral | 29 | 0.00418018 | 0 | 0 | 0 |  |
| U2_risk_sane | B0_market_ew | 20 | strong_down | 11 | -0.0748173 | 0 | 0 | 0 |  |
| U2_risk_sane | B0_market_ew | 20 | strong_up | 11 | 0.0773816 | 0 | 0 | 0 |  |
| U2_risk_sane | M5_plus_industry_breadth_elasticnet_excess | 20 | neutral | 29 | 0.0151507 | 0.0109705 | 0.010439 | 0.586207 | 0.00895577 |
| U2_risk_sane | M5_plus_industry_breadth_elasticnet_excess | 20 | strong_down | 11 | -0.0640996 | 0.0107177 | 0.00709764 | 0.818182 | -0.00720177 |
| U2_risk_sane | M5_plus_industry_breadth_elasticnet_excess | 20 | strong_up | 11 | 0.0694179 | -0.00796364 | -0.0202538 | 0.363636 | -0.00487453 |
| U2_risk_sane | M5_plus_industry_breadth_plus_fund_flow_elasticnet_excess | 20 | neutral | 29 | 0.017748 | 0.0135678 | 0.010439 | 0.655172 | 0.0077399 |
| U2_risk_sane | M5_plus_industry_breadth_plus_fund_flow_elasticnet_excess | 20 | strong_down | 11 | -0.0626876 | 0.0121297 | 0.00709764 | 0.818182 | -0.00591296 |
| U2_risk_sane | M5_plus_industry_breadth_plus_fund_flow_elasticnet_excess | 20 | strong_up | 11 | 0.0722374 | -0.00514421 | -0.0132587 | 0.363636 | 0.00357173 |
| U2_risk_sane | M5_plus_industry_breadth_plus_fund_flow_plus_fundamental_elasticnet_excess | 20 | neutral | 29 | 0.0170991 | 0.0129189 | 0.0121136 | 0.551724 | 0.00154215 |
| U2_risk_sane | M5_plus_industry_breadth_plus_fund_flow_plus_fundamental_elasticnet_excess | 20 | strong_down | 11 | -0.0569188 | 0.0178984 | 0.0282173 | 0.727273 | 0.0171888 |
| U2_risk_sane | M5_plus_industry_breadth_plus_fund_flow_plus_fundamental_elasticnet_excess | 20 | strong_up | 11 | 0.0669974 | -0.0103842 | -0.0176399 | 0.363636 | 4.12629e-05 |
| U2_risk_sane | M5_plus_industry_breadth_plus_fund_flow_plus_fundamental_plus_shareholder_elasticnet_excess | 20 | neutral | 29 | 0.0219679 | 0.0177877 | 0.0147682 | 0.655172 | 0.0149442 |
| U2_risk_sane | M5_plus_industry_breadth_plus_fund_flow_plus_fundamental_plus_shareholder_elasticnet_excess | 20 | strong_down | 11 | -0.0495959 | 0.0252214 | 0.0271623 | 0.818182 | 0.0183482 |
| U2_risk_sane | M5_plus_industry_breadth_plus_fund_flow_plus_fundamental_plus_shareholder_elasticnet_excess | 20 | strong_up | 11 | 0.0676342 | -0.00974741 | -0.00358735 | 0.454545 | -0.0020289 |
| U2_risk_sane | M5_plus_industry_breadth_plus_fund_flow_plus_fundamental_plus_shareholder_plus_northbound_elasticnet_excess | 20 | neutral | 29 | 0.0217528 | 0.0175726 | 0.0147682 | 0.655172 | 0.0151494 |
| U2_risk_sane | M5_plus_industry_breadth_plus_fund_flow_plus_fundamental_plus_shareholder_plus_northbound_elasticnet_excess | 20 | strong_down | 11 | -0.0504095 | 0.0244078 | 0.0271623 | 0.727273 | 0.016801 |
| U2_risk_sane | M5_plus_industry_breadth_plus_fund_flow_plus_fundamental_plus_shareholder_plus_northbound_elasticnet_excess | 20 | strong_up | 11 | 0.0679414 | -0.00944019 | -0.00358735 | 0.454545 | -0.00121532 |
| U2_risk_sane | M5_price_volume_only_elasticnet_excess | 20 | neutral | 29 | 0.022353 | 0.0181728 | 0.0225659 | 0.689655 | 0.0078353 |

## 口径

- `price_volume_only` 是 M5 内部 baseline；后续 spec 按 industry_breadth -> fund_flow -> fundamental -> shareholder 累积加入。
- `topk_excess_after_cost` 使用半 L1 月度换手乘以 cost_bps 的简化成本敏感性。

## 本轮产物

- `data/results/m5_northbound_test_2026-05-05_summary.json`
- `data/results/m5_northbound_test_2026-05-05_leaderboard.csv`
- `data/results/m5_northbound_test_2026-05-05_incremental_delta.csv`
- `data/results/m5_northbound_test_2026-05-05_monthly_long.csv`
- `data/results/m5_northbound_test_2026-05-05_rank_ic.csv`
- `data/results/m5_northbound_test_2026-05-05_quantile_spread.csv`
- `data/results/m5_northbound_test_2026-05-05_feature_coverage.csv`
- `data/results/m5_northbound_test_2026-05-05_feature_importance.csv`
- `data/results/m5_northbound_test_2026-05-05_topk_holdings.csv`
- `data/results/m5_northbound_test_2026-05-05_industry_exposure.csv`
- `data/results/m5_northbound_test_2026-05-05_candidate_pool_width.csv`
- `data/results/m5_northbound_test_2026-05-05_candidate_pool_reject_reason.csv`
- `data/results/m5_northbound_test_2026-05-05_year_slice.csv`
- `data/results/m5_northbound_test_2026-05-05_regime_slice.csv`
- `data/results/m5_northbound_test_2026-05-05_market_states.csv`
- `data/results/m5_northbound_test_2026-05-05_manifest.json`
