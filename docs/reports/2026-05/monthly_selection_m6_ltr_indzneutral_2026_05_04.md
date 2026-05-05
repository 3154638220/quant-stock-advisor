# Monthly Selection M6 Learning-to-Rank

- 生成时间：`2026-05-03T17:07:15.269092+00:00`
- 结果类型：`monthly_selection_m6_ltr`
- 研究主题：`monthly_selection_m6_ltr`
- 研究配置：`dataset_monthly_selection_features_families_price_volume_only_industry_breadth_fund_flow_fundamental_pools_u1_liquid_tradable_u2_risk_sane_topk_20_models_xgboost_rank_ndcg_grades_5_maxfit_0_jobs_all_wf_24m_costbps_10_0`
- 输出 stem：`monthly_selection_m6_ltr_indzneutral_2026_05_04`
- 数据集：`data/cache/monthly_selection_features.parquet`
- 数据库：`data/market.duckdb`
- 训练/评估：按 signal_date walk-forward；每个测试月只用历史月份训练。
- 有效标签月份：`63`
- 单窗训练行上限：`0`（`0` 表示不抽样）

## Leaderboard

| candidate_pool_version | model | model_type | top_k | months | mean_topk_return | median_topk_return | topk_excess_mean | topk_excess_median | topk_hit_rate | topk_minus_nextk_mean | industry_neutral_topk_excess_mean | turnover_mean | cost_drag_mean | topk_excess_after_cost_mean | topk_excess_annualized | rank_ic_mean | rank_ic_std | rank_ic_months | rank_ic_ir | quantile_top_minus_bottom_mean | neutral_median_excess | strong_down_median_excess | strong_up_median_excess |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| U1_liquid_tradable | B0_market_ew | benchmark | 20 | 39 | 0.00290466 | 0.00265337 | 0 | 0 | 0 |  |  |  | 0 | 0 | 0 |  |  |  |  |  | 0 | 0 | 0 |
| U1_liquid_tradable | M6_xgboost_rank_ndcg | xgboost_ranker | 20 | 39 | -0.0060883 | -0.0199936 | -0.00899296 | -0.0104542 | 0.282051 | -6.54146e-05 | -0.00464049 | 0.925 | 0.000925 | -0.010144 | -0.102735 | -0.00814835 | 0.102207 | 39 | -0.0797242 | -0.00437628 | -0.0194282 | 0.037658 | -0.00515215 |
| U2_risk_sane | B0_market_ew | benchmark | 20 | 39 | 0.00290466 | 0.00265337 | 0 | 0 | 0 |  |  |  | 0 | 0 | 0 |  |  |  |  |  | 0 | 0 | 0 |
| U2_risk_sane | M6_xgboost_rank_ndcg | xgboost_ranker | 20 | 39 | -0.00516272 | 0.000743067 | -0.00806737 | -0.0117587 | 0.358974 | -0.00830543 | -0.00549767 | 0.952632 | 0.000952632 | -0.00890087 | -0.0926265 | -0.00499501 | 0.105277 | 39 | -0.0474465 | -0.00220577 | -0.0102033 | -0.00971033 | -0.0444757 |

## U1 Top20 Leading Models

| candidate_pool_version | model | model_type | top_k | months | mean_topk_return | median_topk_return | topk_excess_mean | topk_excess_median | topk_hit_rate | topk_minus_nextk_mean | industry_neutral_topk_excess_mean | turnover_mean | cost_drag_mean | topk_excess_after_cost_mean | topk_excess_annualized | rank_ic_mean | rank_ic_std | rank_ic_months | rank_ic_ir | quantile_top_minus_bottom_mean | neutral_median_excess | strong_down_median_excess | strong_up_median_excess |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| U1_liquid_tradable | B0_market_ew | benchmark | 20 | 39 | 0.00290466 | 0.00265337 | 0 | 0 | 0 |  |  |  | 0 | 0 | 0 |  |  |  |  |  | 0 | 0 | 0 |
| U1_liquid_tradable | M6_xgboost_rank_ndcg | xgboost_ranker | 20 | 39 | -0.0060883 | -0.0199936 | -0.00899296 | -0.0104542 | 0.282051 | -6.54146e-05 | -0.00464049 | 0.925 | 0.000925 | -0.010144 | -0.102735 | -0.00814835 | 0.102207 | 39 | -0.0797242 | -0.00437628 | -0.0194282 | 0.037658 | -0.00515215 |

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
| m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_fundamental_pe_ttm_ind_z | feature_fundamental_pe_ttm_ind | 307350 | 0 | 0 | 0 |  |  |
| m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_fundamental_pb_ind_z | feature_fundamental_pb_ind | 307350 | 0 | 0 | 0 |  |  |
| m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_fundamental_ev_ebitda_ind_z | feature_fundamental_ev_ebitda_ind | 307350 | 0 | 0 | 0 |  |  |
| m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_fundamental_roe_ttm_ind_z | feature_fundamental_roe_ttm_ind | 307350 | 0 | 0 | 0 |  |  |
| m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_fundamental_net_profit_yoy_ind_z | feature_fundamental_net_profit_yoy_ind | 307350 | 0 | 0 | 0 |  |  |
| m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_fundamental_gross_margin_change_ind_z | feature_fundamental_gross_margin_change_ind | 307350 | 0 | 0 | 0 |  |  |
| m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_fundamental_gross_margin_delta_ind_z | feature_fundamental_gross_margin_delta_ind | 307350 | 0 | 0 | 0 |  |  |
| m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_fundamental_debt_to_assets_change_ind_z | feature_fundamental_debt_to_assets_change_ind | 307350 | 0 | 0 | 0 |  |  |
| m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_fundamental_ocf_to_net_profit_ind_z | feature_fundamental_ocf_to_net_profit_ind | 307350 | 0 | 0 | 0 |  |  |
| m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_fundamental_ocf_to_asset_ind_z | feature_fundamental_ocf_to_asset_ind | 307350 | 0 | 0 | 0 |  |  |
| m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_fundamental_asset_turnover_ind_z | feature_fundamental_asset_turnover_ind | 307350 | 0 | 0 | 0 |  |  |
| m6_core_price_volume_industry_breadth_fund_flow_fundamental | price_volume,industry_breadth,fund_flow,fundamental | feature_fundamental_net_margin_stability_ind_z | feature_fundamental_net_margin_stability_ind | 307350 | 0 | 0 | 0 |  |  |

## Year Slice

| candidate_pool_version | model | top_k | year | months | mean_topk_return | mean_topk_excess | median_topk_excess | hit_rate | mean_topk_minus_nextk |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| U1_liquid_tradable | B0_market_ew | 20 | 2023 | 12 | -0.0185141 | 0 | 0 | 0 |  |
| U1_liquid_tradable | B0_market_ew | 20 | 2024 | 12 | -0.00586991 | 0 | 0 | 0 |  |
| U1_liquid_tradable | B0_market_ew | 20 | 2025 | 12 | 0.0329754 | 0 | 0 | 0 |  |
| U1_liquid_tradable | B0_market_ew | 20 | 2026 | 3 | 0.00339507 | 0 | 0 | 0 |  |
| U1_liquid_tradable | M6_xgboost_rank_ndcg | 20 | 2023 | 12 | -0.0249055 | -0.00639142 | -0.0183236 | 0.166667 | -0.0023534 |
| U1_liquid_tradable | M6_xgboost_rank_ndcg | 20 | 2024 | 12 | -0.0105456 | -0.0046757 | -0.0110585 | 0.333333 | -0.00421293 |
| U1_liquid_tradable | M6_xgboost_rank_ndcg | 20 | 2025 | 12 | 0.016521 | -0.0164544 | -0.00663676 | 0.333333 | 0.00688329 |
| U1_liquid_tradable | M6_xgboost_rank_ndcg | 20 | 2026 | 3 | -0.0034274 | -0.00682247 | -0.0316529 | 0.333333 | -0.00211824 |
| U2_risk_sane | B0_market_ew | 20 | 2023 | 12 | -0.0185141 | 0 | 0 | 0 |  |
| U2_risk_sane | B0_market_ew | 20 | 2024 | 12 | -0.00586991 | 0 | 0 | 0 |  |
| U2_risk_sane | B0_market_ew | 20 | 2025 | 12 | 0.0329754 | 0 | 0 | 0 |  |
| U2_risk_sane | B0_market_ew | 20 | 2026 | 3 | 0.00339507 | 0 | 0 | 0 |  |
| U2_risk_sane | M6_xgboost_rank_ndcg | 20 | 2023 | 12 | -0.0181021 | 0.000411979 | -0.00493466 | 0.333333 | -0.00129009 |
| U2_risk_sane | M6_xgboost_rank_ndcg | 20 | 2024 | 12 | -0.0168951 | -0.0110252 | -0.0120644 | 0.416667 | -0.00308992 |
| U2_risk_sane | M6_xgboost_rank_ndcg | 20 | 2025 | 12 | 0.0202036 | -0.0127717 | -0.0188811 | 0.416667 | -0.0253564 |
| U2_risk_sane | M6_xgboost_rank_ndcg | 20 | 2026 | 3 | -0.00794104 | -0.0113361 | -0.0102033 | 0 | 0.010975 |

## Realized Market State Slice

| candidate_pool_version | model | top_k | realized_market_state | months | mean_topk_return | mean_topk_excess | median_topk_excess | hit_rate | mean_topk_minus_nextk |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| U1_liquid_tradable | B0_market_ew | 20 | neutral | 25 | 0.00196727 | 0 | 0 | 0 |  |
| U1_liquid_tradable | B0_market_ew | 20 | strong_down | 7 | -0.0722071 | 0 | 0 | 0 |  |
| U1_liquid_tradable | B0_market_ew | 20 | strong_up | 7 | 0.0813642 | 0 | 0 | 0 |  |
| U1_liquid_tradable | M6_xgboost_rank_ndcg | 20 | neutral | 25 | -0.0188778 | -0.020845 | -0.0194282 | 0.16 | -0.011054 |
| U1_liquid_tradable | M6_xgboost_rank_ndcg | 20 | strong_down | 7 | -0.0509737 | 0.0212334 | 0.037658 | 0.571429 | 0.0138347 |
| U1_liquid_tradable | M6_xgboost_rank_ndcg | 20 | strong_up | 7 | 0.0844738 | 0.00310956 | -0.00515215 | 0.428571 | 0.0252793 |
| U2_risk_sane | B0_market_ew | 20 | neutral | 25 | 0.00196727 | 0 | 0 | 0 |  |
| U2_risk_sane | B0_market_ew | 20 | strong_down | 7 | -0.0722071 | 0 | 0 | 0 |  |
| U2_risk_sane | B0_market_ew | 20 | strong_up | 7 | 0.0813642 | 0 | 0 | 0 |  |
| U2_risk_sane | M6_xgboost_rank_ndcg | 20 | neutral | 25 | -0.0100278 | -0.0119951 | -0.0102033 | 0.36 | -0.00809019 |
| U2_risk_sane | M6_xgboost_rank_ndcg | 20 | strong_down | 7 | -0.0534523 | 0.0187548 | -0.00971033 | 0.428571 | 0.0231733 |
| U2_risk_sane | M6_xgboost_rank_ndcg | 20 | strong_up | 7 | 0.0605022 | -0.0208621 | -0.0444757 | 0.285714 | -0.0405529 |

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

- `data/results/monthly_selection_m6_ltr_indzneutral_2026_05_04_summary.json`
- `data/results/monthly_selection_m6_ltr_indzneutral_2026_05_04_leaderboard.csv`
- `data/results/monthly_selection_m6_ltr_indzneutral_2026_05_04_monthly_long.csv`
- `data/results/monthly_selection_m6_ltr_indzneutral_2026_05_04_rank_ic.csv`
- `data/results/monthly_selection_m6_ltr_indzneutral_2026_05_04_quantile_spread.csv`
- `data/results/monthly_selection_m6_ltr_indzneutral_2026_05_04_feature_coverage.csv`
- `data/results/monthly_selection_m6_ltr_indzneutral_2026_05_04_feature_importance.csv`
- `data/results/monthly_selection_m6_ltr_indzneutral_2026_05_04_topk_holdings.csv`
- `data/results/monthly_selection_m6_ltr_indzneutral_2026_05_04_industry_exposure.csv`
- `data/results/monthly_selection_m6_ltr_indzneutral_2026_05_04_candidate_pool_width.csv`
- `data/results/monthly_selection_m6_ltr_indzneutral_2026_05_04_candidate_pool_reject_reason.csv`
- `data/results/monthly_selection_m6_ltr_indzneutral_2026_05_04_year_slice.csv`
- `data/results/monthly_selection_m6_ltr_indzneutral_2026_05_04_regime_slice.csv`
- `data/results/monthly_selection_m6_ltr_indzneutral_2026_05_04_market_states.csv`
- `data/results/monthly_selection_m6_ltr_indzneutral_2026_05_04_manifest.json`
