# Monthly Selection Oracle

- 生成时间：`2026-04-28T12:27:42.875880+00:00`
- 结果类型：`monthly_selection_oracle`
- 研究主题：`monthly_selection_oracle`
- 研究配置：`dataset_monthly_selection_features_pools_u0_all_tradable-u1_liquid_tradable-u2_risk_sane_topk_20-30-50_buckets_5`
- 输出 stem：`monthly_selection_oracle_2026-04-28`
- 数据集：`data/cache/monthly_selection_features.parquet`
- 有效标签月份：`62`

## Oracle By Candidate Pool

| candidate_pool_version | top_k | months | median_candidate_pool_width | mean_oracle_topk_return | median_oracle_topk_return | mean_oracle_excess_vs_market | median_oracle_excess_vs_market | positive_oracle_excess_share | mean_oracle_minus_pool |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| U0_all_tradable | 20 | 62 | 4919.5 | 0.865728 | 0.84459 | 0.854008 | 0.836051 | 1 | 0.853156 |
| U1_liquid_tradable | 20 | 62 | 3111.5 | 0.753628 | 0.738583 | 0.741908 | 0.701405 | 1 | 0.744685 |
| U2_risk_sane | 20 | 62 | 2797 | 0.693729 | 0.669107 | 0.682009 | 0.65732 | 1 | 0.683617 |
| U0_all_tradable | 30 | 62 | 4919.5 | 0.760462 | 0.731057 | 0.748742 | 0.736413 | 1 | 0.74789 |
| U1_liquid_tradable | 30 | 62 | 3111.5 | 0.66343 | 0.652435 | 0.65171 | 0.616664 | 1 | 0.654487 |
| U2_risk_sane | 30 | 62 | 2797 | 0.609318 | 0.592027 | 0.597599 | 0.578516 | 1 | 0.599206 |
| U0_all_tradable | 50 | 62 | 4919.5 | 0.641405 | 0.609469 | 0.629685 | 0.618893 | 1 | 0.628833 |
| U1_liquid_tradable | 50 | 62 | 3111.5 | 0.560684 | 0.547599 | 0.548964 | 0.518989 | 1 | 0.551741 |
| U2_risk_sane | 50 | 62 | 2797 | 0.512769 | 0.493601 | 0.50105 | 0.492748 | 1 | 0.502657 |

## Feature Bucket Monotonicity

| candidate_pool_version | feature | feature_col | direction | bucket | months | n | mean_forward_return | median_forward_return | future_top20_share | bucket_return_spearman | bucket_top20_spearman | bucket_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| U0_all_tradable | low_limit_move_hits_20d | feature_limit_move_hits_20d_z | -1 | 1 | 62 | 225067 | 0.0102461 | -0.00722924 | 0.214809 | 1 | -1 | 3 |
| U0_all_tradable | low_vol_20d | feature_realized_vol_20d_z | -1 | 1 | 62 | 58988 | 0.00348748 | -0.0236434 | 0.235671 | 0.7 | -1 | 5 |
| U1_liquid_tradable | low_limit_move_hits_20d | feature_limit_move_hits_20d_z | -1 | 1 | 62 | 125134 | 0.006874 | -0.0223324 | 0.221436 | 0.5 | -1 | 3 |
| U2_risk_sane | low_limit_move_hits_20d | feature_limit_move_hits_20d_z | -1 | 1 | 62 | 136696 | 0.00925771 | -0.01434 | 0.215387 | 0.5 | -1 | 3 |
| U1_liquid_tradable | low_vol_20d | feature_realized_vol_20d_z | -1 | 1 | 62 | 40957 | 0.000594748 | -0.0224503 | 0.233149 | 0.3 | -0.9 | 5 |
| U2_risk_sane | turnover_20d | feature_turnover_20d_z | 1 | 1 | 62 | 36639 | 0.00782283 | -0.00708351 | 0.185921 | 0.3 | 1 | 5 |
| U2_risk_sane | low_vol_20d | feature_realized_vol_20d_z | -1 | 1 | 62 | 36639 | 0.00540316 | -0.0195526 | 0.233871 | 0.1 | -1 | 5 |
| U0_all_tradable | momentum_20d | feature_ret_20d_z | 1 | 1 | 62 | 58989 | 0.013502 | 0.00394246 | 0.213934 | 0 | 0.7 | 5 |
| U2_risk_sane | momentum_20d | feature_ret_20d_z | 1 | 1 | 62 | 36639 | 0.0107408 | -0.00145678 | 0.213552 | -0.1 | 0.7 | 5 |
| U0_all_tradable | short_momentum_5d | feature_ret_5d_z | 1 | 1 | 62 | 59026 | 0.0134775 | -0.0146656 | 0.220549 | -0.4 | 0.3 | 5 |
| U1_liquid_tradable | momentum_20d | feature_ret_20d_z | 1 | 1 | 62 | 40957 | 0.0108424 | -0.0022025 | 0.214499 | -0.6 | 0.7 | 5 |
| U1_liquid_tradable | turnover_20d | feature_turnover_20d_z | 1 | 1 | 62 | 40957 | 0.0079245 | -0.00618153 | 0.187738 | -0.6 | 1 | 5 |
| U0_all_tradable | turnover_20d | feature_turnover_20d_z | 1 | 1 | 62 | 58988 | 0.0143083 | 0.0030266 | 0.18617 | -0.7 | 1 | 5 |
| U2_risk_sane | price_position_250d | feature_price_position_250d_z | 1 | 1 | 62 | 37806 | 0.00914586 | -0.00749512 | 0.18862 | -0.7 | 1 | 5 |
| U0_all_tradable | price_position_250d | feature_price_position_250d_z | 1 | 1 | 62 | 60998 | 0.013359 | 0.000203241 | 0.190277 | -0.7 | 1 | 5 |
| U1_liquid_tradable | price_position_250d | feature_price_position_250d_z | 1 | 1 | 62 | 42218 | 0.00962733 | -0.00243561 | 0.189352 | -0.7 | 1 | 5 |
| U2_risk_sane | momentum_60d | feature_ret_60d_z | 1 | 1 | 62 | 36639 | 0.0120937 | -0.00323736 | 0.215901 | -0.7 | 0.4 | 5 |
| U2_risk_sane | short_momentum_5d | feature_ret_5d_z | 1 | 1 | 62 | 36639 | 0.0121747 | -0.0180911 | 0.220664 | -0.9 | 0.3 | 5 |
| U1_liquid_tradable | short_momentum_5d | feature_ret_5d_z | 1 | 1 | 62 | 40976 | 0.0115635 | -0.0177612 | 0.222867 | -0.9 | 0.3 | 5 |
| U0_all_tradable | liquidity_amount_20d | feature_amount_20d_log_z | 1 | 1 | 62 | 58988 | 0.023804 | 0.00742818 | 0.202549 | -1 | 0.9 | 5 |

## Baseline Overlap

| candidate_pool_version | baseline | feature_col | top_k | months | median_candidate_pool_width | mean_oracle_topk_overlap_share | median_oracle_topk_overlap_share | mean_oracle_top20_bucket_hit_share | mean_baseline_topk_return | mean_baseline_excess_vs_market | positive_baseline_excess_share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| U0_all_tradable | low_limit_move_hits_20d | feature_limit_move_hits_20d_z | 30 | 62 | 4919.5 | 0.00591398 | 0 | 0.182258 | 0.013034 | 0.00131408 | 0.483871 |
| U0_all_tradable | low_limit_move_hits_20d | feature_limit_move_hits_20d_z | 20 | 62 | 4919.5 | 0.00403226 | 0 | 0.181452 | 0.0123936 | 0.000673689 | 0.483871 |
| U0_all_tradable | low_limit_move_hits_20d | feature_limit_move_hits_20d_z | 50 | 62 | 4919.5 | 0.00741935 | 0 | 0.176452 | 0.0114755 | -0.000244412 | 0.419355 |
| U0_all_tradable | low_vol_20d | feature_realized_vol_20d_z | 20 | 62 | 4919.5 | 0.00241935 | 0 | 0.135484 | 0.010851 | -0.000868925 | 0.451613 |
| U0_all_tradable | low_vol_20d | feature_realized_vol_20d_z | 50 | 62 | 4919.5 | 0.00225806 | 0 | 0.145484 | 0.0105052 | -0.00121467 | 0.451613 |
| U2_risk_sane | turnover_20d | feature_turnover_20d_z | 20 | 62 | 2797 | 0.00967742 | 0 | 0.241935 | 0.0102787 | -0.00144121 | 0.419355 |
| U0_all_tradable | low_vol_20d | feature_realized_vol_20d_z | 30 | 62 | 4919.5 | 0.0016129 | 0 | 0.137634 | 0.0101623 | -0.00155755 | 0.5 |
| U2_risk_sane | turnover_20d | feature_turnover_20d_z | 50 | 62 | 2797 | 0.0290323 | 0.02 | 0.235161 | 0.00954383 | -0.00217605 | 0.483871 |
| U2_risk_sane | low_limit_move_hits_20d | feature_limit_move_hits_20d_z | 50 | 62 | 2797 | 0.013871 | 0 | 0.183548 | 0.00922958 | -0.0024903 | 0.370968 |
| U1_liquid_tradable | low_vol_20d | feature_realized_vol_20d_z | 50 | 62 | 3111.5 | 0.00451613 | 0 | 0.146774 | 0.00873889 | -0.00298099 | 0.435484 |
| U1_liquid_tradable | low_limit_move_hits_20d | feature_limit_move_hits_20d_z | 50 | 62 | 3111.5 | 0.0106452 | 0 | 0.182903 | 0.00844364 | -0.00327624 | 0.33871 |
| U2_risk_sane | low_vol_20d | feature_realized_vol_20d_z | 50 | 62 | 2797 | 0.00612903 | 0 | 0.143871 | 0.00817546 | -0.00354442 | 0.435484 |
| U2_risk_sane | turnover_20d | feature_turnover_20d_z | 30 | 62 | 2797 | 0.0182796 | 0 | 0.226882 | 0.00803746 | -0.00368242 | 0.451613 |
| U1_liquid_tradable | low_vol_20d | feature_realized_vol_20d_z | 30 | 62 | 3111.5 | 0.00215054 | 0 | 0.141935 | 0.00786421 | -0.00385567 | 0.451613 |
| U2_risk_sane | low_vol_20d | feature_realized_vol_20d_z | 30 | 62 | 2797 | 0.00215054 | 0 | 0.139247 | 0.00766717 | -0.00405272 | 0.451613 |
| U1_liquid_tradable | low_vol_20d | feature_realized_vol_20d_z | 20 | 62 | 3111.5 | 0.0016129 | 0 | 0.134677 | 0.0074114 | -0.00430849 | 0.451613 |
| U2_risk_sane | low_vol_20d | feature_realized_vol_20d_z | 20 | 62 | 2797 | 0.0016129 | 0 | 0.131452 | 0.00721013 | -0.00450975 | 0.451613 |
| U2_risk_sane | low_limit_move_hits_20d | feature_limit_move_hits_20d_z | 30 | 62 | 2797 | 0.00752688 | 0 | 0.176344 | 0.00703416 | -0.00468572 | 0.354839 |
| U1_liquid_tradable | low_limit_move_hits_20d | feature_limit_move_hits_20d_z | 30 | 62 | 3111.5 | 0.00591398 | 0 | 0.175269 | 0.00649792 | -0.00522197 | 0.33871 |
| U2_risk_sane | low_limit_move_hits_20d | feature_limit_move_hits_20d_z | 20 | 62 | 2797 | 0.00403226 | 0 | 0.167742 | 0.00511444 | -0.00660544 | 0.33871 |

## Interpretation Note

- 不应过度在意因子或模型是否命中事后 oracle Top-20。Oracle Top-20 是候选空间上限和可分性诊断，不是训练目标，也不是策略准入 gate。
- 可交易月度选股的核心目标是稳定改善推荐 Top-K 的收益分布，包括 `topk_excess_mean`、`topk_hit_rate`、`topk_minus_nextk`、分桶 spread、年度/市场状态稳定性和成本后表现。
- 一个模型可以几乎不命中每个月事后最强的 20 只股票，但只要它稳定避开差股票、提高 Top-K 平均收益并控制回撤/换手，就仍然是有效研究候选。
- 因此，`baseline_overlap` 应只用于理解“现有特征离 oracle 上限有多远”，不能用于否定能盈利的非 oracle-mimic 排序模型。

## Regime Oracle Capacity

| candidate_pool_version | top_k | realized_market_state | months | mean_market_return | mean_oracle_topk_return | median_oracle_topk_return | mean_oracle_excess_vs_market | median_oracle_excess_vs_market |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| U0_all_tradable | 20 | neutral | 36 | 0.0107482 | 0.84632 | 0.82003 | 0.835572 | 0.808784 |
| U0_all_tradable | 20 | strong_down | 13 | -0.0700111 | 0.701155 | 0.670718 | 0.771166 | 0.731083 |
| U0_all_tradable | 20 | strong_up | 13 | 0.0961418 | 1.08405 | 1.04516 | 0.987906 | 0.975639 |
| U0_all_tradable | 30 | neutral | 36 | 0.0107482 | 0.744701 | 0.70774 | 0.733953 | 0.698033 |
| U0_all_tradable | 30 | strong_down | 13 | -0.0700111 | 0.60174 | 0.56448 | 0.671751 | 0.624032 |
| U0_all_tradable | 30 | strong_up | 13 | 0.0961418 | 0.962828 | 0.900174 | 0.866687 | 0.830649 |
| U0_all_tradable | 50 | neutral | 36 | 0.0107482 | 0.628425 | 0.604932 | 0.617677 | 0.580271 |
| U0_all_tradable | 50 | strong_down | 13 | -0.0700111 | 0.490369 | 0.457089 | 0.56038 | 0.509472 |
| U0_all_tradable | 50 | strong_up | 13 | 0.0961418 | 0.828387 | 0.788302 | 0.732245 | 0.720778 |
| U1_liquid_tradable | 20 | neutral | 36 | 0.0107482 | 0.741817 | 0.6826 | 0.731069 | 0.675639 |
| U1_liquid_tradable | 20 | strong_down | 13 | -0.0700111 | 0.584127 | 0.521122 | 0.654138 | 0.556345 |
| U1_liquid_tradable | 20 | strong_up | 13 | 0.0961418 | 0.955835 | 0.927674 | 0.859693 | 0.860151 |
| U1_liquid_tradable | 30 | neutral | 36 | 0.0107482 | 0.65235 | 0.59715 | 0.641602 | 0.588623 |
| U1_liquid_tradable | 30 | strong_down | 13 | -0.0700111 | 0.501259 | 0.448794 | 0.57127 | 0.484018 |
| U1_liquid_tradable | 30 | strong_up | 13 | 0.0961418 | 0.856284 | 0.848257 | 0.760142 | 0.780734 |
| U1_liquid_tradable | 50 | neutral | 36 | 0.0107482 | 0.550023 | 0.504381 | 0.539275 | 0.499078 |
| U1_liquid_tradable | 50 | strong_down | 13 | -0.0700111 | 0.408368 | 0.363418 | 0.478379 | 0.407212 |
| U1_liquid_tradable | 50 | strong_up | 13 | 0.0961418 | 0.742521 | 0.756233 | 0.64638 | 0.661555 |
| U2_risk_sane | 20 | neutral | 36 | 0.0107482 | 0.673977 | 0.616895 | 0.663229 | 0.616687 |
| U2_risk_sane | 20 | strong_down | 13 | -0.0700111 | 0.53065 | 0.457998 | 0.600661 | 0.501782 |
| U2_risk_sane | 20 | strong_up | 13 | 0.0961418 | 0.911506 | 0.908268 | 0.815365 | 0.840744 |
| U2_risk_sane | 30 | neutral | 36 | 0.0107482 | 0.592781 | 0.558994 | 0.582033 | 0.547488 |
| U2_risk_sane | 30 | strong_down | 13 | -0.0700111 | 0.45188 | 0.390626 | 0.521891 | 0.442383 |
| U2_risk_sane | 30 | strong_up | 13 | 0.0961418 | 0.812552 | 0.827313 | 0.716411 | 0.759789 |
| U2_risk_sane | 50 | neutral | 36 | 0.0107482 | 0.498038 | 0.48481 | 0.48729 | 0.466893 |
| U2_risk_sane | 50 | strong_down | 13 | -0.0700111 | 0.36555 | 0.317813 | 0.435561 | 0.376468 |
| U2_risk_sane | 50 | strong_up | 13 | 0.0961418 | 0.700784 | 0.72306 | 0.604642 | 0.633675 |

## Industry Oracle Distribution

| candidate_pool_version | top_k | industry_level1 | mean_share | months |
| --- | --- | --- | --- | --- |
| U0_all_tradable | 20 | 计算机 | 0.142045 | 44 |
| U0_all_tradable | 20 | 电子 | 0.135 | 40 |
| U0_all_tradable | 20 | 机械设备 | 0.127451 | 51 |
| U0_all_tradable | 20 | 医药生物 | 0.11875 | 40 |
| U0_all_tradable | 20 | 电力设备 | 0.10641 | 39 |
| U0_all_tradable | 20 | 传媒 | 0.101724 | 29 |
| U0_all_tradable | 20 | 基础化工 | 0.10125 | 40 |
| U0_all_tradable | 20 | 公用事业 | 0.0973684 | 19 |
| U0_all_tradable | 20 | 汽车 | 0.0971429 | 35 |
| U0_all_tradable | 20 | 房地产 | 0.0911765 | 17 |
| U0_all_tradable | 20 | 通信 | 0.0890625 | 32 |
| U0_all_tradable | 20 | 国防军工 | 0.0857143 | 14 |
| U0_all_tradable | 20 | 商贸零售 | 0.08 | 25 |
| U0_all_tradable | 20 | 有色金属 | 0.0789474 | 19 |
| U0_all_tradable | 20 | 石油石化 | 0.075 | 4 |
| U0_all_tradable | 20 | 社会服务 | 0.075 | 14 |
| U0_all_tradable | 20 | 建筑装饰 | 0.071875 | 32 |
| U0_all_tradable | 20 | 交通运输 | 0.065625 | 16 |
| U0_all_tradable | 20 | 家用电器 | 0.0636364 | 11 |
| U0_all_tradable | 20 | 环保 | 0.0619048 | 21 |
| U0_all_tradable | 20 | 食品饮料 | 0.0607143 | 14 |
| U0_all_tradable | 20 | 纺织服饰 | 0.06 | 25 |
| U0_all_tradable | 20 | 轻工制造 | 0.0596154 | 26 |
| U0_all_tradable | 20 | 农林牧渔 | 0.0576923 | 13 |
| U0_all_tradable | 20 | 煤炭 | 0.05 | 6 |
| U0_all_tradable | 20 | 综合 | 0.05 | 3 |
| U0_all_tradable | 20 | 钢铁 | 0.05 | 6 |
| U0_all_tradable | 20 | _UNKNOWN_ | 0.05 | 7 |
| U0_all_tradable | 20 | 建筑材料 | 0.05 | 9 |
| U0_all_tradable | 20 | 美容护理 | 0.05 | 5 |

## 口径

- Oracle Top-K：每个 `signal_date`、每个候选池内，按未来 `label_forward_1m_o2o_return` 事后排序取 Top-K。
- Oracle overlap 只作诊断，不作为主评价指标；模型不需要命中每个月事后最强的 Top-20，只要稳定提高可交易 Top-K 收益分布即可。
- `realized_market_state` 使用同一持有期市场等权收益的全样本 20%/80% 分位切片，仅用于 oracle capacity 归因，不作为可交易信号。
- Baseline overlap 使用单因子截面排序与 oracle Top-K / future top 20% bucket 对比。
- 特征分桶使用已在 M2 中按月截面 winsorize/z-score 的特征列。

## 本轮产物

- `data/results/monthly_selection_oracle_2026-04-28_summary.json`
- `data/results/monthly_selection_oracle_2026-04-28_oracle_topk_return_by_month.csv`
- `data/results/monthly_selection_oracle_2026-04-28_oracle_topk_holdings.csv`
- `data/results/monthly_selection_oracle_2026-04-28_oracle_topk_by_candidate_pool.csv`
- `data/results/monthly_selection_oracle_2026-04-28_feature_bucket_monotonicity.csv`
- `data/results/monthly_selection_oracle_2026-04-28_baseline_overlap.csv`
- `data/results/monthly_selection_oracle_2026-04-28_regime_oracle_capacity.csv`
- `data/results/monthly_selection_oracle_2026-04-28_industry_oracle_distribution.csv`
- `data/results/monthly_selection_oracle_2026-04-28_candidate_pool_width.csv`
- `data/results/monthly_selection_oracle_2026-04-28_market_states.csv`
- `data/results/monthly_selection_oracle_2026-04-28_manifest.json`
