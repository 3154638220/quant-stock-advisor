# R2B v2 Weight Audit

- 生成时间：`2026-04-28T10:36:15.768142+00:00`
- `eval_contract_version`: `r0_eval_execution_contract_2026-04-28`
- `execution_contract_version`: `tplus1_open_buy_delta_limit_mask_2026-04-28`
- `industry_map_source_status`: `real_industry_map`
- 目标：审计 `U3_A` gray zone 是否有稳定证据，不产生 production candidate。

## 1. U3_A 月度归因

| month_end           | regime      | breadth   |   baseline_excess |   target_excess |   monthly_delta_vs_baseline |   selected_pairs |
|:--------------------|:------------|:----------|------------------:|----------------:|----------------------------:|-----------------:|
| 2025-01-31 00:00:00 | mild_down   | mid       |           -0.0322 |          0.0041 |                      0.0363 |                0 |
| 2023-03-31 00:00:00 | mild_down   | narrow    |            0.0523 |          0.0838 |                      0.0315 |                0 |
| 2022-07-31 00:00:00 | mild_up     | mid       |           -0.0516 |         -0.0227 |                      0.0289 |                3 |
| 2022-01-31 00:00:00 | strong_down | narrow    |            0.0726 |          0.0726 |                      0.0000 |                0 |
| 2024-03-31 00:00:00 | mild_up     | wide      |           -0.0416 |         -0.0416 |                      0.0000 |                3 |
| 2023-06-30 00:00:00 | neutral     | mid       |           -0.0412 |         -0.0412 |                      0.0000 |                0 |
| 2023-07-31 00:00:00 | neutral     | mid       |            0.0452 |          0.0452 |                      0.0000 |                0 |
| 2023-08-31 00:00:00 | strong_down | narrow    |           -0.0124 |         -0.0124 |                      0.0000 |                0 |
| 2021-02-28 00:00:00 | mild_up     | mid       |            0.0256 |          0.0256 |                      0.0000 |                0 |
| 2023-10-31 00:00:00 | mild_down   | narrow    |           -0.0296 |         -0.0296 |                      0.0000 |                0 |
| 2023-11-30 00:00:00 | mild_up     | mid       |           -0.0433 |         -0.0433 |                      0.0000 |                0 |
| 2023-12-31 00:00:00 | mild_down   | narrow    |            0.0033 |          0.0033 |                      0.0000 |                0 |

| active_replacement_month   | regime      | breadth   |   months |   avg_selected_pairs |   mean_monthly_delta |   median_monthly_delta |   positive_delta_share |
|:---------------------------|:------------|:----------|---------:|---------------------:|---------------------:|-----------------------:|-----------------------:|
| False                      | mild_down   | mid       |        4 |               0.0000 |               0.0000 |                -0.0062 |                 0.2500 |
| False                      | mild_down   | narrow    |        8 |               0.0000 |               0.0039 |                 0.0000 |                 0.1250 |
| False                      | mild_down   | wide      |        1 |               0.0000 |              -0.0123 |                -0.0123 |                 0.0000 |
| False                      | mild_up     | mid       |        6 |               0.0000 |              -0.0000 |                 0.0000 |                 0.0000 |
| False                      | mild_up     | wide      |        2 |               0.0000 |              -0.0089 |                -0.0089 |                 0.0000 |
| False                      | neutral     | mid       |        9 |               0.0000 |              -0.0008 |                 0.0000 |                 0.0000 |
| False                      | neutral     | wide      |        1 |               0.0000 |              -0.0001 |                -0.0001 |                 0.0000 |
| False                      | strong_down | narrow    |       10 |               0.0000 |              -0.0000 |                 0.0000 |                 0.1000 |
| False                      | strong_up   | mid       |        1 |               0.0000 |               0.0000 |                 0.0000 |                 0.0000 |
| False                      | strong_up   | wide      |       11 |               0.0000 |              -0.0015 |                 0.0000 |                 0.0000 |
| True                       | mild_up     | mid       |        3 |               3.0000 |               0.0095 |                 0.0000 |                 0.3333 |
| True                       | mild_up     | wide      |        2 |               3.0000 |               0.0000 |                 0.0000 |                 0.0000 |
| True                       | neutral     | mid       |        2 |               3.0000 |              -0.0001 |                -0.0001 |                 0.0000 |
| True                       | strong_down | mid       |        1 |               3.0000 |              -0.0003 |                -0.0003 |                 0.0000 |
| True                       | strong_down | narrow    |        1 |               3.0000 |               0.0000 |                 0.0000 |                 0.0000 |
| True                       | strong_down | wide      |        1 |               3.0000 |               0.0000 |                 0.0000 |                 0.0000 |
| True                       | strong_up   | wide      |        1 |               3.0000 |               0.0000 |                 0.0000 |                 0.0000 |

## 2. Selected Pair Slices

### By Year

|      year |   selected_pairs |   realized_win_rate |   avg_realized_pair_edge |   median_realized_pair_edge |   avg_pair_edge_score |
|----------:|-----------------:|--------------------:|-------------------------:|----------------------------:|----------------------:|
| 2022.0000 |           9.0000 |              0.6667 |                   0.0267 |                      0.0184 |                0.8110 |
| 2023.0000 |           3.0000 |              1.0000 |                   0.1595 |                      0.1166 |                0.7960 |
| 2024.0000 |           9.0000 |              0.3333 |                   0.0589 |                     -0.0677 |                0.8139 |
| 2025.0000 |           9.0000 |              0.3333 |                  -0.0671 |                     -0.0692 |                0.8198 |
| 2026.0000 |           3.0000 |              0.3333 |                   0.0105 |                     -0.1914 |                0.8026 |

### By Replacement Slot

|   replacement_order |   selected_pairs |   realized_win_rate |   avg_realized_pair_edge |   median_realized_pair_edge |   avg_pair_edge_score |
|--------------------:|-----------------:|--------------------:|-------------------------:|----------------------------:|----------------------:|
|              1.0000 |          11.0000 |              0.2727 |                  -0.0358 |                     -0.0692 |                0.8948 |
|              2.0000 |          11.0000 |              0.6364 |                   0.0567 |                      0.0184 |                0.8106 |
|              3.0000 |          11.0000 |              0.5455 |                   0.0406 |                      0.0161 |                0.7308 |

### By New Industry

| new_industry   |   selected_pairs |   realized_win_rate |   avg_realized_pair_edge |   median_realized_pair_edge |   avg_pair_edge_score |
|:---------------|-----------------:|--------------------:|-------------------------:|----------------------------:|----------------------:|
| 通信           |                5 |              0.8000 |                   0.2342 |                      0.1662 |                0.8222 |
| 电力设备       |                4 |              0.5000 |                   0.0779 |                      0.0785 |                0.8197 |
| 电子           |                4 |              0.5000 |                  -0.0451 |                     -0.0504 |                0.7717 |
| 医药生物       |                3 |              0.3333 |                  -0.0317 |                     -0.0200 |                0.8278 |
| 机械设备       |                3 |              0.3333 |                  -0.1571 |                     -0.2264 |                0.8457 |
| 计算机         |                2 |              1.0000 |                   0.1810 |                      0.1810 |                0.7892 |
| 汽车           |                2 |              0.0000 |                  -0.1401 |                     -0.1401 |                0.7666 |
| 环保           |                1 |              1.0000 |                   0.5045 |                      0.5045 |                0.7840 |
| 纺织服饰       |                1 |              1.0000 |                   0.0874 |                      0.0874 |                0.7408 |
| 公用事业       |                1 |              1.0000 |                   0.0186 |                      0.0186 |                0.7838 |
| 食品饮料       |                1 |              1.0000 |                   0.0184 |                      0.0184 |                0.7909 |
| 有色金属       |                1 |              0.0000 |                  -0.0055 |                     -0.0055 |                0.7782 |
| 石油石化       |                1 |              0.0000 |                  -0.0677 |                     -0.0677 |                0.8995 |
| 社会服务       |                1 |              0.0000 |                  -0.0692 |                     -0.0692 |                0.9226 |
| 美容护理       |                1 |              0.0000 |                  -0.1550 |                     -0.1550 |                0.8668 |

## 3. Feature Bucket Monotonicity

| feature                 |   bucket |   pair_count |   replace_win_rate |   mean_pair_edge |   bucket_edge_spearman |   bucket_win_spearman |
|:------------------------|---------:|-------------:|-------------------:|-----------------:|-----------------------:|----------------------:|
| candidate_score_pct     |        1 |          951 |             0.3617 |          -0.0274 |                 0.5000 |                0.6000 |
| candidate_score_pct     |        2 |          951 |             0.3964 |          -0.0137 |                 0.5000 |                0.6000 |
| candidate_score_pct     |        3 |          951 |             0.3954 |          -0.0025 |                 0.5000 |                0.6000 |
| candidate_score_pct     |        4 |          951 |             0.3722 |          -0.0164 |                 0.5000 |                0.6000 |
| candidate_score_pct     |        5 |          951 |             0.3996 |          -0.0056 |                 0.5000 |                0.6000 |
| score_margin            |        1 |          951 |             0.4332 |           0.0058 |                -0.1000 |               -0.2052 |
| score_margin            |        2 |          951 |             0.3333 |          -0.0381 |                -0.1000 |               -0.2052 |
| score_margin            |        3 |          951 |             0.4332 |           0.0091 |                -0.1000 |               -0.2052 |
| score_margin            |        4 |          951 |             0.4458 |           0.0163 |                -0.1000 |               -0.2052 |
| score_margin            |        5 |          951 |             0.2797 |          -0.0586 |                -0.1000 |               -0.2052 |
| pair_edge_score         |        1 |          951 |             0.4143 |          -0.0092 |                -0.7000 |               -0.9000 |
| pair_edge_score         |        2 |          951 |             0.4248 |          -0.0040 |                -0.7000 |               -0.9000 |
| pair_edge_score         |        3 |          951 |             0.3933 |          -0.0080 |                -0.7000 |               -0.9000 |
| pair_edge_score         |        4 |          951 |             0.3617 |          -0.0151 |                -0.7000 |               -0.9000 |
| pair_edge_score         |        5 |          951 |             0.3312 |          -0.0293 |                -0.7000 |               -0.9000 |
| rel_strength_diff       |        1 |          951 |             0.4364 |          -0.0029 |                 0.0000 |               -0.7000 |
| rel_strength_diff       |        2 |          951 |             0.3722 |          -0.0198 |                 0.0000 |               -0.7000 |
| rel_strength_diff       |        3 |          951 |             0.3775 |          -0.0187 |                 0.0000 |               -0.7000 |
| rel_strength_diff       |        4 |          951 |             0.3743 |          -0.0129 |                 0.0000 |               -0.7000 |
| rel_strength_diff       |        5 |          951 |             0.3649 |          -0.0114 |                 0.0000 |               -0.7000 |
| amount_expansion_diff   |        1 |          951 |             0.4637 |           0.0035 |                -0.9000 |               -1.0000 |
| amount_expansion_diff   |        2 |          951 |             0.4196 |          -0.0097 |                -0.9000 |               -1.0000 |
| amount_expansion_diff   |        3 |          951 |             0.3722 |          -0.0141 |                -0.9000 |               -1.0000 |
| amount_expansion_diff   |        4 |          951 |             0.3617 |          -0.0111 |                -0.9000 |               -1.0000 |
| amount_expansion_diff   |        5 |          951 |             0.3081 |          -0.0343 |                -0.9000 |               -1.0000 |
| turnover_expansion_diff |        1 |          951 |             0.4564 |           0.0004 |                -1.0000 |               -1.0000 |
| turnover_expansion_diff |        2 |          951 |             0.4196 |          -0.0085 |                -1.0000 |               -1.0000 |
| turnover_expansion_diff |        3 |          951 |             0.3859 |          -0.0124 |                -1.0000 |               -1.0000 |
| turnover_expansion_diff |        4 |          951 |             0.3428 |          -0.0173 |                -1.0000 |               -1.0000 |
| turnover_expansion_diff |        5 |          951 |             0.3207 |          -0.0277 |                -1.0000 |               -1.0000 |
| old_defensive_score     |        1 |          962 |             0.2994 |          -0.0403 |                 0.0000 |                0.0000 |
| old_defensive_score     |        2 |         1075 |             0.5126 |           0.0316 |                 0.0000 |                0.0000 |
| old_defensive_score     |        3 |          955 |             0.4115 |          -0.0093 |                 0.0000 |                0.0000 |
| old_defensive_score     |        4 |          953 |             0.3536 |          -0.0231 |                 0.0000 |                0.0000 |
| old_defensive_score     |        5 |          810 |             0.3235 |          -0.0330 |                 0.0000 |                0.0000 |

## 4. Threshold / Capacity Sensitivity

| candidate_id                |   daily_proxy_annualized_excess_vs_market |   delta_vs_baseline_proxy |   strong_up_median_excess |   delta_vs_baseline_strong_up_median_excess |   strong_up_positive_share |   delta_vs_baseline_strong_up_positive_share |   selected_pairs |   active_rebalance_share |   avg_realized_pair_edge |   realized_win_rate |
|:----------------------------|------------------------------------------:|--------------------------:|--------------------------:|--------------------------------------------:|---------------------------:|---------------------------------------------:|-----------------:|-------------------------:|-------------------------:|--------------------:|
| U3_A_audit_thr0.68_replace3 |                                   -0.0834 |                    0.0025 |                   -0.0616 |                                     -0.0004 |                     0.1538 |                                       0.0000 |               33 |                   0.1719 |                   0.0205 |              0.4848 |
| U3_A_audit_thr0.68_replace2 |                                   -0.0850 |                    0.0010 |                   -0.0671 |                                     -0.0059 |                     0.1538 |                                       0.0000 |               22 |                   0.1719 |                   0.0105 |              0.4545 |
| U3_A_audit_thr0.75_replace2 |                                   -0.0850 |                    0.0010 |                   -0.0671 |                                     -0.0059 |                     0.1538 |                                       0.0000 |               22 |                   0.1719 |                   0.0105 |              0.4545 |
| U3_A_audit_thr0.80_replace2 |                                   -0.0855 |                    0.0004 |                   -0.0612 |                                      0.0000 |                     0.1538 |                                       0.0000 |               17 |                   0.1719 |                  -0.0060 |              0.4118 |
| U3_A_audit_thr0.80_replace3 |                                   -0.0855 |                    0.0004 |                   -0.0612 |                                      0.0000 |                     0.1538 |                                       0.0000 |               17 |                   0.1719 |                  -0.0060 |              0.4118 |
| U3_A_audit_thr0.75_replace3 |                                   -0.0866 |                   -0.0006 |                   -0.0671 |                                     -0.0059 |                     0.1538 |                                       0.0000 |               26 |                   0.1719 |                   0.0016 |              0.4615 |
| U3_A_audit_thr0.85_replace2 |                                   -0.0884 |                   -0.0024 |                   -0.0612 |                                      0.0000 |                     0.1538 |                                       0.0000 |               12 |                   0.1719 |                  -0.0327 |              0.3333 |
| U3_A_audit_thr0.85_replace3 |                                   -0.0884 |                   -0.0024 |                   -0.0612 |                                      0.0000 |                     0.1538 |                                       0.0000 |               12 |                   0.1719 |                  -0.0327 |              0.3333 |
| U3_A_audit_thr0.68_replace1 |                                   -0.0884 |                   -0.0025 |                   -0.0612 |                                      0.0000 |                     0.1538 |                                       0.0000 |               11 |                   0.1719 |                  -0.0358 |              0.2727 |
| U3_A_audit_thr0.75_replace1 |                                   -0.0884 |                   -0.0025 |                   -0.0612 |                                      0.0000 |                     0.1538 |                                       0.0000 |               11 |                   0.1719 |                  -0.0358 |              0.2727 |
| U3_A_audit_thr0.80_replace1 |                                   -0.0884 |                   -0.0025 |                   -0.0612 |                                      0.0000 |                     0.1538 |                                       0.0000 |               11 |                   0.1719 |                  -0.0358 |              0.2727 |
| U3_A_audit_thr0.85_replace1 |                                   -0.0884 |                   -0.0025 |                   -0.0612 |                                      0.0000 |                     0.1538 |                                       0.0000 |               11 |                   0.1719 |                  -0.0358 |              0.2727 |

## 5. Simple Baseline Comparison

| candidate_id                           |   daily_proxy_annualized_excess_vs_market |   delta_vs_baseline_proxy |   strong_up_median_excess |   delta_vs_baseline_strong_up_median_excess |   strong_up_positive_share |   delta_vs_baseline_strong_up_positive_share |   selected_pairs |   active_rebalance_share |   avg_realized_pair_edge |   realized_win_rate |
|:---------------------------------------|------------------------------------------:|--------------------------:|--------------------------:|--------------------------------------------:|---------------------------:|---------------------------------------------:|-----------------:|-------------------------:|-------------------------:|--------------------:|
| U3_A_simple_candidate_score_pct_only   |                                   -0.0847 |                    0.0012 |                   -0.0724 |                                     -0.0112 |                     0.1538 |                                       0.0000 |               33 |                   0.1719 |                   0.0157 |              0.3939 |
| U3_A_simple_amount_expansion_diff_only |                                   -0.0867 |                   -0.0007 |                   -0.0725 |                                     -0.0113 |                     0.1538 |                                       0.0000 |               32 |                   0.1719 |                  -0.0144 |              0.4062 |
| U3_A_simple_score_margin_only          |                                   -0.0880 |                   -0.0021 |                   -0.0612 |                                      0.0000 |                     0.1538 |                                       0.0000 |               20 |                   0.1719 |                  -0.0213 |              0.4000 |
| U3_A_simple_overheat_relief_only       |                                   -0.0905 |                   -0.0046 |                   -0.0724 |                                     -0.0112 |                     0.1538 |                                       0.0000 |                3 |                   0.0156 |                  -0.0609 |              0.3333 |
| U3_A_simple_rel_strength_diff_only     |                                   -0.0958 |                   -0.0099 |                   -0.0744 |                                     -0.0132 |                     0.1538 |                                       0.0000 |               33 |                   0.1719 |                  -0.0321 |              0.3939 |

## 6. 判定

- U3_A gray-zone 仍弱，当前不足以启动复杂 R3；最多继续做窄规则复核。
- 本轮 audit 仍不产生 production candidate，不补正式 full backtest，不写默认配置。

## 7. 产出文件

- `data/results/r2b_v2_weight_audit_2026-04-28_monthly_attribution.csv`
- `data/results/r2b_v2_weight_audit_2026-04-28_monthly_attribution_summary.csv`
- `data/results/r2b_v2_weight_audit_2026-04-28_feature_bucket_monotonicity.csv`
- `data/results/r2b_v2_weight_audit_2026-04-28_selected_pairs_by_year.csv`
- `data/results/r2b_v2_weight_audit_2026-04-28_selected_pairs_by_slot.csv`
- `data/results/r2b_v2_weight_audit_2026-04-28_selected_pairs_by_industry.csv`
- `data/results/r2b_v2_weight_audit_2026-04-28_threshold_capacity_sensitivity.csv`
- `data/results/r2b_v2_weight_audit_2026-04-28_simple_baseline_comparison.csv`
- `data/results/r2b_v2_weight_audit_2026-04-28_summary.json`
