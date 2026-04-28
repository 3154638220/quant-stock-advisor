# R2B Oracle Replacement Attribution

- 生成时间：`2026-04-28T09:29:01.978552+00:00`
- `eval_contract_version`: `r0_eval_execution_contract_2026-04-28`
- `execution_contract_version`: `tplus1_open_buy_delta_limit_mask_2026-04-28`
- `industry_map_source`: `akshare.stock_industry_clf_hist_sw+stock_industry_category_cninfo`
- `industry_map_source_status`: `real_industry_map`
- 目标：先判断边界 replacement 是否存在事后可学习上限，再决定是否进入 R2B v2。

## 1. Oracle Replace-3 理论上限

| candidate_id                                                          |   daily_proxy_annualized_excess_vs_market |   best_possible_replace_3_excess |   avg_turnover_half_l1 |   oracle_selected_pairs |   active_rebalance_share |
|:----------------------------------------------------------------------|------------------------------------------:|---------------------------------:|-----------------------:|------------------------:|-------------------------:|
| ORACLE_score__u2_c__S2_bottom_5__candidate_buyable__all_states        |                                    2.6608 |                           2.7468 |                 0.2188 |                     186 |                   0.9688 |
| ORACLE_score__u2_a__S2_bottom_5__candidate_buyable__all_states        |                                    2.6608 |                           2.7468 |                 0.2188 |                     186 |                   0.9688 |
| ORACLE_score__u2_b__S2_bottom_5__candidate_buyable__all_states        |                                    2.6608 |                           2.7468 |                 0.2188 |                     186 |                   0.9688 |
| ORACLE_score__u2_a__S2_bottom_3__candidate_buyable__all_states        |                                    2.4737 |                           2.5596 |                 0.1633 |                     186 |                   0.9688 |
| ORACLE_score__u2_c__S2_bottom_3__candidate_buyable__all_states        |                                    2.4737 |                           2.5596 |                 0.1633 |                     186 |                   0.9688 |
| ORACLE_score__u2_b__S2_bottom_3__candidate_buyable__all_states        |                                    2.4737 |                           2.5596 |                 0.1633 |                     186 |                   0.9688 |
| ORACLE_score__u2_a__S2_bottom_5__candidate_top_pct_90__all_states     |                                    1.3948 |                           1.4807 |                 0.2180 |                     186 |                   0.9688 |
| ORACLE_score__u2_c__S2_bottom_5__candidate_top_pct_90__all_states     |                                    1.3829 |                           1.4688 |                 0.2195 |                     186 |                   0.9688 |
| ORACLE_score__u2_b__S2_bottom_5__candidate_top_pct_90__all_states     |                                    1.3221 |                           1.4080 |                 0.2195 |                     186 |                   0.9688 |
| ORACLE_score__u2_a__S2_bottom_3__candidate_top_pct_90__all_states     |                                    1.2723 |                           1.3582 |                 0.1625 |                     186 |                   0.9688 |
| ORACLE_score__u2_c__S2_bottom_3__candidate_top_pct_90__all_states     |                                    1.2610 |                           1.3470 |                 0.1641 |                     186 |                   0.9688 |
| ORACLE_score__u2_b__S2_bottom_3__candidate_top_pct_90__all_states     |                                    1.2033 |                           1.2892 |                 0.1641 |                     186 |                   0.9688 |
| ORACLE_score__u2_a__S2_bottom_5__candidate_top_pct_95__all_states     |                                    1.1097 |                           1.1957 |                 0.2172 |                     186 |                   0.9688 |
| ORACLE_score__u2_b__S2_bottom_5__candidate_top_pct_95__all_states     |                                    1.0537 |                           1.1396 |                 0.2172 |                     186 |                   0.9688 |
| ORACLE_score__u2_c__S2_bottom_5__candidate_top_pct_95__all_states     |                                    1.0164 |                           1.1023 |                 0.2188 |                     186 |                   0.9688 |
| ORACLE_score__u2_a__S2_bottom_3__candidate_top_pct_95__all_states     |                                    0.9994 |                           1.0853 |                 0.1617 |                     184 |                   0.9688 |
| ORACLE_score__u2_b__S2_bottom_3__candidate_top_pct_95__all_states     |                                    0.9475 |                           1.0335 |                 0.1617 |                     185 |                   0.9688 |
| ORACLE_score__u2_c__S2_bottom_3__candidate_top_pct_95__all_states     |                                    0.9131 |                           0.9991 |                 0.1633 |                     186 |                   0.9688 |
| ORACLE_score__u2_a__S2_bottom_5__candidate_buyable__strong_up_or_wide |                                    0.2742 |                           0.3601 |                 0.1423 |                      48 |                   0.2500 |
| ORACLE_score__u2_b__S2_bottom_5__candidate_buyable__strong_up_or_wide |                                    0.2742 |                           0.3601 |                 0.1423 |                      48 |                   0.2500 |

## 2. Oracle Capacity

| score_col   | old_pool    | candidate_pool       | state_gate        |   avg_oracle_positive_slots |   active_month_share |   full_3_slot_month_share |   avg_selected_edge_when_active |   avg_sum_edge_per_month |
|:------------|:------------|:---------------------|:------------------|----------------------------:|---------------------:|--------------------------:|--------------------------------:|-------------------------:|
| score__u2_a | S2_bottom_5 | candidate_buyable    | all_states        |                      3.0000 |               1.0000 |                    1.0000 |                          1.0286 |                   3.0858 |
| score__u2_b | S2_bottom_5 | candidate_buyable    | all_states        |                      3.0000 |               1.0000 |                    1.0000 |                          1.0286 |                   3.0858 |
| score__u2_c | S2_bottom_5 | candidate_buyable    | all_states        |                      3.0000 |               1.0000 |                    1.0000 |                          1.0286 |                   3.0858 |
| score__u2_a | S2_bottom_3 | candidate_buyable    | all_states        |                      3.0000 |               1.0000 |                    1.0000 |                          0.9962 |                   2.9886 |
| score__u2_b | S2_bottom_3 | candidate_buyable    | all_states        |                      3.0000 |               1.0000 |                    1.0000 |                          0.9962 |                   2.9886 |
| score__u2_c | S2_bottom_3 | candidate_buyable    | all_states        |                      3.0000 |               1.0000 |                    1.0000 |                          0.9962 |                   2.9886 |
| score__u2_a | S2_bottom_5 | candidate_top_pct_90 | all_states        |                      3.0000 |               1.0000 |                    1.0000 |                          0.6471 |                   1.9414 |
| score__u2_c | S2_bottom_5 | candidate_top_pct_90 | all_states        |                      3.0000 |               1.0000 |                    1.0000 |                          0.6451 |                   1.9353 |
| score__u2_b | S2_bottom_5 | candidate_top_pct_90 | all_states        |                      3.0000 |               1.0000 |                    1.0000 |                          0.6334 |                   1.9002 |
| score__u2_a | S2_bottom_3 | candidate_top_pct_90 | all_states        |                      3.0000 |               1.0000 |                    1.0000 |                          0.6147 |                   1.8441 |
| score__u2_c | S2_bottom_3 | candidate_top_pct_90 | all_states        |                      3.0000 |               1.0000 |                    1.0000 |                          0.6127 |                   1.8380 |
| score__u2_b | S2_bottom_3 | candidate_top_pct_90 | all_states        |                      3.0000 |               1.0000 |                    1.0000 |                          0.6010 |                   1.8030 |
| score__u2_a | S2_bottom_5 | candidate_top_pct_95 | all_states        |                      3.0000 |               1.0000 |                    1.0000 |                          0.5398 |                   1.6193 |
| score__u2_b | S2_bottom_5 | candidate_top_pct_95 | all_states        |                      3.0000 |               1.0000 |                    1.0000 |                          0.5190 |                   1.5569 |
| score__u2_a | S2_bottom_3 | candidate_top_pct_95 | all_states        |                      2.9677 |               1.0000 |                    0.9677 |                          0.5098 |                   1.5222 |
| score__u2_c | S2_bottom_5 | candidate_top_pct_95 | all_states        |                      3.0000 |               1.0000 |                    1.0000 |                          0.5070 |                   1.5211 |
| score__u2_b | S2_bottom_3 | candidate_top_pct_95 | all_states        |                      2.9839 |               1.0000 |                    0.9839 |                          0.4871 |                   1.4597 |
| score__u2_c | S2_bottom_3 | candidate_top_pct_95 | all_states        |                      3.0000 |               1.0000 |                    1.0000 |                          0.4746 |                   1.4238 |
| score__u2_c | S2_bottom_5 | candidate_buyable    | strong_up_or_wide |                      0.7742 |               0.2581 |                    0.2581 |                          0.9941 |                   0.7696 |
| score__u2_a | S2_bottom_5 | candidate_buyable    | strong_up_or_wide |                      0.7742 |               0.2581 |                    0.2581 |                          0.9941 |                   0.7696 |

## 3. State Gate Precision

| score_col   | old_pool    | candidate_pool       | gate                             | gate_value   |   pair_count |   hit_rate |   lift_vs_all |   mean_pair_edge |
|:------------|:------------|:---------------------|:---------------------------------|:-------------|-------------:|-----------:|--------------:|-----------------:|
| score__u2_a | S2_bottom_5 | candidate_top_pct_95 | state_strong_up_or_wide          | False        |        27586 |     0.4474 |        0.0249 |          -0.0015 |
| score__u2_a | S2_bottom_3 | candidate_top_pct_95 | state_strong_up_or_wide          | False        |        16620 |     0.4440 |        0.0235 |           0.0011 |
| score__u2_b | S2_bottom_5 | candidate_top_pct_90 | state_strong_up_or_wide          | False        |        58384 |     0.4483 |        0.0202 |          -0.0043 |
| score__u2_c | S2_bottom_5 | candidate_top_pct_95 | state_strong_up_or_wide          | False        |        29755 |     0.4579 |        0.0198 |          -0.0009 |
| score__u2_a | S2_bottom_5 | candidate_top_pct_90 | state_strong_up_or_wide          | False        |        54368 |     0.4474 |        0.0191 |          -0.0023 |
| score__u2_b | S2_bottom_5 | candidate_top_pct_95 | state_strong_up_or_wide          | False        |        29799 |     0.4489 |        0.0191 |          -0.0023 |
| score__u2_a | S2_bottom_3 | candidate_top_pct_90 | state_strong_up_or_wide          | False        |        32751 |     0.4458 |        0.0189 |           0.0005 |
| score__u2_b | S2_bottom_3 | candidate_top_pct_90 | state_strong_up_or_wide          | False        |        35166 |     0.4442 |        0.0187 |          -0.0015 |
| score__u2_a | S2_bottom_5 | candidate_top_pct_95 | state_up_or_wide_not_strong_down | False        |        28331 |     0.4410 |        0.0185 |          -0.0034 |
| score__u2_c | S2_bottom_5 | candidate_top_pct_90 | state_strong_up_or_wide          | False        |        57714 |     0.4557 |        0.0183 |          -0.0015 |
| score__u2_c | S2_bottom_3 | candidate_top_pct_95 | state_strong_up_or_wide          | False        |        17922 |     0.4546 |        0.0183 |           0.0019 |
| score__u2_a | S2_bottom_3 | candidate_top_pct_95 | state_up_or_wide_not_strong_down | False        |        17067 |     0.4386 |        0.0180 |          -0.0003 |
| score__u2_c | S2_bottom_3 | candidate_top_pct_90 | state_strong_up_or_wide          | False        |        34761 |     0.4535 |        0.0172 |           0.0016 |
| score__u2_b | S2_bottom_3 | candidate_top_pct_95 | state_strong_up_or_wide          | False        |        17949 |     0.4443 |        0.0168 |           0.0005 |
| score__u2_a | S2_bottom_3 | candidate_top_pct_90 | state_up_or_wide_not_strong_down | False        |        33636 |     0.4409 |        0.0140 |          -0.0009 |
| score__u2_b | S2_bottom_5 | candidate_top_pct_90 | state_up_or_wide_not_strong_down | False        |        60109 |     0.4414 |        0.0132 |          -0.0061 |
| score__u2_a | S2_bottom_5 | candidate_top_pct_90 | state_up_or_wide_not_strong_down | False        |        55843 |     0.4415 |        0.0132 |          -0.0040 |
| score__u2_b | S2_bottom_3 | candidate_top_pct_90 | state_up_or_wide_not_strong_down | False        |        36201 |     0.4386 |        0.0131 |          -0.0029 |
| score__u2_c | S2_bottom_5 | candidate_top_pct_95 | state_up_or_wide_not_strong_down | False        |        30620 |     0.4512 |        0.0131 |          -0.0026 |
| score__u2_c | S2_bottom_3 | candidate_top_pct_95 | state_up_or_wide_not_strong_down | False        |        18441 |     0.4492 |        0.0129 |           0.0006 |
| score__u2_c | S2_bottom_3 | candidate_buyable    | state_strong_up_or_wide          | False        |       323364 |     0.4655 |        0.0125 |           0.0007 |
| score__u2_b | S2_bottom_3 | candidate_buyable    | state_strong_up_or_wide          | False        |       323364 |     0.4655 |        0.0125 |           0.0007 |
| score__u2_a | S2_bottom_3 | candidate_buyable    | state_strong_up_or_wide          | False        |       323364 |     0.4655 |        0.0125 |           0.0007 |
| score__u2_c | S2_bottom_5 | candidate_buyable    | state_strong_up_or_wide          | False        |       537030 |     0.4642 |        0.0123 |          -0.0026 |

## 4. Feature Bucket Monotonicity

| score_col   | old_pool    | candidate_pool       | feature           |   bucket |   pair_count |   hit_rate |   mean_pair_edge |   bucket_edge_spearman |
|:------------|:------------|:---------------------|:------------------|---------:|-------------:|-----------:|-----------------:|-----------------------:|
| score__u2_a | S2_bottom_3 | candidate_buyable    | rel_strength_diff |        1 |        91387 |     0.4403 |          -0.0092 |                 1.0000 |
| score__u2_b | S2_bottom_3 | candidate_buyable    | rel_strength_diff |        1 |        91387 |     0.4403 |          -0.0092 |                 1.0000 |
| score__u2_b | S2_bottom_3 | candidate_top_pct_95 | rel_strength_diff |        1 |         5055 |     0.4121 |          -0.0168 |                 1.0000 |
| score__u2_c | S2_bottom_3 | candidate_buyable    | rel_strength_diff |        1 |        91387 |     0.4403 |          -0.0092 |                 1.0000 |
| score__u2_c | S2_bottom_3 | candidate_top_pct_95 | rel_strength_diff |        1 |         5044 |     0.4239 |          -0.0146 |                 1.0000 |
| score__u2_a | S2_bottom_3 | candidate_buyable    | rel_strength_diff |        2 |        91386 |     0.4512 |          -0.0085 |                 1.0000 |
| score__u2_b | S2_bottom_3 | candidate_buyable    | rel_strength_diff |        2 |        91386 |     0.4512 |          -0.0085 |                 1.0000 |
| score__u2_b | S2_bottom_3 | candidate_top_pct_95 | rel_strength_diff |        2 |         5054 |     0.4339 |          -0.0065 |                 1.0000 |
| score__u2_c | S2_bottom_3 | candidate_buyable    | rel_strength_diff |        2 |        91386 |     0.4512 |          -0.0085 |                 1.0000 |
| score__u2_c | S2_bottom_3 | candidate_top_pct_95 | rel_strength_diff |        2 |         5043 |     0.4329 |          -0.0070 |                 1.0000 |
| score__u2_a | S2_bottom_3 | candidate_buyable    | rel_strength_diff |        3 |        91387 |     0.4564 |          -0.0029 |                 1.0000 |
| score__u2_b | S2_bottom_3 | candidate_buyable    | rel_strength_diff |        3 |        91387 |     0.4564 |          -0.0029 |                 1.0000 |
| score__u2_b | S2_bottom_3 | candidate_top_pct_95 | rel_strength_diff |        3 |         5054 |     0.4185 |          -0.0058 |                 1.0000 |
| score__u2_c | S2_bottom_3 | candidate_buyable    | rel_strength_diff |        3 |        91387 |     0.4564 |          -0.0029 |                 1.0000 |
| score__u2_c | S2_bottom_3 | candidate_top_pct_95 | rel_strength_diff |        3 |         5044 |     0.4266 |          -0.0043 |                 1.0000 |
| score__u2_a | S2_bottom_3 | candidate_buyable    | rel_strength_diff |        4 |        91386 |     0.4631 |           0.0027 |                 1.0000 |
| score__u2_b | S2_bottom_3 | candidate_buyable    | rel_strength_diff |        4 |        91386 |     0.4631 |           0.0027 |                 1.0000 |
| score__u2_b | S2_bottom_3 | candidate_top_pct_95 | rel_strength_diff |        4 |         5054 |     0.4351 |           0.0002 |                 1.0000 |
| score__u2_c | S2_bottom_3 | candidate_buyable    | rel_strength_diff |        4 |        91386 |     0.4631 |           0.0027 |                 1.0000 |
| score__u2_c | S2_bottom_3 | candidate_top_pct_95 | rel_strength_diff |        4 |         5043 |     0.4376 |          -0.0005 |                 1.0000 |
| score__u2_a | S2_bottom_3 | candidate_buyable    | rel_strength_diff |        5 |        91387 |     0.4543 |           0.0047 |                 1.0000 |
| score__u2_b | S2_bottom_3 | candidate_buyable    | rel_strength_diff |        5 |        91387 |     0.4543 |           0.0047 |                 1.0000 |
| score__u2_b | S2_bottom_3 | candidate_top_pct_95 | rel_strength_diff |        5 |         5055 |     0.4376 |           0.0050 |                 1.0000 |
| score__u2_c | S2_bottom_3 | candidate_buyable    | rel_strength_diff |        5 |        91387 |     0.4543 |           0.0047 |                 1.0000 |
| score__u2_c | S2_bottom_3 | candidate_top_pct_95 | rel_strength_diff |        5 |         5044 |     0.4603 |           0.0132 |                 1.0000 |
| score__u2_a | S2_bottom_3 | candidate_top_pct_90 | score_margin      |        1 |         9203 |     0.3864 |          -0.0220 |                 1.0000 |
| score__u2_b | S2_bottom_3 | candidate_buyable    | score_margin      |        1 |        91387 |     0.4493 |          -0.0075 |                 1.0000 |
| score__u2_c | S2_bottom_3 | candidate_buyable    | score_margin      |        1 |        91387 |     0.4398 |          -0.0096 |                 1.0000 |
| score__u2_a | S2_bottom_3 | candidate_top_pct_90 | score_margin      |        2 |         9203 |     0.4142 |          -0.0160 |                 1.0000 |
| score__u2_b | S2_bottom_3 | candidate_buyable    | score_margin      |        2 |        91386 |     0.4483 |          -0.0060 |                 1.0000 |

## 5. 判定

- `strong_up_or_wide state-gated` 最优 oracle replace-3 的 daily proxy 增量为 `36.01%`，说明候选池存在可学习上限；允许推进 R2B v2 edge-gated replacement。`all_states` 的 `274.68%` 仅作为全样本 oracle 上限诊断，不作为状态门控策略判定口径。
- 若 feature bucket 单调性弱而 oracle 上限存在，下一步优先做 pairwise rule，不训练复杂模型。

## 6. 产出文件

- `data/results/r2b_oracle_replacement_attribution_2026-04-28_oracle_hit_rate_by_state.csv`
- `data/results/r2b_oracle_replacement_attribution_2026-04-28_oracle_capacity_by_month.csv`
- `data/results/r2b_oracle_replacement_attribution_2026-04-28_oracle_capacity_summary.csv`
- `data/results/r2b_oracle_replacement_attribution_2026-04-28_best_possible_replace_3_excess.csv`
- `data/results/r2b_oracle_replacement_attribution_2026-04-28_feature_bucket_monotonicity.csv`
- `data/results/r2b_oracle_replacement_attribution_2026-04-28_state_gate_precision.csv`
- `data/results/r2b_oracle_replacement_attribution_2026-04-28_cost_sensitivity.csv`
- `data/results/r2b_oracle_replacement_attribution_2026-04-28_oracle_selected_pairs.csv`
- `data/results/r2b_oracle_replacement_attribution_2026-04-28_summary.json`
