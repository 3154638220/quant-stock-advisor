# Score 消融实验

- 生成时间：`2026-04-19T10:12:17.548825+00:00`
- 区间：`2021-01-01` 到 `2026-04-19`
- 固定口径：`tplus1_open` / `M` / `top_k=20` / `max_turnover=0.3` / `universe_filter=false`（S5 为外部对照，沿用 P1 静态原参数）

## 全样本与 OOS 汇总

| scenario      |   annualized_return |   sharpe_ratio |   max_drawdown |   turnover_mean |   rolling_oos_median_ann_return | rolling_oos_median_sharpe   |   slice_oos_median_ann_return | slice_oos_median_sharpe   |
|:--------------|--------------------:|---------------:|---------------:|----------------:|--------------------------------:|:----------------------------|------------------------------:|:--------------------------|
| S1_ocf_single |           -0.289365 |      -1.26015  |       0.83885  |        0.260484 |                      -0.264854  |                             |                    -0.376183  |                           |
| S2_vol_single |           -0.101905 |      -0.457181 |       0.510759 |        0.269355 |                      -0.0363144 |                             |                    -0.0424956 |                           |
| S3_dual_7030  |           -0.251672 |      -1.13627  |       0.78584  |        0.254918 |                      -0.210502  |                             |                    -0.276818  |                           |
| S4_legacy3    |           -0.249392 |      -1.13769  |       0.785647 |        0.301613 |                      -0.194303  |                             |                    -0.316478  |                           |
| S5_p1_static  |           -0.176153 |      -0.943241 |       0.65285  |        0.262903 |                      -0.198043  |                             |                    -0.157246  |                           |

## 与 S5 的 Top-K 重合度

| scenario      | reference    |   mean_overlap_count |   median_overlap_count |   mean_overlap_ratio_vs_candidate |   mean_overlap_ratio_vs_reference |   mean_candidate_only_count |   mean_reference_only_count |
|:--------------|:-------------|---------------------:|-----------------------:|----------------------------------:|----------------------------------:|----------------------------:|----------------------------:|
| S1_ocf_single | S5_p1_static |             0.301587 |                      0 |                         0.0150794 |                        0.00603175 |                     19.6984 |                     49.6984 |
| S2_vol_single | S5_p1_static |            20        |                     20 |                         1         |                        0.4        |                      0      |                     30      |
| S3_dual_7030  | S5_p1_static |             1.25397  |                      1 |                         0.0626984 |                        0.0250794  |                     18.746  |                     48.746  |
| S4_legacy3    | S5_p1_static |             1.2381   |                      1 |                         0.0619048 |                        0.0247619  |                     18.7619 |                     48.7619 |

## 分数相关性

| left          | right         |   common_rows |   median_daily_corr |   mean_daily_corr |
|:--------------|:--------------|--------------:|--------------------:|------------------:|
| S1_ocf_single | S2_vol_single |       7008366 |         0.000540348 |         0.0013187 |
| S1_ocf_single | S3_dual_7030  |       7008366 |         0.861051    |         0.862741  |
| S1_ocf_single | S4_legacy3    |       7008366 |         0.711813    |         0.710915  |
| S1_ocf_single | S5_p1_static  |       7008366 |         0.000540348 |         0.0013187 |
| S2_vol_single | S3_dual_7030  |       7008366 |         0.509071    |         0.506398  |
| S2_vol_single | S4_legacy3    |       7008366 |         0.592495    |         0.591773  |
| S2_vol_single | S5_p1_static  |       7008366 |         1           |         1         |
| S3_dual_7030  | S4_legacy3    |       7008366 |         0.912029    |         0.912212  |
| S3_dual_7030  | S5_p1_static  |       7008366 |         0.509071    |         0.506398  |
| S4_legacy3    | S5_p1_static  |       7008366 |         0.592495    |         0.591773  |

## 产物

- `data/results/score_ablation_2026-04-19_summary.csv`
- `data/results/score_ablation_2026-04-19_coverage_summary.csv`
- `data/results/score_ablation_2026-04-19_score_corr.csv`
- `data/results/score_ablation_2026-04-19_topk_overlap_summary.csv`