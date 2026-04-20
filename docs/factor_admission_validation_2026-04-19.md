# F1 候选因子准入验证

- 生成时间：`2026-04-19T14:02:15.537703+00:00`
- 基线：`S2 = vol_to_turnover`，并沿用当前默认研究口径 `prefilter=false + universe=true`
- 候选：`gross_margin_delta`、`ocf_to_asset`
- 比较矩阵：基线、候选单因子、候选 `10% / 20%` 叠加版

## 组合汇总

| scenario              | candidate_factor   | family    | is_baseline   |   annualized_return |   sharpe_ratio |   max_drawdown |   turnover_mean |   rolling_oos_median_ann_return |   slice_oos_median_ann_return |
|:----------------------|:-------------------|:----------|:--------------|--------------------:|---------------:|---------------:|----------------:|--------------------------------:|------------------------------:|
| A0_baseline_s2        | vol_to_turnover    | baseline  | True          |          0.0373737  |     0.318814   |       0.201539 |        0.186486 |                       0.094812  |                     0.0372804 |
| A1_gross_single       | gross_margin_delta | gross     | False         |         -0.413043   |    -2.03692    |       0.933712 |        0.225424 |                      -0.391807  |                    -0.413609  |
| A2_gross_blend_10     | gross_margin_delta | gross     | False         |          0.00457803 |     0.106796   |       0.28807  |        0.180263 |                       0.0929644 |                     0.0324653 |
| A3_gross_blend_20     | gross_margin_delta | gross     | False         |         -0.0424579  |    -0.169726   |       0.363431 |        0.1625   |                       0.0101913 |                    -0.0223068 |
| A4_ocf_asset_single   | ocf_to_asset       | ocf_asset | False         |         -0.201689   |    -1.09642    |       0.707409 |        0.22377  |                      -0.272698  |                    -0.249353  |
| A5_ocf_asset_blend_10 | ocf_to_asset       | ocf_asset | False         |          0.0192868  |     0.201428   |       0.219061 |        0.208108 |                       0.0704841 |                     0.0423601 |
| A6_ocf_asset_blend_20 | ocf_to_asset       | ocf_asset | False         |         -0.0125554  |    -0.00865735 |       0.260719 |        0.175    |                       0.030264  |                     0.0250372 |

## IC 门槛

| factor             | horizon_key    |   n_dates |    ic_mean |   ic_t_value |
|:-------------------|:---------------|----------:|-----------:|-------------:|
| gross_margin_delta | close_21d      |        74 | 0.0058882  |     1.60132  |
| gross_margin_delta | tplus1_open_1d |        75 | 0.00258165 |     0.889182 |
| ocf_to_asset       | close_21d      |        74 | 0.0100361  |     2.55847  |
| ocf_to_asset       | tplus1_open_1d |        75 | 0.00270138 |     0.696922 |

## 准入结论

| scenario              | candidate_factor   | family    | is_baseline   |   annualized_return |   sharpe_ratio |   max_drawdown |   turnover_mean |   rolling_oos_median_ann_return |   slice_oos_median_ann_return | factor             |   close21_ic_mean |   close21_ic_t |   open1_ic_mean |   open1_ic_t |   pass_f1_gate |   delta_ann_vs_baseline |   delta_sharpe_vs_baseline |   delta_rolling_vs_baseline |   delta_slice_vs_baseline | pass_combo_gate   | admission_status   |
|:----------------------|:-------------------|:----------|:--------------|--------------------:|---------------:|---------------:|----------------:|--------------------------------:|------------------------------:|:-------------------|------------------:|---------------:|----------------:|-------------:|---------------:|------------------------:|---------------------------:|----------------------------:|--------------------------:|:------------------|:-------------------|
| A0_baseline_s2        | vol_to_turnover    | baseline  | True          |          0.0373737  |     0.318814   |       0.201539 |        0.186486 |                       0.094812  |                     0.0372804 | nan                |       nan         |      nan       |    nan          |   nan        |            nan |               0         |                   0        |                  0          |                0          | False             | baseline           |
| A1_gross_single       | gross_margin_delta | gross     | False         |         -0.413043   |    -2.03692    |       0.933712 |        0.225424 |                      -0.391807  |                    -0.413609  | gross_margin_delta |         0.0058882 |        1.60132 |      0.00258165 |     0.889182 |              0 |              -0.450417  |                  -2.35574  |                 -0.486619   |               -0.450889   | False             | fail               |
| A2_gross_blend_10     | gross_margin_delta | gross     | False         |          0.00457803 |     0.106796   |       0.28807  |        0.180263 |                       0.0929644 |                     0.0324653 | gross_margin_delta |         0.0058882 |        1.60132 |      0.00258165 |     0.889182 |              0 |              -0.0327957 |                  -0.212019 |                 -0.00184758 |               -0.00481513 | False             | fail               |
| A3_gross_blend_20     | gross_margin_delta | gross     | False         |         -0.0424579  |    -0.169726   |       0.363431 |        0.1625   |                       0.0101913 |                    -0.0223068 | gross_margin_delta |         0.0058882 |        1.60132 |      0.00258165 |     0.889182 |              0 |              -0.0798317 |                  -0.488541 |                 -0.0846207  |               -0.0595872  | False             | fail               |
| A4_ocf_asset_single   | ocf_to_asset       | ocf_asset | False         |         -0.201689   |    -1.09642    |       0.707409 |        0.22377  |                      -0.272698  |                    -0.249353  | ocf_to_asset       |         0.0100361 |        2.55847 |      0.00270138 |     0.696922 |              1 |              -0.239063  |                  -1.41523  |                 -0.36751    |               -0.286633   | False             | fail               |
| A5_ocf_asset_blend_10 | ocf_to_asset       | ocf_asset | False         |          0.0192868  |     0.201428   |       0.219061 |        0.208108 |                       0.0704841 |                     0.0423601 | ocf_to_asset       |         0.0100361 |        2.55847 |      0.00270138 |     0.696922 |              1 |              -0.0180869 |                  -0.117387 |                 -0.0243279  |                0.00507967 | False             | fail               |
| A6_ocf_asset_blend_20 | ocf_to_asset       | ocf_asset | False         |         -0.0125554  |    -0.00865735 |       0.260719 |        0.175    |                       0.030264  |                     0.0250372 | ocf_to_asset       |         0.0100361 |        2.55847 |      0.00270138 |     0.696922 |              1 |              -0.0499291 |                  -0.327472 |                 -0.064548   |               -0.0122432  | False             | fail               |

## 本轮产物

- `data/results/factor_admission_validation_2026-04-19_summary.csv`
- `data/results/factor_admission_validation_2026-04-19_factor_gate.csv`
- `data/results/factor_admission_validation_2026-04-19_admission.csv`
- `data/results/factor_admission_validation_2026-04-19_a*.json`
