# F1 候选因子准入验证

- 生成时间：`2026-04-19T14:26:19.227586+00:00`
- 基线：`S2 = vol_to_turnover`，并沿用当前默认研究口径 `prefilter=false + universe=true`
- 候选：`asset_turnover`、`net_margin_stability`
- 比较矩阵：基线、候选单因子、候选 `10% / 20%` 叠加版

## 组合汇总

| scenario                    | candidate_factor     | family         | is_baseline   |   annualized_return |   sharpe_ratio |   max_drawdown |   turnover_mean |   rolling_oos_median_ann_return |   slice_oos_median_ann_return |
|:----------------------------|:---------------------|:---------------|:--------------|--------------------:|---------------:|---------------:|----------------:|--------------------------------:|------------------------------:|
| A0_baseline_s2              | vol_to_turnover      | baseline       | True          |          0.0373737  |      0.318814  |       0.201539 |        0.186486 |                      0.094812   |                     0.0372804 |
| A10_asset_turnover_single   | asset_turnover       | asset_turnover | False         |         -0.128061   |     -0.675441  |       0.52783  |        0.151887 |                     -0.117126   |                    -0.128268  |
| A11_asset_turnover_blend_10 | asset_turnover       | asset_turnover | False         |         -0.00290179 |      0.0569688 |       0.303111 |        0.18125  |                      0.00651763 |                     0.0473911 |
| A12_asset_turnover_blend_20 | asset_turnover       | asset_turnover | False         |         -0.0227069  |     -0.0763208 |       0.295603 |        0.151042 |                      0.017063   |                     0.0217649 |
| A7_net_margin_single        | net_margin_stability | net_margin     | False         |         -0.34757    |     -1.35458   |       0.891582 |        0.223387 |                     -0.221436   |                    -0.383723  |
| A8_net_margin_blend_10      | net_margin_stability | net_margin     | False         |          0.0243369  |      0.23278   |       0.245359 |        0.158333 |                      0.0852842  |                     0.0318784 |
| A9_net_margin_blend_20      | net_margin_stability | net_margin     | False         |          0.0340294  |      0.291574  |       0.249596 |        0.175    |                      0.0882541  |                     0.0384267 |

## IC 门槛

| factor               | horizon_key    |   n_dates |      ic_mean |   ic_t_value |
|:---------------------|:---------------|----------:|-------------:|-------------:|
| asset_turnover       | close_21d      |        74 |  0.000596992 |     0.112895 |
| asset_turnover       | tplus1_open_1d |        75 | -0.00281517  |    -0.499537 |
| net_margin_stability | close_21d      |        74 |  0.00803495  |     1.70989  |
| net_margin_stability | tplus1_open_1d |        75 |  0.00254756  |     0.614092 |

## 准入结论

| scenario                    | candidate_factor     | family         | is_baseline   |   annualized_return |   sharpe_ratio |   max_drawdown |   turnover_mean |   rolling_oos_median_ann_return |   slice_oos_median_ann_return | factor               |   close21_ic_mean |   close21_ic_t |   open1_ic_mean |   open1_ic_t |   pass_f1_gate |   delta_ann_vs_baseline |   delta_sharpe_vs_baseline |   delta_rolling_vs_baseline |   delta_slice_vs_baseline | pass_combo_gate   | admission_status   |
|:----------------------------|:---------------------|:---------------|:--------------|--------------------:|---------------:|---------------:|----------------:|--------------------------------:|------------------------------:|:---------------------|------------------:|---------------:|----------------:|-------------:|---------------:|------------------------:|---------------------------:|----------------------------:|--------------------------:|:------------------|:-------------------|
| A0_baseline_s2              | vol_to_turnover      | baseline       | True          |          0.0373737  |      0.318814  |       0.201539 |        0.186486 |                      0.094812   |                     0.0372804 | nan                  |     nan           |     nan        |    nan          |   nan        |            nan |              0          |                  0         |                  0          |                0          | False             | baseline           |
| A10_asset_turnover_single   | asset_turnover       | asset_turnover | False         |         -0.128061   |     -0.675441  |       0.52783  |        0.151887 |                     -0.117126   |                    -0.128268  | asset_turnover       |       0.000596992 |       0.112895 |     -0.00281517 |    -0.499537 |              0 |             -0.165435   |                 -0.994256  |                 -0.211938   |               -0.165549   | False             | fail               |
| A11_asset_turnover_blend_10 | asset_turnover       | asset_turnover | False         |         -0.00290179 |      0.0569688 |       0.303111 |        0.18125  |                      0.00651763 |                     0.0473911 | asset_turnover       |       0.000596992 |       0.112895 |     -0.00281517 |    -0.499537 |              0 |             -0.0402755  |                 -0.261846  |                 -0.0882944  |                0.0101107  | False             | fail               |
| A12_asset_turnover_blend_20 | asset_turnover       | asset_turnover | False         |         -0.0227069  |     -0.0763208 |       0.295603 |        0.151042 |                      0.017063   |                     0.0217649 | asset_turnover       |       0.000596992 |       0.112895 |     -0.00281517 |    -0.499537 |              0 |             -0.0600807  |                 -0.395135  |                 -0.077749   |               -0.0155155  | False             | fail               |
| A7_net_margin_single        | net_margin_stability | net_margin     | False         |         -0.34757    |     -1.35458   |       0.891582 |        0.223387 |                     -0.221436   |                    -0.383723  | net_margin_stability |       0.00803495  |       1.70989  |      0.00254756 |     0.614092 |              0 |             -0.384943   |                 -1.6734    |                 -0.316248   |               -0.421003   | False             | fail               |
| A8_net_margin_blend_10      | net_margin_stability | net_margin     | False         |          0.0243369  |      0.23278   |       0.245359 |        0.158333 |                      0.0852842  |                     0.0318784 | net_margin_stability |       0.00803495  |       1.70989  |      0.00254756 |     0.614092 |              0 |             -0.0130368  |                 -0.0860346 |                 -0.0095278  |               -0.00540197 | False             | fail               |
| A9_net_margin_blend_20      | net_margin_stability | net_margin     | False         |          0.0340294  |      0.291574  |       0.249596 |        0.175    |                      0.0882541  |                     0.0384267 | net_margin_stability |       0.00803495  |       1.70989  |      0.00254756 |     0.614092 |              0 |             -0.00334428 |                 -0.0272406 |                 -0.00655795 |                0.00114632 | False             | fail               |

## 本轮产物

- `data/results/factor_admission_validation_2026-04-20_next_summary.csv`
- `data/results/factor_admission_validation_2026-04-20_next_factor_gate.csv`
- `data/results/factor_admission_validation_2026-04-20_next_admission.csv`
- `data/results/factor_admission_validation_2026-04-20_next_a*.json`
