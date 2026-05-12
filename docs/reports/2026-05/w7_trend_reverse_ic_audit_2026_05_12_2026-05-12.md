# W7 Trend Reverse IC Audit

- 生成时间：`2026-05-12T10:23:22.941146+00:00`
- 目的：在 W7 Stage A 失败后，离线验证“多头趋势状态更像短期过热信号”的反向假设。

## Summary

| candidate_pool_version | factor | direction_variant | months | rank_ic_mean | ic_ir | positive_month_ratio | top_bottom_excess_spread_mean | top_bottom_excess_spread_t | reverse_gate_pass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| U1_liquid_tradable | feature_trend_ema_spread | reversed | 63 | 0.0834243 | 0.582253 | 0.714286 | 0.0139149 | 2.36619 | True |
| U1_liquid_tradable | feature_trend_streak_days | reversed | 63 | 0.0606609 | 0.443366 | 0.68254 | 0.0101863 | 1.89994 | True |
| U1_liquid_tradable | feature_trend_bull_state | reversed | 63 | 0.0583071 | 0.488168 | 0.650794 |  |  | True |
| U1_liquid_tradable | feature_trend_bull_ratio_20d | reversed | 63 | 0.0539801 | 0.387286 | 0.68254 | 0.00622134 | 1.45244 | True |
| U1_liquid_tradable | feature_trend_bull_ratio_60d | reversed | 63 | 0.0438086 | 0.312242 | 0.634921 | 0.00533347 | 1.04367 | True |
| U1_liquid_tradable | feature_trend_flip_days_ago | original | 63 | 0.010257 | 0.150169 | 0.52381 | 0.00431583 | 1.95208 | False |
| U1_liquid_tradable | feature_trend_flip_days_ago | reversed | 63 | -0.010257 | -0.150169 | 0.47619 | -0.00443482 | -1.94441 | False |
| U1_liquid_tradable | feature_trend_bull_ratio_60d | original | 63 | -0.0438086 | -0.312242 | 0.365079 | -0.00545482 | -1.06427 | False |
| U1_liquid_tradable | feature_trend_bull_ratio_20d | original | 63 | -0.0539801 | -0.387286 | 0.31746 | -0.00641156 | -1.49637 | False |
| U1_liquid_tradable | feature_trend_bull_state | original | 63 | -0.0583071 | -0.488168 | 0.349206 |  |  | False |
| U1_liquid_tradable | feature_trend_streak_days | original | 63 | -0.0606609 | -0.443366 | 0.31746 | -0.0110541 | -2.06728 | False |
| U1_liquid_tradable | feature_trend_ema_spread | original | 63 | -0.0834243 | -0.582253 | 0.285714 | -0.0139155 | -2.36699 | False |
| U2_risk_sane | feature_trend_ema_spread | reversed | 63 | 0.0601337 | 0.430614 | 0.68254 | 0.00872103 | 1.62657 | True |
| U2_risk_sane | feature_trend_bull_state | reversed | 63 | 0.0433543 | 0.374736 | 0.603175 |  |  | True |
| U2_risk_sane | feature_trend_streak_days | reversed | 63 | 0.0432227 | 0.323339 | 0.650794 | 0.00621429 | 1.24934 | True |
| U2_risk_sane | feature_trend_bull_ratio_20d | reversed | 63 | 0.0384654 | 0.284796 | 0.650794 | 0.00329883 | 0.793053 | False |
| U2_risk_sane | feature_trend_bull_ratio_60d | reversed | 63 | 0.031846 | 0.227504 | 0.603175 | 0.00209529 | 0.436935 | False |
| U2_risk_sane | feature_trend_flip_days_ago | original | 63 | 0.0195695 | 0.28482 | 0.619048 | 0.00597566 | 2.7027 | False |
| U2_risk_sane | feature_trend_flip_days_ago | reversed | 63 | -0.0195695 | -0.28482 | 0.380952 | -0.00568963 | -2.58723 | False |
| U2_risk_sane | feature_trend_bull_ratio_60d | original | 63 | -0.031846 | -0.227504 | 0.396825 | -0.0019504 | -0.409377 | False |
| U2_risk_sane | feature_trend_bull_ratio_20d | original | 63 | -0.0384654 | -0.284796 | 0.349206 | -0.00406887 | -0.98292 | False |
| U2_risk_sane | feature_trend_streak_days | original | 63 | -0.0432227 | -0.323339 | 0.349206 | -0.0069976 | -1.44769 | False |
| U2_risk_sane | feature_trend_bull_state | original | 63 | -0.0433543 | -0.374736 | 0.396825 |  |  | False |
| U2_risk_sane | feature_trend_ema_spread | original | 63 | -0.0601337 | -0.430614 | 0.31746 | -0.00873212 | -1.62849 | False |

## Conclusion

存在反向方向通过离线门槛的趋势项；仍需独立 Stage A 复跑和 M8 Gate，不能直接推广到生产配置。

## Recent Monthly Rows

| signal_date | candidate_pool_version | factor | direction_variant | rank_ic | top_bottom_excess_spread | sample_count | coverage |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2024-04-30 | U2_risk_sane | feature_trend_streak_days | reversed | -0.241578 | -0.0411804 | 3168 | 1 |
| 2024-05-31 | U2_risk_sane | feature_trend_streak_days | reversed | -0.114402 | -0.0289283 | 3160 | 1 |
| 2024-06-28 | U2_risk_sane | feature_trend_streak_days | reversed | 0.119103 | 0.0277733 | 2613 | 1 |
| 2024-07-31 | U2_risk_sane | feature_trend_streak_days | reversed | 0.0682117 | 0.0230381 | 2419 | 1 |
| 2024-08-30 | U2_risk_sane | feature_trend_streak_days | reversed | 0.261348 | 0.0794889 | 2375 | 1 |
| 2024-09-30 | U2_risk_sane | feature_trend_streak_days | reversed | -0.204729 | -0.0925826 | 1586 | 1 |
| 2024-10-31 | U2_risk_sane | feature_trend_streak_days | reversed | -0.0398098 | -0.0223968 | 3732 | 1 |
| 2024-11-29 | U2_risk_sane | feature_trend_streak_days | reversed | 0.136773 | 0.026599 | 4000 | 1 |
| 2024-12-31 | U2_risk_sane | feature_trend_streak_days | reversed | -0.0537829 | -0.0234462 | 3926 | 1 |
| 2025-01-27 | U2_risk_sane | feature_trend_streak_days | reversed | 0.036056 | -0.0169921 | 3340 | 1 |
| 2025-02-28 | U2_risk_sane | feature_trend_streak_days | reversed | 0.154952 | 0.00990007 | 3709 | 1 |
| 2025-03-31 | U2_risk_sane | feature_trend_streak_days | reversed | -0.0555714 | -0.0164782 | 3956 | 1 |
| 2025-04-30 | U2_risk_sane | feature_trend_streak_days | reversed | 0.00808444 | -0.00447426 | 3673 | 1 |
| 2025-05-30 | U2_risk_sane | feature_trend_streak_days | reversed | 0.125001 | 0.0292013 | 3729 | 1 |
| 2025-06-30 | U2_risk_sane | feature_trend_streak_days | reversed | 0.0216066 | -0.0144434 | 3757 | 1 |
| 2025-07-31 | U2_risk_sane | feature_trend_streak_days | reversed | -0.00600166 | 0.00151728 | 4063 | 1 |
| 2025-08-29 | U2_risk_sane | feature_trend_streak_days | reversed | -0.00833718 | -0.00155332 | 4107 | 1 |
| 2025-09-30 | U2_risk_sane | feature_trend_streak_days | reversed | 0.184525 | 0.0333598 | 3926 | 1 |
| 2025-10-31 | U2_risk_sane | feature_trend_streak_days | reversed | 0.0832609 | 0.022444 | 4028 | 1 |
| 2025-11-28 | U2_risk_sane | feature_trend_streak_days | reversed | 0.145125 | 0.0255139 | 4166 | 1 |
| 2025-12-31 | U2_risk_sane | feature_trend_streak_days | reversed | 0.0531468 | -0.00793794 | 3740 | 1 |
| 2026-01-30 | U2_risk_sane | feature_trend_streak_days | reversed | -0.147207 | -0.0381131 | 4209 | 1 |
| 2026-02-27 | U2_risk_sane | feature_trend_streak_days | reversed | 0.0526165 | 0.0274873 | 3815 | 1 |
| 2026-03-31 | U2_risk_sane | feature_trend_streak_days | reversed | -0.098593 | -0.0519184 | 4196 | 1 |

## 口径

- 样本为 `candidate_pool_pass == True` 且存在月度 open-to-open 标签的截面。
- `original` 使用因子原始方向；`reversed` 使用 `-factor`，用于检验过热/反向假设。
- `rank_ic` 使用每月截面 Spearman 相关；默认标签为 `label_forward_1m_excess_vs_market`。
- `top_bottom_excess_spread` 是按方向分桶后最高桶均值减最低桶均值。
- 这是离线诊断，不改生产注册方向，不进入 M8 Baseline Gate。

## Artifacts

- `data/results/w7_trend_reverse_ic_audit_2026_05_12_2026-05-12_summary.csv`
- `data/results/w7_trend_reverse_ic_audit_2026_05_12_2026-05-12_monthly.csv`
- `data/results/w7_trend_reverse_ic_audit_2026_05_12_2026-05-12_summary.json`
- `data/results/w7_trend_reverse_ic_audit_2026_05_12_2026-05-12_manifest.json`
