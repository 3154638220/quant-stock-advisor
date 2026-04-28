# R2 Upside Sleeve v1 (Day 3-5)

- 生成时间：`2026-04-27T15:01:36.610960+00:00`
- 配置快照：`/mnt/ssd/lh/config.yaml.backtest`
- 固定口径：`top_k=20` / `M` / `equal_weight` / `max_turnover=1.0` / `tplus1_open` / `universe_filter=on` / `prefilter=off`
- `p1_experiment_mode`: `daily_proxy_first`
- `legacy_proxy_decision_role`: `diagnostic_only`
- `primary_decision_metric`: `daily_bt_like_proxy_annualized_excess_vs_market`
- Gate：`<0%`→reject / `0%~+3%`→gray_zone / `>=+3%`→full_backtest_candidate

## 1. Leaderboard

| candidate_id             | label                                      |   daily_proxy_annualized_excess_vs_market |   delta_vs_baseline_proxy | gate_decision   |   strong_up_median_excess |   delta_vs_baseline_strong_up_median_excess |   strong_up_positive_share |   delta_vs_baseline_strong_up_positive_share |   strong_down_median_excess |   delta_vs_baseline_strong_down_median_excess |   strong_up_switch_in_minus_out |   avg_turnover_half_l1 |   delta_vs_baseline_turnover |   n_rebalances |
|:-------------------------|:-------------------------------------------|------------------------------------------:|--------------------------:|:----------------|--------------------------:|--------------------------------------------:|---------------------------:|---------------------------------------------:|----------------------------:|----------------------------------------------:|--------------------------------:|-----------------------:|-----------------------------:|---------------:|
| BASELINE_S2              | S2 vol_to_turnover (defensive baseline)    |                                   -0.1126 |                    0.0000 | reject          |                   -0.0626 |                                      0.0000 |                     0.1538 |                                       0.0000 |                      0.0765 |                                        0.0000 |                         -0.0128 |                 0.1242 |                       0.0000 |             64 |
| UPSIDE_B_relstr60_tnexp  | rel_strength_60d + turnover_expansion_5_60 |                                   -0.8568 |                   -0.7442 | reject          |                   -0.1776 |                                     -0.1149 |                     0.0000 |                                      -0.1538 |                     -0.1482 |                                       -0.2248 |                         -0.0104 |                 0.9516 |                       0.8273 |             64 |
| UPSIDE_A_relstr20_amtexp | rel_strength_20d + amount_expansion_5_60   |                                   -0.8590 |                   -0.7465 | reject          |                   -0.1706 |                                     -0.1080 |                     0.0000 |                                      -0.1538 |                     -0.1558 |                                       -0.2323 |                         -0.0086 |                 0.9672 |                       0.8430 |             64 |
| UPSIDE_C_limitup_tail    | limit_up_hits_20d + tail_strength_20d      |                                   -0.8662 |                   -0.7536 | reject          |                   -0.1525 |                                     -0.0898 |                     0.0000 |                                      -0.1538 |                     -0.1383 |                                       -0.2148 |                          0.0029 |                 0.9047 |                       0.7805 |             64 |

## 2. R2 验收（对 BASELINE 的相对改善）

| candidate_id             | status   | daily_proxy_>=_0   | strong_up_median_excess_improves   | strong_up_positive_share_improves   | strong_down_not_materially_worse   |
|:-------------------------|:---------|:-------------------|:-----------------------------------|:------------------------------------|:-----------------------------------|
| BASELINE_S2              | baseline | nan                | nan                                | nan                                 | nan                                |
| UPSIDE_B_relstr60_tnexp  | fail     | fail               | fail                               | fail                                | fail                               |
| UPSIDE_A_relstr20_amtexp | fail     | fail               | fail                               | fail                                | fail                               |
| UPSIDE_C_limitup_tail    | fail     | fail               | fail                               | fail                                | fail                               |

## 3. Regime 切片（candidate × regime）

| candidate_id             | regime      |   months |   median_benchmark_return |   median_strategy_return |   median_excess_return |   positive_excess_share |   benchmark_compound |   strategy_compound |   capture_ratio |
|:-------------------------|:------------|---------:|--------------------------:|-------------------------:|-----------------------:|------------------------:|---------------------:|--------------------:|----------------:|
| BASELINE_S2              | strong_down |       13 |                   -0.0551 |                  -0.0026 |                 0.0765 |                  0.9231 |              -0.6393 |              0.0699 |         -0.1094 |
| BASELINE_S2              | mild_down   |       13 |                   -0.0125 |                   0.0000 |                 0.0131 |                  0.6923 |              -0.1549 |             -0.0501 |          0.3233 |
| BASELINE_S2              | neutral     |       12 |                    0.0145 |                  -0.0056 |                -0.0126 |                  0.4167 |               0.1764 |              0.0520 |          0.2949 |
| BASELINE_S2              | mild_up     |       13 |                    0.0418 |                  -0.0002 |                -0.0340 |                  0.0000 |               0.7073 |              0.0043 |          0.0061 |
| BASELINE_S2              | strong_up   |       13 |                    0.0773 |                   0.0146 |                -0.0626 |                  0.1538 |               2.0764 |              0.2255 |          0.1086 |
| UPSIDE_A_relstr20_amtexp | strong_down |       13 |                   -0.0551 |                  -0.2206 |                -0.1558 |                  0.0000 |              -0.6393 |             -0.9589 |          1.4999 |
| UPSIDE_A_relstr20_amtexp | mild_down   |       13 |                   -0.0125 |                  -0.1247 |                -0.1155 |                  0.0769 |              -0.1549 |             -0.8170 |          5.2747 |
| UPSIDE_A_relstr20_amtexp | neutral     |       12 |                    0.0145 |                  -0.1499 |                -0.1676 |                  0.0000 |               0.1764 |             -0.8655 |         -4.9055 |
| UPSIDE_A_relstr20_amtexp | mild_up     |       13 |                    0.0418 |                  -0.0639 |                -0.1127 |                  0.0000 |               0.7073 |             -0.6521 |         -0.9219 |
| UPSIDE_A_relstr20_amtexp | strong_up   |       13 |                    0.0773 |                  -0.0848 |                -0.1706 |                  0.0000 |               2.0764 |             -0.6774 |         -0.3263 |
| UPSIDE_B_relstr60_tnexp  | strong_down |       13 |                   -0.0551 |                  -0.1927 |                -0.1482 |                  0.0000 |              -0.6393 |             -0.9540 |          1.4922 |
| UPSIDE_B_relstr60_tnexp  | mild_down   |       13 |                   -0.0125 |                  -0.1380 |                -0.1187 |                  0.0769 |              -0.1549 |             -0.8323 |          5.3732 |
| UPSIDE_B_relstr60_tnexp  | neutral     |       12 |                    0.0145 |                  -0.1452 |                -0.1641 |                  0.0000 |               0.1764 |             -0.8426 |         -4.7754 |
| UPSIDE_B_relstr60_tnexp  | mild_up     |       13 |                    0.0418 |                  -0.0535 |                -0.1026 |                  0.0000 |               0.7073 |             -0.6297 |         -0.8903 |
| UPSIDE_B_relstr60_tnexp  | strong_up   |       13 |                    0.0773 |                  -0.0761 |                -0.1776 |                  0.0000 |               2.0764 |             -0.7274 |         -0.3503 |
| UPSIDE_C_limitup_tail    | strong_down |       13 |                   -0.0551 |                  -0.2083 |                -0.1383 |                  0.0000 |              -0.6393 |             -0.9613 |          1.5036 |
| UPSIDE_C_limitup_tail    | mild_down   |       13 |                   -0.0125 |                  -0.1137 |                -0.0999 |                  0.0769 |              -0.1549 |             -0.8127 |          5.2472 |
| UPSIDE_C_limitup_tail    | neutral     |       12 |                    0.0145 |                  -0.1532 |                -0.1635 |                  0.0000 |               0.1764 |             -0.8785 |         -4.9789 |
| UPSIDE_C_limitup_tail    | mild_up     |       13 |                    0.0418 |                  -0.1065 |                -0.1504 |                  0.0000 |               0.7073 |             -0.7483 |         -1.0580 |
| UPSIDE_C_limitup_tail    | strong_up   |       13 |                    0.0773 |                  -0.0748 |                -0.1525 |                  0.0000 |               2.0764 |             -0.6129 |         -0.2952 |

## 4. Breadth 切片（candidate × breadth）

| candidate_id             | breadth   |   months |   median_benchmark_return |   median_strategy_return |   median_excess_return |   positive_excess_share |
|:-------------------------|:----------|---------:|--------------------------:|-------------------------:|-----------------------:|------------------------:|
| BASELINE_S2              | narrow    |       19 |                   -0.0353 |                  -0.0089 |                 0.0281 |                  0.7895 |
| BASELINE_S2              | mid       |       26 |                    0.0172 |                  -0.0018 |                -0.0091 |                  0.4231 |
| BASELINE_S2              | wide      |       19 |                    0.0695 |                   0.0146 |                -0.0526 |                  0.1053 |
| UPSIDE_A_relstr20_amtexp | narrow    |       19 |                   -0.0353 |                  -0.1421 |                -0.1039 |                  0.0526 |
| UPSIDE_A_relstr20_amtexp | mid       |       26 |                    0.0172 |                  -0.1390 |                -0.1538 |                  0.0000 |
| UPSIDE_A_relstr20_amtexp | wide      |       19 |                    0.0695 |                  -0.0639 |                -0.1679 |                  0.0000 |
| UPSIDE_B_relstr60_tnexp  | narrow    |       19 |                   -0.0353 |                  -0.1380 |                -0.0990 |                  0.0526 |
| UPSIDE_B_relstr60_tnexp  | mid       |       26 |                    0.0172 |                  -0.1427 |                -0.1465 |                  0.0000 |
| UPSIDE_B_relstr60_tnexp  | wide      |       19 |                    0.0695 |                  -0.0669 |                -0.1533 |                  0.0000 |
| UPSIDE_C_limitup_tail    | narrow    |       19 |                   -0.0353 |                  -0.1657 |                -0.1109 |                  0.0526 |
| UPSIDE_C_limitup_tail    | mid       |       26 |                    0.0172 |                  -0.1209 |                -0.1572 |                  0.0000 |
| UPSIDE_C_limitup_tail    | wide      |       19 |                    0.0695 |                  -0.0814 |                -0.1499 |                  0.0000 |

## 5. 关键年份 strong_up（2021/2025/2026）

| candidate_id             |   year | regime    |   months |   median_excess_return |   positive_excess_share |
|:-------------------------|-------:|:----------|---------:|-----------------------:|------------------------:|
| BASELINE_S2              |   2021 | strong_up |        3 |                -0.0527 |                  0.3333 |
| BASELINE_S2              |   2025 | strong_up |        2 |                -0.0804 |                  0.0000 |
| BASELINE_S2              |   2026 | strong_up |        1 |                -0.1321 |                  0.0000 |
| UPSIDE_A_relstr20_amtexp |   2021 | strong_up |        3 |                -0.1955 |                  0.0000 |
| UPSIDE_A_relstr20_amtexp |   2025 | strong_up |        2 |                -0.1373 |                  0.0000 |
| UPSIDE_A_relstr20_amtexp |   2026 | strong_up |        1 |                -0.2270 |                  0.0000 |
| UPSIDE_B_relstr60_tnexp  |   2021 | strong_up |        3 |                -0.1776 |                  0.0000 |
| UPSIDE_B_relstr60_tnexp  |   2025 | strong_up |        2 |                -0.1446 |                  0.0000 |
| UPSIDE_B_relstr60_tnexp  |   2026 | strong_up |        1 |                -0.2324 |                  0.0000 |
| UPSIDE_C_limitup_tail    |   2021 | strong_up |        3 |                -0.1525 |                  0.0000 |
| UPSIDE_C_limitup_tail    |   2025 | strong_up |        2 |                -0.1384 |                  0.0000 |
| UPSIDE_C_limitup_tail    |   2026 | strong_up |        1 |                -0.1787 |                  0.0000 |

## 6. Switch quality（candidate × regime）

| candidate_id             | regime      |   rebalances |   mean_switch_in |   mean_switch_out |   mean_switch_in_minus_out |   median_switch_in_minus_out |   switch_in_winning_share |   mean_topk_minus_next |
|:-------------------------|:------------|-------------:|-----------------:|------------------:|---------------------------:|-----------------------------:|--------------------------:|-----------------------:|
| BASELINE_S2              | strong_down |            7 |           0.0091 |           -0.0336 |                     0.0427 |                       0.0268 |                    0.5714 |                 0.0352 |
| BASELINE_S2              | mild_down   |            9 |          -0.0142 |           -0.0019 |                    -0.0123 |                      -0.0016 |                    0.4444 |                 0.0120 |
| BASELINE_S2              | neutral     |            8 |           0.0351 |            0.0105 |                     0.0246 |                       0.0124 |                    0.7500 |                 0.0026 |
| BASELINE_S2              | mild_up     |            4 |           0.0637 |            0.0293 |                     0.0344 |                       0.0081 |                    0.7500 |                -0.0046 |
| BASELINE_S2              | strong_up   |            8 |           0.0322 |            0.0451 |                    -0.0128 |                       0.0031 |                    0.5000 |                -0.0136 |
| UPSIDE_A_relstr20_amtexp | strong_down |           14 |          -0.1065 |           -0.0812 |                    -0.0253 |                      -0.0389 |                    0.2143 |                -0.0143 |
| UPSIDE_A_relstr20_amtexp | mild_down   |           13 |           0.0013 |           -0.0110 |                     0.0123 |                       0.0180 |                    0.5385 |                 0.0176 |
| UPSIDE_A_relstr20_amtexp | neutral     |           11 |          -0.0367 |            0.0073 |                    -0.0440 |                      -0.0342 |                    0.1818 |                -0.0267 |
| UPSIDE_A_relstr20_amtexp | mild_up     |           11 |           0.0233 |            0.0227 |                     0.0005 |                      -0.0088 |                    0.4545 |                 0.0101 |
| UPSIDE_A_relstr20_amtexp | strong_up   |           12 |           0.0804 |            0.0890 |                    -0.0086 |                      -0.0333 |                    0.4167 |                -0.0006 |
| UPSIDE_B_relstr60_tnexp  | strong_down |           14 |          -0.1137 |           -0.0742 |                    -0.0395 |                      -0.0472 |                    0.2857 |                -0.0289 |
| UPSIDE_B_relstr60_tnexp  | mild_down   |           13 |          -0.0133 |           -0.0247 |                     0.0115 |                       0.0118 |                    0.5385 |                 0.0035 |
| UPSIDE_B_relstr60_tnexp  | neutral     |           11 |          -0.0278 |            0.0072 |                    -0.0350 |                      -0.0236 |                    0.2727 |                -0.0287 |
| UPSIDE_B_relstr60_tnexp  | mild_up     |           11 |           0.0221 |            0.0124 |                     0.0097 |                      -0.0174 |                    0.4545 |                -0.0015 |
| UPSIDE_B_relstr60_tnexp  | strong_up   |           12 |           0.0623 |            0.0727 |                    -0.0104 |                      -0.0026 |                    0.5000 |                -0.0201 |
| UPSIDE_C_limitup_tail    | strong_down |           14 |          -0.1186 |           -0.0815 |                    -0.0372 |                      -0.0406 |                    0.0714 |                -0.0221 |
| UPSIDE_C_limitup_tail    | mild_down   |           13 |          -0.0237 |           -0.0369 |                     0.0132 |                       0.0211 |                    0.5385 |                -0.0101 |
| UPSIDE_C_limitup_tail    | neutral     |           11 |          -0.0307 |           -0.0330 |                     0.0023 |                      -0.0245 |                    0.4545 |                -0.0097 |
| UPSIDE_C_limitup_tail    | mild_up     |           11 |          -0.0100 |            0.0201 |                    -0.0301 |                      -0.0178 |                    0.3636 |                -0.0210 |
| UPSIDE_C_limitup_tail    | strong_up   |           12 |           0.0814 |            0.0785 |                     0.0029 |                       0.0051 |                    0.5833 |                -0.0079 |

## 7. 结论 / 下一步

### 7.1 三条候选（A/B/C）单独都不可独立 promotion

| 维度 | 现象 | 解释 |
| --- | --- | --- |
| daily proxy 年化超额 | A=`-85.9%` / B=`-85.7%` / C=`-86.6%`，均 reject | 不是边际负，是塌方负，表明纯上行 sleeve 无法独立担当主组合 |
| `avg_turnover_half_l1` | A=`0.967` / B=`0.952` / C=`0.905`（baseline `0.124`） | 月度近乎 100% 换仓，扣完 vwap 滑点 + impact 后剩余信号被吃光 |
| 各 regime median excess | A/B/C 在 5 档全部 < `-9%` | 防守 / 中性 / 上行月份都跑不赢；不存在“以下行换上行”的可交易 trade-off |
| `strong_up_median_excess` | A `-17.1%` / B `-17.8%` / C `-15.3%` | 即使在策略最匹配的 strong_up，纯上行 sleeve 也劣于 S2（`-6.3%`），机制 1+2 的方向性证据没有转化为可交易超额 |

结论：**plan §3 Day 3-5 的四条验收，三个候选全部 fail。** 这一结果不意外——R1 在三个机制中已说明 S2 在 strong_up 失配，但反向也不成立：把 S2 整盘换成 upside 并不能赚到上涨钱，而且把 strong_down 的防守一并送掉了（A/B/C 在 strong_down median excess 从 baseline `+7.65%` 掉到 `-13~-16%`）。

### 7.2 R1 机制 3 的边际正向证据：UPSIDE_C 的 switch quality

唯一的可利用信号在 §6：

| candidate | strong_up `mean_switch_in_minus_out` | strong_up `switch_in_winning_share` |
| --- | ---: | ---: |
| BASELINE_S2 | `-0.0128` | `0.500` |
| UPSIDE_A | `-0.0086` | `0.417` |
| UPSIDE_B | `-0.0104` | `0.500` |
| **UPSIDE_C** | **`+0.0029`** | **`0.583`** |

UPSIDE_C（`limit_up_hits_20d + tail_strength_20d`）是四个候选中唯一在 strong_up 把 `switch_in_minus_out` 翻正、且换入跑赢比例超过 50% 的组合。它的整体 daily proxy 仍然 reject，但**它的换入边界在 strong_up 与 R1 机制 3 的方向一致**——这恰好是 R3 boundary-aware 想攻击的对象。

### 7.3 下一步建议（写给 Day 6-7）

1. **不允许任何 v1 sleeve 单独写回默认基线**——四条 R2 验收均 fail，且 daily proxy 都 reject。
2. **Day 6-7 必须做组合，而不是替换**：把 S2 作为 `defensive_sleeve` 主体，在 `strong_up + wide_breadth` 状态下只挂极小权重的 `upside_sleeve`。
3. **在三个候选里优先 UPSIDE_C** 作为 Day 6-7 的 upside 输入：唯一一个 strong_up `switch_in_minus_out > 0`、且 turnover 在三个里相对最低（`0.905`）。
4. **状态权重起步要保守**：第一组只测试 `strong_up/wide_breadth → defensive 80% / upside 20%`，其余状态全部 100% defensive。这样总换手相对 S2 baseline 增量上限可估算（≤ `0.20×0.905 + 0.80×0.124 ≈ 0.28`），可控。
5. **若该组合 daily proxy 仍 < 0%**：再回到 Day 3-5 候选层，按 plan §3 备选清单补 `relative_strength + industry_breadth` / `tradable_breakout + turnover_expansion` 两个新 composite，每轮 ≤3。
6. **不补正式 full backtest**——三个候选都低于 gray_zone 下沿，§2 R0 明确禁止。

## 8. 产出文件

- `data/results/p2_upside_sleeve_v1_2026-04-27_leaderboard.csv`
- `data/results/p2_upside_sleeve_v1_2026-04-27_regime_long.csv`
- `data/results/p2_upside_sleeve_v1_2026-04-27_breadth_long.csv`
- `data/results/p2_upside_sleeve_v1_2026-04-27_year_long.csv`
- `data/results/p2_upside_sleeve_v1_2026-04-27_switch_long.csv`
- `data/results/p2_upside_sleeve_v1_2026-04-27_monthly_long.csv`
- `data/results/p2_upside_sleeve_v1_2026-04-27_summary.json`

## 9. 配置参数

- `start`: `2021-01-01`
- `end`: `2026-04-27`
- `top_k`: `20`
- `rebalance_rule`: `M`
- `portfolio_method`: `equal_weight`
- `max_turnover`: `1.0`
- `execution_mode`: `tplus1_open`
- `prefilter`: `{'enabled': False, 'limit_move_max': 2, 'turnover_low_pct': 0.1, 'turnover_high_pct': 0.98, 'price_position_high_pct': 0.9}`
- `universe_filter`: `{'enabled': True, 'min_amount_20d': 50000000, 'require_roe_ttm_positive': True}`
- `benchmark_symbol`: `market_ew_proxy`
- `benchmark_min_history_days`: `551`
- `config_source`: `/mnt/ssd/lh/config.yaml.backtest`
- `p1_experiment_mode`: `daily_proxy_first`
- `legacy_proxy_decision_role`: `diagnostic_only`
- `primary_decision_metric`: `daily_bt_like_proxy_annualized_excess_vs_market`
- `gate_thresholds`: `{'reject': 0.0, 'full_backtest': 0.03}`
