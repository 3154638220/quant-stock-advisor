# R1 Strong-Up 失败归因

- 生成时间：`2026-04-27T14:23:35.608910+00:00`
- 配置快照：`/mnt/ssd/lh/config.yaml.backtest`
- 固定口径：`score=S2=vol_to_turnover` / `top_k=20` / `M` / `equal_weight` / `tplus1_open`
- Regime 分位阈值（基准月收益）：P20=-0.0291 / P40=-0.0002 / P60=0.0283 / P80=0.0614
- Breadth 分位阈值（月内 daily positive ratio 均值）：P30=0.4564 / P70=0.5096

## 结论速览

- strong_up 月份共 13 个，中位超额 `-0.0626`，正超额比例 `0.154`，capture_ratio `0.109`。
- wide breadth 月份共 19 个，中位超额 `-0.0527`。
- strong_up 状态下 switch_in_minus_out 均值 `-0.0186`，换入跑赢比例 `0.375`，topk_minus_next 均值 `-0.0046`。

## 1. Regime 切片（5 档）

| regime      |   months |   median_benchmark_return |   median_strategy_return |   median_excess_return |   positive_excess_share |   benchmark_compound |   strategy_compound |   capture_ratio |
|:------------|---------:|--------------------------:|-------------------------:|-----------------------:|------------------------:|---------------------:|--------------------:|----------------:|
| strong_down |       13 |                -0.0768409 |             -0.00260794  |              0.0765167 |                0.923077 |            -0.657877 |          0.00105365 |     -0.00160159 |
| mild_down   |       13 |                -0.0124372 |              0.00427939  |              0.0130855 |                0.692308 |            -0.166814 |          0.0347599  |     -0.208376   |
| neutral     |       12 |                 0.014402  |             -0.00563773  |             -0.0126102 |                0.416667 |             0.176026 |          0.0520254  |      0.295555   |
| mild_up     |       13 |                 0.041907  |             -0.000171592 |             -0.0341616 |                0        |             0.706474 |          0.00441299 |      0.00624651 |
| strong_up   |       13 |                 0.0772722 |              0.0146222   |             -0.0626499 |                0.153846 |             2.07491  |          0.225535   |      0.108696   |

## 2. Breadth 切片（3 档）

| breadth   |   months |   median_benchmark_return |   median_strategy_return |   median_excess_return |   positive_excess_share |
|:----------|---------:|--------------------------:|-------------------------:|-----------------------:|------------------------:|
| narrow    |       19 |                -0.0374002 |              -0.00891432 |             0.0401133  |                0.789474 |
| mid       |       26 |                 0.0171312 |              -0.00181994 |            -0.00889372 |                0.423077 |
| wide      |       19 |                 0.0690125 |               0.0146222  |            -0.0526727  |                0.105263 |

## 3. 关键年份（2021/2025/2026）regime 表现

|   year | regime      |   months |   median_benchmark_return |   median_strategy_return |   median_excess_return |   positive_excess_share |
|-------:|:------------|---------:|--------------------------:|-------------------------:|-----------------------:|------------------------:|
|   2021 | strong_down |        1 |                -0.0768409 |              0.019174    |             0.0960149  |                1        |
|   2021 | mild_down   |        3 |                -0.0212518 |             -0.0359457   |            -0.0146938  |                0.333333 |
|   2021 | neutral     |        4 |                 0.0217244 |             -0.00219632  |            -0.0213906  |                0.5      |
|   2021 | mild_up     |        1 |                 0.0308414 |              0.0294504   |            -0.00139094 |                0        |
|   2021 | strong_up   |        3 |                 0.0633528 |              0.0106664   |            -0.0526864  |                0.333333 |
|   2025 | mild_down   |        3 |                -0.0142122 |              0.00427939  |             0.00519606 |                0.666667 |
|   2025 | neutral     |        4 |                 0.014402  |             -0.00573879  |            -0.01646    |                0.25     |
|   2025 | mild_up     |        3 |                 0.05227   |              0.0350096   |            -0.0250557  |                0        |
|   2025 | strong_up   |        2 |                 0.0804164 |             -5.09875e-05 |            -0.0804674  |                0        |
|   2026 | strong_down |        1 |                -0.0929515 |              0.00336937  |             0.0963209  |                1        |
|   2026 | mild_up     |        2 |                 0.0367082 |             -0.00638872  |            -0.0430969  |                0        |
|   2026 | strong_up   |        1 |                 0.0792875 |             -0.0529686   |            -0.132256   |                0        |

## 4. 持仓暴露：(regime, group) × feature 均值

group 含义：`top20`（当期持仓）/ `21_40` / `41_60` / `switch_in` / `switch_out` / `benchmark`（投资域等权）。

仅展示 `strong_up` / `wide breadth` 关注的子集（完整表见 CSV）。

| regime      | group      |   rebalances |   rel_strength_20d |   rel_strength_60d |   amount_expansion_5_60 |   turnover_expansion_5_60 |   tail_strength_20d |   limit_up_hits_20d |   log_market_cap |   realized_vol |
|:------------|:-----------|-------------:|-------------------:|-------------------:|------------------------:|--------------------------:|--------------------:|--------------------:|-----------------:|---------------:|
| strong_down | top20      |           13 |         0.0795113  |         0.0855794  |             -0.00202266 |                0.00947203 |         0.000322119 |           0.0307692 |          30.7326 |       0.24807  |
| strong_down | 21_40      |           13 |         0.0699135  |         0.0757284  |             -0.0829152  |               -0.0681731  |         0.000236547 |           0.107692  |          29.866  |       0.309703 |
| strong_down | 41_60      |           13 |         0.0635247  |         0.0599934  |             -0.0981699  |               -0.0540688  |        -0.000305131 |           0.130769  |          29.6245 |       0.329258 |
| strong_down | switch_in  |            8 |         0.0484372  |         0.0525246  |             -0.0728043  |               -0.0833425  |        -0.000259814 |           0.0208333 |          30.701  |       0.270637 |
| strong_down | switch_out |            8 |         0.0751327  |         0.0702918  |             -0.0290543  |               -0.0248872  |         0.000385054 |           0         |          30.5847 |       0.252969 |
| strong_down | benchmark  |           13 |         0.0126457  |         0.0257989  |             -0.278296   |               -0.204365   |        -0.00277267  |           0.338913  |          25.7399 |       0.459749 |
| neutral     | top20      |           12 |        -0.00161508 |         0.0180692  |             -0.198692   |               -0.213848   |         0.000223248 |           0.025     |          30.806  |       0.214945 |
| neutral     | 21_40      |           12 |         0.00910325 |         0.0455041  |             -0.168746   |               -0.16502    |         0.000654648 |           0.120833  |          30.1427 |       0.283981 |
| neutral     | 41_60      |           12 |         0.012869   |         0.0252518  |             -0.136528   |               -0.124843   |         0.000776917 |           0.195833  |          29.7189 |       0.294867 |
| neutral     | switch_in  |            9 |         0.0114457  |         0.0532442  |             -0.0890989  |               -0.367033   |         0.000677924 |           0.0277778 |          30.4044 |       0.240243 |
| neutral     | switch_out |            9 |         0.0310604  |         0.042555   |             -0.113079   |               -0.119427   |         0.00165797  |           0.0802469 |          30.4397 |       0.271452 |
| neutral     | benchmark  |           12 |         0.0141008  |         0.0391633  |             -0.159272   |               -0.156017   |         0.00131928  |           0.291179  |          25.7927 |       0.396649 |
| strong_up   | top20      |           13 |        -0.0415576  |         0.00966735 |              0.0785564  |                0.0647411  |         0.00150287  |           0.0576923 |          30.7765 |       0.248131 |
| strong_up   | 21_40      |           13 |        -0.0214069  |         0.0254378  |              0.106602   |                0.0951618  |         0.00253553  |           0.130769  |          30.0124 |       0.314034 |
| strong_up   | 41_60      |           13 |        -0.0214882  |         0.0234617  |              0.131104   |                0.113642   |         0.0027586   |           0.280769  |          29.6935 |       0.351519 |
| strong_up   | switch_in  |            7 |        -0.0641612  |        -0.0126311  |              0.148457   |                0.0186212  |         0.0014743   |           0.277778  |          30.6483 |       0.303521 |
| strong_up   | switch_out |            7 |        -0.0824914  |        -0.0170942  |              0.0752125  |                0.0824304  |         0.00099601  |           0.286848  |          30.7578 |       0.291279 |
| strong_up   | benchmark  |           13 |         0.0233317  |         0.046463   |              0.140415   |                0.11085    |         0.00448551  |           0.512536  |          25.7713 |       0.489932 |

## 5. Active diff（group - benchmark，按 regime）

| regime      | group      |   rebalances |   active_rel_strength_20d |   active_rel_strength_60d |   active_amount_expansion_5_60 |   active_turnover_expansion_5_60 |   active_tail_strength_20d |   active_limit_up_hits_20d |   active_log_market_cap |   active_realized_vol |
|:------------|:-----------|-------------:|--------------------------:|--------------------------:|-------------------------------:|---------------------------------:|---------------------------:|---------------------------:|------------------------:|----------------------:|
| strong_down | top20      |           13 |               0.0668656   |                0.0597805  |                     0.276274   |                       0.213837   |                0.00309479  |                  -0.308144 |                 4.99272 |             -0.211678 |
| strong_down | 21_40      |           13 |               0.0572678   |                0.0499296  |                     0.195381   |                       0.136192   |                0.00300922  |                  -0.231221 |                 4.12617 |             -0.150045 |
| strong_down | 41_60      |           13 |               0.0508789   |                0.0341946  |                     0.180126   |                       0.150296   |                0.00246754  |                  -0.208144 |                 3.88461 |             -0.13049  |
| strong_down | switch_in  |            8 |               0.0380873   |                0.0295957  |                     0.160532   |                       0.0697028  |                0.00180272  |                  -0.324596 |                 4.87813 |             -0.198806 |
| strong_down | switch_out |            8 |               0.0647828   |                0.047363   |                     0.204282   |                       0.128158   |                0.00244759  |                  -0.345429 |                 4.7619  |             -0.216474 |
| strong_down | benchmark  |           13 |               0           |                0          |                     0          |                       0          |                0           |                   0        |                 0       |              0        |
| neutral     | top20      |           12 |              -0.0157159   |               -0.021094   |                    -0.0394199  |                      -0.0578314  |               -0.00109603  |                  -0.266179 |                 5.01331 |             -0.181704 |
| neutral     | 21_40      |           12 |              -0.00499758  |                0.00634086 |                    -0.00947375 |                      -0.0090033  |               -0.00066463  |                  -0.170346 |                 4.35004 |             -0.112668 |
| neutral     | 41_60      |           12 |              -0.0012318   |               -0.0139115  |                     0.0227443  |                       0.0311735  |               -0.000542361 |                  -0.095346 |                 3.92626 |             -0.101782 |
| neutral     | switch_in  |            9 |              -0.000329524 |                0.0160309  |                     0.068971   |                      -0.207438   |               -0.000562281 |                  -0.266049 |                 4.61097 |             -0.153163 |
| neutral     | switch_out |            9 |               0.0192852   |                0.00534162 |                     0.044991   |                       0.040169   |                0.000417761 |                  -0.21358  |                 4.64622 |             -0.121954 |
| neutral     | benchmark  |           12 |               0           |                0          |                     0          |                       0          |                0           |                   0        |                 0       |              0        |
| strong_up   | top20      |           13 |              -0.0648893   |               -0.0367957  |                    -0.0618583  |                      -0.046109   |               -0.00298264  |                  -0.454844 |                 5.00521 |             -0.241801 |
| strong_up   | 21_40      |           13 |              -0.0447385   |               -0.0210252  |                    -0.0338128  |                      -0.0156883  |               -0.00194998  |                  -0.381767 |                 4.24104 |             -0.175898 |
| strong_up   | 41_60      |           13 |              -0.0448199   |               -0.0230013  |                    -0.00931099 |                       0.00279201 |               -0.00172691  |                  -0.231767 |                 3.92218 |             -0.138413 |
| strong_up   | switch_in  |            7 |              -0.0940739   |               -0.0678045  |                    -0.0711763  |                      -0.159557   |               -0.00401063  |                  -0.263933 |                 4.85218 |             -0.20496  |
| strong_up   | switch_out |            7 |              -0.112404    |               -0.0722675  |                    -0.144421   |                      -0.0957477  |               -0.00448891  |                  -0.254862 |                 4.9616  |             -0.217202 |
| strong_up   | benchmark  |           13 |               0           |                0          |                     0          |                       0          |                0           |                   0        |                 0       |              0        |

## 6. Switch quality（按 regime）

| regime      |   rebalances |   mean_switch_in |   mean_switch_out |   mean_switch_in_minus_out |   median_switch_in_minus_out |   switch_in_winning_share |   mean_topk_minus_next |
|:------------|-------------:|-----------------:|------------------:|---------------------------:|-----------------------------:|--------------------------:|-----------------------:|
| strong_down |            6 |      -0.00679637 |       -0.0361277  |                  0.0293313 |                   0.028317   |                  0.5      |             0.0157683  |
| mild_down   |           10 |       0.0147626  |       -0.00598288 |                  0.0207455 |                   0.0165847  |                  0.6      |             0.0150287  |
| neutral     |            9 |       0.034933   |        0.0106397  |                  0.0242933 |                   0.0117434  |                  0.666667 |            -0.00537082 |
| mild_up     |            3 |       0.03034    |        0.016103   |                  0.014237  |                  -0.0264993  |                  0.333333 |             0.0120697  |
| strong_up   |            8 |       0.0406455  |        0.0592439  |                 -0.0185984 |                  -0.00575405 |                  0.375    |            -0.00455347 |

## 7. 归因结论与可验证机制

### 7.1 主结论

S2 / vol_to_turnover 基线在 **`strong_up` 与 `wide_breadth`** 月份系统性失败：

| 切片 | 月数 | 中位超额 | 正超额比例 | capture_ratio |
| --- | ---: | ---: | ---: | ---: |
| `strong_up` | 13 | `-6.26%` | `0.154` | `0.109` |
| `wide_breadth` | 19 | `-5.27%` | `0.105` | — |
| `strong_down` | 13 | `+7.65%` | `0.923` | `-0.002` |
| `narrow_breadth` | 19 | `+4.01%` | `0.789` | — |

防守 / 上涨参与的反差极端：strong_up 仅吃到基准约 `10.9%` 的复利收益，strong_down 反而保住正收益。

关键年份（2021/2025/2026）strong_up 的中位超额分别为 `-5.27% / -8.05% / -13.23%`，三年都在 strong_up 重伤；strong_down 同期均为正。

### 7.2 三个可验证机制

读法：active 量 = top20（或 switch_in）均值 - 投资域基准均值。

**机制 1 — 持仓在上涨月偏稳定大盘 / 低波 / 无涨停弹性，结构性失配**

strong_up 状态下 top20 的 active 系数：

| 维度 | active 值 | 方向解读 |
| --- | ---: | --- |
| `active_log_market_cap` | `+5.00` | 极端偏大盘（与 strong_down 时一样） |
| `active_realized_vol` | `-0.24` | 低波 |
| `active_limit_up_hits_20d` | `-0.45` | 涨停集群明显少（基准 `0.51`、持仓 `0.06`） |
| `active_rel_strength_20d` | `-0.065` | 近 20 日相对强度反而更弱 |

→ S2 从设计上挑稳定低换手大盘股，这套画像在 strong_up 状态与基准差距最大。

**机制 2 — S2 排序与 strong_up 的成交额 / 换手扩张方向反向**

strong_up 状态下：

| 维度 | top20 active | 21_40 active | 41_60 active |
| --- | ---: | ---: | ---: |
| `active_amount_expansion_5_60` | `-0.062` | `-0.034` | `-0.009` |
| `active_turnover_expansion_5_60` | `-0.046` | `-0.016` | `+0.003` |

值得注意的是：**41_60 桶在 strong_up 状态下扩张差距已经接近 0**。换言之，S2 的截面排序越靠前，反而越避开 strong_up 期间扩张最强的票。这与 `vol_to_turnover` 公式的字面动机（低成交、低换手溢价）一致，因此机制成立的概率非常高。

**机制 3 — 换入边界在 strong_up 状态系统性失效**

switch quality 按 regime：

| regime | switch_in_minus_out | switch_in_winning_share |
| --- | ---: | ---: |
| `strong_down` | `+2.93%` | 0.500 |
| `mild_down` | `+2.07%` | 0.600 |
| `neutral` | `+2.43%` | 0.667 |
| `mild_up` | `+1.42%` | 0.333 |
| **`strong_up`** | **`-1.86%`** | **0.375** |

strong_up 是唯一 switch_in_minus_out 为负的 regime；其余四档都正，验证 R3 boundary-aware 的核心命题（`switch-in` 在 strong_up 比 `switch-out` 还差）。

进一步看 switch_in 在 strong_up 的 active 暴露：

| 维度 | switch_in active | 解读 |
| --- | ---: | --- |
| `active_rel_strength_20d` | `-0.094` | 比 top20 整体（-0.065）还更避开强势票 |
| `active_amount_expansion_5_60` | `-0.071` | 同上，扩张方向更反 |
| `active_turnover_expansion_5_60` | `-0.160` | 极端反向 |
| `active_limit_up_hits_20d` | `-0.264` | 仍然不参与涨停集群 |

→ S2 不只是“一开始挑错票”，**模型在 strong_up 期间的换仓动作系统性挑出了比已持仓更弱的票**。这是 R3 用换入分类器或 pairwise ranker 直接攻击的对象。

### 7.3 给 R2 / R3 的输入

1. **R2 upside sleeve 第一版优先组合**：
   - 候选 A：`rel_strength_20d + amount_expansion_5_60`（机制 1+2 同向）
   - 候选 B：`rel_strength_60d + turnover_expansion_5_60`（中长期版本）
   - 候选 C：`limit_up_hits_20d + tail_strength_20d`（可交易路径版）
   每个候选在 strong_up 都对应 active 系数显著负（基准已确认这些维度更高）。
2. **R3 boundary-aware 标签优先级**：
   - `switch_quality_label`（机制 3 直接对应）应作为第一目标；
   - 训练样本应在 `strong_up + wide_breadth` 上加权或仅在该切片建模，因其他四档 switch_in 已经为正，无须改写。
3. **R2 状态权重的顺向证据**：
   - `defensive_sleeve` 在 strong_down 的 capture_ratio 接近 0 但中位超额 `+7.65%`、跑赢 `92.3%` → 维持高权重；
   - `upside_sleeve` 须只在 `strong_up` / `wide_breadth` 启用，避免在 mild_up（中位超额 `-3.42%` 但 switch_in_minus_out 仍为正 `+1.42%`）做不必要扰动。

## 8. 产出文件

- `data/results/p1_strong_up_failure_attribution_2026-04-27_monthly.csv`
- `data/results/p1_strong_up_failure_attribution_2026-04-27_regime_capture.csv`
- `data/results/p1_strong_up_failure_attribution_2026-04-27_breadth_capture.csv`
- `data/results/p1_strong_up_failure_attribution_2026-04-27_year_capture.csv`
- `data/results/p1_strong_up_failure_attribution_2026-04-27_group_exposure_detail.csv`
- `data/results/p1_strong_up_failure_attribution_2026-04-27_group_exposure.csv`
- `data/results/p1_strong_up_failure_attribution_2026-04-27_group_active_diff.csv`
- `data/results/p1_strong_up_failure_attribution_2026-04-27_switch_quality.csv`
- `data/results/p1_strong_up_failure_attribution_2026-04-27_switch_by_regime.csv`
- `data/results/p1_strong_up_failure_attribution_2026-04-27_summary.json`

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
- `benchmark_min_history_days`: `446`
- `config_source`: `/mnt/ssd/lh/config.yaml.backtest`
