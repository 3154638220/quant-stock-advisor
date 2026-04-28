# 策略优化主计划

**文档角色**：当前唯一主计划（canonical）
**更新时间**：`2026-04-28`
**当前阶段**：R5 配置治理已完成；已建立 promoted registry，当前无任何研究候选进入生产
**生产状态**：当前没有任何 P1/R2/R3 研究候选满足 promotion；不允许写入日更推荐默认配置
**归档入口**：`docs/plan-04-20.md` 仅保留 `2026-04-20` 当日执行记录，不再承担主计划职责

---

## 0. 总判断

当前项目不是“没有模型方向”，而是已经把几个不该继续消耗预算的方向证伪了。

目前最强的资产是研究与评估基础设施，不是某一个 alpha 候选。系统已经具备：

1. daily-proxy-first 准入护栏。
2. open-to-open 执行与 benchmark 对齐。
3. buy-fail diagnostic。
4. regime / breadth / switch quality / topk boundary 诊断。
5. 失败归因与产物归档纪律。

但当前模型效果的核心短板也很明确：

1. S2 / `vol_to_turnover` 有防守价值，但在 strong-up / wide-breadth 行情中系统性落后。
2. P1 原主线的标签近邻微调已经被充分证伪。
3. pure upside Top-20 与 dual sleeve 都失败。
4. R2B 第一轮 replacement 证明边界信号有局部信息，但组合收益没有兑现。
5. R4A 已补齐真实申万行业映射；R4A 之前的 `industry_breadth` 仍使用 symbol prefix fallback，不能作为有效行业 alpha 证据。
6. R2B-O 证明候选池存在很强的事后 replacement 上限，但 `strong_up_or_wide` 状态门控本身没有提高普通 pair 胜率。
7. R2B v2 证明 edge gate 能减少机械替换并改善部分 switch quality，但仍没有把 strong-up 收益兑现为可 promotion alpha。
8. R2B v2 weight audit 进一步证明 `U3_A` gray zone 不稳：手写 `pair_edge_score` bucket 反向，replace-1 更差，不能启动 R3。

新的主线必须从：

```text
选出上涨 Top-20
```

改成：

```text
在可交易状态下，只替换 S2 边界上最弱的少数持仓，并且替换后真实跑赢
```

一句话路线：

```text
真实行业映射
-> oracle replacement attribution 判断可学上限（已完成）
-> R2B v2 edge-gated replacement / overlay（已完成，`U3_A` gray zone）
-> Day 8 gray-zone weight audit（已完成，`U3_A` 不稳）
-> 暂停 R2B/R3 replacement 主线，进入 R5 配置治理
-> 只有 full backtest 与 promotion gate 全过后才进入生产 registry
```

当前禁止动作：

1. 不启动 R3；R2B v2 gray-zone audit 未支持进入模型训练。
2. 不继续加大 dual sleeve upside 权重。
3. 不继续 P1/G0 标签近邻微调。
4. 不把 pure upside Top-20 当 promotion 候选。
5. 不把 R2B v1 任一研究配置写入日更推荐默认配置。

---

## 1. 当前证据

### 1.1 R0 已完成，后续实验必须引用新契约

R0 修复已完成，见：

- `docs/r0_eval_execution_contract_fix_2026-04-28.md`
- `data/results/r0_eval_execution_contract_fix_2026-04-28_summary.json`

后续实验必须记录：

| 字段 | 当前值 |
| --- | --- |
| `eval_contract_version` | `r0_eval_execution_contract_2026-04-28` |
| `execution_contract_version` | `tplus1_open_buy_delta_limit_mask_2026-04-28` |
| `primary_benchmark_return_mode` | `open_to_open` |
| `comparison_benchmark_return_mode` | `close_to_close` |
| `state_lag` | `previous_completed_month` |
| `state_threshold_mode` | 状态决策使用 `expanding`；事后诊断可标注 `diagnostic_only` |

R0 修复点：

1. `limit_up_hits_20d` 拆成纯上涨、纯下跌、任意涨跌停路径。
2. `tplus1_open` 买入失败只冻结新增/增持部分，已有持仓继续持有。
3. benchmark 改为与策略执行一致的 open-to-open。
4. 状态决策阈值改为可交易的 expanding threshold。
5. `daily_run.py` 的 `log` 初始化与 F 类 lint 错误已修复。

### 1.2 R1F 复跑确认旧 R2 失败仍成立

R1F 已完成，见：

- `docs/r1f_fixed_baseline_rerun_2026-04-28.md`
- `data/results/r1f_fixed_baseline_rerun_2026-04-28_summary.json`

关键结果：

| candidate | daily proxy 年化超额 | strong-up 中位超额 | strong-up positive share | strong-down 中位超额 | 平均半 L1 换手 | 判定 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `BASELINE_S2_FIXED` | `-8.59%` | `-6.12%` | `15.38%` | `+6.29%` | `0.084` | baseline / reject |
| `DUAL_V1_FIXED` | `-11.51%` | `-6.24%` | `15.38%` | `+4.84%` | `0.151` | reject |
| `UPSIDE_C_FIXED` | `-47.72%` | `-6.18%` | `15.38%` | `-8.13%` | `0.752` | reject |

含义：

1. R0 修复改善了数值，但没有反转结论。
2. `UPSIDE_C_FIXED` 仍高换手且收益很差，不能作为独立 sleeve。
3. `DUAL_V1_FIXED` 的 switch quality 局部改善没有转化成 strong-up 收益。
4. 旧 R2 失败成立，不应继续做 sleeve 权重网格。

### 1.3 R2B 第一轮 replacement 已失败

R2B v1 已完成，见：

- `docs/r2b_tradable_upside_replacement_v1_2026-04-28.md`
- `data/results/r2b_tradable_upside_replacement_v1_2026-04-28_*.csv`

R2B v1 的组合表达是：

```text
S2 defensive Top-20
+ capped replacement slots
```

不再使用完整 upside Top-20 sleeve。

主要结果：

| candidate | daily proxy 年化超额 | 相对 baseline | strong-up 中位超额 | strong-up positive share | strong-up switch-in-minus-out | 平均半 L1 换手 | 判定 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `BASELINE_S2_FIXED` | `-8.59%` | `0.00pct` | `-6.12%` | `15.38%` | `-2.37%` | `0.084` | baseline / reject |
| `U2_C_s2_residual_elasticity__R2B_OVERLAY_10` | `-8.87%` | `-0.28pct` | `-6.12%` | `15.38%` | `+1.30%` | `0.104` | reject |
| `U2_A_industry_breadth_strength__R2B_OVERLAY_10` | `-8.92%` | `-0.33pct` | `-5.98%` | `15.38%` | `+1.62%` | `0.107` | reject |
| `U2_A_industry_breadth_strength__R2B_REPLACE_3` | `-9.07%` | `-0.48pct` | `-5.96%` | `15.38%` | `+3.13%` | `0.141` | reject |
| `U2_B_tradable_breakout_expansion__R2B_REPLACE_5` | `-12.45%` | `-3.86pct` | `-6.12%` | `15.38%` | `+0.47%` | `0.208` | reject |

补充观察：

1. `U2_A_industry_breadth_strength__STANDALONE_TOP20` 的 daily proxy 改善到 `-1.34%`，strong-up 中位超额转正到 `+1.49%`，strong-up positive share 到 `53.85%`，但平均半 L1 换手达到 `0.952`，不具备 promotion 资格。
2. replacement 版本把换手压回可控区间后，daily proxy 没有优于 baseline，strong-up positive share 也没有改善。
3. 多个 replacement 候选的 strong-up `switch_in_minus_out` 转正，说明边界信号有局部信息，但不足以抵消组合收益和强市胜率问题。
4. 当前缺少真实 `data/cache/industry_map.csv`；行业相关输入和暴露使用 `symbol_prefix_proxy_missing_data_cache_industry_map`，该结果不能替代 R4 的真实行业映射。

结论：

```text
R2B v1 全部 reject，无 gray zone。
不启动 R3。
后续已完成 R4A 真实行业映射、R2B-O oracle attribution、R2B v2 edge-gated replacement 与 Day 8 weight audit。
当前 R2B/R3 replacement 主线暂停，不写生产配置。
```

### 1.4 S2 的真实定位

S2 / `vol_to_turnover` 不是完整主策略，而是 defensive core。

当前默认研究基线：

| 参数 | 当前值 |
| --- | --- |
| `score` | `S2 = vol_to_turnover` |
| `portfolio_method` | `equal_weight` |
| `top_k` | `20` |
| `rebalance_rule` | `M` |
| `max_turnover` | `1.0` |
| `execution_mode` | `tplus1_open` |
| `prefilter` | `false` |
| `universe_filter` | `true` |
| `benchmark` | `market_ew_open_to_open` primary |

R1/R1F 共同确认：

| 切片 | 表现 |
| --- | --- |
| `strong_down` | 防守能力明确 |
| `narrow_breadth` | 弱市/窄市有效 |
| `strong_up` | 上涨参与不足 |
| `wide_breadth` | 扩散行情明显落后 |

所以后续不应要求 S2 单独追上市场，而应使用：

```text
S2 defensive core
+ 少量、有证据、有成本补偿的 replacement
```

### 1.5 R2B-O oracle replacement attribution 已完成

R2B-O 已完成，见：

- `docs/r2b_oracle_replacement_attribution_2026-04-28.md`
- `data/results/r2b_oracle_replacement_attribution_2026-04-28_summary.json`
- `data/results/r2b_oracle_replacement_attribution_2026-04-28_*.csv`

实验口径：

| 字段 | 当前值 |
| --- | --- |
| `eval_contract_version` | `r0_eval_execution_contract_2026-04-28` |
| `execution_contract_version` | `tplus1_open_buy_delta_limit_mask_2026-04-28` |
| `industry_map_source_status` | `real_industry_map` |
| `pair_base_rows` | `4,326,400` |
| 默认 horizon | `20d` |
| 默认 cost buffer | `15bp` |
| oracle replacement cap | `3` |

关键结果：

| 口径 | 最优 oracle daily proxy 年化超额 | 相对 S2 baseline | 平均半 L1 换手 | active rebalance share | 说明 |
| --- | ---: | ---: | ---: | ---: | --- |
| `all_states / S2_bottom_5 / candidate_buyable` | `+266.08%` | `+274.68pct` | `0.219` | `96.88%` | 全样本事后上限，仅作 diagnostic |
| `strong_up_or_wide / S2_bottom_5 / candidate_buyable` | `+27.42%` | `+36.01pct` | `0.142` | `25.00%` | 状态门控下仍有明确上限 |
| `strong_up_or_wide / S2_bottom_5 / candidate_top_pct_90` | `+17.98%` | `+26.57pct` | `0.142` | `25.00%` | 现有 upside score 前 10% 仍保留上限 |
| `BASELINE_S2_FIXED` | `-8.59%` | `0.00pct` | `0.084` | `0.00%` | baseline |

oracle capacity 观察：

1. `all_states` 下几乎每个月都能找到 3 个正 edge 替换，说明 replacement 问题不是“候选池完全没有上限”。
2. `strong_up_or_wide` 下活跃月份约 `25.8%`，平均正 edge slot 为 `0.77`，不是每月都应该替换。
3. `candidate_buyable` 的 oracle 结果对 `U2_A/U2_B/U2_C` 相同，因为 oracle 在完整可买池中事后选边；这不能证明现有 score 已经能学到 edge。
4. `candidate_top_pct_90` 在 `strong_up_or_wide` 下仍有 `+22.99pct` 到 `+26.57pct` 的 oracle proxy 增量，说明现有 score 对候选池裁剪有一定价值。
5. `state_strong_up_or_wide=True` 的普通 pair hit rate 反而低于全样本，状态门控不能单独作为 edge 证据；R2B v2 必须以 pair edge / score margin / feature bucket 为主门控。

R2B-O 判定：

```text
oracle 有明确上限。
现有 score 只能提供候选池裁剪，不足以直接作为替换决策。
进入 R2B v2 edge-gated replacement。
仍不启动 R3 boundary model；先做规则版 pairwise edge gate。
```

### 1.6 R2B v2 edge-gated replacement 已完成

R2B v2 已完成，见：

- `docs/r2b_edge_gated_replacement_v2_2026-04-28.md`
- `scripts/run_r2b_edge_gated_replacement_v2.py`
- `data/results/r2b_edge_gated_replacement_v2_2026-04-28_summary.json`
- `data/results/r2b_edge_gated_replacement_v2_2026-04-28_*.csv`

实验口径：

| 字段 | 当前值 |
| --- | --- |
| `eval_contract_version` | `r0_eval_execution_contract_2026-04-28` |
| `execution_contract_version` | `tplus1_open_buy_delta_limit_mask_2026-04-28` |
| `industry_map_source_status` | `real_industry_map` |
| `pair_base_rows` | `4,280,720` |
| `portfolio_method` | `defensive_core_edge_gated_replacement` |
| `max_replace` | `3` |
| 默认 horizon / cost buffer | `20d / 15bp` |

候选池与 gate：

| candidate | pool | state gate | selected pairs | active month share | 0 替换月份 |
| --- | --- | --- | ---: | ---: | ---: |
| `U3_A_real_industry_leadership__EDGE_GATED` | `S2_bottom_3 + candidate_top_pct_95` | `strong_up_and_wide` | `33` | `17.19%` | `82.81%` |
| `U3_B_buyable_leadership_persistence__EDGE_GATED` | `S2_bottom_3 + candidate_top_pct_90` | `up_or_wide_not_strong_down` | `46` | `25.00%` | `75.00%` |
| `U3_C_pairwise_residual_edge__EDGE_GATED` | `S2_bottom_5 + candidate_buyable` | `strong_up_or_wide` | `51` | `26.56%` | `73.44%` |

关键结果：

| candidate | daily proxy 年化超额 | 相对 S2 baseline | strong-up 中位超额 | strong-up positive share | strong-up switch-in-minus-out | 平均半 L1 换手 | 判定 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `BASELINE_S2_FIXED` | `-8.59%` | `0.00pct` | `-6.12%` | `15.38%` | `-2.37%` | `0.084` | baseline / reject |
| `U3_A_real_industry_leadership__EDGE_GATED` | `-8.34%` | `+0.25pct` | `-6.16%` | `15.38%` | `+2.35%` | `0.132` | gray zone |
| `U3_C_pairwise_residual_edge__EDGE_GATED` | `-9.25%` | `-0.66pct` | `-6.12%` | `23.08%` | `+6.90%` | `0.140` | reject |
| `U3_B_buyable_leadership_persistence__EDGE_GATED` | `-9.82%` | `-1.23pct` | `-6.12%` | `15.38%` | `-1.10%` | `0.133` | reject |

R2B v2 观察：

1. edge gate 已经解决 R2B v1 “状态允许后倾向填满 slot”的问题，0 替换月份占 `73%~83%`。
2. `U3_A` daily proxy 略高于 baseline，strong-up switch 转正，且换手增量 `+0.047` 可控，所以进入 gray zone。
3. `U3_A` 没有改善 strong-up 中位超额或 positive share，因此不是 full-backtest candidate，也不是 production candidate。
4. `U3_C` 的 strong-up positive share 从 `15.38%` 提升到 `23.08%`，switch 也明显转正，但 daily proxy 低于 baseline，不能进入 gray zone。
5. `U3_B` daily proxy、strong-down 防守和 switch quality 均弱于 `U3_A`，直接 reject。
6. 所有 R2B v2 候选 daily proxy 仍为负，没有候选达到 `>= +3%` full-backtest 门槛。

R2B v2 判定：

```text
U3_A 进入 gray zone，只允许补更细 slice / 阶段判断。
U3_B、U3_C reject。
不补正式 full backtest。
不写入生产配置。
复杂 R3 boundary model 仍不启动；后续 Day 8 weight audit 已确认 `U3_A` 不稳。
```

### 1.7 R2B v2 gray-zone weight audit 已完成

R2B v2 weight audit 已完成，见：

- `docs/r2b_v2_weight_audit_2026-04-28.md`
- `scripts/run_r2b_v2_weight_audit.py`
- `data/results/r2b_v2_weight_audit_2026-04-28_summary.json`
- `data/results/r2b_v2_weight_audit_2026-04-28_*.csv`

实验口径：

| 字段 | 当前值 |
| --- | --- |
| `target_candidate` | `U3_A_real_industry_leadership__EDGE_GATED` |
| `industry_map_source_status` | `real_industry_map` |
| `pair_base_rows` | `486,237` |
| `target_rule_pair_rows` | `4,755` |
| `selected_pairs` | `33` |
| 默认 horizon / cost buffer | `20d / 15bp` |

核心审计结果：

| 诊断 | 结果 | 含义 |
| --- | --- | --- |
| `pair_edge_score` bucket | `bucket_edge_spearman=-0.70`，`bucket_win_spearman=-0.90` | 手写综合分越高，真实 pair edge 反而越差 |
| `amount_expansion_diff` bucket | `bucket_edge_spearman=-0.90`，`bucket_win_spearman=-1.00` | 成交扩张差的正权重方向不成立 |
| `turnover_expansion_diff` bucket | `bucket_edge_spearman=-1.00`，`bucket_win_spearman=-1.00` | 换手扩张差的正权重方向不成立 |
| selected pair by year | `2025` 平均 realized edge `-6.71%`，`2026` 中位 realized edge `-19.14%` | 跨年不稳定 |
| selected pair by slot | slot 1 realized edge `-3.58%`，win rate `27.27%` | 最高分 slot 反而最差，不能收缩为 replace-1 |
| threshold / capacity | 最好仍是原 `thr0.68_replace3`，仅 `+0.25pct`；`replace1` 为 `-0.25pct` | gray zone 对具体手写规则敏感，不稳健 |
| simple baseline | `candidate_score_pct_only` 也能达到 `+0.12pct` | 复杂线性权重没有明显优于朴素候选分 |

审计判定：

```text
U3_A gray zone 不具备足够稳定性。
手写线性 pair_edge_score 没有得到 feature bucket 支持。
replace-1 不成立，不能作为 R2B v2.1 的自然收缩方向。
不启动 R3 classifier / ranker。
不补正式 full backtest。
不写生产配置。
R2B/R3 replacement 主线暂停；除非新增更强特征或更明确目标切片，否则不继续在当前 U3 线性规则上消耗预算。
```

---

## 2. 当前优缺点

### 2.1 主要优点

1. **评估契约已经可复用**
   R0 修复后，策略收益、benchmark、涨停买入失败、regime threshold 都有统一口径。

2. **daily-proxy-first 护栏有效**
   旧 `light_strategy_proxy` 和 `full_like_proxy` 已经不再作为准入证据。当前 gate 能更早淘汰明显无效候选。

3. **失败归因能力成熟**
   现在能把失败拆到 state、breadth、switch quality、topk boundary、持仓暴露和换手，而不是只看单条收益曲线。

4. **S2 防守核心有保留价值**
   它不适合做完整主策略，但适合做弱市 core。

5. **研究纪律清楚**
   当前没有把未 promotion 的研究配置写回生产，也没有用 val_rank_ic 替代可交易收益。

6. **replacement 问题存在可学习上限**
   R2B-O 显示在真实行业映射和 open-to-open 口径下，S2 边界附近存在足够多事后正 edge 样本；R2B v2 已把它收敛成可审计的规则版 edge gate，并通过 audit 明确了当前规则版不够稳。

### 2.2 主要缺点

1. **没有可 promotion 的 alpha**
   P1 原主线、旧 R2、R2B v1 都未通过 gate。

2. **upside 输入仍然粗糙**
   当前强势特征更像追涨/高换手，而不是稳定可交易领导力。

3. **主题与行业持续性信息仍粗糙**
   真实申万行业映射已经补齐，但 industry breadth 还需要从静态行业归属升级为可交易的行业扩散、持续性和拥挤度信号。

4. **replacement edge gate 未学到稳定边界**
   R2B v2 已经从状态触发改为 edge 触发，但 audit 显示手写 `pair_edge_score` 方向不稳，不能支撑 R3。

5. **状态门控太粗**
   `strong_up or wide` 会把部分非 risk-on 场景也打开 replacement。

6. **score 仍是手写线性 rank**
   `U2_A/U2_B/U2_C` 是合理 scout，但不是最终模型目标。

7. **目标函数还没有直接对齐 hold/replace**
   真正应该优化的是 `candidate - old_holding - cost_buffer`，而不是候选自身 forward return。

---

## 3. 新主线

新的主线分五层。

### Layer 1：真实行业映射

目的：

```text
让 industry breadth / intra-industry strength 从 board proxy 变成真实行业/主题扩散信号
```

没有真实行业映射前，不允许把行业相关 R2B 结果当作有效 alpha 证据。

### Layer 2：Oracle replacement attribution

目的：

```text
先判断候选池中是否存在足够多“事后能跑赢被替换持仓”的样本
```

如果 oracle 都没有可学习上限，不训练模型。

### Layer 3：R2B v2 edge-gated replacement

目的：

```text
只在有明确边际优势时替换 0 到 3 个名字
```

不再默认填满 slot。

### Layer 4：R3 boundary model

启动条件：

```text
R2B v2 至少进入 gray zone，或 replacement edge 明显稳定
```

第一阶段只做规则版 boundary gate；第二阶段才做 calibrated classifier / pairwise ranker。

### Layer 5：Promotion registry

目的：

```text
防止研究配置误进入生产
```

daily proxy 不是 promotion 终点。任何生产候选必须进入 promoted registry。

---

## 4. 下一阶段执行计划

### R4A：真实行业映射

**状态**：已完成（`2026-04-28`）。

**目标**：建立 `data/cache/industry_map.csv`，替代 symbol prefix fallback。

完成产物：

```text
data/cache/industry_map.csv
data/results/industry_map_quality_2026-04-28_summary.csv
docs/industry_map_quality_2026-04-28.md
```

质量摘要：

| 指标 | 结果 |
| --- | ---: |
| 当前 universe | `5184` |
| 映射覆盖 | `5184` |
| 覆盖率 | `100.00%` |
| unknown 比例 | `0.00%` |
| 重复 symbol | `0` |
| 一级行业数 | `31` |
| 数据源 | `akshare.stock_industry_clf_hist_sw+stock_industry_category_cninfo` |

说明：数据源为申万官方股票行业历史全量表 + 巨潮申万行业分类目录；`industry` 与 `industry_level1` 使用申万一级行业，`industry_level2` 使用申万二级行业。R2B 报告逻辑已增加 `industry_map_source_status`，只有 `real_industry_map` 才允许把 industry breadth 当作 alpha 证据。

历史任务清单保留如下，作为验收口径记录。

任务：

1. 新增或完善行业映射构建脚本，例如：

```text
scripts/build_industry_map.py
```

2. 输出：

```text
data/cache/industry_map.csv
data/results/industry_map_quality_YYYY-MM-DD_summary.csv
docs/industry_map_quality_YYYY-MM-DD.md
```

3. `industry_map.csv` 至少包含：

| 字段 | 要求 |
| --- | --- |
| `symbol` | 6 位股票代码 |
| `industry` | 兼容旧代码的行业列 |
| `industry_level1` | 一级行业 |
| `industry_level2` | 二级行业或可用替代 |
| `source` | 数据源 |
| `asof_date` | 映射日期 |

4. 质量检查必须输出：

| 指标 | 要求 |
| --- | --- |
| 当前 universe 覆盖率 | `>= 90%` |
| unknown 比例 | 必须显式报告 |
| 重复 symbol | 必须为 0 或解释 |
| 行业截面宽度 | 每个行业股票数分布 |
| fallback 使用 | 若使用 fallback，必须标记为 `fallback_only_for_diagnostic` |

5. 修改 R2B 报告逻辑：

```text
industry_map_source != symbol_prefix_proxy_missing_data_cache_industry_map
```

才允许把 industry breadth 结果当作 alpha 证据。

验收：

| 项目 | 要求 |
| --- | --- |
| 文件 | `data/cache/industry_map.csv` 存在 |
| 覆盖率 | 当前 universe 覆盖率 `>= 90%` |
| PIT 说明 | 至少记录 `source/asof_date` |
| 测试 | map 存在时加载真实行业；缺失时报告 fallback |
| 报告 | R2B/R3 必须输出真实行业暴露 |

### R2B-O：Oracle replacement attribution

**状态**：已完成（`2026-04-28`）。

**目标**：在继续造新 score 前，先判断 replacement 问题是否有可学习上限。

核心问题：

```text
在每个可替换状态下，如果事后知道未来收益，
候选池里是否真的存在足够多能跑赢 S2 边界持仓的股票？
```

已落地脚本：

```text
scripts/run_r2b_oracle_replacement_attribution.py
```

输入：

1. S2 defensive Top-20 权重。
2. R4A 真实行业映射。
3. 当前 R2B 候选池或新候选池。
4. open-to-open forward return。
5. 交易成本 buffer。

候选池定义：

| pool | 定义 |
| --- | --- |
| `S2_bottom_3` | S2 Top-20 中 defensive score 最弱 3 个 |
| `S2_bottom_5` | S2 Top-20 中 defensive score 最弱 5 个 |
| `candidate_top_pct_90` | upside score 前 10% |
| `candidate_top_pct_95` | upside score 前 5% |
| `candidate_buyable` | 次日开盘可买且非过热 |

标签：

| label | 定义 |
| --- | --- |
| `pair_edge_5d` | `new_forward_5d - old_forward_5d - cost_buffer` |
| `pair_edge_10d` | `new_forward_10d - old_forward_10d - cost_buffer` |
| `pair_edge_20d` | `new_forward_20d - old_forward_20d - cost_buffer` |
| `replace_win` | `pair_edge_horizon > 0` |

必须输出：

| 产物 | 说明 |
| --- | --- |
| `oracle_hit_rate_by_state` | 各状态下候选跑赢旧持仓比例 |
| `oracle_capacity` | 每月最多有几个正 edge 替换 |
| `best_possible_replace_3_excess` | oracle replace-3 的理论上限 |
| `feature_bucket_monotonicity` | 现有 feature 是否能区分正/负 edge |
| `state_gate_precision` | `strong_up/wide` 是否真的提高 edge 胜率 |
| `cost_sensitivity` | 成本 buffer 对可替换样本数的影响 |

完成产物：

```text
docs/r2b_oracle_replacement_attribution_2026-04-28.md
data/results/r2b_oracle_replacement_attribution_2026-04-28_oracle_hit_rate_by_state.csv
data/results/r2b_oracle_replacement_attribution_2026-04-28_oracle_capacity_by_month.csv
data/results/r2b_oracle_replacement_attribution_2026-04-28_oracle_capacity_summary.csv
data/results/r2b_oracle_replacement_attribution_2026-04-28_best_possible_replace_3_excess.csv
data/results/r2b_oracle_replacement_attribution_2026-04-28_feature_bucket_monotonicity.csv
data/results/r2b_oracle_replacement_attribution_2026-04-28_state_gate_precision.csv
data/results/r2b_oracle_replacement_attribution_2026-04-28_cost_sensitivity.csv
data/results/r2b_oracle_replacement_attribution_2026-04-28_oracle_selected_pairs.csv
data/results/r2b_oracle_replacement_attribution_2026-04-28_summary.json
```

实际判定：

| 结果 | 动作 |
| --- | --- |
| oracle 有明确上限 | 进入 R2B v2 |
| 现有 score 对候选池裁剪有效，但不能直接决定替换 | 做 pairwise feature / rule，不训练复杂模型 |
| `strong_up_or_wide` 普通 pair precision 不提升 | R2B v2 必须加入 edge/confidence gate，不能只靠状态触发 |

原判定框架保留：

| 结果 | 动作 |
| --- | --- |
| oracle 仍无法显著跑赢 baseline | 停止 R2B，回到数据/目标周期 |
| oracle 有上限，但现有 score 分不出来 | 做 pairwise feature / rule，不训练复杂模型 |
| oracle 有上限，简单 rule 能捕捉一部分 | 进入 R2B v2 |
| oracle replace-3 已接近 `>= +3%` daily proxy 上限 | 允许 R2B v2 后补更细 gate |

历史计划产物模板：

```text
docs/r2b_oracle_replacement_attribution_YYYY-MM-DD.md
data/results/r2b_oracle_replacement_attribution_YYYY-MM-DD_*.csv
data/results/r2b_oracle_replacement_attribution_YYYY-MM-DD_summary.json
```

### R2B v2：Edge-gated replacement

**启动条件**：R4A 完成，且 R2B-O 显示存在可学习上限。

**当前状态**：已完成（`2026-04-28`）；`U3_A` 进入 gray zone，`U3_B/U3_C` reject。

完成产物：

```text
scripts/run_r2b_edge_gated_replacement_v2.py
tests/test_r2b_edge_gated_replacement_v2.py
docs/r2b_edge_gated_replacement_v2_2026-04-28.md
data/results/r2b_edge_gated_replacement_v2_2026-04-28_leaderboard.csv
data/results/r2b_edge_gated_replacement_v2_2026-04-28_replacement_diag_long.csv
data/results/r2b_edge_gated_replacement_v2_2026-04-28_replacement_count_distribution.csv
data/results/r2b_edge_gated_replacement_v2_2026-04-28_selected_pairs.csv
data/results/r2b_edge_gated_replacement_v2_2026-04-28_overlap_long.csv
data/results/r2b_edge_gated_replacement_v2_2026-04-28_industry_exposure_long.csv
data/results/r2b_edge_gated_replacement_v2_2026-04-28_regime_long.csv
data/results/r2b_edge_gated_replacement_v2_2026-04-28_breadth_long.csv
data/results/r2b_edge_gated_replacement_v2_2026-04-28_year_long.csv
data/results/r2b_edge_gated_replacement_v2_2026-04-28_year_strong_up_improvement.csv
data/results/r2b_edge_gated_replacement_v2_2026-04-28_switch_long.csv
data/results/r2b_edge_gated_replacement_v2_2026-04-28_monthly_long.csv
data/results/r2b_edge_gated_replacement_v2_2026-04-28_summary.json
```

目标：

```text
S2 defensive core Top-20
在状态和 pair edge 同时通过时：
    替换 0 到 3 个名字
其他情况：
    hold
```

R2B v2 只保留一个主组合表达：

```text
defensive core + edge-weighted overlay / replacement
```

不再同时扩 `Replace-3 / Replace-5 / Overlay-10` 大矩阵。

候选输入最多 3 个：

| candidate | 组成 | 假设 |
| --- | --- | --- |
| `U3_A_real_industry_leadership` | 真实行业 breadth + 行业内强度 + 行业内成交扩张 + 行业扩散持续性 | strong-up 来自行业扩散，不是单票追涨 |
| `U3_B_buyable_leadership_persistence` | 20/60 日相对强度一致性 + 可买突破 + gap/涨停风险过滤 + overheat penalty | 买能持续的领导股，不追一字板 |
| `U3_C_pairwise_residual_edge` | 候选相对旧持仓的强度差、成交差、S2 残差弹性差、流动性差、过热差 | 直接解释为什么新名字应替换旧名字 |

replacement gate 必须从“状态触发”改成“边际优势触发”：

| gate | 要求 |
| --- | --- |
| state gate | `strong_up & wide` 优先；可对比 `up_or_wide_but_not_strong_down` |
| edge gate | `pair_edge_score >= threshold` |
| cost gate | `expected_edge_after_cost > 0` |
| confidence gate | `state_confidence` 或 `score_margin` 达标 |
| buyability gate | 次日开盘可买，不是一字涨停 |
| risk gate | 无 limit-down 近期风险，无极端 overheat |
| industry gate | 替换后行业集中度不过阈值 |
| capacity gate | 每月替换 `0/1/2/3`，不得默认填满 |

R2B v2 的诊断重点：

1. 替换次数分布，而不是平均替换个数。
2. 0 替换月份是否足够多。
3. 替换后 strong-up positive share 是否改善。
4. `switch_in_minus_out` 是否稳定为正。
5. `topk_minus_next` 是否不明显为负。
6. 行业暴露是否可解释。
7. 换手上升是否有收益补偿。

最低验收：

| 指标 | 要求 |
| --- | --- |
| daily proxy | 不低于 `BASELINE_S2_FIXED` |
| strong-up 中位超额 | 至少改善 `+2pct`，或 positive share 提升 `>= +10pct` |
| strong-down 中位超额 | 劣化不超过 `2pct` |
| 平均半 L1 换手增量 | 默认不超过 `+0.10` |
| strong-up switch-in-minus-out | 不显著为负，最好转正 |
| topk-minus-next | strong-up 不明显反向 |
| 2021/2025/2026 | 至少两个关键年份 strong-up 改善 |
| industry source | 必须是真实行业映射，不得是 prefix fallback |

判定：

| 结果 | 动作 |
| --- | --- |
| daily proxy `< 0%` 且低于 baseline | reject，不启动 R3 |
| daily proxy `< 0%` 但高于 baseline，且 strong-up 明显改善 | gray zone，只归档诊断 |
| daily proxy `0% ~ +3%` | gray zone，可补更细 slice，不补 production |
| daily proxy `>= +3%` | full-backtest candidate |

实际判定：

| candidate | 判定 | 动作 |
| --- | --- | --- |
| `U3_A_real_industry_leadership__EDGE_GATED` | gray zone | 只允许补更细 slice / 阶段判断，不补 production |
| `U3_B_buyable_leadership_persistence__EDGE_GATED` | reject | 停止 |
| `U3_C_pairwise_residual_edge__EDGE_GATED` | reject | 停止，保留 switch 诊断 |

### R3：Boundary model

**当前状态**：不启动。R2B v2 虽有 `U3_A` gray zone，但 Day 8 weight audit 未支持其稳定性，当前不训练 classifier / ranker。

启动条件必须满足至少一条：

1. R2B v2 进入 gray zone。
2. R2B v2 daily proxy 不低于 baseline，且 strong-up 明显改善。
3. R2B v2 的 `switch_in_minus_out` 在 strong-up 稳定转正，并非少数月份支撑。
4. oracle attribution 与规则版 gate 都显示有可学习上限。

当前 Day 8 复核结果：

```text
上述条件未满足。
U3_A gray zone 不稳定，replace-1 不成立，feature bucket 反向。
R3 暂停。
```

R3 第一阶段只做规则版：

```text
hold unless replacement has clear edge
```

R3 第二阶段才允许最小模型：

| 模型 | 用途 | 优先级 |
| --- | --- | --- |
| Logistic / calibrated classifier | 判断替换是否通过 | 第一优先 |
| Pairwise ranker | 判断 new 是否优于 old | 第二优先 |
| XGBoost regression | 对照 | 第三优先 |
| 深度序列模型 | 小样本观察 | 暂不作为主线 |

训练样本：

```text
(rebalance_date, old_symbol, new_symbol)
```

标签：

```text
new_forward_o2o_return - old_forward_o2o_return - cost_buffer > 0
```

验证：

1. walk-forward by year。
2. 单独报告 `2021 / 2025 / 2026`。
3. 单独报告 strong-up / wide breadth。
4. 如果 AUC 好但 daily proxy 或 switch quality 差，直接淘汰。

### R5：生产边界与配置治理

目标：

```text
防止研究配置误进入日更推荐
```

规则：

1. 日更推荐只允许使用 promotion registry 中的配置。
2. `daily proxy` 不是 promotion 终点。
3. `gray zone` 不是 production candidate。
4. 当前没有任何 P1/R2/R3 候选满足 promotion。
5. 未 promotion 的研究配置不得写入 `config.yaml.example` 的默认主线。
6. `config.yaml.backtest` 继续作为 canonical 研究入口。
7. 生产 `config.yaml` 与研究 `config.yaml.backtest` 必须显式分离。

建议新增：

```text
configs/promoted/README.md
configs/promoted/promoted_registry.json
```

registry 至少记录：

| 字段 | 说明 |
| --- | --- |
| `config_id` | promoted 配置 ID |
| `promotion_date` | promotion 日期 |
| `full_backtest_report` | 正式回测报告 |
| `daily_proxy_report` | daily proxy 报告 |
| `oos_report` | rolling/slice OOS |
| `state_slice_report` | strong-up/strong-down 等 |
| `boundary_report` | switch/topk boundary |
| `owner_decision` | 人工确认 |

---

## 5. 数据与新数据预算

### 5.1 fund flow

当前质量报告显示：

1. fund flow 最新日期晚于日线最新日期。
2. 覆盖区间内仍存在未匹配行。
3. 部分标的不在当前日线 universe。

处理：

1. 先补齐日线到资金流最新日期。
2. 复跑 `scripts/run_newdata_quality_checks.py`。
3. 只有质量报告 `ok=True`，或明确解释剩余缺口后，才恢复研究预算。

恢复前禁止：

1. 跑 `G2/G4` 主线网格。
2. 把 fund flow 直接拼入 P1/R2 主线。
3. 用资金流单因子救当前失败策略。

### 5.2 shareholder

shareholder PIT 质量较保守，但 alpha 证据弱。

处理：

1. 保留低优先级观察。
2. 不进入 R2B/R3 主线。
3. 只有出现明确机制证据时再重启。

### 5.3 universe ablation

当前默认 universe：

```text
min_amount_20d >= 50,000,000
roe_ttm > 0
```

这可能压制 strong-up 弹性。下一阶段允许做 universe ablation，但只作为诊断，不等于放宽生产股票池。

候选：

| universe | 用途 |
| --- | --- |
| `U0_current` | 当前默认 universe |
| `U1_no_roe_gate` | 检查 ROE 正值门槛是否压制上涨弹性 |
| `U2_liquidity_only` | 只保留流动性门槛 |
| `U3_liquidity_tighter` | 提高流动性门槛，排除难交易小票 |

优先观察：

1. oracle replacement capacity。
2. strong-up/wide 的 positive edge share。
3. buy-fail weight。
4. 换手和成交可行性。
5. 行业暴露是否恶化。

---

## 6. Gate 与 Promotion 标准

### 6.1 Daily proxy gate

R0 修复后继续使用三档：

| daily proxy 年化超额 | 状态 | 处理 |
| ---: | --- | --- |
| `< 0%` | `reject` | 默认停止，不补正式 full backtest |
| `0% ~ +3%` | `gray_zone` | 只归档诊断，默认不补正式 full backtest |
| `>= +3%` | `full_backtest_candidate` | 允许补正式 full backtest |

例外复核条件：

1. R0 修复导致旧候选方向显著改变。
2. daily proxy 为正但正式 full backtest 为负。
3. `abs(daily_minus_full) > 5pct`。
4. 方向错配累计超过 2 次。

### 6.2 Mechanism gate

用于判断是否值得进入下一层研究，不等同于 promotion。

R2B/R3 候选必须报告：

| 指标 | 要求 |
| --- | --- |
| strong-up median excess | 相对 baseline 改善 |
| strong-up positive share | 相对 baseline 改善，目标 `>= +10pct` |
| switch-in-minus-out | 不显著为负，最好转正 |
| topk-minus-next | 不明显反向 |
| replacement count | 不能状态触发后默认填满 |
| turnover delta | 必须有收益补偿 |
| industry exposure | 真实行业映射下可解释 |

### 6.3 Production promotion gate

候选进入 production 前必须同时满足：

1. 正式 full backtest 的 `annualized_excess_vs_market >= 0`。
2. 年度超额中位数 `>= 0`。
3. rolling OOS 超额中位数不为负。
4. slice OOS 超额中位数不为负。
5. `2021 / 2025 / 2026` 不能明显更差。
6. strong-up 中位超额或跑赢比例必须相对当前 S2 基线改善。
7. strong-down 防守不能明显劣化。
8. MaxDD 不明显劣于当前默认研究基线，除非超额改善足够大且可解释。
9. 换手上升必须有明确收益补偿。
10. `switch_in_minus_out` 不能显著为负。
11. `topk_minus_next` 不能明显反向。
12. 执行口径、benchmark 口径、Top-K、成本、universe、state threshold 必须同窗可追溯。
13. 必须记录 `eval_contract_version` 和 `execution_contract_version`。
14. 必须进入 promotion registry。

---

## 7. 未来 10 个工作日队列

### Day 1-2：R4A 行业映射

**状态**：已完成（`2026-04-28`）。

历史任务：

1. 建立 `data/cache/industry_map.csv`。
2. 输出质量报告。
3. 增加加载与 fallback 测试。
4. 更新 R2B 行业暴露读取逻辑。

通过条件：

1. 当前 universe 覆盖率 `>= 90%`。
2. `industry_map_source` 不再是 prefix fallback。
3. 报告记录 `source/asof_date`。

### Day 3-4：R2B-O oracle attribution

**状态**：已完成（`2026-04-28`）。

历史任务：

1. 构造 pairwise old/new 样本。
2. 计算 `5d/10d/20d` pair edge。
3. 输出 oracle capacity 与 state gate precision。
4. 判断是否存在可学习上限。

通过条件：

1. 至少在 strong-up/wide 中看到正 edge capacity。
2. oracle replace-3 对 baseline 有明确理论改善。
3. 成本 buffer 后仍有足够样本。

### Day 5-7：R2B v2

**状态**：已完成（`2026-04-28`）。

任务：

1. 固化 3 个候选池对照：`S2_bottom_3 + candidate_top_pct_95`、`S2_bottom_3 + candidate_top_pct_90`、`S2_bottom_5 + candidate_buyable`。
2. 构造规则版 `pair_edge_score`，优先使用 `score_margin`、`rel_strength_diff`、成交扩张差、过热差和行业约束。
3. 只保留 1 个主组合表达：`defensive core + edge-gated replacement`。
4. 替换数量改为 `0/1/2/3`，不得默认填满。
5. 输出 replacement count distribution、pair edge、switch/topk boundary、行业暴露。

通过条件：

1. daily proxy 不低于 `BASELINE_S2_FIXED`。
2. strong-up positive share 或中位超额明显改善。
3. 换手增量可解释。

实际结果：

1. `U3_A` daily proxy `-8.34%`，略高于 baseline `-8.59%`，但 strong-up 中位超额和 positive share 未改善，判定为 gray zone。
2. `U3_C` strong-up positive share 与 switch 改善，但 daily proxy 低于 baseline，reject。
3. `U3_B` 全面弱于 `U3_A`，reject。
4. 所有候选均未达到 `>= +3%` full-backtest 门槛。

### Day 8：阶段判断

**状态**：已完成（`2026-04-28`）。

任务：

1. 若 R2B v2 全面 reject，停止 replacement 主线。
2. 若 R2B v2 进入 gray zone，补更细 slice。
3. 若 R2B v2 达到 `>= +3%`，允许补正式 full backtest。
4. 只有 gray zone 或更好才启动 R3。

实际执行：

1. 已新增并运行 `scripts/run_r2b_v2_weight_audit.py`。
2. `U3_A` gray zone 未通过 weight audit：`pair_edge_score` bucket 反向，slot 1 真实 edge 为负，`replace1` sensitivity 弱于 baseline。
3. 不启动 R3，不补正式 full backtest。
4. R2B/R3 replacement 主线暂停，除非后续新增更强特征或更明确切片。

### Day 9-10：R5 配置治理

**状态**：已完成（`2026-04-28`）。

任务：

1. 新增 promoted registry 骨架。
2. 明确日更推荐只能读取 promoted 配置。
3. 检查 `config.yaml.example` 不包含未 promotion 研究主线。

完成产物：

```text
configs/promoted/README.md
configs/promoted/promoted_registry.json
docs/r5_config_governance_2026-04-28.md
tests/test_promoted_registry.py
```

当前 registry 结论：

```text
promoted_configs = []
active_promoted_config_id = null
has_promoted_research_candidates = false
```

实际判定：

1. 当前没有任何 P1/R2/R3 研究候选满足 production promotion。
2. `U3_A` gray zone 已被 Day 8 weight audit 判定不稳，不进入生产。
3. `U3_B/U3_C` reject，不进入生产。
4. `config.yaml.backtest` 继续作为 canonical 研究入口；未 promotion 研究配置不得写入 `config.yaml.example`。

建议阶段总结产物：

```text
docs/phase_decision_industry_oracle_replacement_YYYY-MM-DD.md
data/results/phase_decision_industry_oracle_replacement_YYYY-MM-DD_summary.json
```

---

## 8. 停止清单

以下方向冻结，除非出现新数据、口径修复后反转、或明确机制证据：

1. 不继续做 P1/G0 近邻标签微调。
2. 不用旧 `light_strategy_proxy` 或 `full_like_proxy` 触发正式回测。
3. 不用 `val_rank_ic` 替代 daily proxy 或正式 full backtest。
4. 不继续做纯 Top-K、buffer、分层等权的小参数网格。
5. 不继续加大 R2 dual sleeve upside 权重。
6. 不把 pure upside Top-20 当作可 promotion 候选。
7. 不启动 R3，除非 R2B v2 至少进入 gray zone。
8. 不把 `weekly_kdj_*` 写回默认 `composite_extended`。
9. 不扩 `weekly_kdj` interaction 网格。
10. 不重复跑 `G2/G4` 直拼 fund flow。
11. 不继续为 shareholder 做单因子或线性权重主线。
12. 不把任何未 promotion 的研究配置写入生产日更推荐。
13. 不使用全样本 regime 分位数做可交易状态决策。
14. 不把 prefix proxy industry 当真实行业证据。
15. 不在 oracle attribution 失败后训练 boundary model。
16. 不把深度模型作为下一阶段主线。

---

## 9. 保留清单

这些能力继续保留：

1. `daily-proxy-first` runner 和三档 gate。
2. 正式 full backtest 作为 promotion 必要条件。
3. R0 评估/执行契约。
4. open-to-open primary benchmark。
5. buy-fail diagnostic。
6. `*_daily_proxy_leaderboard.csv`。
7. `*_monthly_long.csv` 与 state slice。
8. `*_switch_long.csv` 与 switch quality。
9. `*_topk_boundary.csv` 或等价 boundary diagnostic。
10. strong-up / strong-down / breadth 切片。
11. S2 / `vol_to_turnover` 作为 defensive core。
12. G0 技术特征作为 baseline diagnostic。
13. fund flow / shareholder 数据链路，但必须先过质量 gate。
14. `config.yaml.backtest` 作为 canonical 研究入口。
15. 历史配置快照保留在 `configs/backtests/`。
16. R1 失败归因框架继续作为新候选诊断模板。

---

## 10. 产出规范

每轮新实验至少保留：

### 10.1 配置身份

| 字段 | 要求 |
| --- | --- |
| `research_topic` | 必填 |
| `research_config_id` | 必填 |
| `output_stem` | 必填 |
| `result_type` | 必填 |
| `config_source` | 必填 |

### 10.2 评估契约

| 字段 | 要求 |
| --- | --- |
| `eval_contract_version` | 必填 |
| `execution_contract_version` | 必填 |
| `benchmark_return_mode` | 必填 |
| `state_threshold_mode` | 必填 |
| `state_lag` | 必填 |
| `lookahead_check` | 必须为 `pass` 或标注 diagnostic only |

### 10.3 组合口径

| 字段 | 要求 |
| --- | --- |
| `benchmark_symbol` | 必填 |
| `top_k` | 必填 |
| `rebalance_rule` | 必填 |
| `portfolio_method` | 必填 |
| `execution_mode` | 必填 |
| `max_turnover` | 必填 |
| 成本假设 | 必填 |

### 10.4 数据口径

| 字段 | 要求 |
| --- | --- |
| `prefilter` | 必填 |
| `universe_filter` | 必填 |
| `industry_map_source` | R2B/R3 必填 |
| `feature_spec` | 必填 |
| `label_spec` 或 `rule_spec` | 必填 |
| cache 路径与 schema/version | 若使用 cache 必填 |

### 10.5 结果文件

至少输出：

1. summary CSV 或 JSON。
2. detail CSV 或 period detail。
3. leaderboard。
4. monthly/state slice。
5. switch quality。
6. topk/boundary diagnostic。
7. buy-fail diagnostic。
8. industry exposure。
9. 一页结论文档。

结论文档必须回答：

1. 只改了什么。
2. 相对 S2 defensive baseline 有何变化。
3. 对 primary benchmark 有何变化。
4. strong-up / strong-down 有何变化。
5. switch quality 和 topk boundary 有何变化。
6. 是否允许进入下一层验证。

---

## 11. 证据索引

### 主计划与路线

- `docs/plan.md`
- `docs/plan-04-20.md`

### R0 评估/执行契约

- `docs/r0_eval_execution_contract_fix_2026-04-28.md`
- `data/results/r0_eval_execution_contract_fix_2026-04-28_summary.json`
- `data/results/r0_eval_execution_contract_fix_2026-04-28_benchmark_compare.csv`

### R1F fixed baseline

- `docs/r1f_fixed_baseline_rerun_2026-04-28.md`
- `scripts/run_r1f_fixed_baseline_rerun.py`
- `data/results/r1f_fixed_baseline_rerun_2026-04-28_*.csv`
- `data/results/r1f_fixed_baseline_rerun_2026-04-28_summary.json`

### R2B replacement v1

- `docs/r2b_tradable_upside_replacement_v1_2026-04-28.md`
- `scripts/run_r2b_tradable_upside_replacement_v1.py`
- `data/results/r2b_tradable_upside_replacement_v1_2026-04-28_*.csv`
- `data/results/r2b_tradable_upside_replacement_v1_2026-04-28_summary.json`

### R2B oracle 与 v2 edge gate

- `docs/r2b_oracle_replacement_attribution_2026-04-28.md`
- `scripts/run_r2b_oracle_replacement_attribution.py`
- `data/results/r2b_oracle_replacement_attribution_2026-04-28_*.csv`
- `data/results/r2b_oracle_replacement_attribution_2026-04-28_summary.json`
- `docs/r2b_edge_gated_replacement_v2_2026-04-28.md`
- `scripts/run_r2b_edge_gated_replacement_v2.py`
- `data/results/r2b_edge_gated_replacement_v2_2026-04-28_*.csv`
- `data/results/r2b_edge_gated_replacement_v2_2026-04-28_summary.json`
- `docs/r2b_v2_weight_audit_2026-04-28.md`
- `scripts/run_r2b_v2_weight_audit.py`
- `data/results/r2b_v2_weight_audit_2026-04-28_*.csv`
- `data/results/r2b_v2_weight_audit_2026-04-28_summary.json`

### R1 strong-up 归因

- `docs/p1_strong_up_failure_attribution_2026-04-27.md`
- `scripts/run_p1_strong_up_attribution.py`
- `data/results/p1_strong_up_failure_attribution_2026-04-27_*.csv`

### P1 proxy 校准与失败候选

- `docs/p1_daily_bt_like_proxy_calibration_2026-04-27.md`
- `docs/p1_proxy_calibration_history_2026-04-27.md`
- `docs/p1_failure_diagnostics_2026-04-26.md`
- `docs/p1_label_objective_experiment_2026-04-26.md`
- `docs/p1_marketrel_g1_gap_diagnostics_2026-04-26.md`
- `docs/p1_marketrel_state_diagnostics_2026-04-27.md`
- `docs/p1_monthly_investable_label_smoke_2026-04-27.md`
- `docs/p1_up_capture_market_relative_g0_smoke_2026-04-27.md`
- `docs/p1_rank_fusion_long_horizon_g0_smoke_2026-04-27.md`
- `docs/p1_top_bucket_rank_fusion_g0_smoke_2026-04-27.md`

### 已冻结方向

- `docs/p2_upside_sleeve_v1_2026-04-27.md`
- `docs/p2_regime_aware_dual_sleeve_v1_2026-04-28.md`
- `docs/alpha_factor_scout_2026-04-24_weekly_kdj.md`
- `docs/p1_weekly_interaction_ab_2026-04-26.md`
- `docs/p1_rank_direction_rerun_2026-04-26.md`

### 新数据质量

- `docs/newdata_quality_current_families_fund_flow-shareholder_flow_a_share_fund_flow_holder_a_share_shareholder_daily_a_share_daily_lag_30_width_100.md`
- `docs/alpha_factor_scout_2026-04-23_shareholder_smoke.md`
- `scripts/run_newdata_quality_checks.py`
- `scripts/fetch_fund_flow.py`
- `scripts/fetch_shareholder.py`

### 核心代码入口

- `config.yaml.backtest`
- `configs/backtests/README.md`
- `scripts/run_backtest_eval.py`
- `src/backtest/engine.py`
- `src/models/xtree/p1_workflow.py`
- `src/market/tradability.py`
- `scripts/run_r2b_tradable_upside_replacement_v1.py`
- `scripts/run_r2b_oracle_replacement_attribution.py`
- `scripts/run_r2b_edge_gated_replacement_v2.py`
- `scripts/run_r2b_v2_weight_audit.py`

---

## 12. 当前一句话路线

**停止 P1 标签近邻微调和旧 R2 sleeve 扩权；保留 S2 作为 defensive core；真实行业映射、oracle replacement attribution、R2B v2 edge-gated replacement、gray-zone weight audit 与 R5 配置治理已完成；`U3_A` gray zone 不稳，R2B/R3 replacement 主线暂停，promoted registry 当前为空，没有任何研究候选进入生产配置。**
