# 量化月度选股主计划

**文档角色**：当前唯一主计划（canonical）  
**更新时间**：`2026-05-04`  
**当前阶段**：M10，成本与执行压力（进行中）  
**研究终点**：每月输出可解释、可回测、PIT-safe、可执行约束清楚的 Top-K 股票推荐名单  
**生产状态**：无研究候选进入生产；`configs/promoted/promoted_registry.json` 继续为空  
**当前结论**：M8 行业约束自然化已完成，soft industry risk-budget 在 Top20/Top30 上产生 3 个通过 M8 natural gate 的研究候选；M10 成本敏感性框架已完成（10/30/50 bps 三档），容量分析与涨停/VWAP 压力测试框架已就位；hard-cap baseline 仍是收益上限更强的 stress 对照；披露日历填充（P2-3）与 next_trade_date 边界校验（P2-4）已完成；P0 紧急修复全部完成；仍不能 promotion，待 M10 压力测试结果确认 after-cost excess 在 30bps 下 > 0  
**归档入口**：`docs/reports/2026-04/plan-04-20.md` 仅保留历史执行记录，不再承担主计划职责  
**问题追踪**：`docs/plan-05-03.md`（P0-P3 完整问题清单）；`docs/plan-05-04.md`（工程质量审计，22/22 项全部完成 ✅）

---

## 0. 当前决策

项目主线已经从“旧持仓 replacement”切换为“月度截面选股”。

旧问题：

```text
edge(old_stock, new_stock, date)
= future_return(new_stock) - future_return(old_stock) - cost - risk_penalty
```

当前问题：

```text
score(stock, month_end)
-> rank tradable universe
-> select monthly Top-K
```

当前最重要的判断：

1. **不 promotion M7/M8/M9 报告**。M8 已从 hard cap 诊断推进到自然化约束，但真实成本压力、买入失败和冲击成本仍未完成。
2. **M8 当前通过候选来自 soft industry risk-budget，而不是 hard cap**。`U1 + M5 ElasticNet market_excess + Top20 + gamma0.20` after-cost 月均超额约 `0.018338`，相对 hard-cap best 差 `-0.004031`，Rank IC 约 `0.103597`，Top-K vs next-K 约 `0.013406`，平均最大行业占比约 `0.103947`，集中度 pass rate `1.0`。
3. **hard cap 仍是 stress 上限，不是生产形态**。`U1_liquid_tradable + Top20 + indcap3` after-cost 月均超额约 `0.022369`，平均最大行业占比 `0.150000`，集中度 pass rate `1.0`；后续只作为 M10/M12 对照，不作为主候选形态。
4. **M6 仍只作 watchlist / sleeve**。`U2 + M6_xgboost_rank_ndcg + Top20` 在 industry cap 后仍有收益证据，但 Rank IC 很弱，不能单独 promotion。
5. **下一步不再堆模型**。优先对 M8 natural 候选做 30/50 bps、买入失败、冲击成本、容量和换手压力测试。
6. **生产边界不变**。任何研究候选通过完整 gate 前，不写入 `config.yaml.example` 或 promoted registry。

一句话路线：

```text
PIT 数据
-> 月度截面特征表
-> 多源 tabular ranker
-> 行业/状态/成本约束复核
-> 研究版推荐
-> promotion package
```

---

## 1. 研究契约

### 1.1 任务定义

每个月最后一个可交易信号日 `t`：

1. 只使用 `t` 日收盘后已经可见、且 PIT-safe 的数据。
2. 在 `t+1` 开盘可买的股票池中打分排序。
3. 输出 Top-K 月度研究名单。
4. 标签为 `t+1` 开盘到下一次月度换仓开盘的 open-to-open 收益。

形式化：

```text
x[i, t] = stock i at month-end t 的可观测特征
y[i, t] = stock i 从 t+1 open 到 next rebalance open 的 forward return
s[i, t] = f(x[i, t])
select Top-K by s[i, t]
```

### 1.2 默认口径

| 字段 | 当前默认 |
| --- | --- |
| `rebalance_rule` | `M` |
| `signal_date` | 每月最后一个交易日 |
| `execution_mode` | `tplus1_open` |
| `label_return_mode` | `open_to_open` |
| `benchmark` | `market_ew_open_to_open` |
| `top_k` | `20 / 30 / 50` 并行诊断，主报告 `20 / 30` |
| `portfolio_method` | 研究评价默认等权；生产前必须补行业/单票/换手约束 |
| `candidate_pool` | 宽研究池，只做可交易、数据有效、极端风险准入 |
| `industry_map` | 真实申万行业映射 |

### 1.3 候选池

候选池是准入层，不是 alpha 层。

| pool | 定义 | 当前用途 |
| --- | --- | --- |
| `U0_all_tradable` | 当前 OHLCV 有效且次日开盘可买 | 判断全可交易空间上限 |
| `U1_liquid_tradable` | U0 + 历史长度 + 20 日成交额门槛 | 默认训练池与 M5 主证据 |
| `U2_risk_sane` | U1 + 排除极端涨跌停路径、极端波动/换手、绝对高位 | M6/M7 watchlist 口径 |
| `U3_quality_optional` | U2 + 质量门槛 | 仅作 ablation |

当前规则：

```text
主训练证据：U1_liquid_tradable + U2_risk_sane 并行
主报告 watchlist：U2_risk_sane
下一阶段：U1/U2 均必须加入行业集中度约束对照
```

### 1.4 标签族

| label | 定义 | 用途 |
| --- | --- | --- |
| `label_forward_1m_o2o_return` | 下月 open-to-open 原始收益 | 回归 / 诊断 |
| `label_forward_1m_excess_vs_market` | 原始收益减同月市场等权收益 | 主排序标签 |
| `label_forward_1m_industry_neutral_excess` | 原始收益减同月行业等权收益 | 控制行业 beta |
| `label_future_return_quantile` | 同月截面收益分桶 | 排序 relevance |
| `label_future_top_20pct` | 是否进入未来收益前 20% | top-bucket classifier |
| `label_future_top_10pct` | 是否进入未来收益前 10% | 高置信候选诊断 |
| `label_future_bottom_20pct` | 是否进入未来底部 20% | 风险过滤 |

主线优化 market-relative / industry-neutral ranking，不追求单票精确涨幅。

### 1.5 Oracle 角色

Oracle Top-K 只用于判断上限和可分性，不作为训练目标或 promotion gate。

当前 oracle 结论：

1. `U1_liquid_tradable` Top20 oracle 平均月收益约 `0.753628`，相对市场超额约 `0.741908`。
2. `U2_risk_sane` Top20 oracle 平均月收益约 `0.693729`，相对市场超额约 `0.682009`。
3. 现有 price-volume 单因子对 oracle Top-K overlap 很弱，说明当前特征离 oracle imitation 远。
4. overlap 弱不否定选股可行性；主评价继续看 Top-K 收益、Rank IC、分桶 spread、年度和 regime 稳定性。

---

## 2. 数据路线

### 2.1 数据接入原则

任何新数据进入训练主线前必须通过：

| 检查 | 要求 |
| --- | --- |
| PIT | 特征可用日不晚于信号日 |
| coverage | 覆盖率、缺失率、候选池内覆盖率必须报告 |
| alignment | 能与 `(symbol, signal_date)` 或日线 `(symbol, trade_date)` 对齐 |
| leakage | 公告日、报告期、采集日、更新时间必须区分 |

不满足这些条件的数据只能进入 diagnostic。

### 2.2 已有数据状态

| 数据族 | 当前状态 | 下一步 |
| --- | --- | --- |
| 日线 OHLCV | 主线可用 | 修复最新信号日 `next_trade_date` 与可买性 |
| 行业映射 | `real_industry_map` | 作为行业约束和行业 breadth 基础 |
| 基本面 | PIT 覆盖高，主字段可用 | 修复 `ev_ebitda=0` 覆盖，筛选低覆盖字段 |
| fund flow | 质量 gate 已修，历史覆盖短 | 作为近端/缺失感知特征，不单独主导模型 |
| shareholder | PIT 保守，alpha 证据弱 | 低优先级 ablation |
| 股票名称 | M7 当前 `UNKNOWN` | 接入 PIT-safe 名称表或报告层静态名称缓存 |

### 2.3 下一批优先数据

| 优先级 | 数据 | 原因 |
| --- | --- | --- |
| P0 | 主题 / 概念 breadth | A 股月度强者常由主题扩散驱动 |
| P0 | 北向资金 | 行业偏好和持续买入可能有月度延续性 |
| P1 | 融资融券 | 杠杆拥挤度解释强势延续和尾部风险 |
| P1 | 龙虎榜 | 区分持续机构资金与短炒噪声 |
| P2 | 大宗交易 | 识别筹码转移和潜在压力 |
| P2 | 结构化公告事件 | 业绩预告、合同、回购、减持、解禁等 |
| P3 | 分钟线 / Level-2 | 仅当 tabular ranker 通过稳定 gate 后再投入 |

---

## 3. 模型路线

### 3.1 当前模型分工

| 层级 | 模型 | 当前结论 |
| --- | --- | --- |
| baseline | 单因子 / 线性 blend | 多数不稳定，只保留作对照 |
| M5 稳定底座 | ExtraTrees / ElasticNet excess | 当前最值得继续推进 |
| M5 状态补充 | Logistic top20 | strong-up 参与度较好，但全局 Rank IC 弱 |
| M6 watchlist | XGBoost rank NDCG | 仅 `U2 + Top20/Top30` 通过 watchlist gate |
| M6 弃用候选 | pairwise / fixed ensemble / top20 calibrated 单独版 | 未稳定改善 |

### 3.2 当前关键数字

| 口径 | 模型 | Top-K | after-cost 月均超额 | Rank IC | 分桶 spread | 备注 |
| --- | --- | --- | --- | --- | --- | --- |
| `U1` | M5 `+industry+flow+fundamental` ExtraTrees | 20 | `0.011436` | `0.107500` | `0.014534` | M5 最强收益底座 |
| `U1` | M5 `+industry+flow+fundamental` ElasticNet | 20 | `0.010598` | `0.104957` | `0.017400` | 更可解释 |
| `U2` | M5 `+industry+flow+fundamental` ExtraTrees | 20 | `0.009373` | `0.088744` | `0.012681` | U2 对照 |
| `U2` | M6 XGBoost rank NDCG | 20 | `0.012106` | `0.006662` | `0.000674` | watchlist，收益强但 Rank IC 弱 |
| `U2` | M6 XGBoost rank NDCG | 30 | `0.008630` | `0.006662` | `0.000674` | watchlist |
| `U1` | M6 XGBoost rank NDCG | 20 | `0.000214` | `0.009297` | `-0.001108` | fail |

### 3.3 下一版模型原则

1. M5 ExtraTrees / ElasticNet 作为稳定排序底座。
2. M6 rank NDCG 只作为 `U2` watchlist 模块，不单独 promotion。
3. top-bucket classifier 只用于 regime-aware sleeve 或风险提示，不作为全局主模型。
4. 行业集中度不再只停留在选择层治理；hard cap 只作为诊断 baseline，下一步优先把行业拥挤惩罚、行业中性标签和组合优化目标纳入训练/选择闭环。
5. 任何 ensemble 权重只能由历史 walk-forward 训练窗确定，不能用测试月调权。

---

## 4. 评估标准

### 4.1 主评价指标

| 指标 | 说明 |
| --- | --- |
| `topk_excess_after_cost_mean` | 成本后 Top-K 月均超额 |
| `rank_ic_mean` | 月度 Rank IC 均值 |
| `rank_ic_ir` | Rank IC 稳定性 |
| `topk_hit_rate` | Top-K 跑赢市场月份比例 |
| `topk_minus_nextk` | Top-K 相对 next-K 的边界质量 |
| `quantile_top_minus_bottom_mean` | 分桶单调性 |
| `industry_neutral_topk_excess_mean` | 行业中性超额 |
| `year_median_excess` | 年度稳健性 |
| `strong_up_median_excess` | 强市参与度 |
| `strong_down_median_excess` | 弱市防守 |

### 4.2 必须报告的风险指标

| 指标 | 用途 |
| --- | --- |
| `max_industry_share` | 单行业集中度 |
| `industry_count` | Top-K 行业分散度 |
| `turnover_half_l1` | 月度换手 |
| `cost_sensitivity` | 10/30/50 bps 成本压力 |
| `buy_fail_weight` | 涨停、停牌、不可买影响 |
| `size_exposure` | 大小盘暴露 |
| `feature_coverage` | 低覆盖特征风险 |
| `oracle_overlap` | 仅作诊断 |

### 4.3 Promotion Gate

进入 production candidate 前必须全部满足：

| gate | 要求 |
| --- | --- |
| data gate | PIT / coverage / alignment 全部通过 |
| execution gate | 最新信号日可正常生成，不依赖旧月回退 |
| baseline gate | Top-K after-cost 超额不低于 M5 最强稳定底座 |
| rank gate | Rank IC 为正，且不是单一年份支撑 |
| spread gate | Top-K vs next-K 或 top-bottom spread 为正 |
| year gate | OOS 年度中位超额 `>= 0`，2023/2024/2025/2026 无不可解释崩坏 |
| regime gate | strong-up / strong-down 均不显著塌陷 |
| concentration gate | Top20 单行业占比原则上不高于 `0.30`，Top30 不高于 `0.30-0.35` |
| cost gate | 30 bps 和 50 bps 压力下收益不完全消失 |
| governance gate | manifest、配置身份、证据索引、人工确认齐全 |

---

## 5. 当前里程碑摘要

| 阶段 | 状态 | 关键结论 |
| --- | --- | --- |
| M0 目标切换 | 完成 | replacement 冻结，月度选股成为主线 |
| M1 数据质量 | 完成 | fund flow / shareholder 基础质量 gate 已修复 |
| M2 月度数据集 | 完成 | `monthly_selection_features_v1`，64 个信号月，5197 只股票 |
| M3 Oracle | 完成 | U1/U2 oracle 上限强，price-volume 可分性不足 |
| M4 Baseline | 完成 | price-volume-only 有弱信号，不足以推荐 |
| M5 多源扩展 | 完成 | industry breadth + fundamental 带来稳定增量 |
| M6 LTR | 完成 | `U2 + M6_xgboost_rank_ndcg + Top20/30` 仅进入 watchlist |
| M7 推荐报告 | 完成 | 能输出研究名单，但 Top20/30 全为非银金融 |
| M8 集中度与状态治理 | 完成 | hard cap 诊断与 natural industry soft risk-budget 均已生成；3 个自然化候选通过 M8 natural gate，但仍不 promotion |
| M9 数据完整性修复 | 完成 | 报告层名称、T+1 可买字段、低覆盖特征治理、ST 名称过滤和 M9 gate 已落地 |
| M10 成本与执行压力 | 进行中 | 成本敏感性(10/30/50bps)、容量分析、涨停/VWAP压力测试框架已就位 |
| M11 新数据扩展 | 待启动 | 北向、两融、主题、公告事件 |
| M12 promotion package | 待启动 | 仅在 M8-M11 gate 通过后进入 |

---

## 6. M8：行业集中度与 Regime-Aware 治理

**状态**：完成（2026-04-30）。hard cap 诊断与行业约束自然化均已完成；进入 M10 成本与执行压力。  
**目标**：把 M7 watchlist 从“有历史收益”改造成“有可解释暴露、可生产约束、且约束尽量由模型/损失/组合优化自然表达”的候选。

### 6.0 M8 已完成诊断

已新增并运行：

```text
scripts/run_monthly_selection_concentration_regime.py
```

核心产物：

```text
data/results/monthly_selection_m8_concentration_regime_2026-04-30_leaderboard.csv
data/results/monthly_selection_m8_concentration_regime_2026-04-30_gate.csv
data/results/monthly_selection_m8_concentration_regime_2026-04-30_industry_concentration.csv
docs/reports/2026-04/monthly_selection_m8_concentration_regime_2026-04-30.md
```

主要结果：

1. 行业 cap 显著降低集中度：Top20 `indcap3/4/5` 的平均最大行业占比分别约 `0.15/0.19/0.23`，月度集中度 pass rate 为 `1.0`。
2. `M8_regime_aware_fixed_policy + U1 + Top20 + indcap3` after-cost 月均超额约 `0.022369`，Rank IC 约 `0.097405`，Top-K vs next-K 为正。
3. `M5 ElasticNet + U1 + Top20 + indcap4` after-cost 月均超额约 `0.017749`，Rank IC 约 `0.103597`，解释性最好。
4. `U2 + M6_xgboost_rank_ndcg` 的 industry cap 版收益改善了集中度，但 Rank IC 仍弱，只能作为 sleeve / watchlist 证据。
5. M8 hard-cap gate 有研究候选通过，但这只说明行业集中度是当前系统的关键薄弱点；hard cap 后续只作为 stress baseline，不作为主候选生产形态。

### 6.0.1 M8 自然化验收

已新增并运行：

```text
scripts/run_monthly_selection_m8_natural_industry_constraints.py
```

核心产物：

```text
data/results/monthly_selection_m8_natural_industry_constraints_2026-04-30_leaderboard.csv
data/results/monthly_selection_m8_natural_industry_constraints_2026-04-30_label_compare.csv
data/results/monthly_selection_m8_natural_industry_constraints_2026-04-30_penalty_frontier.csv
data/results/monthly_selection_m8_natural_industry_constraints_2026-04-30_score_decomposition.csv
data/results/monthly_selection_m8_natural_industry_constraints_2026-04-30_optimizer_compare.csv
data/results/monthly_selection_m8_natural_industry_constraints_2026-04-30_gate.csv
docs/reports/2026-04/monthly_selection_m8_natural_industry_constraints_2026-04-30.md
```

自然化实验覆盖：

1. 标签层：`market_excess`、`industry_neutral_excess`、`blended_excess_50_50` 三套 walk-forward 标签。
2. 分数层：行业内 alpha、行业 allocation、`within70_industry30`、sector residual 分解。
3. soft penalty：相对候选池行业拥挤的连续分数惩罚 frontier。
4. 组合层：`soft_industry_risk_budget` 连续行业风险预算选择，不使用每行业最多 N 只 hard cap。

通过 M8 natural gate 的候选：

| pool | 候选 | Top-K | after-cost 月均超额 | hard-cap 差值 | Rank IC | Top-K vs next-K | 平均最大行业占比 | 集中度 pass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `U1` | ElasticNet `market_excess` + soft risk `gamma0.20` | 20 | `0.018338` | `-0.004031` | `0.103597` | `0.013406` | `0.103947` | `1.0` |
| `U2` | ElasticNet `industry_neutral_excess` + soft risk `gamma0.20` | 20 | `0.010460` | `-0.004677` | `0.091521` | `0.004255` | `0.101316` | `1.0` |
| `U1` | ElasticNet `blended_excess_50_50` + soft risk `gamma0.20` | 30 | `0.016142` | `-0.003473` | `0.108379` | `0.010827` | `0.100000` | `1.0` |

结论：

```text
M8 完成：至少一个自然化方案在 U1 + Top20/30 上接近 hard-cap baseline，
且集中度、收益、Rank IC、spread、year/regime gate 同时通过。
```

但：

```text
仍不 promotion。M10 真实成本、买入失败、冲击成本和容量压力尚未完成。
```

### 6.1 行业集中度约束

新增月度 Top-K 选择层的行业只数上限，复用已有的行业 cap 思路。

建议网格：

| Top-K | 行业只数上限候选 |
| --- | --- |
| Top20 | `3 / 4 / 5` |
| Top30 | `4 / 5 / 6 / 8` |
| Top50 | `6 / 8 / 10` |

必须输出：

1. constrained / unconstrained 的 leaderboard 对照。
2. 月度 `max_industry_share`、行业数、行业暴露表。
3. Top-K vs next-K 是否被行业约束破坏。
4. strong-up / strong-down 切片变化。
5. 成本后收益变化。

通过条件：

```text
行业集中度显著下降；
after-cost 超额不低于 M5 稳定底座太多；
Top-K vs next-K 和 year/regime slice 不恶化为负。
```

### 6.2 Regime-aware 模型选择

当前观察：

1. M5 ExtraTrees / ElasticNet 的 Rank IC 与分桶更好。
2. M6 `U2 + rank_ndcg` 的 Top20 收益更好，但 Rank IC 很弱。
3. Logistic top-bucket 在部分 strong-up 切片更有参与度，但全局排序证据弱。

下一步做一个只用历史训练窗确定权重的模型选择层：

| regime | 候选组合 |
| --- | --- |
| neutral | M5 ExtraTrees / ElasticNet 为主 |
| strong_down | M5 ElasticNet、低波/质量风险过滤更高权重 |
| strong_up | M6 rank_ndcg 或 top-bucket sleeve 小权重补强 |
| wide breadth | 行业/主题 breadth 权重提高，但必须受行业 cap 约束 |
| narrow breadth | 降低高 beta / 高集中主题暴露 |

禁止事项：

1. 禁止用 realized market state 调当月权重。
2. 禁止按测试月收益回填 regime 规则。
3. 禁止用 M7 当前非银金融结果直接证明行业集中可接受。

### 6.3 M8 已有产物

建议新增脚本：

```text
scripts/run_monthly_selection_concentration_regime.py
```

建议产物：

```text
data/results/monthly_selection_m8_concentration_regime_YYYY-MM-DD_leaderboard.csv
data/results/monthly_selection_m8_concentration_regime_YYYY-MM-DD_monthly_long.csv
data/results/monthly_selection_m8_concentration_regime_YYYY-MM-DD_industry_concentration.csv
data/results/monthly_selection_m8_concentration_regime_YYYY-MM-DD_regime_slice.csv
data/results/monthly_selection_m8_concentration_regime_YYYY-MM-DD_year_slice.csv
data/results/monthly_selection_m8_concentration_regime_YYYY-MM-DD_topk_holdings.csv
data/results/monthly_selection_m8_concentration_regime_YYYY-MM-DD_gate.csv
docs/reports/YYYY-MM/monthly_selection_m8_concentration_regime_YYYY-MM-DD.md
```

### 6.4 行业约束自然化结果

核心判断：

```text
hard cap 提升 = 有价值的诊断信号
hard cap 本身 != 预测模型能力提升
```

本轮目标不是继续调 `indcap3/4/5`，而是让模型分数、训练目标或组合优化自然地降低行业拥挤，并保持可解释性。该目标已由 `soft_industry_risk_budget` 候选完成验收。

#### 6.4.1 标签层：行业中性收益作为主学习目标

新增或强化以下标签对照：

| label | 定义 | 用途 |
| --- | --- | --- |
| `label_forward_1m_industry_neutral_excess` | 个股下月收益减同月同业等权收益 | 让模型优先学习行业内 alpha |
| `label_forward_1m_market_excess` | 个股下月收益减同月市场等权收益 | 保留行业间配置能力 |
| `label_forward_1m_blended_excess` | `market_excess` 与 `industry_neutral_excess` 的训练窗内固定 blend | 平衡行业 beta 与行业内选股 |

实验要求：

1. 同一模型、同一特征、同一 walk-forward 切分下比较三类标签。
2. 同时报告 raw excess、industry-neutral excess、Rank IC、行业暴露和 Top-K vs next-K。
3. 禁止用测试期收益选择 blend 权重；权重只能由训练窗或固定先验确定。

通过条件：

```text
industry-neutral 或 blended 标签下，
Top-K 行业集中度自然下降；
after-cost 超额不显著低于 hard cap baseline；
Rank IC / spread 不恶化。
```

#### 6.4.2 损失层：行业拥挤 soft penalty

在训练或验证选择目标中加入软惩罚，而不是后处理硬卡数量。

候选形式：

```text
objective = rank_loss
          - lambda_concentration * concentration_penalty(predicted_topk)
          - lambda_turnover * turnover_penalty
```

行业惩罚候选：

| penalty | 说明 |
| --- | --- |
| `hhi` | Top-K 行业权重平方和，惩罚行业过度集中 |
| `max_share_softplus` | 超过目标行业占比后平滑惩罚 |
| `relative_to_universe` | 惩罚相对候选池行业权重的主动偏离 |
| `sector_residual_corr` | 惩罚预测分数与行业 dummy 的过高相关 |

实验要求：

1. `lambda` 只在训练窗内选择，采用 walk-forward 固定到测试月。
2. 每个 penalty 必须和 no-penalty、hard-cap baseline 对照。
3. 输出收益-集中度 frontier，而不是只报告单个最佳点。

#### 6.4.3 模型层：显式学习行业内排序与行业间配置

把总分拆成两个可解释部分：

```text
score_total
= score_within_industry_alpha
+ w_industry * score_industry_allocation
+ w_risk * score_risk_control
```

候选实现：

1. **行业内 ranker**：每个信号月先在行业内学习/标准化分数，降低大行业或热门行业支配 Top-K 的概率。
2. **行业 allocation sleeve**：用行业 breadth、资金流、估值、动量预测行业层相对收益，只给总分小权重。
3. **残差化模型**：训练前对收益或特征做行业残差化，模型预测行业解释不了的部分。
4. **score calibration**：按行业对预测分数做训练窗内校准，避免某些行业分数系统性偏高。

必须报告：

1. `score_within_industry_alpha` 单独表现。
2. `score_industry_allocation` 单独表现。
3. 合成分数表现。
4. 每月行业暴露变化是否来自真实行业观点，而不是模型偏置。

#### 6.4.4 组合优化层：从 hard cap 改为风险预算

选择层保留约束，但把“每行业最多 N 只”替换为更连续、可解释的风险预算。

候选优化目标：

```text
maximize sum(w_i * score_i)
       - gamma_industry * industry_active_risk(w)
       - gamma_turnover * turnover(w, w_prev)
       - gamma_size * size_active_risk(w)
```

约束建议：

| 约束 | 形式 |
| --- | --- |
| 单票权重 | `0 <= w_i <= single_name_cap` |
| 行业主动暴露 | 相对候选池或基准行业权重的偏离上限 |
| 行业 HHI | 不超过训练窗确定阈值 |
| 换手 | 半 L1 换手惩罚或上限 |
| 可买性 | `t+1_open` 不可买直接剔除 |

对照关系：

```text
unconstrained top score
hard cap top score
soft industry-risk optimizer
```

M8 完成条件不是 soft optimizer 必须超过 hard cap，而是：

```text
在接近 hard cap 收益的前提下，
行业集中度由连续风险预算自然控制；
参数由训练窗确定；
跨年和 regime slice 不依赖少数月份。
```

#### 6.4.5 验收产物

新增脚本：

```text
scripts/run_monthly_selection_m8_natural_industry_constraints.py
```

新增结果：

```text
data/results/monthly_selection_m8_natural_industry_YYYY-MM-DD_label_compare.csv
data/results/monthly_selection_m8_natural_industry_YYYY-MM-DD_penalty_frontier.csv
data/results/monthly_selection_m8_natural_industry_YYYY-MM-DD_score_decomposition.csv
data/results/monthly_selection_m8_natural_industry_YYYY-MM-DD_optimizer_compare.csv
data/results/monthly_selection_m8_natural_industry_YYYY-MM-DD_gate.csv
docs/reports/YYYY-MM/monthly_selection_m8_natural_industry_constraints_YYYY-MM-DD.md
```

M8 最终完成 gate 结果：

1. 通过。hard cap 仍保留为 stress baseline，但主候选不靠 hard cap 通过 concentration gate。
2. 通过。`U1 + Top20` 与 `U1 + Top30` 均有自然化候选接近 hard-cap baseline。
3. 通过。行业集中度、收益、Rank IC、spread、year/regime slice 同时可解释。
4. 通过。参数选择来自固定先验网格，未按测试月收益回填。
5. 通过。M8 进入完成状态；下一步是 M10 成本与执行压力，不是 promotion。

---

## 7. M9：数据完整性修复

**状态**：完成（2026-04-30）。  
**目标**：让推荐报告能在最新可买信号日稳定生成，并减少低覆盖字段造成的伪信号。

### 7.0 M9 已完成修复

已新增和修复：

```text
scripts/fetch_stock_names.py
scripts/run_monthly_selection_report.py
```

核心变更：

1. 报告信号日默认选择“最新可执行信号日”：必须存在 `candidate_pool_pass` 且 `next_trade_date` 非空；显式请求无 `next_trade_date` 的日期会报错，不再硬出不可买名单。
2. 推荐表合并回真实 `is_buyable_tplus1_open` 与 `next_trade_date`，修复报告层把可买标的误写成 `no_next_trade_date` 的问题。
3. 新增股票名称缓存 `data/cache/a_share_stock_names.csv`，报告默认读取；M9 实跑名称覆盖 0 个 `UNKNOWN`。
4. 名称可用后，报告目标月剔除 `ST/*ST` 名称标的；本次目标月剔除 129 只，最终推荐 ST 名称数为 0。
5. 新增 M9 特征覆盖策略：`candidate_pool_pass` 覆盖率低于 `30%` 的字段不作为核心特征，只保留缺失标记；`feature_fundamental_ev_ebitda_z` 由 0 覆盖核心字段降级为 `is_missing_feature_fundamental_ev_ebitda`。
6. 报告新增 `feature_policy.csv` 与 `m9_integrity.csv`，manifest 写入 `active_feature_cols` 与 `m9_integrity_pass`。

实跑产物：

```text
data/results/monthly_selection_m9_data_integrity_report_2026-04-30_recommendations.csv
data/results/monthly_selection_m9_data_integrity_report_2026-04-30_feature_policy.csv
data/results/monthly_selection_m9_data_integrity_report_2026-04-30_m9_integrity.csv
data/results/monthly_selection_m9_data_integrity_report_2026-04-30_manifest.json
docs/monthly_selection_m9_data_integrity_report_2026-04-30.md
```

实跑结论：

1. 当前本地 dataset 最新日 `2026-04-29` 仍缺少下一交易日行情，因此不能作为“可执行信号日”；报告自动选择最新可执行信号日 `2026-03-31`，下一交易日 `2026-04-01`。
2. M9 gate 全部通过：`target_candidate_pool_pass_rows=4196`，`target_next_trade_date_present=4196`，`recommendation_buyable=0`，`recommendation_names_readable=0`，`recommendation_excludes_st_names=0`，`zero_coverage_core_features=0`，`low_coverage_core_features_lt_30pct=0`。
3. 这只解决数据完整性与报告可信度；M8 后续已完成自然化验收，但 M10 未完成，因此仍保持 `research_only_not_promoted` 状态。

### 7.1 必修项

1. 已修复 `next_trade_date` 口径：无下一交易日行情的日期不允许作为可执行报告信号日。
2. 已保留数据补齐边界：若需使用 `2026-04-29`，必须先补齐 `2026-04-30` 日线和可买性；在补齐前自动退到最新可执行信号日。
3. 已接入股票名称缓存，避免报告层 `name=UNKNOWN`。
4. 已移除 `feature_fundamental_ev_ebitda_z` 作为 0 覆盖核心字段的影响，只保留缺失标记。
5. 已对覆盖率低于 `30%` 的特征执行 missing-marker-only 策略。

### 7.2 验收

```text
最新信号日可生成 Top20/Top30；
feature_coverage 无零覆盖主字段；
report 中 name 字段可读；
manifest 明确数据截止日、信号日、下一交易日。
```

当前验收结果：

```text
通过。当前可执行信号日为 2026-03-31，下一交易日为 2026-04-01。
若要把 2026-04-29 作为报告信号日，必须先补齐 2026-04-30 的全市场日线。
```

---

## 8. M10：成本与执行压力测试

**状态**：进行中（2026-05-03）。  
**进度**：成本敏感性网格（10/30/50 bps）已在 `run_monthly_benchmark_suite.py` 实现；容量分析（日均成交额）框架已就位；涨停买入失败（idle/redistribute）与 VWAP 冲击对比框架已就位。披露日历填充（P2-3）与 next_trade_date 边界校验（P2-4）已完成。  
**目标**：验证高换手月度 Top-K 在真实成本下是否仍有意义。  
**问题追踪**：详见 `docs/plan-05-03.md` P1-1。

### 8.1 成本网格

| 成本项 | 测试 |
| --- | --- |
| 固定成本 | `10 / 30 / 50 bps` |
| 流动性冲击 | 按 `amount_20d` 分层，小成交额更高 |
| 涨停买入失败 | `idle / redistribute` 两口径 |
| 停牌/无量 | 次日开盘不可买过滤 |
| 单票上限 | 与行业 cap 联动 |
| 换手上限 | 半 L1 月度换手约束 |

### 8.2 通过条件

1. 30 bps 后不低于市场等权。
2. 50 bps 后收益不完全消失，或明确限定可用资金规模。
3. buy-fail 权重可解释，不由涨停不可买贡献主要历史收益。
4. 高换手月份有单独披露。

---

## 9. M11：新数据扩展

**状态**：待启动。  
**目标**：补齐能解释月度 top bucket 的增量信息，而不是继续压榨价量因子。

优先顺序：

1. 主题 / 概念 breadth。
2. 北向资金持股和净买入。
3. 融资融券余额与交易。
4. 龙虎榜机构和席位结构。
5. 解禁、减持、回购、业绩预告等结构化事件。
6. 大宗交易。

每个数据族必须单独跑：

```text
data quality
-> coverage
-> M5-style incremental delta
-> regime/year slice
-> feature importance
-> gate
```

---

## 10. M12：Promotion Package

**状态**：待启动。  
**触发条件**：M8-M10 通过，且 M11 没有引入新泄漏或不稳定暴露。

Promotion package 必须包含：

1. 固定配置文件。
2. full backtest / walk-forward 证据。
3. 最新推荐报告。
4. 成本和执行压力测试。
5. 行业、规模、换手、买入失败、数据覆盖诊断。
6. 年度和 regime 切片。
7. promoted registry 变更说明。
8. 人工确认记录。

只有 promotion package 通过后，才允许更新：

```text
configs/promoted/promoted_registry.json
```

仍不允许直接把研究候选写入：

```text
config.yaml.example
```

---

## 11. 停止清单

以下方向冻结，除非出现明确新证据：

1. 不继续 P1/G0 近邻标签微调。
2. 不继续 dual sleeve 权重网格。
3. 不继续 R2B v1 / R2B v2 replacement 参数网格。
4. 不启动 R3 replacement classifier / pairwise ranker。
5. 不把 `old-new pair` 作为新主线训练样本。
6. 不把 oracle top-K overlap 当作主评价指标。
7. 不用随机 CV 证明时序选股模型有效。
8. 不使用未通过质量检查或非 PIT 的数据。
9. 不把 shareholder 单因子作为主线救援。
10. 不把深度学习作为当前第一主线。
11. 不把未 promotion 的研究配置写入生产月度选股主线。
12. 不把当前 M7 非银金融集中名单直接解释为可交易组合。

---

## 12. 保留清单

这些能力继续保留并迁移到月度选股研究：

1. R0 评估/执行契约。
2. `tplus1_open` 交易口径。
3. open-to-open benchmark。
4. buy-fail diagnostic。
5. monthly / state / breadth / year slice。
6. top-k boundary diagnostic。
7. industry exposure diagnostic。
8. 真实申万行业映射。
9. fund flow / shareholder / fundamental 数据链路。
10. `config.yaml.backtest` 作为 canonical 研究入口。
11. `configs/promoted/promoted_registry.json` 作为生产边界。
12. 历史 R2B/R5 报告作为目标切换证据。

---

## 13. 产出规范

### 13.1 配置身份

| 字段 | 要求 |
| --- | --- |
| `research_topic` | 必填 |
| `research_config_id` | 必填 |
| `output_stem` | 必填 |
| `result_type` | 必填 |
| `config_source` | 必填 |

### 13.2 数据口径

| 字段 | 要求 |
| --- | --- |
| `dataset_version` | 必填 |
| `candidate_pool_version` | 必填 |
| `candidate_pool_rule` | 必填 |
| `candidate_pool_width_by_month` | 必填 |
| `feature_spec` | 必填 |
| `label_spec` | 必填 |
| `pit_policy` | 必填 |
| `universe_filter` | 必填 |
| `industry_map_source` | 必填 |
| `data_quality_report` | 使用新数据时必填 |

### 13.3 训练口径

| 字段 | 要求 |
| --- | --- |
| `model_type` | 必填 |
| `train_window` | 必填 |
| `validation_window` | 必填 |
| `test_window` | 必填 |
| `cv_policy` | 必须是时间序列切分 |
| `hyperparameter_policy` | 必填 |
| `random_seed` | 必填 |

### 13.4 评估口径

| 字段 | 要求 |
| --- | --- |
| `rebalance_rule` | 必填 |
| `execution_mode` | 必填 |
| `benchmark_return_mode` | 必填 |
| `top_k` | 必填 |
| `cost_assumption` | 必填 |
| `buyability_policy` | 必填 |
| `concentration_policy` | M8 起必填 |

### 13.5 结果文件

每轮实验至少输出：

1. `summary.json`
2. `leaderboard.csv`
3. `monthly_long.csv`
4. `rank_ic.csv`
5. `quantile_spread.csv`
6. `topk_holdings.csv`
7. `industry_exposure.csv`
8. `industry_concentration.csv`（M8 起）
9. `candidate_pool_width.csv`
10. `candidate_pool_reject_reason.csv`
11. `feature_coverage.csv`
12. `feature_importance.csv` 或等价解释文件
13. `manifest.json`
14. 一页结论文档

结论文档必须回答：

1. 本轮新增了什么数据、模型或约束。
2. 数据质量是否通过。
3. 使用哪个 candidate pool，过滤规则是否只做准入。
4. 相对 M5 稳定底座是否改善。
5. Top-K 是否稳定跑赢市场。
6. 分桶是否单调。
7. 哪些年份 / regime 失败。
8. 行业集中度是否可接受。
9. 成本压力下是否仍有收益。
10. 是否允许进入下一阶段。

---

## 14. 证据索引

### 当前主线证据

- `docs/reports/2026-04/monthly_selection_dataset_2026-04-28.md`
- `docs/reports/2026-04/monthly_selection_oracle_2026-04-28.md`
- `docs/reports/2026-04/monthly_selection_baselines_2026-04-29.md`
- `docs/reports/2026-04/monthly_selection_m5_multisource_full_2026-04-29.md`
- `docs/reports/2026-04/monthly_selection_m6_ltr_2026-04-29.md`
- `docs/reports/2026-04/monthly_selection_m7_recommendation_report_2026-04-29.md`
- `docs/reports/2026-04/monthly_selection_m8_concentration_regime_2026-04-30.md`
- `docs/monthly_selection_m9_data_integrity_report_2026-04-30.md`
- `data/cache/monthly_selection_features.parquet`
- `data/cache/a_share_stock_names.csv`
- `data/results/monthly_selection_m5_multisource_full_2026-04-29_leaderboard.csv`
- `data/results/monthly_selection_m5_multisource_full_2026-04-29_incremental_delta.csv`
- `data/results/monthly_selection_m6_ltr_2026-04-29_leaderboard.csv`
- `data/results/monthly_selection_m6_ltr_2026-04-29_vs_m5_gate.csv`
- `data/results/monthly_selection_m7_recommendation_report_2026-04-29_recommendations.csv`
- `data/results/monthly_selection_m8_concentration_regime_2026-04-30_leaderboard.csv`
- `data/results/monthly_selection_m8_concentration_regime_2026-04-30_gate.csv`
- `data/results/monthly_selection_m8_concentration_regime_2026-04-30_industry_concentration.csv`
- `data/results/monthly_selection_m9_data_integrity_report_2026-04-30_m9_integrity.csv`
- `data/results/monthly_selection_m9_data_integrity_report_2026-04-30_feature_policy.csv`
- `data/results/monthly_selection_m9_data_integrity_report_2026-04-30_manifest.json`

### 数据质量证据

- `docs/newdata_quality_current_families_fund_flow-shareholder_flow_a_share_fund_flow_holder_a_share_shareholder_daily_a_share_daily_lag_30_width_100.md`
- `docs/reports/2026-04/fundamental_pit_coverage_2026-04-28.md`
- `scripts/run_newdata_quality_checks.py`
- `src/features/fund_flow_factors.py`
- `src/features/shareholder_factors.py`
- `src/features/fundamental_factors.py`

### 目标切换前证据

- `docs/reports/2026-04/r1f_fixed_baseline_rerun_2026-04-28.md`
- `docs/reports/2026-04/p2_regime_aware_dual_sleeve_v1_2026-04-28.md`
- `docs/reports/2026-04/r2b_tradable_upside_replacement_v1_2026-04-28.md`
- `docs/reports/2026-04/r2b_oracle_replacement_attribution_2026-04-28.md`
- `docs/reports/2026-04/r2b_edge_gated_replacement_v2_2026-04-28.md`
- `docs/reports/2026-04/r2b_v2_weight_audit_2026-04-28.md`
- `docs/reports/2026-04/r5_config_governance_2026-04-28.md`

### 核心代码入口

- `config.yaml.backtest`
- `scripts/run_monthly_selection_dataset.py`
- `scripts/run_monthly_selection_oracle.py`
- `scripts/run_monthly_selection_baselines.py`
- `scripts/run_monthly_selection_multisource.py`
- `scripts/run_monthly_selection_ltr.py`
- `scripts/run_monthly_selection_report.py`
- `scripts/fetch_stock_names.py`
- `scripts/run_monthly_selection_concentration_regime.py`
- `scripts/run_monthly_selection_m8_natural_industry_constraints.py`
- `scripts/run_backtest_eval.py`
- `src/market/tradability.py`
- `src/features/tensor_base_factors.py`
- `src/features/intraday_proxy_factors.py`
- `src/features/fund_flow_factors.py`
- `src/features/shareholder_factors.py`
- `src/features/fundamental_factors.py`
- `tests/test_monthly_selection_dataset.py`
- `tests/test_monthly_selection_oracle.py`
- `tests/test_monthly_selection_baselines.py`
- `tests/test_monthly_selection_multisource.py`
- `tests/test_monthly_selection_ltr.py`
- `tests/test_monthly_selection_report.py`
- `tests/test_monthly_selection_concentration_regime.py`
- `tests/test_monthly_selection_m8_natural_industry_constraints.py`

---

## 15. 当前一句话路线

**当前项目已经完成 M0-M9，其中 M8 已从 hard-cap 诊断推进到 soft industry risk-budget 自然化验收；但 M10 成本执行压力、M11 新数据稳定性和 M12 promotion package 尚未完成，因此当前仍不是交易策略。下一步进入 M10，对 M8 natural 候选做 30/50 bps、买入失败、冲击成本、容量和换手压力测试，最终只有完整通过 M10-M12 gate 的候选才允许进入 promoted registry。**
