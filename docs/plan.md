# 量化月度选股主计划

**文档角色**：当前唯一主计划（canonical）  
**更新时间**：`2026-04-28`  
**当前目标**：从“量化持仓 / replacement 换仓”切换为“量化月度选股研究”  
**研究终点**：每月输出可解释、可回测、PIT-safe 的 Top-K 股票推荐名单  
**生产状态**：当前无任何研究候选进入生产；`configs/promoted/promoted_registry.json` 继续为空  
**归档入口**：`docs/plan-04-20.md` 仅保留 `2026-04-20` 当日执行记录，不再承担主计划职责

---

## 0. 总判断

当前项目正式切换目标：

```text
旧目标：
    构建可自动持仓的量化组合，并在已有 S2 defensive 持仓上做 old-new replacement

新目标：
    构建量化月度选股系统，每月从可交易股票池中筛出未来一个月更可能跑赢市场的 Top-K 推荐名单
```

这个切换改变了核心建模对象。

旧主线的建模对象是：

```text
edge(old_stock, new_stock, date)
= future_return(new_stock) - future_return(old_stock) - cost - risk_penalty
```

新主线的建模对象是：

```text
score(stock, month_end)
-> rank all tradable stocks
-> select monthly Top-K
```

因此，后续不再把 `old-new pair`、replacement slot、hold/replace 决策作为主问题。换手与成本仍然需要在评估报告中披露，但它们不再是训练目标的第一约束。

一句话路线：

```text
多源 PIT 数据
-> 月度截面特征表
-> open-to-open 月度标签
-> oracle top-bucket 上限诊断
-> baseline ranker
-> 多源特征扩展
-> learning-to-rank / top-bucket classifier
-> 月度 Top-K 推荐名单与研究报告
```

---

## 1. 为什么转向

### 1.1 已证明的资产

当前项目不是从零开始。已有基础设施继续保留：

1. `tplus1_open` 执行口径。
2. open-to-open primary benchmark。
3. buy-fail diagnostic。
4. monthly / state / breadth / year slice。
5. top-k boundary / switch quality 诊断框架。
6. 真实申万行业映射。
7. fund flow / shareholder 数据链路雏形。
8. promoted registry 与生产配置隔离规则。

### 1.2 已证伪的方向

截至 `2026-04-28`，以下方向已经不再作为主线：

1. P1/G0 标签近邻微调。
2. pure upside Top-20。
3. dual sleeve 扩权。
4. R2B v1 replacement。
5. R2B v2 手写 `pair_edge_score`。
6. R3 replacement classifier / pairwise ranker。

主要证据：

- `docs/r1f_fixed_baseline_rerun_2026-04-28.md`
- `docs/p2_regime_aware_dual_sleeve_v1_2026-04-28.md`
- `docs/r2b_tradable_upside_replacement_v1_2026-04-28.md`
- `docs/r2b_oracle_replacement_attribution_2026-04-28.md`
- `docs/r2b_edge_gated_replacement_v2_2026-04-28.md`
- `docs/r2b_v2_weight_audit_2026-04-28.md`
- `docs/r5_config_governance_2026-04-28.md`

### 1.3 对旧证据的新解释

R2B oracle replacement attribution 的意义不是继续逼 replacement，而是说明：

```text
候选池里存在事后强者。
现有手写规则无法稳定提前识别。
下一步应回到更干净的月度截面排序问题。
```

也就是说，oracle 证明的是“候选空间有信息”，不是“replacement 框架本身是最合适的研究目标”。

---

## 2. 新研究定义

### 2.1 任务定义

每个月最后一个可交易信号日 `t`：

1. 使用 `t` 日收盘后已知、且 PIT-safe 的数据构造特征。
2. 在 `t+1` 开盘可买的股票池中打分排序。
3. 输出 Top-K 月度推荐名单。
4. 标签使用 `t+1` 开盘到下一次月度换仓开盘的收益。

形式化：

```text
x[i, t] = stock i 在月末 t 的可观测特征
y[i, t] = stock i 从 t+1 open 到 next_rebalance open 的 forward return
s[i, t] = f(x[i, t])
select Top-K by s[i, t]
```

### 2.2 默认研究口径

| 字段 | 当前默认 |
| --- | --- |
| `rebalance_rule` | `M` |
| `signal_date` | 每月最后一个交易日 |
| `execution_mode` | `tplus1_open` |
| `label_return_mode` | `open_to_open` |
| `benchmark` | `market_ew_open_to_open` |
| `portfolio_method` | 研究版默认等权，仅用于评价 Top-K |
| `top_k` | `20 / 30 / 50` 并行诊断，默认主报告 `Top20` |
| `raw_universe` | A 股原始股票池 |
| `candidate_pool` | 月度可交易研究池，只做准入过滤，不做 alpha 判断 |
| `industry_map` | 真实申万行业映射 |

### 2.3 月度可交易研究池

新主线仍保留候选池，但候选池的定义从“换仓候选池”改为“月度可交易研究池”。

候选池路径：

```text
Raw A-share universe
-> Eligible monthly candidate pool
-> Model scoring / ranking
-> Top-K monthly selection
```

候选池的职责：

1. 排除不可交易标的。
2. 排除数据无效标的。
3. 排除极端风险标的。
4. 保证训练样本和标签可解释。

候选池不负责判断股票是否有 alpha。真正的 alpha 排序必须交给模型完成。

明确禁止把候选池写成半个手写策略：

```text
错误：
    candidate_pool = 我觉得会涨的股票集合

正确：
    candidate_pool = tradable + data-valid + risk-sane 的宽研究池
```

候选池过滤分三层：

| 层级 | 处理 | 示例 |
| --- | --- | --- |
| 交易可行性 | 默认硬过滤 | ST / 退市整理 / 长期停牌 / 历史不足 / 流动性不足 / 次日不可买 |
| 极端风险 | 极端情况硬过滤，轻度情况保留为 risk flag | 连续一字板、极端波动、异常换手、严重过热、重大退市或监管风险 |
| alpha 信息 | 不在候选池阶段硬过滤，交给模型排序 | 动量、行业强度、资金流、基本面质量、主题热度、机构买入 |

候选池版本必须显式记录，默认至少并行诊断：

| pool | 定义 | 用途 |
| --- | --- | --- |
| `U0_all_tradable` | 只做基础可交易过滤 | 判断全可交易空间上限 |
| `U1_liquid_tradable` | 加流动性、历史长度、基础数据质量门槛 | 默认训练候选池 |
| `U2_risk_sane` | 在 `U1` 上排除极端一字板、极端波动、严重过热等 | 默认报告对照 |
| `U3_quality_optional` | 在 `U2` 上加入基本面质量门槛 | 仅作 ablation，不作为默认主池 |

默认规则：

```text
主训练池：U1_liquid_tradable
主报告对照：U1_liquid_tradable + U2_risk_sane
质量过滤池：U3_quality_optional 仅作诊断
```

### 2.4 标签族

不直接把“未来收益最高的 K 只股票”作为唯一 imitation target。标签分层如下：

| label | 定义 | 用途 |
| --- | --- | --- |
| `forward_1m_o2o_return` | 下月 open-to-open 原始收益 | 回归 / 诊断 |
| `forward_1m_excess_vs_market` | 原始收益减同月市场等权收益 | 主排序标签之一 |
| `forward_1m_industry_neutral_excess` | 原始收益减同月行业等权收益 | 控制行业 beta |
| `future_return_quantile` | 同月截面收益分桶 | 稳健分类 |
| `future_top_20pct` | 是否进入未来收益前 20% | top-bucket classifier |
| `future_top_10pct` | 是否进入未来收益前 10% | 高置信候选 |
| `future_bottom_20pct` | 是否进入未来收益后 20% | 风险过滤 |

主线优先优化：

```text
market-relative / industry-neutral top-bucket ranking
```

而不是硬预测单只股票精确涨幅。

### 2.5 Oracle 定义

新 oracle 不再是 replacement oracle，而是月度截面 oracle：

```text
Oracle Top-K:
    每个月在指定 candidate pool 中，未来一个月 open-to-open 收益最高的 K 只股票

Oracle Top-Bucket:
    每个月在指定 candidate pool 中，未来收益前 10% / 20% 股票集合
```

oracle 的用途：

1. 判断候选空间上限。
2. 判断现有特征对未来 top bucket 是否有分辨力。
3. 构造分桶、排序和 overlap 诊断。
4. 对比不同 candidate pool 是否过度过滤机会。

oracle overlap 不是主评价指标。模型不需要命中每个月最强的极端股票，只要稳定提高 Top-K 收益分布即可。

每次 oracle / baseline / model 结果都必须标注：

```text
candidate_pool_version
candidate_pool_rule
candidate_pool_width_by_month
```

---

## 3. 数据扩展主线

新目标的第一优先级是扩展 PIT-safe 观测变量，而不是继续调 replacement 规则。

### 3.1 数据接入原则

任何新数据进入模型前必须通过四类检查：

| 检查 | 要求 |
| --- | --- |
| PIT | 特征可用日必须不晚于信号日 |
| coverage | 覆盖率、截面宽度、缺失率必须报告 |
| alignment | 必须能与日线 `(symbol, trade_date)` 或月度信号日对齐 |
| leakage | 公告日、报告期、采集日、更新时间必须区分 |

不满足这些条件的数据只能进入 diagnostic，不能进入训练主线。

### 3.2 Tier 0：已有数据修复与固化

优先把仓库已有数据链路变成可训练资产。

| 数据族 | 当前状态 | 下一步 |
| --- | --- | --- |
| 日线 OHLCV | 主线可用 | 补齐到最新研究日期，确保与资金流同窗 |
| 真实行业映射 | 已完成 | 固化为所有行业特征的基础 |
| 基本面 | 有入口 | 做 PIT 覆盖与月度快照表 |
| fund flow | 有表和因子，但质量检查未通过 | 先修日线/资金流日期对齐，再恢复研究预算 |
| shareholder | PIT 接入较保守，alpha 证据弱 | 低优先级保留，不作为第一主线 |

相关代码与文档：

- `src/features/fund_flow_factors.py`
- `src/features/shareholder_factors.py`
- `src/features/fundamental_factors.py`
- `docs/newdata_quality_current_families_fund_flow-shareholder_flow_a_share_fund_flow_holder_a_share_shareholder_daily_a_share_daily_lag_30_width_100.md`

### 3.3 Tier 1：优先新增数据

这些数据最可能解释“月度 top bucket”：

| 数据 | 可能特征 | 研究假设 |
| --- | --- | --- |
| 主力 / 超大单 / 大单资金流 | 净流入均值、连续流入、资金流反转、大小单分歧 | 资金确认比单纯价格动量更接近可交易强势 |
| 北向资金 | 持股变化、净买入、连续买入、行业内排名 | 外资趋势与行业偏好可能有月度延续性 |
| 融资融券 | 融资余额变化、买入偿还比、拥挤度 | 杠杆资金可解释强势延续和尾部风险 |
| 龙虎榜 | 机构净买入、席位集中度、游资一日游标记 | 区分持续资金与短炒噪声 |
| 大宗交易 | 折价率、成交占流通比、买卖方类型 | 识别筹码转移和潜在压力 |
| 真实行业 breadth | 行业上涨家数、创新高比例、成交额占比、行业强度持续性 | 月度选股需要行业/主题扩散信息 |
| 主题 / 概念暴露 | 主题热度、主题内排名、主题 breadth | A 股月度强者常由主题驱动 |

### 3.4 Tier 2：公告、新闻与文本数据

文本数据不直接进入主模型，先做结构化事件层。

| 数据 | 结构化方式 |
| --- | --- |
| 业绩预告 / 快报 / 财报 | 预告类型、利润增速区间、超预期标记 |
| 重大合同 / 中标 / 订单 | 金额占营收比例、行业、持续期 |
| 并购重组 / 定增 / 回购 | 事件类型、进度、金额、摊薄或增厚方向 |
| 减持 / 解禁 / 质押 | 压力强度、时间窗口、比例 |
| 新闻 / 舆情 | 情绪、热度变化、事件类别 |
| 研报 | 评级变化、盈利预测修正、覆盖数量变化 |

文本 embedding 可以作为第二阶段特征，但第一阶段必须先保证事件分类与时间戳 PIT 正确。

### 3.5 Tier 3：分钟线 / Level-2 / 微观结构

这类数据最适合深度学习，但成本和工程复杂度较高。

候选特征：

1. 分钟 VWAP 偏离。
2. 尾盘买入强度。
3. 盘口不平衡。
4. 冲击成本。
5. 撤单率。
6. 分时成交额集中度。
7. 集合竞价量价特征。

只有当 Tier 0 / Tier 1 的 tabular ranker 已经证明有效后，才把这类数据作为深度模型输入。

---

## 4. 月度特征表

### 4.1 目标产物

新增 canonical 月度研究表：

```text
data/cache/monthly_selection_features.parquet
```

每行：

```text
(signal_date, symbol)
```

必须字段：

| 字段 | 说明 |
| --- | --- |
| `signal_date` | 月末信号日 |
| `symbol` | 6 位股票代码 |
| `candidate_pool_version` | 候选池版本，例如 `U1_liquid_tradable` |
| `candidate_pool_pass` | 是否进入该候选池 |
| `candidate_pool_reject_reason` | 未进入候选池的原因，训练池内可为空 |
| `is_buyable_tplus1_open` | 次日开盘可买标记 |
| `industry_level1` | 申万一级行业 |
| `industry_level2` | 申万二级行业 |
| `market_cap` / `log_market_cap` | 规模控制 |
| `amount_20d` | 流动性 |
| `risk_flags` | 极端风险与轻度风险标记 |
| `feature_*` | PIT-safe 特征 |
| `label_*` | 训练标签，仅训练/评估阶段可见 |

### 4.2 特征处理

默认处理：

1. 按 `signal_date` 截面 winsorize。
2. 按 `signal_date` 截面 z-score。
3. 可选行业中性化。
4. 可选市值 + 行业回归残差化。
5. 缺失值按截面中位数填充，并保留 `is_missing_*` 标记。

禁止：

1. 用未来月份数据填充历史缺失。
2. 用全样本分位数做训练特征。
3. 把公告期 `end_date` 当成可用日。
4. 把数据抓取日混同于公告可用日。

---

## 5. 模型路线

### 5.1 Baseline 必须先跑

任何 ML 模型上线前，必须先跑可解释 baseline：

| baseline | 说明 |
| --- | --- |
| `B0_market_ew` | 全市场等权基准 |
| `B1_current_s2` | 当前 `vol_to_turnover` 防守基线，仅作对照 |
| `B2_momentum` | 月度动量基线 |
| `B3_low_vol_quality` | 低波 + 基本面质量 |
| `B4_industry_strength` | 行业强度 + 行业内排名 |
| `B5_fund_flow_strength` | 资金流强度，前提是质量检查通过 |

baseline 的意义是判断新数据和模型是否真的带来增量，而不是被复杂模型掩盖。

### 5.2 第一阶段模型

优先 tabular、稳健、可解释：

| 模型 | 目标 | 优先级 |
| --- | --- | --- |
| ElasticNet / Logistic | 可解释线性基线 | 必跑 |
| RandomForest / ExtraTrees | 非线性 sanity check | 可选 |
| XGBoost / LightGBM regression | 预测相对收益 | 第一优先 |
| XGBoost / LightGBM classifier | 预测 future top bucket | 第一优先 |
| LambdaRank / pairwise ranking | 直接优化月度截面排序 | 主线候选 |
| CatBoost | 行业/主题类别特征丰富时使用 | 可选 |

### 5.3 第二阶段模型

只有当第一阶段模型 OOS 稳定后，才进入：

| 模型 | 前提 |
| --- | --- |
| Tabular ensemble | 至少两个独立特征族有效 |
| Regime-aware ranker | 月度样本足够，且 state 切片有稳定差异 |
| Text embedding + tabular fusion | 公告/新闻事件 PIT 检查通过 |
| Graph model | 行业/主题/供应链图谱可用 |
| TCN / Transformer | 分钟线或长序列数据可用 |

深度学习不作为当前第一主线。原因：

1. 月度样本少。
2. A 股非平稳强。
3. 直接拟合 oracle top-K 容易学噪声。
4. 没有高维序列/文本/图数据时，深度模型的边际优势不明确。

---

## 6. 评估标准

### 6.1 主评价指标

模型不以 oracle overlap 为主指标。主指标是：

| 指标 | 说明 |
| --- | --- |
| `rank_ic_mean` | 月度 Rank IC 均值 |
| `rank_ic_ir` | Rank IC 稳定性 |
| `topk_excess_mean` | Top-K 月均超额 |
| `topk_excess_annualized` | Top-K 年化超额 |
| `topk_hit_rate` | Top-K 跑赢市场月份比例 |
| `topk_minus_nextk` | Top-K 相对 next-K 的 spread |
| `top_quantile_minus_bottom_quantile` | 分层单调性 |
| `industry_neutral_topk_excess` | 行业中性超额 |
| `year_median_excess` | 年度稳健性 |
| `strong_up_capture` | 强市参与度 |
| `strong_down_drawdown` | 弱市防守 |

### 6.2 必须报告但不作为第一训练目标

| 指标 | 用途 |
| --- | --- |
| `turnover` | 月度推荐名单稳定性 |
| `cost_sensitivity` | 真实执行可行性 |
| `buy_fail_weight` | 涨停/停牌影响 |
| `industry_exposure` | 行业集中度 |
| `size_exposure` | 小盘/大盘风格 |
| `oracle_overlap` | 仅作诊断，不作主 gate |

### 6.3 Gate

研究候选进入下一阶段，至少满足：

| gate | 要求 |
| --- | --- |
| data gate | 所有特征族 PIT / coverage / alignment 通过 |
| candidate pool gate | 候选池只做可交易、数据有效、极端风险过滤，且规则与月度宽度可追溯 |
| baseline gate | Top-K 超额不低于最强简单 baseline |
| rank gate | Rank IC 均值为正，且不是单一年份支撑 |
| spread gate | Top-K 相对 next-K 或 bottom bucket 为正 |
| slice gate | strong-up 不显著落后，strong-down 不灾难性恶化 |
| cost gate | 加入合理成本后不完全消失 |
| stability gate | 2021 / 2025 / 2026 不出现不可解释的大幅崩坏 |

进入“可作为月度推荐候选”前，额外要求：

1. OOS 年度中位超额 `>= 0`。
2. rolling 12M 超额中位数 `>= 0`。
3. Top-K hit rate 高于市场随机基线。
4. 行业/市值暴露可解释。
5. 推荐名单换手有披露，且成本敏感性不过度脆弱。

---

## 7. 执行计划

### M0：目标切换与文档治理

**状态**：当前进行中。

任务：

1. 重构 `docs/plan.md`。
2. 明确 R2B/R3 replacement 冻结为历史方向。
3. 明确新主线为月度选股。
4. 保留 promoted registry 生产边界。

验收：

```text
docs/plan.md 不再把 replacement 作为下一阶段主线。
```

### M1：数据质量修复

目标：

```text
先让已有多源数据可用于月度特征表。
```

任务：

1. 补齐日线到资金流最新日期。
2. 复跑 `scripts/run_newdata_quality_checks.py`。
3. 修复 fund flow 与 daily 的日期 / universe 对齐问题。
4. 输出新的 newdata quality 报告。
5. 基本面与 shareholder 做 PIT 月度覆盖摘要。

通过条件：

| 数据族 | 要求 |
| --- | --- |
| fund flow | `ok=True`，或剩余缺口有可解释白名单 |
| shareholder | notice_date / fallback lag 规则清楚 |
| fundamental | PIT 可用日与覆盖率可追溯 |
| industry | `real_industry_map` |

### M2：月度特征与标签表

目标：

```text
构建 monthly selection canonical dataset。
```

建议新增脚本：

```text
scripts/run_monthly_selection_dataset.py
```

产物：

```text
data/cache/monthly_selection_features.parquet
data/results/monthly_selection_dataset_YYYY-MM-DD_quality.csv
docs/monthly_selection_dataset_YYYY-MM-DD.md
```

必须包含：

1. 月末信号日。
2. 次日开盘可买标记。
3. open-to-open 月度标签。
4. market-relative 标签。
5. industry-neutral 标签。
6. feature coverage 表。
7. label 分布表。
8. raw universe 宽度表。
9. candidate pool 版本、规则与月度宽度表。
10. candidate pool reject reason 分布。

### M3：Oracle top-bucket 诊断

目标：

```text
判断月度选股任务的上限与现有特征可分性。
```

建议新增脚本：

```text
scripts/run_monthly_selection_oracle.py
```

输出：

| 产物 | 说明 |
| --- | --- |
| `oracle_topk_return_by_month` | 每月 oracle Top-K 理论收益 |
| `oracle_topk_by_candidate_pool` | 不同候选池内的 oracle 上限对比 |
| `feature_bucket_monotonicity` | 各特征分桶与未来收益关系 |
| `baseline_overlap` | 简单 baseline 与 oracle top bucket overlap |
| `regime_oracle_capacity` | 不同市场状态下 oracle 上限 |
| `industry_oracle_distribution` | oracle Top-K 行业分布 |
| `candidate_pool_width` | 不同候选池的月度宽度和过滤强度 |

判定：

| 结果 | 动作 |
| --- | --- |
| 默认候选池 oracle 上限很弱 | 回到 candidate pool / label 定义 |
| 宽池 oracle 强、窄池 oracle 弱 | 候选池过滤过重，回退过滤规则 |
| oracle 有上限但现有特征不可分 | 优先扩数据 |
| oracle 有上限且部分特征单调 | 进入 M4 baseline ranker |

### M4：Baseline ranker

目标：

```text
用简单模型建立可复现月度选股基线。
```

建议新增脚本：

```text
scripts/run_monthly_selection_baselines.py
```

模型：

1. 单因子 rank。
2. 线性 blend。
3. ElasticNet。
4. Logistic top-bucket classifier。
5. XGBoost / LightGBM baseline。

输出：

```text
data/results/monthly_selection_baselines_YYYY-MM-DD_leaderboard.csv
data/results/monthly_selection_baselines_YYYY-MM-DD_monthly_long.csv
data/results/monthly_selection_baselines_YYYY-MM-DD_quantile_spread.csv
docs/monthly_selection_baselines_YYYY-MM-DD.md
```

### M5：多源数据扩展

目标：

```text
逐个数据家族验证增量，而不是一次性把所有新数据拼进去。
```

实验顺序：

1. `price_volume_only`
2. `+ industry_breadth`
3. `+ fund_flow`
4. `+ fundamental`
5. `+ shareholder`
6. `+ northbound / margin / event`，按数据可用性推进

每个家族必须输出：

1. 覆盖率变化。
2. Rank IC 增量。
3. Top-K 超额增量。
4. 分桶单调性。
5. 特征重要性。
6. 年度和 regime slice。

### M6：Learning-to-rank 主模型

目标：

```text
从 baseline 进入真正的月度截面排序模型。
```

候选：

1. LightGBM LambdaRank。
2. XGBoost ranking。
3. top-bucket classifier + rank calibration。
4. regression / classifier / ranker ensemble。

训练规则：

1. 按时间 walk-forward。
2. 禁止随机打乱 cross-validation 作为主证据。
3. 每个训练窗只用当时可见数据。
4. 超参选择只能用训练窗内验证集。
5. 报告每个 OOS 月份的 Top-K 明细。

### M7：月度推荐报告

目标：

```text
形成研究版月度选股报告。
```

推荐报告至少包含：

| 字段 | 说明 |
| --- | --- |
| `rank` | 推荐排名 |
| `symbol` | 股票代码 |
| `name` | 股票名称 |
| `score` | 模型分数 |
| `score_percentile` | 截面分位 |
| `industry` | 行业 |
| `feature_contrib` | 主要贡献特征 |
| `risk_flags` | 涨停、过热、流动性、解禁等风险 |
| `last_month_rank` | 上月是否已入选 |
| `buyability` | 次日可买性 |

推荐名单不是生产持仓，不自动生成交易指令。

---

## 8. 停止清单

以下方向冻结，除非出现明确新证据：

1. 不继续 P1/G0 近邻标签微调。
2. 不继续 dual sleeve 权重网格。
3. 不继续 R2B v1 / R2B v2 replacement 参数网格。
4. 不启动 R3 replacement classifier / pairwise ranker。
5. 不把 `old-new pair` 作为新主线训练样本。
6. 不把 oracle top-K overlap 当作主评价指标。
7. 不用随机 CV 证明时序选股模型有效。
8. 不使用未通过质量检查的 fund flow。
9. 不使用非 PIT 的公告、财务、新闻数据。
10. 不把 shareholder 单因子作为主线救援。
11. 不把未 promotion 的研究配置写入生产日更推荐。
12. 不把 `config.yaml.example` 改成研究候选默认配置。
13. 不把深度学习作为第一阶段主线。

---

## 9. 保留清单

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

## 10. 产出规范

### 10.1 配置身份

| 字段 | 要求 |
| --- | --- |
| `research_topic` | 必填 |
| `research_config_id` | 必填 |
| `output_stem` | 必填 |
| `result_type` | 必填 |
| `config_source` | 必填 |

### 10.2 数据口径

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

### 10.3 训练口径

| 字段 | 要求 |
| --- | --- |
| `model_type` | 必填 |
| `train_window` | 必填 |
| `validation_window` | 必填 |
| `test_window` | 必填 |
| `cv_policy` | 必须是时间序列切分 |
| `hyperparameter_policy` | 必填 |
| `random_seed` | 必填 |

### 10.4 评估口径

| 字段 | 要求 |
| --- | --- |
| `rebalance_rule` | 必填 |
| `execution_mode` | 必填 |
| `benchmark_return_mode` | 必填 |
| `top_k` | 必填 |
| `cost_assumption` | 必填 |
| `buyability_policy` | 必填 |

### 10.5 结果文件

每轮实验至少输出：

1. `summary.json`
2. `leaderboard.csv`
3. `monthly_long.csv`
4. `rank_ic.csv`
5. `quantile_spread.csv`
6. `topk_holdings.csv`
7. `industry_exposure.csv`
8. `candidate_pool_width.csv`
9. `candidate_pool_reject_reason.csv`
10. `feature_importance.csv` 或等价解释文件
11. 一页结论文档

结论文档必须回答：

1. 本轮新增了什么数据或模型。
2. 数据质量是否通过。
3. 使用了哪个 candidate pool，过滤规则是否只做准入。
4. 相对 baseline 是否改善。
5. Top-K 是否稳定跑赢市场。
6. 分桶是否单调。
7. 哪些年份 / regime 失败。
8. 是否允许进入下一阶段。

---

## 11. 证据索引

### 主计划与归档

- `docs/plan.md`
- `docs/plan-04-20.md`

### 目标切换前的关键证据

- `docs/r0_eval_execution_contract_fix_2026-04-28.md`
- `docs/r1f_fixed_baseline_rerun_2026-04-28.md`
- `docs/p2_regime_aware_dual_sleeve_v1_2026-04-28.md`
- `docs/r2b_tradable_upside_replacement_v1_2026-04-28.md`
- `docs/r2b_oracle_replacement_attribution_2026-04-28.md`
- `docs/r2b_edge_gated_replacement_v2_2026-04-28.md`
- `docs/r2b_v2_weight_audit_2026-04-28.md`
- `docs/r5_config_governance_2026-04-28.md`

### 月度选股相关历史基础

- `docs/p1_monthly_investable_label_smoke_2026-04-27.md`
- `docs/p1_monthly_investable_up_capture_g0_smoke_2026-04-27.md`
- `docs/p1_top_bucket_rank_fusion_g0_smoke_2026-04-27.md`
- `docs/p1_rank_fusion_long_horizon_g0_smoke_2026-04-27.md`
- `docs/p1_daily_bt_like_proxy_calibration_2026-04-27.md`

### 新数据质量

- `docs/newdata_quality_current_families_fund_flow-shareholder_flow_a_share_fund_flow_holder_a_share_shareholder_daily_a_share_daily_lag_30_width_100.md`
- `scripts/run_newdata_quality_checks.py`
- `scripts/fetch_fund_flow.py`
- `scripts/fetch_shareholder.py`
- `src/features/fund_flow_factors.py`
- `src/features/shareholder_factors.py`
- `src/features/fundamental_factors.py`

### 核心代码入口

- `config.yaml.backtest`
- `src/models/xtree/p1_workflow.py`
- `src/features/tensor_base_factors.py`
- `src/features/intraday_proxy_factors.py`
- `src/features/fund_flow_factors.py`
- `src/features/shareholder_factors.py`
- `src/features/fundamental_factors.py`
- `src/market/tradability.py`
- `scripts/run_backtest_eval.py`

---

## 12. 当前一句话路线

**项目主线已从“量化持仓 replacement”切换为“量化月度选股”：先构建宽而干净的月度可交易研究池，只做可交易、数据有效和极端风险准入过滤；再修复并扩展 PIT-safe 多源数据，构建月度截面特征与标签表，用 oracle top-bucket 诊断上限，用 baseline 与 learning-to-rank 模型学习稳定提高 Top-K 收益分布；R2B/R3 replacement 主线冻结，生产 promoted registry 继续为空。**
