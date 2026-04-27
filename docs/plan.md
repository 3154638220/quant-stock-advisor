# 策略优化主计划

**文档角色**：当前唯一主计划（canonical）  
**更新时间**：`2026-04-27`  
**当前阶段**：P1 标签和 proxy 准入硬化，P2 数据质量修复，P4 研究和生产边界固化  
**归档入口**：`docs/plan-04-20.md` 仅保留 `2026-04-20` 当日执行记录，不再承担主计划职责

---

## 0. 一页结论

项目的工程底座已经能支撑严肃研究：AkShare 到 DuckDB 的本地数据链路、GPU 因子计算、benchmark-first 回测、XGBoost 分组训练、daily full-backtest-like proxy、正式 full backtest、质量报告和实验产物都有了。当前瓶颈不在“还能不能多加几个因子”，而在三个地方：

1. **训练标签和正式交易口径仍不一致**：旧 light proxy 曾连续高估失败样本，`market_relative + G1` 和 `regression + rank_fusion + G0` 都出现了 proxy 为正但正式回测负超额。
2. **上涨参与不足是主问题**：默认研究基线和 P1 树模型都在强上涨月份明显跑输，全市场等权基准上涨时捕获率不足。
3. **新数据家族尚未达到研究级质量**：资金流和股东人数已接入，但质量 gate 未通过，不能继续作为主线扩实验。

当前推进策略：

1. P1 只做能先改善 `daily_bt_like_proxy_annualized_excess_vs_market` 的最小标签或 proxy 修复假设。
2. P2 先修数据质量，不在质量失败的数据上继续花研究预算。
3. P3 interaction alpha 只允许从失败诊断里提炼，每轮最多 3 个机制假设。
4. P4 固化研究和生产边界，任何 light proxy、signal diagnostic、admission 结果都不能直接进入日更推荐。

一句话：**下一阶段不是扩网格，而是把“候选为什么会在正式月频 Top-20、T+1 open、换手约束、market_ew 基准下赚钱”先证明清楚。**

---

## 1. 当前项目判断

### 1.1 能力现状

| 层面 | 已具备能力 | 当前短板 | 推进判断 |
| --- | --- | --- | --- |
| 数据 | 日线、fund flow、shareholder、fundamental、缓存和质量检查 | fund flow 和日线时间错位；shareholder 存在 `notice_date < end_date` | 先修质量，再谈新 alpha |
| 因子 | 技术面、K 线结构、fund flow、shareholder、weekly KDJ、interaction 原料 | 线性技术面边际收益低，weekly KDJ 直接拼接失败 | 冻结线性拼接，转机制诊断 |
| 回测 | `tplus1_open`、成本、全市场等权基准、换手约束、walk-forward、年度切片 | 旧 proxy 和正式 full backtest 曾严重背离 | daily proxy 作为 P1 前置准入 |
| 树模型 | G0 到 G6 分组、rank/regression、rank_fusion/market_relative 标签 | 标签目标仍没有稳定转化成 benchmark-first 收益 | P1 继续，但收窄到标签和执行对齐 |
| 生产 | `daily_run.py` 推荐链路、CSV 输出、LLM 关注度扫描 | 研究结论容易被误读成生产可用 | 默认配置与研究配置继续解耦 |

### 1.2 当前默认研究基线

除非实验明确另行声明，后续 benchmark-first 对照使用：

| 参数 | 当前值 |
| --- | --- |
| `score` | `S2 = vol_to_turnover` |
| `portfolio_method` | `equal_weight` |
| `top_k` | `20` |
| `rebalance_rule` | `M` |
| `max_turnover` | `0.3` |
| `execution_mode` | `tplus1_open` |
| `prefilter` | `false` |
| `universe_filter` | `true` |
| `benchmark_symbol` | `market_ew_proxy` |

这条基线是**研究对照**，不是生产策略。它的已知问题是上涨月参与不足，尤其 `2021 / 2025 / 2026` 明显落后。

### 1.3 已确认的失败机制

1. V3 基线不是下跌防守失败，而是上涨月份参与不足。上涨月中位超额 `-3.43%`，上涨月跑赢基准比例仅 `18.4%`；下跌月中位超额 `+2.96%`，下跌月跑赢比例 `80.8%`。
2. `G1 = G0 + weekly_kdj_*` 的 light proxy 曾较强，但同窗正式回测没有超过 `G0`，不能 promotion。
3. `rank + market_relative + G1` 的正式 full backtest 年化超额为 `-33.97%`，`2022/2024/2025` 合计退化显著。
4. `regression + rank_fusion + G0` 的 light proxy 为正，但正式 full backtest 年化超额为 `-20.87%`，rolling 和 slice OOS 中位超额均为负。
5. 旧 full-like proxy 仍高估失败样本；daily full-backtest-like proxy 在 5 个历史失败样本上方向一致，平均偏差约 `+0.60pct`，可以作为下一轮 P1 准入层。
6. 资金流和股东人数目前是候选原料，不是主线：两者最新质量报告均未通过 gate。

---

## 2. 推进原则

### 2.1 北极星指标

项目下一阶段只有一个主目标：**构造能稳定跑赢 `market_ew_proxy` 的可解释候选**。

候选不是看单次收益最高，而是必须同时解释：

1. 为什么能改善上涨月参与。
2. 为什么不会显著牺牲下跌月防守。
3. 为什么在 `Top-20 + M + equal_weight + max_turnover=0.3 + tplus1_open` 下仍成立。
4. 为什么在年度、rolling OOS、slice OOS 上不是单段偶然。

### 2.2 研究漏斗

任何新候选必须按下面顺序推进：

| 层级 | 结果类型 | 作用 | 通过后才能做什么 |
| --- | --- | --- | --- |
| 数据质量 | `newdata_quality_summary` | 判断数据是否可解释、是否 PIT 安全 | 进入因子或模型研究 |
| 信号诊断 | `signal_diagnostic` | 看方向、覆盖、分层、状态暴露 | 进入 light proxy |
| 轻量代理 | `light_strategy_proxy` | 快速筛候选和做 A/B | 进入 daily proxy 或 admission |
| daily proxy | `daily_bt_like_proxy_*` | 贴近正式 T+1 open、成本、market_ew 的 P1 准入 | 进入状态诊断和正式回测 |
| 正式回测 | `full_backtest` | promotion 的必要条件 | 进入默认研究基线候选 |

旧 light proxy 只能用于探索，不能再作为 P1 是否补正式回测的直接依据。

### 2.3 Stop Rules

满足任一条件即停止当前方向：

1. P1 候选的 `daily_bt_like_proxy_annualized_excess_vs_market < 0`，默认不补正式 full backtest。
2. P1 候选需要 `tree_score_auto_flipped=True` 才能工作，先诊断方向问题，不 promotion。
3. 新数据家族的质量报告 `ok=False`，不得进入大规模 tree/scout 网格。
4. interaction 候选没有明确失败机制来源，只是“再多试一些”，不进入队列。
5. 正式回测的年度超额中位数、rolling OOS 中位超额、slice OOS 中位超额任一明显为负，不进入默认研究基线。
6. 候选只改善旧 proxy，不改善 daily proxy 或市场状态分层，直接归档。

---

## 3. 未来 30 天执行队列

### 3.1 P1 主线：标签和执行口径对齐

**目标**：让训练目标更接近正式策略真正赚亏的对象，而不是继续追逐截面 Rank IC 或旧 proxy。

优先级最高的三个假设：

| 假设 | 要解决的问题 | 最小实现 | 验收方式 |
| --- | --- | --- | --- |
| H1：月频可投资标签 | 训练标签仍偏 5/10/20 日前向收益，和正式月频持仓收益错位 | 新增面向月末调仓的标签模式，至少显式对齐 `M`、`Top-K`、`tplus1_open` 的收益观测 | G0 先过 daily proxy，再看年度和状态分层 |
| H2：Top-20 边界损失约束 | 旧诊断显示 `21-40` 桶存在机会损失，硬切 Top-20 可能不稳 | 在 P1 summary 固化 Top-20 vs `21-40` 桶差、换入换出收益和边界稳定性，不先改组合参数 | 候选必须改善 Top-20 相对 `21-40`，而不是只改变排序外观 |
| H3：市场状态条件诊断 | `market_relative + G1` 在窄广度和高波动月份退化，强上涨月仍不行 | 每个 P1 候选默认输出强上涨、强下跌、窄广度、高波动状态切片 | 状态切片不能比 G0 明显更差，强上涨月必须有改善证据 |

执行顺序：

1. 先在 `G0` 上验证标签或 proxy 修复，不同时扩 `G1/G2/G3/G4`。
2. `G0` 的 daily proxy 不过 0，停止该标签方向。
3. `G0` daily proxy 过 0 后，再补 `G1`，检验 weekly KDJ 是否提供真实增量。
4. `G1` 只要在窄广度、高波动或强上涨状态明显劣于 `G0`，停止 weekly KDJ 方向。
5. 只有 daily proxy、状态诊断、正式 full backtest 同时成立，才进入 promotion 候选。

当前可复用的命令模板：

```bash
python scripts/run_p1_tree_groups.py \
  --config config.yaml.backtest \
  --groups G0 \
  --label-horizons 5,10,20 \
  --label-mode rank_fusion \
  --xgboost-objective regression \
  --proxy-horizon 5 \
  --rebalance-rule M \
  --top-k 20 \
  --val-frac 0.2 \
  --run-full-backtest \
  --backtest-config config.yaml.backtest \
  --backtest-start 2021-01-01 \
  --backtest-top-k 20 \
  --backtest-max-turnover 0.3 \
  --backtest-portfolio-method equal_weight \
  --out-tag p1_next_label_proxy_smoke
```

说明：现有 runner 已默认启用 daily proxy admission gate。这个命令只作为字段和 gate 的模板，不能把已失败的旧组合原样当作新候选重复跑；后续新增标签模式或机制假设时，替换 `--label-mode`、`--xgboost-objective` 和 `--out-tag`，但必须沿用这个 gate。

### 3.2 P2 数据质量修复

**目标**：让 fund flow 和 shareholder 从“已接入”变成“可解释、可研究”。

当前质量结论：

| 家族 | 状态 | 关键问题 | 下一步 |
| --- | --- | --- | --- |
| `fund_flow` | 暂停新增研究预算 | 资金流最新 `2026-04-24`，日线最新 `2026-04-13`；未匹配 `73,892` 行，其中 `43,734` 行晚于日线最新日期 | 先补齐日线到 `2026-04-24`，再复跑质量报告 |
| `shareholder` | 暂停新增研究预算 | `notice_date` 覆盖率为 `1.0`，但存在 `notice_date < end_date` | 追溯源表和 PIT 逻辑，修复前不做模型网格 |

复核命令：

```bash
python scripts/run_newdata_quality_checks.py \
  --config config.yaml.backtest \
  --families fund_flow,shareholder \
  --output-prefix newdata_quality_current
```

通过标准：

1. `fund_flow` 的日线时间错位解释清楚，已知标的日期错位降到可人工复核范围。
2. `shareholder` 不再出现 `notice_date < end_date`。
3. 质量报告明确写出“是否恢复研究预算”。

### 3.3 P3 Interaction Alpha

**目标**：只从明确失败机制出发，构造少量条件表达。

当前允许保留但不立即扩网格的方向：

1. 低周线 J，但非高波动的反转修正。
2. 资金流背离叠加低换手的条件表达。
3. 弱近期动量但价格位置未破坏的上涨参与修正。

进入条件：

1. 对应机制必须先在 P1 或 V3 失败诊断中出现。
2. 每轮最多 3 个 interaction 假设。
3. 先跑 `signal_diagnostic` 或 daily proxy smoke，不直接进 full backtest。
4. 如果只是放大已有反转簇，立即淘汰。

### 3.4 P4 研究和生产边界

**目标**：避免把研究层的“看起来不错”误写成生产默认。

近期要固化：

1. `config.yaml.backtest` 继续作为研究快照，生产 `config.yaml` 不承载未 promotion 研究结论。
2. 每个研究产物必须写清 `research_topic`、`research_config_id`、`output_stem`、`result_type`。
3. `docs/backtest_report.md` 属于历史报告，不能作为当前默认研究基线的唯一依据。
4. P1 bundle 必须能追溯训练窗口、label spec、feature group、cache、代码入口。
5. 日更推荐只接受已经 promotion 的配置；P1 目前没有任何方案满足。

---

## 4. Promotion Gate

任何新方案要进入默认研究基线候选，必须同时满足：

1. 正式 full backtest 的 `annualized_excess_vs_market >= 0`。
2. 年度超额中位数 `>= 0`。
3. rolling OOS 和 slice OOS 的超额中位数不为负。
4. `2021 / 2025 / 2026` 不能明显更差，且必须解释改善或退化来源。
5. 强上涨月份的中位超额或跑赢比例必须相对当前基线改善。
6. 下跌月份防守不能明显劣化。
7. MaxDD 不明显劣于当前默认研究基线，除非超额改善足够大且可解释。
8. 换手上升必须有明确收益补偿。
9. 树模型必须使用同一标签口径、同一训练窗口、同一执行口径做同窗比较。
10. 依赖 `tree_score_auto_flipped=True` 的模型不能 promotion，必须先解释方向问题。

结果判读优先级：

1. 正式 full backtest 大于 daily proxy。
2. daily proxy 大于旧 light proxy。
3. 状态分层大于单一全样本均值。
4. 同窗对照大于跨窗口历史对照。
5. benchmark-first 指标大于绝对收益指标。

---

## 5. 模块状态看板

| 模块 | 状态 | 当前判断 | 下一步 |
| --- | --- | --- | --- |
| P0 评估硬化 | 第一轮完成 | 不再阻塞 P1，但仍要维护结果身份和 schema | 将 daily proxy、状态切片、正式回测字段纳入固定报告模板 |
| P1 树模型 | 主线 | daily proxy 已接入并通过历史失败样本校准；旧标签和目标函数均未 promotion | 只做标签和执行口径对齐，先 G0 后 G1 |
| P2 新数据 | 维护 | fund flow 和 shareholder 质量 gate 未过 | 补日线、修 PIT，再复跑质量报告 |
| P3 Interaction Alpha | 诊断驱动 | 不再扩 weekly KDJ 或同质反转网格 | 每轮最多 3 个机制假设，先诊断后 proxy |
| P4 生产边界 | 已开始 | 研究链路不能影响日更推荐 | 固化 canonical config、bundle registry、报告字段 |

---

## 6. P1 分组状态

| 分组 | 定义 | 当前判断 | 下一动作 |
| --- | --- | --- | --- |
| `G0` | baseline technical | 正式诊断基准，仍负超额 | 作为所有新标签/proxy 修复的第一验证对象 |
| `G1` | `G0 + weekly_kdj_*` | light proxy 好，但正式回测没有兑现 | 仅在 G0 先过 daily proxy 后再测 |
| `G2` | `G0 + fund_flow_*` | full backtest 劣化，且 fund flow 质量未过 | 暂停 |
| `G3` | `G0 + shareholder_*` | 无增量，且 PIT 质量未过 | 暂停 |
| `G4` | `G0 + weekly_kdj_* + fund_flow_*` | 未超过 `G1`，数据质量也不足 | 暂停 |
| `G5` | `G0 + weekly_kdj_* + weekly_kdj_interaction_*` | light proxy 未超过 `G1` | 仅保留为机制诊断 |
| `G6` | `G0 + weekly_kdj_interaction_*` | light proxy 未超过 `G1` | 仅保留为机制诊断 |

P1 禁区：

1. 不继续扩 `weekly_kdj` interaction 网格。
2. 不把 `weekly_kdj_*` 写回默认 `composite_extended`。
3. 不重复跑 `G2/G4` 直拼 fund flow。
4. 不用跨窗口、跨标签口径、跨缓存版本的结果做 promotion 判断。
5. 不把 `signal_diagnostic` 或旧 light proxy 当成正式回测结论。

---

## 7. 冻结结论

这些判断当前不再反复重跑，除非后续出现新数据、代码修复或口径修正：

1. `prefilter=true` 在当前主线下会误伤有效候选，默认研究口径继续关闭。
2. `universe_filter=true` 在 `prefilter=false` 条件下仍是正向交互，默认继续开启。
3. `Top-30 / Top-40 / 20-35 / 20-40 / 更小缓冲 / 分层等权` 已止损。
4. `risk_parity` 保留为诊断工具，不回到主线。
5. `mean_variance` 继续停用。
6. `asset_turnover`、`net_margin_stability` 未通过 benchmark-first 准入，不进入默认主线。
7. 股东人数单因子降级为低优先级。
8. `weekly_kdj_j / weekly_kdj_oversold_depth / weekly_kdj_rebound` 不默认加入 `composite_extended`。
9. `weekly_kdj_interaction_*` 的 `G5/G6` 轻量 A/B 未超过 `G1`，暂不进入 full backtest 队列。
10. `fund_flow_*` 已在训练和推理面板可用，但 `G2/G4` full backtest 未给出正增量，暂不推进为 P1 主线。

---

## 8. 证据索引

### 8.1 默认基线和上涨参与不足

入口：`docs/benchmark_gap_diagnostics_2026-04-20_v3.md`

关键结论：

1. 当前弱势主要是上涨月份参与不足，不是下跌防守失败。
2. 持仓偏更大市值、更高成交额、低波动、近期相对动量偏弱，更像稳定票。
3. `21-40` 桶平均前向收益接近 `01-20`，但中位数仍不稳定，适合做边界诊断，不适合重启 Top-K 网格。

### 8.2 P1 标签、目标和 proxy 断层

入口：

- `docs/p1_label_objective_experiment_2026-04-26.md`
- `docs/p1_marketrel_g1_gap_diagnostics_2026-04-26.md`
- `docs/p1_full_like_proxy_calibration_smoke_2026-04-27.md`
- `docs/p1_proxy_calibration_history_2026-04-27.md`
- `docs/p1_daily_bt_like_proxy_calibration_2026-04-27.md`
- `docs/p1_marketrel_state_diagnostics_2026-04-27.md`

关键结论：

1. `rank + market_relative + G1` light proxy 最强，但 full backtest 严重失败。
2. `regression + rank_fusion + G0` light proxy 为正，但正式回测仍显著负超额。
3. `market_relative + G1` 的问题集中在标签口径和 proxy 断层，执行成本只是次要放大项。
4. daily full-backtest-like proxy 在 5 个历史失败样本上全部为负，和正式 full backtest 方向一致。
5. 窄广度、高波动月份是 G1 相对 G0 的主要退化来源。

### 8.3 Weekly KDJ

入口：

- `docs/alpha_factor_scout_2026-04-24_weekly_kdj.md`
- `docs/p1_weekly_interaction_ab_2026-04-26.md`
- `docs/p1_rank_direction_rerun_2026-04-26.md`
- `docs/p1_rankfix_same_window_full_backtest_2026-04-26.md`
- `docs/p1_failure_diagnostics_2026-04-26.md`

关键结论：

1. `weekly_kdj_j` 有负向 Rank IC 证据。
2. 直接线性加入或简单 interaction 没有形成正式回测增量。
3. 后续若继续使用，只能作为条件诊断变量，不作为直接拼接特征组进入 full backtest。

### 8.4 新数据

入口：

- `docs/newdata_quality_current_families_fund_flow-shareholder_flow_a_share_fund_flow_holder_a_share_shareholder_daily_a_share_daily_lag_30_width_100.md`
- `docs/alpha_factor_scout_2026-04-23_shareholder_smoke.md`
- `data/results/p1_full_backtest_g2_fundflow.json`
- `data/results/p1_full_backtest_g4_fundflow.json`
- `scripts/run_newdata_quality_checks.py`

关键结论：

1. fund flow 和 shareholder 链路保留。
2. 本轮质量 gate 均未通过，短期暂停新增研究预算。
3. 新数据不能作为 P1 promotion 主线，先修时间对齐和 PIT 异常。

---

## 9. 产出规范

每轮新实验至少保留：

1. 配置或命名身份：
   - `research_topic`
   - `research_config_id`
   - `output_stem`
   - `result_type`
2. 结果文件：
   - summary CSV
   - detail CSV 或 period detail
   - JSON
   - 必要时的 manifest
3. 一页结论文档，必须写清：
   - 只改了什么
   - 对默认研究基线的变化
   - 对 `market_ew_proxy` 的变化
   - 是否改变主计划
   - 是否允许进入下一层验证

每份正式回测还必须显式写清：

1. `benchmark_symbol`
2. `benchmark_min_history_days` 或等价说明
3. `top_k`
4. `rebalance_rule`
5. `portfolio_method`
6. `execution_mode`
7. `prefilter`
8. `universe_filter`
9. 成本假设
10. cache 路径与 schema/version
11. 若发生 fallback，必须记录 fallback 原因

---

## 10. 当前不该做的事情

1. 不继续做 `Top-K`、缓冲带、分层等权的小参数网格。
2. 不把 `weekly_kdj_*` 写回默认 `composite_extended`。
3. 不继续为股东人数做单因子或线性权重主线。
4. 不把 `fund_flow_*` 直接拼进 P1 主线重复跑 `G2/G4`。
5. 不把 `signal_diagnostic` 或旧 `light_strategy_proxy` 当成正式回测结论。
6. 不用跨窗口、跨标签口径、跨缓存版本的结果做 promotion 判断。
7. 不在没有明确失败机制的情况下扩 expression 网格。
8. 不把任何未 promotion 的研究配置写入生产日更推荐。

---

## 11. 一句话路线图

未来一个阶段的最优推进方式是：

**先用 daily full-backtest-like proxy 管住 P1 候选，再把标签目标对齐到正式月频交易收益；同时修复 fund flow 和 shareholder 的质量 gate。只有当候选能改善上涨参与、守住下跌防守，并在正式 full backtest 与 OOS 切片中跑赢 `market_ew_proxy`，才允许进入默认研究基线候选。**
