# 策略优化主计划

**文档角色**：当前唯一主计划（canonical）
**更新时间**：`2026-04-26`
**当前阶段**：P1 树模型目标/标签诊断 + P2 新数据质量归档
**归档入口**：`docs/plan-04-20.md` 仅保留 `2026-04-20` 当日执行记录，不再承担主计划职责

---

## 0. 当前决策

项目已经从“系统性负收益修复”进入“能否稳定跑赢全市场等权基准”的阶段。当前没有任何新方案可以写回默认主线。

当前执行判断：

1. `V3 = S2(vol_to_turnover) + equal_weight + Top-20 + M + prefilter=false + universe=true` 是默认研究基线，但只是对照基线，不是可生产策略。
2. 线性技术面主干已经接近饱和，继续做 `Top-K / 缓冲带 / 分层等权 / 同类反转因子加权` 的收益很低。
3. `weekly_kdj_*` 只保留为树模型或条件表达素材，不再直接加入 `composite_extended`。
4. P0 评估硬化已完成第一轮收口，不再阻塞 P1。
5. P1 rank-fix 同窗正式对照与 G0/G1 失效诊断已完成：`G1 = G0 + weekly_kdj_*` 没有超过同口径 `G0`。
6. P1 的核心问题不是“缺更多特征”，而是训练目标、标签口径、light proxy 与 full backtest 之间的断层。
7. 资金流和股东人数链路已经接入，但研究结论偏弱；它们目前是候选原料，不是主线。

一句话：**先补新数据质量归档，再只做 `regression + rank_fusion + G0` 正式回测和 `market_relative + G1` 断层诊断；在这些结果出来前，不扩线性网格、不扩 weekly KDJ 网格、不重复 G2/G4 直拼。**

---

## 1. 下一步执行队列

### 1.1 立即做：新数据质量归档

目的：把当前 DuckDB 中 `fund_flow` 和 `shareholder` 的覆盖、缺失、重复、全零、时间错位写入正式研究产物，决定后续是否继续给 P2 研究预算。

建议命令：

```bash
python scripts/run_newdata_quality_checks.py \
  --config config.yaml.backtest \
  --families fund_flow,shareholder \
  --output-prefix newdata_quality_current
```

完成标准：

1. 生成 `data/results/*newdata_quality_current*_summary.csv`。
2. 生成 `data/results/*newdata_quality_current*.json` 与 manifest。
3. 生成 `docs/*newdata_quality_current*.md`。
4. 报告中明确写出 `fund_flow` 和 `shareholder` 是否继续保留研究预算。

### 1.2 紧接着做：`regression + rank_fusion + G0` 正式回测

目的：验证第一轮目标函数实验中最值得补 full backtest 的候选。只跑 `G0`，不扩特征组。

建议命令：

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
  --out-tag p1_label_objective_reg_rankfusion
```

判读规则：

1. 若 full backtest 仍显著负超额，目标函数方向降温，P1 先转向断层诊断。
2. 若 full backtest 明显改善但未过 gate，只允许做同窗 `G1` 对照，不允许直接 promotion。
3. 若过 gate，仍需补 rolling / slice OOS、年度拆解和关键年份解释。

### 1.3 可并行或随后做：`market_relative + G1` 断层诊断

目的：解释为什么 `rank + market_relative + G1` 在 light proxy 上最好，但 full backtest 严重失败。

诊断范围：

1. rank bucket：特别是 Top-20 与 `21-40` 桶的机会损失。
2. 持仓暴露：市值、成交额、波动率、近期动量、价格位置、换手。
3. 年份退化：重点解释 `2022 / 2024 / 2025`。
4. 换手与执行：确认 light proxy 是否低估了月频换仓与成本冲击。

完成标准：输出一页诊断文档，结论必须写清“是标签口径失效、proxy 断层、执行口径断层，还是三者共同作用”。

---

## 2. 默认研究口径

除非实验明确另行声明，后续所有 benchmark-first 比较都以这一组口径为默认参考：

- `score = S2 = vol_to_turnover`
- `portfolio_method = equal_weight`
- `top_k = 20`
- `rebalance_rule = M`
- `max_turnover = 0.3`
- `execution_mode = tplus1_open`
- `prefilter = false`
- `universe_filter = true`
- `benchmark_symbol = market_ew_proxy`

这条基线的含义是“当前最稳的研究对照”，不是“生产默认答案”。已知它仍存在稳定负超额：上涨月份参与不足，尤其在 `2021 / 2025 / 2026` 明显落后。

---

## 3. Promotion Gate

任何新方案要进入默认研究基线候选，必须同时满足：

1. 正式 full backtest 的 `annualized_excess_vs_market >= 0`。
2. 年度超额中位数 `>= 0`。
3. `rolling / slice OOS` 的超额中位数不为负。
4. `2021 / 2025 / 2026` 不能明显更差，至少要解释清楚改善或退化来源。
5. MaxDD 不明显劣于当前默认研究基线，除非超额改善足够大且可解释。
6. 换手上升必须有明确收益补偿。
7. 树模型必须使用同一标签口径、同一训练窗口、同一执行口径做同窗比较。
8. 如果模型依赖 `tree_score_auto_flipped=True` 才能工作，必须先解释方向问题，不能 promotion。

结果层级：

| 层级 | `result_type` | 用途 | 能否直接 promotion |
| --- | --- | --- | --- |
| 信号诊断 | `signal_diagnostic` | 看方向、覆盖、频率感知风险收益 | 不能 |
| 轻量代理 | `light_strategy_proxy` | 快速筛候选、做 A/B | 不能 |
| 因子准入 | `admission` | 检查 `IC + combo + benchmark` 门槛 | 仍需 full backtest |
| 正式回测 | `full_backtest` | promotion 的必要条件 | 可以作为最终依据 |

轻量结果只能决定“要不要进入下一层”，不能直接解释成正式策略收益。

---

## 4. 状态看板

| 模块 | 状态 | 当前判断 | 下一步 |
| --- | --- | --- | --- |
| P0 评估硬化 | 第一轮完成 | 不再阻塞 P1 | 维护 schema/version、身份字段、结果分层 |
| P1 树模型 | 主线 | G1 未超过同窗 G0，目标/标签断层待解释 | 补 `regression + rank_fusion + G0` full backtest |
| P2 新数据 | 链路完成、价值未证 | fund flow / shareholder 均不能直接进主线 | 跑正式质量报告 |
| P3 interaction alpha | 诊断驱动 | 不再扩同质网格 | 每轮最多 3 个明确机制假设 |
| P4 研究/生产边界 | 已开始 | 不能把研究轻量结果写入日更推荐 | 继续固化 canonical config 与 bundle 追踪 |

---

## 5. P1 树模型主线

### 5.1 目标

验证非线性模型是否能把“线性主干里重复或弱势的特征”转化成真实 benchmark-first 增量。当前重点不是再加特征，而是解释训练目标、标签方向、年份失效和执行口径。

### 5.2 分组状态

| 分组 | 定义 | 当前判断 |
| --- | --- | --- |
| `G0` | baseline technical | rank-fix 同窗 full backtest 已补；仍负超额，作为正式诊断基准 |
| `G1` | `G0 + weekly_kdj_*` | light proxy 好，但同窗 full backtest 未超过 `G0` |
| `G2` | `G0 + fund_flow_*` | full backtest 明显劣化，暂停主线 promotion |
| `G3` | `G0 + shareholder_*` | 无增量，降级 |
| `G4` | `G0 + weekly_kdj_* + fund_flow_*` | 未超过 `G1`，暂停 |
| `G5` | `G0 + weekly_kdj_* + weekly_kdj_interaction_*` | light proxy 未超过 `G1`，观察 |
| `G6` | `G0 + weekly_kdj_interaction_*` | light proxy 未超过 `G1`，观察 |

### 5.3 已完成结论

1. rank 标签方向问题已修正，`tree_score_auto_flipped=False`。
2. rank-fix 同窗 full backtest 已补：`G1` 没有 benchmark-first 增量。
3. G0/G1 年份失效诊断已补：两组共同弱点仍是上涨月捕获不足。
4. `weekly_kdj_*` 更像强化头部反转排序，没有稳定解决 `21-40` 桶机会损失。
5. 第一轮目标/标签实验已完成：
   - `rank + rank_fusion`、`regression + rank_fusion`、`rank + market_relative` 已做 light proxy。
   - `rank + market_relative` 的 G0/G1 已补 full backtest，未过 gate，且 G1 明显劣化。
   - `regression + rank_fusion + G0` 仍待正式回测。

### 5.4 P1 禁区

1. 不继续扩 `weekly_kdj` interaction 网格。
2. 不把 `weekly_kdj_*` 写回默认 `composite_extended`。
3. 不用跨窗口、跨标签口径、跨缓存版本的结果做 promotion 判断。
4. 不把 `signal_diagnostic` 或 `light_strategy_proxy` 当成正式回测结论。

---

## 6. P2 新数据家族

### 6.1 资金流

当前判断：保留数据链路，暂停主线 promotion。

已知状态：

1. 在线抓取和 cache 阻塞已经解除。
2. `a_share_fund_flow` 已有约 `58.1` 万行、`4,866` 个标的，最新 `trade_date = 2026-04-24`。
3. `G2 = G0 + fund_flow_*` 明显劣化。
4. `G4 = G0 + weekly_kdj_* + fund_flow_*` 未超过 `G1`。

下一步：

1. 跑正式质量报告并归档到 `docs/`。
2. 只在出现明确机制假设时重新研究，例如“资金流只在低换手反转候选里有效”。
3. 不再重复 `G2/G4` 这种直接拼接。

### 6.2 股东人数

当前判断：低优先级保留。

已知状态：

1. 单因子未通过 IC、组合层和基准层验证。
2. `G3 = G0 + shareholder_*` 无增量。
3. 只有 PIT/覆盖质量明显好于当前认知时，才继续投入研究预算。

下一步：

1. 跑正式 PIT/覆盖质量报告。
2. 若质量报告没有改善认知，则不再做大规模表达式或树模型网格。

---

## 7. P3 Gated / Interaction Alpha

当前判断：需要更结构化的 alpha 表达，但不能靠“多试一些”推进。

规则：

1. 每轮最多提出 `3` 个有明确失效机制来源的 interaction 假设。
2. 每个假设先跑 `signal_diagnostic` 或 light proxy，再决定是否进入 scout。
3. 若一个 interaction 只是重复放大已有反转簇，应立即淘汰。
4. `realized_vol` 反向表达保留为防守 overlay 候选，不按新 alpha 主线推进。

可保留但不立即执行的候选方向：

- “低周线 J + 非高波动”的反转修正
- “资金流背离 + 低换手”的条件表达
- “弱近期动量但未破坏价格位置”的上涨参与修正

这些方向必须先由失效诊断证明有对应问题，再进入实验。

---

## 8. P4 研究链路与生产链路分离

当前边界：

- 生产链路：`daily_run.py`、推荐 CSV、日更配置。
- 研究链路：benchmark-first 回测、alpha scout、admission、P1 树模型训练、新数据质量检查。

规则：

1. 两条链路不能混用默认配置。
2. 研究分支的 light proxy 结果不能直接影响日更推荐。
3. `config.yaml.example` 只表达可运行默认，不承载未 promotion 的研究结论。
4. 每个模型 bundle 都要能追溯训练窗口、特征组、label spec、数据 cache 和代码入口。
5. 每份研究报告必须写清“是否改变主计划”；没有改变时，明确写“仅归档，不 promotion”。

当前 canonical research configs：

- `v3_market_ew_full_backtest`
- `p1_tree_full_backtest`
- 待补：`newdata_quality_current`

---

## 9. 冻结结论

这些判断当前不再反复重跑，除非后续出现新数据或口径修正：

1. `prefilter=true` 在当前主线下会误伤有效候选，默认研究口径继续关闭。
2. `universe_filter=true` 在 `prefilter=false` 条件下仍是正向交互，默认继续开启。
3. `Top-30 / Top-40 / 20-35 / 20-40 / 更小缓冲 / 分层等权` 已止损。
4. `risk_parity` 保留为诊断工具，不回到主线。
5. `mean_variance` 继续停用。
6. `asset_turnover`、`net_margin_stability` 未通过 benchmark-first 准入，不进入默认主线。
7. 股东人数单因子降级为低优先级。
8. `weekly_kdj_j / weekly_kdj_oversold_depth / weekly_kdj_rebound` 不默认加入 `composite_extended`。
9. `weekly_kdj_interaction_*` 的 `G5/G6` 轻量 A/B 未超过 `G1`，暂不进入 full backtest 队列。
10. `fund_flow_*` 已在训练/推理面板真实可用，但 `G2/G4` full backtest 未给出正增量，暂不推进为 P1 主线。

---

## 10. 证据索引

### V3 基准差距

入口：`docs/benchmark_gap_diagnostics_2026-04-20_v3.md`

结论：当前弱势主要不是下跌月份防守失败，而是上涨月份参与不足。策略持仓偏更大市值、更高成交额、低波动、近期相对动量偏弱，更像“稳定票”，不是强势扩散行情里的弹性票。

### Weekly KDJ

入口：

- `docs/alpha_factor_scout_2026-04-24_weekly_kdj.md`
- `docs/p1_weekly_interaction_ab_2026-04-26.md`
- `docs/p1_rank_direction_rerun_2026-04-26.md`
- `docs/p1_rankfix_same_window_full_backtest_2026-04-26.md`
- `docs/p1_failure_diagnostics_2026-04-26.md`

结论：`weekly_kdj_j` 有负向 Rank IC 证据，但直接线性加入或简单 interaction 没有形成正式回测增量。

### P1 目标与标签

入口：`docs/p1_label_objective_experiment_2026-04-26.md`

结论：

1. `rank + market_relative` 的 G1 light proxy 最好，但 full backtest 严重失败。
2. `regression + rank_fusion` 提高了 G0 light proxy，但仍需补正式回测。
3. 训练期 proxy Top-K 与正式月频换手、执行成本、全样本时段之间存在断层。

### 新数据

入口：

- `docs/alpha_factor_scout_2026-04-23_shareholder_smoke.md`
- `data/results/p1_full_backtest_g2_fundflow.json`
- `data/results/p1_full_backtest_g4_fundflow.json`
- `scripts/run_newdata_quality_checks.py`

结论：资金流和股东人数链路保留，但短期不作为 P1 promotion 主线。

---

## 11. 产出规范

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

- `benchmark_symbol`
- `benchmark_min_history_days` 或等价说明
- `top_k`
- `rebalance_rule`
- `portfolio_method`
- `execution_mode`
- `prefilter`
- `universe_filter`
- 成本假设
- cache 路径与 schema/version
- 若发生 fallback，必须记录 fallback 原因

---

## 12. 当前不该做的事情

1. 不继续做 `Top-K`、缓冲带、分层等权的小参数网格。
2. 不把 `weekly_kdj_*` 写回默认 `composite_extended`。
3. 不继续为股东人数做单因子或线性权重主线。
4. 不把 `fund_flow_*` 直接拼进 P1 主线重复跑 `G2/G4`。
5. 不把 `signal_diagnostic` 或 `light_strategy_proxy` 当成正式回测结论。
6. 不用跨窗口、跨标签口径、跨缓存版本的结果做 promotion 判断。
7. 不在没有明确失败机制的情况下扩 expression 网格。

---

## 13. 一句话结论

当前最优路线不是再扩线性因子和参数网格，而是：

**先用 P0 已硬化的评估链路补齐新数据质量归档，再用最小正式回测验证 `regression + rank_fusion + G0`；只有解释清楚 P1 为什么 light proxy 好但 full backtest 跑输后，才继续做少量诊断驱动的 interaction alpha。**
