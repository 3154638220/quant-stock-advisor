# 策略优化主计划

**文档角色**：当前唯一主计划（canonical）
**更新时间**：`2026-04-27`
**当前阶段**：P1 proxy 准入硬化 + P2 数据链路维护
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

一句话：**新数据质量归档、`regression + rank_fusion + G0` 正式回测、`market_relative + G1` 断层诊断、daily proxy 校准与市场状态分层均已完成，均不支持 promotion；下一步只允许先过 daily proxy 准入的最小标签/proxy 修复假设，不扩线性网格、不扩 weekly KDJ 网格、不重复 G2/G4 直拼。**

---

## 1. 下一步执行队列

### 1.1 已完成：新数据质量归档

目的：把当前 DuckDB 中 `fund_flow` 和 `shareholder` 的覆盖、缺失、重复、全零、时间错位写入正式研究产物，决定后续是否继续给 P2 研究预算。

执行命令：

```bash
python scripts/run_newdata_quality_checks.py \
  --config config.yaml.backtest \
  --families fund_flow,shareholder \
  --output-prefix newdata_quality_current
```

归档入口：`docs/newdata_quality_current_families_fund_flow-shareholder_flow_a_share_fund_flow_holder_a_share_shareholder_daily_a_share_daily_lag_30_width_100.md`

结论：`fund_flow` 与 `shareholder` 均未通过本轮质量 gate，短期暂停新增 P2 研究预算，仅保留数据链路维护。

1. `fund_flow`：`581,433` 行、`4,866` 个标的、最新 `trade_date = 2026-04-24`，但日线表最新仅到 `2026-04-13`；`73,892` 行无法匹配日线表，其中 `43,734` 行晚于日线最新日期，`30,151` 行来自日线表完全没有的 `281` 个标的，仅 `7` 行属于日线已覆盖标的但具体日期未匹配。
2. `shareholder`：`14,075` 行、`5,354` 个标的、`notice_date` 覆盖率 `1.0`，但存在 `notice_date < end_date` 异常记录。
3. 已生成 `data/results/*newdata_quality_current*_summary.csv`。
4. 已生成 `data/results/*newdata_quality_current*.json` 与 manifest。
5. 已生成 `docs/*newdata_quality_current*.md`。

### 1.2 已完成：`regression + rank_fusion + G0` 正式回测

目的：验证第一轮目标函数实验中最值得补 full backtest 的候选。只跑 `G0`，不扩特征组。

执行命令：

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

产物入口：

- `data/results/p1_label_objective_reg_rankfusion_rb_m_top20_lh_5-10-20_px_5_val20_lbl_rank_fusion_obj_regression_20260426_141830_summary.csv`
- `data/results/p1_label_objective_reg_rankfusion_rb_m_top20_lh_5-10-20_px_5_val20_lbl_rank_fusion_obj_regression_20260426_141830.json`
- `data/results/p1_full_backtest_g0.json`

结论：不 promotion，目标函数方向降温，P1 先转向断层诊断。

1. Light proxy 仍为正：`annualized_excess_vs_market = +26.13%`，`val_rank_ic = 0.0972`，`tree_score_auto_flipped=False`。
2. 正式 full backtest 显著负超额：`full_backtest_annualized_excess_vs_market = -20.87%`，含成本年化收益 `-7.61%`，MaxDD `45.51%`。
3. OOS 不支持 promotion：rolling OOS 超额中位数 `-10.97%`，slice OOS 超额中位数 `-24.55%`，年度超额中位数 `-17.21%`。
4. 关键年份仍弱：`2021/2025/2026` 平均超额 `-23.72%`。
5. 注意：本轮 summary 中 `pass_p1_*_gate=True` 是相对单组 baseline delta 的内部字段；按本文 Promotion Gate 的绝对口径，本轮明确未过 gate。

### 1.3 已完成：`market_relative + G1` 断层诊断

目的：解释为什么 `rank + market_relative + G1` 在 light proxy 上最好，但 full backtest 严重失败。

诊断范围：

1. rank bucket：特别是 Top-20 与 `21-40` 桶的机会损失。
2. 持仓暴露：市值、成交额、波动率、近期动量、价格位置、换手。
3. 年份退化：重点解释 `2022 / 2024 / 2025`。
4. 换手与执行：确认 light proxy 是否低估了月频换仓与成本冲击。

归档入口：`docs/p1_marketrel_g1_gap_diagnostics_2026-04-26.md`

结论：本轮是**标签口径失效 + light proxy 断层共同作用**，执行成本只是次要放大项；不 promotion，不扩 `market_relative + weekly_kdj` 网格。

1. Light proxy 与正式回测严重背离：`G1` light proxy 年化超额 `+17.98%`，正式 full backtest 对 `market_ew` 年化超额 `-33.97%`。
2. Top-20 边界不稳：`G1` Top-20 平均前向收益 `2.76%`，但中位数 `-0.20%`；`41-60` 桶中位数反而为 `+0.56%`。
3. 暴露发生结构漂移：`G1` 相对基准偏小市值、低价格位置、更高波动/换手；这与默认研究基线的稳健票倾向相反，但没有换来上涨月捕获。
4. 年份退化集中在 `2022/2024/2025`：`G1-G0` 月度超额合计分别为 `-40.41% / -62.85% / -5.89%`。
5. 执行成本不是主因：`G1` turnover mean `34.68%`，成本拖累年化约 `-0.62%`，远小于年化超额缺口。

### 1.4 已完成第一轮：full-like light proxy 校准

目的：只针对已确认断层做最小机制假设，不再扩同质因子网格。第一版先修 light proxy，让它显式模拟正式口径中的月频 Top-20 和 turnover cap。

已完成：

1. 新增 full-like proxy：`Top-20 + max_turnover=0.3`，输出 `full_like_proxy_*` 与 `proxy_gap_full_like_minus_unconstrained`。
2. 已跑 `market_relative` G0/G1 smoke，入口：`docs/p1_full_like_proxy_calibration_smoke_2026-04-27.md`。
3. Smoke 显示新版 proxy 更保守：G0 从 `+3.04%` 压到 `+1.42%`，G1 从 `+16.42%` 压到 `+9.74%`。
4. 已复用 5 个历史 full backtest bundle 做跨失败样本校准，入口：`docs/p1_proxy_calibration_history_2026-04-27.md`。
5. 已新增 daily full-backtest-like proxy：用 score 构造 `equal_weight + Top-K + max_turnover` 调仓权重，直接复用日频 open-to-open 收益、交易成本和 `market_ew` 对齐口径，输出 `daily_bt_like_proxy_*` 与 `proxy_gap_daily_bt_like_minus_unconstrained`。
6. 已复用同 5 个历史 full backtest bundle 校准 daily proxy，入口：`docs/p1_daily_bt_like_proxy_calibration_2026-04-27.md`。

结论：`Top-20 + max_turnover=0.3` 的 forward-return full-like proxy 仍不足以作为准入 gate；daily full-backtest-like proxy 可以作为 P1 下一轮正式回测前的准入层。

1. `proxy_horizon=5` 时，5 个样本的 full-like proxy 全部仍为正，相对正式 full backtest 平均高估 `+28.58pct`。
2. `proxy_horizon=20` 时，背离反而更大，平均高估 `+38.10pct`；因此问题不是简单的 `5d` horizon 太短。
3. daily proxy 在 5 个历史样本中全部为负，和正式 full backtest 方向一致；相对正式 full backtest 平均偏差 `+0.60pct`，最大绝对偏差 `2.44pct`。

已完成追加：

1. 已对 `market_relative + G0/G1` 增加年度/市场状态分层诊断，入口：`docs/p1_marketrel_state_diagnostics_2026-04-27.md`。
2. 关键年份 `2022/2024/2025` 合计超额：`G1 = -142.55%`，明显弱于 `G0 = -33.40%`。
3. `G1` 的强上涨月中位超额仍为 `-5.28%`，强下跌月中位超额为 `-1.39%`；说明 `market_relative` 没有修复上涨参与，还削弱了下跌月防守。
4. `G1-G0` 退化主要集中在窄广度与高波动月份：窄广度月份合计 `-95.96%`，高波动月份合计 `-82.82%`。

下一步：

1. 已把 daily proxy 固化为 P1 候选准入 gate：`run_p1_tree_groups.py --run-full-backtest` 默认要求 `daily_bt_like_proxy_annualized_excess_vs_market >= 0`，否则跳过正式 full backtest 并写入 `full_backtest_skipped_reason`。
2. `market_relative + weekly_kdj` 不保留为可继续扩展方向；只能作为失败样本用于校验新 proxy / 标签设计。
3. 若继续用 weekly KDJ，只允许作为条件诊断变量，不作为直接拼接特征组进入 full backtest。

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
| P1 树模型 | 主线 | daily full-backtest-like proxy 已接入并通过历史失败样本校准；`market_relative + G1` 的年度/市场状态退化已解释；runner 已默认用 daily proxy 拦截 full backtest | 只设计能先改善 daily proxy 的最小标签/proxy 假设 |
| P2 新数据 | 质量归档完成 | fund flow / shareholder 均未过质量 gate，暂停新增研究预算 | 只维护数据链路，先排查时间错位与 PIT 异常 |
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
   - `regression + rank_fusion + G0` 已补 full backtest：light proxy 为正，但正式 full backtest 年化超额 `-20.87%`，不 promotion。
6. `market_relative + G1` 断层诊断已补：主要是标签口径与 light proxy 未刻画正式月频 Top-20 边界和市场状态，执行成本不是主因。

### 5.4 P1 禁区

1. 不继续扩 `weekly_kdj` interaction 网格。
2. 不把 `weekly_kdj_*` 写回默认 `composite_extended`。
3. 不用跨窗口、跨标签口径、跨缓存版本的结果做 promotion 判断。
4. 不把 `signal_diagnostic` 或 `light_strategy_proxy` 当成正式回测结论。

---

## 6. P2 新数据家族

### 6.1 资金流

当前判断：保留数据链路，暂停新增研究预算与主线 promotion。

已知状态：

1. 在线抓取和 cache 阻塞已经解除。
2. `a_share_fund_flow` 已有约 `58.1` 万行、`4,866` 个标的，最新 `trade_date = 2026-04-24`。
3. `G2 = G0 + fund_flow_*` 明显劣化。
4. `G4 = G0 + weekly_kdj_* + fund_flow_*` 未超过 `G1`。
5. `newdata_quality_current` 质量报告未通过：资金流未匹配行已拆解为日线滞后、额外市场标的和 `7` 行已知标的日期错位；主阻塞是先补齐日线到 `2026-04-24`，再复核剩余错位。

下一步：

1. 先补齐日线表到资金流最新日期 `2026-04-24`，再复跑质量报告；若仍有已知标的日期错位，只修那部分。
2. 只在出现明确机制假设时重新研究，例如“资金流只在低换手反转候选里有效”。
3. 不再重复 `G2/G4` 这种直接拼接。

### 6.2 股东人数

当前判断：保留数据链路，暂停新增研究预算。

已知状态：

1. 单因子未通过 IC、组合层和基准层验证。
2. `G3 = G0 + shareholder_*` 无增量。
3. 只有 PIT/覆盖质量明显好于当前认知时，才继续投入研究预算。
4. `newdata_quality_current` 质量报告未通过：存在 `notice_date < end_date` 异常记录；虽然 `notice_date` 覆盖率为 `1.0`，但 PIT 可信度需要先修复。

下一步：

1. 排查并修复 `notice_date < end_date` 的来源。
2. 修复前不做大规模表达式或树模型网格。

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
- `newdata_quality_current`

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
- `docs/p1_marketrel_g1_gap_diagnostics_2026-04-26.md`
- `docs/p1_full_like_proxy_calibration_smoke_2026-04-27.md`
- `docs/p1_proxy_calibration_history_2026-04-27.md`
- `docs/p1_proxy_calibration_history_h20_2026-04-27.md`

结论：`weekly_kdj_j` 有负向 Rank IC 证据，但直接线性加入或简单 interaction 没有形成正式回测增量。

### P1 目标与标签

入口：`docs/p1_label_objective_experiment_2026-04-26.md`

结论：

1. `rank + market_relative` 的 G1 light proxy 最好，但 full backtest 严重失败。
2. `regression + rank_fusion` 提高了 G0 light proxy，但补正式回测后仍显著负超额。
3. `market_relative + G1` 断层诊断确认主因是标签口径失效和 light proxy 断层共同作用，执行成本只是次要放大项。
4. full-like proxy 第一轮历史校准仍显著高估正式回测；`h=20` 不能修复。daily full-backtest-like proxy 已把 5 个历史样本的平均正式回测偏差降到 `+0.60pct`，可作为下一轮 P1 准入 gate。
5. `market_relative + G1` 市场状态分层已补：关键年份 `2022/2024/2025` 合计超额 `-142.55%`，强上涨月和强下跌月中位超额均为负，窄广度/高波动月份是 G1 相对 G0 的主要退化来源。

### 新数据

入口：

- `docs/alpha_factor_scout_2026-04-23_shareholder_smoke.md`
- `data/results/p1_full_backtest_g2_fundflow.json`
- `data/results/p1_full_backtest_g4_fundflow.json`
- `docs/newdata_quality_current_families_fund_flow-shareholder_flow_a_share_fund_flow_holder_a_share_shareholder_daily_a_share_daily_lag_30_width_100.md`
- `scripts/run_newdata_quality_checks.py`

结论：资金流和股东人数链路保留，但本轮质量 gate 均未通过；短期暂停新增 P2 研究预算，也不作为 P1 promotion 主线。

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

**新数据质量归档、`regression + rank_fusion + G0` 正式回测、`market_relative + G1` 断层诊断已经收口；下一步只做标签/proxy 的最小修复假设，先证明能修复正式月频 Top-20 边界和年份退化，再谈任何新 alpha 或 promotion。**
