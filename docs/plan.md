# 策略优化主计划

**文档角色**：当前唯一主计划（canonical）  
**更新时间**：`2026-04-27`  
**当前阶段**：路线重构；停止 P1 近邻标签微调，转向 `regime-aware + upside sleeve + boundary-aware`  
**归档入口**：`docs/plan-04-20.md` 仅保留 `2026-04-20` 当日执行记录，不再承担主计划职责

---

## 0. 总判断

当前项目没有进入“整个系统无路可走”的瓶颈，但 **P1 原主线已经进入局部死胡同**。

这里的 P1 原主线指：

`G0 技术特征` + `XGBoost rank/regression` + `forward return / rank_fusion / market_relative / up_capture / path-quality 标签变体` + `月频 Top-20`

这个方向已经不再值得继续做近邻实验。后续不再把“换一个标签形状、调一个 horizon 权重、换一个 proxy horizon、扩一点 Top-K 或 buffer”作为主研究预算。

新的主线是：

1. **保留 daily-proxy-first 作为准入护栏**：它继续决定候选是否值得补正式 full backtest。
2. **停止 P1 标签微调线**：现有 G0 标签候选已经连续 daily reject。
3. **把 S2 防守能力和上涨参与能力拆开建模**：S2/vol_to_turnover 更像 defensive sleeve，不应被强行要求在上涨扩散期追上市场。
4. **新增 upside sleeve**：专门针对 strong up / broad breadth / 成交额扩张环境构造进攻型候选。
5. **新增 boundary-aware 调仓目标**：直接优化 `switch-in` 是否优于 `switch-out`、`Top-20` 是否显著优于 `21-60`，而不是继续优化普通截面收益标签。

一句话：**从“单模型 Top-20 选股”切换为“状态识别 + 防守/进攻双袖套 + 换入边界控制”的研究框架。**

---

## 1. 当前证据

### 1.1 评估口径已经基本修正

旧 `light_strategy_proxy` 和 `full_like_proxy` 已经证明会系统性高估失败候选，后续只保留为 historical diagnostic。

当前准入指标为：

`daily_bt_like_proxy_annualized_excess_vs_market`

校准结果：

| 项目 | 结论 |
| --- | --- |
| 校准样本 | 5 个历史失败/对照样本 |
| daily proxy 方向 | 与正式 full backtest 方向一致 |
| daily proxy 正超额样本 | `0/5` |
| daily proxy 相对正式 full backtest 平均偏差 | `+0.60pct` |
| daily proxy 最大绝对偏差 | `2.44pct` |
| 旧 full-like proxy 相对正式回测平均高估 | `+28.74pct` |

证据入口：`docs/p1_daily_bt_like_proxy_calibration_2026-04-27.md`

### 1.2 P1 近邻标签候选连续失败

| 候选 | 旧 proxy 年化超额 | daily proxy 年化超额 | 判定 |
| --- | ---: | ---: | --- |
| `monthly_investable + regression + G0` | `+10.60%` | `-31.18%` | reject |
| `monthly_investable_market_relative + regression + G0` | `+20.02%` | `-43.42%` | reject |
| `up_capture_market_relative + regression + G0` | `+19.67%` | `-39.21%` | reject |
| `monthly_investable_up_capture_market_relative + regression + G0` | `+18.72%` | `-47.48%` | reject |
| `rank_fusion long-horizon weighted + regression + G0` | `+16.83%` | `-29.52%` | reject |
| `top_bucket_rank_fusion + regression + G0` | `+15.76%` | `-28.09%` | reject |
| `calmar path-quality + rank_fusion + regression + G0` | `+14.54%` | `-22.99%` | reject |

关键含义：

1. 失败不是单个标签定义偶然不合适。
2. `val_rank_ic` 和旧 proxy 经常好看，但不能转化为月频可交易收益。
3. G0 当前特征能学到某种截面排序关系，但这类关系和正式执行链路错位。
4. 继续做 `sharpe/truncate/horizon_weight/top_bucket` 的近邻变体，信息增量很低。

证据入口：

- `docs/p1_monthly_investable_label_smoke_2026-04-27.md`
- `docs/p1_up_capture_market_relative_g0_smoke_2026-04-27.md`
- `docs/p1_monthly_investable_up_capture_g0_smoke_2026-04-27.md`
- `docs/p1_rank_fusion_long_horizon_g0_smoke_2026-04-27.md`
- `docs/p1_top_bucket_rank_fusion_g0_smoke_2026-04-27.md`
- `data/results/p1_calmar_path_quality_g0_smoke_rb_m_top20_lh_5-10-20_px_5_val20_lbl_rank_fusion_lt_calmar_obj_regression_20260427_132502_report.md`

### 1.3 真实失败机制：上涨参与不足

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
| `benchmark_symbol` | `market_ew_proxy` |

已确认：

1. 基线不是下跌防守失败，而是上涨月份参与不足。
2. 上涨月中位超额 `-3.43%`，上涨月跑赢比例 `18.4%`。
3. 下跌月中位超额 `+2.96%`，下跌月跑赢比例 `80.8%`。
4. 持仓偏更大市值、更高成交额、低波动、近期相对动量偏弱，更像稳定票。
5. `21-40` 桶平均前向收益接近 `01-20`，但中位数不稳定，不能简单扩 Top-K。

证据入口：`docs/benchmark_gap_diagnostics_2026-04-20_v3.md`

### 1.4 换入边界反复报警

多轮候选都出现类似结构：

1. `Top-K minus next bucket` 薄弱或转负。
2. `switch-in minus switch-out` 经常为负。
3. 换手高时收益没有补偿，换手降下来时收益也没有兑现。

这说明下一步不能只做 “Top-K 参数网格” 或 “换手约束网格”。问题应改写为：

> 新换入的股票，是否真的比被换出的股票更值得买？

---

## 2. 新路线图

### R0：评估护栏继续保留

**目标**：保证后续任何候选都先过可交易口径，不再被旧 proxy 误导。

| 口径 | 角色 | 是否能触发正式 full backtest | 是否能 promotion |
| --- | --- | --- | --- |
| 旧 `light_strategy_proxy` | legacy diagnostic only | 否 | 否 |
| `full_like_proxy` | legacy diagnostic only | 否 | 否 |
| `daily_bt_like_proxy` | admission gate | 是 | 否 |
| 正式 `full_backtest` | promotion 必要条件 | 是 | 是 |

默认 gate：

| daily proxy 年化超额 | 状态 | 处理 |
| ---: | --- | --- |
| `< 0%` | `reject` | 停止，不补正式 full backtest |
| `0% ~ +3%` | `gray_zone` | 只归档诊断，默认不补正式 full backtest |
| `>= +3%` | `full_backtest_candidate` | 允许补正式 full backtest |

只有出现以下情况才重启 proxy 对齐排查：

1. daily proxy 为正但正式 full backtest 为负。
2. `abs(daily_minus_full) > 5pct`。
3. 方向错配累计超过 2 次。

### R1：Strong-Up 失败归因

**目标**：先解释上涨月到底漏掉了什么，再决定新增哪些特征或规则。

本阶段不训练新模型，只做归因。

必须输出：

1. `2021 / 2025 / 2026` 的上涨月持仓对比。
2. `Top-20 / 21-40 / 41-60 / switch-in / switch-out / benchmark` 的暴露对比。
3. 暴露维度至少包括：
   - 近 20/60 日相对强度
   - 成交额扩张
   - 换手扩张
   - realized volatility
   - price position
   - log market cap
   - 行业或概念暴露，若映射可用
   - 涨停、跳空、尾盘强度等可交易路径特征
4. strong up / mild up / strong down / narrow breadth / wide breadth 状态切片。
5. 归因结论必须明确写出：当前策略在上涨月是缺少弹性、错过行业扩散、换入太慢，还是被防守因子压制。

验收标准：

1. 归因结果能解释至少 `2021 / 2025 / 2026` 中两个关键年份的主要落后来源。
2. 归因必须给出 1 到 3 个可验证机制，不能只写“上涨参与不足”。
3. 没有归因结论前，不再新增 P1 标签候选。

建议产物：

```text
docs/p1_strong_up_failure_attribution_2026-04-27.md
data/results/p1_strong_up_failure_attribution_2026-04-27_*.csv
```

### R2：Regime-Aware 双袖套

**目标**：不要让一个防守型信号同时承担防守和进攻任务。

组合拆成两个 sleeve：

| sleeve | 目的 | 候选信号 |
| --- | --- | --- |
| `defensive_sleeve` | 下跌月防守和低波动稳定收益 | `S2 = vol_to_turnover`，保留现有强项 |
| `upside_sleeve` | strong up / wide breadth 环境提高上涨参与 | 相对强度、成交额扩张、行业扩散、低拥挤动量、可交易突破 |

第一版不训练复杂模型，先做可解释 composite：

1. 市场状态为 strong up 或 wide breadth 时，提高 `upside_sleeve` 权重。
2. 市场状态为 strong down 或 narrow breadth 时，提高 `defensive_sleeve` 权重。
3. 中性状态保持保守，不强行追涨。

候选必须和默认研究基线同窗比较：

| 指标 | 最低要求 |
| --- | --- |
| daily proxy 年化超额 | `>= 0%` 才能继续 |
| strong up 中位超额 | 相对 S2 明显改善 |
| strong up 跑赢比例 | 相对 S2 明显改善 |
| strong down 中位超额 | 不明显劣化 |
| 年度关键期 | `2021 / 2025 / 2026` 至少两个年份改善 |
| 换手 | 上升必须有收益补偿 |

进入 full backtest 的要求仍然是 daily proxy `>= +3%`。

### R3：Boundary-Aware 调仓目标

**目标**：把模型目标从“预测普通截面收益”改成“判断换入是否值得”。

当前证据表明，`switch-in minus switch-out` 是比普通 Rank IC 更接近真实失败的指标。R3 只在 R1/R2 证明存在可利用机制后启动。

候选目标：

1. `switch_quality_label`：本期新进 Top-K 在下一持有期是否跑赢被换出的股票。
2. `top_vs_next_label`：Top-20 是否显著跑赢 21-60，而不只是 rank 略高。
3. `hold_vs_replace_label`：继续持有旧仓是否优于替换。
4. `state_conditional_label`：仅在 strong up / wide breadth 状态下训练进攻边界，在弱市保持 defensive sleeve。

可选算法：

| 算法 | 适用位置 | 备注 |
| --- | --- | --- |
| Logistic / Calibrated classifier | 换入是否通过 | 第一优先，便于解释 |
| LambdaRank / pairwise ranker | Top vs next / switch pair | 只在标签构造稳定后使用 |
| XGBoost regression | 继续作为基线对照 | 不能再作为唯一主线 |
| 深度序列模型 | 只做小样本对照 | 不作为下一阶段主线 |

验收标准：

1. `switch_in_minus_out >= 0`，且不是靠极少数月份撑起来。
2. `topk_minus_next >= 0`，并且 strong up 状态不为负。
3. daily proxy 至少进入 gray zone，才允许继续迭代。
4. 若 `val_rank_ic` 好但 daily proxy 或 switch quality 差，直接淘汰。

### R4：新数据质量修复

**目标**：让 fund flow 和 shareholder 从“已接入”变成“可解释、可研究”。

| 家族 | 当前状态 | 下一步 |
| --- | --- | --- |
| `fund_flow` | 暂停新增研究预算；资金流日期晚于日线，匹配质量未过 gate | 先补齐日线到资金流最新日期，再复跑质量报告 |
| `shareholder` | PIT 异常已保守修复，质量复跑 `ok=True`，但 alpha 证据弱 | 只保留低优先级观察，不进主线 |

恢复研究预算的条件：

1. `fund_flow` 的日线时间错位解释清楚。
2. 晚于日线最新日期的未匹配行不再污染质量判断。
3. 质量报告明确写出是否恢复研究预算。
4. 恢复前不得跑 `G2/G3/G4` 主线网格。

复核命令：

```bash
python scripts/run_newdata_quality_checks.py \
  --config config.yaml.backtest \
  --families fund_flow,shareholder \
  --output-prefix newdata_quality_current
```

### R5：生产边界

**目标**：研究配置不能误进入日更推荐。

规则：

1. 日更推荐只接受已经 promotion 的配置。
2. daily proxy 不是 promotion 终点。
3. 任何研究配置写入生产前，必须有正式 full backtest、年度切片、rolling OOS、slice OOS、状态切片和 Top-K 边界报告。
4. 当前没有任何 P1 树模型候选满足 promotion。

---

## 3. 未来 10 个工作日执行队列

### Day 1-2：重做失败归因

执行 R1，不训练新模型。

输出：

1. strong-up 失败归因报告。
2. `Top-20 / 21-40 / switch-in / switch-out` 暴露表。
3. 能解释关键落后年份的 1 到 3 个机制假设。

通过条件：

1. 明确指出 S2 在上涨月漏掉的候选类型。
2. 明确指出现有 G0/XGBoost 候选为何换入质量偏弱。
3. 给 R2 upside sleeve 至少 1 个可测试特征组合。

### Day 3-5：构造第一版 Upside Sleeve

先做规则或线性 composite，不训练复杂模型。

候选数量限制：每轮最多 3 个。

优先候选：

1. `relative_strength + amount_expansion`
2. `industry_breadth + relative_strength`
3. `tradable_breakout + turnover_expansion`

验收：

1. daily proxy 不为负。
2. strong up 中位超额改善。
3. strong down 不明显劣化。
4. 换手上升有收益补偿。

### Day 6-7：双袖套权重规则

在 `defensive_sleeve` 和 `upside_sleeve` 之间做状态权重。

限制：

1. 不做大网格。
2. 状态权重最多测试 3 组。
3. 只允许基于预先定义的市场状态，不允许事后按收益调状态。

候选权重示例：

| 状态 | defensive | upside |
| --- | ---: | ---: |
| strong up / wide breadth | `40%` | `60%` |
| neutral | `70%` | `30%` |
| strong down / narrow breadth | `90%` | `10%` |

### Day 8-10：Boundary-Aware 最小模型

只有 R2 至少进入 gray zone 才启动。

第一版只做分类或 pairwise，不做深度模型。

目标：

1. 预测 `switch-in` 是否跑赢 `switch-out`。
2. 预测 `Top-20` 是否显著跑赢 `21-60`。
3. 给组合层一个 `replace / hold` 门控。

通过条件：

1. `switch_in_minus_out` 转正。
2. daily proxy 不劣于 R2。
3. strong up 改善不消失。

---

## 4. 停止清单

以下方向冻结，除非出现新数据、代码修复或明确机制证据：

1. 不继续做 `monthly_investable`、`up_capture`、`rank_fusion`、`top_bucket_rank_fusion`、`calmar/sharpe/truncate` 的近邻标签微调。
2. 不把旧 `light_strategy_proxy` 或 `full_like_proxy` 当成入围证据。
3. 不用 `val_rank_ic` 替代 daily proxy 或正式 full backtest。
4. 不继续做 `Top-K`、缓冲带、分层等权的小参数网格。
5. 不把 `weekly_kdj_*` 写回默认 `composite_extended`。
6. 不扩 `weekly_kdj` interaction 网格。
7. 不重复跑 `G2/G4` 直拼 fund flow。
8. 不继续为 shareholder 做单因子或线性权重主线。
9. 不在没有 R1 归因结论的情况下扩 expression 网格。
10. 不把任何未 promotion 的研究配置写入生产日更推荐。
11. 不把 historical config 快照重新平铺到项目根目录。
12. 不把深度模型作为下一阶段主线；它只能作为小对照实验。

---

## 5. 保留清单

这些能力继续保留：

1. `daily-proxy-first` runner 和三档 gate。
2. `*_daily_proxy_leaderboard.csv`。
3. `*_daily_proxy_monthly_state.csv` 和 `*_daily_proxy_state_summary.csv`。
4. `*_topk_boundary.csv`。
5. P1 fixed report template 中的身份、label spec、组合口径、成本和 cache 字段。
6. S2/vol_to_turnover 作为 defensive sleeve 候选。
7. `G0` 技术特征作为 baseline，不再作为唯一 alpha 来源。
8. fund flow / shareholder 数据链路，但必须先过质量 gate。
9. `risk_parity` 作为诊断工具，不回主线。
10. `config.yaml.backtest` 作为 canonical 研究入口。

---

## 6. Promotion 标准

候选进入 production 前必须同时满足：

1. 正式 full backtest 的 `annualized_excess_vs_market >= 0`。
2. 年度超额中位数 `>= 0`。
3. rolling OOS 超额中位数不为负。
4. slice OOS 超额中位数不为负。
5. `2021 / 2025 / 2026` 不能明显更差。
6. strong up 中位超额或跑赢比例必须相对当前 S2 基线改善。
7. strong down 防守不能明显劣化。
8. MaxDD 不明显劣于当前默认研究基线，除非超额改善足够大且可解释。
9. 换手上升必须有明确收益补偿。
10. `switch_in_minus_out` 不能显著为负。
11. `topk_minus_next` 不能明显反向。
12. 训练窗口、标签口径、执行口径、Top-K、成本、benchmark 必须同窗可追溯。

---

## 7. 产出规范

每轮新实验至少保留：

1. 配置身份：
   - `research_topic`
   - `research_config_id`
   - `output_stem`
   - `result_type`
   - `config_source`
2. 结果文件：
   - summary CSV
   - detail CSV 或 period detail
   - JSON
   - 必要时的 manifest
3. 一页结论文档：
   - 只改了什么
   - 对 S2 defensive baseline 的变化
   - 对 `market_ew_proxy` 的变化
   - 对 strong up / strong down 的变化
   - 对 `switch_in_minus_out` 和 `topk_minus_next` 的变化
   - 是否允许进入下一层验证

每份 P1/R2/R3 输出还必须显式写清：

1. `p1_experiment_mode` 或新的 route id。
2. `legacy_proxy_decision_role=diagnostic_only`。
3. `benchmark_symbol`。
4. `top_k`。
5. `rebalance_rule`。
6. `portfolio_method`。
7. `execution_mode`。
8. `prefilter`。
9. `universe_filter`。
10. 成本假设。
11. cache 路径与 schema/version。
12. feature spec。
13. label spec 或 rule spec。
14. `daily_proxy_first_status`。
15. daily proxy leaderboard 路径。
16. Top-K 边界诊断路径。
17. 市场状态切片路径。
18. 若发生 fallback，必须记录 fallback 原因。
19. 若使用历史快照，优先记录解析后的 `configs/backtests/...` 路径；旧短名只作为兼容入口保留。

---

## 8. 证据索引

### 8.1 基线和上涨参与不足

- `docs/benchmark_gap_diagnostics_2026-04-20_v3.md`
- `docs/score_ablation_2026-04-19.md`
- `docs/alpha_factor_scout_2026-04-20.md`
- `docs/factor_admission_validation_2026-04-20_next.md`

### 8.2 P1 失败和 proxy 校准

- `docs/p1_label_objective_experiment_2026-04-26.md`
- `docs/p1_failure_diagnostics_2026-04-26.md`
- `docs/p1_marketrel_g1_gap_diagnostics_2026-04-26.md`
- `docs/p1_marketrel_state_diagnostics_2026-04-27.md`
- `docs/p1_proxy_calibration_history_2026-04-27.md`
- `docs/p1_proxy_calibration_history_h20_2026-04-27.md`
- `docs/p1_daily_bt_like_proxy_calibration_2026-04-27.md`
- `docs/p1_daily_proxy_first_2026-04-27.md`

### 8.3 已失败的 P1 标签候选

- `docs/p1_monthly_investable_label_smoke_2026-04-27.md`
- `docs/p1_up_capture_market_relative_g0_smoke_2026-04-27.md`
- `docs/p1_monthly_investable_up_capture_g0_smoke_2026-04-27.md`
- `docs/p1_rank_fusion_long_horizon_g0_smoke_2026-04-27.md`
- `docs/p1_top_bucket_rank_fusion_g0_smoke_2026-04-27.md`
- `data/results/p1_calmar_path_quality_g0_smoke_rb_m_top20_lh_5-10-20_px_5_val20_lbl_rank_fusion_lt_calmar_obj_regression_20260427_132502_report.md`

### 8.4 Weekly KDJ

- `docs/alpha_factor_scout_2026-04-24_weekly_kdj.md`
- `docs/p1_weekly_interaction_ab_2026-04-26.md`
- `docs/p1_rank_direction_rerun_2026-04-26.md`
- `docs/p1_rankfix_same_window_full_backtest_2026-04-26.md`

### 8.5 新数据

- `docs/newdata_quality_current_families_fund_flow-shareholder_flow_a_share_fund_flow_holder_a_share_shareholder_daily_a_share_daily_lag_30_width_100.md`
- `docs/alpha_factor_scout_2026-04-23_shareholder_smoke.md`
- `data/results/p1_full_backtest_g2_fundflow.json`
- `data/results/p1_full_backtest_g4_fundflow.json`
- `scripts/run_newdata_quality_checks.py`

---

## 9. 当前一句话路线

**停止 P1 标签近邻微调；保留 daily proxy 作为护栏；先做 strong-up 失败归因，再构造 regime-aware 的 defensive/upside 双袖套，最后只在机制成立时引入 boundary-aware 模型来控制换入质量。**
