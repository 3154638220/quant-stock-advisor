# 策略优化主计划

**文档角色**：当前唯一主计划（canonical）  
**更新时间**：`2026-04-27`  
**当前阶段**：P1 daily-proxy-first 实验循环，P2 数据质量修复，P4 结果身份和生产边界固化  
**归档入口**：`docs/plan-04-20.md` 仅保留 `2026-04-20` 当日执行记录，不再承担主计划职责

---

## 0. 当前结论和下一步

P1 不再围绕旧 light proxy 做解释和对齐。旧 proxy 已经证明自己会系统性高估失败候选，后续只保留为历史诊断字段，不再承担 admission、promotion 或正式回测触发职责。

当前主判断：

1. daily full-backtest-like proxy 在 5 个历史失败/对照样本上与正式 full backtest 方向一致，`daily_bt_like_proxy_excess` 全部为负。
2. daily proxy 相对正式 full backtest 的平均偏差约 `+0.60pct`，最大绝对偏差约 `2.44pct`；旧 full-like proxy 相对正式回测平均高估约 `+28.74pct`。
3. 因此 P1 主实验口径切换为 **daily-proxy-first**：先看 daily proxy，再决定是否补正式 full backtest。
4. H1 的两个最小月频可投资标签已落地并做完 `G0` smoke，但都被 daily proxy 截停：
   - `monthly_investable + regression + G0`：旧 light proxy 年化超额 `+10.60%`，full-like proxy `+4.53%`，daily backtest-like proxy `-31.18%`。
   - `monthly_investable_market_relative + regression + G0`：旧 light proxy 年化超额 `+20.02%`，full-like proxy `+18.12%`，daily backtest-like proxy `-43.42%`。
5. P1 runner 已固化 daily-proxy-first 输出：
   - summary 主 `result_type=daily_bt_like_proxy`
   - 旧 proxy 明确标记为 `legacy_proxy_decision_role=diagnostic_only`
   - 新增 `*_daily_proxy_leaderboard.csv`
   - 新增 `daily_proxy_first_status`
   - 新增 `pass_p1_daily_proxy_full_backtest_gate`
   - 保留 `*_topk_boundary.csv`
   - 保留 `*_daily_proxy_monthly_state.csv` 与 `*_daily_proxy_state_summary.csv`
6. 相关 P1 单测已覆盖新增标签、Top-K 边界、状态切片、daily-proxy-first 三档 gate 和 leaderboard，`pytest tests/test_p1_tree_groups.py` 当前为 `39 passed`。

下一步的唯一 P1 主线：

**直接做 daily-proxy-first leaderboard，不再单独做旧 proxy 和 daily proxy 的断层对齐项目。**

---

## 1. Daily-Proxy-First 纪律

### 1.1 口径角色

| 口径 | 新角色 | 是否能触发正式回测 | 是否能 promotion |
| --- | --- | --- | --- |
| 旧 `light_strategy_proxy` | legacy diagnostic only | 否 | 否 |
| `full_like_proxy` | legacy diagnostic only | 否 | 否 |
| `daily_bt_like_proxy` | P1 主准入指标 | 是 | 否 |
| 正式 `full_backtest` | promotion 必要条件 | 已是终层 | 是 |

旧 proxy 仍可用于解释“为什么某个候选看起来漂亮”，但不能用于决定候选是否进入下一层。

### 1.2 三档 Gate

P1 runner 当前默认使用三档 daily proxy gate：

| daily proxy 年化超额 | `daily_proxy_first_status` | 处理 |
| ---: | --- | --- |
| `< 0%` | `reject` | 停止，不补正式 full backtest |
| `0% ~ +3%` | `gray_zone` | 只归档诊断，默认不补正式 full backtest |
| `>= +3%` | `full_backtest_candidate` | 允许补正式 full backtest |

对应参数：

| 参数 | 默认值 | 作用 |
| --- | ---: | --- |
| `--daily-proxy-admission-threshold` | `0.0` | 硬停止线 |
| `--daily-proxy-full-backtest-threshold` | `0.03` | 正式 full backtest 安全边际线 |

`+3%` 来自当前校准最大绝对偏差约 `2.44pct` 后加一点安全边际。它不是 promotion 线，只是正式回测触发线。

### 1.3 不再主动做 Proxy 对齐工程

后续不再单独维护“对齐 daily proxy 和正式 full backtest”的项目。每次有候选通过 daily proxy 并补正式 full backtest 后，只记录健康检查字段：

1. `daily_bt_like_proxy_annualized_excess_vs_market`
2. `full_backtest_annualized_excess_vs_market`
3. `daily_minus_full`
4. `direction_match`

只有出现下面任一情况，才启动 proxy 对齐排查：

1. daily proxy 为正但正式 full backtest 为负。
2. `abs(daily_minus_full) > 5pct`。
3. 方向错配累计超过 2 次。

---

## 2. 当前研究基线

除非实验明确另行声明，后续 benchmark-first 对照使用：

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

这条基线是研究对照，不是生产策略。核心问题仍是上涨月参与不足，尤其 `2021 / 2025 / 2026` 明显落后。

已确认失败机制：

1. V3 基线不是下跌防守失败，而是上涨月份参与不足。上涨月中位超额 `-3.43%`，上涨月跑赢比例 `18.4%`；下跌月中位超额 `+2.96%`，下跌月跑赢比例 `80.8%`。
2. `G1 = G0 + weekly_kdj_*` 的 light proxy 曾较强，但同窗正式回测没有超过 `G0`，不能 promotion。
3. `rank + market_relative + G1` 的正式 full backtest 年化超额为 `-33.97%`，`2022/2024/2025` 合计退化显著。
4. `regression + rank_fusion + G0` 的 light proxy 为正，但正式 full backtest 年化超额为 `-20.87%`，rolling 和 slice OOS 中位超额均为负。
5. 资金流和股东人数仍是候选原料，不是主线；两者最新质量报告均未通过 gate。

---

## 3. 未来 10 个工作日执行队列

### 3.1 P1-A：Daily Proxy First Leaderboard

**目标**：完全废弃旧 proxy 的决策权，所有 P1 候选直接进入 daily proxy leaderboard。

第一轮只允许 `G0`，每次最多 3 个候选：

| 候选 | 目的 | 处理 |
| --- | --- | --- |
| `rank_fusion + regression + G0` | 旧失败基线，统一 leaderboard 口径 | 只作对照 |
| `monthly_investable + regression + G0` | H1 当前失败样本 | 已 daily reject，不扩组 |
| 新的最小 daily-proxy-first 机制候选 | 从上涨参与、边界或换手问题中选一个 | 只跑 `G0` |

候选不能靠旧 proxy 入围。旧 proxy 可以输出，但只作为 `legacy_*` 字段解释用。

标准命令模板：

```bash
python scripts/run_p1_tree_groups.py \
  --config config.yaml.backtest \
  --groups G0 \
  --history-start 2020-01-01 \
  --sample-start 2021-01-01 \
  --label-horizons 5,10,20 \
  --label-mode monthly_investable \
  --xgboost-objective regression \
  --proxy-horizon 5 \
  --rebalance-rule M \
  --top-k 20 \
  --val-frac 0.2 \
  --daily-proxy-admission-threshold 0.0 \
  --daily-proxy-full-backtest-threshold 0.03 \
  --run-full-backtest \
  --backtest-config config.yaml.backtest \
  --backtest-start 2021-01-01 \
  --backtest-top-k 20 \
  --backtest-max-turnover 1.0 \
  --backtest-portfolio-method equal_weight \
  --out-tag p1_daily_proxy_first_smoke
```

说明：`monthly_investable` 在这里仅作为命令结构示例。新的标签或目标口径必须先实现并注册，再替换 `--label-mode` 和 `--out-tag`。当前已失败的 H1 候选不能原样重跑后当成新证据。

验收输出：

1. `*_summary.csv`
2. `*_daily_proxy_leaderboard.csv`
3. `*_daily_proxy_monthly_state.csv`
4. `*_daily_proxy_state_summary.csv`
5. `*_topk_boundary.csv`
6. `*_bundle_manifest.csv`
7. JSON payload 中的 `daily_proxy_first_leaderboard`

判读顺序：

1. 先看 `daily_proxy_first_status`。
2. 再看 strong up / strong down / high vol / narrow breadth 状态切片。
3. 再看 Top-K boundary 和换入换出。
4. 只有 `full_backtest_candidate` 才补正式 full backtest。

### 3.2 P1-B：Daily Proxy 通过后的正式 Full Backtest

触发条件：

1. `daily_proxy_first_status=full_backtest_candidate`
2. `daily_bt_like_proxy_annualized_excess_vs_market >= +3%`
3. strong up 状态不明显劣化。
4. strong down 防守不明显劣化。
5. Top-K boundary 不明显反向。
6. 不依赖 `tree_score_auto_flipped=True` 才工作。

正式回测通过后仍不能直接 promotion，必须进入 P1-C。

### 3.3 P1-C：Promotion Review

promotion 必须同时满足：

1. 正式 full backtest 的 `annualized_excess_vs_market >= 0`。
2. 年度超额中位数 `>= 0`。
3. rolling OOS 和 slice OOS 的超额中位数不为负。
4. `2021 / 2025 / 2026` 不能明显更差，且必须解释改善或退化来源。
5. 强上涨月份的中位超额或跑赢比例必须相对当前基线改善。
6. 下跌月份防守不能明显劣化。
7. MaxDD 不明显劣于当前默认研究基线，除非超额改善足够大且可解释。
8. 换手上升必须有明确收益补偿。
9. 训练窗口、标签口径、执行口径、Top-K、成本、benchmark 必须同窗可追溯。

### 3.4 P2：新数据质量修复

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

恢复研究预算的标准：

1. `fund_flow` 的日线时间错位解释清楚，晚于日线最新日期的未匹配行不再污染质量判断。
2. `shareholder` 不再出现 `notice_date < end_date`，或该异常被证明为源数据字段语义问题并有明确 PIT 处理。
3. 质量报告明确写出“是否恢复研究预算”。
4. 恢复前不得跑 `G2/G3/G4` 主线网格。

### 3.5 P3：Interaction Alpha

**目标**：只从明确失败机制出发，构造少量条件表达。

允许保留但不立即扩网格的方向：

1. 低周线 J，但非高波动的反转修正。
2. 资金流背离叠加低换手的条件表达。
3. 弱近期动量但价格位置未破坏的上涨参与修正。

进入条件：

1. 对应机制必须来自 daily proxy leaderboard 的失败状态、Top-K 边界或换手诊断。
2. 每轮最多 3 个 interaction 假设。
3. 先跑 daily proxy smoke，不直接进 full backtest。
4. 如果只是放大已有反转簇，立即淘汰。

### 3.6 P4：研究和生产边界

**目标**：避免把研究层的“看起来不错”误写成生产默认。

已完成：

1. 根目录只保留当前 canonical 研究入口 `config.yaml.backtest` 和生产模板 `config.yaml.example`。
2. 历史 backtest 场景快照统一收纳到 `configs/backtests/`。
3. 已补充 `configs/README.md` 与 `configs/backtests/README.md`。
4. 配置加载器兼容旧命令：传入 `config.yaml.backtest.r7_s2_prefilter_off_universe_on` 这类旧快照名时，会自动解析到 `configs/backtests/`。
5. 覆盖宽度动态场景生成的新快照默认写入 `configs/backtests/`。
6. 根目录临时探针文件已收纳到 `tmp/`。

下一步：

1. 所有研究产物必须写清 `research_topic`、`research_config_id`、`output_stem`、`result_type` 和 `config_source`。
2. P1 bundle 必须能追溯训练窗口、label spec、feature group、cache、代码入口、执行口径、daily gate 状态。
3. 新增或更新固定报告模板，把 daily proxy leaderboard、状态切片、Top-K 边界、正式回测字段纳入同一页结论。
4. 日更推荐只接受已经 promotion 的配置；P1 当前没有任何方案满足。

---

## 4. 模块状态看板

| 模块 | 状态 | 当前判断 | 下一步 |
| --- | --- | --- | --- |
| P0 评估硬化 | 第一轮完成 | daily proxy 已能识别历史失败样本，新增状态切片、Top-K 边界和 leaderboard | 固化报告模板和字段 |
| P1 树模型 | daily-proxy-first 主线 | H1 两个 G0 候选 daily reject；旧 proxy 决策权已废弃 | 只在 G0 上做少量机制候选 |
| P2 新数据 | 维护 | fund flow 和 shareholder 质量 gate 未过 | 补日线、修 PIT，再复跑质量报告 |
| P3 Interaction Alpha | 诊断驱动 | 不再扩 weekly KDJ 或同质反转网格 | 等 daily proxy leaderboard 给机制后，每轮最多 3 个假设 |
| P4 生产边界 | 第一轮目录整理完成 | 历史研究配置已收纳到 `configs/backtests/`，旧快照名保持兼容 | 继续固化 bundle registry、报告字段和 promotion 写回规则 |

---

## 5. P1 分组状态

| 分组 | 定义 | 当前判断 | 下一动作 |
| --- | --- | --- | --- |
| `G0` | baseline technical | 当前唯一允许继续做 daily-proxy-first 标签/目标验证的对象 | 每轮最多 3 个候选 |
| `G1` | `G0 + weekly_kdj_*` | light proxy 好，但正式回测没有兑现 | 只有 G0 daily proxy 达到 full backtest candidate 后再测 |
| `G2` | `G0 + fund_flow_*` | full backtest 劣化，且 fund flow 质量未过 | 暂停 |
| `G3` | `G0 + shareholder_*` | 无增量，且 PIT 质量未过 | 暂停 |
| `G4` | `G0 + weekly_kdj_* + fund_flow_*` | 未超过 `G1`，数据质量也不足 | 暂停 |
| `G5` | `G0 + weekly_kdj_* + weekly_kdj_interaction_*` | light proxy 未超过 `G1` | 仅保留为机制诊断 |
| `G6` | `G0 + weekly_kdj_interaction_*` | light proxy 未超过 `G1` | 仅保留为机制诊断 |

P1 禁区：

1. 不继续扩 `weekly_kdj` interaction 网格。
2. 不把 `weekly_kdj_*` 写回默认 `composite_extended`。
3. 不重复跑 `G2/G4` 直拼 fund flow。
4. 不用旧 proxy、跨窗口、跨标签口径、跨缓存版本的结果做 promotion 判断。
5. 不把 `signal_diagnostic`、旧 light proxy 或 full-like proxy 当成正式回测结论。
6. 不把 H1 当前两个失败候选推进到 `G1/G2/G3/G4`。
7. 不为了让 daily proxy 变好而打开大网格。

---

## 6. 冻结结论

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

## 7. 证据索引

### 7.1 默认基线和上涨参与不足

入口：`docs/benchmark_gap_diagnostics_2026-04-20_v3.md`

关键结论：

1. 当前弱势主要是上涨月份参与不足，不是下跌防守失败。
2. 持仓偏更大市值、更高成交额、低波动、近期相对动量偏弱，更像稳定票。
3. `21-40` 桶平均前向收益接近 `01-20`，但中位数仍不稳定，适合做边界诊断，不适合重启 Top-K 网格。

### 7.2 P1 标签、目标和 proxy 断层

入口：

- `docs/p1_label_objective_experiment_2026-04-26.md`
- `docs/p1_marketrel_g1_gap_diagnostics_2026-04-26.md`
- `docs/p1_full_like_proxy_calibration_smoke_2026-04-27.md`
- `docs/p1_proxy_calibration_history_2026-04-27.md`
- `docs/p1_proxy_calibration_history_h20_2026-04-27.md`
- `docs/p1_daily_bt_like_proxy_calibration_2026-04-27.md`
- `docs/p1_marketrel_state_diagnostics_2026-04-27.md`
- `docs/p1_monthly_investable_label_smoke_2026-04-27.md`
- `docs/p1_daily_proxy_first_2026-04-27.md`

关键结论：

1. `rank + market_relative + G1` light proxy 最强，但 full backtest 严重失败。
2. `regression + rank_fusion + G0` light proxy 为正，但正式回测仍显著负超额。
3. daily full-backtest-like proxy 在 5 个历史失败样本上全部为负，和正式 full backtest 方向一致。
4. `monthly_investable` 和 `monthly_investable_market_relative` 的 `G0` smoke 均未通过 daily proxy gate。
5. 旧 proxy 已降级为 legacy diagnostic，不再承担决策权。

### 7.3 Weekly KDJ

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

### 7.4 新数据

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

## 8. 产出规范

每轮新实验至少保留：

1. 配置或命名身份：
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
3. 一页结论文档，必须写清：
   - 只改了什么
   - 对默认研究基线的变化
   - 对 `market_ew_proxy` 的变化
   - 是否改变主计划
   - 是否允许进入下一层验证

每份 P1 输出还必须显式写清：

1. `p1_experiment_mode=daily_proxy_first`
2. `legacy_proxy_decision_role=diagnostic_only`
3. `benchmark_symbol`
4. `top_k`
5. `rebalance_rule`
6. `portfolio_method`
7. `execution_mode`
8. `prefilter`
9. `universe_filter`
10. 成本假设
11. cache 路径与 schema/version
12. label spec
13. feature group
14. `daily_proxy_first_status`
15. `daily_proxy_admission_threshold`
16. `daily_proxy_full_backtest_threshold`
17. daily proxy leaderboard 路径
18. Top-K 边界诊断路径
19. 市场状态切片路径
20. 若发生 fallback，必须记录 fallback 原因
21. 若使用历史快照，优先记录解析后的 `configs/backtests/...` 路径；旧短名只作为兼容入口保留

---

## 9. 当前不该做的事情

1. 不继续做 `Top-K`、缓冲带、分层等权的小参数网格。
2. 不把 `weekly_kdj_*` 写回默认 `composite_extended`。
3. 不继续为股东人数做单因子或线性权重主线。
4. 不把 `fund_flow_*` 直接拼进 P1 主线重复跑 `G2/G4`。
5. 不把 `signal_diagnostic`、旧 `light_strategy_proxy` 或 full-like proxy 当成正式回测结论。
6. 不用跨窗口、跨标签口径、跨缓存版本的结果做 promotion 判断。
7. 不在没有明确失败机制的情况下扩 expression 网格。
8. 不把任何未 promotion 的研究配置写入生产日更推荐。
9. 不再把历史研究配置快照平铺到项目根目录。
10. 不把 H1 当前两个已失败的月频标签候选继续扩到 `G1/G2/G3/G4`。
11. 不再为了旧 proxy 做单独断层对齐项目。
12. 不把 daily proxy 当成 promotion 终点。

---

## 10. 一句话路线图

未来一个阶段的最优推进方式是：

**把 P1 全面切到 daily-proxy-first：旧 proxy 只做 legacy diagnostic，daily proxy 负责候选准入和正式回测触发，正式 full backtest 仍是 promotion 必要条件；同时修复 fund flow 和 shareholder 的质量 gate，并把研究产物身份、leaderboard、状态切片、Top-K 边界和 promotion 写回规则固化。**
