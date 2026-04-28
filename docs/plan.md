# 策略优化主计划

**文档角色**：当前唯一主计划（canonical）  
**更新时间**：`2026-04-28`  
**当前阶段**：评估/执行口径硬化，随后重启 `upside input`，最后才进入 `boundary-aware replacement`  
**生产状态**：当前没有任何 P1/R2/R3 研究候选满足 promotion，不允许写入日更推荐默认配置  
**归档入口**：`docs/plan-04-20.md` 仅保留 `2026-04-20` 当日执行记录，不再承担主计划职责

---

## 0. 总判断

当前项目不是“系统无路可走”，而是已经走到一个更清楚的分岔点：

1. 研究基础设施、回测链路、daily-proxy-first 护栏、诊断报告和产物归档已经明显成熟。
2. P1 原主线已经被充分证伪，不应继续投入“标签近邻微调”预算。
3. S2 / `vol_to_turnover` 有明确防守价值，但不适合作为单一主策略承担上涨扩散期收益。
4. R2 第一版 pure upside sleeve 和 dual sleeve 均失败，不能 promotion，也不应继续简单加大 upside 权重。
5. 在继续做新模型前，必须先修正若干会影响 upside 结论解释的评估/执行口径风险。

一句话路线：

> 先把评估和执行契约修硬；再重做更可交易的 upside 输入；然后只在边界替换层使用 upside；最后才考虑 boundary-aware 模型。

新的主线不是：

`单模型 Top-20 选股`

也不是：

`防守篮子 + 进攻篮子按状态粗暴拼接`

而是：

`评估契约一致 + defensive core + tradable upside replacement + switch boundary control`

---

## 1. 当前证据与诊断结论

### 1.1 项目的主要优点

项目当前最强的部分是研究基础设施，而不是某个 alpha 候选。

已经形成的有效能力：

1. **daily-proxy-first 护栏**  
   旧 `light_strategy_proxy` 和 `full_like_proxy` 已经降级为 legacy diagnostic。当前准入指标改为 `daily_bt_like_proxy_annualized_excess_vs_market`，并且历史校准显示它与正式 full backtest 方向一致。

2. **证据链完整**  
   每轮实验能够产出 leaderboard、monthly state、regime slice、breadth slice、switch quality、Top-K boundary、JSON summary 和一页结论文档。

3. **研究纪律明显变好**  
   现在已经明确“不用 val_rank_ic 替代可交易收益”“不把旧 proxy 当 promotion 证据”“不把未 promotion 研究配置写进生产”。

4. **失败归因能力已经建立**  
   R1 不只是说“上涨参与不足”，而是拆出了市值/低波/涨停弹性、成交额/换手扩张、switch-in 边界失效三类机制。

5. **工程覆盖尚可**  
   当前 `pytest -q` 通过。测试覆盖数据质量、PIT、回测引擎、交易成本、组合约束、P1/P2 诊断、因子管线等核心部位。

这些能力都要保留。后续失败时也必须继续按同样证据链归档，而不是回到只看单条收益曲线。

### 1.2 P1 原主线已经停止

P1 原主线指：

`G0 技术特征` + `XGBoost rank/regression` + `forward return / rank_fusion / market_relative / up_capture / path-quality 标签变体` + `月频 Top-20`

已经连续失败：

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

1. 失败不是某一个 label 写法不合适。
2. `val_rank_ic` 能学到某种截面排序，但不能稳定转化为月频 Top-20 可交易收益。
3. 继续调 `horizon_weight`、`sharpe/calmar/truncate`、`top_bucket`、`proxy_horizon` 的信息增量很低。
4. P1/G0 可以作为诊断基线和对照，但不再是主研究预算。

### 1.3 S2 的真实定位是 defensive sleeve

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

R1 归因显示：

| 切片 | 月数 | 中位超额 | 正超额比例 | 结论 |
| --- | ---: | ---: | ---: | --- |
| `strong_down` | 13 | `+7.65%` | `92.3%` | 防守能力明确 |
| `narrow_breadth` | 19 | `+4.01%` | `78.9%` | 弱市/窄市有效 |
| `strong_up` | 13 | `-6.26%` | `15.4%` | 上涨参与不足 |
| `wide_breadth` | 19 | `-5.27%` | `10.5%` | 扩散行情明显落后 |

因此，S2 不应被要求单独追上市场。它应该是防守核心，而不是完整主策略。

### 1.4 R1 已确认三个可验证机制

R1 输出：

- `docs/p1_strong_up_failure_attribution_2026-04-27.md`
- `scripts/run_p1_strong_up_attribution.py`
- `data/results/p1_strong_up_failure_attribution_2026-04-27_*.csv`

机制 1：strong-up 持仓偏稳定大盘、低波、无涨停弹性。

strong-up 状态下 top20 active 暴露：

| 维度 | active 值 | 解读 |
| --- | ---: | --- |
| `active_log_market_cap` | `+5.00` | 极端偏大盘 |
| `active_realized_vol` | `-0.24` | 低波 |
| `active_limit_up_hits_20d` | `-0.45` | 极少参与涨停集群 |
| `active_rel_strength_20d` | `-0.065` | 近 20 日相对强度偏弱 |

机制 2：S2 排序与成交额/换手扩张方向反向。

strong-up 状态下：

| 维度 | top20 active | 21-40 active | 41-60 active |
| --- | ---: | ---: | ---: |
| `active_amount_expansion_5_60` | `-0.062` | `-0.034` | `-0.009` |
| `active_turnover_expansion_5_60` | `-0.046` | `-0.016` | `+0.003` |

机制 3：switch-in 边界在 strong-up 系统性失效。

| regime | `switch_in_minus_out` | `switch_in_winning_share` |
| --- | ---: | ---: |
| `strong_down` | `+2.93%` | `0.500` |
| `mild_down` | `+2.07%` | `0.600` |
| `neutral` | `+2.43%` | `0.667` |
| `mild_up` | `+1.42%` | `0.333` |
| `strong_up` | `-1.86%` | `0.375` |

这说明真正该攻击的不是普通 Rank IC，而是：

> 新换入股票是否真的比被换出股票更值得买？

### 1.5 R2 第一版已经失败

R2 Day 3-5 测试 pure upside sleeve：

| 候选 | daily proxy 年化超额 | strong-up 中位超额 | strong-down 中位超额 | 平均半 L1 换手 | 判定 |
| --- | ---: | ---: | ---: | ---: | --- |
| `BASELINE_S2` | `-11.3%` | `-6.26%` | `+7.65%` | `0.124` | baseline |
| `UPSIDE_A = rel_strength_20d + amount_expansion_5_60` | `-85.9%` | `-17.1%` | `-15.6%` | `0.967` | reject |
| `UPSIDE_B = rel_strength_60d + turnover_expansion_5_60` | `-85.7%` | `-17.8%` | `-14.8%` | `0.952` | reject |
| `UPSIDE_C = limit_up_hits_20d + tail_strength_20d` | `-86.6%` | `-15.3%` | `-13.8%` | `0.905` | reject |

唯一边际正向信号：

| candidate | strong-up `mean_switch_in_minus_out` | strong-up `switch_in_winning_share` |
| --- | ---: | ---: |
| `BASELINE_S2` | `-0.0128` | `0.500` |
| `UPSIDE_A` | `-0.0086` | `0.417` |
| `UPSIDE_B` | `-0.0104` | `0.500` |
| `UPSIDE_C` | `+0.0029` | `0.583` |

解释：

1. pure upside sleeve 不能独立担当主组合。
2. 粗糙强势特征会带来接近全换仓，成本和路径风险吞噬收益。
3. UPSIDE_C 的 switch-in 边界有一点信息，但组合收益没有兑现。
4. 它更适合成为“替换候选”，不适合成为独立篮子。

### 1.6 R2 双袖套也失败

R2 Day 6-7 测试 defensive/upside 状态权重：

| 候选 | daily proxy 年化超额 | strong-up 中位超额 | strong-down 中位超额 | 平均半 L1 换手 | 判定 |
| --- | ---: | ---: | ---: | ---: | --- |
| `BASELINE_S2` | `-12.8%` | `-6.52%` | `+6.72%` | `0.057` | baseline / reject |
| `DUAL_V1_80_20_TRIGGER_ONLY` | `-23.0%` | `-8.59%` | `+2.07%` | `0.172` | reject |
| `DUAL_V2_60_40_MILD_85_15` | `-39.1%` | `-8.74%` | `-0.42%` | `0.276` | reject |
| `DUAL_V3_40_60_MILD_70_30_NEUTRAL_90_10` | `-52.0%` | `-9.73%` | `-3.92%` | `0.434` | reject |

关键含义：

1. 保守 80/20 也没有改善 strong-up。
2. 加大 upside 权重后，daily proxy、strong-down、防守能力和换手单调劣化。
3. 当前 upside 输入层不可用，不应启动 R3 最小模型。
4. 需要回到 upside feature 和组合表达，而不是继续调 sleeve 权重。

---

## 2. 必须先处理的评估/执行口径风险

以下问题会影响 upside 结论解释。它们不一定推翻已有失败结论，但足以要求后续研究先完成 R0 修复。

**R0 处理状态（2026-04-28）**：核心代码修复已完成，见 `docs/r0_eval_execution_contract_fix_2026-04-28.md` 与 `data/results/r0_eval_execution_contract_fix_2026-04-28_summary.json`。后续 R1F/R2B 使用 `eval_contract_version=r0_eval_execution_contract_2026-04-28`。

### 2.1 `limit_up_hits_20d` 语义错误

当前实现中，`limit_up_hits_20d` 使用的是：

```text
abs(pct_chg) >= limit_threshold
```

这会把涨停和跌停都计入同一个变量。它实际更接近：

```text
limit_move_hits_20d
```

而不是纯粹的涨停次数。

影响：

1. UPSIDE_C 的“涨停弹性”解释被污染。
2. 强势票和极端下跌票可能被同一个变量混合。
3. 后续如果直接用该变量做进攻 sleeve，会把波动/跌停风险误当成上涨弹性。

修复方向：

1. 新增 `limit_up_hits_20d`：只统计 `pct_chg >= threshold`。
2. 新增 `limit_down_hits_20d`：只统计 `pct_chg <= -threshold`。
3. 可保留 `limit_move_hits_20d` 作为波动路径诊断，但不得命名为 up。

### 2.2 `tplus1_open` 涨停买入失败近似可能误伤强势票

当前回测引擎使用 `open(t+1) / open(t) - 1 >= 9.5%` 近似判断“开盘涨停买不到”。

问题：

1. 这更像持有期 open-to-open 收益达到涨停，不等价于入场开盘触及涨停。
2. 对创业板、科创板、北交所的 20%/30% 涨跌幅限制不够精确。
3. 该逻辑对 strong-up 和涨停弹性候选尤其敏感，可能系统性惩罚 upside sleeve。
4. 买不到只应作用在新买入或增持部分，不应错误影响已有持仓。

修复方向：

1. 使用真实入场日 `open(entry_date)` 对 `pre_close(entry_date)` 判断是否一字涨停。
2. 按股票代码识别 10%/20%/30% 涨跌幅限制。
3. 只对新增或增持权重应用买入失败处理。
4. 对继续持有部分，不应用“买入失败”逻辑。
5. 输出调仓日 buy-fail diagnostic：失败标的、失败权重、重分配权重、闲置权重。

### 2.3 策略收益与基准收益口径仍需显式对齐

当前策略主要采用 `tplus1_open`，但 `market_ew_proxy` 多处仍用 close-to-close 等权收益作为基准。

影响：

1. daily proxy 与 full backtest 方向校准有效，但经济解释仍可能有残余口径差。
2. strong-up capture、月度超额和 state slice 可能混入开盘/收盘口径差。

修复方向：

1. 新增 `market_ew_o2o_proxy`：用全市场 open-to-open 日收益等权。
2. daily proxy、R1/R2/R3 状态切片、full backtest 默认使用与策略执行一致的 benchmark。
3. 保留 close-to-close `market_ew_proxy` 作为对照，但不得混作 primary decision metric。

### 2.4 Regime 阈值不能用全样本未来分位数进入交易决策

当前 R1 归因用全样本分位数切 `strong_up/strong_down` 可以接受，因为它是事后诊断。

但 R2 若把 regime 用作状态权重，阈值必须可交易：

1. 使用 expanding quantile。
2. 或使用 rolling quantile。
3. 或使用训练窗固定阈值，然后 OOS 使用该阈值。

后续任何 state-aware 候选必须显式记录：

| 字段 | 要求 |
| --- | --- |
| `state_threshold_mode` | `expanding` / `rolling` / `train_fixed` |
| `state_lag` | 至少 `previous_completed_month` |
| `state_feature_available_date` | 必须早于调仓决策日 |
| `lookahead_check` | 必须为 `pass` |

### 2.5 Universe gate 可能压制 strong-up 弹性

当前研究基线启用：

```text
min_amount_20d >= 50,000,000
roe_ttm > 0
```

这对稳定性有帮助，但可能天然压制：

1. 小盘弹性。
2. 亏损反转票。
3. 主题扩散阶段的非成熟盈利标的。

后续应单独做 strong-up universe ablation：

| 候选 | 用途 |
| --- | --- |
| `U0_current` | 当前默认 universe |
| `U1_no_roe_gate` | 检验 ROE 正值门槛是否压制上涨弹性 |
| `U2_liquidity_only` | 只保留流动性门槛 |
| `U3_liquidity_tighter` | 更高流动性门槛，排除无法交易小票 |

注意：universe ablation 只作为诊断，不等于直接放宽生产股票池。

### 2.6 dual sleeve 的组合表达过粗

当前 dual sleeve 是 defensive Top-20 和 upside Top-20 两个篮子线性混合。触发 upside 时组合持仓名义上可达到约 40 个名字。

问题：

1. 它不是在解决 Top-20 边界替换，而是在新增一个高换手篮子。
2. 它天然增加换手、稀释防守暴露。
3. 它无法回答“哪个新买入值得替代哪个旧持仓”。

后续 dual sleeve 不再作为主表达。新表达应改为：

```text
defensive core Top-20
+ capped replacement slots
+ explicit hold/replace gate
```

第一版只允许替换最弱的 3 到 5 个名字，而不是新增完整 upside 篮子。

### 2.7 生产入口与工程卫生风险

当前 `pytest -q` 通过，但 `ruff check src scripts tests` 失败，且包含一个实际生产入口风险：

1. `scripts/daily_run.py` 中 `log` 在初始化前被使用。
2. 多个脚本存在 import 排序、未用变量、模糊变量名等 lint 欠账。
3. CI 配置里包含 ruff job，当前状态不适合直接提交为稳定主线。

修复要求：

1. `daily_run.py` 的 `log` 必须在首次使用前初始化。
2. `ruff check src scripts tests` 至少不能有 `F` 类错误。
3. 新增 R0 相关测试后，`pytest -q` 必须继续通过。

---

## 3. 新路线图

### R0：评估与执行契约硬化

**目标**：先确保后续 upside 失败或改善是 alpha 问题，而不是评估/执行口径误差。

R0 是强制阶段。R0 未完成前，不启动新的 R2/R3 候选。

#### R0.1 修复涨跌停路径特征

任务：

1. 将当前 `limit_up_hits_20d` 拆分为：
   - `limit_up_hits_20d`
   - `limit_down_hits_20d`
   - `limit_move_hits_20d`
2. 更新 R1/R2 报告中的 feature spec。
3. 所有使用 UPSIDE_C 的地方改为明确使用纯 `limit_up_hits_20d`。
4. 增加单元测试：
   - 涨停只增加 `limit_up_hits_20d`
   - 跌停只增加 `limit_down_hits_20d`
   - 两者都增加 `limit_move_hits_20d`

验收：

| 指标 | 要求 |
| --- | --- |
| feature semantics | 名称和计算一致 |
| tests | 覆盖涨停/跌停方向 |
| report | 明确记录旧变量语义修复 |

#### R0.2 修复 `tplus1_open` 买入失败处理

任务：

1. 基于 entry date 的 `open / pre_close` 判断是否一字涨停。
2. 买入失败只应用于新增/增持权重。
3. 已持仓部分继续持有，不因“无法买入”被错误归零。
4. 对 10%/20%/30% 板块限幅分别处理。
5. 输出 buy-fail diagnostic。

验收：

| 场景 | 预期 |
| --- | --- |
| 新买入一字涨停 | 新增权重买入失败 |
| 已持仓一字涨停 | 已持仓部分保留收益 |
| 创业板/科创板 20% | 不按 9.5% 错误判定 |
| 全部买入失败 | 资金闲置或按配置重分配，必须记录 |

#### R0.3 建立 open-to-open market benchmark

任务：

1. 新增 `build_market_ew_open_to_open_benchmark` 或等价函数。
2. daily proxy 与 full backtest 在 `tplus1_open` 下默认使用 open-to-open benchmark。
3. 报告同时保留 close-to-close benchmark 对照。

验收：

| 字段 | 要求 |
| --- | --- |
| `primary_benchmark_return_mode` | `open_to_open` |
| `comparison_benchmark_return_mode` | `close_to_close` 可选 |
| `daily_proxy_first_metric` | 使用 primary benchmark |

#### R0.4 修复 regime 阈值可交易性

任务：

1. R1 事后归因可继续使用 full-sample quantile，但必须标注为 `diagnostic_only`。
2. R2/R3 状态决策必须使用 `expanding`、`rolling` 或 `train_fixed` 阈值。
3. 输出每个 rebalance 使用的 state source month、threshold source 和 lag。

验收：

| 项目 | 要求 |
| --- | --- |
| no-lookahead state | pass |
| state diagnostic | 每个 rebalance 可追溯 |
| report | 显式记录 threshold mode |

#### R0.5 修复生产入口与 lint

任务：

1. 修复 `scripts/daily_run.py` 中 `log` 初始化顺序。
2. 清理或压制合理的 ruff 问题。
3. 至少保证 `ruff check` 不再出现 `F821`、`F401`、`F841` 等真实错误。

验收：

| 命令 | 要求 |
| --- | --- |
| `pytest -q` | pass |
| `python -m ruff check src scripts tests` | pass，或至少无真实错误并在报告中列出剩余风格债 |

R0 建议产物：

```text
docs/r0_eval_execution_contract_fix_2026-04-28.md
data/results/r0_eval_execution_contract_fix_2026-04-28_summary.json
data/results/r0_eval_execution_contract_fix_2026-04-28_benchmark_compare.csv
```

---

### R1F：Fixed Baseline 最小复跑

**目标**：R0 修复后，用最少候选判断旧失败是否仍成立。

R1F 不是新研究扩展，而是口径修复后的 sanity check。

固定只复跑三组：

| 候选 | 用途 |
| --- | --- |
| `BASELINE_S2_FIXED` | 修复后防守基线 |
| `UPSIDE_C_FIXED` | 修复后的纯涨停弹性 + tail strength 候选，仅诊断 |
| `DUAL_V1_FIXED` | 修复后的 80/20 trigger-only 对照 |

不得在这一阶段新增大网格。

必须比较：

1. 修复前 vs 修复后 daily proxy。
2. 修复前 vs 修复后 strong-up 中位超额。
3. 修复前 vs 修复后 strong-down 中位超额。
4. 修复前 vs 修复后 switch quality。
5. 修复前 vs 修复后 buy-fail weight。
6. 修复前 vs 修复后 benchmark 口径差。

判定：

| 情况 | 动作 |
| --- | --- |
| 三组仍明显 reject | 旧 R2 失败成立，进入 R2B 重做 upside 输入 |
| `UPSIDE_C_FIXED` switch quality 改善但组合仍差 | 只允许进入 replacement 诊断，不允许独立 sleeve |
| `DUAL_V1_FIXED` 进入 gray zone | 补完整 state/boundary 诊断，暂不 full backtest |
| 任一候选 daily proxy `>= +3%` | 才允许补正式 full backtest |

建议产物：

```text
docs/r1f_fixed_baseline_rerun_2026-04-29.md
data/results/r1f_fixed_baseline_rerun_2026-04-29_*.csv
```

---

### R2B：重做更可交易的 upside 输入

**目标**：不再测试“纯上涨 Top-20 独立篮子”，而是构造能服务边界替换的可交易 upside 输入。

核心原则：

1. upside 输入必须能解释为什么“这个新名字”比 S2 边界上的旧名字更值得买。
2. 强势不等于涨停，不等于高波动，不等于全换仓。
3. 每轮最多 3 个候选。
4. 每个候选先做 stand-alone diagnostic，但 promotion gate 以 replacement 组合为主。

#### R2B.1 需要新增或修复的特征族

| 特征族 | 目的 | 要求 |
| --- | --- | --- |
| `industry_breadth` | 判断上涨是否为行业/主题扩散 | 需要行业映射 |
| `intra_industry_strength` | 避免只买全市场强势但行业内落后票 | 行业内 rank |
| `tradable_breakout` | 捕捉可买的突破，不追一字板 | 必须通过 buyability filter |
| `amount_turnover_expansion` | 捕捉成交确认 | 加入过热惩罚 |
| `overheat_penalty` | 排除极端连续涨停/高位放量出货 | 必须与 upside 同时使用 |
| `size_liquidity_guard` | 避免不可成交小票 | 不等同于大盘偏置 |

#### R2B.2 第一轮候选

第一轮最多 3 个：

| 候选 | 组成 | 假设 |
| --- | --- | --- |
| `U2_A_industry_breadth_strength` | 行业 breadth + 行业内相对强度 + 成交额扩张 | strong-up 来自行业扩散，而非单票追涨 |
| `U2_B_tradable_breakout_expansion` | 可买突破 + turnover/amount expansion + overheat penalty | 捕捉能买到的趋势延续 |
| `U2_C_s2_residual_elasticity` | 对市值/波动/S2 暴露残差化后的弹性特征 | 只补 S2 缺失的弹性，不复制高换手噪声 |

每个候选必须输出：

1. stand-alone Top-20 diagnostic。
2. replacement Top-3 / Top-5 diagnostic。
3. 与 S2 baseline 的 overlap、turnover、sector exposure。
4. strong-up / strong-down / wide breadth / narrow breadth 切片。
5. switch-in/out 与 topk-vs-next。

#### R2B.3 组合表达改为 replacement sleeve

第一版组合不再是：

```text
80% defensive Top-20 + 20% upside Top-20
```

而是：

```text
S2 defensive core Top-20
在 strong-up/wide-breadth 且 replacement gate 通过时：
    允许替换 S2 最弱的 3 到 5 个持仓
其他状态：
    保持 defensive core
```

候选组合：

| 候选 | 规则 |
| --- | --- |
| `R2B_REPLACE_3` | 最多替换 3 个名字 |
| `R2B_REPLACE_5` | 最多替换 5 个名字 |
| `R2B_OVERLAY_10` | 只给 upside replacement 总权重 10%，不扩完整篮子 |

替换必须满足：

1. 新候选可买。
2. 新候选 upside score 进入当日候选池前分位。
3. 新候选相对被替换持仓有足够 score margin。
4. 新候选不能触发 overheat / limit-down / liquidity 风险。
5. 替换后行业集中度不超过阈值。

#### R2B.4 验收标准

R2B 候选必须相对 `BASELINE_S2_FIXED` 评估。

| 指标 | 最低要求 |
| --- | --- |
| daily proxy 年化超额 | 不低于 baseline，且最好 `>= 0%` |
| strong-up 中位超额 | 至少改善 `+2pct`，或正超额比例明显改善 |
| strong-up positive share | 至少改善 `+10pct` |
| strong-down 中位超额 | 劣化不超过 `2pct`，否则必须有总超额补偿 |
| 平均半 L1 换手增量 | 默认不超过 `+0.10`；超过必须有收益补偿 |
| switch-in-minus-out | strong-up 不显著为负 |
| topk-minus-next | strong-up 不明显反向 |
| 2021/2025/2026 | 至少两个关键年份 strong-up 改善 |

只有 daily proxy `>= +3%` 才允许补正式 full backtest。

建议产物：

```text
docs/r2b_tradable_upside_replacement_v1_2026-04-30.md
data/results/r2b_tradable_upside_replacement_v1_2026-04-30_*.csv
```

---

### R3：Boundary-Aware 替换模型

**启动条件**：R2B 至少满足以下任一条件：

1. replacement 候选进入 gray zone。
2. replacement 候选 daily proxy 不低于 baseline，且 strong-up 明显改善。
3. 规则版 `switch_in_minus_out` 在 strong-up 转正，并非由极少数月份支撑。

若 R2B 仍全面失败，不启动 R3。

#### R3.1 第一阶段只做规则版 boundary gate

先不训练模型。

规则版目标：

```text
hold unless replacement has clear edge
```

规则输入：

1. upside score margin。
2. switch candidate 的可买性。
3. 行业 breadth。
4. turnover/amount expansion。
5. overheat penalty。
6. 被替换持仓的 S2 排名恶化程度。
7. 当前 regime/breadth。

输出：

1. `replace / hold` 决策。
2. 每次替换的解释字段。
3. 替换前后预期暴露变化。

验收：

| 指标 | 要求 |
| --- | --- |
| replacement count | 每月可解释，不能全换 |
| switch-in-minus-out | strong-up 转正或明显改善 |
| turnover | 增量可控 |
| daily proxy | 不低于 R2B |

#### R3.2 第二阶段才做最小模型

模型优先级：

| 算法 | 用途 | 优先级 |
| --- | --- | --- |
| Logistic / calibrated classifier | 判断替换是否通过 | 第一优先 |
| Pairwise ranker | 判断 switch-in 是否优于 switch-out | 第二优先 |
| XGBoost regression | 基线对照 | 第三优先 |
| 深度序列模型 | 小样本对照，不作为主线 | 低优先级 |

标签：

| 标签 | 定义 |
| --- | --- |
| `switch_quality_label` | 新换入是否跑赢被换出 |
| `top_vs_next_label` | Top-20 是否显著优于 21-60 |
| `hold_vs_replace_label` | 继续持有旧仓是否优于替换 |
| `state_conditional_label` | 仅在 strong-up/wide-breadth 学替换，在弱市保持 defensive |

R3 验收：

1. `switch_in_minus_out >= 0`，且不是靠 1 到 2 个月支撑。
2. `topk_minus_next >= 0`，strong-up 不为负。
3. daily proxy 至少不低于 R2B。
4. strong-up 改善不消失。
5. strong-down 防守不明显劣化。
6. 若 `val_rank_ic` 好但 daily proxy 或 switch quality 差，直接淘汰。

建议产物：

```text
docs/r3_boundary_replacement_v1_2026-05-02.md
data/results/r3_boundary_replacement_v1_2026-05-02_*.csv
```

---

### R4：数据与行业基础设施

**目标**：让 upside 输入有行业/主题扩散信息，并修复新数据质量。

#### R4.1 行业映射优先级最高

R1 已经说明行业分布缺失限制了 strong-up 归因。

任务：

1. 建立或刷新 `data/cache/industry_map.csv`。
2. 至少包含：
   - `symbol`
   - `industry_level1`
   - `industry_level2` 或可用替代
   - `source`
   - `asof_date`
3. 所有 R2B/R3 报告输出行业暴露。
4. 新增行业 breadth 与行业内 rank 特征。

验收：

| 指标 | 要求 |
| --- | --- |
| 覆盖率 | 当前 universe 覆盖率 `>= 90%` |
| PIT 风险 | 至少记录 source/asof_date |
| 报告 | R2B 起必须有行业暴露表 |

#### R4.2 fund flow 暂不进 alpha 主线

当前质量报告显示：

1. fund flow 最新日期晚于日线最新日期。
2. 覆盖区间内仍存在未匹配行。
3. 部分标的不在当前日线 universe。

处理：

1. 先补齐日线到资金流最新日期。
2. 复跑 `run_newdata_quality_checks.py`。
3. 只有质量报告 `ok=True` 或明确解释剩余缺口后，才恢复研究预算。

恢复前禁止：

1. 跑 `G2/G4` 主线网格。
2. 把 fund flow 直接拼入 P1/R2 主线。
3. 用资金流单因子救当前失败策略。

#### R4.3 shareholder 低优先级保留

shareholder PIT 质量已较保守，但 alpha 证据弱。

处理：

1. 保留低优先级观察。
2. 不进入 R2B/R3 主线。
3. 只有出现明确机制证据时再重启。

---

### R5：生产边界与配置治理

**目标**：防止研究配置误进入日更推荐。

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

## 4. 未来 10 个工作日执行队列

### Day 1-2：R0 评估/执行契约修复

任务：

1. 修复 `limit_up_hits_20d` 语义。
2. 修复 `tplus1_open` 买入失败处理。
3. 增加 open-to-open market benchmark。
4. 修复 regime threshold 可交易模式。
5. 修复 `daily_run.py` 的 `log` 初始化问题。
6. 跑 `pytest` 和 `ruff`。

通过条件：

1. R0 测试通过。
2. 报告说明修复前后口径差。
3. 后续实验能引用 `eval_contract_version` 和 `execution_contract_version`。

### Day 3：R1F fixed baseline 最小复跑

只复跑：

1. `BASELINE_S2_FIXED`
2. `UPSIDE_C_FIXED`
3. `DUAL_V1_FIXED`

通过条件：

1. 明确旧 R2 失败是否仍成立。
2. 若修复后结果变化很大，必须先解释变化来源。
3. 不新增候选网格。

### Day 4-6：R2B 第一轮 tradable upside replacement

任务：

1. 补行业映射。
2. 构造 `industry_breadth`、`intra_industry_strength`、`tradable_breakout`、`overheat_penalty`。
3. 只测试 3 个 upside 输入。
4. 每个输入都跑 replacement 组合，不再只看 pure upside Top-20。

通过条件：

1. 至少一个 replacement 候选 strong-up 改善。
2. daily proxy 不低于 `BASELINE_S2_FIXED`。
3. 换手增量可解释。

### Day 7：R2B replacement 规则收敛

任务：

1. 比较 Replace-3、Replace-5、Overlay-10。
2. 只保留最稳的一种表达。
3. 输出 switch/topk boundary 诊断。

通过条件：

1. 如果仍全面 reject，停止 R2B 第一轮。
2. 如果进入 gray zone，允许补更细 slice，但不直接 promotion。
3. 如果 `>= +3%`，允许补正式 full backtest。

### Day 8-9：R3 规则版 boundary gate

仅在 R2B 通过启动条件时执行。

任务：

1. 先做规则版 `hold/replace`。
2. 不训练复杂模型。
3. 验证 switch quality 是否稳定改善。

通过条件：

1. `switch_in_minus_out` 不显著为负。
2. daily proxy 不低于 R2B。
3. strong-up 改善不消失。

### Day 10：阶段决策

输出一份阶段总结：

1. R0 修复是否改变旧结论。
2. R2B 是否找到可交易 upside 输入。
3. R3 是否有启动价值。
4. 是否允许任何候选补正式 full backtest。
5. 是否继续研究、冻结、或回到数据/行业基础设施。

建议产物：

```text
docs/phase_decision_eval_contract_to_upside_replacement_2026-05-08.md
```

---

## 5. 停止清单

以下方向冻结，除非出现新数据、口径修复后反转、或明确机制证据：

1. 不继续做 P1/G0 近邻标签微调。
2. 不用旧 `light_strategy_proxy` 或 `full_like_proxy` 触发正式回测。
3. 不用 `val_rank_ic` 替代 daily proxy 或正式 full backtest。
4. 不继续做纯 Top-K、buffer、分层等权的小参数网格。
5. 不继续加大 R2 dual sleeve upside 权重。
6. 不把 pure upside Top-20 当作可 promotion 候选。
7. 不在 R0 完成前新增 R2/R3 实验。
8. 不在 R2B 未进入 gray zone 前启动 boundary-aware 训练模型。
9. 不把 `weekly_kdj_*` 写回默认 `composite_extended`。
10. 不扩 `weekly_kdj` interaction 网格。
11. 不重复跑 `G2/G4` 直拼 fund flow。
12. 不继续为 shareholder 做单因子或线性权重主线。
13. 不把任何未 promotion 的研究配置写入生产日更推荐。
14. 不把 historical config 快照重新平铺到项目根目录。
15. 不把深度模型作为下一阶段主线。
16. 不使用全样本 regime 分位数做可交易状态决策。
17. 不在 `limit_up_hits_20d` 语义修复前继续使用它做进攻 sleeve。

---

## 6. 保留清单

这些能力继续保留：

1. `daily-proxy-first` runner 和三档 gate。
2. 正式 full backtest 作为唯一 promotion 必要条件。
3. `*_daily_proxy_leaderboard.csv`。
4. `*_daily_proxy_monthly_state.csv` 和 `*_daily_proxy_state_summary.csv`。
5. `*_topk_boundary.csv`。
6. switch quality 诊断。
7. strong-up / strong-down / breadth 切片。
8. S2 / `vol_to_turnover` 作为 defensive sleeve 候选。
9. G0 技术特征作为 baseline diagnostic，不再作为唯一 alpha 来源。
10. fund flow / shareholder 数据链路，但必须先过质量 gate。
11. `risk_parity` 作为诊断工具，不回当前主线。
12. `config.yaml.backtest` 作为 canonical 研究入口。
13. 历史配置快照保留在 `configs/backtests/`。
14. R1 失败归因框架继续作为每轮新候选的诊断模板。

---

## 7. Gate 与 Promotion 标准

### 7.1 Daily proxy gate

R0 修复后，daily proxy gate 继续使用三档：

| daily proxy 年化超额 | 状态 | 处理 |
| ---: | --- | --- |
| `< 0%` | `reject` | 停止，不补正式 full backtest |
| `0% ~ +3%` | `gray_zone` | 只归档诊断，默认不补正式 full backtest |
| `>= +3%` | `full_backtest_candidate` | 允许补正式 full backtest |

例外复核条件：

1. R0 修复导致旧候选方向显著改变。
2. daily proxy 为正但正式 full backtest 为负。
3. `abs(daily_minus_full) > 5pct`。
4. 方向错配累计超过 2 次。

### 7.2 R2B replacement gate

R2B 候选除了 daily proxy，还必须满足：

1. strong-up 中位超额相对 baseline 改善。
2. strong-down 防守不明显劣化。
3. 换手上升有收益补偿。
4. `switch_in_minus_out` 不显著为负。
5. `topk_minus_next` 不明显反向。
6. 行业集中度和可买性风险可解释。

### 7.3 Production promotion 标准

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

## 8. 产出规范

每轮新实验至少保留：

1. 配置身份：
   - `research_topic`
   - `research_config_id`
   - `output_stem`
   - `result_type`
   - `config_source`
2. 评估契约：
   - `eval_contract_version`
   - `execution_contract_version`
   - `benchmark_return_mode`
   - `state_threshold_mode`
   - `state_lag`
3. 组合口径：
   - `benchmark_symbol`
   - `top_k`
   - `rebalance_rule`
   - `portfolio_method`
   - `execution_mode`
   - `max_turnover`
   - 成本假设
4. 数据口径：
   - `prefilter`
   - `universe_filter`
   - `industry_map_source`
   - `feature_spec`
   - `label_spec` 或 `rule_spec`
   - cache 路径与 schema/version
5. 结果文件：
   - summary CSV
   - detail CSV 或 period detail
   - JSON summary
   - manifest
6. 诊断文件：
   - daily proxy leaderboard
   - state slice
   - breadth slice
   - Top-K boundary
   - switch quality
   - buy-fail diagnostic
   - industry exposure
7. 一页结论文档：
   - 只改了什么
   - 对 S2 defensive baseline 的变化
   - 对 primary benchmark 的变化
   - 对 strong-up / strong-down 的变化
   - 对 switch quality 和 topk boundary 的变化
   - 是否允许进入下一层验证

若发生 fallback，必须记录 fallback 原因。

若使用历史快照，优先记录解析后的 `configs/backtests/...` 路径；旧短名只作为兼容入口保留。

---

## 9. 证据索引

### 9.1 主计划与路线

- `docs/plan.md`
- `docs/plan-04-20.md`

### 9.2 基线和上涨参与不足

- `docs/benchmark_gap_diagnostics_2026-04-20_v3.md`
- `docs/score_ablation_2026-04-19.md`
- `docs/alpha_factor_scout_2026-04-20.md`
- `docs/factor_admission_validation_2026-04-20_next.md`

### 9.3 R1 strong-up 失败归因

- `docs/p1_strong_up_failure_attribution_2026-04-27.md`
- `scripts/run_p1_strong_up_attribution.py`
- `data/results/p1_strong_up_failure_attribution_2026-04-27_*.csv`

### 9.4 R2 upside / dual sleeve 失败

- `docs/p2_upside_sleeve_v1_2026-04-27.md`
- `scripts/run_p2_upside_sleeve_v1.py`
- `data/results/p2_upside_sleeve_v1_2026-04-27_*.csv`
- `docs/p2_regime_aware_dual_sleeve_v1_2026-04-28.md`
- `scripts/run_p2_regime_aware_dual_sleeve_v1.py`
- `data/results/p2_regime_aware_dual_sleeve_v1_2026-04-28_*.csv`

### 9.5 P1 proxy 校准与失败候选

- `docs/p1_label_objective_experiment_2026-04-26.md`
- `docs/p1_failure_diagnostics_2026-04-26.md`
- `docs/p1_marketrel_g1_gap_diagnostics_2026-04-26.md`
- `docs/p1_marketrel_state_diagnostics_2026-04-27.md`
- `docs/p1_proxy_calibration_history_2026-04-27.md`
- `docs/p1_proxy_calibration_history_h20_2026-04-27.md`
- `docs/p1_daily_bt_like_proxy_calibration_2026-04-27.md`
- `docs/p1_daily_proxy_first_2026-04-27.md`
- `docs/p1_monthly_investable_label_smoke_2026-04-27.md`
- `docs/p1_up_capture_market_relative_g0_smoke_2026-04-27.md`
- `docs/p1_monthly_investable_up_capture_g0_smoke_2026-04-27.md`
- `docs/p1_rank_fusion_long_horizon_g0_smoke_2026-04-27.md`
- `docs/p1_top_bucket_rank_fusion_g0_smoke_2026-04-27.md`

### 9.6 Weekly KDJ 与已冻结方向

- `docs/alpha_factor_scout_2026-04-24_weekly_kdj.md`
- `docs/p1_weekly_interaction_ab_2026-04-26.md`
- `docs/p1_rank_direction_rerun_2026-04-26.md`
- `docs/p1_rankfix_same_window_full_backtest_2026-04-26.md`

### 9.7 新数据质量

- `docs/newdata_quality_current_families_fund_flow-shareholder_flow_a_share_fund_flow_holder_a_share_shareholder_daily_a_share_daily_lag_30_width_100.md`
- `docs/alpha_factor_scout_2026-04-23_shareholder_smoke.md`
- `scripts/run_newdata_quality_checks.py`
- `scripts/fetch_fund_flow.py`
- `scripts/fetch_shareholder.py`

### 9.8 核心代码入口

- `config.yaml.backtest`
- `configs/backtests/README.md`
- `scripts/run_backtest_eval.py`
- `src/backtest/engine.py`
- `src/models/xtree/p1_workflow.py`
- `src/market/tradability.py`
- `scripts/daily_run.py`

---

## 10. 当前一句话路线

**停止 P1 标签近邻微调；冻结旧 R2 sleeve 权重扩展；先完成 R0 评估/执行契约修复；用 fixed baseline 复核旧结论；然后以 `industry breadth + tradable breakout + boundary replacement` 重做 upside；只有 replacement 至少进入 gray zone 后，才启动 boundary-aware 模型。**
