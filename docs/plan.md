# 策略优化与研发计划

**文档版本**：2026-04-14（项目代码全量分析版）  
**依据**：`docs/backtest_report.md`、`docs/backtest_report_data.json` 全样本与 Walk-Forward 结果，以及对仓库内 `src/`、`scripts/` 全部源码的系统性审查。

**目的**：在可复现的回测框架下，系统性提升**风险调整后收益**与**相对全市场等权基准的超额**，并把最大回撤控制在可接受区间；本文件为执行级路线图，与回测报告同步迭代。

---

## 1. 基线与结论（当前是否「满意」）

| 维度 | 当前（含成本，2021-01-01～2026-04-14） | 结论 |
|------|----------------------------------------|------|
| 年化收益（CAGR） | **+10.59%**（基准全市场等权 **+13.47%**） | 绝对收益尚可，**相对基准跑输约 2.2%/年**，不满足「alpha 策略」预期 |
| 夏普 | **0.499** | 勉强贴合格线，风险补偿一般 |
| 最大回撤 | **-49.88%**（基准约 -33.87%） | **不可接受**：尾部风险显著高于简单等权 |
| 年度稳定性 | 6 个自然年中 **4 年负超额**（2022、2023、2025、2026 偏弱） | 市况敏感，策略脆弱 |
| Walk-Forward（滚动 15 折） | 年化均值 +34.14%（受 Fold 10 极端值拉高），**中位数**约 +11.6% | 单折 63 日噪声大，中位数更接近实际 OOS 能力 |
| Walk-Forward（分段 3 折） | OOS 均值年化 +12.70%，夏普 +0.515，回撤 -27.34% | 与全样本基本一致，无明显过拟合信号 |

**结论**：以「跑赢简单等权、回撤可控」为标尺，**当前结果不能算满意**；需在因子、组合构建、模型与验证四线并行改进。

---

## 2. 根因归纳（与代码/配置深度对齐）

### 2.1 因子维度单一，信号同质

- `composite_extended`（`config.yaml.example` 中 16 个因子）全部为技术面短周期信号（动量、RSI、偏离度、涨跌幅等），权重绝对值前五名均为反转/动量类（各约 13.6%）。
- 无基本面（PE/PB/ROE）、无结构性因子（行业轮动）、无资金流向因子，截面分散度低。
- `src/features/tensor_alpha.py`、`tensor_base_factors.py`、`intraday_proxy_factors.py` 提供了计算基础，但尚未扩展至基本面域。

### 2.2 权重静态，未利用已有 IC 监控能力

- 因子权重硬编码在 `config.yaml.example` 的 `signals.composite_extended` 节，属于「凭经验配置」。
- **`src/features/ic_monitor.py`（`ICMonitor` 类）已完整实现**滚动 IC 统计与告警，但未接入 `daily_run.py` 的因子加权路径；`src/features/factor_eval.py` 中 `information_coefficient`、`rank_ic`、`layered_returns` 等评估工具同样未被日常调用。
- 滚动 ICIR 加权的基础设施已就位，缺的是把 `ic_monitor.json` 结果反馈到 `composite_extended_linear_score` 权重的「闭环」代码。

### 2.3 组合构建偏粗，优化器未接入主链路

- 主回测固定 **Top-50 等权**，无行业/市值约束。
- **`src/portfolio/optimizer.py` 已实现** `optimize_risk_parity`、`optimize_min_variance`、`optimize_mean_variance` 三种方法，`weights_from_cov_method` 提供统一入口；`src/portfolio/covariance.py` 实现 Ledoit-Wolf 收缩。
- `src/portfolio/weights.py` 中 `build_topk_weights` 是当前唯一生产权重构建路径，**行业约束逻辑尚未添加**。
- 上述优化器与协方差在 `tests/` 有单元测试，可直接接入回测引擎。

### 2.4 XGBoost 模型 Rank IC 为负，两套系统未统一评价

- 工件 `data/models/xgboost_panel_ee2108f515be/bundle.json` 显示 `train_rank_ic = -0.161`、`val_rank_ic = -0.074`，若 `config.yaml` 中 `sort_by: xgboost`，生产实际选出的是未来表现最差的股票。
- 当前 `run_backtest_eval.py` 主报告使用 `composite_extended` 线性路径，但 `config.yaml.example` 默认 `sort_by: xgboost`——**生产与回测默认配置不一致**。
- `src/models/inference.py` 有 `rank_ic_guard` 保护逻辑（`config.yaml.example` 中 `quantile: 0.25, min_history: 4`），但不能弥补模型本身方向错误。
- `scripts/train/train_xgboost.py`、`train_baseline.py`、`train_deep_sequence.py`、`train_timeseries.py` 均已存在，缺的是**正确的时序切分标签与重训验收流程**。

### 2.5 信号与调仓错配

- 多数因子窗口 3～20 日，月末调仓（约 21 个交易日）导致信号在执行时已大幅衰减。
- 执行口径为 `close_to_close`，而因子是 T 日收盘值——理论上存在一日的信号可用性延迟（T 日信号、T+1 日实际能入场），但该偏差在回测中已被接受并记录。

### 2.6 Regime 机制有实现，效果未独立量化

- `src/market/regime.py` 实现 bull/bear/oscillation 三态分类 + 连续动态因子权重调整，已接入 `daily_run.py`。
- 当前没有回测脚本对 **「启用 regime 调权 vs 关闭」** 做对照实验，无法判断其净贡献；`config.yaml.example` 中 `regime.enabled: true` 为默认，可能引入不必要的方差。

### 2.7 Walk-Forward 报告解读偏差

- 滚动 WF 均值被 Fold 10（年化 +437.88%，总收益 +52.29%）严重拉高，均值 +34.14% 不具代表性。
- 脚本当前输出均值聚合；**中位数** 约 +11.6%，更贴近真实预期；分段 WF（3 折，每折约 1 年）给出更可信的 OOS 回撤估计（-27.34%）。
- `run_backtest_eval.py` 报告中 WF 段落仅导出均值，需增加 p25/中位数/p75 输出。

### 2.8 LLM 支线尚未闭环

- `src/llm/` 实现了 Ollama 客户端、东财飙升榜关注度扫描、新闻/财报情绪分析，但输出为独立 CSV/JSON，**未作为截面因子注入 `composite_extended` 或过滤层**。
- README 明确标注「LLM 尚未闭环」；在基础因子改进完成前，LLM 因子的优先级应次于 P2。

---

## 3. 目标与验收标准（建议）

以下数值为**方向性目标**，以同一套 `run_backtest_eval.py` 参数族与成本假设复测为准；若更换基准或股票池，需单独说明。

| 阶段 | 指标 | 目标 |
|------|------|------|
| **第一阶段** | 超额年化（vs `market_ew`） | 由 -2.17% **转正**，目标 **+1%～+3%/年** |
| | 最大回撤 | 由 -49.88% 收窄至 **≤ -35%** |
| | 年度稳定性 | 6 个自然年中 **正超额年数 ≥ 4 年** |
| **第二阶段** | 夏普（含成本） | 提升至 **0.6+** |
| | 最大回撤 | 进一步收窄至 **≤ -30%** |
| | WF 中位数超额 | **> 0** |
| **通用约束** | 成本敏感性 | 任何策略须同时报含/不含成本，换手 ≤ 60% |
| | OOS 一致性 | 分段 WF（长窗口）OOS 夏普与全样本差距 **< 0.1** |

---

## 4. 工作分解（按优先级与代码可操作性排序）

### P0 — 快速修复：可立刻落地，收益确定性高

#### P0-A：统一生产与回测默认配置（1～2 天）

- **问题**：`config.yaml.example` 中 `sort_by: xgboost`，XGBoost Rank IC 为负，导致生产实际选出的是差股。
- **行动**：
  1. 将 `config.yaml.example` 中 `signals.sort_by` 改为 `composite_extended`（与回测报告对齐）。
  2. 在 `run_backtest_eval.py` 与 `daily_run.py` 顶部注释中明确「默认排序键」，防止日后静默修改。
  3. XGBoost 路径待 P3 重训验收后再恢复。
- **文件**：`config.yaml.example`、`scripts/run_backtest_eval.py`、`scripts/daily_run.py`

#### P0-B：Walk-Forward 报告增加中位数/分位聚合（0.5～1 天）

- **问题**：当前滚动 WF 报告仅输出均值，Fold 10 将均值拉至不可信区间。
- **行动**：在 `src/backtest/walk_forward.py` 的聚合逻辑中，额外计算并输出 `median_ann_return`、`p25_ann_return`、`p75_ann_return`；`run_backtest_eval.py` 的 Markdown 报告同步展示。
- **文件**：`src/backtest/walk_forward.py`、`scripts/run_backtest_eval.py`

#### P0-C：行业持仓上限约束（2～3 天）

- **问题**：无行业约束导致集中暴露（如 2022/2025 偏弱年份可能对应某行业系统性调整）。
- **行动**：
  1. 在 `src/portfolio/weights.py` 的 `build_topk_weights` 中增加 `industry_cap` 参数（如每行业最多持 N 只或 X%），按 AkShare 行业分类实现。
  2. 行业分类数据从 DuckDB `a_share_daily` 或单独 `a_share_info` 表读取（若无则补充一次性抓取脚本）。
  3. 在回测引擎中传递行业映射，验证约束前后回撤差异。
- **文件**：`src/portfolio/weights.py`、`src/backtest/engine.py`、可能需新增 `scripts/fetch_industry.py`

#### P0-D：Regime 对照实验（1 天）

- **问题**：`regime` 模块已接入但净贡献未知，可能增加不必要方差。
- **行动**：在 `run_backtest_eval.py` 中增加 `--no-regime` 选项，输出关闭 regime 调权的对照结果；纳入报告第四节。
- **文件**：`scripts/run_backtest_eval.py`、`src/market/regime.py`

---

### P1 — 因子库扩展（预期 alpha 增量最大）

#### P1-A：基本面因子最小集（1～2 周）

当前因子全为技术面，基本面/质量因子可提供与短期动量低相关的独立 alpha 源。

- **计划引入**（严格按公告日对齐，不允许前视偏差）：
  - 估值：市盈率（TTM）、市净率、EV/EBITDA
  - 盈利质量：ROE（TTM）、净利润同比增速、毛利率变化
  - 资产负债：资产负债率变化、经营现金流/净利润
- **数据路径**：AkShare 已支持财务数据接口；新增 `src/data_fetcher/fundamental_client.py`，数据入 DuckDB `a_share_fundamental` 表，按季度公告日 join 日线表（point-in-time 对齐）。
- **因子计算**：在 `src/features/` 下新增 `fundamental_factors.py`，实现截面 winsorize、z-score 与市值中性化（复用 `src/features/standardize.py`、`neutralize.py`）。
- **接入**：在 `config.yaml.example` 的 `signals.composite_extended` 节增加基本面因子权重，初始以均匀小权重（3%～5%）试水；同步更新 `src/models/rank_score.py` 中 `composite_extended_linear_score` 的可接收因子集。
- **验收**：新因子在因子评估脚本中 Rank IC 均值 > 0.02，且加入前后全样本夏普有正向贡献。

#### P1-B：资金流与情绪因子（可选，按数据可得性）

- 北向资金净流入（AkShare 有接口）
- 融资买入额占成交比
- 分析师一致预期 EPS 变化速度

以上因子数据频率与可回溯深度需先评估，再决定是否纳入回测区间（2021 年以前可能缺失）。

#### P1-C：LLM 情绪因子正式闭环（P1 后期）

- 在 `src/llm/attention_scanner.py` 输出的关注度分数基础上，计算截面标准化的「LLM 情绪 z 分」。
- 接入 `composite_extended` 权重（初始权重 ≤ 5%），通过回测验证净贡献后决定是否扩大。
- 需解决历史回填问题：LLM 情绪因子没有历史数据，初期只能做前向跟踪，不能用于历史回测，需单独说明。

---

### P2 — 权重与组合优化（利用已有实现）

#### P2-A：滚动 ICIR 加权替代静态权重（1～2 周）

**代码已就绪，需实现「闭环」**：

- `src/features/ic_monitor.py` 已实现 `ICMonitor.rolling_ic_stats()`，可输出每因子近期 ICIR。
- 当前缺少的逻辑：
  1. 在 `daily_run.py` 或独立的 `scripts/update_ic_weights.py` 中，读取 `ic_monitor.json` → 计算滚动 ICIR → 导出 `ic_weights.json`。
  2. 在 `src/models/rank_score.py` 的 `composite_extended_linear_score` 函数中增加 `weights_override` 参数，优先使用 `ic_weights.json` 而非 `config.yaml` 静态权重。
  3. 在 `run_backtest_eval.py` 中实现历史 ICIR 滚动权重的回测路径（向量化实现，避免前视偏差）。
- **衰减方案**：ICIR 加权 + 指数衰减（近期 IC 权重更高）+ clip 防止单因子权重过大（上限 25%）。

#### P2-B：组合优化层接入（1～2 周）

**优化器已就绪，需接入主链路**：

- `src/portfolio/optimizer.py`（`optimize_risk_parity`、`optimize_min_variance`）+ `src/portfolio/covariance.py`（Ledoit-Wolf）均已实现并通过单元测试。
- 当前 `build_topk_weights` 固定等权，需在其内部或调用处增加可选的优化权重路径：
  ```
  if portfolio_method == "equal_weight":
      weights = equal_weight(topk_stocks)
  elif portfolio_method in ("risk_parity", "min_variance"):
      Sigma = estimate_covariance(topk_returns, method="ledoit_wolf")
      weights = weights_from_cov_method(portfolio_method, Sigma)
  ```
- 在 `config.yaml.example` 新增 `portfolio.method`（默认 `equal_weight`，可选 `risk_parity`、`min_variance`）。
- **验收指标**：风险平价 vs 等权的全样本回撤对比，以及含成本夏普。

#### P2-C：Top-K 与调仓频率网格搜索（3～5 天）

- 参数网格：`top_k` ∈ {30, 40, 50, 60}、`max_turnover` ∈ {0.3, 0.4, 0.5}、调仓频率 ∈ {月末, 双周}。
- 目标函数：含成本夏普（主）+ Calmar 比率（副），**禁止使用裸收益**。
- 在 `run_backtest_eval.py` 中新增 `--grid-search` 选项，结果导出为 CSV，避免手动逐次运行。

---

### P3 — 机器学习重训与统一评价

#### P3-A：XGBoost 根因排查与重训（1～2 周）

- **排查清单**（针对 Rank IC 为负）：
  1. 标签泄露：确认 `train_xgboost.py` 中标签 `forward_ret` 的对齐方式，检查是否使用了 T+1 当天的收盘价而不是真正的未来价格。
  2. 特征标准化方向：确认 `momentum`/`rsi` 等反转因子在树模型特征中是否需要翻转符号（线性模型用负权重，树模型直接处理单调关系）。
  3. 交叉验证方式：必须使用**严格时间序列切分**（`TimeSeriesSplit`），禁止 k-fold。
  4. 标签设计：当前标签见 `config.yaml.example` 中 `labels.horizons: [5, 10, 20]`，`fusion: rank_fusion`；确认 rank 是在**截面**还是全样本计算。
- **重训规范**：
  - `scripts/train/train_xgboost.py` 修复后，验证集 Rank IC > 0.03 才允许写入 `bundle_dir`。
  - 可选对照：同期训练 LightGBM ranker（`objective='rank:pairwise'`），对比 Rank IC 与分层收益。
- **文件**：`scripts/train/train_xgboost.py`、`src/models/inference.py`

#### P3-B：深度序列模型现状评估（3～5 天）

- `scripts/train/train_deep_sequence.py` 已实现，`src/backtest/` 中 `deep_sequence` 路径存在，但生产回测未启用。
- 评估步骤：拉取最新 `data/models/deep_sequence_latest/` 工件，在独立测试集上计算 Rank IC；若 IC > 0 则接入对照回测，否则先修复。
- 序列模型的优势在于**捕捉时序模式**，与截面线性因子互补；但推理延迟和 Jetson GPU 资源需提前评估。

#### P3-C：生产一致性校验（持续）

- 每次模型更新后，在 `daily_run.py` 的 `eval` 子命令结果中核对：推荐 CSV 的 Rank IC（推荐当日 vs 后 N 日收益的截面相关）。
- `src/cli/eval_recommend.py` 已实现此逻辑，需定期运行并写入 `data/experiments/experiments.jsonl`。

---

### P4 — 验证体系与工程运营

#### P4-A：Walk-Forward 聚合增强（已列 P0-B，此为深化）

- 延长单折测试窗：在 `run_backtest_eval.py` 中支持 `--wf-test-window` 参数，默认保留 63 日（当前），可配置为 126 日（半年）减少单折噪声。
- 分段 WF 有效折数过少（当前 3 折），可通过调整 `n_splits` 或改用 `expanding_window=True` 的扩张窗方案增加样本量。

#### P4-B：配置治理（持续）

- 明确 `config.yaml`（生产）与回测默认的差异点，在 `config.yaml.example` 中加注释说明每个关键字段对回测结果的影响。
- 关键对照字段：`signals.sort_by`、`portfolio.method`、`regime.enabled`、`backtest.eval_rebalance_rule`、`prefilter.*`。
- 考虑新增 `config.yaml.backtest` 作为回测专用配置快照，与生产 `config.yaml` 解耦。

#### P4-C：实验记录规范化（持续）

- `data/experiments/experiments.jsonl` 已存在，但写入路径不统一（部分脚本直接写，部分不写）。
- 统一：每次 `run_backtest_eval.py` 运行结果（参数 + 关键指标）自动追加到 `experiments.jsonl`，便于横向对比。
- 可考虑在 `src/models/experiment_recorder.py` 中增加 `append_backtest_result()` 工厂函数。

---

## 5. 里程碑与时间顺序

| 阶段 | 关键任务 | 涉及文件 | 验收产出 |
|------|----------|----------|----------|
| **M1**（1 周内） | P0-A 配置统一 + P0-B WF 中位数 + P0-D Regime 对照 | `config.yaml.example`、`walk_forward.py`、`run_backtest_eval.py` | 回测报告更新，WF 中位数明确，生产默认使用 `composite_extended` |
| **M2**（2～3 周） | P0-C 行业上限约束 + P2-C Top-K 网格搜索 | `portfolio/weights.py`、`backtest/engine.py`、`run_backtest_eval.py` | 行业约束前后回撤对比，网格搜索最优参数表 |
| **M3**（3～5 周） | P1-A 基本面因子最小集入库 + P2-A ICIR 加权闭环 | `data_fetcher/fundamental_client.py`、`features/fundamental_factors.py`、`ic_monitor.py` + `rank_score.py` | 新因子 Rank IC > 0.02，ICIR 权重时间序列可审计，全样本含基本面对比实验 |
| **M4**（5～7 周） | P2-B 风险平价接入 + P3-A XGBoost 重训验收 | `portfolio/optimizer.py`（已实现）、`train_xgboost.py`、`inference.py` | 风险平价 vs 等权回撤对比，XGBoost 验证集 Rank IC > 0.03 |
| **M5**（7～10 周） | P3-B 深度序列评估 + P1-B 资金流因子（可选）+ P1-C LLM 因子尝试 | `train_deep_sequence.py`、LLM 模块 | 模型横向 Rank IC 对比表，推荐默认配置更新 |

具体日历由资源与数据准备情况调整；每一里程碑结束应保留**可复现命令**（见 `backtest_report.md` 附录），并在 `data/experiments/experiments.jsonl` 中记录关键指标快照。

---

## 6. 风险与依赖

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| **基本面数据前视偏差** | 使用未来信息导致虚高回测 | AkShare 财务接口需核实「公告日」字段是否为真实披露日；回测 join 必须使用 `merge_asof` 按公告日对齐 |
| **XGBoost 方向仍错误** | 生产持续选差股 | M1 前先禁用 xgboost 路径（P0-A）；M4 重训后独立验证，不混入线性策略报告 |
| **Regime 引入额外方差** | 状态切换时机错误放大损失 | M1 期的对照实验（P0-D）量化净贡献，若负贡献则默认关闭 |
| **组合优化数值不稳定** | 协方差估计在股票数多时可能奇异 | `covariance.py` 已用 Ledoit-Wolf 收缩；优化器有 `bounds=(1e-8, 1.0)` 下界保护；需在至少 3 年历史数据下测试 |
| **过拟合风险（因子增多）** | 样本内调参造成 OOS 失效 | 因子引入必须先做 OOS 评估（分段 WF 的测试期，或单独留出 2024 年后的数据作为最终检验集）；禁止仅用全样本调参 |
| **计算成本上升** | GPU（Jetson）资源瓶颈 | 基本面因子为日频 join，计算量小；协方差计算在 Top-K 子集上（≤ 60 只），可接受；深度模型推理延迟需提前在 Jetson 上实测 |
| **数据质量**（ST、停牌、复权） | 因子计算与回测收益对不上 | `src/data_fetcher/data_quality.py` 已实现落库检查；基本面数据需额外的单测（如对比已知财报数字）；ST 过滤在 `prefilter` 中已部分实现 |

---

## 7. 相关文件索引

### 核心流水线

| 文件 | 职责 |
|------|------|
| `scripts/daily_run.py` | 日更主入口：拉数 → 因子 → 排序 → 权重 → 推荐 CSV |
| `scripts/run_backtest_eval.py` | 回测评估入口（本计划验收基准） |
| `src/settings.py` | 配置加载（`QUANT_CONFIG` → `config.yaml` → `config.yaml.example`） |
| `config.yaml.example` | 全局配置模板，包含全部可调参数 |

### 因子层

| 文件 | 职责 |
|------|------|
| `src/features/tensor_alpha.py` | GPU 张量因子（动量、RSI、ATR 等） |
| `src/features/tensor_base_factors.py` | 扩展基础因子 bundle |
| `src/features/intraday_proxy_factors.py` | 日内 K 线结构代理因子 |
| `src/features/factor_eval.py` | IC/RankIC/分层收益评估工具 |
| `src/features/ic_monitor.py` | 滚动 IC 持久化监控与告警（**已实现，待接入权重闭环**） |
| `src/features/standardize.py`、`neutralize.py`、`orthogonalize.py` | 截面标准化、中性化、正交化 |

### 信号与模型层

| 文件 | 职责 |
|------|------|
| `src/models/rank_score.py` | `composite_extended_linear_score`；排序键计算 |
| `src/models/inference.py` | XGBoost/深度序列推理，含 `rank_ic_guard` |
| `scripts/train/train_xgboost.py` | XGBoost 截面排序训练（**待修复 Rank IC 为负问题**） |
| `scripts/train/train_deep_sequence.py` | 深度序列模型训练 |

### 市场状态

| 文件 | 职责 |
|------|------|
| `src/market/regime.py` | Bull/Bear/Oscillation 分类 + 因子权重动态调整 |
| `src/market/tradability.py` | 涨跌停、停牌过滤 |

### 组合与回测层

| 文件 | 职责 |
|------|------|
| `src/portfolio/weights.py` | `build_topk_weights`（**待添加行业约束**） |
| `src/portfolio/optimizer.py` | ERC / 最小方差 / 均值-方差优化（**已实现，待接入主链路**） |
| `src/portfolio/covariance.py` | Ledoit-Wolf 协方差收缩（**已实现**） |
| `src/backtest/engine.py` | 向量回测引擎，支持 `close_to_close` / `tplus1_open` |
| `src/backtest/walk_forward.py` | Walk-Forward 验证（**待增加中位数聚合**） |
| `src/backtest/portfolio_eval.py` | 组合绩效评估 |

### 数据层

| 文件 | 职责 |
|------|------|
| `src/data_fetcher/akshare_client.py` | AkShare 日线拉取与 DuckDB 写入 |
| `src/data_fetcher/akshare_resilience.py` | 超时/重试/缓存韧性层 |
| `src/data_fetcher/db_manager.py` | DuckDB 管理 |
| `src/data_fetcher/data_quality.py` | 落库前质量检查 |

### 文档与结果

| 文件 | 职责 |
|------|------|
| `docs/backtest_report.md` | 最新回测报告（含基线指标、WF 结果、根因诊断） |
| `docs/backtest_report_data.json` | 回测结果机器可读快照 |
| `data/experiments/experiments.jsonl` | 实验记录（参数 + 指标，需规范化写入） |
| `data/models/xgboost_panel_*/bundle.json` | XGBoost 工件（当前 Rank IC 为负，待重训） |

---

*本计划随回测与实现进度更新；重大变更请同步修订本节日期与里程碑状态。*
