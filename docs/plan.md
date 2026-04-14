# 量化系统优化规划

本文档汇总对整条量化流水线的系统性优化分析，便于跟踪与迭代。

---

## 整体架构现状

```
AkShare → DuckDB → PyTorch因子 → 排序(线性/XGBoost/序列) → 组合优化 → 推荐CSV → 回测
                                                    ↕（独立）
                                         LLM新闻/关注度扫描
```

---

## 一、紧急问题（影响系统可用性）

### 1. `config.yaml` 彻底缺失

`config.yaml` 已从仓库删除，但 `Dockerfile` 仍有 `COPY config.yaml ./`，导致 **Docker 构建必然失败**。`README.md` 中文档也全部引用该文件。当前系统靠代码内置默认值维持运行，属于「隐形状态」。

**修复状态（已完成）**：

- 已新增 `config.yaml.example` 作为模板配置。
- `Dockerfile` 已移除 `COPY config.yaml ./`，改为仅拷贝模板文件。
- 配置加载链路已支持 `QUANT_CONFIG -> config.yaml -> config.yaml.example` 回退。
- `README.md` 已更新配置初始化方式（先复制模板再本地覆盖）。

### 2. 依赖声明缺口

`requirements-base.txt` 缺少以下实际运行时依赖：

- `scipy`（`src/portfolio/optimizer.py` 中 SLSQP 优化器）
- `ollama`（`src/llm/client.py`）
- `httpx` / `aiohttp`（resilience 模块）

新环境部署时会出现 `ImportError`。`pyproject.toml` 也未声明任何 `install_requires`。

**修复状态（已完成）**：

- `requirements-base.txt` 已补充：`scipy`、`ollama`、`httpx`、`aiohttp`。
- `pyproject.toml` 已补齐 `project.dependencies`，与运行时依赖保持一致。

### 3. XGBoost 模型可能系统性做反

`docs/backtest_report.md` 明确记录历史 bundle 的 **Rank IC 为负**，而生产链路 `sort_by: xgboost` 若沿用旧 bundle，排序方向可能整体反转。需立即核查当前工件的 IC 方向。

**修复状态（已完成）**：

- 在 `src/models/inference.py` 的 `predict_xgboost_tree` 中新增方向保护：
  - 自动读取 bundle 的 `val_rank_ic` / `train_rank_ic`；
  - 当可用 Rank IC < 0 时，自动翻转 `tree_score` 方向，避免 Top-K 反向排序。
- 已新增单测覆盖该行为（`tests/test_tree_model.py`）。

---

## 二、数据层优化

### 2.1 AkShare 数据稳定性

`src/data_fetcher/akshare_resilience.py` 已有 `SIGALRM` 超时和 JSON 快照回退，但：

- `SIGALRM` 只能用于主线程，多线程并发拉取时无效
- 建议改为 `concurrent.futures.ThreadPoolExecutor` + `Future.result(timeout=...)` 实现真正的超时

### 2.2 DuckDB 写入瓶颈

`db_manager.py` 的并发拉取在大股票池（5000+ 标的）下会产生连接竞争。建议：

- 批量写入改为先攒 Arrow Table 再单次 `INSERT`
- 对 `trade_date` + `symbol` 建联合索引加速回测期间的区间查询

### 2.3 衍生列回填脚本依赖

`scripts/backfill_derived_daily.py` 是手工触发的，若新增因子字段后忘记回填，历史数据和增量数据会产生字段不一致。建议在 `db_manager.py` 初始化时自动检测缺失列并触发回填。

---

## 三、因子工程优化

### 3.1 因子 IC 衰减未被监控

`src/features/factor_eval.py` 有 IC / RankIC 计算，但没有持久化记录和告警。因子效果会随市场状态衰退，需要建立因子 IC 的滚动监控看板（写入 DuckDB 或 JSON），当近期 IC 均值跌破阈值时发出告警。

### 3.2 中性化应作为默认选项

`算法改进.md` 和 `src/features/neutralize.py` 都支持市值+行业回归中性化，但在 `daily_run.py` 的默认路径中未必启用。**市值效应在 A 股极强**，不中性化的因子 IC 很可能大部分来自市值暴露，并非真实 alpha。

### 3.3 K 线结构因子标签未完整映射

`src/models/recommend_explain.py` 的 `FACTOR_LABELS` 未覆盖 `intraday_proxy_factors.py` 中的全部因子，推荐报告的解释性差。

---

## 四、模型层优化

### 4.1 XGBoost 训练的标签质量

`src/features/tree_dataset.py` 已支持夏普/卡玛/截断标签，但需要验证：

- 是否真正使用了**截面相对标签**（而非绝对收益）
- 标签窗口（5d/10d/20d）是否做了多目标融合

建议将标签构造参数纳入实验追踪（`src/models/experiment.py`），使每次训练结果可复现和对比。

**修复状态（已完成）**：

- `scripts/train/train_xgboost.py` 已支持多窗口标签融合（`label-horizons` / `label-weights`，默认可配置 5/10/20）。
- 融合标签采用按交易日截面的相对排序融合（`rank_fusion`），明确使用截面相对标签信号。
- 标签构造参数（窗口、权重、变换、融合模式）已写入 `bundle.json` 与实验日志（`params_json`）。

### 4.2 缺乏模型版本管理

`src/models/artifacts.py` 有 bundle 存取，但没有版本比较机制。当新训练的模型 OOS 表现不如旧模型时，系统会静默替换。建议：

- 新 bundle 写入前先计算验证集 Rank IC，低于历史 P25 分位则拒绝替换
- 保留最近 N 个版本，支持回滚

**修复状态（已完成）**：

- `src/models/artifacts.py` 新增版本治理能力：历史版本列表、指标读取、分位门禁判断、发布目录+历史快照管理。
- `src/models/xtree/train.py` 已接入门禁：候选模型 `val_rank_ic` 低于历史 P25（可配置）会拒绝发布。
- 发布时自动写入 `xgboost_panel_history/active_bundle.json`，记录 active/previous 版本并保留最近 N 份快照，支持回滚。

### 4.3 时序模型（LSTM/TCN）欠缺生产验证

`src/models/timeseries/` 已有完整架构，但 `docs/backtest_report.md` 中的 `config_source: builtin_defaults` 显示历史回测仅用了线性模型。深度序列模型尚未经过严格的 walk-forward OOS 验证。

**修复状态（已完成）**：

- `src/models/timeseries/train.py` 新增可选 walk-forward OOS 校验（可配置 train/test/step/epochs），输出聚合 OOS 指标到模型 metrics。
- `scripts/train/train_deep_sequence.py` 默认启用 `--walk-forward-oos`，训练后自动执行 OOS 验证。
- `scripts/train/train_timeseries.py` 增加 `--time-val-split`（默认开启）与可选 `--walk-forward-oos` 参数，避免随机切分泄漏未来。

---

## 五、组合优化层

### 5.1 协方差矩阵估计质量

`src/portfolio/covariance.py` 使用 Ledoit-Wolf 收缩，在股票池大、历史数据短时效果有限。建议补充：

- **行业因子模型**协方差（Barra 风格，用行业虚拟变量做因子分解）
- 或 **指数加权**协方差（对近期数据加权更重）

**修复状态（已完成）**：

- `src/portfolio/covariance.py` 已新增 `ewma` 协方差估计（`cov_shrinkage: ewma`，支持 `cov_ewma_halflife`）。
- 已新增 `industry_factor` 协方差估计（`cov_shrinkage: industry_factor`），按行业虚拟变量做因子分解并重构 `Σ = BFB' + D`。
- `daily_run` 与 `portfolio_eval` 已接入上述新方法，可通过 `portfolio.cov_shrinkage` 配置切换。

### 5.2 换手约束过于简单

`src/portfolio/weights.py` 有换手上限，但没有考虑**交易冲击成本**：大市值票的换手成本和小市值票差异很大。建议引入以市值分档的成本系数。

**修复状态（已完成）**：

- `src/portfolio/weights.py` 的换手约束已支持成本加权形式：
  `0.5 * sum(coeff_i * |Δw_i|) <= max_turnover`。
- 已新增按市值分档的成本系数构造（小盘更高、大盘更低），通过 `portfolio.turnover_cost_model` 配置启用。
- `daily_run` 与 `portfolio_eval` 已接入成本加权换手约束。

### 5.3 Regime 状态切换粒度粗

`src/market/regime.py` 的三状态（牛/熊/震荡）对信号权重的影响是静态配置的。更优做法是用**隐马尔可夫模型（HMM）** 或基于波动率指标（如 VIX 类）的动态权重，而不是硬编码分档。

**修复状态（已完成）**：

- `src/market/regime.py` 已新增“趋势 + 波动率”连续动态权重机制（可配置开关）。
- `get_regime_weights` 在保留三状态标签的同时，可根据 `short_return` 与 `realized_vol_ann` 输出连续权重倍数，避免纯硬切换。
- `daily_run` 已将 `classify_regime` 的结果透传到动态权重计算链路。

---

## 六、LLM 链路深度集成

当前 LLM 模块与主流水线完全割裂，输出仅为独立 JSON 报告。可分两步集成：

**第一步（低风险）**：将 `attention_scanner.py` 的「关注度得分」作为一个**软因子**加入线性组合，权重设小（5% 以内），先观察 IC。

**第二步（中期）**：将个股新闻情绪分（`news_analyzer.py`）按时间戳对齐到日线，构造「情绪动量因子」，纳入树模型特征。需严格处理**前视偏差**（只用截止收盘前已发布的新闻）。

---

## 七、回测与评估

### 7.1 Walk-forward 窗口需延长

当前短测试窗导致年化波动极大，单个 fold 的结论不具统计显著性。建议测试窗至少 **63 个交易日（约 3 个月）**，训练窗 **252 天以上**。

**修复状态（已完成）**：

- `scripts/run_backtest_eval.py` 的 rolling walk-forward 已固定使用 `train_days=252 / test_days=63 / step_days=63`。
- 时间切片验证 `contiguous_time_splits` 也已提高最小训练窗约束（默认 `min_train_days=252`），避免短窗误判。

### 7.2 缺少基准对比

回测结果未与沪深300 / 中证500 / 等权指数做**超额收益（Alpha）** 对比，无法判断系统贡献的真实价值。

**修复状态（已完成）**：

- `scripts/run_backtest_eval.py` 新增多基准对比：`510300(沪深300ETF)`、`510500(中证500ETF)` 与全市场等权基准。
- 报告输出与 JSON 均新增 `benchmarks` 与 `excess_vs_benchmarks`，可直接查看策略相对各基准的年化/夏普/回撤超额。

### 7.3 滑点模型过于乐观

`BacktestConfig` 的 `execution_mode` 使用 close-to-close，现实中尾盘集合竞价流动性有限，大单冲击不可忽视。建议补充 **VWAP 执行模式**。

**修复状态（已完成）**：

- `src/backtest/engine.py` 的 `BacktestConfig.execution_mode` 已支持 `vwap`。
- `vwap` 模式在调仓日按换手额外扣减执行冲击（可配置 `vwap_slippage_bps_per_side`、`vwap_impact_bps`），用于近似模拟 VWAP 成交与冲击成本。
- `scripts/run_backtest_eval.py` 已支持从配置读取并使用 `vwap` 执行模式，回测报告会记录实际执行口径。

---

## 八、工程质量

| 问题 | 当前状态 | 建议 |
|------|----------|------|
| CI/CD | 已接入自动化测试流水线 | GitHub Actions 在 push / PR 触发 `pytest` |
| 日志 | 已支持结构化日志 | 默认 JSON 行日志，可配置回退 text |
| `data/` 纳入 git | 已明确排除 | `.gitignore` 已排除 `data/` 与 DuckDB 产物 |
| 测试覆盖 | 已补齐端到端链路 | 新增 fetch → factor → rank → backtest 集成测试 |

**修复状态（已完成）**：

- 新增 `.github/workflows/pytest.yml`，每次 push / PR 自动执行 `pytest`。
- `src/logging_config.py` 新增 JSON line formatter，入口脚本支持 `logging.format` 配置（`json` / `text`）。
- `config.yaml.example` 新增 `logging.format` 示例配置，默认 `json`。
- `tests/test_e2e_pipeline.py` 新增离线 e2e 集成测试，覆盖 fetch → factor → rank → backtest 主链路。
- `.gitignore` 已明确排除 `data/`、`*.duckdb`、`*.duckdb.wal`，避免大文件入库。

---

## 优先级路线图

```
P0（本周）:   修复 Dockerfile COPY config.yaml / 补全 requirements
P1（本月）:   XGBoost bundle IC 方向核查 + 中性化默认启用
P2（Q2）:     因子 IC 监控 + 模型版本管理 + walk-forward 窗口调整
P3（Q3）:     LLM 情绪因子集成 + Regime 动态化 + 行业协方差模型
```

---

*文档生成自系统优化分析；随实现进度可在此文件勾选或追加条目。*
