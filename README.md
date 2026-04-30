# quant-system

本地化 **A 股**量化流水线：用 **AkShare** 增量写入 **DuckDB**，在 **GPU（PyTorch）** 上批量计算量价因子，按配置生成每日推荐池（CSV），并支持组合权重、回测与 walk-forward 等研究能力。默认**不推送云端**，结果落在 `data/results/`。

---

## 已实现能力（代码现状）


| 模块                | 说明                                                                                                                                                                     |
| ----------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **数据**            | AkShare 日线增量、股票池缓存、写入前数据质量检查（`src/data_fetcher/`）                                                                                                                      |
| **新闻驱动关注度扫描**     | 从东方财富飙升榜/人气榜出发，用 Ollama + qwen2.5 识别因真实新闻事件（而非市场波动）导致关注度飙升的股票；可选市场宏观分析与财务指标补充（`src/llm/`、`scripts/llm_daily_analysis.py`）                                              |
| **因子**            | GPU 张量动量/RSI/ATR/波动率/换手/量价相关；K 线结构高频降频代理因子（日内振幅、影线比率、隔夜跳空、尾盘强度等）；截面 winsorize、z-score、行业与市值双重中性化（`src/features/`）                                                      |
| **因子正交化**         | Löwdin 对称正交化与 Gram-Schmidt 正交化，在传入 XGBoost/线性模型前剥离因子冗余（`src/features/orthogonalize.py`，训练时 `--orthogonalize` 启用）                                                       |
| **信号**            | 线性 composite / composite_extended；**XGBoost** 截面 **Learning-to-Rank**（默认 `XGBRanker`，可选回归）；**LSTM/GRU/TCN/Transformer** 序列模型（`src/models/`、`scripts/train/train_*.py`） |
| **Regime Switch** | 大盘状态分类器（牛/熊/震荡），根据基准近期收益与波动率动态调整 composite_extended 因子权重；可在 `config.yaml`（由 `config.yaml.example` 复制）中 `regime:` 节配置（`src/market/regime.py`）                           |
| **组合**            | 等权/按得分；**Ledoit–Wolf 收缩协方差** + ridge；ERC / 最小方差 / 均值–方差（SciPy）；单票与行业上限、换手约束（`src/portfolio/`）                                                                          |
| **回测与评估**         | 向量回测引擎（含涨停买入失败重分配 `redistribute` 模式）；双边约 19 bps 保守成本假设；绩效面板、IC/分层、时间切片与 rolling walk-forward（`src/backtest/`）                                                          |
| **日更入口**          | `scripts/daily_run.py`（拉数→因子→Regime Switch→推荐 CSV，输出列含 `regime`）；`scripts/fetch_only.py`（仅拉数）                                                                          |
| **事后评估**          | `daily_run.py eval`：对已有 `recommend_*.csv` 标注前向收益（读 DuckDB）                                                                                                             |


算法与统计细节见 **[项目算法建模详解.md](docs/项目算法建模详解.md)**。

---

## 技术栈

- **数据**：AkShare → DuckDB（路径见 `config.yaml`，初始模板为 `config.yaml.example`）
- **计算**：Polars / NumPy；因子主路径 **PyTorch**（Jetson 上**不默认**引入 RAPIDS）
- **机器学习**：scikit-learn 基线、XGBoost、PyTorch 序列模型
- **组合优化**：SciPy（SLSQP 等）
- **包管理**：Python **3.10.x**（`requires-python` 与 `.python-version` 为 3.10.12）、可编辑安装 `pip install -e .`

---

## 仓库结构

```text
.
├── config.yaml.example      # 全局配置模板（先复制为 config.yaml 再按本机调整）
├── config.yaml              # 本地运行配置（建议不入库，支持 QUANT_CONFIG 覆盖路径）
├── config.yaml.backtest     # 当前 canonical 研究回测快照
├── configs/
│   ├── promoted/            # production promotion registry
│   ├── backtests/           # 历史研究配置快照与场景变体
│   └── experiments/         # 临时/探索性配置，按主题归档
├── pyproject.toml           # 包名 quant-system、pytest 配置
├── environment.yml          # Conda 环境 quant-system（Python 3.10）
├── requirements-base.txt    # 通用 pip 依赖（不含 Jetson 专用 torch）
├── requirements.txt         # x86：额外从 PyPI 装 torch
├── data/                    # DuckDB、缓存、日志、结果 CSV、模型工件（默认路径）
├── docs/                    # 文档与报告
│   ├── README.md            # 文档目录边界
│   ├── 项目算法建模详解.md   # 算法与统计细节说明
│   ├── 荐股算法流程说明.md
│   ├── backtest_report.md
│   ├── plan.md
│   └── reports/             # 按月份归档的研究报告与证据链
├── src/
│   ├── data_fetcher/        # 拉数、DuckDB、质量检查
│   │   └── akshare_resilience.py  # 网络超时/重试/快照回退
│   ├── features/
│   │   ├── tensor_alpha.py          # 动量、RSI 张量因子
│   │   ├── tensor_base_factors.py   # 扩展基础因子（波动率、换手、量价等）
│   │   ├── intraday_proxy_factors.py# K 线结构高频降频代理因子
│   │   ├── orthogonalize.py         # 因子正交化：Löwdin / Gram-Schmidt
│   │   ├── panel.py                 # 宽表透视
│   │   ├── standardize.py           # 截面标准化
│   │   ├── neutralize.py            # 截面/行业/市值中性化
│   │   ├── tree_dataset.py          # 树模型训练面板
│   │   └── factor_eval.py           # IC/分层评估
│   ├── models/              # 打分、基线、XGBoost、时序网络、推理
│   ├── portfolio/           # 协方差、优化、权重
│   ├── backtest/            # 回测（含涨停重分配）、成本、绩效、walk-forward
│   ├── market/
│   │   ├── tradability.py   # 涨跌停、停牌、预过滤
│   │   └── regime.py        # 大盘状态分类器
│   ├── cli/
│   │   └── eval_recommend.py# 推荐 CSV 前向收益标注（由 daily_run.py 调用）
│   └── llm/
│       ├── client.py              # Ollama 客户端封装
│       ├── prompts.py             # Prompt 模板（关注度判定、情绪、宏观、财务）
│       ├── attention_scanner.py   # 关注度飙升扫描 + LLM 新闻驱动过滤（核心）
│       ├── news_analyzer.py       # 市场宏观 / 个股新闻情绪（辅助）
│       └── financial_analyzer.py  # 财务报表提取（辅助）
├── scripts/
│   ├── daily_run.py         # 日更主入口（拉数→因子→推荐 CSV）
│   ├── fetch_only.py        # 仅增量更新数据库
│   ├── llm_daily_analysis.py# 新闻驱动关注度扫描入口
│   ├── llm_analysis_report.py# 扫描结果 → Markdown / HTML 报告
│   ├── run_backtest_eval.py # 回测评估
│   ├── train/               # 离线训练脚本入口
│   │   ├── train_baseline.py        # Ridge / ElasticNet / Random Forest
│   │   ├── train_xgboost.py         # 截面 XGBoost（Learning-to-Rank）
│   │   ├── train_deep_sequence.py   # OHLCV 序列深度模型（端到端入口）
│   │   └── train_timeseries.py      # 序列模型（CSV 面板输入）
│   └── ...                  # 环境、bootstrap、Docker 辅助
├── tests/                   # pytest
└── notebooks/               # 探索与原型（.gitkeep 占位）
```

本地运行产物边界：`data/`、`tmp/`、`.pytest_cache/`、`.ruff_cache/`、`__pycache__/`、`.conda_envs/`、`.conda_pkgs/`、`.miniforge3/` 都是可再生或本机私有内容，默认不进入版本控制和 Docker build context。

---

## 环境安装

### 版本要求

- **Python**：**3.10.x**（推荐 **3.10.12**，见 `.python-version` 与 `pyproject.toml`）

### 配置文件初始化

首次使用请先创建本地配置文件：

```bash
cp config.yaml.example config.yaml
```

可选：通过环境变量指定其他配置路径（脚本默认读取 `config.yaml`，不存在时会回退到 `config.yaml.example`）：

```bash
export QUANT_CONFIG=/absolute/path/to/config.yaml
```

研究回测默认入口是根目录 `config.yaml.backtest`；历史场景快照统一收纳在 `configs/backtests/`，主题探索配置放在 `configs/experiments/`。为了兼容旧报告和旧命令，`--config config.yaml.backtest.r7_s2_prefilter_off_universe_on` 这类旧快照名会自动解析到 `configs/backtests/` 下同名文件。

生产 promotion 边界记录在 `configs/promoted/promoted_registry.json`。截至 `2026-04-28`，registry 中 `promoted_configs` 为空，当前没有任何 P1/R2/R3 研究候选进入生产；`daily proxy` 与 `gray zone` 均不能替代正式 promotion。

### Conda（推荐，含 Jetson）

```bash
conda env create -f environment.yml
conda activate quant-system
bash scripts/bootstrap_conda_env.sh
```

`bootstrap_conda_env.sh` 会安装 `requirements-base.txt`、**NVIDIA Jetson 官方 PyTorch wheel**（默认 URL 在 `jetson/torch-wheel.url`，可用环境变量 `TORCH_WHEEL_URL` 覆盖）、并执行 `pip install -e .`。若 wheel 为较新版本，可能需按 [NVIDIA PyTorch for Jetson](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html) 先装 **cuSPARSELt** 等前置依赖（仓库提供 `scripts/install_cusparselt.sh` 可参考）。

### 桌面 x86_64

可直接：

```bash
pip install -r requirements.txt
pip install -e .
```

**不要在 aarch64（Jetson）上用 `requirements.txt` 安装 PyTorch**，应走上面的 bootstrap。

### Docker（L4T / Jetson）

基础镜像与构建说明见仓库根目录 `Dockerfile`。推荐一键构建：

```bash
bash scripts/docker_build_jetson.sh
```

（默认 `docker build --network=host`，避免部分宿主机上 **build 阶段 DNS 解析失败**导致 pip 无法访问 PyPI。）若你已在 `/etc/docker/daemon.json` 中配置 `"dns": ["8.8.8.8","1.1.1.1"]` 等，也可直接 `docker build -t my-jetson-pytorch:latest .`。

基础镜像自带的 NVIDIA pip 源（`jetson.webredirect` / NGC）与 PyPI 混用时，在部分网络下会出现 pip 与 PyPI 的 SSL 异常；当前 `Dockerfile` 已在构建阶段改为 **仅使用 PyPI** 并移除用户级 NVIDIA pip 配置。

构建完成后，一键进入容器与烟测可使用 `scripts/docker_run.sh`（需 `nvidia-container-toolkit`）。

### 环境自检

```bash
conda activate quant-system
python scripts/env_check.py
```

若本机网络偶发抖动、`AkShare` 常见 `ReadTimeout` / SSL / DNS 问题，可先做网络诊断：

```bash
python scripts/akshare_network_doctor.py
```

需要把 DNS 固化到 `systemd-resolved` 与 Docker（Jetson/Linux，需 root）时，可执行：

```bash
sudo python scripts/akshare_network_doctor.py --apply-dns
```

未手动 `activate` 时可用：

```bash
bash scripts/with_conda.sh python scripts/daily_run.py --help
```

---

## 快速开始

**仅更新数据库（不跑因子）：**

```bash
python scripts/fetch_only.py --max-symbols 50
```

**完整流水线（默认子命令 `run`）：**

```bash
python scripts/daily_run.py --max-symbols 50
python scripts/daily_run.py run --skip-fetch --top-k 20
python scripts/daily_run.py --symbols 600519,000001
```

**对最新推荐 CSV 做前向收益标注：**

```bash
python scripts/daily_run.py eval --latest --horizon 5
python scripts/daily_run.py eval --csv data/results/recommend_2026-03-27.csv
```

**新闻驱动关注度扫描（需 Ollama 运行中，模型已下载）：**

从东方财富飙升榜出发，逐只拉取近期新闻，用本地 LLM 判断关注度飙升是否由真实新闻事件驱动（而非股价涨停/板块联动等市场波动），输出经过滤的新闻驱动型高关注度股票。

```bash
# 默认：扫描飙升榜前 50 只，LLM 逐只判定
python scripts/llm_daily_analysis.py

# 扫描前 30 只，使用轻量模型
python scripts/llm_daily_analysis.py --top-k 30 --model qwen2.5:3b

# 指定分析基准时刻（用于区分盘中/盘后新闻时序）
python scripts/llm_daily_analysis.py --analysis-time 2026-03-30T15:30:00

# 同时做市场宏观分析
python scripts/llm_daily_analysis.py --with-macro

# 对新闻驱动的标的额外做财务指标提取
python scripts/llm_daily_analysis.py --with-financial

# 生成 Markdown / HTML 报告
python scripts/llm_analysis_report.py --latest
```

输出文件：

- `data/results/llm_attention_<date>.json` — 完整扫描结果（含全部飙升股 + LLM 判定）
- `data/results/llm_attention_<date>.csv` — 仅新闻驱动股票（催化剂、显著性、类别等）
- `data/results/llm_report_<date>.md / .html` — 可读报告（由 `llm_analysis_report.py` 生成）

**本地算力与模型（默认 `qwen2.5:32b`）：**

- **换模型**：`config.yaml`（或 `QUANT_CONFIG` 指向文件）中的 `llm.model` 默认 `qwen2.5:32b`；命令行 `--model` 可覆盖；先 `ollama pull <名称>`。降级可选 `qwen2.5:14b`、`qwen2.5:7b`。Q4 量化显存粗估：7B≈5GB、14B≈10GB、32B≈20GB、70B≈40GB+。
- **吃满 GPU**：`llm.ollama_options` 中 `num_gpu: -1`、`num_ctx: 8192`。
- **超时**：大模型推理慢时提高 `llm.timeout_sec`（300~600），或 `--timeout 600`。
- **扫描量**：`llm.attention.max_surge_stocks` 控制扫描飙升榜前 N 只，越大耗时越长（每只需一次 LLM 推理）。
- **去噪窗口**：`llm.attention.news_lookback_hours` 默认 48，仅保留近窗口新闻并按时间倒序喂给模型，降低“旧新闻干扰”。
- **时序锚点**：可通过 `--analysis-time` 传入分析基准时刻，帮助模型区分盘中驱动与盘后复盘叙事。

**烟测（固定标的、接口不稳时）：**

```bash
bash scripts/verify_first_run.sh
# 可选：VERIFY_SYMBOLS=600519,000001 bash scripts/verify_first_run.sh
```

推荐结果默认写入 `data/results/recommend_*.csv`（含 `regime` 列）；`eval` 默认生成 `data/results/eval_*.csv`。

**排序键**：`--sort-by` 可选 `momentum`、`rsi`、`composite`、`composite_extended`、`xgboost`、`deep_sequence`；日常荐股默认使用 `xgboost`（树模型与序列模型需先训练并在 `config.yaml` 中配置工件路径）。完整参数见：

```bash
python scripts/daily_run.py --help
```

---

## 配置要点（`config.yaml` / `config.yaml.example`）

- **运行配置边界**：`config.yaml.example` 和本地 `config.yaml` 服务生产/日更链路；`config.yaml.backtest` 与 `configs/backtests/` 服务研究回测链路；`configs/promoted/promoted_registry.json` 记录允许进入生产的 promoted 配置。未 promotion 的研究结论不要写回生产配置。
- **paths**：`duckdb_path`、`results_dir`、`models_dir` 等
- **akshare**：复权、超时、重试、股票池缓存策略
- **akshare**：现还包含 HTTP 连接/读超时、HTTP 重试、本地快照回退与 DNS 诊断脚本配置
- **akshare**：日线源优先级 `daily_source_preference` 与批量并发 `fetch_workers`（当前机器默认 2）
- **gpu**：`device`（`cuda`/`cpu`）、`batch_symbols`、`dtype`
- **features**：动量/RSI/ATR 窗口、回看交易日、winsorize 分位、K 线结构因子窗口（`tail_window`、`vpt_window`、`range_skew_window`）
- **prefilter**：股票池硬规则预过滤（ST、涨跌停次数、换手极值、绝对高位）
- **orthogonalize**：`enabled`（是否正交化）、`method`（`symmetric` / `gram_schmidt`）
- **signals**：`top_k`、`sort_by`、composite 权重、`composite_extended`（含 K 线结构因子权重）、树模型与 `deep_sequence` 段
- **regime**：`enabled`、`short_window`、`long_window`、各市场状态的因子权重倍数
- **portfolio**：权重模式（默认 `risk_parity`）、风险模型、单票/行业上限（默认 5%）、换手与成本
- **backtest**：`execution_mode`（`close_to_close` / `tplus1_open` / `vwap`，默认 `tplus1_open`）、`limit_up_mode`（`idle` / `redistribute`）、`vwap_slippage_bps_per_side`、`vwap_impact_bps`
- **transaction_costs**：`slippage_bps_per_side` 默认调整为 4.5 bps，合计双边约 **19 bps**
- **llm**：`model`、`timeout_sec`、`ollama_options`；`llm.attention` 子节控制关注度扫描参数（`max_surge_stocks`、`max_news_per_stock`、`news_lookback_hours`）

---

## 模型训练（离线）

`scripts/train/` 下为训练入口脚本，例如：

- `train_baseline.py` — Ridge / ElasticNet / Random Forest  
- `train_xgboost.py` — 截面 XGBoost（含新增 K 线结构因子与 `--orthogonalize` 选项）
- `train_deep_sequence.py`、`train_timeseries.py` — 序列深度模型

训练产出默认落在 `data/models`、`data/experiments`（与 `config.yaml` 中路径一致）。

**升级与重训（与当前代码对齐时）：**

- **树模型**：若仍使用旧版工件，请重新运行 `python scripts/train/train_xgboost.py --config config.yaml`。现在 `default_tree_factor_names()` 包含 8 个新 K 线结构因子（`intraday_range`、`upper_shadow_ratio` 等），特征列须对齐。可选 `--orthogonalize` 在训练前对截面因子矩阵做 Löwdin 正交化；正交化方法由 `config.yaml` 中 `orthogonalize.method` 控制，也可通过 `--orthogonalize-method gram_schmidt` 指定。
- **标签变换**：支持 `--label-transform raw/sharpe/calmar/truncate`，默认从 `config.yaml` 的 `label.transform` 读取（当前为 `truncate`，截断极端收益顶 2%）。
- **深度序列**：重训后新 checkpoint 在输出头前多了 **Dropout**；推理加载已使用 `**strict=False`**，旧 bundle 仍可加载，但建议用新超参重训。

---

## 测试

```bash
pip install -e ".[dev]"
pytest
```

---

## 研究流程

当前仓库建议把研究链路固定为下面这个顺序，避免把轻量诊断、scout 和正式回测混在一起解释：

1. `fetch / derive`
   先更新 DuckDB 与派生数据，必要时刷新 `prepared_factors` cache。
2. `light diagnostic`
   先用 `scripts/run_signal_diagnostic.py` 看信号方向、频率感知年化、波动、回撤和基准相对表现。
   这一层现在统一标记为 `result_type=signal_diagnostic`，是独立的 canonical 轻量诊断产物，不等价于正式 full backtest。
   该脚本现在也会像 scout/admission 一样输出统一的 `research_topic / research_config_id / output_stem`，默认产物名为 `{output_stem}_summary.csv`、`{output_stem}_period_detail.csv` 和 `docs/{output_stem}.md`。
3. `scout`
   用 `scripts/run_alpha_factor_scout.py`、`scripts/run_alpha_directional_scout.py`、`scripts/run_alpha_expression_scout.py` 做 benchmark-first 候选侦察。
   这类结果继续统一标记为 `result_type=light_strategy_proxy`，用于候选筛查，不直接替代 admission/full backtest。
4. `P1 daily-proxy-first`
   P1 树模型不再用旧 light proxy 触发正式回测。`scripts/run_p1_tree_groups.py` 默认输出 `result_type=daily_bt_like_proxy`、`*_daily_proxy_leaderboard.csv`、状态切片和 Top-K 边界诊断；旧 `light_strategy_proxy` / `full_like_proxy` 仅保留为 `legacy_proxy_decision_role=diagnostic_only`。
   默认 gate 是 `<0%` 直接 reject，`0%~+3%` 归档为 gray zone，`>=+3%` 才允许补正式 full backtest。
5. `admission`
   候选只有在 `IC gate + combo gate + benchmark-first gate` 都过线后，才进入正式准入判断。
   对应脚本是 `scripts/run_factor_admission_validation.py`。
6. `full backtest`
   最终用 `scripts/run_backtest_eval.py` 做正式回测、walk-forward 和切片验证。
7. `optional publish`
   只有研究结论稳定后，才考虑回写默认配置或接入日更推荐链路。

这个顺序的核心纪律是：

- 不把 `signal_diagnostic` 或 `light_strategy_proxy` 数值直接当成正式策略收益。
- P1 旧 proxy 只做 legacy diagnostic，不参与 admission、full backtest 触发或 promotion。
- P1 daily proxy 只负责准入和正式回测触发，不替代正式 full backtest。
- 不跳过 scout/admission 直接把新因子写进默认主线。
- `prepared_factors` cache 命中前提必须与当前 schema/version 一致，否则应重建。

---

## 路线图（未默认落地）

以下在 **[项目算法建模详解.md](docs/项目算法建模详解.md)** 中亦标注为规划或可选扩展：**GNN**（板块/产业链图）、**深度强化学习**（如 FinRL）、新闻驱动关注度因子接入量化流水线等（当前 LLM 关注度扫描独立输出，暂未接入因子打分体系）。当前仓库主线仍是日线量价 + 统计/树/序列模型 + 凸优化组合 + 向量回测。

---

## Jetson 与算力预期（简述）

- **ARM + JetPack**：优先使用 NVIDIA 提供的 **PyTorch** 构建做 GPU 因子；**弱化 RAPIDS** 默认依赖，降低维护成本。  
- **统一内存**利于较大模型与批量张量任务；**CUDA 核心数与带宽**通常低于同代桌面旗舰，大规模全图 GNN 或高并行 RL 需按「可跑通、可迭代」设预期。

更完整的硬件与四条算力路径讨论，若仍需保留可参考历史文档或自行归档至 `docs/`（当前以代码与 `项目算法建模详解.md` 为准）。
