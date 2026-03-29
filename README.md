# quant-system

本地化 **A 股**量化流水线：用 **AkShare** 增量写入 **DuckDB**，在 **GPU（PyTorch）** 上批量计算量价因子，按配置生成每日推荐池（CSV），并支持组合权重、回测与 walk-forward 等研究能力。默认**不推送云端**，结果落在 `data/results/`。

---

## 已实现能力（代码现状）

| 模块 | 说明 |
|------|------|
| **数据** | AkShare 日线增量、股票池缓存、写入前数据质量检查（`src/data_fetcher/`） |
| **因子** | GPU 张量动量/RSI/ATR/波动率/换手/量价相关；K 线结构高频降频代理因子（日内振幅、影线比率、隔夜跳空、尾盘强度等）；截面 winsorize、z-score、行业与市值双重中性化（`src/features/`） |
| **因子正交化** | Löwdin 对称正交化与 Gram-Schmidt 正交化，在传入 XGBoost/线性模型前剥离因子冗余（`src/features/orthogonalize.py`，训练时 `--orthogonalize` 启用） |
| **信号** | 线性 composite / composite_extended；**XGBoost** 截面 **Learning-to-Rank**（默认 `XGBRanker`，可选回归）；**LSTM/GRU/TCN/Transformer** 序列模型（`src/models/`、`models/train_*.py`） |
| **Regime Switch** | 大盘状态分类器（牛/熊/震荡），根据基准近期收益与波动率动态调整 composite_extended 因子权重；可在 `config.yaml` 中 `regime:` 节配置（`src/market/regime.py`） |
| **组合** | 等权/按得分；**Ledoit–Wolf 收缩协方差** + ridge；ERC / 最小方差 / 均值–方差（SciPy）；单票与行业上限、换手约束（`src/portfolio/`） |
| **回测与评估** | 向量回测引擎（含涨停买入失败重分配 `redistribute` 模式）；双边约 19 bps 保守成本假设；绩效面板、IC/分层、时间切片与 rolling walk-forward（`src/backtest/`） |
| **日更入口** | `scripts/daily_run.py`（拉数→因子→Regime Switch→推荐 CSV，输出列含 `regime`）；`scripts/fetch_only.py`（仅拉数） |
| **事后评估** | `daily_run.py eval`：对已有 `recommend_*.csv` 标注前向收益（读 DuckDB） |

算法与统计细节见 **[项目算法建模详解.md](项目算法建模详解.md)**；迭代计划见 **[算法改进.md](算法改进.md)**。

---

## 技术栈

- **数据**：AkShare → DuckDB（路径见 `config.yaml`）
- **计算**：Polars / NumPy；因子主路径 **PyTorch**（Jetson 上**不默认**引入 RAPIDS）
- **机器学习**：scikit-learn 基线、XGBoost、PyTorch 序列模型
- **组合优化**：SciPy（SLSQP 等）
- **包管理**：Python **3.10.x**（`requires-python` 与 `.python-version` 为 3.10.12）、可编辑安装 `pip install -e .`

---

## 仓库结构

```text
.
├── config.yaml              # 全局配置（路径、GPU、因子、信号、组合、回测、regime 等）
├── pyproject.toml           # 包名 quant-system、pytest 配置
├── environment.yml          # Conda 环境 quant-system（Python 3.10）
├── requirements-base.txt    # 通用 pip 依赖（不含 Jetson 专用 torch）
├── requirements.txt         # x86：额外从 PyPI 装 torch
├── data/                    # DuckDB、缓存、日志、结果 CSV、模型工件（默认路径）
├── src/
│   ├── data_fetcher/        # 拉数、DuckDB、质量检查
│   ├── features/
│   │   ├── tensor_alpha.py          # 动量、RSI 张量因子
│   │   ├── tensor_base_factors.py   # 扩展基础因子（波动率、换手、量价等）
│   │   ├── intraday_proxy_factors.py# K 线结构高频降频代理因子（新增）
│   │   ├── orthogonalize.py         # 因子正交化：Löwdin / Gram-Schmidt（新增）
│   │   ├── panel.py                 # 宽表透视
│   │   ├── standardize.py           # 截面标准化
│   │   ├── neutralize.py            # 截面/行业/市值中性化
│   │   ├── tree_dataset.py          # 树模型训练面板
│   │   └── factor_eval.py           # IC/分层评估
│   ├── models/              # 打分、基线、XGBoost、时序网络、推理
│   ├── portfolio/           # 协方差、优化、权重
│   ├── backtest/            # 回测（含涨停重分配）、成本、绩效、walk-forward
│   └── market/
│       ├── tradability.py   # 涨跌停、停牌、预过滤
│       └── regime.py        # 大盘状态分类器（新增）
├── models/                  # 训练脚本入口（baseline / xgboost / deep_sequence 等）
├── scripts/                 # 日更、环境、bootstrap、Docker 辅助
├── tests/                   # pytest
├── notebooks/               # 探索与原型
└── jetson/                  # Jetson torch wheel 默认 URL 等
```

---

## 环境安装

### 版本要求

- **Python**：**3.10.x**（推荐 **3.10.12**，见 `.python-version` 与 `pyproject.toml`）

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

基础镜像与构建说明见仓库根目录 `Dockerfile`；一键构建与烟测可使用 `scripts/docker_run.sh`（需 `nvidia-container-toolkit`）。

### 环境自检

```bash
conda activate quant-system
python scripts/env_check.py
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

## 配置要点（`config.yaml`）

- **paths**：`duckdb_path`、`results_dir`、`models_dir` 等
- **akshare**：复权、超时、重试、股票池缓存策略
- **gpu**：`device`（`cuda`/`cpu`）、`batch_symbols`、`dtype`
- **features**：动量/RSI/ATR 窗口、回看交易日、winsorize 分位、K 线结构因子窗口（`tail_window`、`vpt_window`、`range_skew_window`）
- **prefilter**：股票池硬规则预过滤（ST、涨跌停次数、换手极值、绝对高位）
- **orthogonalize**：`enabled`（是否正交化）、`method`（`symmetric` / `gram_schmidt`）
- **signals**：`top_k`、`sort_by`、composite 权重、`composite_extended`（含 K 线结构因子权重）、树模型与 `deep_sequence` 段
- **regime**：`enabled`、`short_window`、`long_window`、各市场状态的因子权重倍数
- **portfolio**：权重模式（默认 `risk_parity`）、风险模型、单票/行业上限（默认 5%）、换手与成本
- **backtest**：`execution_mode`（默认 `tplus1_open`）、`limit_up_mode`（`idle` / `redistribute`）
- **transaction_costs**：`slippage_bps_per_side` 默认调整为 4.5 bps，合计双边约 **19 bps**

---

## 模型训练（离线）

项目根目录 `models/` 下为训练入口脚本，例如：

- `train_baseline.py` — Ridge / ElasticNet / Random Forest  
- `train_xgboost.py` — 截面 XGBoost（含新增 K 线结构因子与 `--orthogonalize` 选项）
- `train_deep_sequence.py`、`train_timeseries.py` — 序列深度模型  

训练产出默认落在 `data/models`、`data/experiments`（与 `config.yaml` 中路径一致）。

**升级与重训（与当前代码对齐时）：**

- **树模型**：若仍使用旧版工件，请重新运行 `python models/train_xgboost.py --config config.yaml`。现在 `default_tree_factor_names()` 包含 8 个新 K 线结构因子（`intraday_range`、`upper_shadow_ratio` 等），特征列须对齐。可选 `--orthogonalize` 在训练前对截面因子矩阵做 Löwdin 正交化；正交化方法由 `config.yaml` 中 `orthogonalize.method` 控制，也可通过 `--orthogonalize-method gram_schmidt` 指定。
- **标签变换**：支持 `--label-transform raw/sharpe/calmar/truncate`，默认从 `config.yaml` 的 `label.transform` 读取（当前为 `truncate`，截断极端收益顶 2%）。
- **深度序列**：重训后新 checkpoint 在输出头前多了 **Dropout**；推理加载已使用 **`strict=False`**，旧 bundle 仍可加载，但建议用新超参重训。

---

## 测试

```bash
pip install -e ".[dev]"
pytest
```

---

## 路线图（未默认落地）

以下在 **[项目算法建模详解.md](项目算法建模详解.md)** 中亦标注为规划或可选扩展：**GNN**（板块/产业链图）、**深度强化学习**（如 FinRL）、**本地 LLM** 处理另类文本数据等。当前仓库主线仍是日线量价 + 统计/树/序列模型 + 凸优化组合 + 向量回测。

---

## Jetson 与算力预期（简述）

- **ARM + JetPack**：优先使用 NVIDIA 提供的 **PyTorch** 构建做 GPU 因子；**弱化 RAPIDS** 默认依赖，降低维护成本。  
- **统一内存**利于较大模型与批量张量任务；**CUDA 核心数与带宽**通常低于同代桌面旗舰，大规模全图 GNN 或高并行 RL 需按「可跑通、可迭代」设预期。  

更完整的硬件与四条算力路径讨论，若仍需保留可参考历史文档或自行归档至 `docs/`（当前以代码与 `项目算法建模详解.md` 为准）。
