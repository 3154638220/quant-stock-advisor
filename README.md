# quant-system

本仓库是一套本地化 A 股量化研究与荐股流水线：用 AkShare 增量写入 DuckDB，构建日线量价、行业宽度、资金流、基本面与可选 LLM 事件特征，按统一的 T+1/open-to-open 口径生成推荐名单、研究报告与回测证据。默认不推送云端，运行产物落在 `data/` 下。

当前代码里有两条需要明确区分的链路：

| 链路 | 入口 | 状态 | 输出 |
| --- | --- | --- | --- |
| 日更推荐链路 | `scripts/daily_run.py` | 本地日常运行入口，默认 `composite_extended`；可切换 XGBoost / 深度序列模型 | `data/results/recommend_*.csv`、推荐解释报告、eval 标注 |
| 月度选股链路 | `scripts/run_monthly_selection_*.py` | 生产默认方法已定为 `U1_liquid_tradable + Top20 + indcap3 hard-cap baseline` | `monthly_selection_*` 数据集、leaderboard、Top20 月度名单 |

生产准入边界在 `configs/promoted/promoted_registry.json`。截至仓库当前记录 `2026-04-30`，active promoted 配置为 `monthly_selection_u1_top20_indcap3_hardcap_baseline`；未登记的研究结果在 promotion gate 通过前，不应写回 `config.yaml.example` 或本地生产 `config.yaml` 的默认主线。

算法与统计细节见 [docs/项目算法建模详解.md](docs/项目算法建模详解.md)，荐股流程见 [docs/荐股算法流程说明.md](docs/荐股算法流程说明.md)，当前研究计划见 [docs/plan.md](docs/plan.md)。

## 已实现能力

| 模块 | 说明 |
| --- | --- |
| 数据 | AkShare 日线增量、股票池缓存、DuckDB 存储、写入前质量检查、网络重试与本地缓存回退，位于 `src/data_fetcher/` |
| 可交易性 | ST/停牌近似、涨跌停板比例、次日开盘一字涨停不可买过滤、月度候选池 U0/U1/U2，位于 `src/market/tradability.py` |
| 日线因子 | GPU/CPU 张量批量计算动量、RSI、ATR、波动率、换手、量价相关、乖离、价格区间位置、周线 KDJ 与 K 线结构代理因子 |
| 多源因子 | PIT 基本面、资金流、股东户数、行业宽度、可选 LLM 新闻关注度情绪；月度链路做覆盖率和缺失标记诊断 |
| 特征处理 | 截面 winsorize、z-score、缺失填充、市值/行业中性化、可选 Löwdin 或 Gram-Schmidt 正交化 |
| 打分模型 | 线性 `composite` / `composite_extended`、XGBoost 截面排序、LSTM/GRU/TCN/Transformer 序列模型、月度 M6 learning-to-rank |
| 市场状态 | Regime Switch 根据基准短中期收益与波动率调整线性因子权重，可用 `510300` 或 `market_ew_proxy` |
| 组合权重 | 等权、分数权重、分层等权、风险平价、最小方差、均值-方差；支持 Ledoit-Wolf、EWMA、行业因子协方差 |
| 回测评估 | `tplus1_open`、`close_to_close`、`vwap` 执行口径，涨停买入失败 `idle` / `redistribute`，交易成本、walk-forward、切片诊断 |
| 研究治理 | 研究身份、配置快照、manifest、promoted registry，避免轻量诊断或 gray zone 结果误入生产 |

## 技术栈

- 数据：AkShare、DuckDB、Pandas / Polars / NumPy。
- 计算：PyTorch 张量因子，自动在 CUDA 不可用时回退 CPU。
- 机器学习：scikit-learn、XGBoost、PyTorch 序列模型。
- 组合优化：SciPy、Ledoit-Wolf / EWMA / 行业因子协方差。
- 运行环境：Python 3.10.x，可编辑安装 `pip install -e .`。

## 仓库结构

```text
.
├── config.yaml.example      # 本地生产/日更配置模板
├── config.yaml.backtest     # 当前 canonical 研究回测配置入口
├── configs/
│   ├── backtests/           # 历史研究配置快照与场景变体
│   ├── experiments/         # 临时探索配置
│   └── promoted/            # 生产 promotion registry
├── docs/
│   ├── README.md            # 文档目录说明
│   ├── plan.md              # 当前月度选股研究计划
│   ├── 荐股算法流程说明.md
│   ├── 项目算法建模详解.md
│   └── reports/             # 按月份归档的研究报告与证据链
├── src/
│   ├── data_fetcher/        # AkShare、DuckDB、质量检查、基本面/资金流/股东数据
│   ├── features/            # 因子、标准化、中性化、正交化、IC 评估
│   ├── market/              # 可交易性与 Regime Switch
│   ├── models/              # 打分、训练、推理、工件管理
│   ├── portfolio/           # 协方差、权重优化、约束
│   ├── backtest/            # 回测引擎、成本、绩效、walk-forward
│   ├── cli/                 # 推荐 CSV 事后评估
│   └── llm/                 # 本地 Ollama 新闻关注度扫描
├── scripts/
│   ├── daily_run.py                         # 日更推荐主入口
│   ├── fetch_only.py                        # 仅更新日线
│   ├── run_monthly_selection_dataset.py     # M2 月度 canonical dataset
│   ├── run_monthly_selection_oracle.py      # M3 oracle 上限诊断
│   ├── run_monthly_selection_baselines.py   # M4 baseline
│   ├── run_monthly_selection_multisource.py # M5 多源扩展
│   ├── run_monthly_selection_ltr.py         # M6 learning-to-rank
│   ├── run_monthly_selection_report.py      # M7 研究版推荐报告
│   ├── run_backtest_eval.py                 # 正式回测评估
│   └── train/                               # 离线训练入口
└── tests/
```

`data/`、`tmp/`、`.pytest_cache/`、`.ruff_cache/`、`__pycache__/`、`.conda_envs/`、`.conda_pkgs/`、`.miniforge3/` 是本机私有或可再生产物，默认不进入版本控制和 Docker build context。

## 环境安装

Python 版本要求是 3.10.x，推荐 3.10.12。

```bash
cp config.yaml.example config.yaml
```

可通过环境变量覆盖配置路径：

```bash
export QUANT_CONFIG=/absolute/path/to/config.yaml
```

Conda / Jetson 推荐：

```bash
conda env create -f environment.yml
conda activate quant-system
bash scripts/bootstrap_conda_env.sh
```

x86_64 桌面环境可直接：

```bash
pip install -r requirements.txt
pip install -e .
```

Jetson 上不要用 `requirements.txt` 安装 PyTorch，应走 `bootstrap_conda_env.sh` 或 NVIDIA 官方 wheel。环境自检：

```bash
python scripts/env_check.py
```

AkShare 网络不稳时可先跑：

```bash
python scripts/akshare_network_doctor.py
```

Docker / Jetson 辅助脚本仍保留：

```bash
bash scripts/docker_build_jetson.sh
bash scripts/docker_run.sh
```

Jetson 构建优先使用 NVIDIA 官方 PyTorch wheel；若遇到 cuSPARSELt 前置依赖问题，可参考 `scripts/install_cusparselt.sh`。

## 快速开始

仅更新日线数据库：

```bash
python scripts/fetch_only.py --max-symbols 50
```

运行日更推荐：

```bash
python scripts/daily_run.py --max-symbols 50
python scripts/daily_run.py run --skip-fetch --top-k 20
python scripts/daily_run.py --symbols 600519,000001 --skip-fetch
```

评估与解释已有推荐：

```bash
python scripts/daily_run.py eval --latest --horizon 5
python scripts/recommend_report.py --latest
```

离线训练日更模型：

```bash
python scripts/train/train_xgboost.py --config config.yaml
python scripts/train/train_deep_sequence.py --config config.yaml --kind gru --seq-len 30
```

运行月度选股研究链路：

```bash
python scripts/run_monthly_selection_dataset.py --config config.yaml.backtest
python scripts/run_monthly_selection_oracle.py --config config.yaml.backtest
python scripts/run_monthly_selection_baselines.py --config config.yaml.backtest
python scripts/run_monthly_selection_multisource.py --config config.yaml.backtest
python scripts/run_monthly_selection_ltr.py --config config.yaml.backtest
python scripts/run_monthly_selection_report.py --config config.yaml.backtest
```

数据扩展入口：

```bash
python scripts/build_industry_map.py
python scripts/fetch_fundamental.py --max-symbols 100
python scripts/fetch_fund_flow.py --max-symbols 100
python scripts/fetch_shareholder.py --latest-n 4
```

本地 LLM 新闻关注度扫描需要 Ollama 正在运行且模型已拉取：

```bash
python scripts/llm_daily_analysis.py --with-macro
python scripts/llm_analysis_report.py --latest
```

## 配置边界

| 配置 | 用途 |
| --- | --- |
| `config.yaml.example` | 本地生产/日更模板，默认不承载未 promotion 的研究候选 |
| `config.yaml` | 本机私有运行配置，可由 `QUANT_CONFIG` 覆盖 |
| `config.yaml.backtest` | 当前 canonical 研究回测入口 |
| `configs/backtests/` | 历史研究快照与变体 |
| `configs/promoted/promoted_registry.json` | 唯一生产准入注册表 |

关键配置段：

- `paths`：DuckDB、结果、日志、模型与实验目录。
- `database`：日线、基本面、资金流、股东表名。
- `features`：窗口、K 线结构代理因子、是否做市值中性化。
- `signals`：`top_k`、`sort_by`、线性权重、树模型/序列模型工件、IC 动态权重。
- `prefilter`：日更硬过滤规则。
- `regime`：大盘状态分类与动态权重。
- `portfolio`：权重方法、单票上限、行业约束、协方差估计、换手约束。
- `backtest`：执行口径、调仓频率、涨停买入失败处理。
- `transaction_costs`：佣金、滑点、印花税，默认双边约 19 bps。
- `llm`：Ollama 模型、超时、扫描规模与新闻回看窗口。

## 当前研究纪律

月度选股当前生产默认方法为 `U1_liquid_tradable + Top20 + indcap3 hard-cap baseline`，对应 promoted config `monthly_selection_u1_top20_indcap3_hardcap_baseline`。交易时点为：持有月最后一个交易日卖出当月 Top20，下一交易日开盘买入下一月 Top20。其他 M5/M6/M8 变体仍按研究产物处理，除非进入 `configs/promoted/promoted_registry.json`。

旧 replacement / sleeve / gray zone 候选已经在 2026-04 的报告中被冻结或拒绝。轻量 `signal_diagnostic`、`light_strategy_proxy`、daily proxy、gray zone 都不是 promotion 终点。只有正式 full backtest、OOS、状态切片、边界诊断、成本、universe 与人工确认完整可追溯后，才允许新增 promoted 配置。

## 测试

```bash
pip install -e ".[dev]"
pytest
```

如果只改文档，通常不需要跑完整测试；改代码、配置或默认策略时至少应跑相关 pytest，并保留生成的研究 manifest / report。

## 风险提示

本系统输出的是研究与筛选结果，不构成任何投资建议。实际交易仍需结合流动性、公告、行业事件、仓位管理和个人风险承受能力独立判断。
