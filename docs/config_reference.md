# 配置项参考

本文档列出 `config.yaml` / `config.yaml.example` 中与 2026-05-03 改进计划相关的核心新增配置。

## database

| 字段 | 类型 | 默认值 | 有效值 | 说明 |
|---|---:|---:|---|---|
| `price_adjust` | string | `qfq` | `qfq` / `hfq` / `none` | 日线价格复权口径。价格类特征应使用前复权数据，避免除权跳变污染信号。 |

## signals.ic_weighting

| 字段 | 类型 | 默认值 | 有效值 | 说明 |
|---|---:|---:|---|---|
| `enabled` | bool | `true` | `true` / `false` | 是否启用滚动 ICIR 动态因子权重。 |
| `window` | int | `20` | `>= 2` | 计算 ICIR 的滚动期数。 |
| `min_periods` | int | `12` | `1..window` | 计算滚动 ICIR 所需的最少观测数。 |
| `softmax_temperature` | float | `2.0` | `> 0` | ICIR softmax 温度，越大权重越均匀。 |
| `weights_path` | path | `data/cache/ic_weights.json` | 任意路径 | IC 权重缓存文件；存在时优先读取。 |
| `monitor_path` | path | `data/logs/ic_monitor.json` | 任意路径 | JSONL 兼容 IC 监控记录路径。 |
| `half_life` | float | `20.0` | `> 0` | 旧版权重更新脚本的指数衰减半衰期。 |
| `clip_abs_weight` | float | `0.25` | `0..1` | 旧版权重更新脚本的单因子权重绝对值上限。 |

## portfolio

| 字段 | 类型 | 默认值 | 有效值 | 说明 |
|---|---:|---:|---|---|
| `covariance_method` | string | `auto` | `sample` / `ledoit_wolf` / `auto` / `ewma` / `factor` / `industry_factor` | 组合优化协方差估计方法。`auto` 在样本协方差病态时切换 Ledoit-Wolf。 |
| `condition_number_threshold` | float | `1000` | `> 0` | `auto` 模式下触发收缩估计的条件数阈值。 |
| `max_monthly_turnover` | float | `0.60` | `0..1` | 月度优化层目标换手上限。 |
| `max_turnover` | float | `1.0` | `0..1` | 组合权重构造中的半 L1 换手上限；`1.0` 表示不约束。 |
| `cov_lookback_days` | int | `60` | `>= 2` | 协方差估计使用的历史交易日窗口。 |
| `cov_ridge` | float | `1e-6` | `>= 0` | 协方差矩阵对角线 ridge，提升数值稳定性。 |
| `cov_shrinkage` | string | `ledoit_wolf` | `sample` / `ledoit_wolf` / `auto` / `ewma` / `factor` / `industry_factor` | 回测 runner 使用的协方差估计方式。 |
| `cov_ewma_halflife` | float | `20` | `> 0` | EWMA 协方差半衰期。 |
| `risk_aversion` | float | `1.0` | `> 0` | 均值方差优化风险厌恶系数。 |

## portfolio.turnover_cost_model

| 字段 | 类型 | 默认值 | 有效值 | 说明 |
|---|---:|---:|---|---|
| `enabled` | bool | `true` | `true` / `false` | 是否按市值分层调整换手约束成本系数。 |
| `size_col` | string | `log_market_cap` | 数值列名 | 用于区分大小市值的列。 |
| `small_cap_coeff` | float | `1.6` | `> 0` | 小市值换手惩罚系数。 |
| `mid_cap_coeff` | float | `1.0` | `> 0` | 中市值换手惩罚系数。 |
| `large_cap_coeff` | float | `0.7` | `> 0` | 大市值换手惩罚系数。 |
| `q_small` | float | `0.33` | `0..1` | 小市值分位阈值。 |
| `q_large` | float | `0.67` | `0..1` | 大市值分位阈值。 |

## backtest

| 字段 | 类型 | 默认值 | 有效值 | 说明 |
|---|---:|---:|---|---|
| `limit_up_mode` | string | `redistribute` | `idle` / `redistribute` | 涨停打开买入失败后的处理方式；仅约束新增/增持权重，不影响既有持仓收益。 |
| `vwap_slippage_bps_per_side` | float | `3.0` | `>= 0` | VWAP 执行模式单边滑点。 |
| `vwap_impact_bps` | float | `8.0` | `>= 0` | 旧版固定 VWAP 冲击成本。 |

## transaction_costs

| 字段 | 类型 | 默认值 | 有效值 | 说明 |
|---|---:|---:|---|---|
| `impact_model` | string | `sqrt_adv` | `fixed_bps` / `sqrt_adv` | 市场冲击模型。`sqrt_adv` 按成交参与率使用平方根律。 |
| `impact_k` | float | `0.10` | `>= 0` | 平方根律冲击系数。 |

## monthly selection configs

| 字段 | 类型 | 默认值 | 有效值 | 说明 |
|---|---:|---:|---|---|
| `availability_lag_days` | int | `45` | `>= 0` | 旧 PIT 兜底延迟，已弃用；优先使用实际披露日历。 |
| `pit_fallback_lag_days` | int | `45` | `>= 0` | 披露日历缺失时的保守兜底延迟。 |
| `hpo_enabled` | bool | `false` | `true` / `false` | 是否启用 XGBoost 时序交叉验证调参。 |
| `hpo_n_trials` | int | `30` | `>= 1` | Optuna trial 数。 |
| `hpo_cv_folds` | int | `3` | `>= 1` | 时序 CV 折数。 |
| `window_type` | string | `expanding` | `expanding` / `rolling` | Walk-forward 训练窗口类型。 |
| `halflife_months` | float | `36.0` | `>= 0` | 扩张窗口样本权重半衰期；`0` 表示等权。 |
