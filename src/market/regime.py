"""
大盘状态分类器（Regime Switch / Market State Classifier）。

在系统最外层增加一个简单的「大盘状态分类器」，判断当前市场处于：
- **bull**（单边牛市）：大盘近期持续上涨，趋势因子信号可信度高
- **bear**（单边熊市）：大盘近期持续下跌，低波/防御因子权重应提升
- **oscillation**（震荡市）：大盘横向震荡，反转因子与均值回归因子更有效

根据市场状态动态调整 composite_extended 因子权重：
- 牛市：增加动量权重，减少反转权重
- 熊市：增加低波、反转、大市值权重，减少动量权重
- 震荡：均衡配置，反转和低波权重适度提升

Usage
-----
>>> from src.market.regime import classify_regime, get_regime_weights
>>> regime, meta = classify_regime(benchmark_series, asof_date)
>>> weights = get_regime_weights(base_weights, regime)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

REGIME_BULL = "bull"
REGIME_BEAR = "bear"
REGIME_OSCILLATION = "oscillation"

# 占位符：用全市场等权日收益（与回测脚本 ``build_market_ew_benchmark`` 一致）作为 regime 基准，
# 避免依赖沪深300 ETF（如 510300）是否入库。
MARKET_EW_PROXY = "market_ew_proxy"


@dataclass
class RegimeConfig:
    """大盘状态分类器超参数，可从 config.yaml 中的 regime 节读取。"""

    # 短期趋势窗口（交易日）
    short_window: int = 20
    # 中期趋势窗口（交易日）
    long_window: int = 60
    # 单边牛市判定阈值：短期收益 > bull_threshold 且波动率低于 vol_threshold
    bull_return_threshold: float = 0.05
    # 单边熊市判定阈值：短期收益 < -bear_threshold
    bear_return_threshold: float = 0.04
    # 波动率阈值（年化）：高波动倾向震荡市
    vol_threshold_ann: float = 0.25
    # 牛市动量权重倍数（相对基准权重）
    bull_momentum_multiplier: float = 1.5
    # 牛市反转权重倍数（压制）
    bull_reversal_multiplier: float = 0.3
    # 熊市低波权重倍数
    bear_lowvol_multiplier: float = 2.0
    # 熊市动量权重倍数（压制）
    bear_momentum_multiplier: float = 0.4
    # 熊市大市值权重倍数
    bear_size_multiplier: float = 1.5
    # 震荡市反转权重倍数
    oscillation_reversal_multiplier: float = 1.3
    # 是否启用基于波动率/趋势的连续动态权重（替代纯状态硬切换）
    dynamic_weighting_enabled: bool = True
    # 动态权重目标波动率（年化）；高于该值逐步降风险
    dynamic_vol_target_ann: float = 0.22
    # 波动率平滑尺度（越小越敏感）
    dynamic_vol_scale: float = 0.08
    # 趋势平滑尺度（按短期累计收益）
    dynamic_trend_scale: float = 0.05
    # 动态权重变化强度
    dynamic_strength: float = 3.0


def regime_config_from_mapping(m: Mapping[str, Any]) -> RegimeConfig:
    """从 config.yaml 的 regime 节加载超参数。"""
    d = dict(m) if m else {}
    return RegimeConfig(
        short_window=int(d.get("short_window", 20)),
        long_window=int(d.get("long_window", 60)),
        bull_return_threshold=float(d.get("bull_return_threshold", 0.05)),
        bear_return_threshold=float(d.get("bear_return_threshold", 0.04)),
        vol_threshold_ann=float(d.get("vol_threshold_ann", 0.25)),
        bull_momentum_multiplier=float(d.get("bull_momentum_multiplier", 1.5)),
        bull_reversal_multiplier=float(d.get("bull_reversal_multiplier", 0.3)),
        bear_lowvol_multiplier=float(d.get("bear_lowvol_multiplier", 2.0)),
        bear_momentum_multiplier=float(d.get("bear_momentum_multiplier", 0.4)),
        bear_size_multiplier=float(d.get("bear_size_multiplier", 1.5)),
        oscillation_reversal_multiplier=float(d.get("oscillation_reversal_multiplier", 1.3)),
        dynamic_weighting_enabled=bool(d.get("dynamic_weighting_enabled", True)),
        dynamic_vol_target_ann=float(d.get("dynamic_vol_target_ann", 0.22)),
        dynamic_vol_scale=float(d.get("dynamic_vol_scale", 0.08)),
        dynamic_trend_scale=float(d.get("dynamic_trend_scale", 0.05)),
        dynamic_strength=float(d.get("dynamic_strength", 3.0)),
    )


@dataclass
class RegimeResult:
    """大盘状态分类结果。"""

    regime: str  # bull / bear / oscillation
    short_return: float  # 短期滚动收益
    long_return: float   # 中期滚动收益
    realized_vol_ann: float  # 短期年化波动率
    n_days_used: int     # 实际使用天数
    meta: Dict[str, Any] = field(default_factory=dict)


def classify_regime(
    benchmark_series: pd.Series,
    asof_date,
    *,
    cfg: Optional[RegimeConfig] = None,
) -> Tuple[str, RegimeResult]:
    """
    根据大盘基准日收益序列对当前市场状态进行分类。

    Parameters
    ----------
    benchmark_series
        大盘基准日收益（不是价格）序列，索引为交易日，值为简单日收益。
        可使用沪深300 ETF（510300）日收益，或全市场等权日收益（与 ``market_ew_proxy`` 一致）。
    asof_date
        截面日期，截取 asof_date 及之前的数据。

    Returns
    -------
    regime_label : str
        "bull" / "bear" / "oscillation"
    result : RegimeResult
        含详细指标的结果对象
    """
    cfg = cfg or RegimeConfig()
    asof_ts = pd.Timestamp(asof_date).normalize()

    if benchmark_series.empty:
        return REGIME_OSCILLATION, RegimeResult(
            regime=REGIME_OSCILLATION,
            short_return=0.0,
            long_return=0.0,
            realized_vol_ann=0.0,
            n_days_used=0,
            meta={"reason": "empty_benchmark"},
        )

    s = benchmark_series.copy()
    s.index = pd.to_datetime(s.index).normalize()
    s = s[s.index <= asof_ts].sort_index()
    s = pd.to_numeric(s, errors="coerce").dropna()

    n = len(s)
    if n < max(cfg.short_window, 5):
        return REGIME_OSCILLATION, RegimeResult(
            regime=REGIME_OSCILLATION,
            short_return=0.0,
            long_return=0.0,
            realized_vol_ann=0.0,
            n_days_used=n,
            meta={"reason": "insufficient_data"},
        )

    short_w = min(cfg.short_window, n)
    long_w = min(cfg.long_window, n)

    short_rets = s.iloc[-short_w:].to_numpy()
    long_rets = s.iloc[-long_w:].to_numpy()

    # 累积收益（复利累积，避免简单求和在长窗口下的累积误差）
    short_return = float(np.prod(1.0 + short_rets) - 1.0)
    long_return = float(np.prod(1.0 + long_rets) - 1.0)

    # 短期年化波动率
    vol_daily = float(np.std(short_rets, ddof=0))
    vol_ann = vol_daily * np.sqrt(252.0)

    # 分类逻辑：短期趋势 + 波动率联合判断
    is_bull = (
        short_return > cfg.bull_return_threshold
        and long_return > 0.0
        and vol_ann < cfg.vol_threshold_ann
    )
    is_bear = (
        short_return < -cfg.bear_return_threshold
        or (short_return < 0 and long_return < -cfg.bear_return_threshold)
    )

    if is_bull:
        regime = REGIME_BULL
    elif is_bear:
        regime = REGIME_BEAR
    else:
        regime = REGIME_OSCILLATION

    result = RegimeResult(
        regime=regime,
        short_return=short_return,
        long_return=long_return,
        realized_vol_ann=vol_ann,
        n_days_used=n,
        meta={
            "short_window": short_w,
            "long_window": long_w,
            "vol_ann": vol_ann,
            "is_bull_conditions": is_bull,
            "is_bear_conditions": is_bear,
        },
    )
    return regime, result


def get_regime_weights(
    base_weights: Mapping[str, float],
    regime: str,
    *,
    cfg: Optional[RegimeConfig] = None,
    regime_result: Optional[RegimeResult] = None,
) -> Dict[str, float]:
    """
    根据市场状态动态调整因子权重。

    在 ``base_weights`` 基础上，根据大盘状态乘以对应倍数后重新归一化（按绝对值之和）。

    调整规则：
    - **牛市（bull）**：动量 ×1.5，反转因子乘以 0.3（压制反转倾向），高位惩罚适度保留
    - **熊市（bear）**：动量 ×0.4（压制趋势追击），低波 ×2.0，大市值 ×1.5
    - **震荡（oscillation）**：基本保持原权重，反转 ×1.3

    Parameters
    ----------
    base_weights
        原始因子权重字典（可含负权重，如 ``{"momentum": 0.25, "bias_short": -0.08}``）。
    regime
        "bull" / "bear" / "oscillation"

    Returns
    -------
    adjusted_weights : Dict[str, float]，按绝对值之和归一化
    """
    cfg = cfg or RegimeConfig()

    # 各状态对各类因子的调整倍数映射
    # 键：因子名关键字（子串匹配），值：倍数
    BULL_MULTIPLIERS = {
        "momentum": cfg.bull_momentum_multiplier,
        "reversal": cfg.bull_reversal_multiplier,
        "short_reversal": cfg.bull_reversal_multiplier,
        "recent_return": cfg.bull_reversal_multiplier,
    }
    BEAR_MULTIPLIERS = {
        "momentum": cfg.bear_momentum_multiplier,
        "realized_vol": cfg.bear_lowvol_multiplier,
        "atr": cfg.bear_lowvol_multiplier,
        "log_market_cap": cfg.bear_size_multiplier,
        "bias_short": cfg.bear_lowvol_multiplier,
        "bias_long": cfg.bear_lowvol_multiplier,
        "max_single_day_drop": cfg.bear_lowvol_multiplier,
    }
    OSCILLATION_MULTIPLIERS = {
        "short_reversal": cfg.oscillation_reversal_multiplier,
        "recent_return": cfg.oscillation_reversal_multiplier,
        "reversal": cfg.oscillation_reversal_multiplier,
    }

    def _pick_multiplier(factor_name: str, mapping: Mapping[str, float]) -> float:
        # 精确匹配优先，其次前缀匹配（key 后跟 "_" 或完全相等）
        if factor_name in mapping:
            return float(mapping[factor_name])
        for key, m in mapping.items():
            if factor_name.startswith(key + "_") or factor_name == key:
                return float(m)
        return 1.0

    if regime == REGIME_BULL:
        multipliers = BULL_MULTIPLIERS
    elif regime == REGIME_BEAR:
        multipliers = BEAR_MULTIPLIERS
    else:
        multipliers = OSCILLATION_MULTIPLIERS

    adjusted: Dict[str, float] = {}
    for factor, w in base_weights.items():
        mult = _pick_multiplier(factor, multipliers)
        # 动态模式：根据近期趋势和波动率连续调整，不再完全依赖离散状态。
        if cfg.dynamic_weighting_enabled and regime_result is not None:
            bull_m = _pick_multiplier(factor, BULL_MULTIPLIERS)
            bear_m = _pick_multiplier(factor, BEAR_MULTIPLIERS)
            osc_m = _pick_multiplier(factor, OSCILLATION_MULTIPLIERS)
            trend_scale = max(float(cfg.dynamic_trend_scale), 1e-6)
            vol_scale = max(float(cfg.dynamic_vol_scale), 1e-6)
            k = max(float(cfg.dynamic_strength), 1e-6)

            trend_score = np.tanh(float(regime_result.short_return) / trend_scale)
            vol_score = np.tanh(
                (float(regime_result.realized_vol_ann) - float(cfg.dynamic_vol_target_ann))
                / vol_scale
            )
            risk_on = 1.0 / (1.0 + np.exp(-k * (trend_score - vol_score)))
            high_vol = 1.0 / (1.0 + np.exp(-k * vol_score))
            blend_tb = risk_on * bull_m + (1.0 - risk_on) * bear_m
            mult = (1.0 - high_vol) * blend_tb + high_vol * osc_m
        adjusted[factor] = w * mult

    # 重新归一化（按绝对值之和，保留符号方向）
    total_abs = sum(abs(v) for v in adjusted.values())
    if total_abs < 1e-15:
        return dict(base_weights)

    # 保持总权重绝对值之和与原始一致
    orig_abs = sum(abs(v) for v in base_weights.values())
    scale = orig_abs / total_abs
    return {k: v * scale for k, v in adjusted.items()}


def _market_ew_daily_returns_from_frame(
    df: pd.DataFrame,
    *,
    date_col: str = "trade_date",
    close_col: str = "close",
    min_symbol_obs: int = 30,
) -> pd.Series:
    """
    由长表日线计算全市场等权日收益（与 ``scripts/run_backtest_eval.build_market_ew_benchmark`` 同口径）。
    """
    if df.empty or close_col not in df.columns:
        return pd.Series(dtype=float)
    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col]).dt.normalize()
    d = d[(d[close_col] > 0)].sort_values(["symbol", date_col])
    d["ret"] = d.groupby("symbol")[close_col].pct_change()
    d = d.dropna(subset=["ret"])
    if min_symbol_obs > 1:
        cnt = d.groupby("symbol")[date_col].count()
        good = cnt[cnt >= int(min_symbol_obs)].index
        d = d[d["symbol"].isin(good)]
    if d.empty:
        return pd.Series(dtype=float)
    cs = d.groupby(date_col)["ret"].mean()
    cs.index = pd.to_datetime(cs.index).normalize()
    return cs.sort_index()


def get_benchmark_returns_from_db(
    db,
    benchmark_symbol: str,
    *,
    lookback_days: int = 90,
    asof_date=None,
    date_col: str = "trade_date",
    close_col: str = "close",
) -> pd.Series:
    """
    从 DuckDB 读取基准标的（如 510300）的日收益序列。

    Parameters
    ----------
    db
        DuckDBManager 实例（上下文管理器内使用）。
    benchmark_symbol
        基准标的代码，如 "510300"；或使用占位符 ``market_ew_proxy`` 表示全市场等权收益。
    lookback_days
        回看天数（用于 BDay 推算起始日期）。
    asof_date
        截面日期，默认为今日。

    Returns
    -------
    pd.Series，索引为交易日，值为简单日收益。
    """
    if asof_date is None:
        asof_ts = pd.Timestamp("today").normalize()
    else:
        asof_ts = pd.Timestamp(asof_date).normalize()

    start = asof_ts - pd.offsets.BDay(lookback_days + 5)
    sym_raw = str(benchmark_symbol).strip()
    if sym_raw.lower() in (MARKET_EW_PROXY.lower(),):
        try:
            df = db.read_daily_frame(symbols=None, start=start, end=asof_ts)
        except Exception:
            return pd.Series(dtype=float)
        min_obs = max(5, min(int(lookback_days), 30))
        return _market_ew_daily_returns_from_frame(
            df, date_col=date_col, close_col=close_col, min_symbol_obs=min_obs
        )

    sym = sym_raw.zfill(6)

    try:
        df = db.read_daily_frame(
            symbols=[sym],
            start=start,
            end=asof_ts,
        )
    except Exception:
        return pd.Series(dtype=float)

    if df.empty or close_col not in df.columns:
        return pd.Series(dtype=float)

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col]).dt.normalize()
    df = df.sort_values(date_col)
    closes = pd.to_numeric(df[close_col], errors="coerce")
    rets = closes.pct_change()
    rets.index = df[date_col].values
    return rets.dropna()
