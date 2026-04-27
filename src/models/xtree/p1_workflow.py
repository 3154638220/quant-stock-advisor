"""P1 树模型研究编排：特征分组、面板增强与 daily-proxy-first 汇总。"""

from __future__ import annotations

import json
import re
from collections import OrderedDict
from typing import Any, Iterable

import numpy as np
import pandas as pd

from scripts.light_strategy_proxy import infer_periods_per_year, summarize_signal_diagnostic
from src.backtest.engine import BacktestConfig, build_open_to_open_returns, run_backtest
from src.features.tree_dataset import default_tree_factor_names

WEEKLY_KDJ_TREE_FEATURES: tuple[str, ...] = (
    "weekly_kdj_k",
    "weekly_kdj_d",
    "weekly_kdj_j",
    "weekly_kdj_oversold",
    "weekly_kdj_oversold_depth",
    "weekly_kdj_rebound",
)

FUND_FLOW_TREE_FEATURES: tuple[str, ...] = (
    "main_inflow_z_5d",
    "main_inflow_z_10d",
    "main_inflow_z_20d",
    "super_inflow_z_5d",
    "super_inflow_z_10d",
    "super_inflow_z_20d",
    "flow_divergence_5d",
    "flow_divergence_10d",
    "flow_divergence_20d",
    "main_inflow_streak",
)

SHAREHOLDER_TREE_FEATURES: tuple[str, ...] = (
    "holder_count_change_pct",
    "holder_change_rate_z",
    "holder_count_log_z",
    "holder_concentration_proxy",
)

WEEKLY_KDJ_INTERACTION_TREE_FEATURES: tuple[str, ...] = (
    "wk_j_contrarian_x_vol_to_turnover",
    "wk_j_contrarian_low_turnover_gate",
    "wk_j_contrarian_weak_momentum_gate",
    "wk_oversold_depth_low_vol_gate",
)

DEFAULT_P1_BENCHMARK_KEY_YEARS: tuple[int, ...] = (2021, 2025, 2026)


def baseline_tree_feature_names() -> tuple[str, ...]:
    """P1 的 G0 基线技术面：默认树模型因子剔除 weekly_kdj 家族。"""
    banned = set(WEEKLY_KDJ_TREE_FEATURES)
    return tuple(name for name in default_tree_factor_names() if name not in banned)


def p1_tree_feature_groups(
    *,
    include_interaction_groups: bool = False,
) -> "OrderedDict[str, tuple[str, ...]]":
    """P1 研究固定分组，保证 G0~G4 口径稳定可复现。"""
    g0 = baseline_tree_feature_names()
    groups: "OrderedDict[str, tuple[str, ...]]" = OrderedDict(
        [
            ("G0", g0),
            ("G1", g0 + WEEKLY_KDJ_TREE_FEATURES),
            ("G2", g0 + FUND_FLOW_TREE_FEATURES),
            ("G3", g0 + SHAREHOLDER_TREE_FEATURES),
            ("G4", g0 + WEEKLY_KDJ_TREE_FEATURES + FUND_FLOW_TREE_FEATURES),
        ]
    )
    if include_interaction_groups:
        groups["G5"] = g0 + WEEKLY_KDJ_TREE_FEATURES + WEEKLY_KDJ_INTERACTION_TREE_FEATURES
        groups["G6"] = g0 + WEEKLY_KDJ_INTERACTION_TREE_FEATURES
    return groups


def union_p1_tree_feature_names() -> tuple[str, ...]:
    """P1 runner 所需的全量特征并集。"""
    out: list[str] = []
    seen: set[str] = set()
    for names in p1_tree_feature_groups().values():
        for name in names:
            if name not in seen:
                seen.add(name)
                out.append(name)
    return tuple(out)


def panel_generation_feature_names() -> tuple[str, ...]:
    """
    供 ``long_factor_panel_from_daily`` 直接生成的原生树因子。

    资金流 / 股东人数列需要在面板生成后再 attach，因此这里仅保留 baseline + weekly_kdj。
    """
    g0 = baseline_tree_feature_names()
    out: list[str] = []
    seen: set[str] = set()
    for name in g0 + WEEKLY_KDJ_TREE_FEATURES:
        if name not in seen:
            seen.add(name)
            out.append(name)
    return tuple(out)


def _slugify_token(value: Any) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "na"


def build_p1_tree_research_config_id(
    *,
    rebalance_rule: str,
    top_k: int,
    label_horizons: Iterable[int],
    label_weights: Iterable[float] | None = None,
    proxy_horizon: int,
    val_frac: float,
    label_mode: str = "",
    label_transform: str = "raw",
    xgboost_objective: str = "",
) -> str:
    """把核心研究参数压成稳定 config id，便于结果/工件追踪。"""
    horizon_values = [int(x) for x in label_horizons]
    horizons = "-".join(str(x) for x in horizon_values)
    rebalance = _slugify_token(rebalance_rule or "d")
    val_pct = int(round(float(val_frac) * 100))
    parts = [
        f"rb_{rebalance}",
        f"top{int(top_k)}",
        f"lh_{horizons}",
        f"px_{int(proxy_horizon)}",
        f"val{val_pct}",
    ]
    if label_weights is not None:
        weights = [float(x) for x in label_weights]
        default_equal = (
            len(weights) == len(horizon_values)
            and len(weights) > 0
            and all(np.isclose(w, 1.0 / len(weights), rtol=0.0, atol=1e-12) for w in weights)
        )
        if weights and not default_equal:
            parts.append("lw_" + "-".join(f"{int(round(w * 100)):02d}" for w in weights))
    if label_mode:
        parts.append(f"lbl_{_slugify_token(label_mode)}")
    transform = _slugify_token(label_transform or "raw")
    if transform != "raw":
        parts.append(f"lt_{transform}")
    if xgboost_objective:
        parts.append(f"obj_{_slugify_token(xgboost_objective)}")
    return "_".join(parts)


def build_p1_training_label(
    panel: pd.DataFrame,
    *,
    label_columns: Iterable[str],
    label_weights: Iterable[float],
    label_mode: str = "rank_fusion",
    date_col: str = "trade_date",
    out_col: str = "forward_ret_fused",
) -> tuple[pd.DataFrame, str, dict[str, Any]]:
    """
    构造 P1 树模型训练标签。

    ``rank_fusion`` 复用既有截面 rank 融合；``top_bucket_rank_fusion`` 只保留
    截面顶部/底部 20% 的 rank 信号，用于最小化测试 Top-K 边界噪声；
    ``raw_fusion`` 直接融合原始收益；
    ``market_relative`` / ``benchmark_relative`` 先按日扣掉截面等权收益，再融合。
    ``up_capture_market_relative`` 继续使用同日截面等权相对收益，但在市场前向收益为正的
    截面上放大标签幅度，用于最小化测试上涨参与不足机制。
    当前 full backtest 以 ``market_ew_proxy`` 为 benchmark，因此 benchmark-relative
    在训练面板内采用同日截面等权前瞻收益作为可复现代理。
    """
    label_cols = [str(c) for c in label_columns]
    weights = [float(w) for w in label_weights]
    if len(label_cols) != len(weights):
        raise ValueError("label_columns 与 label_weights 长度不一致")
    if not label_cols:
        raise ValueError("label_columns 不能为空")
    missing = [c for c in label_cols if c not in panel.columns]
    if missing:
        raise ValueError(f"缺少标签列: {missing}")
    if date_col not in panel.columns:
        raise ValueError(f"缺少日期列: {date_col}")

    w = np.asarray(weights, dtype=np.float64)
    if not np.isfinite(w).all() or np.sum(np.abs(w)) <= 1e-12:
        raise ValueError("label_weights 非法")
    w = w / np.sum(np.abs(w))

    mode = _slugify_token(label_mode or "rank_fusion")
    if mode not in {
        "rank_fusion",
        "top_bucket_rank_fusion",
        "raw_fusion",
        "market_relative",
        "benchmark_relative",
        "up_capture_market_relative",
    }:
        raise ValueError(
            "label_mode 须为 rank_fusion/top_bucket_rank_fusion/raw_fusion/market_relative/"
            "benchmark_relative/up_capture_market_relative"
        )

    out = panel.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce").dt.normalize()
    score = np.zeros(len(out), dtype=np.float64)
    valid = out[date_col].notna().to_numpy(dtype=bool)
    component_cols: list[str] = []
    for col, wi in zip(label_cols, w):
        vals = pd.to_numeric(out[col], errors="coerce")
        if mode in {"rank_fusion", "top_bucket_rank_fusion"}:
            rank_pct = vals.groupby(out[date_col], sort=False).rank(method="average", pct=True)
            if mode == "top_bucket_rank_fusion":
                upper = ((rank_pct - 0.8) / 0.2).clip(lower=0.0, upper=1.0)
                lower = ((0.4 - rank_pct) / 0.2).clip(lower=0.0, upper=1.0)
                comp = upper - lower
            else:
                comp = rank_pct - 0.5
        elif mode in {"market_relative", "benchmark_relative", "up_capture_market_relative"}:
            market_ret = vals.groupby(out[date_col], sort=False).transform("mean")
            comp = vals - market_ret
            if mode == "up_capture_market_relative":
                comp = comp * np.where(market_ret.to_numpy(dtype=np.float64) > 0.0, 2.0, 1.0)
        else:
            comp = vals
        comp_np = comp.to_numpy(dtype=np.float64)
        score += wi * comp_np
        valid &= np.isfinite(comp_np)
        component_cols.append(col)

    out[out_col] = np.where(valid, score, np.nan)
    meta = {
        "label_mode": mode,
        "label_scope": (
            "cross_section_top_bottom_bucket"
            if mode == "top_bucket_rank_fusion"
            else "cross_section_relative"
            if mode == "rank_fusion"
            else mode
        ),
        "label_component_columns": ",".join(component_cols),
        "label_weights_normalized": ",".join(f"{x:.8g}" for x in w),
        "label_top_bucket_quantile": 0.2 if mode == "top_bucket_rank_fusion" else "",
        "label_market_proxy": (
            "same_date_cross_section_equal_weight"
            if mode in {"market_relative", "benchmark_relative", "up_capture_market_relative"}
            else ""
        ),
        "label_up_capture_multiplier": 2.0 if mode == "up_capture_market_relative" else 1.0,
    }
    return out[np.isfinite(out[out_col])].copy(), out_col, meta


def _daily_asset_return_matrix(
    daily_df: pd.DataFrame,
    *,
    execution_mode: str = "tplus1_open",
    symbol_col: str = "symbol",
    date_col: str = "trade_date",
) -> pd.DataFrame:
    """构造与正式执行模式一致的单资产日收益矩阵。"""
    if daily_df.empty:
        return pd.DataFrame()
    daily = daily_df.copy()
    daily[symbol_col] = daily[symbol_col].astype(str).str.zfill(6)
    daily[date_col] = pd.to_datetime(daily[date_col], errors="coerce").dt.normalize()
    exe = str(execution_mode).lower().strip()
    if exe == "tplus1_open":
        out = build_open_to_open_returns(
            daily,
            date_col=date_col,
            sym_col=symbol_col,
            zero_if_limit_up_open=False,
        )
    elif exe == "close_to_close":
        d = daily[
            [symbol_col, date_col, "close"]
        ].copy()
        d["close"] = pd.to_numeric(d["close"], errors="coerce")
        d = d.dropna(subset=[date_col, "close"]).sort_values([symbol_col, date_col])
        d["_ret"] = d.groupby(symbol_col, sort=False)["close"].pct_change()
        out = d.pivot(index=date_col, columns=symbol_col, values="_ret").sort_index()
    else:
        raise ValueError(f"monthly_investable 标签当前仅支持 tplus1_open/close_to_close，收到: {execution_mode}")
    out.index = pd.to_datetime(out.index, errors="coerce").normalize()
    return out.sort_index().astype(np.float64)


def build_investable_period_return_panel(
    panel: pd.DataFrame,
    daily_df: pd.DataFrame,
    *,
    rebalance_rule: str,
    execution_mode: str = "tplus1_open",
    out_col: str = "forward_ret_investable",
    date_col: str = "trade_date",
    symbol_col: str = "symbol",
) -> pd.DataFrame:
    """
    为每个调仓日构造真实可投资持有期收益。

    信号日为每个 rebalance period 的最后一个交易日；收益从下一交易日开盘买入后开始，
    一直持有到下一次调仓信号生效前的最后一个 open-to-open 区间。这个窗口与
    ``run_backtest(..., execution_mode='tplus1_open')`` 对月频权重的使用方式一致。
    """
    required = {date_col, symbol_col}
    missing = sorted(required - set(panel.columns))
    if missing:
        raise ValueError(f"panel 缺少列: {missing}")
    if out_col in panel.columns:
        panel = panel.drop(columns=[out_col])

    df = panel.copy()
    df[symbol_col] = df[symbol_col].astype(str).str.zfill(6)
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce").dt.normalize()
    df = df.dropna(subset=[date_col]).copy()
    if df.empty:
        return pd.DataFrame(columns=[*panel.columns, out_col])

    chosen = select_rebalance_dates(df[date_col].unique(), rebalance_rule=rebalance_rule)
    if len(chosen) < 2:
        return pd.DataFrame(columns=[*df.columns, out_col])
    asset_returns = _daily_asset_return_matrix(
        daily_df,
        execution_mode=execution_mode,
        symbol_col=symbol_col,
        date_col=date_col,
    )
    if asset_returns.empty:
        return pd.DataFrame(columns=[*df.columns, out_col])

    rows: list[pd.DataFrame] = []
    signal_dates = pd.to_datetime(chosen["trade_date"], errors="coerce").dt.normalize().tolist()
    for signal_date, next_signal_date in zip(signal_dates[:-1], signal_dates[1:]):
        if pd.isna(signal_date) or pd.isna(next_signal_date) or next_signal_date <= signal_date:
            continue
        window = asset_returns[(asset_returns.index > signal_date) & (asset_returns.index <= next_signal_date)]
        if window.empty:
            continue
        period_ret = (1.0 + window.fillna(0.0)).prod(axis=0) - 1.0
        period_df = period_ret.rename(out_col).reset_index().rename(columns={"index": symbol_col})
        period_df[symbol_col] = period_df[symbol_col].astype(str).str.zfill(6)
        day = df[df[date_col] == signal_date].merge(period_df, on=symbol_col, how="inner")
        day = day[np.isfinite(pd.to_numeric(day[out_col], errors="coerce"))].copy()
        if not day.empty:
            rows.append(day)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=[*df.columns, out_col])


def build_p1_monthly_investable_label(
    panel: pd.DataFrame,
    daily_df: pd.DataFrame,
    *,
    rebalance_rule: str,
    execution_mode: str = "tplus1_open",
    label_mode: str = "monthly_investable",
    date_col: str = "trade_date",
    out_col: str = "forward_ret_investable",
) -> tuple[pd.DataFrame, str, dict[str, Any]]:
    """构造与月频正式执行一致的 P1 标签。"""
    mode = _slugify_token(label_mode or "monthly_investable")
    if mode not in {
        "monthly_investable",
        "monthly_investable_market_relative",
        "monthly_investable_up_capture_market_relative",
    }:
        raise ValueError(
            "monthly label_mode 须为 monthly_investable/monthly_investable_market_relative/"
            "monthly_investable_up_capture_market_relative"
        )
    out = build_investable_period_return_panel(
        panel,
        daily_df,
        rebalance_rule=rebalance_rule,
        execution_mode=execution_mode,
        out_col=out_col,
        date_col=date_col,
    )
    if out.empty:
        meta = {
            "label_mode": mode,
            "label_scope": mode,
            "label_component_columns": out_col,
            "label_weights_normalized": "1",
            "label_market_proxy": "",
            "label_rebalance_rule": str(rebalance_rule),
            "label_execution_mode": str(execution_mode),
        }
        return out, out_col, meta

    out[date_col] = pd.to_datetime(out[date_col], errors="coerce").dt.normalize()
    vals = pd.to_numeric(out[out_col], errors="coerce")
    if mode in {"monthly_investable_market_relative", "monthly_investable_up_capture_market_relative"}:
        market_ret = vals.groupby(out[date_col], sort=False).transform("mean")
        rel = vals - market_ret
        if mode == "monthly_investable_up_capture_market_relative":
            rel = rel * np.where(market_ret.to_numpy(dtype=np.float64) > 0.0, 2.0, 1.0)
        out[out_col] = rel
        market_proxy = "same_rebalance_date_cross_section_equal_weight"
    else:
        out[out_col] = vals
        market_proxy = ""
    out = out[np.isfinite(pd.to_numeric(out[out_col], errors="coerce"))].copy()
    meta = {
        "label_mode": mode,
        "label_scope": mode,
        "label_component_columns": out_col,
        "label_weights_normalized": "1",
        "label_market_proxy": market_proxy,
        "label_rebalance_rule": str(rebalance_rule),
        "label_execution_mode": str(execution_mode),
        "label_alignment": "rebalance_period_tplus1_open_to_next_rebalance",
        "label_up_capture_multiplier": 2.0 if mode == "monthly_investable_up_capture_market_relative" else 1.0,
    }
    return out, out_col, meta


def build_p1_tree_output_stem(
    *,
    out_tag: str,
    research_config_id: str,
    generated_at: Any,
) -> str:
    """生成 P1 树模型研究输出 stem，统一命名口径。"""
    ts = pd.Timestamp(generated_at).strftime("%Y%m%d_%H%M%S")
    return f"{_slugify_token(out_tag)}_{research_config_id}_{ts}"


def attach_p1_experimental_features(
    panel: pd.DataFrame,
    *,
    db_path: str,
    fund_flow_table: str = "a_share_fund_flow",
    shareholder_table: str = "a_share_shareholder",
    shareholder_lag_days: int = 30,
) -> pd.DataFrame:
    """在树模型面板上接入 P1 新数据链路。"""
    from src.features.fund_flow_factors import attach_fund_flow
    from src.features.shareholder_factors import attach_shareholder_factors

    out = attach_fund_flow(panel, db_path, table_name=fund_flow_table)
    out = attach_shareholder_factors(
        out,
        db_path,
        table_name=shareholder_table,
        availability_lag_days=shareholder_lag_days,
    )
    return out


def _cross_section_z(df: pd.DataFrame, col: str, *, date_col: str = "trade_date") -> pd.Series:
    vals = pd.to_numeric(df[col], errors="coerce")
    mean = vals.groupby(df[date_col], sort=False).transform("mean")
    std = vals.groupby(df[date_col], sort=False).transform("std")
    return (vals - mean) / std.replace(0.0, np.nan)


def attach_weekly_kdj_interaction_features(
    panel: pd.DataFrame,
    *,
    low_turnover_quantile: float = 0.4,
    weak_momentum_quantile: float = 0.6,
    low_vol_quantile: float = 0.7,
) -> pd.DataFrame:
    """
    构造 P1 的 weekly_kdj gated / interaction 特征。

    原始证据显示 ``weekly_kdj_j`` 低位更偏正向，因此这里统一使用
    ``-weekly_kdj_j`` 作为 contrarian 强度，再用低换手、非强动量、非高波动
    三类条件做门控。
    """
    required = {
        "trade_date",
        "weekly_kdj_j",
        "weekly_kdj_oversold_depth",
        "vol_to_turnover",
        "turnover_roll_mean",
        "momentum",
        "realized_vol",
    }
    missing = sorted(required - set(panel.columns))
    if missing:
        raise ValueError(f"缺少 weekly_kdj interaction 依赖列: {missing}")

    out = panel.copy()
    out["trade_date"] = pd.to_datetime(out["trade_date"], errors="coerce").dt.normalize()
    contrarian = -pd.to_numeric(out["weekly_kdj_j"], errors="coerce")
    out["wk_j_contrarian_x_vol_to_turnover"] = contrarian * _cross_section_z(out, "vol_to_turnover")

    turnover = pd.to_numeric(out["turnover_roll_mean"], errors="coerce")
    momentum = pd.to_numeric(out["momentum"], errors="coerce")
    realized_vol = pd.to_numeric(out["realized_vol"], errors="coerce")
    low_turnover_cut = turnover.groupby(out["trade_date"], sort=False).transform(
        lambda s: s.quantile(float(low_turnover_quantile))
    )
    weak_momentum_cut = momentum.groupby(out["trade_date"], sort=False).transform(
        lambda s: s.quantile(float(weak_momentum_quantile))
    )
    low_vol_cut = realized_vol.groupby(out["trade_date"], sort=False).transform(
        lambda s: s.quantile(float(low_vol_quantile))
    )

    out["wk_j_contrarian_low_turnover_gate"] = contrarian.where(turnover <= low_turnover_cut, 0.0)
    out["wk_j_contrarian_weak_momentum_gate"] = contrarian.where(momentum <= weak_momentum_cut, 0.0)
    out["wk_oversold_depth_low_vol_gate"] = pd.to_numeric(
        out["weekly_kdj_oversold_depth"], errors="coerce"
    ).where(realized_vol <= low_vol_cut, 0.0)
    return out


def resolve_available_feature_names(
    panel: pd.DataFrame,
    feature_names: Iterable[str],
) -> tuple[list[str], list[str]]:
    """过滤掉面板中不存在或全为空的列。"""
    available: list[str] = []
    missing: list[str] = []
    for name in feature_names:
        if name not in panel.columns:
            missing.append(name)
            continue
        ser = pd.to_numeric(panel[name], errors="coerce")
        if ser.notna().any():
            available.append(name)
        else:
            missing.append(name)
    return available, missing


def select_rebalance_dates(
    dates: Iterable[Any],
    *,
    rebalance_rule: str,
) -> pd.DataFrame:
    """按调仓频率选出每个 period 的最终调仓日。"""
    idx = pd.DatetimeIndex(pd.to_datetime(list(dates), errors="coerce")).dropna().sort_values().unique()
    if len(idx) == 0:
        return pd.DataFrame(columns=["trade_date", "period"])

    rule = str(rebalance_rule).strip().upper() or "D"
    multiple = ""
    unit = rule
    while unit and unit[0].isdigit():
        multiple += unit[0]
        unit = unit[1:]
    step = max(int(multiple), 1) if multiple else 1

    if unit in {"M", "ME"}:
        base_period = pd.Series(idx.to_period("M").astype(str), index=idx)
    elif unit.startswith("W"):
        weekly_freq = unit if "-" in unit else "W-FRI"
        base_period = pd.Series(idx.to_period(weekly_freq).astype(str), index=idx)
    elif unit in {"Q", "QE"}:
        base_period = pd.Series(idx.to_period("Q").astype(str), index=idx)
    else:
        base_period = pd.Series(idx.strftime("%Y-%m-%d"), index=idx)

    ordered_periods = pd.Index(pd.unique(base_period))
    period_to_gid = {period: i // step for i, period in enumerate(ordered_periods)}
    grouped = (
        pd.DataFrame(
            {
                "trade_date": idx,
                "base_period": base_period.to_numpy(),
                "group_id": base_period.map(period_to_gid).to_numpy(),
            }
        )
        .groupby("group_id", sort=True)
        .agg(trade_date=("trade_date", "last"), period=("base_period", "last"))
        .reset_index(drop=True)
    )
    grouped["trade_date"] = pd.to_datetime(grouped["trade_date"]).dt.normalize()
    return grouped


def build_tree_light_proxy_detail(
    scored_panel: pd.DataFrame,
    *,
    score_col: str,
    proxy_return_col: str,
    rebalance_rule: str,
    top_k: int,
    scenario: str,
) -> pd.DataFrame:
    """
    用验证集预测分数构造 benchmark-first 轻量代理。

    口径：
    - 每个 rebalance period 只取最后一个交易日做一次截面选股
    - `strategy_return` = Top-K 在该日的平均前瞻收益
    - `benchmark_return` = 同日全截面的平均前瞻收益
    """
    if top_k < 1:
        raise ValueError("top_k 须 >= 1")
    if score_col not in scored_panel.columns:
        raise ValueError(f"缺少 score_col={score_col}")
    if proxy_return_col not in scored_panel.columns:
        raise ValueError(f"缺少 proxy_return_col={proxy_return_col}")

    df = scored_panel.copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["trade_date"]).copy()
    if df.empty:
        return pd.DataFrame(
            columns=[
                "period",
                "trade_date",
                "strategy_return",
                "benchmark_return",
                "scenario",
                "excess_return",
                "benchmark_up",
                "strategy_up",
                "beat_benchmark",
            ]
        )

    chosen = select_rebalance_dates(df["trade_date"].unique(), rebalance_rule=rebalance_rule)
    rows: list[dict[str, Any]] = []
    for row in chosen.itertuples(index=False):
        day = df[df["trade_date"] == row.trade_date].copy()
        if day.empty:
            continue
        day[score_col] = pd.to_numeric(day[score_col], errors="coerce")
        day[proxy_return_col] = pd.to_numeric(day[proxy_return_col], errors="coerce")
        day = day[np.isfinite(day[score_col]) & np.isfinite(day[proxy_return_col])].copy()
        if day.empty:
            continue

        top = day.nlargest(top_k, columns=score_col, keep="all").head(top_k)
        strategy_ret = float(pd.to_numeric(top[proxy_return_col], errors="coerce").mean())
        benchmark_ret = float(pd.to_numeric(day[proxy_return_col], errors="coerce").mean())
        excess_ret = strategy_ret - benchmark_ret
        rows.append(
            {
                "period": row.period,
                "trade_date": pd.Timestamp(row.trade_date),
                "strategy_return": strategy_ret,
                "benchmark_return": benchmark_ret,
                "scenario": scenario,
                "excess_return": excess_ret,
                "benchmark_up": benchmark_ret > 0.0,
                "strategy_up": strategy_ret > 0.0,
                "beat_benchmark": excess_ret > 0.0,
            }
        )

    return pd.DataFrame(rows)


def _equal_weight_turnover(prev: set[str], cur: set[str], *, top_k: int) -> float:
    if top_k <= 0:
        return float("nan")
    if not prev and not cur:
        return 0.0
    if not prev:
        return 1.0
    return float(1.0 - (len(prev & cur) / float(top_k)))


def build_tree_turnover_aware_proxy_detail(
    scored_panel: pd.DataFrame,
    *,
    score_col: str,
    proxy_return_col: str,
    rebalance_rule: str,
    top_k: int,
    max_turnover: float,
    scenario: str,
) -> pd.DataFrame:
    """
    构造更接近正式回测的轻量代理。

    与 ``build_tree_light_proxy_detail`` 一样只使用验证集前瞻收益；额外模拟
    ``equal_weight + Top-K + max_turnover`` 的月频持仓延续，避免每期无约束重选
    导致 proxy 高估可执行收益。
    """
    if top_k < 1:
        raise ValueError("top_k 须 >= 1")
    if score_col not in scored_panel.columns:
        raise ValueError(f"缺少 score_col={score_col}")
    if proxy_return_col not in scored_panel.columns:
        raise ValueError(f"缺少 proxy_return_col={proxy_return_col}")
    if "symbol" not in scored_panel.columns:
        raise ValueError("缺少 symbol 列")

    df = scored_panel.copy()
    df["symbol"] = df["symbol"].astype(str).str.zfill(6)
    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["trade_date"]).copy()
    out_cols = [
        "period",
        "trade_date",
        "strategy_return",
        "benchmark_return",
        "scenario",
        "excess_return",
        "benchmark_up",
        "strategy_up",
        "beat_benchmark",
        "selected_count",
        "turnover_half_l1",
        "retained_from_prev_count",
        "target_topk_overlap_count",
    ]
    if df.empty:
        return pd.DataFrame(columns=out_cols)

    max_turnover = float(max(0.0, min(1.0, max_turnover)))
    chosen = select_rebalance_dates(df["trade_date"].unique(), rebalance_rule=rebalance_rule)
    prev_holdings: set[str] = set()
    rows: list[dict[str, Any]] = []
    for row in chosen.itertuples(index=False):
        day = df[df["trade_date"] == row.trade_date].copy()
        if day.empty:
            continue
        day[score_col] = pd.to_numeric(day[score_col], errors="coerce")
        day[proxy_return_col] = pd.to_numeric(day[proxy_return_col], errors="coerce")
        day = day[np.isfinite(day[score_col]) & np.isfinite(day[proxy_return_col])].copy()
        if day.empty:
            continue

        ranked = day.sort_values(score_col, ascending=False).drop_duplicates(subset=["symbol"], keep="first")
        target = ranked.head(top_k).copy()
        selected = target.copy()
        target_symbols = set(target["symbol"].astype(str))
        target_overlap_count = int(len(target_symbols & prev_holdings))

        if prev_holdings and max_turnover < 1.0 and len(selected) >= top_k:
            required_overlap = int(np.ceil(top_k * (1.0 - max_turnover)))
            if target_overlap_count < required_overlap:
                need = required_overlap - target_overlap_count
                selected_symbols = set(selected["symbol"].astype(str))
                cand_prev = ranked[
                    ranked["symbol"].astype(str).isin(prev_holdings)
                    & ~ranked["symbol"].astype(str).isin(selected_symbols)
                ].head(need)
                if not cand_prev.empty:
                    drop_cnt = min(len(cand_prev), len(selected))
                    non_overlap = selected[~selected["symbol"].astype(str).isin(prev_holdings)]
                    if len(non_overlap) >= drop_cnt:
                        to_drop = non_overlap.nsmallest(drop_cnt, score_col).index
                    else:
                        to_drop = selected.nsmallest(drop_cnt, score_col).index
                    selected = selected.drop(index=to_drop)
                    selected = pd.concat([selected, cand_prev], ignore_index=True)
                    selected = (
                        selected.drop_duplicates(subset=["symbol"], keep="first")
                        .sort_values(score_col, ascending=False)
                        .head(top_k)
                    )

        selected_symbols = set(selected["symbol"].astype(str))
        strategy_ret = float(pd.to_numeric(selected[proxy_return_col], errors="coerce").mean())
        benchmark_ret = float(pd.to_numeric(day[proxy_return_col], errors="coerce").mean())
        excess_ret = strategy_ret - benchmark_ret
        rows.append(
            {
                "period": row.period,
                "trade_date": pd.Timestamp(row.trade_date),
                "strategy_return": strategy_ret,
                "benchmark_return": benchmark_ret,
                "scenario": scenario,
                "excess_return": excess_ret,
                "benchmark_up": benchmark_ret > 0.0,
                "strategy_up": strategy_ret > 0.0,
                "beat_benchmark": excess_ret > 0.0,
                "selected_count": int(len(selected_symbols)),
                "turnover_half_l1": _equal_weight_turnover(prev_holdings, selected_symbols, top_k=top_k),
                "retained_from_prev_count": int(len(selected_symbols & prev_holdings)),
                "target_topk_overlap_count": target_overlap_count,
            }
        )
        prev_holdings = selected_symbols

    return pd.DataFrame(rows, columns=out_cols)


def _select_turnover_capped_equal_weight_symbols(
    ranked: pd.DataFrame,
    *,
    symbol_col: str,
    score_col: str,
    top_k: int,
    max_turnover: float,
    prev_holdings: set[str],
) -> tuple[list[str], int]:
    target = ranked.head(top_k).copy()
    selected = target.copy()
    target_symbols = set(target[symbol_col].astype(str))
    target_overlap_count = int(len(target_symbols & prev_holdings))

    if prev_holdings and max_turnover < 1.0 and len(selected) >= top_k:
        required_overlap = int(np.ceil(top_k * (1.0 - max_turnover)))
        if target_overlap_count < required_overlap:
            need = required_overlap - target_overlap_count
            selected_symbols = set(selected[symbol_col].astype(str))
            cand_prev = ranked[
                ranked[symbol_col].astype(str).isin(prev_holdings)
                & ~ranked[symbol_col].astype(str).isin(selected_symbols)
            ].head(need)
            if not cand_prev.empty:
                drop_cnt = min(len(cand_prev), len(selected))
                non_overlap = selected[~selected[symbol_col].astype(str).isin(prev_holdings)]
                if len(non_overlap) >= drop_cnt:
                    to_drop = non_overlap.nsmallest(drop_cnt, score_col).index
                else:
                    to_drop = selected.nsmallest(drop_cnt, score_col).index
                selected = selected.drop(index=to_drop)
                selected = pd.concat([selected, cand_prev], ignore_index=True)
                selected = (
                    selected.drop_duplicates(subset=[symbol_col], keep="first")
                    .sort_values(score_col, ascending=False)
                    .head(top_k)
                )

    return [str(s).zfill(6) for s in selected[symbol_col].astype(str).tolist()], target_overlap_count


def build_tree_score_weight_matrix(
    scored_panel: pd.DataFrame,
    *,
    score_col: str,
    rebalance_rule: str,
    top_k: int,
    max_turnover: float,
    symbol_col: str = "symbol",
    date_col: str = "trade_date",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """由树模型分数构造 equal-weight Top-K 调仓权重，并显式应用换手上限。"""
    if top_k < 1:
        raise ValueError("top_k 须 >= 1")
    required = {symbol_col, date_col, score_col}
    missing = sorted(required - set(scored_panel.columns))
    if missing:
        raise ValueError(f"scored_panel 缺少列: {missing}")

    df = scored_panel[[symbol_col, date_col, score_col]].copy()
    df[symbol_col] = df[symbol_col].astype(str).str.zfill(6)
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce").dt.normalize()
    df[score_col] = pd.to_numeric(df[score_col], errors="coerce")
    df = df.dropna(subset=[symbol_col, date_col, score_col]).copy()
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    max_turnover = float(max(0.0, min(1.0, max_turnover)))
    chosen = select_rebalance_dates(df[date_col].unique(), rebalance_rule=rebalance_rule)
    prev_holdings: set[str] = set()
    rows: list[pd.Series] = []
    diag_rows: list[dict[str, Any]] = []
    all_symbols = sorted(df[symbol_col].unique().tolist())

    for row in chosen.itertuples(index=False):
        day = df[df[date_col] == row.trade_date].copy()
        if day.empty:
            continue
        ranked = day.sort_values(score_col, ascending=False).drop_duplicates(subset=[symbol_col], keep="first")
        if ranked.empty:
            continue
        selected_symbols, target_overlap_count = _select_turnover_capped_equal_weight_symbols(
            ranked,
            symbol_col=symbol_col,
            score_col=score_col,
            top_k=int(top_k),
            max_turnover=max_turnover,
            prev_holdings=prev_holdings,
        )
        if not selected_symbols:
            continue
        selected_set = set(selected_symbols)
        weight = 1.0 / float(len(selected_symbols))
        w = pd.Series(0.0, index=all_symbols, name=pd.Timestamp(row.trade_date), dtype=np.float64)
        for sym in selected_symbols:
            w.loc[sym] = weight
        rows.append(w)
        diag_rows.append(
            {
                "period": row.period,
                "trade_date": pd.Timestamp(row.trade_date),
                "selected_count": int(len(selected_symbols)),
                "turnover_half_l1": _equal_weight_turnover(prev_holdings, selected_set, top_k=top_k),
                "retained_from_prev_count": int(len(selected_set & prev_holdings)),
                "target_topk_overlap_count": int(target_overlap_count),
            }
        )
        prev_holdings = selected_set

    weights = pd.DataFrame(rows) if rows else pd.DataFrame()
    if not weights.empty:
        weights.index = pd.to_datetime(weights.index).normalize()
        weights = weights.sort_index()
    return weights, pd.DataFrame(diag_rows)


def _market_ew_close_to_close_benchmark(
    daily_df: pd.DataFrame,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    min_days: int,
) -> pd.Series:
    df = daily_df[
        (pd.to_datetime(daily_df["trade_date"], errors="coerce") >= start)
        & (pd.to_datetime(daily_df["trade_date"], errors="coerce") <= end)
        & (pd.to_numeric(daily_df["close"], errors="coerce") > 0)
    ].copy()
    if df.empty:
        return pd.Series(dtype=np.float64)
    df["symbol"] = df["symbol"].astype(str).str.zfill(6)
    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce").dt.normalize()
    sym_cnt = df.groupby("symbol")["trade_date"].count()
    good = sym_cnt[sym_cnt >= int(max(1, min_days))].index
    if len(good) == 0:
        good = sym_cnt.index
    df = df[df["symbol"].isin(good)].sort_values(["symbol", "trade_date"])
    df["ret"] = pd.to_numeric(df.groupby("symbol")["close"].pct_change(), errors="coerce")
    out = df.dropna(subset=["ret"]).groupby("trade_date")["ret"].mean()
    out.index = pd.to_datetime(out.index).normalize()
    return out.sort_index().astype(np.float64)


def build_tree_daily_backtest_like_proxy_detail(
    scored_panel: pd.DataFrame,
    daily_df: pd.DataFrame,
    *,
    score_col: str,
    rebalance_rule: str,
    top_k: int,
    max_turnover: float,
    scenario: str,
    cost_params: Any = None,
    execution_mode: str = "tplus1_open",
    execution_lag: int = 1,
    limit_up_mode: str = "idle",
    vwap_slippage_bps_per_side: float = 3.0,
    vwap_impact_bps: float = 8.0,
    market_ew_min_days: int | None = None,
    precomputed_asset_returns: pd.DataFrame | None = None,
    precomputed_market_benchmark: pd.Series | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """
    构造 full-backtest-like 日频轻量代理。

    该层直接复用正式回测引擎的日频收益、成本和 ``market_ew`` 对齐口径，
    但只跑当前验证窗分数生成的一条权重序列，不进入完整 OOS/grid/report。
    """
    weights, weight_diag = build_tree_score_weight_matrix(
        scored_panel,
        score_col=score_col,
        rebalance_rule=rebalance_rule,
        top_k=top_k,
        max_turnover=max_turnover,
    )
    out_cols = [
        "period",
        "trade_date",
        "strategy_return",
        "benchmark_return",
        "scenario",
        "excess_return",
        "benchmark_up",
        "strategy_up",
        "beat_benchmark",
        "turnover_half_l1",
    ]
    if weights.empty:
        return pd.DataFrame(columns=out_cols), {"n_rebalances": 0, "avg_turnover_half_l1": float("nan")}

    daily = daily_df.copy()
    daily["symbol"] = daily["symbol"].astype(str).str.zfill(6)
    daily["trade_date"] = pd.to_datetime(daily["trade_date"], errors="coerce").dt.normalize()
    start = pd.Timestamp(weights.index.min()).normalize()
    end = pd.Timestamp(daily["trade_date"].max()).normalize()
    target_cols = sorted(set(weights.columns) | set(daily["symbol"].unique()))

    exe = str(execution_mode).lower().strip()
    if precomputed_asset_returns is not None:
        asset_returns = precomputed_asset_returns.copy().sort_index()
        asset_returns.index = pd.to_datetime(asset_returns.index, errors="coerce").normalize()
        asset_returns = asset_returns.reindex(columns=target_cols).fillna(0.0)
        asset_returns = asset_returns[(asset_returns.index >= start) & (asset_returns.index <= end)]
    elif exe == "tplus1_open":
        asset_returns = build_open_to_open_returns(daily, zero_if_limit_up_open=False).sort_index()
        asset_returns = asset_returns.reindex(columns=target_cols).fillna(0.0)
        asset_returns = asset_returns[(asset_returns.index >= start) & (asset_returns.index <= end)]
    else:
        d = daily[
            (daily["trade_date"] >= start)
            & (daily["trade_date"] <= end)
            & (pd.to_numeric(daily["close"], errors="coerce") > 0)
        ].sort_values(["symbol", "trade_date"]).copy()
        d["ret"] = d.groupby("symbol")["close"].pct_change()
        asset_returns = d.pivot(index="trade_date", columns="symbol", values="ret").sort_index()
        asset_returns = asset_returns.reindex(columns=target_cols).fillna(0.0)

    weights = weights.reindex(columns=target_cols, fill_value=0.0)
    if asset_returns.empty:
        return pd.DataFrame(columns=out_cols), {"n_rebalances": int(len(weights)), "avg_turnover_half_l1": float("nan")}
    asset_returns = asset_returns[asset_returns.index >= start]
    if weights.index.min() > asset_returns.index.min():
        seed = weights.iloc[[0]].copy()
        seed.index = pd.DatetimeIndex([asset_returns.index.min()])
        weights = pd.concat([seed, weights], axis=0)
        weights = weights[~weights.index.duplicated(keep="last")].sort_index()

    bt = BacktestConfig(
        cost_params=cost_params,
        execution_mode=exe,
        execution_lag=int(execution_lag),
        limit_up_mode=str(limit_up_mode),
        vwap_slippage_bps_per_side=float(vwap_slippage_bps_per_side),
        vwap_impact_bps=float(vwap_impact_bps),
    )
    res = run_backtest(asset_returns, weights, config=bt)

    n_trade_days = int(asset_returns.index.nunique())
    min_days = int(market_ew_min_days) if market_ew_min_days is not None else max(20, int(0.35 * max(n_trade_days, 1)))
    if precomputed_market_benchmark is not None:
        bench = precomputed_market_benchmark.copy().sort_index()
        bench.index = pd.to_datetime(bench.index, errors="coerce").normalize()
        bench = bench[(bench.index >= start) & (bench.index <= end)].astype(np.float64)
    else:
        bench = _market_ew_close_to_close_benchmark(daily, start=start, end=end, min_days=min_days)
    common = res.daily_returns.index.intersection(bench.index).sort_values()
    if common.empty:
        detail = pd.DataFrame(columns=out_cols)
    else:
        strat = res.daily_returns.reindex(common).fillna(0.0)
        b = bench.reindex(common).fillna(0.0)
        detail = pd.DataFrame(
            {
                "period": pd.DatetimeIndex(common).strftime("%Y-%m-%d"),
                "trade_date": common,
                "strategy_return": strat.to_numpy(dtype=np.float64),
                "benchmark_return": b.to_numpy(dtype=np.float64),
                "scenario": scenario,
            }
        )
        detail["excess_return"] = detail["strategy_return"] - detail["benchmark_return"]
        detail["benchmark_up"] = detail["benchmark_return"] > 0.0
        detail["strategy_up"] = detail["strategy_return"] > 0.0
        detail["beat_benchmark"] = detail["excess_return"] > 0.0
        detail = detail.merge(
            res.rebalance_turnover.rename("turnover_half_l1").reset_index().rename(columns={"index": "trade_date"}),
            on="trade_date",
            how="left",
        )
    meta = {
        "n_rebalances": int(len(weight_diag)),
        "avg_turnover_half_l1": float(pd.to_numeric(weight_diag.get("turnover_half_l1"), errors="coerce").mean())
        if not weight_diag.empty
        else float("nan"),
        "market_ew_min_days": int(min_days),
        "daily_periods_per_year": 252.0,
        "backtest_like_strategy_annualized_return_engine": float(res.panel.annualized_return),
        "backtest_like_strategy_max_drawdown_engine": float(res.panel.max_drawdown),
    }
    return detail, meta


def summarize_tree_daily_backtest_like_proxy(
    detail_df: pd.DataFrame,
) -> dict[str, float]:
    """汇总 full-backtest-like 日频轻量代理。"""
    summary = summarize_signal_diagnostic(detail_df, periods_per_year=252.0)
    summary["periods_per_year"] = 252.0
    if not detail_df.empty:
        summary["avg_turnover_half_l1"] = float(
            pd.to_numeric(detail_df.get("turnover_half_l1"), errors="coerce").mean()
        )
    else:
        summary["avg_turnover_half_l1"] = float("nan")
    return summary


def build_tree_topk_boundary_diagnostic(
    scored_panel: pd.DataFrame,
    daily_df: pd.DataFrame,
    *,
    score_col: str,
    rebalance_rule: str,
    top_k: int,
    execution_mode: str = "tplus1_open",
    scenario: str,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """
    诊断 Top-K 边界机会损失和持仓切换质量。

    对每个真实调仓日按树模型分数排序，比较 Top-K 与下一档 Top-K 的可投资持有期收益，
    并记录换入、换出、上期 Top-K 掉入 21-40 桶等边界稳定性指标。
    """
    if top_k < 1:
        raise ValueError("top_k 须 >= 1")
    required = {"symbol", "trade_date", score_col}
    missing = sorted(required - set(scored_panel.columns))
    if missing:
        raise ValueError(f"scored_panel 缺少列: {missing}")

    base = scored_panel[["symbol", "trade_date", score_col]].copy()
    base["symbol"] = base["symbol"].astype(str).str.zfill(6)
    base["trade_date"] = pd.to_datetime(base["trade_date"], errors="coerce").dt.normalize()
    base[score_col] = pd.to_numeric(base[score_col], errors="coerce")
    base = base.dropna(subset=["trade_date", score_col]).copy()
    ret_panel = build_investable_period_return_panel(
        base,
        daily_df,
        rebalance_rule=rebalance_rule,
        execution_mode=execution_mode,
        out_col="investable_period_return",
    )
    out_cols = [
        "scenario",
        "period",
        "trade_date",
        "topk_mean_return",
        "next_bucket_mean_return",
        "topk_minus_next_bucket_return",
        "topk_median_return",
        "next_bucket_median_return",
        "topk_positive_share",
        "next_bucket_positive_share",
        "selected_count",
        "next_bucket_count",
        "topk_turnover_half_l1",
        "retained_from_prev_count",
        "switched_in_mean_return",
        "switched_out_mean_return",
        "switched_in_minus_out_return",
        "prev_topk_now_next_bucket_mean_return",
    ]
    if ret_panel.empty:
        return pd.DataFrame(columns=out_cols), {}

    chosen = select_rebalance_dates(ret_panel["trade_date"].unique(), rebalance_rule=rebalance_rule)
    period_map = {
        pd.Timestamp(r.trade_date).normalize(): str(r.period)
        for r in chosen.itertuples(index=False)
    }
    rows: list[dict[str, Any]] = []
    prev_topk: set[str] = set()

    def _mean_for(day_df: pd.DataFrame, symbols: set[str]) -> float:
        if not symbols:
            return float("nan")
        vals = pd.to_numeric(
            day_df.loc[day_df["symbol"].astype(str).isin(symbols), "investable_period_return"],
            errors="coerce",
        )
        return float(vals.mean()) if vals.notna().any() else float("nan")

    for trade_date, day in ret_panel.groupby("trade_date", sort=True):
        day = day.sort_values(score_col, ascending=False).drop_duplicates(subset=["symbol"], keep="first").copy()
        if day.empty:
            continue
        top = day.head(top_k).copy()
        nxt = day.iloc[top_k : top_k * 2].copy()
        top_syms = set(top["symbol"].astype(str))
        next_syms = set(nxt["symbol"].astype(str))
        switched_in = top_syms - prev_topk
        switched_out = prev_topk - top_syms
        prev_now_next = prev_topk & next_syms
        top_ret = pd.to_numeric(top["investable_period_return"], errors="coerce")
        next_ret = pd.to_numeric(nxt["investable_period_return"], errors="coerce")
        switched_in_ret = _mean_for(day, switched_in)
        switched_out_ret = _mean_for(day, switched_out)
        rows.append(
            {
                "scenario": scenario,
                "period": period_map.get(pd.Timestamp(trade_date).normalize(), str(pd.Timestamp(trade_date).date())),
                "trade_date": pd.Timestamp(trade_date),
                "topk_mean_return": float(top_ret.mean()) if top_ret.notna().any() else float("nan"),
                "next_bucket_mean_return": float(next_ret.mean()) if next_ret.notna().any() else float("nan"),
                "topk_minus_next_bucket_return": (
                    float(top_ret.mean() - next_ret.mean()) if top_ret.notna().any() and next_ret.notna().any()
                    else float("nan")
                ),
                "topk_median_return": float(top_ret.median()) if top_ret.notna().any() else float("nan"),
                "next_bucket_median_return": float(next_ret.median()) if next_ret.notna().any() else float("nan"),
                "topk_positive_share": float((top_ret > 0.0).mean()) if len(top_ret) else float("nan"),
                "next_bucket_positive_share": float((next_ret > 0.0).mean()) if len(next_ret) else float("nan"),
                "selected_count": int(len(top_syms)),
                "next_bucket_count": int(len(next_syms)),
                "topk_turnover_half_l1": _equal_weight_turnover(prev_topk, top_syms, top_k=top_k),
                "retained_from_prev_count": int(len(prev_topk & top_syms)),
                "switched_in_mean_return": switched_in_ret,
                "switched_out_mean_return": switched_out_ret,
                "switched_in_minus_out_return": (
                    switched_in_ret - switched_out_ret
                    if np.isfinite(switched_in_ret) and np.isfinite(switched_out_ret)
                    else float("nan")
                ),
                "prev_topk_now_next_bucket_mean_return": _mean_for(day, prev_now_next),
            }
        )
        prev_topk = top_syms

    detail = pd.DataFrame(rows, columns=out_cols)
    if detail.empty:
        return detail, {}
    summary = {
        "topk_boundary_topk_mean_return": float(pd.to_numeric(detail["topk_mean_return"], errors="coerce").mean()),
        "topk_boundary_next_bucket_mean_return": float(
            pd.to_numeric(detail["next_bucket_mean_return"], errors="coerce").mean()
        ),
        "topk_boundary_topk_minus_next_mean_return": float(
            pd.to_numeric(detail["topk_minus_next_bucket_return"], errors="coerce").mean()
        ),
        "topk_boundary_switch_in_minus_out_mean_return": float(
            pd.to_numeric(detail["switched_in_minus_out_return"], errors="coerce").mean()
        ),
        "topk_boundary_prev_topk_now_next_bucket_mean_return": float(
            pd.to_numeric(detail["prev_topk_now_next_bucket_mean_return"], errors="coerce").mean()
        ),
        "topk_boundary_avg_turnover_half_l1": float(
            pd.to_numeric(detail["topk_turnover_half_l1"], errors="coerce").mean()
        ),
        "topk_boundary_periods": int(len(detail)),
    }
    return detail, summary


def _tercile_state(values: pd.Series, *, low_label: str, mid_label: str, high_label: str) -> pd.Series:
    vals = pd.to_numeric(values, errors="coerce")
    out = pd.Series(mid_label, index=values.index, dtype=object)
    finite = vals.dropna()
    if len(finite) < 3:
        out.loc[vals.notna()] = mid_label
        return out
    lo = float(finite.quantile(1.0 / 3.0))
    hi = float(finite.quantile(2.0 / 3.0))
    out.loc[vals <= lo] = low_label
    out.loc[vals >= hi] = high_label
    return out


def build_market_state_table_from_daily(
    daily_df: pd.DataFrame,
    *,
    execution_mode: str = "tplus1_open",
) -> pd.DataFrame:
    """按月构造强上涨、强下跌、波动和广度状态。"""
    asset_returns = _daily_asset_return_matrix(daily_df, execution_mode=execution_mode)
    if asset_returns.empty:
        return pd.DataFrame()
    market = asset_returns.mean(axis=1, skipna=True)
    breadth = (asset_returns > 0.0).mean(axis=1)
    daily = pd.DataFrame(
        {
            "market_return": market,
            "breadth_positive_share": breadth,
        }
    ).dropna(subset=["market_return"])
    if daily.empty:
        return pd.DataFrame()
    monthly = daily.resample("ME").agg(
        benchmark_return=("market_return", lambda s: float((1.0 + s).prod() - 1.0)),
        benchmark_daily_vol=("market_return", lambda s: float(pd.to_numeric(s, errors="coerce").std() * np.sqrt(252.0))),
        breadth_positive_share=("breadth_positive_share", "mean"),
        n_trade_days=("market_return", "count"),
    )
    monthly = monthly.reset_index()
    monthly = monthly.rename(columns={monthly.columns[0]: "month_end"})
    monthly["year"] = monthly["month_end"].dt.year
    monthly["month"] = monthly["month_end"].dt.month
    monthly["return_state"] = _tercile_state(
        monthly["benchmark_return"],
        low_label="strong_down",
        mid_label="neutral_return",
        high_label="strong_up",
    )
    monthly["vol_state"] = _tercile_state(
        monthly["benchmark_daily_vol"],
        low_label="low_vol",
        mid_label="mid_vol",
        high_label="high_vol",
    )
    monthly["breadth_state"] = _tercile_state(
        monthly["breadth_positive_share"],
        low_label="narrow_breadth",
        mid_label="mid_breadth",
        high_label="wide_breadth",
    )
    return monthly


def summarize_tree_daily_proxy_state_slices(
    detail_df: pd.DataFrame,
    daily_df: pd.DataFrame,
    *,
    execution_mode: str = "tplus1_open",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """把 daily backtest-like proxy 汇总到市场状态切片。"""
    if detail_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    required = {"trade_date", "strategy_return", "benchmark_return", "group", "proxy_variant"}
    missing = sorted(required - set(detail_df.columns))
    if missing:
        raise ValueError(f"detail_df 缺少列: {missing}")
    d = detail_df[detail_df["proxy_variant"] == "daily_backtest_like"].copy()
    if d.empty:
        return pd.DataFrame(), pd.DataFrame()
    d["trade_date"] = pd.to_datetime(d["trade_date"], errors="coerce").dt.normalize()
    d["month_end"] = d["trade_date"].dt.to_period("M").dt.to_timestamp("M")
    d["strategy_return"] = pd.to_numeric(d["strategy_return"], errors="coerce")
    d["benchmark_return"] = pd.to_numeric(d["benchmark_return"], errors="coerce")
    monthly_rows: list[dict[str, Any]] = []
    for (group, month_end), sub in d.groupby(["group", "month_end"], sort=True):
        strat = pd.to_numeric(sub["strategy_return"], errors="coerce").dropna()
        bench = pd.to_numeric(sub["benchmark_return"], errors="coerce").dropna()
        if strat.empty or bench.empty:
            continue
        strategy_return = float((1.0 + strat).prod() - 1.0)
        benchmark_return = float((1.0 + bench).prod() - 1.0)
        monthly_rows.append(
            {
                "group": group,
                "month_end": pd.Timestamp(month_end),
                "year": int(pd.Timestamp(month_end).year),
                "month": int(pd.Timestamp(month_end).month),
                "strategy_return": strategy_return,
                "benchmark_return": benchmark_return,
                "excess_return": strategy_return - benchmark_return,
                "beat_benchmark": strategy_return > benchmark_return,
                "n_trade_days": int(len(sub)),
            }
        )
    monthly = pd.DataFrame(monthly_rows)
    if monthly.empty:
        return monthly, pd.DataFrame()
    state = build_market_state_table_from_daily(daily_df, execution_mode=execution_mode)
    if not state.empty:
        monthly = monthly.merge(
            state[
                [
                    "month_end",
                    "return_state",
                    "vol_state",
                    "breadth_state",
                    "benchmark_daily_vol",
                    "breadth_positive_share",
                ]
            ],
            on="month_end",
            how="left",
        )

    rows: list[dict[str, Any]] = []
    for state_axis in ("return_state", "vol_state", "breadth_state"):
        if state_axis not in monthly.columns:
            continue
        for (group, state_value), part in monthly.groupby(["group", state_axis], sort=True):
            vals = pd.to_numeric(part["excess_return"], errors="coerce")
            rows.append(
                {
                    "group": group,
                    "state_axis": state_axis,
                    "state": state_value,
                    "n_months": int(len(part)),
                    "mean_excess_return": float(vals.mean()) if vals.notna().any() else float("nan"),
                    "median_excess_return": float(vals.median()) if vals.notna().any() else float("nan"),
                    "beat_rate": float(pd.to_numeric(part["beat_benchmark"], errors="coerce").mean()),
                    "strategy_mean_return": float(pd.to_numeric(part["strategy_return"], errors="coerce").mean()),
                    "benchmark_mean_return": float(pd.to_numeric(part["benchmark_return"], errors="coerce").mean()),
                }
            )
    return monthly, pd.DataFrame(rows)


def summarize_tree_group_result(
    detail_df: pd.DataFrame,
    *,
    rebalance_rule: str,
) -> dict[str, float]:
    """统一输出树模型 P1 轻量代理摘要。"""
    periods_per_year = infer_periods_per_year(rebalance_rule)
    summary = summarize_signal_diagnostic(detail_df, periods_per_year=periods_per_year)
    summary["periods_per_year"] = float(periods_per_year)
    return summary


def summarize_p1_full_backtest_payload(
    payload: dict[str, Any],
    *,
    benchmark_key_years: Iterable[int] | None = None,
) -> dict[str, float | str]:
    """
    从 ``run_backtest_eval`` 的 JSON 抽取 P1 promotion gate 需要的核心字段。

    优先读取 benchmark-first 指标；如果当前 JSON 尚未提供 rolling/slice 的超额口径，
    对应字段保留为 ``NaN``，避免误把缺失当成通过。
    """
    years = tuple(int(y) for y in (benchmark_key_years or DEFAULT_P1_BENCHMARK_KEY_YEARS))
    full_sample = payload.get("full_sample", {}) or {}
    with_cost = full_sample.get("with_cost", {}) or {}
    excess_vs_market = full_sample.get("excess_vs_market", {}) or {}
    params = payload.get("parameters", {}) or {}
    meta = payload.get("meta", {}) or {}
    tree_model = meta.get("tree_model", {}) or {}
    prepared_cache = meta.get("prepared_factors_cache", {}) or {}
    rolling_agg = ((payload.get("walk_forward_rolling", {}) or {}).get("agg", {}) or {})
    slice_agg = ((payload.get("walk_forward_slices", {}) or {}).get("agg", {}) or {})
    rolling_excess = ((payload.get("walk_forward_rolling", {}) or {}).get("excess_vs_market", {}) or {})
    slice_excess = ((payload.get("walk_forward_slices", {}) or {}).get("excess_vs_market", {}) or {})
    yearly_rows = payload.get("yearly", []) or []

    yearly_df = pd.DataFrame(yearly_rows)
    if not yearly_df.empty:
        yearly_df["year"] = pd.to_numeric(yearly_df["year"], errors="coerce")
        yearly_df["excess"] = pd.to_numeric(yearly_df["excess"], errors="coerce")
        yearly_excess = yearly_df["excess"].to_numpy(dtype=np.float64)
        valid_yearly = yearly_excess[np.isfinite(yearly_excess)]
        yearly_excess_median = float(np.median(valid_yearly)) if valid_yearly.size else float("nan")
        key_df = yearly_df[yearly_df["year"].isin(years)].copy()
        key_vals = pd.to_numeric(key_df["excess"], errors="coerce").to_numpy(dtype=np.float64)
        key_vals = key_vals[np.isfinite(key_vals)]
        key_year_mean = float(np.mean(key_vals)) if key_vals.size else float("nan")
    else:
        yearly_excess_median = float("nan")
        key_year_mean = float("nan")

    return {
        "full_backtest_annualized_excess_vs_market": float(excess_vs_market.get("annualized_return", np.nan)),
        "full_backtest_sharpe_ratio": float(with_cost.get("sharpe_ratio", np.nan)),
        "full_backtest_annualized_return": float(with_cost.get("annualized_return", np.nan)),
        "full_backtest_max_drawdown": float(with_cost.get("max_drawdown", np.nan)),
        "rolling_oos_median_ann_return": float(rolling_agg.get("median_ann_return", np.nan)),
        "slice_oos_median_ann_return": float(slice_agg.get("median_ann_return", np.nan)),
        "rolling_oos_median_ann_excess_vs_market": float(
            rolling_agg.get("median_ann_excess_vs_market", rolling_excess.get("median_ann_excess_return", np.nan))
        ),
        "slice_oos_median_ann_excess_vs_market": float(
            slice_agg.get("median_ann_excess_vs_market", slice_excess.get("median_ann_excess_return", np.nan))
        ),
        "yearly_excess_median_vs_market": yearly_excess_median,
        "key_year_excess_mean_vs_market": key_year_mean,
        "benchmark_key_years": ",".join(str(y) for y in years),
        "full_backtest_research_topic": str(payload.get("research_topic", "")),
        "full_backtest_research_config_id": str(payload.get("research_config_id", "")),
        "full_backtest_output_stem": str(payload.get("output_stem", "")),
        "full_backtest_tree_bundle_dir": str(tree_model.get("bundle_dir") or params.get("tree_bundle_dir") or ""),
        "full_backtest_tree_feature_group": str(
            tree_model.get("feature_group") or params.get("tree_feature_group") or ""
        ),
        "full_backtest_tree_label_spec": json.dumps(
            tree_model.get("label_spec") or {}, ensure_ascii=False, sort_keys=True
        ),
        "full_backtest_tree_score_auto_flipped": bool(tree_model.get("tree_score_auto_flipped"))
        if tree_model.get("tree_score_auto_flipped") is not None
        else "",
        "full_backtest_prepared_factors_cache": str(
            prepared_cache.get("path") or params.get("prepared_factors_cache") or ""
        ),
        "full_backtest_prepared_cache_schema_version": str(
            prepared_cache.get("schema_version")
            or (params.get("prepared_factors_cache_meta") or {}).get("prepared_factors_schema_version")
            or ""
        ),
    }


def _non_degrade(series: pd.Series) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce")
    return vals.notna() & (vals >= 0.0)


def classify_daily_proxy_first_decision(
    daily_proxy_excess: Any,
    *,
    admission_threshold: float = 0.0,
    full_backtest_threshold: float | None = None,
) -> dict[str, Any]:
    """
    P1 daily-proxy-first 的统一判读。

    ``admission_threshold`` 是硬停止线；低于它时不补正式回测。``full_backtest_threshold``
    是给正式回测留出的安全边际；介于两者之间的候选进入灰区，只归档诊断。
    """
    full_threshold = float(admission_threshold if full_backtest_threshold is None else full_backtest_threshold)
    admission = float(admission_threshold)
    full_threshold = max(full_threshold, admission)
    try:
        val = float(daily_proxy_excess)
    except (TypeError, ValueError):
        val = float("nan")

    if not np.isfinite(val):
        return {
            "daily_proxy_first_status": "no_daily_proxy",
            "daily_proxy_first_reason": "daily proxy 缺失或非有限值",
            "pass_p1_daily_proxy_admission_gate": False,
            "pass_p1_daily_proxy_full_backtest_gate": False,
            "daily_proxy_safety_margin_to_admission": float("nan"),
            "daily_proxy_safety_margin_to_full_backtest": float("nan"),
        }
    admission_margin = val - admission
    full_margin = val - full_threshold
    if val < admission:
        return {
            "daily_proxy_first_status": "reject",
            "daily_proxy_first_reason": (
                f"daily proxy {val:.6g} < admission threshold {admission:.6g}; "
                "停止，不补正式 full backtest"
            ),
            "pass_p1_daily_proxy_admission_gate": False,
            "pass_p1_daily_proxy_full_backtest_gate": False,
            "daily_proxy_safety_margin_to_admission": float(admission_margin),
            "daily_proxy_safety_margin_to_full_backtest": float(full_margin),
        }
    if val < full_threshold:
        return {
            "daily_proxy_first_status": "gray_zone",
            "daily_proxy_first_reason": (
                f"daily proxy {val:.6g} >= admission threshold {admission:.6g}, "
                f"但 < full backtest threshold {full_threshold:.6g}; 只归档诊断"
            ),
            "pass_p1_daily_proxy_admission_gate": True,
            "pass_p1_daily_proxy_full_backtest_gate": False,
            "daily_proxy_safety_margin_to_admission": float(admission_margin),
            "daily_proxy_safety_margin_to_full_backtest": float(full_margin),
        }
    return {
        "daily_proxy_first_status": "full_backtest_candidate",
        "daily_proxy_first_reason": (
            f"daily proxy {val:.6g} >= full backtest threshold {full_threshold:.6g}; "
            "允许补正式 full backtest"
        ),
        "pass_p1_daily_proxy_admission_gate": True,
        "pass_p1_daily_proxy_full_backtest_gate": True,
        "daily_proxy_safety_margin_to_admission": float(admission_margin),
        "daily_proxy_safety_margin_to_full_backtest": float(full_margin),
    }


def summarize_tree_score_direction(metrics: dict[str, Any]) -> dict[str, float | bool]:
    """
    对齐 ``predict_xgboost_tree`` 的自动翻转语义，给 P1 summary 显式展示有效方向。

    推理端优先看 ``val_rank_ic``，缺失时再看 ``train_rank_ic``；若该值为负，会把
    ``tree_score`` 乘以 -1。因此 P1 比较表需要同时保留 raw IC 与 effective IC。
    """
    chosen = np.nan
    for key in ("val_rank_ic", "train_rank_ic"):
        try:
            val = float(metrics.get(key, np.nan))
        except (TypeError, ValueError):
            continue
        if np.isfinite(val):
            chosen = val
            break
    auto_flipped = bool(np.isfinite(chosen) and chosen < 0.0)
    return {
        "tree_score_auto_flipped": auto_flipped,
        "effective_val_rank_ic": float(abs(chosen)) if np.isfinite(chosen) else float("nan"),
    }


def summarize_p1_label_diagnostics(
    panel: pd.DataFrame,
    *,
    target_column: str,
    label_columns: Iterable[str],
    label_weights: Iterable[float],
    proxy_return_col: str,
    date_col: str = "trade_date",
) -> dict[str, Any]:
    """诊断 P1 树模型标签口径，尤其是目标列与 light proxy 收益列的方向一致性。"""
    label_cols = [str(c) for c in label_columns]
    weights = [float(w) for w in label_weights]
    out: dict[str, Any] = {
        "target_column": target_column,
        "label_columns": ",".join(label_cols),
        "label_weights": ",".join(f"{w:.8g}" for w in weights),
        "proxy_return_col": proxy_return_col,
        "target_higher_is_better": True,
    }
    required = {date_col, target_column, proxy_return_col}
    missing = sorted(required - set(panel.columns))
    if missing:
        out.update(
            {
                "label_diagnostic_status": "missing_columns",
                "label_diagnostic_missing_columns": ",".join(missing),
            }
        )
        return out

    df = panel[[date_col, target_column, proxy_return_col]].copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce").dt.normalize()
    df[target_column] = pd.to_numeric(df[target_column], errors="coerce")
    df[proxy_return_col] = pd.to_numeric(df[proxy_return_col], errors="coerce")
    valid = df.dropna(subset=[date_col, target_column, proxy_return_col]).copy()
    out["label_row_count"] = int(len(panel))
    out["label_valid_row_count"] = int(len(valid))
    out["label_missing_rate"] = float(1.0 - (len(valid) / len(panel))) if len(panel) else float("nan")
    out["label_date_count"] = int(valid[date_col].nunique()) if not valid.empty else 0
    if valid.empty:
        out["label_diagnostic_status"] = "empty"
        return out

    out["target_mean"] = float(valid[target_column].mean())
    out["target_std"] = float(valid[target_column].std())
    out["proxy_return_mean"] = float(valid[proxy_return_col].mean())
    out["proxy_return_std"] = float(valid[proxy_return_col].std())

    per_day_corr: list[float] = []
    for _, g in valid.groupby(date_col, sort=False):
        if len(g) < 3:
            continue
        corr = g[target_column].rank(method="average").corr(
            g[proxy_return_col].rank(method="average")
        )
        if pd.notna(corr) and np.isfinite(float(corr)):
            per_day_corr.append(float(corr))
    out["target_proxy_rank_corr_mean"] = float(np.mean(per_day_corr)) if per_day_corr else float("nan")
    out["target_proxy_rank_corr_negative_rate"] = (
        float(np.mean(np.asarray(per_day_corr) < 0.0)) if per_day_corr else float("nan")
    )
    out["target_proxy_rank_corr_days"] = int(len(per_day_corr))
    out["label_diagnostic_status"] = "ok"
    return out


def build_tree_direction_diagnostic_table(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """汇总 P1 各组 raw Rank IC、自动翻转和目标方向是否稳定。"""
    if not rows:
        return pd.DataFrame()
    cols = [
        "group",
        "val_rank_ic",
        "train_rank_ic",
        "tree_score_auto_flipped",
        "effective_val_rank_ic",
        "annualized_excess_vs_market",
    ]
    df = pd.DataFrame(rows).copy()
    present = [c for c in cols if c in df.columns]
    out = df[present].copy()
    raw = pd.to_numeric(out.get("val_rank_ic"), errors="coerce")
    out["raw_val_rank_ic_negative"] = raw < 0.0
    if "tree_score_auto_flipped" in out.columns:
        auto_flipped = out["tree_score_auto_flipped"].astype(bool)
    else:
        auto_flipped = pd.Series(False, index=out.index)
    out["needs_direction_diagnosis"] = out["raw_val_rank_ic_negative"] | auto_flipped
    return out


def build_group_comparison_table(
    rows: list[dict[str, Any]],
    *,
    baseline_group: str = "G0",
) -> pd.DataFrame:
    """把各组结果整理成一张可直接落盘/写文档的比较表。"""
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).copy()
    out = out.sort_values("group", kind="mergesort").reset_index(drop=True)
    base = out.loc[out["group"] == baseline_group]
    if base.empty:
        out["delta_vs_baseline_val_rank_ic"] = np.nan
        out["delta_vs_baseline_proxy_excess"] = np.nan
        return out

    val_ic_col = "effective_val_rank_ic" if "effective_val_rank_ic" in out.columns else "val_rank_ic"
    base_val_ic = float(pd.to_numeric(base[val_ic_col], errors="coerce").iloc[0])
    base_proxy = float(pd.to_numeric(base["annualized_excess_vs_market"], errors="coerce").iloc[0])
    out["delta_vs_baseline_val_rank_ic"] = pd.to_numeric(out[val_ic_col], errors="coerce") - base_val_ic
    out["delta_vs_baseline_proxy_excess"] = (
        pd.to_numeric(out["annualized_excess_vs_market"], errors="coerce") - base_proxy
    )
    delta_pairs = {
        "daily_bt_like_proxy_annualized_excess_vs_market": "delta_vs_baseline_daily_bt_like_proxy_excess",
        "full_backtest_annualized_excess_vs_market": "delta_vs_baseline_full_backtest_excess",
        "rolling_oos_median_ann_return": "delta_vs_baseline_rolling_oos_ann_return",
        "slice_oos_median_ann_return": "delta_vs_baseline_slice_oos_ann_return",
        "rolling_oos_median_ann_excess_vs_market": "delta_vs_baseline_rolling_oos_excess",
        "slice_oos_median_ann_excess_vs_market": "delta_vs_baseline_slice_oos_excess",
        "yearly_excess_median_vs_market": "delta_vs_baseline_yearly_excess_median",
        "key_year_excess_mean_vs_market": "delta_vs_baseline_key_year_excess_mean",
        "topk_boundary_topk_minus_next_mean_return": "delta_vs_baseline_topk_boundary_spread",
        "topk_boundary_switch_in_minus_out_mean_return": "delta_vs_baseline_switch_in_minus_out",
    }
    for col, delta_col in delta_pairs.items():
        if col not in out.columns:
            continue
        base_val = float(pd.to_numeric(base[col], errors="coerce").iloc[0])
        out[delta_col] = pd.to_numeric(out[col], errors="coerce") - base_val

    if "daily_bt_like_proxy_annualized_excess_vs_market" in out.columns:
        daily_proxy = pd.to_numeric(out["daily_bt_like_proxy_annualized_excess_vs_market"], errors="coerce")
        if "daily_proxy_admission_threshold" in out.columns:
            threshold = pd.to_numeric(out["daily_proxy_admission_threshold"], errors="coerce").fillna(0.0)
        else:
            threshold = pd.Series(0.0, index=out.index)
        if "daily_proxy_full_backtest_threshold" in out.columns:
            full_threshold = pd.to_numeric(
                out["daily_proxy_full_backtest_threshold"],
                errors="coerce",
            ).fillna(threshold)
        else:
            full_threshold = threshold
        decisions = [
            classify_daily_proxy_first_decision(
                val,
                admission_threshold=float(adm),
                full_backtest_threshold=float(full),
            )
            for val, adm, full in zip(daily_proxy, threshold, full_threshold)
        ]
        for key in (
            "daily_proxy_first_status",
            "daily_proxy_first_reason",
            "pass_p1_daily_proxy_admission_gate",
            "pass_p1_daily_proxy_full_backtest_gate",
            "daily_proxy_safety_margin_to_admission",
            "daily_proxy_safety_margin_to_full_backtest",
        ):
            out[key] = [d[key] for d in decisions]
        out["primary_result_type"] = "daily_bt_like_proxy"
        out["primary_decision_metric"] = "daily_bt_like_proxy_annualized_excess_vs_market"
        out["legacy_proxy_decision_role"] = "diagnostic_only"

    if "delta_vs_baseline_full_backtest_excess" in out.columns:
        out["pass_p1_val_rank_ic_gate"] = _non_degrade(out["delta_vs_baseline_val_rank_ic"])
        out["pass_p1_proxy_gate"] = _non_degrade(out["delta_vs_baseline_proxy_excess"])
        out["pass_p1_full_backtest_gate"] = _non_degrade(out["delta_vs_baseline_full_backtest_excess"])
        if "delta_vs_baseline_key_year_excess_mean" in out.columns:
            out["pass_p1_key_year_gate"] = _non_degrade(out["delta_vs_baseline_key_year_excess_mean"])
        else:
            out["pass_p1_key_year_gate"] = False
        if "delta_vs_baseline_rolling_oos_excess" in out.columns and out["delta_vs_baseline_rolling_oos_excess"].notna().any():
            out["pass_p1_rolling_oos_gate"] = _non_degrade(out["delta_vs_baseline_rolling_oos_excess"])
        elif "delta_vs_baseline_rolling_oos_ann_return" in out.columns:
            out["pass_p1_rolling_oos_gate"] = _non_degrade(out["delta_vs_baseline_rolling_oos_ann_return"])
        else:
            out["pass_p1_rolling_oos_gate"] = False
        if "delta_vs_baseline_slice_oos_excess" in out.columns and out["delta_vs_baseline_slice_oos_excess"].notna().any():
            out["pass_p1_slice_oos_gate"] = _non_degrade(out["delta_vs_baseline_slice_oos_excess"])
        elif "delta_vs_baseline_slice_oos_ann_return" in out.columns:
            out["pass_p1_slice_oos_gate"] = _non_degrade(out["delta_vs_baseline_slice_oos_ann_return"])
        else:
            out["pass_p1_slice_oos_gate"] = False
        gate_cols = [
            "pass_p1_full_backtest_gate",
            "pass_p1_val_rank_ic_gate",
            "pass_p1_rolling_oos_gate",
            "pass_p1_slice_oos_gate",
            "pass_p1_key_year_gate",
        ]
        if "pass_p1_daily_proxy_admission_gate" in out.columns:
            gate_cols.append("pass_p1_daily_proxy_admission_gate")
        out["pass_p1_promotion_gate"] = out[gate_cols].all(axis=1)
        out.loc[out["group"] == baseline_group, "pass_p1_promotion_gate"] = True
    return out


def build_daily_proxy_first_leaderboard(summary_df: pd.DataFrame) -> pd.DataFrame:
    """生成只按 daily proxy 决策的 P1 leaderboard。"""
    if summary_df.empty:
        return pd.DataFrame()
    preferred = [
        "group",
        "daily_proxy_first_status",
        "daily_bt_like_proxy_annualized_excess_vs_market",
        "daily_proxy_admission_threshold",
        "daily_proxy_full_backtest_threshold",
        "daily_proxy_safety_margin_to_full_backtest",
        "daily_bt_like_proxy_strategy_annualized_return",
        "daily_bt_like_proxy_benchmark_annualized_return",
        "daily_bt_like_proxy_strategy_sharpe_ratio",
        "daily_bt_like_proxy_strategy_max_drawdown",
        "daily_bt_like_proxy_period_beat_rate",
        "daily_bt_like_proxy_n_periods",
        "daily_bt_like_proxy_n_rebalances",
        "daily_bt_like_proxy_avg_turnover_half_l1",
        "topk_boundary_topk_minus_next_mean_return",
        "topk_boundary_switch_in_minus_out_mean_return",
        "topk_boundary_avg_turnover_half_l1",
        "val_rank_ic",
        "effective_val_rank_ic",
        "tree_score_auto_flipped",
        "label_mode",
        "xgboost_objective",
        "research_config_id",
        "output_stem",
        "bundle_dir",
        "full_backtest_annualized_excess_vs_market",
        "full_backtest_skipped_reason",
        "daily_proxy_first_reason",
    ]
    cols = [c for c in preferred if c in summary_df.columns]
    out = summary_df[cols].copy()
    status_order = {
        "full_backtest_candidate": 0,
        "gray_zone": 1,
        "reject": 2,
        "no_daily_proxy": 3,
    }
    out["_status_order"] = out.get("daily_proxy_first_status", pd.Series("", index=out.index)).map(status_order).fillna(9)
    if "daily_bt_like_proxy_annualized_excess_vs_market" in out.columns:
        out["_daily_sort"] = pd.to_numeric(out["daily_bt_like_proxy_annualized_excess_vs_market"], errors="coerce")
    else:
        out["_daily_sort"] = np.nan
    out = out.sort_values(["_status_order", "_daily_sort", "group"], ascending=[True, False, True], kind="mergesort")
    return out.drop(columns=["_status_order", "_daily_sort"]).reset_index(drop=True)


def _format_report_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (bool, np.bool_)):
        return "true" if bool(value) else "false"
    try:
        val = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(val):
        return ""
    return f"{val:.4g}"


def _format_report_pct(value: Any) -> str:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(val):
        return ""
    return f"{val:.2%}"


def _markdown_escape(value: Any) -> str:
    text = _format_report_value(value)
    return text.replace("|", "\\|").replace("\n", " ")


def _markdown_table(df: pd.DataFrame, *, columns: list[str], max_rows: int = 12) -> str:
    cols = [c for c in columns if c in df.columns]
    if df.empty or not cols:
        return "_无可用数据_"
    view = df.loc[:, cols].head(max_rows).copy()
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = [
        "| " + " | ".join(_markdown_escape(row[col]) for col in cols) + " |"
        for _, row in view.iterrows()
    ]
    return "\n".join([header, sep, *rows])


def build_p1_daily_proxy_first_report(
    *,
    summary_df: pd.DataFrame,
    daily_leaderboard_df: pd.DataFrame,
    state_summary_df: pd.DataFrame,
    boundary_df: pd.DataFrame,
    payload: dict[str, Any],
) -> str:
    """生成 P1 daily-proxy-first 固定一页报告。"""
    cfg = payload.get("config", {}) or {}
    output_stem = str(payload.get("output_stem", ""))
    research_config_id = str(payload.get("research_config_id", ""))
    generated_at = str(payload.get("generated_at_utc", ""))
    label_meta = payload.get("label_meta", {}) or {}
    label_spec = cfg.get("label_spec", {}) or {}
    transaction_costs = cfg.get("transaction_costs", {}) or {}

    if daily_leaderboard_df.empty:
        headline = "无 daily proxy leaderboard，不能判读。"
    else:
        status_counts = daily_leaderboard_df.get("daily_proxy_first_status", pd.Series(dtype=object)).value_counts()
        candidate_count = int(status_counts.get("full_backtest_candidate", 0))
        gray_count = int(status_counts.get("gray_zone", 0))
        reject_count = int(status_counts.get("reject", 0))
        best = daily_leaderboard_df.iloc[0]
        headline = (
            f"本轮 full_backtest_candidate={candidate_count}，gray_zone={gray_count}，reject={reject_count}。"
            f"当前第一名 {best.get('group', '')} 的 daily proxy 超额为 "
            f"{_format_report_pct(best.get('daily_bt_like_proxy_annualized_excess_vs_market'))}，"
            f"状态为 {best.get('daily_proxy_first_status', '')}。"
        )

    mechanism_rows: list[dict[str, Any]] = []
    if not summary_df.empty:
        for _, row in summary_df.iterrows():
            daily_excess = row.get("daily_bt_like_proxy_annualized_excess_vs_market")
            strong_up = np.nan
            strong_down = np.nan
            if not state_summary_df.empty and "group" in state_summary_df.columns:
                state_rows = state_summary_df[
                    (state_summary_df["group"] == row.get("group"))
                    & (state_summary_df.get("state_axis") == "return_state")
                ]
                up = state_rows[state_rows.get("state") == "strong_up"]
                down = state_rows[state_rows.get("state") == "strong_down"]
                if not up.empty:
                    strong_up = up["median_excess_return"].iloc[0]
                if not down.empty:
                    strong_down = down["median_excess_return"].iloc[0]
            mechanism_rows.append(
                {
                    "group": row.get("group", ""),
                    "daily_status": row.get("daily_proxy_first_status", ""),
                    "daily_excess": _format_report_pct(daily_excess),
                    "strong_up_median_excess": _format_report_pct(strong_up),
                    "strong_down_median_excess": _format_report_pct(strong_down),
                    "topk_minus_next": _format_report_pct(row.get("topk_boundary_topk_minus_next_mean_return")),
                    "switch_in_minus_out": _format_report_pct(row.get("topk_boundary_switch_in_minus_out_mean_return")),
                    "avg_turnover": _format_report_pct(row.get("topk_boundary_avg_turnover_half_l1")),
                }
            )
    mechanism_df = pd.DataFrame(mechanism_rows)

    if mechanism_df.empty:
        mechanism_takeaway = "机制诊断缺失，下一步先补齐状态切片和 Top-K 边界输出。"
    else:
        bad_switch = mechanism_df["switch_in_minus_out"].astype(str).str.startswith("-").any()
        bad_up = mechanism_df["strong_up_median_excess"].astype(str).str.startswith("-").any()
        if bad_switch and bad_up:
            mechanism_takeaway = "主要风险仍集中在 strong up 捕获不足和换入质量偏弱，暂不适合扩到 G1/G2/G3/G4。"
        elif bad_switch:
            mechanism_takeaway = "Top-K 边界的换入收益仍弱于换出，下一轮应优先解释换入时点或退出损失。"
        elif bad_up:
            mechanism_takeaway = "strong up 状态仍未修复，下一轮应优先解释上涨月持有路径。"
        else:
            mechanism_takeaway = "状态切片和边界没有明显反向，若 daily proxy 达到阈值才考虑正式 full backtest。"

    leaderboard_md = _markdown_table(
        daily_leaderboard_df,
        columns=[
            "group",
            "daily_proxy_first_status",
            "daily_bt_like_proxy_annualized_excess_vs_market",
            "daily_proxy_safety_margin_to_full_backtest",
            "topk_boundary_topk_minus_next_mean_return",
            "topk_boundary_switch_in_minus_out_mean_return",
            "daily_bt_like_proxy_avg_turnover_half_l1",
            "full_backtest_skipped_reason",
        ],
    )
    mechanism_md = _markdown_table(
        mechanism_df,
        columns=[
            "group",
            "daily_status",
            "daily_excess",
            "strong_up_median_excess",
            "strong_down_median_excess",
            "topk_minus_next",
            "switch_in_minus_out",
            "avg_turnover",
        ],
    )
    state_md = _markdown_table(
        state_summary_df,
        columns=[
            "group",
            "state_axis",
            "state",
            "n_months",
            "median_excess_return",
            "beat_rate",
            "strategy_mean_return",
            "benchmark_mean_return",
        ],
        max_rows=18,
    )
    boundary_md = _markdown_table(
        boundary_df,
        columns=[
            "group",
            "period",
            "topk_mean_return",
            "next_bucket_mean_return",
            "topk_minus_next_bucket_return",
            "switched_in_minus_out_return",
            "topk_turnover_half_l1",
        ],
        max_rows=18,
    )

    paths = [
        ("summary_csv", payload.get("summary_csv", "")),
        ("daily_proxy_leaderboard_csv", payload.get("daily_proxy_leaderboard_csv", "")),
        ("daily_proxy_state_summary_csv", payload.get("daily_proxy_state_summary_csv", "")),
        ("topk_boundary_csv", payload.get("topk_boundary_csv", "")),
        ("bundle_manifest_csv", payload.get("bundle_manifest_csv", "")),
    ]
    path_lines = "\n".join(f"- `{name}`: `{path}`" for name, path in paths)

    return f"""# P1 Daily Proxy First Report

生成时间：`{generated_at}`
研究身份：`{research_config_id}`
输出 stem：`{output_stem}`

## 结论

{headline}

{mechanism_takeaway}

## 身份和口径

| 字段 | 值 |
| --- | --- |
| research_topic | `{payload.get("research_topic", "")}` |
| research_config_id | `{research_config_id}` |
| output_stem | `{output_stem}` |
| result_type | `daily_bt_like_proxy` |
| config_source | `{cfg.get("config_source", "")}` |
| p1_experiment_mode | `{cfg.get("p1_experiment_mode", "")}` |
| legacy_proxy_decision_role | `{cfg.get("legacy_proxy_decision_role", "")}` |
| label_mode | `{label_meta.get("label_mode", "")}` |
| label_scope | `{label_meta.get("label_scope", "")}` |
| label_horizons | `{",".join(str(x) for x in cfg.get("label_horizons", []))}` |
| label_weights | `{",".join(str(x) for x in cfg.get("label_weights", []))}` |
| label_transform | `{cfg.get("label_transform", label_spec.get("label_transform", ""))}` |
| label_truncate_quantile | `{cfg.get("label_truncate_quantile", label_spec.get("label_truncate_quantile", ""))}` |
| label_component_columns | `{label_spec.get("label_component_columns", label_meta.get("label_component_columns", ""))}` |
| target_column | `{label_spec.get("target_column", "")}` |
| proxy_horizon | `{cfg.get("proxy_horizon", "")}` |
| xgboost_objective | `{payload.get("xgboost_objective", "")}` |
| benchmark_symbol | `market_ew_proxy` |
| top_k | `{cfg.get("top_k", "")}` |
| rebalance_rule | `{cfg.get("rebalance_rule", "")}` |
| portfolio_method | `{cfg.get("portfolio_method", cfg.get("backtest_portfolio_method", ""))}` |
| execution_mode | `{cfg.get("execution_mode", "")}` |
| proxy_max_turnover | `{_format_report_pct(cfg.get("proxy_max_turnover"))}` |
| backtest_config | `{cfg.get("backtest_config", "")}` |
| backtest_start | `{cfg.get("backtest_start", "")}` |
| backtest_end | `{cfg.get("backtest_end", "")}` |
| backtest_top_k | `{cfg.get("backtest_top_k", "")}` |
| backtest_max_turnover | `{_format_report_pct(cfg.get("backtest_max_turnover"))}` |
| backtest_portfolio_method | `{cfg.get("backtest_portfolio_method", "")}` |
| backtest_prepared_factors_cache | `{cfg.get("backtest_prepared_factors_cache", "")}` |
| transaction_costs_bps | `buy_commission={transaction_costs.get("commission_buy_bps", "")}, sell_commission={transaction_costs.get("commission_sell_bps", "")}, slippage={transaction_costs.get("slippage_bps_per_side", "")}, stamp_duty={transaction_costs.get("stamp_duty_sell_bps", "")}` |
| daily_proxy_admission_threshold | `{_format_report_pct(cfg.get("daily_proxy_admission_threshold"))}` |
| daily_proxy_full_backtest_threshold | `{_format_report_pct(cfg.get("daily_proxy_full_backtest_threshold"))}` |

## Daily Proxy Leaderboard

{leaderboard_md}

## 机制诊断

{mechanism_md}

## 状态切片

{state_md}

## Top-K 边界样本

{boundary_md}

## 文件

{path_lines}
"""
