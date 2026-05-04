"""标签构建与 daily-proxy 评估管线。

从 src/models/xtree/p1_workflow.py 提取核心逻辑：
- 训练标签构建（rank_fusion / market_relative / monthly_investable 等模式）
- 可投资持有期收益面板
- 调仓日选择

不放 CLI 参数解析与文件 I/O 编排。
"""

from __future__ import annotations

import re
from typing import Any, Iterable

import numpy as np
import pandas as pd

from src.backtest.engine import build_open_to_open_returns


def _slugify_token(value: Any) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "na"


def build_p1_training_label(
    panel: pd.DataFrame,
    *,
    label_columns: Iterable[str],
    label_weights: Iterable[float],
    label_mode: str = "rank_fusion",
    date_col: str = "trade_date",
    out_col: str = "forward_ret_fused",
) -> tuple[pd.DataFrame, str, dict[str, Any]]:
    """构造 P1 树模型训练标签。

    ``rank_fusion`` 复用既有截面 rank 融合；``top_bucket_rank_fusion`` 只保留
    截面顶部/底部 20% 的 rank 信号；``raw_fusion`` 直接融合原始收益；
    ``market_relative`` / ``benchmark_relative`` 先按日扣掉截面等权收益，再融合。
    ``up_capture_market_relative`` 在市场前向收益为正的截面上放大标签幅度。
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
        "rank_fusion", "top_bucket_rank_fusion", "raw_fusion",
        "market_relative", "benchmark_relative", "up_capture_market_relative",
    }:
        raise ValueError(
            "label_mode 须为 rank_fusion/top_bucket_rank_fusion/raw_fusion/"
            "market_relative/benchmark_relative/up_capture_market_relative"
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
            "cross_section_top_bottom_bucket" if mode == "top_bucket_rank_fusion"
            else "cross_section_relative" if mode == "rank_fusion" else mode
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
    if daily_df.empty:
        return pd.DataFrame()
    daily = daily_df.copy()
    daily[symbol_col] = daily[symbol_col].astype(str).str.zfill(6)
    daily[date_col] = pd.to_datetime(daily[date_col], errors="coerce").dt.normalize()
    exe = str(execution_mode).lower().strip()
    if exe == "tplus1_open":
        out = build_open_to_open_returns(daily, date_col=date_col, sym_col=symbol_col, zero_if_limit_up_open=False)
    elif exe == "close_to_close":
        d = daily[[symbol_col, date_col, "close"]].copy()
        d["close"] = pd.to_numeric(d["close"], errors="coerce")
        d = d.dropna(subset=[date_col, "close"]).sort_values([symbol_col, date_col])
        d["_ret"] = d.groupby(symbol_col, sort=False)["close"].pct_change()
        out = d.pivot(index=date_col, columns=symbol_col, values="_ret").sort_index()
    else:
        raise ValueError(f"当前仅支持 tplus1_open/close_to_close，收到: {execution_mode}")
    out.index = pd.to_datetime(out.index, errors="coerce").normalize()
    return out.sort_index().astype(np.float64)


from src.pipeline.monthly_dataset import select_month_end_signal_dates


def select_rebalance_dates(
    all_dates: Iterable[pd.Timestamp], rebalance_rule: str
) -> pd.DataFrame:
    dates = select_month_end_signal_dates(list(all_dates), rebalance_rule=rebalance_rule)
    return pd.DataFrame({"trade_date": sorted(set(dates))})


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
    asset_returns = _daily_asset_return_matrix(daily_df, execution_mode=execution_mode, symbol_col=symbol_col, date_col=date_col)
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
    mode = _slugify_token(label_mode or "monthly_investable")
    if mode not in {
        "monthly_investable", "monthly_investable_market_relative",
        "monthly_investable_up_capture_market_relative",
    }:
        raise ValueError(
            "monthly label_mode 须为 monthly_investable/monthly_investable_market_relative/"
            "monthly_investable_up_capture_market_relative"
        )
    out = build_investable_period_return_panel(
        panel, daily_df, rebalance_rule=rebalance_rule, execution_mode=execution_mode,
        out_col=out_col, date_col=date_col,
    )
    if out.empty:
        meta = {
            "label_mode": mode, "label_scope": mode,
            "label_component_columns": out_col, "label_weights_normalized": "1",
            "label_market_proxy": "", "label_rebalance_rule": str(rebalance_rule),
            "label_execution_mode": str(execution_mode),
        }
        return out, out_col, meta

    if mode == "monthly_investable_market_relative":
        market_ret = out.groupby(date_col, sort=False)[out_col].transform("mean")
        out[out_col] = out[out_col] - market_ret
    elif mode == "monthly_investable_up_capture_market_relative":
        market_ret = out.groupby(date_col, sort=False)[out_col].transform("mean")
        multiplier = np.where(market_ret.to_numpy(dtype=np.float64) > 0.0, 2.0, 1.0)
        out[out_col] = (out[out_col] - market_ret) * multiplier

    meta = {
        "label_mode": mode, "label_scope": mode,
        "label_component_columns": out_col, "label_weights_normalized": "1",
        "label_market_proxy": (
            "same_date_cross_section_equal_weight"
            if mode != "monthly_investable" else ""
        ),
        "label_up_capture_multiplier": 2.0 if mode == "monthly_investable_up_capture_market_relative" else 1.0,
    }
    return out, out_col, meta
