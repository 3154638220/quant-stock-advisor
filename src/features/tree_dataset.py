"""
历史面板：因子张量 + 前瞻收益，供树模型（XGBoost）训练。

默认前瞻收益与 A 股 T+1 及「次日开盘买入」对齐：``open(t+1+h)/open(t+1)-1``（见 ``forward_returns_tplus1_open``）。
可选 ``forward_settlement="close_to_close"`` 保留旧标签 ``close(t+h)/close(t)-1``。
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from src.features.panel import pivot_close_wide, pivot_field_aligned_to_close, wide_close_to_numpy
from src.features.tensor_alpha import compute_momentum_rsi_torch
from src.features.tensor_base_factors import (
    compute_base_factor_bundle,
    daily_returns_from_close,
    forward_returns_from_close,
    forward_returns_tplus1_open,
)


def default_tree_factor_names() -> Tuple[str, ...]:
    """
    与 recommend 表扩展因子列名一致（原始值，非 z）。

    包含 K 线结构高频降频代理因子（需传入 open_px 到 compute_base_factor_bundle）。
    """
    return (
        "momentum",
        "rsi",
        "atr",
        "realized_vol",
        "turnover_roll_mean",
        "vol_ret_corr",
        "short_reversal",
        "vol_to_turnover",
        "volume_skew_log",
        "bias_short",
        "bias_long",
        "max_single_day_drop",
        "recent_return",
        "price_position",
        "log_market_cap",
        "weekly_kdj_k",
        "weekly_kdj_d",
        "weekly_kdj_j",
        "weekly_kdj_oversold",
        "weekly_kdj_oversold_depth",
        "weekly_kdj_rebound",
        # K 线结构高频降频代理因子
        "intraday_range",
        "upper_shadow_ratio",
        "lower_shadow_ratio",
        "close_open_return",
        "overnight_gap",
        "tail_strength",
        "volume_price_trend",
        "intraday_range_skew",
    )


def _sharpe_label_tplus1(
    open_px: torch.Tensor,
    horizon: int,
    *,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    夏普比率标签：``mean(daily_ret) / std(daily_ret)`` 在 T+1 开盘后 horizon 日内。

    连板妖股波动率极大，标签值被大幅削弱；稳健上涨的股票标签值提升。
    """
    if open_px.dim() != 2:
        raise ValueError("open_px 须为二维")
    n_sym, n_day = open_px.shape
    out = torch.full_like(open_px, float("nan"))
    rets = daily_returns_from_close(open_px)
    for t in range(n_day - horizon - 1):
        buy_day = t + 1
        window = rets[:, buy_day + 1 : buy_day + 1 + horizon]
        if window.shape[1] < 2:
            continue
        fin = torch.isfinite(window)
        n_valid = fin.sum(dim=1).float()
        w0 = torch.where(fin, window, torch.zeros_like(window))
        mean_r = w0.sum(dim=1) / n_valid.clamp(min=1.0)
        var_r = (torch.where(fin, (window - mean_r.unsqueeze(1)) ** 2,
                             torch.zeros_like(window))).sum(dim=1) / n_valid.clamp(min=1.0)
        std_r = torch.sqrt(var_r.clamp(min=0.0))
        sr = mean_r / (std_r + eps)
        valid_mask = n_valid >= 2.0
        out[:, t] = torch.where(valid_mask, sr,
                                torch.tensor(float("nan"), device=open_px.device, dtype=open_px.dtype))
    return out


def _calmar_label_tplus1(
    open_px: torch.Tensor,
    horizon: int,
    *,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    卡玛比率标签：``total_return / max_drawdown`` 在 T+1 开盘后 horizon 日内。

    直接惩罚在持有期内发生过深幅回撤的股票。
    """
    if open_px.dim() != 2:
        raise ValueError("open_px 须为二维")
    n_sym, n_day = open_px.shape
    out = torch.full_like(open_px, float("nan"))
    for t in range(n_day - horizon - 1):
        buy_day = t + 1
        sell_day = buy_day + horizon
        if sell_day >= n_day:
            break
        px_window = open_px[:, buy_day : sell_day + 1]
        total_ret = px_window[:, -1] / (px_window[:, 0] + eps) - 1.0
        cum_max = torch.cummax(px_window, dim=1).values
        drawdown = (px_window - cum_max) / (cum_max + eps)
        max_dd = drawdown.min(dim=1).values.abs()
        max_dd = max_dd.clamp(min=eps)
        calmar = total_ret / max_dd
        valid = torch.isfinite(px_window).all(dim=1)
        out[:, t] = torch.where(valid, calmar,
                                torch.tensor(float("nan"), device=open_px.device, dtype=open_px.dtype))
    return out


def _truncate_extreme_labels(
    labels: np.ndarray,
    *,
    upper_quantile: float = 0.98,
) -> np.ndarray:
    """
    截断极端标签：把前瞻收益最高的那部分截断为分位值。

    告诉模型不要去猜谁是极少数妖股，只要找出前 20% 的好股票。
    """
    labels = labels.copy()
    finite = labels[np.isfinite(labels)]
    if len(finite) < 10:
        return labels
    cap = float(np.quantile(finite, upper_quantile))
    labels[np.isfinite(labels) & (labels > cap)] = cap
    return labels


def long_factor_panel_from_daily(
    df: pd.DataFrame,
    *,
    horizon: int,
    min_valid_days: int,
    momentum_window: int,
    rsi_period: int,
    atr_period: int,
    vol_window: int,
    turnover_window: int,
    vp_corr_window: int,
    reversal_window: int,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    factor_names: Optional[Sequence[str]] = None,
    forward_settlement: str = "tplus1_open",
    label_transform: str = "raw",
    label_truncate_quantile: float = 0.98,
    bias_window_short: int = 20,
    bias_window_long: int = 60,
    max_drop_window: int = 20,
    recent_return_window: int = 3,
    price_position_window: int = 250,
    # K 线结构高频降频因子窗口
    tail_window: int = 10,
    vpt_window: int = 20,
    range_skew_window: int = 20,
    # 因子正交化（在传入树模型前）
    orthogonalize: bool = False,
    orthogonalize_method: str = "symmetric",
) -> pd.DataFrame:
    """
    将日线长表转为 (symbol, trade_date) 长表，含原始因子列与 ``forward_ret_{h}d``。

    对每个交易日 ``t``（需能计算前瞻收益），在截面上每只股票一行。

    forward_settlement
        ``tplus1_open``（默认）：``open(t+1+h)/open(t+1)-1``；``close_to_close``：``close(t+h)/close(t)-1``。

    label_transform
        ``raw``（默认）：使用原始前瞻收益。
        ``sharpe``：使用夏普比率标签（惩罚波动）。
        ``calmar``：使用卡玛比率标签（惩罚回撤）。
        ``truncate``：截断极端标签后使用原始收益。

    label_truncate_quantile
        截断分位数，仅 ``label_transform='truncate'`` 时生效。

    orthogonalize
        若为 True，在构建 panel 后按交易日截面对因子矩阵做正交化，
        剥离高度相关因子之间的冗余（推荐在 XGBoost 训练前启用）。

    orthogonalize_method
        ``symmetric``（默认，Löwdin 对称正交化）或 ``gram_schmidt``。
    """
    if df.empty:
        return pd.DataFrame()
    if horizon < 1:
        raise ValueError("horizon 须 >= 1")
    fs = str(forward_settlement).lower().strip()
    if fs not in ("tplus1_open", "close_to_close"):
        raise ValueError("forward_settlement 须为 tplus1_open 或 close_to_close")

    lt = str(label_transform).lower().strip()
    if lt not in ("raw", "sharpe", "calmar", "truncate"):
        raise ValueError("label_transform 须为 raw/sharpe/calmar/truncate")

    names = tuple(factor_names) if factor_names is not None else default_tree_factor_names()

    wide, sym_list, dates = pivot_close_wide(df, min_valid_days=min_valid_days)
    close_np = wide_close_to_numpy(wide)
    n_sym, n_day = close_np.shape
    min_days = horizon + 2 if fs == "tplus1_open" else horizon + 1
    if n_day < min_days:
        return pd.DataFrame()

    dev = torch.device(device)
    close_t = torch.from_numpy(close_np).to(device=dev, dtype=dtype)

    wide_h = pivot_field_aligned_to_close(df, "high", wide)
    wide_l = pivot_field_aligned_to_close(df, "low", wide)
    wide_vol = pivot_field_aligned_to_close(df, "volume", wide)
    wide_to = pivot_field_aligned_to_close(df, "turnover", wide)
    wide_o = pivot_field_aligned_to_close(df, "open", wide)
    open_np = wide_o.to_numpy(dtype=np.float64)
    high_t = torch.from_numpy(wide_h.to_numpy(dtype=np.float64)).to(device=dev, dtype=dtype)
    low_t = torch.from_numpy(wide_l.to_numpy(dtype=np.float64)).to(device=dev, dtype=dtype)
    vol_t = torch.from_numpy(wide_vol.to_numpy(dtype=np.float64)).to(device=dev, dtype=dtype)
    to_t = torch.from_numpy(wide_to.to_numpy(dtype=np.float64)).to(device=dev, dtype=dtype)

    mom, rsi = compute_momentum_rsi_torch(
        close_np,
        momentum_window=momentum_window,
        rsi_period=rsi_period,
        device=device,
        dtype=dtype,
    )
    open_t = torch.from_numpy(open_np).to(device=dev, dtype=dtype)

    bundle = compute_base_factor_bundle(
        close_t,
        volume=vol_t,
        turnover=to_t,
        high=high_t,
        low=low_t,
        open_px=open_t,
        trade_dates=dates,
        vol_window=vol_window,
        turnover_window=turnover_window,
        vp_corr_window=vp_corr_window,
        reversal_window=reversal_window,
        atr_period=atr_period,
        annualize_vol=False,
        bias_window_short=bias_window_short,
        bias_window_long=bias_window_long,
        max_drop_window=max_drop_window,
        recent_return_window=recent_return_window,
        price_position_window=price_position_window,
        tail_window=tail_window,
        vpt_window=vpt_window,
        range_skew_window=range_skew_window,
        include_intraday=True,
    )
    if lt == "sharpe" and fs == "tplus1_open":
        fwd = _sharpe_label_tplus1(open_t, horizon)
    elif lt == "calmar" and fs == "tplus1_open":
        fwd = _calmar_label_tplus1(open_t, horizon)
    elif fs == "tplus1_open":
        fwd = forward_returns_tplus1_open(open_t, horizon)
    else:
        fwd = forward_returns_from_close(close_t, horizon)
    fwd_np = fwd.detach().cpu().numpy()
    if lt == "truncate":
        for t_idx in range(n_day):
            fwd_np[:, t_idx] = _truncate_extreme_labels(
                fwd_np[:, t_idx], upper_quantile=label_truncate_quantile
            )

    tensors: dict[str, np.ndarray] = {
        "momentum": mom.detach().cpu().numpy(),
        "rsi": rsi.detach().cpu().numpy(),
    }
    for key in (
        "atr",
        "realized_vol",
        "turnover_roll_mean",
        "vol_ret_corr",
        "short_reversal",
        "vol_to_turnover",
        "volume_skew_log",
        "bias_short",
        "bias_long",
        "max_single_day_drop",
        "recent_return",
        "price_position",
        "log_market_cap",
        "weekly_kdj_k",
        "weekly_kdj_d",
        "weekly_kdj_j",
        "weekly_kdj_oversold",
        "weekly_kdj_oversold_depth",
        "weekly_kdj_rebound",
        "limit_move_count",
        # K 线结构高频降频代理因子
        "intraday_range",
        "upper_shadow_ratio",
        "lower_shadow_ratio",
        "close_open_return",
        "overnight_gap",
        "tail_strength",
        "volume_price_trend",
        "intraday_range_skew",
    ):
        t = bundle.get(key)
        tensors[key] = t.detach().cpu().numpy() if t is not None else np.full((n_sym, n_day), np.nan)

    y_col = f"forward_ret_{horizon}d"
    rows: List[dict] = []
    for t in range(n_day):
        if fs == "tplus1_open":
            if t + 1 + horizon >= n_day:
                continue
        elif t + horizon >= n_day:
            continue
        trade_date = dates[t]
        if hasattr(trade_date, "date"):
            td = trade_date.date()
        else:
            td = pd.Timestamp(trade_date).date()

        y_sl = fwd_np[:, t]
        mask = np.isfinite(y_sl)
        for name in names:
            if name not in tensors:
                raise ValueError(f"未知因子名: {name!r}")
            mask &= np.isfinite(tensors[name][:, t])

        for i in np.where(mask)[0]:
            sym = sym_list[i]
            rec: dict = {
                "symbol": sym,
                "trade_date": pd.Timestamp(td),
                y_col: float(y_sl[i]),
            }
            for name in names:
                rec[name] = float(tensors[name][i, t])
            rows.append(rec)

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out = out.sort_values(["trade_date", "symbol"]).reset_index(drop=True)

    # 可选：按交易日截面对因子矩阵做正交化，剥离冗余相关性
    if orthogonalize and len(names) >= 2:
        from src.features.orthogonalize import orthogonalize_panel_by_date

        # 仅对实际存在于 out 中的因子列正交化（跳过全 nan 列）
        valid_names = [n for n in names if n in out.columns]
        if len(valid_names) >= 2:
            out = orthogonalize_panel_by_date(
                out,
                list(valid_names),
                method=orthogonalize_method,
                date_col="trade_date",
                suffix="_orth",
            )
            # 用正交化列替换原始列（保留原始列供调试对比）
            for n in valid_names:
                orth_col = f"{n}_orth"
                if orth_col in out.columns:
                    out[n] = out[orth_col]
                    out = out.drop(columns=[orth_col])

    return out


def long_ohlcv_history_from_daily(
    df: pd.DataFrame,
    *,
    horizon: int,
    min_valid_days: int,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    forward_settlement: str = "tplus1_open",
) -> pd.DataFrame:
    """
    日线长表 → (symbol, trade_date) 长表，含 **原始** OHLCV 与 ``forward_ret_{h}d``。

    用于阶段三：以过去 ``seq_len`` 根 K 线的 OHLCV 序列为特征、前瞻收益为标签训练 GRU/LSTM/Transformer。

    forward_settlement 默认 ``tplus1_open``，与 ``long_factor_panel_from_daily`` 一致。
    """
    if df.empty:
        return pd.DataFrame()
    if horizon < 1:
        raise ValueError("horizon 须 >= 1")
    fs = str(forward_settlement).lower().strip()
    if fs not in ("tplus1_open", "close_to_close"):
        raise ValueError("forward_settlement 须为 tplus1_open 或 close_to_close")

    wide, sym_list, dates = pivot_close_wide(df, min_valid_days=min_valid_days)
    close_np = wide_close_to_numpy(wide)
    n_sym, n_day = close_np.shape
    min_days = horizon + 2 if fs == "tplus1_open" else horizon + 1
    if n_day < min_days:
        return pd.DataFrame()

    dev = torch.device(device)
    close_t = torch.from_numpy(close_np).to(device=dev, dtype=dtype)

    wide_o = pivot_field_aligned_to_close(df, "open", wide)
    wide_h = pivot_field_aligned_to_close(df, "high", wide)
    wide_l = pivot_field_aligned_to_close(df, "low", wide)
    wide_vol = pivot_field_aligned_to_close(df, "volume", wide)

    o_np = wide_o.to_numpy(dtype=np.float64)
    h_np = wide_h.to_numpy(dtype=np.float64)
    l_np = wide_l.to_numpy(dtype=np.float64)
    v_np = wide_vol.to_numpy(dtype=np.float64)

    if fs == "tplus1_open":
        open_t = torch.from_numpy(o_np).to(device=dev, dtype=dtype)
        fwd = forward_returns_tplus1_open(open_t, horizon)
    else:
        fwd = forward_returns_from_close(close_t, horizon)
    fwd_np = fwd.detach().cpu().numpy()
    y_col = f"forward_ret_{horizon}d"

    rows: List[dict] = []
    for t in range(n_day):
        if fs == "tplus1_open":
            if t + 1 + horizon >= n_day:
                continue
        elif t + horizon >= n_day:
            continue
        trade_date = dates[t]
        if hasattr(trade_date, "date"):
            td = trade_date.date()
        else:
            td = pd.Timestamp(trade_date).date()

        y_sl = fwd_np[:, t]
        mask = np.isfinite(y_sl)
        for i in np.where(mask)[0]:
            sym = sym_list[i]
            o_ = o_np[i, t]
            h_ = h_np[i, t]
            l_ = l_np[i, t]
            c_ = close_np[i, t]
            vol_ = v_np[i, t]
            if not all(np.isfinite([o_, h_, l_, c_, vol_])):
                continue
            rows.append(
                {
                    "symbol": sym,
                    "trade_date": pd.Timestamp(td),
                    "open": float(o_),
                    "high": float(h_),
                    "low": float(l_),
                    "close": float(c_),
                    "volume": float(vol_),
                    y_col: float(y_sl[i]),
                }
            )

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["symbol", "trade_date"]).reset_index(drop=True)
    return out


def long_ohlcv_last_window_table(
    sym_list: Sequence[str],
    dates: Sequence,
    *,
    wide_open: pd.DataFrame,
    wide_high: pd.DataFrame,
    wide_low: pd.DataFrame,
    wide_close: pd.DataFrame,
    wide_volume: pd.DataFrame,
    seq_len: int,
) -> pd.DataFrame:
    """
    从与 ``pivot_close_wide`` 对齐的 OHLCV 宽表截取**最近** ``seq_len`` 个交易日，展开为长表。

    供 ``predict_timeseries_bundle_last`` 等序列模型推理工具使用。
    """
    if seq_len < 1:
        raise ValueError("seq_len 须 >= 1")
    cols = list(dates)
    if len(cols) < seq_len:
        raise ValueError("交易日数量不足 seq_len")
    take = cols[-seq_len:]
    rows: List[dict] = []
    for si, sym in enumerate(sym_list):
        for dt in take:
            o = float(wide_open.loc[sym, dt])
            h = float(wide_high.loc[sym, dt])
            lo = float(wide_low.loc[sym, dt])
            c = float(wide_close.loc[sym, dt])
            v = float(wide_volume.loc[sym, dt])
            if not all(np.isfinite([o, h, lo, c, v])):
                continue
            td = dt.date() if hasattr(dt, "date") else pd.Timestamp(dt).date()
            rows.append(
                {
                    "symbol": sym,
                    "trade_date": pd.Timestamp(td),
                    "open": o,
                    "high": h,
                    "low": lo,
                    "close": c,
                    "volume": v,
                }
            )
    return pd.DataFrame(rows)


def time_based_train_val_split(
    df: pd.DataFrame,
    *,
    date_col: str = "trade_date",
    val_frac: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """按交易日排序后，后段日期为验证集（避免随机切分泄漏未来）。"""
    if df.empty:
        return df.copy(), df.copy()
    dates = sorted(pd.to_datetime(df[date_col]).dt.normalize().unique())
    if len(dates) < 3:
        return df.copy(), df.iloc[0:0].copy()
    cut = max(1, int(len(dates) * (1.0 - float(val_frac))))
    cutoff_date = dates[cut - 1]
    dnorm = pd.to_datetime(df[date_col]).dt.normalize()
    train = df[dnorm <= cutoff_date].copy()
    val = df[dnorm > cutoff_date].copy()
    return train, val
