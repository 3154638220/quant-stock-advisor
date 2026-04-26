"""
GPU 批量因子示例：10 日动量、Wilder RSI（PyTorch）。

扩展包含一个「已完成周线」KDJ 对齐工具：把日线 OHLC 压到周线后计算 K/D/J，
并将每个交易日映射到最近一个已完成周线值。这样在周内不会读取尚未收官的周线。

无 NVIDIA GPU 时可将 device 设为 ``cpu``；Jetson 上建议使用 ``cuda``。
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


def _to_close_tensor(
    close: Union[np.ndarray, pd.DataFrame, torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """接受 (N, T) 收盘价矩阵或含 close 列的宽表。"""
    if isinstance(close, torch.Tensor):
        x = close
    elif isinstance(close, pd.DataFrame):
        # 行=标的，列=时间，值为 close
        x = torch.from_numpy(close.to_numpy(dtype=np.float64))
    else:
        x = torch.from_numpy(np.asarray(close, dtype=np.float64))
    return x.to(device=device, dtype=dtype)


def momentum_n(
    close: torch.Tensor,
    window: int,
) -> torch.Tensor:
    """
    批量 N 日动量: close(t)/close(t-N) - 1。

    Parameters
    ----------
    close : Tensor
        形状 ``(num_symbols, num_days)``。
    window : int
        例如 10 表示 10 个交易日动量。

    Returns
    -------
    Tensor
        与 ``close`` 同形状，前 ``window`` 列为 NaN。
    """
    if close.dim() != 2:
        raise ValueError("close 须为二维 (symbols, days)")
    if window < 1:
        raise ValueError("window 须 >= 1")
    out = torch.full_like(close, float("nan"))
    # 对齐：第 window 列起，用 window 天前的收盘价作分母（不循环卷绕）
    out[:, window:] = close[:, window:] / close[:, :-window] - 1.0
    return out


def rsi_wilder(
    close: torch.Tensor,
    period: int = 14,
) -> torch.Tensor:
    """
    Wilder RSI，时间维为最后一维；与 ``close`` 列数一致。

    首根 K 线无涨跌故 RSI 为 NaN；自第 ``period`` 根收盘价起有第一个有效 RSI。
    """
    if close.dim() != 2:
        raise ValueError("close 须为二维 (symbols, days)")
    if period < 2:
        raise ValueError("period 须 >= 2")

    diff = close[:, 1:] - close[:, :-1]
    gain = torch.clamp(diff, min=0.0)
    loss = torch.clamp(-diff, min=0.0)
    b, tm1 = gain.shape
    rsi = torch.full((b, close.shape[1]), float("nan"), device=close.device, dtype=close.dtype)
    avg_g = gain[:, :period].mean(dim=1)
    avg_l = loss[:, :period].mean(dim=1)
    rs = avg_g / (avg_l + 1e-12)
    rsi[:, period] = 100.0 - 100.0 / (1.0 + rs)
    for j in range(period, tm1):
        avg_g = (avg_g * (period - 1) + gain[:, j]) / period
        avg_l = (avg_l * (period - 1) + loss[:, j]) / period
        rs = avg_g / (avg_l + 1e-12)
        rsi[:, j + 1] = 100.0 - 100.0 / (1.0 + rs)
    return rsi


def compute_momentum_rsi_torch(
    close: Union[np.ndarray, pd.DataFrame, torch.Tensor],
    *,
    momentum_window: int = 10,
    rsi_period: int = 14,
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    对多标的收盘价矩阵一次性计算动量与 RSI（最后一维为时间）。

    Returns
    -------
    momentum : Tensor
        形状与 ``close`` 相同。
    rsi : Tensor
        与 ``close`` 时间对齐；长度与 close 列数一致（首列为 NaN）。
    """
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ct = _to_close_tensor(close, dev, dtype)
    mom = momentum_n(ct, momentum_window)
    rsi = rsi_wilder(ct, rsi_period)
    return mom, rsi


def _normalize_trade_dates(
    trade_dates: Sequence[object],
) -> pd.DatetimeIndex:
    dt = pd.to_datetime(list(trade_dates), errors="coerce")
    if dt.isna().any():
        raise ValueError("trade_dates 含无法解析的日期")
    return pd.DatetimeIndex(dt)


def _week_group_slices(
    trade_dates: Sequence[object],
) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray]:
    dt = _normalize_trade_dates(trade_dates)
    periods = dt.to_period("W-FRI")
    codes, _ = pd.factorize(periods)
    starts: list[int] = []
    ends: list[int] = []
    last = -1
    for idx, code in enumerate(codes):
        if idx == 0 or code != last:
            starts.append(idx)
            if idx > 0:
                ends.append(idx - 1)
            last = int(code)
    ends.append(len(codes) - 1)
    return dt, np.asarray(starts, dtype=np.int64), np.asarray(ends, dtype=np.int64)


def _nanmax_2d(x: torch.Tensor) -> torch.Tensor:
    """逐行 nanmax，兼容较旧 PyTorch。"""
    fin = torch.isfinite(x)
    masked = torch.where(fin, x, torch.tensor(float("-inf"), device=x.device, dtype=x.dtype))
    vals = masked.max(dim=1).values
    any_valid = fin.any(dim=1)
    return torch.where(
        any_valid,
        vals,
        torch.tensor(float("nan"), device=x.device, dtype=x.dtype),
    )


def _nanmin_2d(x: torch.Tensor) -> torch.Tensor:
    """逐行 nanmin，兼容较旧 PyTorch。"""
    fin = torch.isfinite(x)
    masked = torch.where(fin, x, torch.tensor(float("inf"), device=x.device, dtype=x.dtype))
    vals = masked.min(dim=1).values
    any_valid = fin.any(dim=1)
    return torch.where(
        any_valid,
        vals,
        torch.tensor(float("nan"), device=x.device, dtype=x.dtype),
    )


def weekly_kdj_from_daily(
    close: Union[np.ndarray, pd.DataFrame, torch.Tensor],
    high: Union[np.ndarray, pd.DataFrame, torch.Tensor],
    low: Union[np.ndarray, pd.DataFrame, torch.Tensor],
    *,
    trade_dates: Sequence[object],
    n: int = 9,
    k_smooth: float = 3.0,
    d_smooth: float = 3.0,
    initial_k: float = 50.0,
    initial_d: float = 50.0,
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    基于日线 OHLC 计算「最近已完成周线」KDJ，并对齐回日频。

    映射规则：
    - 周内（非该周最后一个交易日）使用上一根已完成周线的 K/D/J；
    - 该周最后一个交易日收盘后，可使用本周已完成周线的 K/D/J。

    Parameters
    ----------
    close, high, low
        形状均为 ``(num_symbols, num_days)``。
    trade_dates
        与时间维一一对应的交易日序列。
    n
        RSV 的周线回看窗口，默认 9。
    """
    if n < 1:
        raise ValueError("n 须 >= 1")
    if k_smooth <= 0 or d_smooth <= 0:
        raise ValueError("k_smooth/d_smooth 须 > 0")

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    c = _to_close_tensor(close, dev, dtype)
    h = _to_close_tensor(high, dev, dtype)
    l = _to_close_tensor(low, dev, dtype)
    if not (c.shape == h.shape == l.shape):
        raise ValueError("close/high/low 形状须一致")
    if c.dim() != 2:
        raise ValueError("close/high/low 须为二维 (symbols, days)")

    _, starts, ends = _week_group_slices(trade_dates)
    if len(ends) == 0:
        nan = torch.full_like(c, float("nan"))
        return nan, nan.clone(), nan.clone()

    num_symbols, num_days = c.shape
    num_weeks = len(ends)
    wk_close = torch.full((num_symbols, num_weeks), float("nan"), device=dev, dtype=dtype)
    wk_high = torch.full_like(wk_close, float("nan"))
    wk_low = torch.full_like(wk_close, float("nan"))

    for w_idx, (s_idx, e_idx) in enumerate(zip(starts, ends)):
        sl_h = h[:, s_idx : e_idx + 1]
        sl_l = l[:, s_idx : e_idx + 1]
        wk_close[:, w_idx] = c[:, e_idx]
        wk_high[:, w_idx] = _nanmax_2d(sl_h)
        wk_low[:, w_idx] = _nanmin_2d(sl_l)

    k_week = torch.full_like(wk_close, float("nan"))
    d_week = torch.full_like(wk_close, float("nan"))
    j_week = torch.full_like(wk_close, float("nan"))
    prev_k = torch.full((num_symbols,), float(initial_k), device=dev, dtype=dtype)
    prev_d = torch.full((num_symbols,), float(initial_d), device=dev, dtype=dtype)
    alpha_k = 1.0 / float(k_smooth)
    alpha_d = 1.0 / float(d_smooth)

    for w_idx in range(num_weeks):
        lo_idx = max(0, w_idx - n + 1)
        win_high = _nanmax_2d(wk_high[:, lo_idx : w_idx + 1])
        win_low = _nanmin_2d(wk_low[:, lo_idx : w_idx + 1])
        denom = win_high - win_low
        valid = torch.isfinite(wk_close[:, w_idx]) & torch.isfinite(win_high) & torch.isfinite(win_low)
        rsv = torch.full((num_symbols,), float("nan"), device=dev, dtype=dtype)
        good = valid & (denom.abs() > 1e-12)
        flat = valid & ~good
        rsv[good] = (wk_close[good, w_idx] - win_low[good]) / denom[good] * 100.0
        rsv[flat] = 50.0

        cur_k = prev_k * (1.0 - alpha_k)
        cur_d = prev_d * (1.0 - alpha_d)
        cur_k[valid] = prev_k[valid] * (1.0 - alpha_k) + alpha_k * rsv[valid]
        cur_d[valid] = prev_d[valid] * (1.0 - alpha_d) + alpha_d * cur_k[valid]
        cur_j = 3.0 * cur_k - 2.0 * cur_d
        cur_k[~valid] = float("nan")
        cur_d[~valid] = float("nan")
        cur_j[~valid] = float("nan")
        k_week[:, w_idx] = cur_k
        d_week[:, w_idx] = cur_d
        j_week[:, w_idx] = cur_j
        prev_k = torch.where(valid, cur_k, prev_k)
        prev_d = torch.where(valid, cur_d, prev_d)

    k_day = torch.full((num_symbols, num_days), float("nan"), device=dev, dtype=dtype)
    d_day = torch.full_like(k_day, float("nan"))
    j_day = torch.full_like(k_day, float("nan"))
    for w_idx, (s_idx, e_idx) in enumerate(zip(starts, ends)):
        prev_idx = w_idx - 1
        if s_idx < e_idx and prev_idx >= 0:
            k_day[:, s_idx:e_idx] = k_week[:, prev_idx].unsqueeze(1)
            d_day[:, s_idx:e_idx] = d_week[:, prev_idx].unsqueeze(1)
            j_day[:, s_idx:e_idx] = j_week[:, prev_idx].unsqueeze(1)
        k_day[:, e_idx] = k_week[:, w_idx]
        d_day[:, e_idx] = d_week[:, w_idx]
        j_day[:, e_idx] = j_week[:, w_idx]
    return k_day, d_day, j_day


def demo_from_random_walk(
    num_symbols: int = 256,
    num_days: int = 300,
    *,
    device: Optional[str] = None,
) -> None:
    """随机游走价格张量上跑一轮因子，用于无数据时的冒烟测试。"""
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(0)
    ret = torch.randn(num_symbols, num_days - 1, device=dev, dtype=torch.float32) * 0.02
    close = 100.0 * torch.cumprod(1.0 + ret, dim=1)
    close = F.pad(close, (1, 0), value=100.0)
    mom, rsi = compute_momentum_rsi_torch(close, device=dev)
    print(
        "device",
        dev,
        "mom_last_valid",
        float(mom[:, -1].nanmean()),
        "rsi_last",
        float(rsi[:, -1].nanmean()),
    )


# --- JAX 侧等价写法（需自行按 Jetson 文档安装 jax/jaxlib；与 PyTorch 二选一即可）---
# import jax
# import jax.numpy as jnp
#
# def momentum_n_jax(close: jnp.ndarray, window: int) -> jnp.ndarray:
#     out = jnp.full_like(close, jnp.nan)
#     out = out.at[:, window:].set(
#         close[:, window:] / close[:, :-window] - 1.0
#     )
#     return out


if __name__ == "__main__":
    demo_from_random_walk()
