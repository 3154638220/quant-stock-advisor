"""
GPU 批量因子示例：10 日动量、Wilder RSI（PyTorch）。

无 NVIDIA GPU 时可将 device 设为 ``cpu``；Jetson 上建议使用 ``cuda``。
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

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
