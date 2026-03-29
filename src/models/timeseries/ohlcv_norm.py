"""OHLCV 序列锚定归一化：与训练/推理一致，无量纲化价格尺度。"""

from __future__ import annotations

import numpy as np

# 特征列顺序须与 bundle / inference_config 一致
OHLCV_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "volume")


def normalize_ohlcv_anchor(
    x: np.ndarray,
    *,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    对 ``(T, 5)`` 或 ``(N, T, 5)`` 的 OHLCV 做锚定归一化。

    - 价格：各行 open/high/low/close 除以**窗口末行**收盘价，使末行 close≈1。
    - 成交量：``log1p(volume)`` 后在窗口内做零均值单位方差（单序列 T 维）。

    推理时与训练使用同一规则，不依赖训练集统计量。
    """
    a = np.asarray(x, dtype=np.float64)
    if a.ndim == 2:
        return _normalize_one(a, eps=eps)
    if a.ndim == 3:
        out = np.empty_like(a, dtype=np.float64)
        for i in range(a.shape[0]):
            out[i] = _normalize_one(a[i], eps=eps)
        return out
    raise ValueError("x 须为 (T,5) 或 (N,T,5)")


def _normalize_one(w: np.ndarray, *, eps: float) -> np.ndarray:
    if w.shape[-1] != 5:
        raise ValueError("最后一维须为 5 (OHLCV)")
    last_c = float(w[-1, 3])
    if not np.isfinite(last_c) or last_c <= 0:
        return w
    p = w[:, :4].copy() / (last_c + eps)
    lv = np.log1p(np.maximum(w[:, 4], 0.0))
    m = float(np.mean(lv))
    s = float(np.std(lv))
    if not np.isfinite(s) or s < eps:
        vnorm = np.zeros_like(lv)
    else:
        vnorm = (lv - m) / (s + eps)
    out = np.column_stack([p[:, 0], p[:, 1], p[:, 2], p[:, 3], vnorm])
    return out.astype(np.float64)
