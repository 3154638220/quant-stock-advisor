"""
K 线结构高频降频代理因子（日线 OHLCV 衍生）。

在无法获取分钟级数据的情况下，通过日线 OHLCV 结构近似刻画日内微观特征：
- 日内振幅（``intraday_range``）：(high - low) / open，衡量日内波动强度
- 上影线比率（``upper_shadow_ratio``）：上影线长度 / (high - low)，刻画上方压力
- 下影线比率（``lower_shadow_ratio``）：下影线长度 / (high - low)，刻画下方支撑
- 日内涨跌（``close_open_return``）：close/open - 1，近似「尾盘相对开盘」方向
- 隔夜跳空（``overnight_gap``）：open/prev_close - 1，衡量隔夜消息冲击强度
- 尾盘强度（``tail_strength``）：滚动 close_open_return 均值，刻画尾盘持续买入/卖出
- 量价趋势因子（``volume_price_trend``）：量 × |日内收益|，刻画量价同向性
- 振幅偏度（``range_skew``）：滚动日内振幅的截面偏度，识别近期波动分布

所有输入为 ``(num_symbols, num_days)`` 的 PyTorch 张量，与 ``tensor_base_factors`` 接口一致。
"""

from __future__ import annotations

from typing import Optional

import torch

from .tensor_base_factors import (
    _rolling_nan_mean,
    _rolling_nan_std,
    _rowwise_skew,
    daily_returns_from_close,
)


def intraday_range(
    high: torch.Tensor,
    low: torch.Tensor,
    open_px: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    日内振幅：``(high - low) / open``。

    值越大表示日内波动越剧烈；妖股通常振幅极大。
    """
    if high.shape != low.shape or high.shape != open_px.shape:
        raise ValueError("high/low/open 形状须一致")
    return (high - low) / (open_px.abs() + eps)


def upper_shadow_ratio(
    high: torch.Tensor,
    low: torch.Tensor,
    open_px: torch.Tensor,
    close: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    上影线比率：``(high - max(open, close)) / (high - low)``。

    值接近 1 表示上影线长、上方抛压大（冲高回落）；
    妖股常见上影线极长，是"洗盘"或"出货"信号。
    """
    if not (high.shape == low.shape == open_px.shape == close.shape):
        raise ValueError("high/low/open/close 形状须一致")
    body_top = torch.maximum(open_px, close)
    upper = high - body_top
    total = high - low
    upper = upper.clamp(min=0.0)
    ratio = upper / (total.abs() + eps)
    return ratio.clamp(0.0, 1.0)


def lower_shadow_ratio(
    high: torch.Tensor,
    low: torch.Tensor,
    open_px: torch.Tensor,
    close: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    下影线比率：``(min(open, close) - low) / (high - low)``。

    值接近 1 表示下影线长、下方支撑强（下探回升）；
    惯性上涨或主力护盘的股票下影线较长。
    """
    if not (high.shape == low.shape == open_px.shape == close.shape):
        raise ValueError("high/low/open/close 形状须一致")
    body_bot = torch.minimum(open_px, close)
    lower = body_bot - low
    total = high - low
    lower = lower.clamp(min=0.0)
    ratio = lower / (total.abs() + eps)
    return ratio.clamp(0.0, 1.0)


def close_open_return(
    close: torch.Tensor,
    open_px: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    日内涨跌幅：``close / open - 1``。

    正值：收盘强于开盘（尾盘力量强）；
    负值：开盘高走低（尾盘抛压）。
    """
    if close.shape != open_px.shape:
        raise ValueError("close/open 形状须一致")
    return close / (open_px.abs() + eps) - 1.0


def overnight_gap(
    open_px: torch.Tensor,
    close: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    隔夜跳空：``open(t) / close(t-1) - 1``；首列为 ``nan``。

    正值：隔夜高开（利好消息或博弈）；
    负值：隔夜低开（利空或夜盘期货下跌）。
    """
    if close.dim() != 2:
        raise ValueError("close 须为二维 (symbols, days)")
    out = torch.full_like(open_px, float("nan"))
    prev_close = close[:, :-1]
    cur_open = open_px[:, 1:]
    out[:, 1:] = cur_open / (prev_close.abs() + eps) - 1.0
    return out


def rolling_tail_strength(
    close: torch.Tensor,
    open_px: torch.Tensor,
    window: int,
) -> torch.Tensor:
    """
    尾盘强度：``close_open_return`` 在过去 ``window`` 天的滚动均值。

    持续正值表示近期尾盘持续上攻，买力旺盛；
    持续负值表示尾盘抛压，机构减仓迹象。
    """
    if window < 1:
        raise ValueError("window 须 >= 1")
    cor = close_open_return(close, open_px)
    return _rolling_nan_mean(cor, window)


def rolling_volume_price_trend(
    close: torch.Tensor,
    volume: torch.Tensor,
    window: int,
    *,
    log_volume: bool = True,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    量价趋势（滚动累积）：``sum(log(1+volume) * daily_ret)`` 在过去 ``window`` 天。

    正值：放量上涨；负值：放量下跌；量缩价跌接近 0。
    """
    if close.shape != volume.shape:
        raise ValueError("close/volume 形状须一致")
    if window < 1:
        raise ValueError("window 须 >= 1")
    rets = daily_returns_from_close(close)
    vx = torch.log(torch.clamp(volume, min=0.0) + 1.0) if log_volume else volume
    vpt = vx * rets
    out = torch.full_like(close, float("nan"))
    for t in range(window, close.shape[1]):
        sl = vpt[:, t - window + 1 : t + 1]
        fin = torch.isfinite(sl)
        sl0 = torch.where(fin, sl, torch.zeros_like(sl))
        out[:, t] = sl0.sum(dim=1)
    return out


def rolling_intraday_range_skew(
    high: torch.Tensor,
    low: torch.Tensor,
    open_px: torch.Tensor,
    window: int,
) -> torch.Tensor:
    """
    日内振幅偏度：过去 ``window`` 天内振幅序列的偏度。

    右偏（正偏度）表示偶发性大幅震荡，多为妖股特征；
    接近 0 表示每日振幅相对均匀，走势较为稳健。
    """
    if window < 3:
        raise ValueError("window 须 >= 3（偏度至少需 3 个点）")
    rng = intraday_range(high, low, open_px)
    out = torch.full_like(high, float("nan"))
    for t in range(window, high.shape[1]):
        sl = rng[:, t - window + 1 : t + 1]
        out[:, t] = _rowwise_skew(sl)
    return out


def compute_intraday_proxy_bundle(
    close: torch.Tensor,
    open_px: torch.Tensor,
    high: torch.Tensor,
    low: torch.Tensor,
    volume: Optional[torch.Tensor] = None,
    *,
    tail_window: int = 10,
    vpt_window: int = 20,
    range_skew_window: int = 20,
) -> dict:
    """
    一次性计算所有日线 K 线结构代理因子。

    Parameters
    ----------
    close, open_px, high, low
        必需，形状 ``(num_symbols, num_days)``。
    volume
        可选；传入后计算 ``volume_price_trend``。

    Returns
    -------
    dict，键：
    - ``intraday_range``：日内振幅
    - ``upper_shadow_ratio``：上影线比率
    - ``lower_shadow_ratio``：下影线比率
    - ``close_open_return``：当日尾盘 vs 开盘涨跌
    - ``overnight_gap``：隔夜跳空
    - ``tail_strength``：尾盘强度（滚动均值）
    - ``volume_price_trend``：量价趋势（需 volume）
    - ``intraday_range_skew``：振幅偏度
    """
    for name, t in [("close", close), ("open_px", open_px), ("high", high), ("low", low)]:
        if t.dim() != 2:
            raise ValueError(f"{name} 须为二维 (symbols, days)")
    if not (close.shape == open_px.shape == high.shape == low.shape):
        raise ValueError("close/open/high/low 形状须一致")

    out: dict = {
        "intraday_range": intraday_range(high, low, open_px),
        "upper_shadow_ratio": upper_shadow_ratio(high, low, open_px, close),
        "lower_shadow_ratio": lower_shadow_ratio(high, low, open_px, close),
        "close_open_return": close_open_return(close, open_px),
        "overnight_gap": overnight_gap(open_px, close),
        "tail_strength": rolling_tail_strength(close, open_px, tail_window),
        "intraday_range_skew": rolling_intraday_range_skew(high, low, open_px, range_skew_window),
    }
    if volume is not None:
        if volume.shape != close.shape:
            raise ValueError("volume 与 close 形状须一致")
        out["volume_price_trend"] = rolling_volume_price_trend(
            close, volume, vpt_window
        )
    else:
        out["volume_price_trend"] = None

    return out
