"""
扩展基础因子（张量批量）：波动率、换手、量价相关、短期反转。

与 ``tensor_alpha.momentum_n`` 等一致，输入为 ``(num_symbols, num_days)``，
时间维为最后一维；缺失处以 ``nan`` 传播，由上游宽表保证对齐。
"""

from __future__ import annotations

from typing import Optional

import torch

from .tensor_alpha import momentum_n, weekly_kdj_from_daily


# 延迟导入避免循环（intraday_proxy_factors 依赖本模块辅助函数）
def _get_intraday_bundle():
    from .intraday_proxy_factors import compute_intraday_proxy_bundle
    return compute_intraday_proxy_bundle


def daily_returns_from_close(close: torch.Tensor) -> torch.Tensor:
    """日收益率：首列为 ``nan``。"""
    if close.dim() != 2:
        raise ValueError("close 须为二维 (symbols, days)")
    out = torch.full_like(close, float("nan"))
    out[:, 1:] = close[:, 1:] / close[:, :-1] - 1.0
    return out


def forward_returns_from_close(close: torch.Tensor, horizon: int) -> torch.Tensor:
    """
    前瞻收益：``close(t+h)/close(t)-1``，尾部 ``horizon`` 列为 ``nan``。
    """
    if close.dim() != 2:
        raise ValueError("close 须为二维 (symbols, days)")
    if horizon < 1:
        raise ValueError("horizon 须 >= 1")
    out = torch.full_like(close, float("nan"))
    out[:, :-horizon] = close[:, horizon:] / close[:, :-horizon] - 1.0
    return out


def forward_returns_tplus1_open(open_px: torch.Tensor, horizon: int) -> torch.Tensor:
    """
    A 股 T+1 可卖语义下的前瞻收益（与次日开盘买入对齐）。

    信号在交易日 ``t`` 收盘后产生，``t+1`` 开盘买入，持有 ``horizon`` 个开盘到开盘区间，
    收益为 ``open(t+1+horizon)/open(t+1)-1``。例如 ``horizon=1`` 对应 ``(T+2 开盘)/(T+1 开盘)-1``。

    尾部 ``horizon+1`` 列为 ``nan``（缺次日或缺终点开盘）。
    """
    if open_px.dim() != 2:
        raise ValueError("open_px 须为二维 (symbols, days)")
    if horizon < 1:
        raise ValueError("horizon 须 >= 1")
    n_day = open_px.shape[1]
    if n_day <= horizon + 1:
        return torch.full_like(open_px, float("nan"))
    out = torch.full_like(open_px, float("nan"))
    # 列 t 对应信号日 t：open(t+1+h)/open(t+1)-1，需 t+1+horizon < n_day
    k = n_day - horizon - 1
    out[:, :k] = open_px[:, 1 + horizon :] / open_px[:, 1 : n_day - horizon] - 1.0
    return out


def _nanstd_time_window(sl: torch.Tensor) -> torch.Tensor:
    """
    ``sl`` 形状 ``(N, W)``：逐行在有效值上计算总体标准差（与 ``torch.nanstd(..., correction=0)`` 一致）。

    部分 PyTorch 版本无 ``torch.nanstd``，此处用手工矩实现。
    """
    fin = torch.isfinite(sl)
    n = fin.sum(dim=1).to(dtype=sl.dtype).clamp(min=1.0)
    x0 = torch.where(fin, sl, torch.zeros_like(sl))
    mean = x0.sum(dim=1) / n
    x2 = torch.where(fin, sl * sl, torch.zeros_like(sl))
    ex2 = x2.sum(dim=1) / n
    var = ex2 - mean * mean
    return torch.sqrt(torch.clamp(var, min=0.0))


def _rolling_nan_std(x: torch.Tensor, window: int) -> torch.Tensor:
    """沿时间维滚动样本标准差（ddof=0）。"""
    out = torch.full_like(x, float("nan"))
    for t in range(window, x.shape[1]):
        sl = x[:, t - window + 1 : t + 1]
        out[:, t] = _nanstd_time_window(sl)
    return out


def rolling_realized_volatility(
    close: torch.Tensor,
    window: int,
    *,
    annualize: bool = False,
    trading_days_per_year: int = 252,
) -> torch.Tensor:
    """
    已实现波动率：滚动窗口内日收益标准差。

    Parameters
    ----------
    annualize
        若为 True，乘以 ``sqrt(trading_days_per_year)``。
    """
    rets = daily_returns_from_close(close)
    vol = _rolling_nan_std(rets, window)
    if annualize:
        vol = vol * (float(trading_days_per_year) ** 0.5)
    return vol


def _rowwise_skew(sl: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """``sl`` 形状 ``(N, W)``：逐行样本偏度（Pearson），有效样本数 < 3 为 ``nan``。"""
    fin = torch.isfinite(sl)
    n = fin.sum(dim=1).to(dtype=sl.dtype)
    x0 = torch.where(fin, sl, torch.zeros_like(sl))
    n_safe = torch.clamp(n, min=1.0)
    mean = x0.sum(dim=1) / n_safe
    xc = torch.where(fin, sl - mean.unsqueeze(1), torch.zeros_like(sl))
    var = (xc * xc).sum(dim=1) / n_safe
    std = torch.sqrt(torch.clamp(var, min=0.0))
    m3 = (xc * xc * xc).sum(dim=1) / n_safe
    sk = m3 / (std**3 + eps)
    ok = n >= 3.0
    return torch.where(
        ok,
        sk,
        torch.tensor(float("nan"), device=sl.device, dtype=sl.dtype),
    )


def rolling_log_volume_skew(volume: torch.Tensor, window: int) -> torch.Tensor:
    """
    窗口内 ``log(1+volume)`` 的截面式逐股样本偏度，刻画成交量分布不对称性。
    """
    if volume.dim() != 2:
        raise ValueError("volume 须为二维 (symbols, days)")
    if window < 3:
        raise ValueError("window 须 >= 3（偏度至少需 3 个点）")
    lv = torch.log(torch.clamp(volume, min=0.0) + 1.0)
    out = torch.full_like(volume, float("nan"))
    for t in range(window, volume.shape[1]):
        sl = lv[:, t - window + 1 : t + 1]
        out[:, t] = _rowwise_skew(sl)
    return out


def vol_to_turnover_ratio(
    realized_vol: torch.Tensor,
    turnover_roll_mean: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    量价交叉：高波动 / 换手（近似「高波动环境下的相对低换手」强度）。

    ``realized_vol`` 与 ``turnover_roll_mean`` 须同形；换手列单位与数据源一致，本函数不做换算。
    """
    if realized_vol.shape != turnover_roll_mean.shape:
        raise ValueError("realized_vol 与 turnover_roll_mean 形状须一致")
    return realized_vol / (turnover_roll_mean + eps)


def rolling_turnover_mean(turnover: torch.Tensor, window: int) -> torch.Tensor:
    """换手率滚动均值（AkShare 列为「换手率」百分数或小数依数据源而定，本函数不做单位变换）。"""
    if turnover.dim() != 2:
        raise ValueError("turnover 须为二维 (symbols, days)")
    if window < 1:
        raise ValueError("window 须 >= 1")
    out = torch.full_like(turnover, float("nan"))
    for t in range(window, turnover.shape[1]):
        sl = turnover[:, t - window + 1 : t + 1]
        out[:, t] = torch.nanmean(sl, dim=1)
    return out


def _rowwise_pearson_xy(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    eps: float = 1e-12,
) -> torch.Tensor:
    """``x,y`` 形状 ``(N, W)``，逐行 Pearson，全无效行为 ``nan``。"""
    if x.shape != y.shape:
        raise ValueError("x/y 形状须一致")
    m = torch.isfinite(x) & torch.isfinite(y)
    n = m.sum(dim=1).to(dtype=x.dtype)
    ok = n >= 2
    x0 = torch.where(m, x, torch.zeros_like(x))
    y0 = torch.where(m, y, torch.zeros_like(y))
    sum_x = x0.sum(dim=1)
    sum_y = y0.sum(dim=1)
    sum_x2 = (x0 * x0).sum(dim=1)
    sum_y2 = (y0 * y0).sum(dim=1)
    sum_xy = (x0 * y0).sum(dim=1)
    n_safe = torch.clamp(n, min=1.0)
    mean_x = sum_x / n_safe
    mean_y = sum_y / n_safe
    cov = sum_xy / n_safe - mean_x * mean_y
    var_x = sum_x2 / n_safe - mean_x * mean_x
    var_y = sum_y2 / n_safe - mean_y * mean_y
    pos = (var_x > eps) & (var_y > eps) & ok
    den = torch.sqrt(torch.clamp(var_x * var_y, min=0.0)) + eps
    r = cov / den
    r = torch.where(pos, r, torch.tensor(float("nan"), device=x.device, dtype=x.dtype))
    return r


def rolling_volume_return_corr(
    volume: torch.Tensor,
    close: torch.Tensor,
    window: int,
    *,
    log_volume: bool = True,
) -> torch.Tensor:
    """
    量价相关：窗口内成交量（可选 ``log(1+volume)``）与日收益的 Pearson 相关系数。

    常用于刻画「放量上涨/缩量」等同向性；取值 [-1, 1]。
    """
    if volume.shape != close.shape:
        raise ValueError("volume 与 close 形状须一致")
    rets = daily_returns_from_close(close)
    vx = torch.log(torch.clamp(volume, min=0.0) + 1.0) if log_volume else volume
    out = torch.full_like(close, float("nan"))
    for t in range(window, close.shape[1]):
        a = vx[:, t - window + 1 : t + 1]
        b = rets[:, t - window + 1 : t + 1]
        out[:, t] = _rowwise_pearson_xy(a, b)
    return out


def _rolling_nan_mean(x: torch.Tensor, window: int) -> torch.Tensor:
    """沿时间维滚动均值（忽略 nan）。"""
    out = torch.full_like(x, float("nan"))
    for t in range(window, x.shape[1]):
        sl = x[:, t - window + 1 : t + 1]
        out[:, t] = torch.nanmean(sl, dim=1)
    return out


def _rolling_nan_min(x: torch.Tensor, window: int) -> torch.Tensor:
    """沿时间维滚动最小值（nan 位置忽略，全 nan 则结果为 nan）。"""
    out = torch.full_like(x, float("nan"))
    for t in range(window, x.shape[1]):
        sl = x[:, t - window + 1 : t + 1]
        fin = torch.isfinite(sl)
        big = torch.tensor(float("inf"), device=x.device, dtype=x.dtype)
        masked = torch.where(fin, sl, big)
        vals = masked.min(dim=1).values
        any_valid = fin.any(dim=1)
        out[:, t] = torch.where(
            any_valid, vals, torch.tensor(float("nan"), device=x.device, dtype=x.dtype)
        )
    return out


def _rolling_nan_max(x: torch.Tensor, window: int) -> torch.Tensor:
    """沿时间维滚动最大值（nan 位置忽略，全 nan 则结果为 nan）。"""
    out = torch.full_like(x, float("nan"))
    for t in range(window, x.shape[1]):
        sl = x[:, t - window + 1 : t + 1]
        fin = torch.isfinite(sl)
        small = torch.tensor(float("-inf"), device=x.device, dtype=x.dtype)
        masked = torch.where(fin, sl, small)
        vals = masked.max(dim=1).values
        any_valid = fin.any(dim=1)
        out[:, t] = torch.where(
            any_valid, vals, torch.tensor(float("nan"), device=x.device, dtype=x.dtype)
        )
    return out


def bias_from_close(
    close: torch.Tensor,
    window: int,
    *,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    乖离率 BIAS：当前收盘价偏离 N 日均线的比例。

    ``BIAS = (close - MA_window) / MA_window``，高位股 BIAS 极高。
    """
    if close.dim() != 2:
        raise ValueError("close 须为二维 (symbols, days)")
    if window < 1:
        raise ValueError("window 须 >= 1")
    ma = _rolling_nan_mean(close, window)
    return (close - ma) / (ma.abs() + eps)


def max_single_day_drop(
    close: torch.Tensor,
    window: int,
) -> torch.Tensor:
    """
    过去 ``window`` 天内的单日最大跌幅（取 min(daily_return)）。

    妖股伴随剧烈震荡，该值绝对值通常较大（更负）。
    """
    if close.dim() != 2:
        raise ValueError("close 须为二维 (symbols, days)")
    if window < 2:
        raise ValueError("window 须 >= 2")
    rets = daily_returns_from_close(close)
    return _rolling_nan_min(rets, window)


def price_position_in_range(
    close: torch.Tensor,
    window: int,
) -> torch.Tensor:
    """
    当前价格在过去 ``window`` 天高低区间中的位置：``(close - min) / (max - min)``。

    值接近 1 表示处于近期绝对高位；接近 0 表示处于近期低位。
    """
    if close.dim() != 2:
        raise ValueError("close 须为二维 (symbols, days)")
    if window < 2:
        raise ValueError("window 须 >= 2")
    hi = _rolling_nan_max(close, window)
    lo = _rolling_nan_min(close, window)
    rng = hi - lo
    rng = torch.where(rng.abs() < 1e-12, torch.ones_like(rng), rng)
    return (close - lo) / rng


def log_float_market_cap_proxy(
    close: torch.Tensor,
    volume: torch.Tensor,
    turnover: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    对数流通市值代理：``log(close * volume / (turnover/100 + eps))``。

    换手率为百分数时 ``turnover/100`` 近似流通股数比例，
    ``close * volume / (turnover/100)`` 近似流通市值。妖股绝大多数是小微盘。
    """
    if close.shape != volume.shape or close.shape != turnover.shape:
        raise ValueError("close/volume/turnover 形状须一致")
    turnover_ratio = turnover / 100.0 + eps
    float_cap = close * volume / turnover_ratio
    return torch.log(torch.clamp(float_cap, min=1.0))


def limit_move_count(
    close: torch.Tensor,
    window: int,
    *,
    threshold: float = 0.09,
) -> torch.Tensor:
    """
    过去 ``window`` 天内涨停或跌停（|日收益| >= threshold）的次数。

    用于预过滤：妖股在短期内常有多次涨跌停。
    """
    if close.dim() != 2:
        raise ValueError("close 须为二维 (symbols, days)")
    rets = daily_returns_from_close(close)
    hit = (rets.abs() >= threshold).to(dtype=close.dtype)
    hit = torch.where(torch.isfinite(rets), hit, torch.zeros_like(hit))
    out = torch.full_like(close, float("nan"))
    for t in range(window, close.shape[1]):
        sl = hit[:, t - window + 1 : t + 1]
        out[:, t] = sl.sum(dim=1)
    return out


def short_term_reversal(close: torch.Tensor, window: int) -> torch.Tensor:
    """
    短期反转：``-N 日动量``，即过去 ``window`` 日收益取负。

    与动量因子相反：历史上表现为短期反转策略常用构造。
    """
    return -momentum_n(close, window)


def true_range(
    high: torch.Tensor,
    low: torch.Tensor,
    close: torch.Tensor,
) -> torch.Tensor:
    """
    真实波幅 TR：``max(H-L, |H-C_prev|, |L-C_prev|)``；首列为 ``nan``（无前收）。
    """
    if close.dim() != 2:
        raise ValueError("close 须为二维 (symbols, days)")
    if high.shape != close.shape or low.shape != close.shape:
        raise ValueError("high/low/close 形状须一致")
    tr = torch.full_like(close, float("nan"))
    prev = close[:, :-1]
    h = high[:, 1:]
    lo = low[:, 1:]
    hl = h - lo
    hc = (h - prev).abs()
    lc = (lo - prev).abs()
    tr[:, 1:] = torch.maximum(torch.maximum(hl, hc), lc)
    return tr


def atr_wilder(
    high: torch.Tensor,
    low: torch.Tensor,
    close: torch.Tensor,
    period: int = 14,
) -> torch.Tensor:
    """
    Wilder 平滑 ATR：首段 TR 简单均值初始化，其后 ``ATR_t = (ATR_{t-1}*(p-1)+TR_t)/p``。

    第 ``period`` 列（0-based）为首个有效 ATR（与常见实现一致）。
    """
    if period < 2:
        raise ValueError("period 须 >= 2")
    tr = true_range(high, low, close)
    atr = torch.full_like(close, float("nan"))
    atr[:, period] = torch.nanmean(tr[:, 1 : period + 1], dim=1)
    for j in range(period + 1, close.shape[1]):
        atr[:, j] = (atr[:, j - 1] * (period - 1) + tr[:, j]) / period
    return atr


def compute_base_factor_bundle(
    close: torch.Tensor,
    volume: Optional[torch.Tensor] = None,
    turnover: Optional[torch.Tensor] = None,
    high: Optional[torch.Tensor] = None,
    low: Optional[torch.Tensor] = None,
    open_px: Optional[torch.Tensor] = None,
    trade_dates: Optional[list] = None,
    *,
    vol_window: int = 20,
    turnover_window: int = 20,
    vp_corr_window: int = 20,
    reversal_window: int = 5,
    atr_period: int = 14,
    annualize_vol: bool = False,
    bias_window_short: int = 20,
    bias_window_long: int = 60,
    max_drop_window: int = 20,
    recent_return_window: int = 3,
    price_position_window: int = 250,
    limit_move_window: int = 5,
    limit_move_threshold: float = 0.09,
    # K 线结构高频降频因子窗口（需传入 open_px、high、low）
    tail_window: int = 10,
    vpt_window: int = 20,
    range_skew_window: int = 20,
    include_intraday: bool = True,
) -> dict:
    """
    一次性计算多类基础因子（张量字典，键与业务含义见下）。

    若未传入 ``volume`` / ``turnover`` / ``high``+``low``，对应键值为 ``None``。

    基础因子：
    - ``bias_short`` / ``bias_long``：乖离率（偏离均线距离，高位股极高）
    - ``max_single_day_drop``：过去 N 天内最大单日跌幅（识别筹码松动）
    - ``recent_return``：近 3 日涨幅（短期暴涨惩罚项）
    - ``price_position``：当前价格在过去 1 年区间中的位置（绝对高位过滤）
    - ``log_market_cap``：对数流通市值代理（小微盘过滤）
    - ``limit_move_count``：过去 5 天涨跌停次数（妖股识别）
    - ``weekly_kdj_k`` / ``weekly_kdj_d`` / ``weekly_kdj_j``：最近已完成周线 KDJ
    - ``weekly_kdj_oversold``：周线 ``J <= -5`` 触发
    - ``weekly_kdj_oversold_depth``：周线超卖深度 ``max(0, -5-J)``
    - ``weekly_kdj_rebound``：周线 J 脱离极端超卖后的回升确认

    K 线结构高频降频代理因子（需传入 ``open_px``、``high``、``low``）：
    - ``intraday_range``：日内振幅 (high-low)/open
    - ``upper_shadow_ratio``：上影线比率（上方抛压）
    - ``lower_shadow_ratio``：下影线比率（下方支撑）
    - ``close_open_return``：日内涨跌（尾盘相对开盘）
    - ``overnight_gap``：隔夜跳空
    - ``tail_strength``：尾盘强度滚动均值
    - ``volume_price_trend``：量价趋势（需 volume）
    - ``intraday_range_skew``：振幅偏度
    """
    out: dict = {
        "realized_vol": rolling_realized_volatility(
            close, vol_window, annualize=annualize_vol
        ),
        "short_reversal": short_term_reversal(close, reversal_window),
    }
    if high is not None and low is not None:
        if high.shape != close.shape or low.shape != close.shape:
            raise ValueError("high/low 与 close 形状须一致")
        out["atr"] = atr_wilder(high, low, close, atr_period)
    else:
        out["atr"] = None
    if turnover is not None:
        out["turnover_roll_mean"] = rolling_turnover_mean(turnover, turnover_window)
    else:
        out["turnover_roll_mean"] = None
    if volume is not None:
        out["vol_ret_corr"] = rolling_volume_return_corr(
            volume, close, vp_corr_window
        )
        wskew = max(3, int(vol_window))
        out["volume_skew_log"] = rolling_log_volume_skew(volume, wskew)
    else:
        out["vol_ret_corr"] = None
        out["volume_skew_log"] = None
    rv = out.get("realized_vol")
    tom = out.get("turnover_roll_mean")
    if rv is not None and tom is not None:
        out["vol_to_turnover"] = vol_to_turnover_ratio(rv, tom)
    else:
        out["vol_to_turnover"] = None

    out["bias_short"] = bias_from_close(close, bias_window_short)
    out["bias_long"] = bias_from_close(close, bias_window_long)
    out["max_single_day_drop"] = max_single_day_drop(close, max_drop_window)
    out["recent_return"] = momentum_n(close, recent_return_window)
    out["price_position"] = price_position_in_range(close, price_position_window)
    out["limit_move_count"] = limit_move_count(
        close, limit_move_window, threshold=limit_move_threshold
    )
    weekly_kdj_keys = (
        "weekly_kdj_k",
        "weekly_kdj_d",
        "weekly_kdj_j",
        "weekly_kdj_oversold",
        "weekly_kdj_oversold_depth",
        "weekly_kdj_rebound",
    )
    if high is not None and low is not None and trade_dates is not None:
        wk_k, wk_d, wk_j = weekly_kdj_from_daily(
            close,
            high,
            low,
            trade_dates=trade_dates,
            device=close.device,
            dtype=close.dtype,
        )
        out["weekly_kdj_k"] = wk_k
        out["weekly_kdj_d"] = wk_d
        out["weekly_kdj_j"] = wk_j
        oversold = torch.where(
            torch.isfinite(wk_j),
            (wk_j <= -5.0).to(dtype=close.dtype),
            torch.full_like(wk_j, float("nan")),
        )
        out["weekly_kdj_oversold"] = oversold
        depth = torch.where(
            torch.isfinite(wk_j),
            torch.clamp(-5.0 - wk_j, min=0.0),
            torch.full_like(wk_j, float("nan")),
        )
        out["weekly_kdj_oversold_depth"] = depth
        rebound = torch.full_like(wk_j, float("nan"))
        prev_j = wk_j[:, :-1]
        cur_j = wk_j[:, 1:]
        valid_pair = torch.isfinite(prev_j) & torch.isfinite(cur_j)
        rebound[:, 1:] = torch.where(
            valid_pair,
            ((prev_j <= -5.0) & (cur_j > prev_j)).to(dtype=close.dtype),
            torch.full_like(cur_j, float("nan")),
        )
        out["weekly_kdj_rebound"] = rebound
    else:
        for k in weekly_kdj_keys:
            out[k] = None
    if volume is not None and turnover is not None:
        out["log_market_cap"] = log_float_market_cap_proxy(close, volume, turnover)
    else:
        out["log_market_cap"] = None

    # K 线结构高频降频代理因子（需 open_px、high、low 全部传入）
    intraday_keys = (
        "intraday_range",
        "upper_shadow_ratio",
        "lower_shadow_ratio",
        "close_open_return",
        "overnight_gap",
        "tail_strength",
        "volume_price_trend",
        "intraday_range_skew",
    )
    if include_intraday and open_px is not None and high is not None and low is not None:
        _compute_intraday = _get_intraday_bundle()
        intraday = _compute_intraday(
            close,
            open_px,
            high,
            low,
            volume=volume,
            tail_window=tail_window,
            vpt_window=vpt_window,
            range_skew_window=range_skew_window,
        )
        for k in intraday_keys:
            out[k] = intraday.get(k)
    else:
        for k in intraday_keys:
            out[k] = None

    return out
