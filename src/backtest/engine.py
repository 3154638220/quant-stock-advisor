"""标准回测接口：信号（权重）、调仓频率、成本模型、风险约束 → 日收益序列与统一绩效。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.backtest.performance_panel import PerformancePanel, compute_performance_panel
from src.backtest.risk_metrics import risk_config_from_mapping
from src.backtest.transaction_costs import TransactionCostParams, turnover_cost_drag


@dataclass
class BacktestConfig:
    """回测配置：成本、无风险利率、年化基准、风险约束（与现有 risk 配置兼容）。"""

    cost_params: Optional[TransactionCostParams] = None
    risk_free_daily: float = 0.0
    periods_per_year: float = 252.0
    # 若权重行和 > 1 或需限制总敞口，先缩放至 sum(w) <= max_gross_exposure（再归一化）
    max_gross_exposure: float = 1.0
    # 可选：与 risk_metrics.risk_config_from_mapping 同结构的极端行情等（预留扩展）
    risk_cfg: Dict[str, Any] = field(default_factory=dict)
    # ``close_to_close``：asset_returns 为当日收盘/昨收 -1；
    # ``tplus1_open``：须为 open(t+1)/open(t)-1（隔夜+日内至次日开盘）；
    # ``vwap``：与 close_to_close 同收益口径，但在调仓日额外扣减 VWAP 执行冲击。
    execution_mode: str = "close_to_close"
    # 仅 ``close_to_close`` / ``vwap``：组合收益用 ``w_{t-1-L}^T r_t``（``L`` 为本字段；0=历史默认）。
    execution_lag: int = 0
    # 涨停买入失败处理模式（仅 tplus1_open 模式下有效）：
    # ``idle``：资金闲置（涨停票的新增/增持权重冻结，已有持仓继续获得收益）
    # ``redistribute``：将涨停票的新增权重均匀分配给同日其他可买标的
    # P1-4: 两种模式均只冻结新增/增持权重，不冻结已有持仓收益。
    limit_up_mode: str = "idle"
    # 真实开盘涨停不可买 mask，索引为入场日，列为标的。
    # True 表示该日 open / pre_close 触及对应板块涨停；仅影响新增/增持权重。
    limit_up_open_mask: Optional[pd.DataFrame] = None
    # vwap 模式额外执行惩罚（按 half L1 换手比例缩放）：
    # extra_drag = turnover * (vwap_slippage_bps_per_side + vwap_impact_bps * turnover) / 1e4
    vwap_slippage_bps_per_side: float = 3.0
    vwap_impact_bps: float = 8.0


@dataclass
class BacktestResult:
    """引擎输出：日收益、换手、绩效面板。"""

    daily_returns: pd.Series
    rebalance_turnover: pd.Series  # 仅调仓日有值，其余 NaN
    panel: PerformancePanel
    meta: Dict[str, Any] = field(default_factory=dict)


def _align_weights_columns(
    weights: pd.DataFrame,
    asset_cols: List[str],
) -> pd.DataFrame:
    """将权重表列对齐到收益表列顺序；缺失列视为 0。"""
    w = weights.reindex(columns=asset_cols, fill_value=0.0)
    return w.astype(np.float64)


def build_limit_up_open_mask(
    daily_long: pd.DataFrame,
    *,
    date_col: str = "trade_date",
    sym_col: str = "symbol",
) -> pd.DataFrame:
    """构造入场日一字涨停不可买 mask，索引为交易日，列为标的。"""
    from src.market.tradability import is_open_limit_up_unbuyable

    if daily_long.empty:
        raise ValueError("daily_long 为空")
    need = {date_col, sym_col, "open", "close"}
    miss = need - set(daily_long.columns)
    if miss:
        raise ValueError(f"daily_long 缺少列: {miss}")

    df = daily_long.copy()
    df[sym_col] = df[sym_col].astype(str).str.zfill(6)
    df[date_col] = pd.to_datetime(df[date_col]).dt.normalize()
    chunks: List[pd.DataFrame] = []
    for sym, g in df.groupby(sym_col, sort=False):
        g = g.sort_values(date_col)
        open_px = pd.to_numeric(g["open"], errors="coerce")
        if "pre_close" in g.columns:
            prev_close = pd.to_numeric(g["pre_close"], errors="coerce")
        else:
            prev_close = pd.to_numeric(g["close"], errors="coerce").shift(1)
        mask = [
            is_open_limit_up_unbuyable(float(o), float(pc), str(sym))
            if np.isfinite(float(o)) and np.isfinite(float(pc))
            else False
            for o, pc in zip(open_px, prev_close)
        ]
        chunks.append(
            pd.DataFrame(
                {
                    date_col: g[date_col].values,
                    sym_col: sym,
                    "_limit_up_open": mask,
                }
            )
        )
    long_mask = pd.concat(chunks, ignore_index=True)
    wide = long_mask.pivot(index=date_col, columns=sym_col, values="_limit_up_open")
    wide = wide.sort_index().fillna(False)
    return wide.astype(bool)


def build_open_to_open_returns(
    daily_long: pd.DataFrame,
    *,
    date_col: str = "trade_date",
    sym_col: str = "symbol",
    zero_if_limit_up_open: bool = False,
) -> pd.DataFrame:
    """
    由日线长表构造「开盘到次日开盘」单日简单收益宽表，索引为交易日，与 ``run_backtest(..., execution_mode='tplus1_open')`` 输入一致。

    第 ``t`` 行、``s`` 列：``open(t+1,s)/open(t,s)-1``（最后一行全为 ``nan`` 或 0）。

    .. warning::

       **P1-4**: ``zero_if_limit_up_open=True`` 会将涨停日收益置 0，这混淆了
       「当日无法新买入」（买入约束）与「现有持仓在涨停日仍有收益」（持仓收益）。
       该参数仅适用于评估**新建仓**信号（无存量持仓的场景），**不应用于组合回测**。
       组合回测请使用 ``run_backtest(..., limit_up_mode='idle')`` 并在引擎层
       自动区分增量与存量权重。
    """
    if daily_long.empty:
        raise ValueError("daily_long 为空")
    need = {date_col, sym_col, "open", "close"}
    miss = need - set(daily_long.columns)
    if miss:
        raise ValueError(f"daily_long 缺少列: {miss}")

    df = daily_long.copy()
    df[sym_col] = df[sym_col].astype(str).str.zfill(6)
    df[date_col] = pd.to_datetime(df[date_col]).dt.normalize()
    chunks: List[pd.DataFrame] = []
    limit_mask = build_limit_up_open_mask(daily_long, date_col=date_col, sym_col=sym_col) if zero_if_limit_up_open else None
    for sym, g in df.groupby(sym_col, sort=False):
        g = g.sort_values(date_col)
        o = pd.to_numeric(g["open"], errors="coerce")
        o2o = o.shift(-1) / o - 1.0
        if limit_mask is not None and sym in limit_mask.columns:
            sym_mask = limit_mask.reindex(pd.to_datetime(g[date_col]).dt.normalize())[sym].fillna(False).to_numpy(bool)
            o2o = o2o.mask(sym_mask, 0.0)
        chunks.append(
            pd.DataFrame(
                {
                    date_col: g[date_col].values,
                    sym_col: sym,
                    "_o2o": o2o.values,
                }
            )
        )
    long_o2o = pd.concat(chunks, ignore_index=True)
    wide = long_o2o.pivot(index=date_col, columns=sym_col, values="_o2o")
    wide = wide.sort_index()
    return wide.astype(np.float64)


def _redistribute_limit_up_weights(
    w: np.ndarray,
    limit_up_mask: np.ndarray,
) -> np.ndarray:
    """
    将涨停票的权重均匀重分配给同日其余可买标的（顺延逻辑）。

    Parameters
    ----------
    w
        当日权重向量（归一化后），长度 k。
    limit_up_mask
        布尔数组，True 表示该票涨停买入失败。

    Returns
    -------
    w_new : ndarray，重分配后权重（仍归一化）
    """
    w = np.asarray(w, dtype=np.float64).copy()
    lim = np.asarray(limit_up_mask, dtype=bool)
    if not lim.any():
        return w
    stranded = float(np.sum(w[lim]))
    w[lim] = 0.0
    # 将滞留资金按现有权重比例重分配给非涨停票
    available = ~lim
    avail_sum = float(np.sum(w[available]))
    if avail_sum > 1e-15:
        w[available] += w[available] / avail_sum * stranded
    # 若所有票均涨停，资金闲置（全部置 0）
    return w


def _apply_limit_up_buy_fail(
    target_w: np.ndarray,
    prior_w: np.ndarray,
    limit_up_mask: np.ndarray,
    *,
    mode: str,
) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    """只冻结新增/增持部分；已有持仓继续承受当日收益。"""
    target = np.asarray(target_w, dtype=np.float64).copy()
    prior = np.asarray(prior_w, dtype=np.float64)
    lim = np.asarray(limit_up_mask, dtype=bool)
    buy_delta = np.maximum(target - prior, 0.0)
    failed_delta = np.where(lim, buy_delta, 0.0)
    failed_total = float(np.nansum(failed_delta))
    if failed_total <= 1e-15:
        return target, failed_delta, 0.0, 0.0, 0.0

    effective = target - failed_delta
    redistributed_total = 0.0
    if mode == "redistribute":
        available = ~lim
        avail_sum = float(np.nansum(effective[available]))
        if avail_sum > 1e-15:
            effective[available] += effective[available] / avail_sum * failed_total
            redistributed_total = failed_total
    idle_total = failed_total - redistributed_total
    return effective, failed_delta, failed_total, redistributed_total, idle_total


def _apply_gross_exposure(w: np.ndarray, max_gross: float) -> np.ndarray:
    """若 sum(w) > max_gross，整体缩放。"""
    s = float(np.nansum(np.maximum(w, 0.0)))
    if s <= 0:
        return np.zeros_like(w, dtype=np.float64)
    cap = float(max_gross)
    if cap > 0 and s > cap:
        return w * (cap / s)
    return w


def build_daily_weights(
    trading_index: pd.DatetimeIndex,
    weights_rebalance: pd.DataFrame,
    *,
    max_gross_exposure: float = 1.0,
) -> pd.DataFrame:
    """
    将调仓日权重前向填充到完整交易日索引；调仓日须为 ``trading_index`` 的子集。
    """
    if weights_rebalance.empty:
        raise ValueError("weights_rebalance 为空")
    wr = weights_rebalance.sort_index()
    wr.index = pd.to_datetime(wr.index).normalize()
    ti = pd.DatetimeIndex(pd.to_datetime(trading_index).normalize())
    # 仅保留交易日上存在的调仓日
    wr = wr[wr.index.isin(ti)]
    if wr.empty:
        raise ValueError("调仓日不在 trading_index 中")
    cols = wr.columns.tolist()
    out = pd.DataFrame(index=ti, columns=cols, dtype=np.float64)
    for i, dt in enumerate(ti):
        # 最后一个 <= dt 的调仓行
        sub = wr.loc[wr.index <= dt]
        if sub.empty:
            raise ValueError(f"日期 {dt} 之前无调仓权重")
        row = sub.iloc[-1].to_numpy(dtype=np.float64)
        row = _apply_gross_exposure(row, max_gross_exposure)
        s = row.sum()
        if s > 0:
            row = row / s
        out.iloc[i] = row
    return out


def run_backtest(
    asset_returns: pd.DataFrame,
    weights_signal: pd.DataFrame,
    *,
    config: Optional[BacktestConfig] = None,
    # 调仓频率：若给定，从 weights_signal 中按规则重采样（仅保留该频率的调仓日）
    rebalance_rule: Optional[str] = None,
) -> BacktestResult:
    """
    标准回测：输入资产日收益宽表 + 信号权重宽表（按调仓日行），输出日组合收益与绩效面板。

    Parameters
    ----------
    asset_returns
        索引为交易日，列为标的，值为**当日**简单收益。
    weights_signal
        索引为调仓日（须为 ``asset_returns.index`` 的子集），列为标的，非负权重（将归一化）。
        即「信号」已体现在权重上；若需由得分转权重，请在组合层先调用 ``build_portfolio_weights``。
    rebalance_rule
        若给定（如 ``"W-FRI"``），先将 ``weights_signal`` 按该规则对齐到 ``asset_returns`` 的日历后
        再前向填充；否则认为 ``weights_signal`` 的每一行即调仓日。

    Notes
    -----
    组合日收益：``r_{p,t} = w_{t-1}^T r_t``（前一日权重 × 当日资产收益），首日为 0。

    - **close_to_close**（默认）：``r_t`` 为当日收盘相对昨收的收益；基础约定为 ``w_{t-1}^T r_t``。若 ``execution_lag>0``，则改为 ``w_{t-1-L}^T r_t``（``L`` 为滞后天数），用于避免「收盘可得信号却按收盘成交」的不可实现假设。
    - **tplus1_open**：``r_t`` 须为 ``open(t+1)/open(t)-1``（见 ``build_open_to_open_returns``），与「T+1 最早次日开盘卖」一致，避免用日内 T+0 夸大夏普；收益矩阵末行可为 ``nan``（无下一开盘），将按 0 处理。此模式下忽略 ``execution_lag``。
    - **vwap**：``r_t`` 仍使用 close-to-close 收益，但在调仓日额外扣减与换手相关的 VWAP 执行冲击，降低尾盘成交过于乐观的偏差；可与 ``execution_lag`` 同用。

    成本：在**发生调仓**的交易日，按相对上一有效权重的 half L1 换手，用 ``turnover_cost_drag`` 从当日收益中扣减。
    默认参数（commission_buy=2.5bps, commission_sell=2.5bps, slippage=2.0bps/side, stamp_duty=5.0bps）合计约 14 bps 双边，
    如需对齐「双边千分之二（20bps）」的保守假设，可将 ``slippage_bps_per_side`` 调为 4~5 bps，
    或在配置中将 ``transaction_costs.slippage_bps_per_side`` 设为 4.5。

    涨停买入失败（仅 ``tplus1_open`` 模式）：
    - 若 ``BacktestConfig.limit_up_open_mask`` 提供真实 open/pre_close mask，只冻结新增/增持权重；
      已有持仓继续按 open-to-open 收益持有。
    - ``limit_up_mode="idle"``：冻结的新增资金闲置。
    - ``limit_up_mode="redistribute"``：将冻结的新增资金按可买标的当前权重比例重分配。
    """
    cfg = config or BacktestConfig()
    cost = cfg.cost_params
    exe = str(cfg.execution_mode).lower().strip()
    if exe not in ("close_to_close", "tplus1_open", "vwap"):
        raise ValueError("execution_mode 须为 close_to_close、tplus1_open 或 vwap")
    lag = int(cfg.execution_lag)
    if lag < 0:
        raise ValueError("execution_lag 须 >= 0")
    if exe == "tplus1_open":
        lag = 0
    ar = asset_returns.sort_index().copy()
    ar.index = pd.to_datetime(ar.index).normalize()
    sym_cols = [c for c in ar.columns]

    ws = weights_signal.sort_index().copy()
    ws.index = pd.to_datetime(ws.index).normalize()
    ws = _align_weights_columns(ws, sym_cols)

    if rebalance_rule is not None:
        # 将用户给定权重序列重采样到规则锚点（取该周期内最后一行）
        rs = ws.resample(rebalance_rule).last().dropna(how="all")
        rs = rs[rs.index.isin(ar.index)]
        ws = rs

    trading_index = ar.index
    daily_w = build_daily_weights(
        trading_index,
        ws,
        max_gross_exposure=cfg.max_gross_exposure,
    )

    r_mat = ar.to_numpy(dtype=np.float64)
    if exe == "tplus1_open":
        r_mat = np.where(np.isfinite(r_mat), r_mat, 0.0)
    w_mat = daily_w.to_numpy(dtype=np.float64)
    n, k = r_mat.shape
    if w_mat.shape != (n, k):
        raise ValueError("内部权重矩阵与收益矩阵形状不一致")

    limit_up_mode = str(cfg.limit_up_mode).lower().strip()
    if limit_up_mode not in ("idle", "redistribute"):
        limit_up_mode = "idle"
    limit_mask_mat: np.ndarray | None = None
    limit_detection = "disabled_no_mask"
    if exe == "tplus1_open" and cfg.limit_up_open_mask is not None:
        lm = cfg.limit_up_open_mask.copy()
        lm.index = pd.to_datetime(lm.index).normalize()
        lm = lm.reindex(index=trading_index, columns=sym_cols, fill_value=False).fillna(False)
        limit_mask_mat = lm.to_numpy(dtype=bool)
        limit_detection = "open_preclose_mask"

    port = np.zeros(n, dtype=np.float64)
    turn_series = np.full(n, np.nan, dtype=np.float64)
    rebalance_dates = ws.index.intersection(trading_index)
    buy_fail_rows: list[dict[str, Any]] = []

    # r_{p,t} = w_{t-1-L}^T r_t（L=execution_lag；tplus1_open 固定 L=0）
    port[0] = 0.0
    if exe == "tplus1_open" and limit_mask_mat is not None:
        actual_prev = np.zeros(k, dtype=np.float64)
        for i in range(1, n):
            jw = i - 1
            target_w = w_mat[jw].copy()
            r_today = r_mat[i]
            lim_mask = limit_mask_mat[i]
            w_effective, failed_delta, failed_total, redistributed_total, idle_total = _apply_limit_up_buy_fail(
                target_w,
                actual_prev,
                lim_mask,
                mode=limit_up_mode,
            )
            if failed_total > 1e-15:
                redistributed_ratio = redistributed_total / failed_total if failed_total > 1e-15 else 0.0
                for col_idx in np.flatnonzero(failed_delta > 1e-15):
                    failed_weight = float(failed_delta[col_idx])
                    redistributed_weight = failed_weight * redistributed_ratio
                    buy_fail_rows.append(
                        {
                            "trade_date": trading_index[i],
                            "signal_weight_date": trading_index[jw],
                            "symbol": sym_cols[col_idx],
                            "mode": limit_up_mode,
                            "target_weight": float(target_w[col_idx]),
                            "prior_weight": float(actual_prev[col_idx]),
                            "failed_weight": failed_weight,
                            "redistributed_weight": float(redistributed_weight),
                            "idle_weight": float(failed_weight - redistributed_weight),
                            "rebalance_failed_total_weight": float(failed_total),
                            "rebalance_redistributed_total_weight": float(redistributed_total),
                            "rebalance_idle_total_weight": float(idle_total),
                            "effective_weight": float(w_effective[col_idx]),
                        }
                    )

            port[i] = float(np.dot(w_effective, r_today))
            half_l1 = 0.5 * float(np.sum(np.abs(w_effective - actual_prev)))
            if half_l1 > 1e-15:
                turn_series[i] = half_l1
                if cost is not None:
                    port[i] -= turnover_cost_drag(half_l1, cost)
            actual_prev = w_effective
    else:
        for i in range(1, n):
            jw = i - 1 - lag
            if jw < 0:
                port[i] = 0.0
            else:
                w_prev = w_mat[jw].copy()
                r_today = r_mat[i]
                port[i] = float(np.dot(w_prev, r_today))

            w_new = w_mat[i]
            w_old = w_mat[i - 1]
            half_l1 = 0.5 * float(np.sum(np.abs(w_new - w_old)))
            if half_l1 > 1e-15:
                turn_series[i] = half_l1
                if cost is not None:
                    port[i] -= turnover_cost_drag(half_l1, cost)
                if exe == "vwap":
                    # VWAP 模式将尾盘成交冲击显式体现在调仓日：换手越大，冲击惩罚越高。
                    base_bps = max(float(cfg.vwap_slippage_bps_per_side), 0.0)
                    impact_bps = max(float(cfg.vwap_impact_bps), 0.0)
                    extra_drag = half_l1 * (base_bps + impact_bps * half_l1) / 1e4
                    port[i] -= float(extra_drag)

    s = pd.Series(port, index=trading_index, name="portfolio_ret")
    turn = pd.Series(turn_series, index=trading_index, name="turnover_half_l1")

    panel = compute_performance_panel(
        s.to_numpy(dtype=np.float64),
        turnover=turn.to_numpy(dtype=np.float64),
        risk_free_daily=cfg.risk_free_daily,
        periods_per_year=cfg.periods_per_year,
    )

    meta = {
        "n_rebalances": int(len(rebalance_dates)),
        "symbols": sym_cols,
        "risk_cfg_resolved": risk_config_from_mapping(cfg.risk_cfg),
        "execution_mode": exe,
        "execution_lag": int(lag),
        "limit_up_mode": limit_up_mode,
        "limit_up_detection": limit_detection,
        "buy_fail_diagnostic": buy_fail_rows,
        "buy_fail_event_count": int(len(buy_fail_rows)),
        "buy_fail_total_weight": float(sum(row["failed_weight"] for row in buy_fail_rows)),
        "buy_fail_redistributed_weight": float(sum(row["redistributed_weight"] for row in buy_fail_rows)),
        "buy_fail_idle_weight": float(sum(row["idle_weight"] for row in buy_fail_rows)),
    }
    return BacktestResult(daily_returns=s, rebalance_turnover=turn, panel=panel, meta=meta)


def result_to_dict(res: BacktestResult) -> Dict[str, Any]:
    """便于落盘：序列化摘要。"""
    return {
        "panel": res.panel.to_dict(),
        "meta": res.meta,
        "daily_returns": res.daily_returns,
        "rebalance_turnover": res.rebalance_turnover,
    }
