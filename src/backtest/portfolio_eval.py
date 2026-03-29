"""推荐表上的组合层前向收益：无成本 vs 有成本、换手与风险摘要。"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from src.backtest.risk_metrics import (
    drawdown_alert,
    index_cumulative_return,
    max_drawdown_from_returns,
    realized_volatility,
    risk_off_multiplier_from_index,
    volatility_alert,
)
from src.backtest.transaction_costs import (
    TransactionCostParams,
    net_simple_return_from_long_hold,
    turnover_cost_drag,
)
from src.portfolio.covariance import mean_cov_returns_from_daily_long
from src.portfolio.weights import (
    build_portfolio_weights,
    load_prev_weights_series,
    portfolio_config_from_mapping,
)


def weighted_portfolio_return(
    weights: np.ndarray,
    forward_returns: np.ndarray,
) -> float:
    """对有效前向收益行重归一化权重后求加权收益。"""
    w = np.asarray(weights, dtype=np.float64).ravel()
    r = np.asarray(forward_returns, dtype=np.float64).ravel()
    if w.size != r.size:
        raise ValueError("weights 与 forward_returns 长度须一致")
    m = np.isfinite(r) & np.isfinite(w) & (w >= 0)
    if not np.any(m):
        return float("nan")
    w2 = w[m]
    r2 = r[m]
    s = w2.sum()
    if s <= 0:
        return float("nan")
    w2 = w2 / s
    return float(np.dot(w2, r2))


def summarize_portfolio_eval(
    rec_with_fwd: pd.DataFrame,
    *,
    forward_col: str,
    daily_df: pd.DataFrame,
    portfolio_cfg: Dict[str, Any],
    cost_params: TransactionCostParams,
    risk_cfg: Dict[str, Any],
    asof: pd.Timestamp,
    prev_weights_path: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    在含 ``forward_col`` 的推荐表上追加 ``weight``，并返回汇总字典（无成本/有成本可对比）。

    Parameters
    ----------
    rec_with_fwd
        已含前向收益列。
    daily_df
        含基准指数在内的日线长表（用于极端行情与可选指数展示）。
    asof
        推荐基准日（与表内 asof_trade_date 一致）。
    prev_weights_path
        若给定，则读取并与当前权重算换手，在净收益上叠加换手摩擦近似。
    """
    from pathlib import Path

    pcfg = portfolio_config_from_mapping(portfolio_cfg)
    df = rec_with_fwd.copy()
    sym_col = "symbol" if "symbol" in df.columns else "代码"
    symbols = tuple(df[sym_col].astype(str).str.zfill(6).tolist())

    prev_aligned: Optional[np.ndarray] = None
    if prev_weights_path:
        prev_aligned = load_prev_weights_series(Path(prev_weights_path), symbols=symbols)

    wm = str(pcfg["weight_method"]).lower()
    cov_methods = ("risk_parity", "min_variance", "mean_variance")
    cov_mtx = None
    exp_ret = None
    if wm in cov_methods:
        shr = str(pcfg.get("cov_shrinkage", "ledoit_wolf")).lower()
        if shr not in ("ledoit_wolf", "sample"):
            shr = "ledoit_wolf"
        mu_arr, cov_mtx = mean_cov_returns_from_daily_long(
            daily_df,
            symbols,
            asof=pd.Timestamp(asof).normalize(),
            lookback_days=int(pcfg.get("cov_lookback_days", 60)),
            ridge=float(pcfg.get("cov_ridge", 1e-6)),
            shrinkage=shr,  # type: ignore[arg-type]
        )
        if wm == "mean_variance":
            exp_ret = mu_arr

    w = build_portfolio_weights(
        df,
        weight_method=pcfg["weight_method"],
        score_col=pcfg["score_col"],
        max_single_weight=pcfg["max_single_weight"],
        max_industry_weight=pcfg.get("max_industry_weight"),
        industry_col=pcfg.get("industry_col"),
        prev_weights_aligned=prev_aligned,
        max_turnover=pcfg["max_turnover"],
        cov_matrix=cov_mtx,
        expected_returns=exp_ret,
        risk_aversion=float(pcfg.get("risk_aversion", 1.0)),
    )
    df["weight"] = w

    r_mult, idx_ret, risk_note = risk_off_multiplier_from_index(
        daily_df,
        benchmark_symbol=str(risk_cfg.get("benchmark_symbol", "510300")),
        asof=pd.Timestamp(asof).normalize(),
        lookback_trading_days=int(risk_cfg.get("extreme_lookback_days", 5)),
        drop_threshold=float(risk_cfg.get("extreme_drop_threshold", 0.05)),
        risk_off_factor=float(risk_cfg.get("risk_off_factor", 0.0)),
    )
    w_eff = w * float(r_mult)
    s_eff = w_eff.sum()
    if s_eff > 0:
        w_eff = w_eff / s_eff
    else:
        w_eff = np.zeros_like(w_eff)

    fwd = pd.to_numeric(df[forward_col], errors="coerce").to_numpy(dtype=np.float64)
    if s_eff <= 1e-15:
        gross = 0.0
    else:
        gross = weighted_portfolio_return(w_eff, fwd)
    net_long_hold = (
        0.0 if not np.isfinite(gross) else net_simple_return_from_long_hold(gross, cost_params)
    )

    half_l1 = 0.0
    if prev_aligned is not None:
        half_l1 = 0.5 * float(np.sum(np.abs(w - prev_aligned)))
    turn_drag = turnover_cost_drag(half_l1, cost_params)

    # 截面波动率代理：成分前向收益的标准差（非时间序列组合波动）
    m = np.isfinite(fwd) & (w_eff > 0)
    cross_section_std = (
        float(np.std(fwd[m], ddof=1)) if np.sum(m) > 1 else float("nan")
    )

    # 若仅有单日截面，时间序列回撤/年化波动用单点占位；仍输出可计算接口
    ret_1d = np.array([gross], dtype=np.float64)
    mdd = max_drawdown_from_returns(ret_1d)
    vol_ann = realized_volatility(ret_1d)

    dd_warn, dd_msg = drawdown_alert(
        mdd,
        float(risk_cfg.get("max_drawdown_alert", 0.15)),
    )
    vol_warn, vol_msg = volatility_alert(
        vol_ann,
        float(risk_cfg.get("max_volatility_ann", 0.55)),
    )

    summary: Dict[str, Any] = {
        "portfolio_gross_ret": gross,
        "portfolio_net_ret_long_hold": net_long_hold,
        "turnover_cost_drag_approx": turn_drag,
        "cost_model": {
            "buy_fraction": cost_params.buy_fraction(),
            "sell_fraction": cost_params.sell_fraction(),
        },
        "risk_off_multiplier": r_mult,
        "risk_off_note": risk_note,
        "benchmark_cum_ret": idx_ret,
        "turnover_half_l1": half_l1,
        "cross_section_fwd_std": cross_section_std,
        "max_drawdown_sample": mdd,
        "vol_ann_sample": vol_ann,
        "drawdown_alert": dd_warn,
        "drawdown_alert_msg": dd_msg,
        "volatility_alert": vol_warn,
        "volatility_alert_msg": vol_msg,
        "index_ret_raw": index_cumulative_return(
            daily_df,
            symbol=str(risk_cfg.get("benchmark_symbol", "510300")),
            end_date=pd.Timestamp(asof).normalize(),
            lookback_trading_days=int(risk_cfg.get("extreme_lookback_days", 5)),
        ),
    }
    return df, summary
