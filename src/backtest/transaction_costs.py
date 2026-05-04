"""A 股口径交易成本：佣金、滑点、卖出印花税（可配置），以及市场冲击成本模型。

P2-1: 支持固定 bps 和平方根律（sqrt_adv）两种市场冲击模型。
平方根律模型按股票粒度计算冲击成本，对小市值（20 日均量 < 5000 万）更准确。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional

import numpy as np


@dataclass(frozen=True)
class TransactionCostParams:
    """费率以「基点」计：1 bps = 0.01% = 1e-4。

    P2-1 新增：impact_model 和 impact_k 用于平方根律市场冲击模型。
    """

    commission_buy_bps: float = 2.5
    commission_sell_bps: float = 2.5
    slippage_bps_per_side: float = 2.0
    stamp_duty_sell_bps: float = 5.0
    # P2-1: 市场冲击模型
    impact_model: str = "fixed_bps"  # "fixed_bps" | "sqrt_adv"
    impact_k: float = 0.10  # 平方根律系数（仅 sqrt_adv 模式）

    def buy_fraction(self) -> float:
        """开仓一侧（买）总摩擦：佣金 + 滑点。"""
        return (self.commission_buy_bps + self.slippage_bps_per_side) * 1e-4

    def sell_fraction(self) -> float:
        """平仓一侧（卖）总摩擦：佣金 + 滑点 + 印花税（仅卖出）。"""
        return (
            self.commission_sell_bps
            + self.slippage_bps_per_side
            + self.stamp_duty_sell_bps
        ) * 1e-4


def transaction_cost_params_from_mapping(m: Mapping[str, Any]) -> TransactionCostParams:
    d = dict(m) if isinstance(m, dict) else {}
    return TransactionCostParams(
        commission_buy_bps=float(d.get("commission_buy_bps", 2.5)),
        commission_sell_bps=float(d.get("commission_sell_bps", 2.5)),
        slippage_bps_per_side=float(d.get("slippage_bps_per_side", 2.0)),
        stamp_duty_sell_bps=float(d.get("stamp_duty_sell_bps", 5.0)),
        # P2-1: 市场冲击模型配置
        impact_model=str(d.get("impact_model", "fixed_bps")),
        impact_k=float(d.get("impact_k", 0.10)),
    )


def net_simple_return_from_long_hold(
    gross_simple_return: float,
    costs: TransactionCostParams,
) -> float:
    """
    单次持仓期：期初按权重买入、期末全部平仓的近似净简单收益。

    采用 (1-f_buy) * (1 + gross) * (1-f_sell) - 1，与「无成本」的 gross 可直接对比。
    """
    fb = costs.buy_fraction()
    fs = costs.sell_fraction()
    g = float(gross_simple_return)
    return (1.0 - fb) * (1.0 + g) * (1.0 - fs) - 1.0


def turnover_cost_drag(
    turnover_half_l1: float,
    costs: TransactionCostParams,
) -> float:
    """
    仅对换手部分计双边摩擦（不含印花税时买侧/卖侧对称近似）。

    ``turnover_half_l1`` = 0.5 * sum(|w_new - w_old|)。用于增量换手成本近似。
    """
    t = max(0.0, min(1.0, float(turnover_half_l1)))
    # t 为单边换手量（买方或卖方各占 t），双边成本 = t * (buy_cost + sell_cost)
    per_unit = costs.buy_fraction() + costs.sell_fraction()
    return t * per_unit


def sqrt_adv_impact_bps(
    trade_value: float,
    adv: float,
    *,
    impact_k: float = 0.10,
) -> float:
    """P2-1: 平方根律市场冲击成本（基点）。

    基于 Almgren-Chriss 框架的简化：impact_bps = k * sqrt(participation) * 10000。

    Parameters
    ----------
    trade_value : 交易金额（元）
    adv : 日均成交额（元），通常为 20 日均值
    impact_k : 平方根律系数，默认 0.10

    Returns
    -------
    float : 冲击成本（基点）

    Notes
    -----
    - 参与率 10% 时冲击成本显著高于固定 bps
    - 参与率 0.1% 时冲击成本低于固定 bps
    - 对小市值股票（ADV < 5000 万）更准确
    """
    if adv <= 0 or not np.isfinite(adv) or trade_value <= 0:
        return 0.0
    participation = float(trade_value) / float(adv)
    if participation <= 0:
        return 0.0
    return float(float(impact_k) * np.sqrt(participation) * 10000.0)  # type: ignore[no-any-return]


def per_stock_impact_drag(
    weights: np.ndarray,
    prev_weights: np.ndarray,
    *,
    adv_vector: np.ndarray,
    portfolio_value: float = 1.0,
    costs: Optional[TransactionCostParams] = None,
) -> float:
    """P2-1: 按股票粒度计算平方根律冲击成本对组合收益的拖累。

    Parameters
    ----------
    weights : 新权重向量 (n,)
    prev_weights : 上一期权重向量 (n,)
    adv_vector : 各股票日均成交额 (n,)，单位元
    portfolio_value : 组合总价值（用于换算交易金额），默认 1.0（权重即金额比例）
    costs : 成本参数，若 impact_model="sqrt_adv" 则使用平方根律

    Returns
    -------
    float : 冲击成本占总组合价值的比例
    """
    w_new = np.asarray(weights, dtype=np.float64).ravel()
    w_old = np.asarray(prev_weights, dtype=np.float64).ravel()
    adv = np.asarray(adv_vector, dtype=np.float64).ravel()
    n = len(w_new)
    if n == 0 or len(w_old) != n or len(adv) != n:
        return 0.0

    if costs is None or costs.impact_model != "sqrt_adv":
        # 固定 bps 模式：使用原有 turnover_cost_drag
        half_l1 = 0.5 * float(np.sum(np.abs(w_new - w_old)))
        if costs is not None:
            return turnover_cost_drag(half_l1, costs)
        return 0.0

    # sqrt_adv 模式：按股票粒度计算
    total_drag = 0.0
    pv = max(float(portfolio_value), 1.0)
    for i in range(n):
        delta = abs(float(w_new[i] - w_old[i]))
        if delta < 1e-15:
            continue
        trade_value = delta * pv
        impact_bps = sqrt_adv_impact_bps(trade_value, float(adv[i]), impact_k=costs.impact_k)
        total_drag += delta * impact_bps / 10000.0  # bps → 比例

    return float(total_drag)


def cost_params_dict_for_logging(costs: TransactionCostParams) -> Dict[str, Any]:
    return {
        "buy_fraction": costs.buy_fraction(),
        "sell_fraction": costs.sell_fraction(),
        "commission_buy_bps": costs.commission_buy_bps,
        "commission_sell_bps": costs.commission_sell_bps,
        "slippage_bps_per_side": costs.slippage_bps_per_side,
        "stamp_duty_sell_bps": costs.stamp_duty_sell_bps,
        "impact_model": costs.impact_model,
        "impact_k": costs.impact_k,
    }
