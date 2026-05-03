"""A 股口径交易成本：佣金、滑点、卖出印花税（可配置）。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping


@dataclass(frozen=True)
class TransactionCostParams:
    """费率以「基点」计：1 bps = 0.01% = 1e-4。"""

    commission_buy_bps: float = 2.5
    commission_sell_bps: float = 2.5
    slippage_bps_per_side: float = 2.0
    stamp_duty_sell_bps: float = 5.0

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


def cost_params_dict_for_logging(costs: TransactionCostParams) -> Dict[str, float]:
    return {
        "buy_fraction": costs.buy_fraction(),
        "sell_fraction": costs.sell_fraction(),
        "commission_buy_bps": costs.commission_buy_bps,
        "commission_sell_bps": costs.commission_sell_bps,
        "slippage_bps_per_side": costs.slippage_bps_per_side,
        "stamp_duty_sell_bps": costs.stamp_duty_sell_bps,
    }
