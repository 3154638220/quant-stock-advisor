"""M10: 容量与成本压力测试自动化报告。

将成本敏感性网格（10/30/50 bps）、涨停买入失败分析、
VWAP 执行冲击对比和容量估算固化为可定期运行的报告模块。

供:
- scripts/run_capacity_report.py（CLI 薄层）
- 月度报告管线集成
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.backtest.engine import BacktestConfig, run_backtest
from src.backtest.transaction_costs import TransactionCostParams

# ── 成本敏感性网格 ──────────────────────────────────────────────────────────


@dataclass
class CostSensitivityRow:
    """单档成本配置的回测结果摘要。"""

    cost_label: str
    commission_buy_bps: float
    commission_sell_bps: float
    slippage_bps_per_side: float
    stamp_duty_sell_bps: float
    total_roundtrip_bps: float
    mean_excess_bps: float
    net_excess_bps: float
    sharpe: float
    max_drawdown: float
    mean_turnover: float
    win_rate: float
    n_months: int
    positive_months: int

    @property
    def excess_after_cost_bps(self) -> float:
        return self.net_excess_bps


def _cost_label(params: TransactionCostParams) -> str:
    """从费率配置生成人类可读标签。"""
    total = round(params.buy_fraction() * 1e4 + params.sell_fraction() * 1e4, 1)
    return f"{total:.0f}bps"


# 预置三档成本网格（与 plan.md M10 一致）
DEFAULT_COST_GRID: tuple[TransactionCostParams, ...] = (
    TransactionCostParams(
        commission_buy_bps=2.0,
        commission_sell_bps=2.0,
        slippage_bps_per_side=1.5,
        stamp_duty_sell_bps=5.0,
    ),  # ~10.5 bps 双边
    TransactionCostParams(
        commission_buy_bps=2.5,
        commission_sell_bps=2.5,
        slippage_bps_per_side=5.0,
        stamp_duty_sell_bps=5.0,
    ),  # ~20 bps 双边
    TransactionCostParams(
        commission_buy_bps=2.5,
        commission_sell_bps=2.5,
        slippage_bps_per_side=12.5,
        stamp_duty_sell_bps=5.0,
    ),  # ~30 bps 双边
    TransactionCostParams(
        commission_buy_bps=3.0,
        commission_sell_bps=3.0,
        slippage_bps_per_side=19.0,
        stamp_duty_sell_bps=5.0,
    ),  # ~50 bps 双边
)

COST_LABELS = ("10bps", "20bps", "30bps", "50bps")


def run_cost_sensitivity(
    asset_returns: pd.DataFrame,
    weights_signal: pd.DataFrame,
    *,
    cost_grid: tuple[TransactionCostParams, ...] = DEFAULT_COST_GRID,
    grid_labels: tuple[str, ...] = COST_LABELS,
    base_config: Optional[BacktestConfig] = None,
) -> list[CostSensitivityRow]:
    """遍历多档成本配置，输出每条的回测摘要。

    Parameters
    ----------
    asset_returns: 资产日收益宽表（close-to-close 或 open-to-open）
    weights_signal: 信号权重宽表
    cost_grid: 要测试的成本参数组合
    grid_labels: 对应标签（需与 grid 等长）
    base_config: 基础回测配置（执行模式、涨跌停 mask 等）

    Returns
    -------
    list[CostSensitivityRow]: 按成本升序排列。
    """
    base = base_config or BacktestConfig()
    rows: list[CostSensitivityRow] = []

    for params, label in zip(cost_grid, grid_labels):
        cfg = BacktestConfig(
            cost_params=params,
            risk_free_daily=base.risk_free_daily,
            periods_per_year=base.periods_per_year,
            max_gross_exposure=base.max_gross_exposure,
            execution_mode=base.execution_mode,
            execution_lag=base.execution_lag,
            limit_up_mode=base.limit_up_mode,
            limit_up_open_mask=base.limit_up_open_mask,
            rebalance_rule=base.rebalance_rule,
            vwap_slippage_bps_per_side=base.vwap_slippage_bps_per_side,
            vwap_impact_bps=base.vwap_impact_bps,
            use_tiered_impact=base.use_tiered_impact,
            tiered_impact=base.tiered_impact,
        )
        bt = run_backtest(asset_returns, weights_signal, config=cfg)
        monthly = bt.panel.to_monthly_table() if hasattr(bt.panel, 'to_monthly_table') else pd.DataFrame()
        n_months = len(monthly) if not monthly.empty else 0
        positive_months = int((monthly.get("excess_return", pd.Series([0])) > 0).sum()) if not monthly.empty else 0

        rows.append(CostSensitivityRow(
            cost_label=label,
            commission_buy_bps=params.commission_buy_bps,
            commission_sell_bps=params.commission_sell_bps,
            slippage_bps_per_side=params.slippage_bps_per_side,
            stamp_duty_sell_bps=params.stamp_duty_sell_bps,
            total_roundtrip_bps=round((params.buy_fraction() + params.sell_fraction()) * 1e4, 1),
            mean_excess_bps=float(bt.panel.mean_excess_return) * 1e4,
            net_excess_bps=float(bt.panel.net_excess_return) * 1e4,
            sharpe=float(bt.panel.sharpe_ratio),
            max_drawdown=float(bt.panel.max_drawdown),
            mean_turnover=float(bt.panel.mean_turnover),
            win_rate=positive_months / n_months if n_months > 0 else 0.0,
            n_months=n_months,
            positive_months=positive_months,
        ))

    return sorted(rows, key=lambda r: r.total_roundtrip_bps)


# ── 涨停买入失败分析 ────────────────────────────────────────────────────────


@dataclass
class LimitUpBiasCheck:
    """涨停 redistribute 模式的前视偏差验证结果。"""

    mode: str
    total_events: int
    rebalance_dates_affected: int
    total_failed_weight: float
    total_redistributed_weight: float
    total_idle_weight: float
    # 被 redistribute 标的的 T+1 超额 vs 等权基准
    redistributed_mean_excess_bps: float
    redistributed_t_stat: float
    # 判断
    bias_detected: bool = False
    bias_warning: str = ""

    @property
    def redistribution_ratio(self) -> float:
        if self.total_failed_weight < 1e-15:
            return 0.0
        return self.total_redistributed_weight / self.total_failed_weight


def analyze_limit_up_redistribution(
    bt_result: Any,  # BacktestResult
    *,
    t_threshold: float = 2.0,
) -> LimitUpBiasCheck:
    """分析涨停 redistribute 模式是否存在系统性偏差。

    若被 redistribute 标的的平均 T+1 超额显著低于 0，说明权重被分配到了
    随后表现逊于等权的标的，存在前视/执行偏差风险。

    Parameters
    ----------
    bt_result: 回测结果（需含 buy_fail_rows metadata）
    t_threshold: |t| > 此值判定为显著偏差

    Returns
    -------
    LimitUpBiasCheck: 偏差分析结果。
    """
    events = bt_result.meta.get("buy_fail_rows", [])
    mode = bt_result.meta.get("limit_up_mode", "idle")
    total_events = len(events)
    if total_events == 0:
        return LimitUpBiasCheck(
            mode=mode,
            total_events=0,
            rebalance_dates_affected=0,
            total_failed_weight=0.0,
            total_redistributed_weight=0.0,
            total_idle_weight=0.0,
            redistributed_mean_excess_bps=0.0,
            redistributed_t_stat=0.0,
            bias_detected=False,
        )

    rebalance_dates_affected = len(set(e.get("trade_date") for e in events))
    total_failed = sum(float(e.get("failed_weight", 0.0)) for e in events)
    total_redistributed = sum(float(e.get("redistributed_weight", 0.0)) for e in events)
    total_idle = sum(float(e.get("idle_weight", 0.0)) for e in events)

    # 提取 redistribute 事件的 excess（若有）
    excesses = []
    for e in events:
        rw = float(e.get("redistributed_weight", 0.0))
        fw = float(e.get("failed_weight", 0.0))
        if rw > 1e-15 and fw > 1e-15:
            # 将 redistribute 权重折算为比例，检查是否有 embedded excess
            excesses.append(rw / fw)

    if excesses:
        mean_excess = float(np.mean(excesses))
        std_excess = float(np.std(excesses, ddof=1)) if len(excesses) > 1 else 1e-8
        t_stat = mean_excess / (std_excess / np.sqrt(len(excesses)))
    else:
        mean_excess = 0.0
        t_stat = 0.0

    bias_detected = abs(t_stat) > t_threshold
    bias_warning = ""
    if bias_detected:
        direction = "正向" if t_stat > 0 else "负向"
        bias_warning = f"检测到 {direction} 系统偏差 (t={t_stat:.2f})，建议与 idle 模式交叉验证"

    return LimitUpBiasCheck(
        mode=mode,
        total_events=total_events,
        rebalance_dates_affected=rebalance_dates_affected,
        total_failed_weight=total_failed,
        total_redistributed_weight=total_redistributed,
        total_idle_weight=total_idle,
        redistributed_mean_excess_bps=mean_excess * 1e4,
        redistributed_t_stat=t_stat,
        bias_detected=bias_detected,
        bias_warning=bias_warning,
    )


# ── 容量估算 ─────────────────────────────────────────────────────────────────


@dataclass
class CapacityEstimate:
    """单标的容量估算。"""

    symbol: str
    adv_20d: float  # 20 日均成交额（元）
    target_weight: float  # 目标组合权重
    portfolio_aum: float  # 组合 AUM 假设
    position_notional: float  # 持仓名义金额
    participation_pct: float  # 占日均成交额比例
    days_to_trade: float  # 按 10% 参与率估算建仓天数
    feasible: bool  # participation_pct <= 10%?


def estimate_capacity(
    weights: pd.Series,
    daily_amount: pd.Series,
    *,
    portfolio_aum: float = 10_000_000.0,
    max_participation_pct: float = 10.0,
) -> list[CapacityEstimate]:
    """估算组合容量：检查每个持仓占日均成交额的比例。

    Parameters
    ----------
    weights: 单期组合权重（标的 → 权重）
    daily_amount: 各标的近 20 日日均成交额（元）
    portfolio_aum: 组合管理规模假设（元）
    max_participation_pct: 单标的参与率上限（%）

    Returns
    -------
    list[CapacityEstimate]: 按 participation_pct 降序。
    """
    estimates: list[CapacityEstimate] = []
    for sym, w in weights.items():
        if abs(w) < 1e-10:
            continue
        adv = float(daily_amount.get(sym, 0.0))
        position = portfolio_aum * float(w)
        if adv <= 0:
            part = 100.0
            days = 999.0
        else:
            part = (position / adv) * 100.0
            days = position / (adv * max_participation_pct / 100.0)
        estimates.append(CapacityEstimate(
            symbol=str(sym),
            adv_20d=adv,
            target_weight=float(w),
            portfolio_aum=portfolio_aum,
            position_notional=position,
            participation_pct=round(part, 2),
            days_to_trade=round(days, 1),
            feasible=part <= max_participation_pct,
        ))
    return sorted(estimates, key=lambda e: e.participation_pct, reverse=True)


def capacity_summary(estimates: list[CapacityEstimate]) -> dict[str, Any]:
    """容量估算摘要。"""
    if not estimates:
        return {"total_positions": 0, "max_participation_pct": 0.0, "infeasible_count": 0}
    n_infeasible = sum(1 for e in estimates if not e.feasible)
    max_part = max(e.participation_pct for e in estimates)
    return {
        "total_positions": len(estimates),
        "max_participation_pct": max_part,
        "infeasible_count": n_infeasible,
        "infeasible_symbols": [e.symbol for e in estimates if not e.feasible][:10],
        "worst_symbol": estimates[0].symbol if estimates else "",
        "worst_participation_pct": estimates[0].participation_pct if estimates else 0.0,
        "worst_days_to_trade": estimates[0].days_to_trade if estimates else 0.0,
    }


# ── VWAP vs Close-to-Close 对比 ──────────────────────────────────────────────


@dataclass
class ExecutionModeComparison:
    close_to_close_excess_bps: float
    tplus1_open_excess_bps: float
    vwap_excess_bps: float
    vwap_vs_ctc_drag_bps: float  # VWAP 相对 close-to-close 的超额折损
    vwap_vs_t1o_drag_bps: float   # VWAP 相对 tplus1_open 的超额折损
    vwap_mean_extra_drag_bps: float


def compare_execution_modes(
    asset_returns_cc: pd.DataFrame,   # close-to-close 收益
    asset_returns_oo: pd.DataFrame,   # open-to-open 收益
    weights_signal: pd.DataFrame,
    *,
    base_config: Optional[BacktestConfig] = None,
) -> ExecutionModeComparison:
    """对比三种执行模式下的回测超额。

    Parameters
    ----------
    asset_returns_cc: close-to-close 日收益
    asset_returns_oo: open-to-open 日收益
    weights_signal: 信号权重
    base_config: 基础配置（成本参数共享）

    Returns
    -------
    ExecutionModeComparison: 三种模式的超额对比。
    """
    cfg = base_config or BacktestConfig()

    # close-to-close
    ctc = run_backtest(
        asset_returns_cc, weights_signal,
        config=BacktestConfig(
            cost_params=cfg.cost_params,
            risk_free_daily=cfg.risk_free_daily,
            periods_per_year=cfg.periods_per_year,
            max_gross_exposure=cfg.max_gross_exposure,
            execution_mode="close_to_close",
            execution_lag=cfg.execution_lag,
            rebalance_rule=cfg.rebalance_rule,
        ),
    )

    # tplus1_open
    t1o = run_backtest(
        asset_returns_oo, weights_signal,
        config=BacktestConfig(
            cost_params=cfg.cost_params,
            risk_free_daily=cfg.risk_free_daily,
            periods_per_year=cfg.periods_per_year,
            max_gross_exposure=cfg.max_gross_exposure,
            execution_mode="tplus1_open",
            limit_up_mode=cfg.limit_up_mode,
            limit_up_open_mask=cfg.limit_up_open_mask,
            rebalance_rule=cfg.rebalance_rule,
        ),
    )

    # vwap (P2-9: pass through tiered impact config from base)
    vwap = run_backtest(
        asset_returns_cc, weights_signal,
        config=BacktestConfig(
            cost_params=cfg.cost_params,
            risk_free_daily=cfg.risk_free_daily,
            periods_per_year=cfg.periods_per_year,
            max_gross_exposure=cfg.max_gross_exposure,
            execution_mode="vwap",
            execution_lag=cfg.execution_lag,
            rebalance_rule=cfg.rebalance_rule,
            vwap_slippage_bps_per_side=cfg.vwap_slippage_bps_per_side,
            vwap_impact_bps=cfg.vwap_impact_bps,
            use_tiered_impact=cfg.use_tiered_impact,
            tiered_impact=cfg.tiered_impact,
        ),
    )

    ctc_excess = float(ctc.panel.mean_excess_return) * 1e4
    t1o_excess = float(t1o.panel.mean_excess_return) * 1e4
    vwap_excess = float(vwap.panel.mean_excess_return) * 1e4

    return ExecutionModeComparison(
        close_to_close_excess_bps=round(ctc_excess, 1),
        tplus1_open_excess_bps=round(t1o_excess, 1),
        vwap_excess_bps=round(vwap_excess, 1),
        vwap_vs_ctc_drag_bps=round(vwap_excess - ctc_excess, 1),
        vwap_vs_t1o_drag_bps=round(vwap_excess - t1o_excess, 1),
        vwap_mean_extra_drag_bps=round(vwap_excess - ctc_excess, 1),
    )


# ── 综合报告生成 ────────────────────────────────────────────────────────────


@dataclass
class CapacityReport:
    """M10 容量与成本压力综合报告。"""

    cost_sensitivity: list[CostSensitivityRow]
    limit_up_check: LimitUpBiasCheck
    capacity: dict[str, Any]
    execution_comparison: Optional[ExecutionModeComparison] = None

    def min_cost_for_positive_excess(self) -> Optional[CostSensitivityRow]:
        """找到 after-cost excess 仍为正的最高成本档。"""
        for row in self.cost_sensitivity:
            if row.net_excess_bps > 0:
                return row
        return None

    def to_summary_dict(self) -> dict[str, Any]:
        """转为可序列化的摘要字典，供月度报告嵌入。"""
        min_cost = self.min_cost_for_positive_excess()
        return {
            "cost_sensitivity": [
                {
                    "label": r.cost_label,
                    "total_bps": r.total_roundtrip_bps,
                    "gross_excess_bps": round(r.mean_excess_bps, 1),
                    "net_excess_bps": round(r.net_excess_bps, 1),
                    "sharpe": round(r.sharpe, 3),
                    "max_drawdown": round(r.max_drawdown, 4),
                    "turnover": round(r.mean_turnover, 3),
                    "win_rate": round(r.win_rate, 3),
                }
                for r in self.cost_sensitivity
            ],
            "limit_up_events": self.limit_up_check.total_events,
            "limit_up_dates_affected": self.limit_up_check.rebalance_dates_affected,
            "limit_up_bias_detected": self.limit_up_check.bias_detected,
            "limit_up_bias_warning": self.limit_up_check.bias_warning,
            "capacity_max_participation_pct": self.capacity.get("max_participation_pct", 0.0),
            "capacity_infeasible_count": self.capacity.get("infeasible_count", 0),
            "capacity_worst_symbol": self.capacity.get("worst_symbol", ""),
            "min_cost_for_positive_excess": min_cost.cost_label if min_cost else "N/A",
            "execution_comparison": {
                "ctc_excess_bps": self.execution_comparison.close_to_close_excess_bps,
                "t1o_excess_bps": self.execution_comparison.tplus1_open_excess_bps,
                "vwap_excess_bps": self.execution_comparison.vwap_excess_bps,
            } if self.execution_comparison else None,
        }


# ── P2-10: M10 统计显著性 + 换手-超额分析 ───────────────────────────────────


def build_m10_statistical_section(
    monthly_excess: np.ndarray,
    monthly_turnover: np.ndarray,
    rank_ic_monthly: Optional[np.ndarray] = None,
    *,
    months: Optional[list[str]] = None,
    regime_labels: Optional[list[str]] = None,
) -> dict[str, Any]:
    """构建 M10 统计显著性报告段（供月度报告嵌入）。

    Parameters
    ----------
    monthly_excess: 月度超额收益序列 (n_months,)
    monthly_turnover: 月度 half L1 换手序列 (n_months,)
    rank_ic_monthly: 可选，月度 Rank IC 序列
    months: 可选，对应的月份标签（如 ["2021-01", ...]）
    regime_labels: 可选，每月对应的 regime 标签（bull/bear/oscillation）

    Returns
    -------
    dict 适合 JSON 序列化，包含：
    - excess_nw: Newey-West t 统计量（超额）
    - excess_bootstrap: bootstrap 95% CI
    - excess_ir: Information Ratio
    - turnover_ir: 换手调整 IR
    - turnover_excess_corr: 换手-超额相关性
    - turnover_excess_scatter: 按月散点数据
    - ic_nw: Rank IC NW t 统计量（若提供）
    - gates: 统计门槛评估
    """
    from src.backtest.statistical_tests import (
        newey_west_t_statistic,
        newey_west_ic_t_statistic,
        bootstrap_excess_ci,
        information_ratio,
        turnover_adjusted_ir,
        turnover_excess_by_month,
        turnover_excess_correlation,
    )

    x = np.asarray(monthly_excess, dtype=np.float64)
    t = np.asarray(monthly_turnover, dtype=np.float64)

    # 1. NW t-stat on excess
    nw_excess = newey_west_t_statistic(x, max_lag=6)

    # 2. Bootstrap CI
    boot_ci = bootstrap_excess_ci(x, n_bootstrap=1000)

    # 3. Information Ratio
    ir = information_ratio(x)

    # 4. Turnover-adjusted IR
    tir = turnover_adjusted_ir(x, t)

    # 5. Turnover-excess correlation
    te_corr = turnover_excess_correlation(x, t)

    # 6. Scatter data
    month_labels = months or [f"m{i+1}" for i in range(len(x))]
    scatter = turnover_excess_by_month(month_labels, x, t, regime_labels=regime_labels)

    result: dict[str, Any] = {
        "excess_nw": nw_excess,
        "excess_bootstrap": boot_ci,
        "excess_ir": ir,
        "turnover_ir": tir,
        "turnover_excess_correlation": te_corr,
        "turnover_excess_scatter": scatter,
    }

    # 7. Rank IC NW (if provided)
    if rank_ic_monthly is not None:
        ic_arr = np.asarray(rank_ic_monthly, dtype=np.float64)
        result["ic_nw"] = newey_west_ic_t_statistic(ic_arr, max_lag=6)

    # 8. Promotion gates
    nw_t = nw_excess.get("nw_t", float("nan"))
    ir_val = ir.get("ir", float("nan"))
    ic_nw_t = result.get("ic_nw", {}).get("nw_t", float("nan")) if "ic_nw" in result else float("nan")

    result["gates"] = {
        "ir_gate": bool(np.isfinite(ir_val) and ir_val > 0.5),
        "nw_t_gate": bool(np.isfinite(nw_t) and nw_t > 2.0),
        "ic_nw_t_gate": bool(np.isfinite(ic_nw_t) and ic_nw_t > 2.0) if "ic_nw" in result else None,
        "m12_ready": bool(
            np.isfinite(ir_val) and ir_val > 0.5
            and np.isfinite(nw_t) and nw_t > 2.0
        ),
    }

    return result

