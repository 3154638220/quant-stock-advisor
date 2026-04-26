"""轻量策略代理结果的频率感知聚合、年化与 signal diagnostic 汇总。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.backtest.performance_panel import compute_performance_panel


def infer_periods_per_year(rebalance_rule: str) -> float:
    rule = str(rebalance_rule).strip().upper()
    if not rule:
        return 252.0

    multiple = ""
    unit = rule
    while unit and unit[0].isdigit():
        multiple += unit[0]
        unit = unit[1:]
    step = max(int(multiple), 1) if multiple else 1

    if unit in {"M", "ME"}:
        return 12.0 / step
    if unit.startswith("W"):
        return 52.0 / step
    if unit in {"Q", "QE"}:
        return 4.0 / step
    return 252.0 / step


def annualize_period_return(ret: float, n_periods: int, periods_per_year: float) -> float:
    if not np.isfinite(ret) or n_periods <= 0 or ret <= -1.0:
        return float("nan")
    return float((1.0 + ret) ** (periods_per_year / float(n_periods)) - 1.0)


def build_light_proxy_period_detail(
    strategy_daily: pd.Series,
    benchmark_daily: pd.Series,
    *,
    rebalance_rule: str,
    scenario: str = "",
) -> pd.DataFrame:
    common = pd.DatetimeIndex(strategy_daily.index).intersection(pd.DatetimeIndex(benchmark_daily.index)).sort_values()
    if common.empty:
        return pd.DataFrame(
            columns=[
                "period",
                "strategy_return",
                "benchmark_return",
                "scenario",
                "excess_return",
                "benchmark_up",
                "strategy_up",
                "beat_benchmark",
            ]
        )
    strat = pd.to_numeric(strategy_daily.reindex(common), errors="coerce").fillna(0.0)
    bench = pd.to_numeric(benchmark_daily.reindex(common), errors="coerce").fillna(0.0)
    dates = pd.DatetimeIndex(common)

    rule = str(rebalance_rule).strip().upper()
    multiple = ""
    unit = rule
    while unit and unit[0].isdigit():
        multiple += unit[0]
        unit = unit[1:]
    step = max(int(multiple), 1) if multiple else 1

    if unit in {"M", "ME"}:
        base_labels = pd.Series(dates.to_period("M").astype(str), index=dates)
    elif unit.startswith("W"):
        weekly_freq = unit if "-" in unit else "W-FRI"
        base_labels = pd.Series(dates.to_period(weekly_freq).astype(str), index=dates)
    elif unit in {"Q", "QE"}:
        base_labels = pd.Series(dates.to_period("Q").astype(str), index=dates)
    else:
        base_labels = pd.Series(dates.strftime("%Y-%m-%d"), index=dates)

    ordered_labels = pd.Index(pd.unique(base_labels))
    label_to_group = {label: idx // step for idx, label in enumerate(ordered_labels)}
    group_id = base_labels.map(label_to_group)

    period_df = pd.DataFrame(
        {
            "group_id": group_id.to_numpy(),
            "base_period": base_labels.to_numpy(),
            "strategy_return": strat.to_numpy(),
            "benchmark_return": bench.to_numpy(),
        },
        index=dates,
    )
    aggregated = (
        period_df.groupby("group_id", sort=True)
        .agg(
            period=("base_period", "last"),
            strategy_return=("strategy_return", lambda s: float((1.0 + s).prod() - 1.0)),
            benchmark_return=("benchmark_return", lambda s: float((1.0 + s).prod() - 1.0)),
        )
        .reset_index(drop=True)
    )
    aggregated["scenario"] = scenario
    aggregated["excess_return"] = aggregated["strategy_return"] - aggregated["benchmark_return"]
    aggregated["benchmark_up"] = aggregated["benchmark_return"] > 0.0
    aggregated["strategy_up"] = aggregated["strategy_return"] > 0.0
    aggregated["beat_benchmark"] = aggregated["excess_return"] > 0.0
    return aggregated


def summarize_light_strategy_proxy(
    period_df: pd.DataFrame,
    *,
    periods_per_year: float,
) -> dict[str, float]:
    if period_df.empty:
        return {
            "strategy_annualized_return": float("nan"),
            "benchmark_annualized_return": float("nan"),
            "annualized_excess_vs_market": float("nan"),
            "n_periods": 0,
        }

    strat = pd.to_numeric(period_df["strategy_return"], errors="coerce")
    bench = pd.to_numeric(period_df["benchmark_return"], errors="coerce")
    excess = pd.to_numeric(period_df["excess_return"], errors="coerce")
    n_periods = int(excess.notna().sum())
    return {
        "strategy_annualized_return": annualize_period_return(
            float((1.0 + strat.fillna(0.0)).prod() - 1.0),
            int(strat.notna().sum()),
            periods_per_year,
        ),
        "benchmark_annualized_return": annualize_period_return(
            float((1.0 + bench.fillna(0.0)).prod() - 1.0),
            int(bench.notna().sum()),
            periods_per_year,
        ),
        "annualized_excess_vs_market": annualize_period_return(
            float((1.0 + excess.fillna(0.0)).prod() - 1.0),
            n_periods,
            periods_per_year,
        ),
        "n_periods": n_periods,
    }


def summarize_signal_diagnostic(
    period_df: pd.DataFrame,
    *,
    periods_per_year: float,
) -> dict[str, float]:
    """输出 signal diagnostic 层的 canonical 轻量诊断指标。"""
    if period_df.empty:
        return {
            "strategy_total_return": float("nan"),
            "benchmark_total_return": float("nan"),
            "excess_total_return": float("nan"),
            "strategy_annualized_return": float("nan"),
            "benchmark_annualized_return": float("nan"),
            "annualized_excess_vs_market": float("nan"),
            "strategy_annualized_vol": float("nan"),
            "benchmark_annualized_vol": float("nan"),
            "excess_annualized_vol": float("nan"),
            "strategy_sharpe_ratio": float("nan"),
            "benchmark_sharpe_ratio": float("nan"),
            "excess_sharpe_ratio": float("nan"),
            "strategy_max_drawdown": float("nan"),
            "benchmark_max_drawdown": float("nan"),
            "excess_max_drawdown": float("nan"),
            "period_win_rate": float("nan"),
            "period_beat_rate": float("nan"),
            "benchmark_up_rate": float("nan"),
            "n_periods": 0,
        }

    strat = pd.to_numeric(period_df["strategy_return"], errors="coerce").to_numpy(dtype=np.float64)
    bench = pd.to_numeric(period_df["benchmark_return"], errors="coerce").to_numpy(dtype=np.float64)
    excess = pd.to_numeric(period_df["excess_return"], errors="coerce").to_numpy(dtype=np.float64)

    strategy_panel = compute_performance_panel(strat, periods_per_year=periods_per_year)
    benchmark_panel = compute_performance_panel(bench, periods_per_year=periods_per_year)
    excess_panel = compute_performance_panel(excess, periods_per_year=periods_per_year)

    beat = pd.to_numeric(period_df["beat_benchmark"], errors="coerce")
    strategy_up = pd.to_numeric(period_df["strategy_up"], errors="coerce")
    benchmark_up = pd.to_numeric(period_df["benchmark_up"], errors="coerce")

    return {
        "strategy_total_return": float(strategy_panel.total_return),
        "benchmark_total_return": float(benchmark_panel.total_return),
        "excess_total_return": float(excess_panel.total_return),
        "strategy_annualized_return": float(strategy_panel.annualized_return),
        "benchmark_annualized_return": float(benchmark_panel.annualized_return),
        "annualized_excess_vs_market": float(excess_panel.annualized_return),
        "strategy_annualized_vol": float(np.nanstd(strat, ddof=1) * np.sqrt(periods_per_year))
        if np.isfinite(strat).sum() >= 2
        else float("nan"),
        "benchmark_annualized_vol": float(np.nanstd(bench, ddof=1) * np.sqrt(periods_per_year))
        if np.isfinite(bench).sum() >= 2
        else float("nan"),
        "excess_annualized_vol": float(np.nanstd(excess, ddof=1) * np.sqrt(periods_per_year))
        if np.isfinite(excess).sum() >= 2
        else float("nan"),
        "strategy_sharpe_ratio": float(strategy_panel.sharpe_ratio),
        "benchmark_sharpe_ratio": float(benchmark_panel.sharpe_ratio),
        "excess_sharpe_ratio": float(excess_panel.sharpe_ratio),
        "strategy_max_drawdown": float(strategy_panel.max_drawdown),
        "benchmark_max_drawdown": float(benchmark_panel.max_drawdown),
        "excess_max_drawdown": float(excess_panel.max_drawdown),
        "period_win_rate": float(strategy_up.mean()) if strategy_up.notna().any() else float("nan"),
        "period_beat_rate": float(beat.mean()) if beat.notna().any() else float("nan"),
        "benchmark_up_rate": float(benchmark_up.mean()) if benchmark_up.notna().any() else float("nan"),
        "n_periods": int(excess_panel.n_periods),
    }
