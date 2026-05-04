#!/usr/bin/env python3
"""Build the promoted monthly-selection benchmark suite.

The monthly research scripts already produce ``monthly_long.csv`` with the
strategy return, the all-market equal-weight label benchmark, and the selected
candidate-pool mean return.  This script turns those series into a compact
benchmark comparison and optionally joins investable/index benchmarks supplied
as daily OHLC CSV files.
"""

from __future__ import annotations

import argparse
import shlex
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.research_identity import make_research_identity, slugify_token
from src.models.experiment import append_experiment_result
from src.models.research_contract import (
    ArtifactRef,
    DataSlice,
    ExperimentResult,
    build_result_id,
    utc_now_iso,
    write_research_manifest,
)
from src.reporting.markdown_report import format_markdown_table as _format_markdown_table

DEFAULT_MODEL = "M8_regime_aware_fixed_policy__indcap3"
DEFAULT_POOL = "U1_liquid_tradable"
DEFAULT_TOP_K = 20


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    path: Path
    symbol: str = ""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="生成月度选股多基准对比报告")
    p.add_argument("--monthly-long", type=str, default="data/results/monthly_selection_m8_concentration_regime_2026-05-01_monthly_long.csv")
    p.add_argument("--model", type=str, default=DEFAULT_MODEL)
    p.add_argument("--candidate-pool", type=str, default=DEFAULT_POOL)
    p.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    p.add_argument("--as-of-date", type=str, default="")
    p.add_argument("--results-dir", type=str, default="data/results")
    p.add_argument("--output-prefix", type=str, default="monthly_selection_benchmark_suite")
    p.add_argument(
        "--index-csv",
        action="append",
        default=[],
        help=(
            "可选指数日线 CSV，格式 name=path 或 name:symbol=path。"
            "CSV 需包含 trade_date/date 与 open；若含多 symbol 可用 name:symbol=path 过滤。"
        ),
    )
    # P1-1: M10 压力测试参数
    p.add_argument(
        "--db-path",
        type=str,
        default="",
        help="DuckDB 路径（用于容量分析，查询日线成交额）。",
    )
    p.add_argument(
        "--daily-table",
        type=str,
        default="a_share_daily",
        help="日线数据表名（用于容量分析）。",
    )
    p.add_argument(
        "--stress-monthly-long",
        type=str,
        default="",
        help="对比用 monthly_long CSV（如 limit_up=redistribute 或 vwap 模式输出）。",
    )
    p.add_argument(
        "--stress-label",
        type=str,
        default="stress",
        help="对比用 monthly_long 的标签。",
    )
    # P0-1: 多成本档位真实回测输入（替代线性缩放）
    p.add_argument(
        "--multi-cost-monthly-long",
        action="append",
        default=[],
        help=(
            "多成本档 monthly_long，格式 cost_bps=path。"
            "例如 --multi-cost-monthly-long 30=data/results/m8_natural_30bps_monthly_long.csv"
            " --multi-cost-monthly-long 50=data/results/m8_natural_50bps_monthly_long.csv"
        ),
    )
    return p.parse_args()


def _resolve_project_path(raw: str | Path) -> Path:
    p = Path(raw)
    return p if p.is_absolute() else ROOT / p


def _project_relative(path: str | Path) -> str:
    p = Path(path).resolve()
    try:
        return str(p.relative_to(ROOT.resolve()))
    except ValueError:
        return str(p)


def _to_float_series(values: Any) -> pd.Series:
    return pd.to_numeric(pd.Series(values), errors="coerce")


def compounded_return(values: pd.Series) -> float:
    x = pd.to_numeric(values, errors="coerce").dropna()
    if x.empty:
        return np.nan
    return float((1.0 + x).prod() - 1.0)


def annualized_return(values: pd.Series, periods_per_year: int = 12) -> float:
    x = pd.to_numeric(values, errors="coerce").dropna()
    if x.empty:
        return np.nan
    total = float((1.0 + x).prod())
    if total <= 0:
        return np.nan
    return float(total ** (periods_per_year / len(x)) - 1.0)


def max_drawdown(values: pd.Series) -> float:
    x = pd.to_numeric(values, errors="coerce").fillna(0.0)
    if x.empty:
        return np.nan
    nav = (1.0 + x).cumprod()
    peak = nav.cummax()
    dd = nav / peak - 1.0
    return float(dd.min())


def sharpe_monthly(values: pd.Series) -> float:
    x = pd.to_numeric(values, errors="coerce").dropna()
    if len(x) < 2:
        return np.nan
    std = float(x.std(ddof=1))
    if not np.isfinite(std) or std <= 1e-12:
        return np.nan
    return float(x.mean() / std * np.sqrt(12.0))


def information_ratio(strategy: pd.Series, benchmark: pd.Series) -> float:
    aligned = pd.concat([pd.to_numeric(strategy, errors="coerce"), pd.to_numeric(benchmark, errors="coerce")], axis=1).dropna()
    if len(aligned) < 2:
        return np.nan
    excess = aligned.iloc[:, 0] - aligned.iloc[:, 1]
    std = float(excess.std(ddof=1))
    if not np.isfinite(std) or std <= 1e-12:
        return np.nan
    return float(excess.mean() / std * np.sqrt(12.0))


def summarize_series(name: str, returns: pd.Series, *, role: str, primary: bool = False) -> dict[str, Any]:
    x = pd.to_numeric(returns, errors="coerce")
    return {
        "benchmark": name,
        "role": role,
        "primary": bool(primary),
        "months": int(x.notna().sum()),
        "total_return": compounded_return(x),
        "annualized_return": annualized_return(x),
        "mean_monthly_return": float(x.mean()) if x.notna().any() else np.nan,
        "median_monthly_return": float(x.median()) if x.notna().any() else np.nan,
        "monthly_positive_rate": float((x.dropna() > 0).mean()) if x.notna().any() else np.nan,
        "max_drawdown": max_drawdown(x),
        "sharpe": sharpe_monthly(x),
    }


def summarize_relative(strategy_name: str, strategy: pd.Series, benchmark_name: str, benchmark: pd.Series) -> dict[str, Any]:
    aligned = pd.concat(
        [
            pd.to_numeric(strategy, errors="coerce").rename("strategy"),
            pd.to_numeric(benchmark, errors="coerce").rename("benchmark"),
        ],
        axis=1,
    ).dropna()
    if aligned.empty:
        return {
            "strategy": strategy_name,
            "benchmark": benchmark_name,
            "months": 0,
            "excess_total_return": np.nan,
            "annualized_excess_arithmetic": np.nan,
            "mean_monthly_excess": np.nan,
            "median_monthly_excess": np.nan,
            "monthly_hit_rate": np.nan,
            "win_months": 0,
            "information_ratio": np.nan,
        }
    strat_ann = annualized_return(aligned["strategy"])
    bench_ann = annualized_return(aligned["benchmark"])
    excess = aligned["strategy"] - aligned["benchmark"]
    return {
        "strategy": strategy_name,
        "benchmark": benchmark_name,
        "months": int(len(aligned)),
        "excess_total_return": compounded_return(aligned["strategy"]) - compounded_return(aligned["benchmark"]),
        "annualized_excess_arithmetic": strat_ann - bench_ann if np.isfinite(strat_ann) and np.isfinite(bench_ann) else np.nan,
        "mean_monthly_excess": float(excess.mean()),
        "median_monthly_excess": float(excess.median()),
        "monthly_hit_rate": float((excess > 0).mean()),
        "win_months": int((excess > 0).sum()),
        "information_ratio": information_ratio(aligned["strategy"], aligned["benchmark"]),
    }


def parse_index_specs(items: list[str]) -> list[BenchmarkSpec]:
    specs: list[BenchmarkSpec] = []
    for item in items:
        if "=" not in item:
            raise ValueError(f"--index-csv 需要 name=path 或 name:symbol=path: {item}")
        left, raw_path = item.split("=", 1)
        if ":" in left:
            name, symbol = left.split(":", 1)
        else:
            name, symbol = left, ""
        name = name.strip()
        symbol = symbol.strip()
        if not name:
            raise ValueError(f"--index-csv name 不能为空: {item}")
        specs.append(BenchmarkSpec(name=name, symbol=symbol, path=_resolve_project_path(raw_path.strip())))
    return specs


def load_index_csv_monthly_returns(spec: BenchmarkSpec, schedule: pd.DataFrame) -> tuple[pd.Series, dict[str, Any]]:
    meta: dict[str, Any] = {
        "benchmark": spec.name,
        "source": str(spec.path.relative_to(ROOT)) if spec.path.is_relative_to(ROOT) else str(spec.path),
        "symbol_filter": spec.symbol,
        "status": "missing",
        "covered_months": 0,
    }
    if not spec.path.exists():
        return pd.Series(np.nan, index=schedule.index, name=spec.name), meta

    df = pd.read_csv(spec.path, dtype={"symbol": str, "code": str, "ts_code": str})
    date_col = next((c for c in ["trade_date", "date", "datetime"] if c in df.columns), None)
    if date_col is None or "open" not in df.columns:
        meta["status"] = "invalid_schema"
        meta["columns"] = list(df.columns)
        return pd.Series(np.nan, index=schedule.index, name=spec.name), meta
    if spec.symbol:
        symbol_col = next((c for c in ["symbol", "code", "ts_code"] if c in df.columns), None)
        if symbol_col is not None:
            df = df[df[symbol_col].astype(str).str.replace(r"\D", "", regex=True).str.zfill(6).eq(spec.symbol.zfill(6))]
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce").dt.normalize()
    df["open"] = pd.to_numeric(df["open"], errors="coerce")
    px = df.dropna(subset=[date_col, "open"]).drop_duplicates(date_col, keep="last").set_index(date_col)["open"].sort_index()
    vals: list[float] = []
    for row in schedule.itertuples(index=False):
        buy = pd.Timestamp(row.buy_trade_date).normalize()
        sell = pd.Timestamp(row.sell_trade_date).normalize()
        if buy in px.index and sell in px.index and np.isfinite(px.loc[buy]) and np.isfinite(px.loc[sell]) and px.loc[buy] != 0:
            vals.append(float(px.loc[sell] / px.loc[buy] - 1.0))
        else:
            vals.append(np.nan)
    out = pd.Series(vals, index=schedule.index, name=spec.name)
    meta["status"] = "ok" if out.notna().any() else "no_aligned_dates"
    meta["covered_months"] = int(out.notna().sum())
    meta["first_date"] = str(px.index.min().date()) if not px.empty else ""
    meta["last_date"] = str(px.index.max().date()) if not px.empty else ""
    return out, meta


def load_promoted_monthly(path: Path, *, model: str, pool: str, top_k: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {
        "signal_date",
        "buy_trade_date",
        "sell_trade_date",
        "candidate_pool_version",
        "model",
        "top_k",
        "topk_return",
        "market_ew_return",
        "candidate_pool_mean_return",
        "cost_drag",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"monthly_long 缺少列: {missing}")
    out = df[
        df["candidate_pool_version"].astype(str).eq(pool)
        & df["model"].astype(str).eq(model)
        & pd.to_numeric(df["top_k"], errors="coerce").eq(int(top_k))
    ].copy()
    if out.empty:
        raise ValueError(f"monthly_long 中找不到 pool={pool} model={model} top_k={top_k}")
    for col in ["signal_date", "buy_trade_date", "sell_trade_date"]:
        out[col] = pd.to_datetime(out[col], errors="coerce").dt.normalize()
    for col in ["topk_return", "market_ew_return", "candidate_pool_mean_return", "cost_drag"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.sort_values("signal_date").reset_index(drop=True)


def build_benchmark_suite(monthly: pd.DataFrame, index_specs: list[BenchmarkSpec]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    schedule = monthly[["signal_date", "buy_trade_date", "sell_trade_date"]].copy()
    strategy_net = (monthly["topk_return"] - monthly["cost_drag"].fillna(0.0)).rename("model_top20_net")
    strategy_gross = monthly["topk_return"].rename("model_top20_gross")
    benchmark_series: dict[str, pd.Series] = {
        "model_top20_net": strategy_net,
        "model_top20_gross": strategy_gross,
        "u1_candidate_pool_ew": monthly["candidate_pool_mean_return"].rename("u1_candidate_pool_ew"),
        "all_a_market_ew": monthly["market_ew_return"].rename("all_a_market_ew"),
    }
    index_meta: list[dict[str, Any]] = []
    for spec in index_specs:
        ser, meta = load_index_csv_monthly_returns(spec, schedule)
        benchmark_series[spec.name] = ser
        index_meta.append(meta)

    summary_rows = [
        summarize_series("model_top20_net", strategy_net, role="strategy", primary=False),
        summarize_series("model_top20_gross", strategy_gross, role="strategy_gross", primary=False),
        summarize_series("u1_candidate_pool_ew", benchmark_series["u1_candidate_pool_ew"], role="alpha_internal", primary=False),
        summarize_series("all_a_market_ew", benchmark_series["all_a_market_ew"], role="broad_equal_weight", primary=False),
    ]
    for spec in index_specs:
        primary = spec.name.lower() in {"csi1000", "zz1000", "中证1000"}
        role = "primary_index" if primary else "secondary_index"
        summary_rows.append(summarize_series(spec.name, benchmark_series[spec.name], role=role, primary=primary))

    relative_rows = []
    for name, ser in benchmark_series.items():
        if name.startswith("model_top20"):
            continue
        relative_rows.append(summarize_relative("model_top20_net", strategy_net, name, ser))

    series = pd.concat([schedule, *[s.reset_index(drop=True).rename(name) for name, s in benchmark_series.items()]], axis=1)
    return pd.DataFrame(summary_rows), pd.DataFrame(relative_rows), series, index_meta


# ── P1-1: M10 成本敏感性测试 ──────────────────────────────────────────────

COST_SENSITIVITY_GRID: list[dict[str, Any]] = [
    {"cost_bps": 10.0, "label": "baseline_10bps"},
    {"cost_bps": 30.0, "label": "stress_30bps"},
    {"cost_bps": 50.0, "label": "stress_50bps"},
]


def build_cost_sensitivity(
    monthly: pd.DataFrame,
    *,
    base_cost_bps: float = 10.0,
    multi_cost_monthly: dict[float, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """P1-1: 计算不同成本档位下的 after-cost excess 表现。

    优先使用 multi_cost_monthly（真实回测结果），每个 DataFrame 的 cost_drag
    已包含对应 cost_bps 下的实际交易成本。若未提供，回退为基于 baseline
    cost_drag 的线性缩放（低估高成本场景，不推荐用于 gate 判断）。

    Parameters
    ----------
    monthly : DataFrame
        baseline (通常 10bps) 的 monthly_long。
    base_cost_bps : float
        baseline 对应的 cost_bps。
    multi_cost_monthly : dict or None
        {cost_bps: DataFrame} 映射，每个 DataFrame 是完整回测的 monthly_long。
    """
    if multi_cost_monthly:
        return _build_cost_sensitivity_from_real_runs(multi_cost_monthly)

    # 回退：线性缩放（仅用于快速估计，不作为 gate 判断依据）
    return _build_cost_sensitivity_linear_scale(monthly, base_cost_bps=base_cost_bps)


def _build_cost_sensitivity_from_real_runs(
    multi_cost: dict[float, pd.DataFrame],
) -> pd.DataFrame:
    """P0-1: 从真实多成本回测构建 cost sensitivity 表。"""
    rows: list[dict[str, Any]] = []
    for bps in sorted(multi_cost.keys()):
        df = multi_cost[bps]
        if df.empty:
            rows.append({"cost_bps": bps, "label": f"real_{bps:.0f}bps",
                         "after_cost_excess_mean": np.nan,
                         "after_cost_excess_total": np.nan,
                         "positive_months_ratio": np.nan, "breakeven": False,
                         "source": "real_backtest"})
            continue
        topk_ret = pd.to_numeric(df["topk_return"], errors="coerce")
        market_ret = pd.to_numeric(df["market_ew_return"], errors="coerce")
        cost_drag = pd.to_numeric(df["cost_drag"], errors="coerce").fillna(0.0)
        after_cost_excess = topk_ret - cost_drag - market_ret
        excess_valid = after_cost_excess.dropna()
        if excess_valid.empty:
            rows.append({"cost_bps": bps, "label": f"real_{bps:.0f}bps",
                         "after_cost_excess_mean": np.nan,
                         "after_cost_excess_total": np.nan,
                         "positive_months_ratio": np.nan, "breakeven": False,
                         "source": "real_backtest"})
            continue
        rows.append({
            "cost_bps": bps,
            "label": f"real_{bps:.0f}bps",
            "after_cost_excess_mean": float(excess_valid.mean()),
            "after_cost_excess_total": float((1.0 + excess_valid).prod() - 1.0),
            "positive_months_ratio": float((excess_valid > 0).mean()),
            "breakeven": bool(excess_valid.mean() > 0),
            "source": "real_backtest",
        })

    df = pd.DataFrame(rows)
    # 估算 breakeven cost bps（线性插值）
    pos = df[df["after_cost_excess_mean"] > 0]
    neg = df[df["after_cost_excess_mean"] <= 0]
    breakeven_bps = np.nan
    if not pos.empty and not neg.empty:
        p_low = pos.loc[pos["cost_bps"].idxmin()]
        n_high = neg.loc[neg["cost_bps"].idxmax()]
        x0, y0 = float(n_high["cost_bps"]), float(n_high["after_cost_excess_mean"])
        x1, y1 = float(p_low["cost_bps"]), float(p_low["after_cost_excess_mean"])
        if abs(y1 - y0) > 1e-12:
            breakeven_bps = float(x0 - y0 * (x1 - x0) / (y1 - y0))
    elif not pos.empty:
        breakeven_bps = float(pos["cost_bps"].max()) * 1.5
    elif not neg.empty:
        breakeven_bps = float(neg["cost_bps"].min()) * 0.5
    df["breakeven_bps"] = breakeven_bps
    df["method"] = "real_backtest"
    return df


def _build_cost_sensitivity_linear_scale(
    monthly: pd.DataFrame,
    *,
    base_cost_bps: float = 10.0,
) -> pd.DataFrame:
    """回退方法：对 baseline cost_drag 做线性缩放。

    ⚠️ 注意：此方法低估高成本场景的实际影响，不应用于 promotion gate 判断。
    """
    if monthly.empty:
        return pd.DataFrame()
    topk_ret = pd.to_numeric(monthly["topk_return"], errors="coerce")
    market_ret = pd.to_numeric(monthly["market_ew_return"], errors="coerce")
    base_drag = pd.to_numeric(monthly["cost_drag"], errors="coerce").fillna(0.0)

    rows: list[dict[str, Any]] = []
    for entry in COST_SENSITIVITY_GRID:
        bps = float(entry["cost_bps"])
        label = str(entry["label"])
        scaled_drag = base_drag * (bps / max(float(base_cost_bps), 1e-6))
        after_cost_excess = topk_ret - scaled_drag - market_ret
        excess_valid = after_cost_excess.dropna()
        if excess_valid.empty:
            rows.append({"cost_bps": bps, "label": label, "after_cost_excess_mean": np.nan,
                         "after_cost_excess_total": np.nan, "positive_months_ratio": np.nan,
                         "breakeven": False, "source": "linear_scale"})
            continue
        rows.append({
            "cost_bps": bps,
            "label": label,
            "after_cost_excess_mean": float(excess_valid.mean()),
            "after_cost_excess_total": float((1.0 + excess_valid).prod() - 1.0),
            "positive_months_ratio": float((excess_valid > 0).mean()),
            "breakeven": bool(excess_valid.mean() > 0),
            "source": "linear_scale",
        })

    df = pd.DataFrame(rows)
    pos = df[df["after_cost_excess_mean"] > 0]
    neg = df[df["after_cost_excess_mean"] <= 0]
    breakeven_bps = np.nan
    if not pos.empty and not neg.empty:
        p_low = pos.loc[pos["cost_bps"].idxmin()] if not pos.empty else None
        n_high = neg.loc[neg["cost_bps"].idxmax()] if not neg.empty else None
        if p_low is not None and n_high is not None:
            x0, y0 = float(n_high["cost_bps"]), float(n_high["after_cost_excess_mean"])
            x1, y1 = float(p_low["cost_bps"]), float(p_low["after_cost_excess_mean"])
            if abs(y1 - y0) > 1e-12:
                breakeven_bps = float(x0 - y0 * (x1 - x0) / (y1 - y0))
    elif not pos.empty:
        breakeven_bps = float(pos["cost_bps"].max()) * 1.5
    elif not neg.empty:
        breakeven_bps = float(neg["cost_bps"].min()) * 0.5
    df["breakeven_bps"] = breakeven_bps
    df["method"] = "linear_scale"
    return df


# ── P1-1: M10 容量分析 ──────────────────────────────────────────────────

def build_capacity_analysis(
    monthly: pd.DataFrame,
    *,
    db_path: str = "",
    daily_table: str = "a_share_daily",
    top_k: int = 20,
) -> pd.DataFrame:
    """P1-1: 从日线数据计算 Top-K 推荐名单的容量估计。

    对每月信号日，查找 Top-K 推荐股票在近 20 日的日均成交额，
    估算冲击成本 1% 时的最大可交易量（日均成交额 × 0.01）。

    若无 daily 数据路径或 DuckDB 不可用，返回空 DataFrame。
    """
    if not db_path or monthly.empty:
        return pd.DataFrame()
    db = Path(db_path)
    if not db.exists():
        return pd.DataFrame()

    import duckdb
    try:
        con = duckdb.connect(str(db), read_only=True)
    except Exception:
        return pd.DataFrame()

    try:
        # 获取月度信号日序列
        schedule = monthly[["signal_date", "buy_trade_date"]].drop_duplicates().copy()
        schedule["signal_date"] = pd.to_datetime(schedule["signal_date"], errors="coerce").dt.normalize()
        schedule = schedule.dropna(subset=["signal_date"])

        rows: list[dict[str, Any]] = []
        for _, srow in schedule.iterrows():
            sd = pd.Timestamp(srow["signal_date"]).normalize()
            # Top-K 选股（从 monthly 中提取该信号日的 Top-K）
            slice_df = monthly[
                (pd.to_datetime(monthly["signal_date"], errors="coerce").dt.normalize() == sd)
            ]
            if slice_df.empty:
                continue
            # 按 topk_return 排序（正收益更好），取 top_k 只
            top_symbols = (
                slice_df.sort_values("topk_return", ascending=False)
                .head(int(top_k))["symbol"]
                .dropna()
                .astype(str)
                .str.zfill(6)
                .tolist()
            )
            if not top_symbols:
                continue

            # 查询近 20 日日均成交额
            sym_list = ", ".join(f"'{s}'" for s in top_symbols)
            lookback_start = (sd - pd.Timedelta(days=40)).strftime("%Y-%m-%d")
            lookback_end = sd.strftime("%Y-%m-%d")
            try:
                adv_df = con.execute(
                    f"""
                    SELECT symbol,
                           AVG(amount) AS avg_amount_20d
                    FROM {daily_table}
                    WHERE trade_date >= '{lookback_start}'
                      AND trade_date <= '{lookback_end}'
                      AND symbol IN ({sym_list})
                    GROUP BY symbol
                    """
                ).fetchdf()
            except Exception:
                adv_df = pd.DataFrame()

            if adv_df.empty:
                rows.append({
                    "signal_date": sd,
                    "top_symbols": len(top_symbols),
                    "min_avg_amount": np.nan,
                    "median_avg_amount": np.nan,
                    "total_capacity_est": np.nan,
                })
                continue

            adv_df["avg_amount"] = pd.to_numeric(adv_df["avg_amount"], errors="coerce")
            adv_valid = adv_df["avg_amount"].dropna()
            if adv_valid.empty:
                rows.append({
                    "signal_date": sd,
                    "top_symbols": len(top_symbols),
                    "min_avg_amount": np.nan,
                    "median_avg_amount": np.nan,
                    "total_capacity_est": np.nan,
                })
                continue

            # 假设冲击成本 1% 时最大可交易量 = 日均成交额 × 0.01
            capacity_est = float(adv_valid.sum() * 0.01)
            rows.append({
                "signal_date": sd,
                "top_symbols": len(top_symbols),
                "min_avg_amount": float(adv_valid.min()),
                "median_avg_amount": float(adv_valid.median()),
                "total_capacity_est": capacity_est,
            })
    finally:
        con.close()

    return pd.DataFrame(rows)


def build_limit_up_stress_comparison(
    monthly_base: pd.DataFrame,
    monthly_limit_up: pd.DataFrame | None = None,
    *,
    base_label: str = "baseline_idle",
    stress_label: str = "stress_redistribute",
) -> pd.DataFrame:
    """P1-1: 对比 baseline 与 limit-up redistrib 模式的 after-cost excess 差异。

    若未提供 stress 文件，返回空 DataFrame。
    """
    if monthly_limit_up is None or monthly_limit_up.empty or monthly_base.empty:
        return pd.DataFrame()

    def _extract_excess(df: pd.DataFrame, label: str) -> pd.Series:
        topk = pd.to_numeric(df["topk_return"], errors="coerce")
        market = pd.to_numeric(df["market_ew_return"], errors="coerce")
        drag = pd.to_numeric(df["cost_drag"], errors="coerce").fillna(0.0)
        excess = (topk - drag - market).rename(label)
        return excess.dropna()

    base_excess = _extract_excess(monthly_base, base_label)
    stress_excess = _extract_excess(monthly_limit_up, stress_label)
    if base_excess.empty or stress_excess.empty:
        return pd.DataFrame()

    # 对齐信号日
    aligned = pd.concat([base_excess, stress_excess], axis=1).dropna()
    if aligned.empty:
        return pd.DataFrame()

    delta = aligned[stress_label] - aligned[base_label]
    return pd.DataFrame([{
        "comparison": f"{stress_label}_vs_{base_label}",
        "months": int(len(aligned)),
        "baseline_mean_excess": float(aligned[base_label].mean()),
        "stress_mean_excess": float(aligned[stress_label].mean()),
        "delta_mean": float(delta.mean()),
        "delta_total": float((1.0 + delta).prod() - 1.0),
        "positive_delta_ratio": float((delta > 0).mean()),
        "baseline_sharpe": float(aligned[base_label].mean() / (aligned[base_label].std() + 1e-12) * np.sqrt(12)),
        "stress_sharpe": float(aligned[stress_label].mean() / (aligned[stress_label].std() + 1e-12) * np.sqrt(12)),
    }])


def _pct_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").map(lambda x: "" if pd.isna(x) else f"{x:.2%}")
    return out


def build_doc(
    *,
    monthly_path: Path,
    model: str,
    pool: str,
    top_k: int,
    summary: pd.DataFrame,
    relative: pd.DataFrame,
    index_meta: list[dict[str, Any]],
    cost_sensitivity: pd.DataFrame,
    capacity: pd.DataFrame = None,
    limit_up_stress: pd.DataFrame = None,
    artifacts: list[str] = None,
) -> str:
    generated_at = pd.Timestamp.utcnow().isoformat()
    if capacity is None:
        capacity = pd.DataFrame()
    if limit_up_stress is None:
        limit_up_stress = pd.DataFrame()
    if artifacts is None:
        artifacts = []
    summary_view = _pct_cols(
        summary,
        ["total_return", "annualized_return", "mean_monthly_return", "median_monthly_return", "monthly_positive_rate", "max_drawdown"],
    )
    relative_view = _pct_cols(
        relative,
        ["excess_total_return", "annualized_excess_arithmetic", "mean_monthly_excess", "median_monthly_excess", "monthly_hit_rate"],
    )
    meta = pd.DataFrame(index_meta) if index_meta else pd.DataFrame(
        [{"benchmark": "csi1000/csi2000", "status": "not_supplied", "note": "提供 --index-csv 后自动纳入主基准。"}]
    )
    # P1-1: 成本敏感性表格格式化
    cost_view = pd.DataFrame()
    cost_method = "linear_scale (⚠️ 不应用于 promotion gate)"
    if not cost_sensitivity.empty:
        cost_view = cost_sensitivity.copy()
        cost_method = str(cost_sensitivity["method"].iloc[0]) if "method" in cost_sensitivity.columns else "linear_scale"
        if cost_method == "linear_scale":
            cost_method += " (⚠️ 不应用于 promotion gate — 低估高成本影响)"
        else:
            cost_method += " (✅ 真实回测 cost_drag)"
        cost_view["after_cost_excess_mean"] = pd.to_numeric(
            cost_view["after_cost_excess_mean"], errors="coerce"
        ).map(lambda x: "" if pd.isna(x) else f"{x:.4%}")
        cost_view["after_cost_excess_total"] = pd.to_numeric(
            cost_view["after_cost_excess_total"], errors="coerce"
        ).map(lambda x: "" if pd.isna(x) else f"{x:.2%}")
        cost_view["positive_months_ratio"] = pd.to_numeric(
            cost_view["positive_months_ratio"], errors="coerce"
        ).map(lambda x: "" if pd.isna(x) else f"{x:.1%}")
        breakeven = cost_sensitivity["breakeven_bps"].dropna()
        breakeven_str = f"{float(breakeven.iloc[0]):.1f} bps" if len(breakeven) > 0 else "N/A"
    else:
        breakeven_str = "N/A"
    cost_section = f"""## Cost Sensitivity (M10)

- 方法: `{cost_method}`
- 30bps gate: after_cost_excess_mean > 0 是 promotion 前提条件。
{_format_markdown_table(cost_view[["label", "cost_bps", "after_cost_excess_mean", "after_cost_excess_total", "positive_months_ratio", "breakeven"]], max_rows=10)}

- 估算 Breakeven 成本: `{breakeven_str}`
- 解释: `after_cost_excess = topk_return - cost_drag - market_ew_return`
""" if not cost_sensitivity.empty else ""

    # P1-1: 容量分析 section
    capacity_section = ""
    if not capacity.empty:
        cap_view = capacity.copy()
        cap_view["min_avg_amount"] = pd.to_numeric(
            cap_view["min_avg_amount"], errors="coerce"
        ).map(lambda x: "" if pd.isna(x) else f"¥{x/1e4:.0f}万")
        cap_view["median_avg_amount"] = pd.to_numeric(
            cap_view["median_avg_amount"], errors="coerce"
        ).map(lambda x: "" if pd.isna(x) else f"¥{x/1e4:.0f}万")
        cap_view["total_capacity_est"] = pd.to_numeric(
            cap_view["total_capacity_est"], errors="coerce"
        ).map(lambda x: "" if pd.isna(x) else f"¥{x/1e4:.0f}万")
        cap_min_amount = capacity["min_avg_amount"].dropna()
        cap_total = capacity["total_capacity_est"].dropna()
        cap_min_str = f"¥{float(cap_min_amount.min())/1e4:.0f}万" if not cap_min_amount.empty else "N/A"
        cap_total_str = f"¥{float(cap_total.median())/1e4:.0f}万" if not cap_total.empty else "N/A"
        capacity_section = f"""## Capacity Analysis (M10)

{_format_markdown_table(cap_view[["signal_date", "top_symbols", "min_avg_amount", "median_avg_amount", "total_capacity_est"]], max_rows=20)}

- Top{top_k} 最小日均成交额: `{cap_min_str}`
- Top{top_k} 组合容量中位数（1% 冲击约束）: `{cap_total_str}`
- 容量估计方法: `total_capacity = sum(avg_amount_20d) × 0.01`（冲击成本 1% 假设）
- 组合容量 > ¥1 亿是 promotion 前提条件之一。
""" if not capacity.empty else ""

    # P1-1: limit-up/VWAP 压力测试 section
    limit_up_section = ""
    if not limit_up_stress.empty:
        lu_view = limit_up_stress.copy()
        for c in ["baseline_mean_excess", "stress_mean_excess", "delta_mean", "delta_total", "positive_delta_ratio"]:
            if c in lu_view.columns:
                lu_view[c] = pd.to_numeric(lu_view[c], errors="coerce").map(
                    lambda x: "" if pd.isna(x) else f"{x:.4%}"
                )
        lu_view["baseline_sharpe"] = pd.to_numeric(
            lu_view["baseline_sharpe"], errors="coerce"
        ).map(lambda x: "" if pd.isna(x) else f"{x:.2f}")
        lu_view["stress_sharpe"] = pd.to_numeric(
            lu_view["stress_sharpe"], errors="coerce"
        ).map(lambda x: "" if pd.isna(x) else f"{x:.2f}")
        lu_delta_mean = limit_up_stress["delta_mean"].dropna()
        lu_delta_str = f"{float(lu_delta_mean.iloc[0]):.4%}" if len(lu_delta_mean) > 0 else "N/A"
        limit_up_section = f"""## Limit-Up / VWAP Stress Test (M10)

{_format_markdown_table(lu_view, max_rows=10)}

- 月均 excess delta: `{lu_delta_str}`
- 涨停买入失败影响 < 5% after-cost excess 损失为 promotion 前提条件之一。
""" if not limit_up_stress.empty else ""

    artifact_lines = "\n".join(f"- `{x}`" for x in artifacts)
    return f"""# Monthly Selection Benchmark Suite

- 生成时间：`{generated_at}`
- monthly_long：`{monthly_path.relative_to(ROOT) if monthly_path.is_relative_to(ROOT) else monthly_path}`
- 模型：`{model}`
- 候选池：`{pool}`
- Top-K：`{top_k}`
- 基准体系：主基准中证1000（若提供指数行情）、辅助中证2000、内部 alpha 基准 U1 候选池等权、宽基参照全A等权。

## Return Summary

{_format_markdown_table(summary_view, max_rows=30)}

## Excess vs Benchmarks

{_format_markdown_table(relative_view, max_rows=30)}

{cost_section}
{capacity_section}
{limit_up_section}
## Index Inputs

{_format_markdown_table(meta, max_rows=30)}

## 口径

- `model_top20_net` = `topk_return - cost_drag`，首月无上一期持仓时成本按 0 处理。
- `u1_candidate_pool_ew` = 同一 `U1_liquid_tradable` 候选池内股票月度收益等权平均，是内部 alpha 基准。
- `all_a_market_ew` = dataset 标签层全市场 open-to-open 等权基准，不是中证指数。
- 指数 CSV 若提供，按 `buy_trade_date` 开盘到 `sell_trade_date` 开盘计算 open-to-open 月收益，和模型执行时钟对齐。

## 本轮产物

{artifact_lines}
"""


def main() -> int:
    started_at = time.perf_counter()
    args = parse_args()
    monthly_path = _resolve_project_path(args.monthly_long)
    as_of = args.as_of_date.strip() or pd.Timestamp.now().strftime("%Y-%m-%d")
    results_dir = _resolve_project_path(args.results_dir)
    docs_dir = ROOT / "docs" / "reports" / as_of[:7]
    results_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)

    monthly = load_promoted_monthly(monthly_path, model=args.model, pool=args.candidate_pool, top_k=int(args.top_k))
    index_specs = parse_index_specs(args.index_csv)
    summary, relative, series, index_meta = build_benchmark_suite(monthly, index_specs)

    # P0-1: 优先使用多成本真实回测；否则回退线性缩放
    multi_cost_monthly: dict[float, pd.DataFrame] | None = None
    if args.multi_cost_monthly_long:
        multi_cost_monthly = {}
        for item in args.multi_cost_monthly_long:
            if "=" not in item:
                print(f"[benchmark] 跳过无效 multi-cost 参数: {item}", file=sys.stderr)
                continue
            bps_str, path_str = item.split("=", 1)
            try:
                bps = float(bps_str)
            except ValueError:
                print(f"[benchmark] 跳过无效 cost_bps: {bps_str}", file=sys.stderr)
                continue
            cost_path = _resolve_project_path(path_str.strip())
            if not cost_path.exists():
                print(f"[benchmark] multi-cost 文件不存在: {cost_path}", file=sys.stderr)
                continue
            try:
                cost_monthly = load_promoted_monthly(
                    cost_path, model=args.model, pool=args.candidate_pool, top_k=int(args.top_k)
                )
                multi_cost_monthly[bps] = cost_monthly
                print(f"[benchmark] 加载 real cost_bps={bps:.0f} from {cost_path}", flush=True)
            except Exception as exc:
                print(f"[benchmark] 加载失败 cost_bps={bps}: {exc}", file=sys.stderr)

    cost_sensitivity = build_cost_sensitivity(
        monthly,
        base_cost_bps=10.0,
        multi_cost_monthly=multi_cost_monthly if multi_cost_monthly else None,
    )

    # P1-1: M10 容量分析 & 涨停/VWAP 压力测试对比
    capacity = build_capacity_analysis(
        monthly,
        db_path=args.db_path,
        daily_table=args.daily_table,
        top_k=int(args.top_k),
    )
    limit_up_vwap_comparison = pd.DataFrame()
    if args.stress_monthly_long:
        stress_path = _resolve_project_path(args.stress_monthly_long)
        if stress_path.exists():
            try:
                stress_monthly = load_promoted_monthly(
                    stress_path,
                    model=args.model,
                    pool=args.candidate_pool,
                    top_k=int(args.top_k),
                )
                limit_up_vwap_comparison = build_limit_up_stress_comparison(
                    monthly, stress_monthly,
                    base_label="baseline",
                    stress_label=args.stress_label,
                )
            except Exception:
                pass

    output_stem = f"{args.output_prefix}_{as_of}"
    paths = {
        "summary": results_dir / f"{output_stem}_summary.csv",
        "relative": results_dir / f"{output_stem}_relative.csv",
        "series": results_dir / f"{output_stem}_monthly_series.csv",
        "cost_sensitivity": results_dir / f"{output_stem}_cost_sensitivity.csv",
        "capacity": results_dir / f"{output_stem}_capacity.csv",
        "limit_up_stress": results_dir / f"{output_stem}_limit_up_stress.csv",
        "manifest": results_dir / f"{output_stem}_manifest.json",
        "doc": docs_dir / f"{output_stem}.md",
    }
    summary.to_csv(paths["summary"], index=False)
    relative.to_csv(paths["relative"], index=False)
    series.to_csv(paths["series"], index=False)
    cost_sensitivity.to_csv(paths["cost_sensitivity"], index=False)
    capacity.to_csv(paths["capacity"], index=False)
    limit_up_vwap_comparison.to_csv(paths["limit_up_stress"], index=False)
    artifact_paths = [
        _project_relative(paths["summary"]),
        _project_relative(paths["relative"]),
        _project_relative(paths["series"]),
        _project_relative(paths["cost_sensitivity"]),
        _project_relative(paths["capacity"]),
        _project_relative(paths["limit_up_stress"]),
        _project_relative(paths["manifest"]),
        _project_relative(paths["doc"]),
    ]
    paths["doc"].write_text(
        build_doc(
            monthly_path=monthly_path,
            model=args.model,
            pool=args.candidate_pool,
            top_k=int(args.top_k),
            summary=summary,
            relative=relative,
            index_meta=index_meta,
            cost_sensitivity=cost_sensitivity,
            capacity=capacity,
            limit_up_stress=limit_up_vwap_comparison,
            artifacts=artifact_paths,
        ),
        encoding="utf-8",
    )

    # --- standard research contract ---
    identity = make_research_identity(
        result_type="monthly_selection_benchmark_suite",
        research_topic="monthly_selection_benchmark_suite",
        research_config_id=(
            f"model_{slugify_token(args.model)}_pool_{slugify_token(args.candidate_pool)}"
            f"_topk_{int(args.top_k)}"
        ),
        output_stem=output_stem,
        parent_result_id="",
    )
    signal_dates = pd.to_datetime(monthly["signal_date"], errors="coerce").dropna()
    data_slice = DataSlice(
        dataset_name="monthly_selection_monthly_long",
        source_tables=("monthly_selection_monthly_long",),
        date_start=str(signal_dates.min().date()) if not signal_dates.empty else "",
        date_end=str(signal_dates.max().date()) if not signal_dates.empty else "",
        asof_trade_date=as_of,
        signal_date_col="signal_date",
        symbol_col="symbol",
        candidate_pool_version=args.candidate_pool,
        rebalance_rule="M",
        execution_mode="tplus1_open",
        label_return_mode="open_to_open",
        feature_set_id=args.model,
        feature_columns=(),
        label_columns=("topk_return", "market_ew_return", "candidate_pool_mean_return"),
        pit_policy="derived_from_monthly_selection_research_outputs",
        config_path=None,
        extra={
            "monthly_long_path": _project_relative(monthly_path),
            "top_k": int(args.top_k),
            "index_input_count": len(index_specs),
        },
    )
    artifact_refs = (
        ArtifactRef("summary_csv", _project_relative(paths["summary"]), "csv", False, "基准收益汇总"),
        ArtifactRef("relative_csv", _project_relative(paths["relative"]), "csv", False, "相对基准表现"),
        ArtifactRef("monthly_series_csv", _project_relative(paths["series"]), "csv", False, "月度收益序列"),
        ArtifactRef("cost_sensitivity_csv", _project_relative(paths["cost_sensitivity"]), "csv", False, "成本敏感性分析"),
        ArtifactRef("capacity_csv", _project_relative(paths["capacity"]), "csv", False, "容量分析"),
        ArtifactRef("limit_up_stress_csv", _project_relative(paths["limit_up_stress"]), "csv", False, "涨停/VWAP压力测试"),
        ArtifactRef("report_md", _project_relative(paths["doc"]), "md", False, "基准对比报告"),
        ArtifactRef("manifest_json", _project_relative(paths["manifest"]), "json", False, "标准研究 manifest"),
    )
    strategy_summary = summary[summary["benchmark"].astype(str).eq("model_top20_net")]
    strategy_row = strategy_summary.iloc[0].to_dict() if not strategy_summary.empty else {}
    relative_u1 = relative[relative["benchmark"].astype(str).eq("u1_candidate_pool_ew")]
    relative_u1_row = relative_u1.iloc[0].to_dict() if not relative_u1.empty else {}
    metrics = {
        "months": int(series["model_top20_net"].notna().sum()) if "model_top20_net" in series.columns else 0,
        "benchmark_count": int(len(summary)),
        "relative_benchmark_count": int(len(relative)),
        "model_top20_net_total_return": strategy_row.get("total_return"),
        "model_top20_net_annualized_return": strategy_row.get("annualized_return"),
        "excess_vs_u1_total_return": relative_u1_row.get("excess_total_return"),
        "information_ratio_vs_u1": relative_u1_row.get("information_ratio"),
    }
    gates = {
        "data_gate": {
            "passed": bool(len(monthly) > 0 and signal_dates.notna().any()),
            "monthly_rows": int(len(monthly)),
            "date_start": data_slice.date_start,
            "date_end": data_slice.date_end,
        },
        "benchmark_gate": {
            "passed": bool({"u1_candidate_pool_ew", "all_a_market_ew"}.issubset(set(summary["benchmark"].astype(str)))),
            "index_input_count": len(index_specs),
        },
        "governance_gate": {
            "passed": True,
            "manifest_schema": "research_result_v1",
        },
    }
    result = ExperimentResult(
        result_id=build_result_id(identity, [data_slice], metrics),
        identity=identity,
        script_name=_project_relative(Path(__file__).resolve()),
        command=shlex.join([sys.executable, *sys.argv]),
        created_at=utc_now_iso(),
        duration_sec=round(time.perf_counter() - started_at, 6),
        seed=None,
        data_slices=(data_slice,),
        config={"config_path": "", "config_hash": None},
        params={
            "cli": {k: str(v) for k, v in vars(args).items()},
            "benchmark_policy": {
                "primary": "csi1000 when index input is supplied",
                "secondary": ["csi2000", "u1_candidate_pool_ew", "all_a_market_ew"],
                "internal_alpha_benchmark": "u1_candidate_pool_ew",
            },
            "index_inputs": index_meta,
        },
        metrics=metrics,
        gates=gates,
        artifacts=artifact_refs,
        promotion={
            "production_eligible": False,
            "registry_status": "not_registered",
            "blocking_reasons": ["benchmark_suite_is_diagnostic_research_only"],
        },
        notes="Monthly selection benchmark comparison; does not change promoted config.",
    )
    write_research_manifest(
        paths["manifest"],
        result,
        extra={
            "generated_at_utc": result.created_at,
            "legacy_result_type": "monthly_selection_benchmark_suite",
            "legacy_artifacts": artifact_paths,
            "monthly_long": _project_relative(monthly_path),
            "model": args.model,
            "candidate_pool": args.candidate_pool,
            "top_k": int(args.top_k),
        },
    )
    append_experiment_result(results_dir.parent / "experiments", result)
    # --- end standard research contract ---

    print(f"[benchmark-suite] summary={paths['summary']}")
    print(f"[benchmark-suite] relative={paths['relative']}")
    print(f"[benchmark-suite] series={paths['series']}")
    print(f"[benchmark-suite] manifest={paths['manifest']}")
    print(f"[benchmark-suite] doc={paths['doc']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
