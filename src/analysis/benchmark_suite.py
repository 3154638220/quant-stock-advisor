"""月度选股基准对比分析核心模块。

从 scripts/run_monthly_benchmark_suite.py 提取：
- 统计函数（compound/ann/annual/max_drawdown/sharpe/IR）
- Benchmark 数据加载与对比
- M10 成本敏感性分析
- M10 容量分析
- 涨停/VWAP 压力测试对比
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ── 统计函数 ──────────────────────────────────────────────────────────────


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
        "benchmark": name, "role": role, "primary": bool(primary),
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
    aligned = pd.concat([
        pd.to_numeric(strategy, errors="coerce").rename("strategy"),
        pd.to_numeric(benchmark, errors="coerce").rename("benchmark"),
    ], axis=1).dropna()
    if aligned.empty:
        return {
            "strategy": strategy_name, "benchmark": benchmark_name, "months": 0,
            "excess_total_return": np.nan, "annualized_excess_arithmetic": np.nan,
            "mean_monthly_excess": np.nan, "median_monthly_excess": np.nan,
            "monthly_hit_rate": np.nan, "win_months": 0, "information_ratio": np.nan,
        }
    strat_ann = annualized_return(aligned["strategy"])
    bench_ann = annualized_return(aligned["benchmark"])
    excess = aligned["strategy"] - aligned["benchmark"]
    return {
        "strategy": strategy_name, "benchmark": benchmark_name,
        "months": int(len(aligned)),
        "excess_total_return": compounded_return(aligned["strategy"]) - compounded_return(aligned["benchmark"]),
        "annualized_excess_arithmetic": strat_ann - bench_ann if np.isfinite(strat_ann) and np.isfinite(bench_ann) else np.nan,
        "mean_monthly_excess": float(excess.mean()),
        "median_monthly_excess": float(excess.median()),
        "monthly_hit_rate": float((excess > 0).mean()),
        "win_months": int((excess > 0).sum()),
        "information_ratio": information_ratio(aligned["strategy"], aligned["benchmark"]),
    }


# ── Index CSV 加载 ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    path: Path
    symbol: str = ""


def parse_index_specs(items: list[str], project_root: Path) -> list[BenchmarkSpec]:
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
        p = Path(raw_path.strip())
        specs.append(BenchmarkSpec(name=name, symbol=symbol, path=p if p.is_absolute() else project_root / p))
    return specs


def _resolve_index_path(p: Path, project_root: Path) -> Path:
    return p if p.is_absolute() else project_root / p


def load_index_csv_monthly_returns(spec: BenchmarkSpec, schedule: pd.DataFrame, project_root: Path) -> tuple[pd.Series, dict[str, Any]]:
    meta: dict[str, Any] = {
        "benchmark": spec.name,
        "source": str(spec.path.relative_to(project_root)) if spec.path.is_relative_to(project_root) else str(spec.path),
        "symbol_filter": spec.symbol, "status": "missing", "covered_months": 0,
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
        "signal_date", "buy_trade_date", "sell_trade_date",
        "candidate_pool_version", "model", "top_k",
        "topk_return", "market_ew_return", "candidate_pool_mean_return", "cost_drag",
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
        ser, meta = load_index_csv_monthly_returns(spec, schedule, Path.cwd())
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


# ── M10 成本敏感性 ──────────────────────────────────────────────────────────


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
    """计算不同成本档位下的 after-cost excess 表现。"""
    if multi_cost_monthly:
        return _build_cost_sensitivity_from_real_runs(multi_cost_monthly)
    return _build_cost_sensitivity_linear_scale(monthly, base_cost_bps=base_cost_bps)


def _build_cost_sensitivity_from_real_runs(multi_cost: dict[float, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for bps in sorted(multi_cost.keys()):
        df = multi_cost[bps]
        if df.empty:
            rows.append({"cost_bps": bps, "label": f"real_{bps:.0f}bps",
                         "after_cost_excess_mean": np.nan, "after_cost_excess_total": np.nan,
                         "positive_months_ratio": np.nan, "breakeven": False, "source": "real_backtest"})
            continue
        topk_ret = pd.to_numeric(df["topk_return"], errors="coerce")
        market_ret = pd.to_numeric(df["market_ew_return"], errors="coerce")
        cost_drag = pd.to_numeric(df["cost_drag"], errors="coerce").fillna(0.0)
        after_cost_excess = topk_ret - cost_drag - market_ret
        excess_valid = after_cost_excess.dropna()
        if excess_valid.empty:
            rows.append({"cost_bps": bps, "label": f"real_{bps:.0f}bps",
                         "after_cost_excess_mean": np.nan, "after_cost_excess_total": np.nan,
                         "positive_months_ratio": np.nan, "breakeven": False, "source": "real_backtest"})
            continue
        rows.append({
            "cost_bps": bps, "label": f"real_{bps:.0f}bps",
            "after_cost_excess_mean": float(excess_valid.mean()),
            "after_cost_excess_total": float((1.0 + excess_valid).prod() - 1.0),
            "positive_months_ratio": float((excess_valid > 0).mean()),
            "breakeven": bool(excess_valid.mean() > 0), "source": "real_backtest",
        })
    df = pd.DataFrame(rows)
    breakeven_bps = _estimate_breakeven(df)
    df["breakeven_bps"] = breakeven_bps
    df["method"] = "real_backtest"
    return df


def _build_cost_sensitivity_linear_scale(monthly: pd.DataFrame, *, base_cost_bps: float = 10.0) -> pd.DataFrame:
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
            "cost_bps": bps, "label": label,
            "after_cost_excess_mean": float(excess_valid.mean()),
            "after_cost_excess_total": float((1.0 + excess_valid).prod() - 1.0),
            "positive_months_ratio": float((excess_valid > 0).mean()),
            "breakeven": bool(excess_valid.mean() > 0), "source": "linear_scale",
        })
    df = pd.DataFrame(rows)
    breakeven_bps = _estimate_breakeven(df)
    df["breakeven_bps"] = breakeven_bps
    df["method"] = "linear_scale"
    return df


def _estimate_breakeven(df: pd.DataFrame) -> float:
    pos = df[df["after_cost_excess_mean"] > 0]
    neg = df[df["after_cost_excess_mean"] <= 0]
    if not pos.empty and not neg.empty:
        p_low = pos.loc[pos["cost_bps"].idxmin()]
        n_high = neg.loc[neg["cost_bps"].idxmax()]
        x0, y0 = float(n_high["cost_bps"]), float(n_high["after_cost_excess_mean"])
        x1, y1 = float(p_low["cost_bps"]), float(p_low["after_cost_excess_mean"])
        if abs(y1 - y0) > 1e-12:
            return float(x0 - y0 * (x1 - x0) / (y1 - y0))
    elif not pos.empty:
        return float(pos["cost_bps"].max()) * 1.5
    elif not neg.empty:
        return float(neg["cost_bps"].min()) * 0.5
    return np.nan


# ── M10 容量分析 ────────────────────────────────────────────────────────────


def build_capacity_analysis(
    monthly: pd.DataFrame,
    *,
    db_path: str = "",
    daily_table: str = "a_share_daily",
    top_k: int = 20,
) -> pd.DataFrame:
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
        schedule = monthly[["signal_date", "buy_trade_date"]].drop_duplicates().copy()
        schedule["signal_date"] = pd.to_datetime(schedule["signal_date"], errors="coerce").dt.normalize()
        schedule = schedule.dropna(subset=["signal_date"])
        rows: list[dict[str, Any]] = []
        for _, srow in schedule.iterrows():
            sd = pd.Timestamp(srow["signal_date"]).normalize()
            slice_df = monthly[
                (pd.to_datetime(monthly["signal_date"], errors="coerce").dt.normalize() == sd)
            ]
            if slice_df.empty:
                continue
            top_symbols = (
                slice_df.sort_values("topk_return", ascending=False)
                .head(int(top_k))["symbol"].dropna().astype(str).str.zfill(6).tolist()
            )
            if not top_symbols:
                continue
            sym_list = ", ".join(f"'{s}'" for s in top_symbols)
            lookback_start = (sd - pd.Timedelta(days=40)).strftime("%Y-%m-%d")
            lookback_end = sd.strftime("%Y-%m-%d")
            try:
                adv_df = con.execute(
                    f"""SELECT symbol, AVG(amount) AS avg_amount_20d
                    FROM {daily_table}
                    WHERE trade_date >= '{lookback_start}'
                      AND trade_date <= '{lookback_end}'
                      AND symbol IN ({sym_list})
                    GROUP BY symbol"""
                ).fetchdf()
            except Exception:
                adv_df = pd.DataFrame()
            if adv_df.empty:
                rows.append({"signal_date": sd, "top_symbols": len(top_symbols),
                             "min_avg_amount": np.nan, "median_avg_amount": np.nan,
                             "total_capacity_est": np.nan})
                continue
            adv_df["avg_amount"] = pd.to_numeric(adv_df["avg_amount"], errors="coerce")
            adv_valid = adv_df["avg_amount"].dropna()
            if adv_valid.empty:
                rows.append({"signal_date": sd, "top_symbols": len(top_symbols),
                             "min_avg_amount": np.nan, "median_avg_amount": np.nan,
                             "total_capacity_est": np.nan})
                continue
            rows.append({
                "signal_date": sd, "top_symbols": len(top_symbols),
                "min_avg_amount": float(adv_valid.min()),
                "median_avg_amount": float(adv_valid.median()),
                "total_capacity_est": float(adv_valid.sum() * 0.01),
            })
    finally:
        con.close()
    return pd.DataFrame(rows)


# ── M10 涨停/VWAP 压力测试 ──────────────────────────────────────────────────


def build_limit_up_stress_comparison(
    monthly_base: pd.DataFrame,
    monthly_limit_up: pd.DataFrame | None = None,
    *,
    base_label: str = "baseline_idle",
    stress_label: str = "stress_redistribute",
) -> pd.DataFrame:
    if monthly_limit_up is None or monthly_limit_up.empty or monthly_base.empty:
        return pd.DataFrame()

    def _extract_excess(df: pd.DataFrame, label: str) -> pd.Series:
        topk = pd.to_numeric(df["topk_return"], errors="coerce")
        market = pd.to_numeric(df["market_ew_return"], errors="coerce")
        drag = pd.to_numeric(df["cost_drag"], errors="coerce").fillna(0.0)
        return (topk - drag - market).rename(label).dropna()

    base_excess = _extract_excess(monthly_base, base_label)
    stress_excess = _extract_excess(monthly_limit_up, stress_label)
    if base_excess.empty or stress_excess.empty:
        return pd.DataFrame()

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


def _format_markdown_table(df, max_rows=30):  # 延迟导入避免循环
    from src.reporting.markdown_report import format_markdown_table
    return format_markdown_table(df, max_rows=max_rows)


def build_benchmark_doc(
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
    project_root: Path = None,
) -> str:
    root = project_root or Path.cwd()
    generated_at = pd.Timestamp.utcnow().isoformat()
    if capacity is None:
        capacity = pd.DataFrame()
    if limit_up_stress is None:
        limit_up_stress = pd.DataFrame()
    if artifacts is None:
        artifacts = []
    summary_view = _pct_cols(summary, [
        "total_return", "annualized_return", "mean_monthly_return",
        "median_monthly_return", "monthly_positive_rate", "max_drawdown",
    ])
    relative_view = _pct_cols(relative, [
        "excess_total_return", "annualized_excess_arithmetic",
        "mean_monthly_excess", "median_monthly_excess", "monthly_hit_rate",
    ])
    meta = pd.DataFrame(index_meta) if index_meta else pd.DataFrame(
        [{"benchmark": "csi1000/csi2000", "status": "not_supplied",
          "note": "提供 --index-csv 后自动纳入主基准。"}]
    )
    cost_view = pd.DataFrame()
    cost_method = "linear_scale"
    breakeven_str = "N/A"
    if not cost_sensitivity.empty:
        cost_view = cost_sensitivity.copy()
        cost_method = str(cost_sensitivity["method"].iloc[0]) if "method" in cost_sensitivity.columns else "linear_scale"
        if cost_method == "linear_scale":
            cost_method += " (⚠️ 不应用于 promotion gate)"
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
    cost_section = f"""## Cost Sensitivity (M10)

- 方法: `{cost_method}`
- 30bps gate: after_cost_excess_mean > 0 是 promotion 前提条件。
{_format_markdown_table(cost_view[["label", "cost_bps", "after_cost_excess_mean", "after_cost_excess_total", "positive_months_ratio", "breakeven"]], max_rows=10)}

- 估算 Breakeven 成本: `{breakeven_str}`
""" if not cost_sensitivity.empty else ""

    capacity_section = ""
    if not capacity.empty:
        cap_view = capacity.copy()
        cap_view["min_avg_amount"] = pd.to_numeric(cap_view["min_avg_amount"], errors="coerce").map(
            lambda x: "" if pd.isna(x) else f"¥{x/1e4:.0f}万"
        )
        cap_view["median_avg_amount"] = pd.to_numeric(cap_view["median_avg_amount"], errors="coerce").map(
            lambda x: "" if pd.isna(x) else f"¥{x/1e4:.0f}万"
        )
        cap_view["total_capacity_est"] = pd.to_numeric(cap_view["total_capacity_est"], errors="coerce").map(
            lambda x: "" if pd.isna(x) else f"¥{x/1e4:.0f}万"
        )
        cap_total = capacity["total_capacity_est"].dropna()
        cap_total_str = f"¥{float(cap_total.median())/1e4:.0f}万" if not cap_total.empty else "N/A"
        capacity_section = f"""## Capacity Analysis (M10)

{_format_markdown_table(cap_view[["signal_date", "top_symbols", "min_avg_amount", "median_avg_amount", "total_capacity_est"]], max_rows=20)}

- Top{top_k} 组合容量中位数（1% 冲击约束）: `{cap_total_str}`
"""

    limit_up_section = ""
    if not limit_up_stress.empty:
        lu_view = limit_up_stress.copy()
        for c in ["baseline_mean_excess", "stress_mean_excess", "delta_mean", "delta_total", "positive_delta_ratio"]:
            if c in lu_view.columns:
                lu_view[c] = pd.to_numeric(lu_view[c], errors="coerce").map(
                    lambda x: "" if pd.isna(x) else f"{x:.4%}"
                )
        lu_view["baseline_sharpe"] = pd.to_numeric(lu_view["baseline_sharpe"], errors="coerce").map(
            lambda x: "" if pd.isna(x) else f"{x:.2f}"
        )
        lu_view["stress_sharpe"] = pd.to_numeric(lu_view["stress_sharpe"], errors="coerce").map(
            lambda x: "" if pd.isna(x) else f"{x:.2f}"
        )
        limit_up_section = f"""## Limit-Up / VWAP Stress Test (M10)

{_format_markdown_table(lu_view, max_rows=10)}
"""

    artifact_lines = "\n".join(f"- `{x}`" for x in artifacts)
    monthly_path_str = str(monthly_path.relative_to(root)) if monthly_path.is_relative_to(root) else str(monthly_path)
    return f"""# Monthly Selection Benchmark Suite

- 生成时间：`{generated_at}`
- monthly_long：`{monthly_path_str}`
- 模型：`{model}` · 候选池：`{pool}` · Top-K：`{top_k}`

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

- `model_top20_net` = `topk_return - cost_drag`。
- `u1_candidate_pool_ew` = 同一候选池内股票月度收益等权平均。
- `all_a_market_ew` = dataset 标签层全市场 open-to-open 等权基准。

## 本轮产物

{artifact_lines}
"""
