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
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_monthly_selection_baselines import _format_markdown_table, _json_sanitize


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
    return p.parse_args()


def _resolve_project_path(raw: str | Path) -> Path:
    p = Path(raw)
    return p if p.is_absolute() else ROOT / p


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
    artifacts: list[str],
) -> str:
    generated_at = pd.Timestamp.utcnow().isoformat()
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

    output_stem = f"{args.output_prefix}_{as_of}"
    paths = {
        "summary": results_dir / f"{output_stem}_summary.csv",
        "relative": results_dir / f"{output_stem}_relative.csv",
        "series": results_dir / f"{output_stem}_monthly_series.csv",
        "manifest": results_dir / f"{output_stem}_manifest.json",
        "doc": docs_dir / f"{output_stem}.md",
    }
    summary.to_csv(paths["summary"], index=False)
    relative.to_csv(paths["relative"], index=False)
    series.to_csv(paths["series"], index=False)
    manifest = {
        "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
        "result_type": "monthly_selection_benchmark_suite",
        "monthly_long": str(monthly_path.relative_to(ROOT)) if monthly_path.is_relative_to(ROOT) else str(monthly_path),
        "model": args.model,
        "candidate_pool": args.candidate_pool,
        "top_k": int(args.top_k),
        "benchmark_policy": {
            "primary": "csi1000 when index input is supplied",
            "secondary": ["csi2000", "u1_candidate_pool_ew", "all_a_market_ew"],
            "internal_alpha_benchmark": "u1_candidate_pool_ew",
        },
        "index_inputs": index_meta,
        "artifacts": [
            str(paths["summary"].relative_to(ROOT)),
            str(paths["relative"].relative_to(ROOT)),
            str(paths["series"].relative_to(ROOT)),
            str(paths["doc"].relative_to(ROOT)),
        ],
    }
    paths["manifest"].write_text(json.dumps(_json_sanitize(manifest), ensure_ascii=False, indent=2), encoding="utf-8")
    artifact_paths = [
        str(paths["summary"].relative_to(ROOT)),
        str(paths["relative"].relative_to(ROOT)),
        str(paths["series"].relative_to(ROOT)),
        str(paths["manifest"].relative_to(ROOT)),
        str(paths["doc"].relative_to(ROOT)),
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
            artifacts=artifact_paths,
        ),
        encoding="utf-8",
    )
    print(f"[benchmark-suite] summary={paths['summary']}")
    print(f"[benchmark-suite] relative={paths['relative']}")
    print(f"[benchmark-suite] series={paths['series']}")
    print(f"[benchmark-suite] doc={paths['doc']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
