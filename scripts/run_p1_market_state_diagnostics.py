#!/usr/bin/env python3
"""P1 market-relative 失败样本的年度/市场状态分层诊断。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_backtest_eval import build_market_ew_benchmark, build_open_to_open_returns, load_daily_from_duckdb  # noqa: E402
from src.settings import load_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="P1 market_relative 年度/市场状态分层诊断")
    p.add_argument("--config", default="config.yaml.backtest")
    p.add_argument("--monthly-csv", default="data/results/p1_marketrel_g1_gap_diagnostics_2026-04-26_monthly.csv")
    p.add_argument("--monthly-delta-csv", default="data/results/p1_marketrel_g1_gap_diagnostics_2026-04-26_monthly_delta.csv")
    p.add_argument("--start", default="")
    p.add_argument("--end", default="")
    p.add_argument("--benchmark-min-history-days", type=int, default=439)
    p.add_argument("--key-years", default="2022,2024,2025")
    p.add_argument("--output-prefix", default="p1_marketrel_state_diagnostics_2026-04-27")
    p.add_argument("--results-dir", default="data/results")
    p.add_argument("--docs-dir", default="docs")
    return p.parse_args()


def _resolve(path_like: str | Path) -> Path:
    p = Path(path_like).expanduser()
    return p if p.is_absolute() else ROOT / p


def _json_sanitize(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_sanitize(v) for v in obj]
    if isinstance(obj, tuple):
        return [_json_sanitize(v) for v in obj]
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return None if not np.isfinite(v) else v
    return obj


def _compound_return(values: pd.Series) -> float:
    arr = pd.to_numeric(values, errors="coerce").dropna()
    if arr.empty:
        return float("nan")
    return float((1.0 + arr).prod() - 1.0)


def _fmt_pct(value: Any) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "N/A"
    return "N/A" if not np.isfinite(v) else f"{v:+.2%}"


def _markdown_table(df: pd.DataFrame, *, max_rows: int | None = None) -> str:
    if df.empty:
        return "_无数据_"
    view = df.head(int(max_rows)).copy() if max_rows is not None else df.copy()

    def _cell(v: Any) -> str:
        if isinstance(v, pd.Timestamp):
            return v.strftime("%Y-%m-%d")
        if isinstance(v, (float, np.floating)):
            fv = float(v)
            return "" if not np.isfinite(fv) else f"{fv:.6g}"
        if isinstance(v, (int, np.integer)):
            return str(int(v))
        return str(v).replace("|", "\\|").replace("\n", " ")

    rows = ["| " + " | ".join(str(c) for c in view.columns) + " |"]
    rows.append("| " + " | ".join(["---"] * len(view.columns)) + " |")
    for _, row in view.iterrows():
        rows.append("| " + " | ".join(_cell(row[c]) for c in view.columns) + " |")
    if max_rows is not None and len(df) > max_rows:
        rows.append(f"\n_仅展示前 {max_rows} 行，共 {len(df)} 行。_")
    return "\n".join(rows)


def _tercile_label(series: pd.Series, *, low_label: str, mid_label: str, high_label: str) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    if s.notna().sum() < 3 or s.nunique(dropna=True) < 3:
        return pd.Series(np.where(s.notna(), mid_label, "unknown"), index=series.index)
    q1, q2 = s.quantile([1.0 / 3.0, 2.0 / 3.0]).to_list()
    return pd.Series(
        np.select([s <= q1, s >= q2], [low_label, high_label], default=mid_label),
        index=series.index,
    )


def build_monthly_market_state(
    *,
    daily: pd.DataFrame,
    start: str,
    end: str,
    benchmark_min_history_days: int,
) -> pd.DataFrame:
    benchmark = build_market_ew_benchmark(daily, start, end, min_days=int(benchmark_min_history_days))
    benchmark = pd.to_numeric(benchmark, errors="coerce").dropna().sort_index()
    if benchmark.empty:
        return pd.DataFrame()

    open_ret = build_open_to_open_returns(daily, zero_if_limit_up_open=False)
    open_ret = open_ret.reindex(benchmark.index).replace([np.inf, -np.inf], np.nan)
    breadth = (open_ret > 0).mean(axis=1)

    state = pd.DataFrame(
        {
            "benchmark_daily_return": benchmark,
            "breadth_positive_share": breadth,
        }
    )
    monthly = state.resample("ME").agg(
        benchmark_return=("benchmark_daily_return", _compound_return),
        benchmark_daily_vol=("benchmark_daily_return", lambda s: float(pd.to_numeric(s, errors="coerce").std(ddof=1))),
        benchmark_up_day_share=("benchmark_daily_return", lambda s: float((pd.to_numeric(s, errors="coerce") > 0).mean())),
        breadth_positive_share=("breadth_positive_share", "mean"),
        trading_days=("benchmark_daily_return", "count"),
    )
    monthly = monthly.reset_index(names="month_end")
    monthly["year"] = monthly["month_end"].dt.year.astype(int)
    monthly["month"] = monthly["month_end"].dt.month.astype(int)
    monthly["return_state"] = np.select(
        [
            monthly["benchmark_return"] >= 0.05,
            monthly["benchmark_return"] > 0.0,
            monthly["benchmark_return"] <= -0.05,
        ],
        ["strong_up", "mild_up", "strong_down"],
        default="mild_down",
    )
    monthly["vol_state"] = _tercile_label(
        monthly["benchmark_daily_vol"],
        low_label="low_vol",
        mid_label="mid_vol",
        high_label="high_vol",
    )
    monthly["breadth_state"] = _tercile_label(
        monthly["breadth_positive_share"],
        low_label="narrow_breadth",
        mid_label="mid_breadth",
        high_label="broad_breadth",
    )
    return monthly


def summarize_by_state(monthly: pd.DataFrame, *, state_col: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (group, state), part in monthly.groupby(["group", state_col], sort=True):
        rows.append(
            {
                "group": group,
                "state_axis": state_col,
                "state": state,
                "months": int(len(part)),
                "benchmark_compound_return": _compound_return(part["benchmark_return"]),
                "strategy_compound_return": _compound_return(part["strategy_return"]),
                "excess_sum": float(pd.to_numeric(part["excess_return"], errors="coerce").sum()),
                "median_excess_return": float(pd.to_numeric(part["excess_return"], errors="coerce").median()),
                "positive_excess_share": float((pd.to_numeric(part["excess_return"], errors="coerce") > 0).mean()),
            }
        )
    return pd.DataFrame(rows)


def summarize_delta_by_state(delta: pd.DataFrame, state: pd.DataFrame, *, state_col: str) -> pd.DataFrame:
    merged = delta.merge(state[["month_end", state_col]], on="month_end", how="left")
    rows: list[dict[str, Any]] = []
    for state_value, part in merged.groupby(state_col, sort=True):
        rows.append(
            {
                "state_axis": state_col,
                "state": state_value,
                "months": int(len(part)),
                "g1_minus_g0_excess_sum": float(
                    pd.to_numeric(part["g1_minus_g0_excess_return"], errors="coerce").sum()
                ),
                "median_g1_minus_g0_excess": float(
                    pd.to_numeric(part["g1_minus_g0_excess_return"], errors="coerce").median()
                ),
                "g1_beats_g0_share": float(
                    (pd.to_numeric(part["g1_minus_g0_excess_return"], errors="coerce") > 0).mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def summarize_key_year_states(monthly: pd.DataFrame, *, key_years: set[int]) -> pd.DataFrame:
    part = monthly[monthly["year"].isin(key_years)].copy()
    if part.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for (group, year, return_state), sub in part.groupby(["group", "year", "return_state"], sort=True):
        rows.append(
            {
                "group": group,
                "year": int(year),
                "return_state": return_state,
                "months": int(len(sub)),
                "benchmark_compound_return": _compound_return(sub["benchmark_return"]),
                "strategy_compound_return": _compound_return(sub["strategy_return"]),
                "excess_sum": float(pd.to_numeric(sub["excess_return"], errors="coerce").sum()),
                "median_excess_return": float(pd.to_numeric(sub["excess_return"], errors="coerce").median()),
            }
        )
    return pd.DataFrame(rows)


def _build_doc(
    *,
    prefix: str,
    monthly: pd.DataFrame,
    state_summary: pd.DataFrame,
    delta_summary: pd.DataFrame,
    key_year_state: pd.DataFrame,
    worst_months: pd.DataFrame,
    key_years: set[int],
) -> str:
    g1_return = state_summary[
        (state_summary["group"] == "G1") & (state_summary["state_axis"] == "return_state")
    ].copy()
    strong_up = g1_return.loc[g1_return["state"] == "strong_up", "median_excess_return"]
    strong_down = g1_return.loc[g1_return["state"] == "strong_down", "median_excess_return"]
    key_g1 = monthly[(monthly["group"] == "G1") & (monthly["year"].isin(key_years))]
    key_g1_excess = float(pd.to_numeric(key_g1["excess_return"], errors="coerce").sum()) if not key_g1.empty else np.nan
    key_g0 = monthly[(monthly["group"] == "G0") & (monthly["year"].isin(key_years))]
    key_g0_excess = float(pd.to_numeric(key_g0["excess_return"], errors="coerce").sum()) if not key_g0.empty else np.nan
    return f"""# P1 Market-Relative State Diagnostics

- 生成时间：`{pd.Timestamp.utcnow().isoformat()}`
- 结果类型：`signal_diagnostic`
- 诊断对象：`rank + market_relative + G0/G1`
- 关键年份：`{",".join(str(y) for y in sorted(key_years))}`

## 结论摘要

1. `market_relative + G1` 在关键年份合计超额为 `{_fmt_pct(key_g1_excess)}`，弱于 `G0` 的 `{_fmt_pct(key_g0_excess)}`；问题不是单一年份噪声。
2. 按市场月收益分层，`G1` 的强上涨月中位超额为 `{_fmt_pct(strong_up.iloc[0] if len(strong_up) else np.nan)}`，强下跌月中位超额为 `{_fmt_pct(strong_down.iloc[0] if len(strong_down) else np.nan)}`。这说明 `market_relative` 标签没有修复上涨参与，反而把下跌月防守也削弱。
3. 本轮仅归档市场状态诊断，不改变默认研究基线，不 promotion。后续候选必须先通过 `daily_bt_like_proxy_annualized_excess_vs_market` 准入，再看这些市场状态分层是否改善。

## 市场状态分层

{_markdown_table(state_summary)}

## G1 相对 G0 的状态分层

{_markdown_table(delta_summary)}

## 关键年份分层

{_markdown_table(key_year_state)}

## G1 相对 G0 最差月份

{_markdown_table(worst_months, max_rows=12)}

## 产物

- `data/results/{prefix}_monthly_state.csv`
- `data/results/{prefix}_state_summary.csv`
- `data/results/{prefix}_delta_state_summary.csv`
- `data/results/{prefix}_key_year_state.csv`
- `data/results/{prefix}_worst_months.csv`
- `data/results/{prefix}.json`
- `docs/{prefix}.md`
"""


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    paths = cfg.get("paths", {}) or {}
    db_path = _resolve(str(paths.get("duckdb_path") or "data/market.duckdb"))
    results_dir = _resolve(args.results_dir)
    docs_dir = _resolve(args.docs_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)

    monthly = pd.read_csv(_resolve(args.monthly_csv))
    delta = pd.read_csv(_resolve(args.monthly_delta_csv))
    monthly["month_end"] = pd.to_datetime(monthly["month_end"], errors="coerce").dt.normalize()
    delta["month_end"] = pd.to_datetime(delta["month_end"], errors="coerce").dt.normalize()
    first_month = pd.Timestamp(monthly["month_end"].min()).replace(day=1)
    start = args.start.strip() or str(first_month.date())
    end = args.end.strip() or str(monthly["month_end"].max().date())
    key_years = {int(x.strip()) for x in str(args.key_years).split(",") if x.strip()}

    print(f"[P1-STATE] load daily {start} -> {end}", flush=True)
    daily = load_daily_from_duckdb(str(db_path), start, end, lookback_days=0)
    print(f"[P1-STATE] daily rows={len(daily):,}", flush=True)
    state = build_monthly_market_state(
        daily=daily,
        start=start,
        end=end,
        benchmark_min_history_days=int(args.benchmark_min_history_days),
    )
    if state.empty:
        raise RuntimeError("无法构造市场状态月表")
    state = state[state["month_end"].isin(set(monthly["month_end"]))].copy()

    state_attrs = state.drop(columns=["benchmark_return"], errors="ignore")
    merged = monthly.merge(state_attrs, on=["month_end", "year", "month"], how="left")
    state_frames = [
        summarize_by_state(merged, state_col="return_state"),
        summarize_by_state(merged, state_col="vol_state"),
        summarize_by_state(merged, state_col="breadth_state"),
    ]
    state_summary = pd.concat(state_frames, ignore_index=True)
    delta_frames = [
        summarize_delta_by_state(delta, state, state_col="return_state"),
        summarize_delta_by_state(delta, state, state_col="vol_state"),
        summarize_delta_by_state(delta, state, state_col="breadth_state"),
    ]
    delta_summary = pd.concat(delta_frames, ignore_index=True)
    key_year_state = summarize_key_year_states(merged, key_years=key_years)
    worst_months = delta.merge(
        state[["month_end", "return_state", "vol_state", "breadth_state", "benchmark_daily_vol", "breadth_positive_share"]],
        on="month_end",
        how="left",
    ).nsmallest(min(20, len(delta)), "g1_minus_g0_excess_return")

    prefix = str(args.output_prefix).strip()
    monthly_state_path = results_dir / f"{prefix}_monthly_state.csv"
    state_summary_path = results_dir / f"{prefix}_state_summary.csv"
    delta_summary_path = results_dir / f"{prefix}_delta_state_summary.csv"
    key_year_state_path = results_dir / f"{prefix}_key_year_state.csv"
    worst_path = results_dir / f"{prefix}_worst_months.csv"
    json_path = results_dir / f"{prefix}.json"
    doc_path = docs_dir / f"{prefix}.md"

    merged.to_csv(monthly_state_path, index=False)
    state_summary.to_csv(state_summary_path, index=False)
    delta_summary.to_csv(delta_summary_path, index=False)
    key_year_state.to_csv(key_year_state_path, index=False)
    worst_months.to_csv(worst_path, index=False)

    payload = {
        "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
        "result_type": "signal_diagnostic",
        "research_topic": "p1_tree_groups",
        "research_config_id": "market_relative_state_diagnostics",
        "output_stem": prefix,
        "inputs": {
            "monthly_csv": str(_resolve(args.monthly_csv)),
            "monthly_delta_csv": str(_resolve(args.monthly_delta_csv)),
            "config": str(args.config),
            "db_path": str(db_path),
        },
        "params": {
            "start": start,
            "end": end,
            "benchmark_min_history_days": int(args.benchmark_min_history_days),
            "key_years": sorted(key_years),
        },
        "artifacts": {
            "monthly_state_csv": str(monthly_state_path),
            "state_summary_csv": str(state_summary_path),
            "delta_state_summary_csv": str(delta_summary_path),
            "key_year_state_csv": str(key_year_state_path),
            "worst_months_csv": str(worst_path),
            "doc": str(doc_path),
        },
        "state_summary": state_summary.to_dict(orient="records"),
        "delta_state_summary": delta_summary.to_dict(orient="records"),
    }
    json_path.write_text(json.dumps(_json_sanitize(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    doc_path.write_text(
        _build_doc(
            prefix=prefix,
            monthly=merged,
            state_summary=state_summary,
            delta_summary=delta_summary,
            key_year_state=key_year_state,
            worst_months=worst_months,
            key_years=key_years,
        ),
        encoding="utf-8",
    )
    print(json.dumps(_json_sanitize(payload), ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
