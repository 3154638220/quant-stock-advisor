#!/usr/bin/env python3
"""P1 树模型 G0/G1 失效诊断：年份、上涨月捕获、暴露与 Top-K 机会损失。"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_backtest_eval import (  # noqa: E402
    BacktestConfig,
    build_market_ew_benchmark,
    build_open_to_open_returns,
    build_score,
    build_topk_weights,
    load_config,
    load_daily_from_duckdb,
    transaction_cost_params_from_mapping,
)
from scripts.run_benchmark_gap_diagnostics import (  # noqa: E402
    _prepare_exposure_frame,
    build_rank_coverage_tables,
    summarize_exposures,
    summarize_market_capture,
    summarize_monthly_excess,
)
from src.backtest.engine import run_backtest  # noqa: E402


EXPOSURE_FEATURES: tuple[str, ...] = (
    "log_market_cap",
    "amount_20d",
    "turnover_roll_mean",
    "realized_vol",
    "recent_return",
    "momentum_12_1",
    "price_position",
    "vol_to_turnover",
)


@dataclass(frozen=True)
class P1GroupSpec:
    group: str
    result_json: Path
    bundle_dir: str
    tree_features: list[str]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="运行 P1 G0/G1 树模型失效诊断")
    p.add_argument(
        "--g0-json",
        default="data/results/p1_full_backtest_g0_rank_direction_20260426.json",
        help="G0 full backtest JSON",
    )
    p.add_argument(
        "--g1-json",
        default="data/results/p1_full_backtest_g1_rankfix_same_window_20260426.json",
        help="G1 full backtest JSON",
    )
    p.add_argument("--output-prefix", default="p1_failure_diagnostics_2026-04-26")
    p.add_argument("--docs-dir", default="docs")
    p.add_argument("--results-dir", default="data/results")
    return p.parse_args()


def _resolve_path(path_like: str | Path) -> Path:
    p = Path(path_like).expanduser()
    return p if p.is_absolute() else PROJECT_ROOT / p


def _read_payload(path_like: str | Path) -> dict[str, Any]:
    path = _resolve_path(path_like)
    return json.loads(path.read_text(encoding="utf-8"))


def _tree_features_from_payload(payload: dict[str, Any]) -> list[str]:
    raw = (payload.get("parameters") or {}).get("tree_features") or []
    if isinstance(raw, str):
        return [x.strip() for x in raw.split(",") if x.strip()]
    return [str(x).strip() for x in raw if str(x).strip()]


def _group_spec(group: str, json_path: str | Path) -> P1GroupSpec:
    payload = _read_payload(json_path)
    params = payload.get("parameters") or {}
    meta = payload.get("meta") or {}
    tree_model = meta.get("tree_model") or {}
    return P1GroupSpec(
        group=group,
        result_json=_resolve_path(json_path),
        bundle_dir=str(tree_model.get("bundle_dir") or params.get("tree_bundle_dir") or ""),
        tree_features=_tree_features_from_payload(payload),
    )


def _compound_return(values: pd.Series) -> float:
    arr = pd.to_numeric(values, errors="coerce").dropna()
    if arr.empty:
        return float("nan")
    return float((1.0 + arr).prod() - 1.0)


def summarize_key_year_months(monthly: pd.DataFrame, *, key_years: tuple[int, ...]) -> pd.DataFrame:
    """按年份汇总调仓月/月度超额贡献，突出关键落后年份。"""
    if monthly.empty:
        return pd.DataFrame()
    df = monthly.copy()
    rows: list[dict[str, Any]] = []
    for year, part in df.groupby("year", sort=True):
        part = part.sort_values("month_end")
        worst = part.nsmallest(min(3, len(part)), "excess_return")
        up = part[part["benchmark_return"] > 0]
        rows.append(
            {
                "year": int(year),
                "is_key_year": int(int(year) in set(key_years)),
                "months": int(len(part)),
                "strategy_compound_return": _compound_return(part["strategy_return"]),
                "benchmark_compound_return": _compound_return(part["benchmark_return"]),
                "excess_sum": float(pd.to_numeric(part["excess_return"], errors="coerce").sum()),
                "median_monthly_excess": float(pd.to_numeric(part["excess_return"], errors="coerce").median()),
                "benchmark_up_months": int(len(up)),
                "benchmark_up_positive_excess_share": (
                    float((pd.to_numeric(up["excess_return"], errors="coerce") > 0).mean()) if len(up) else np.nan
                ),
                "worst_months": ",".join(pd.to_datetime(worst["month_end"]).dt.strftime("%Y-%m").tolist()),
                "worst_months_excess_sum": float(pd.to_numeric(worst["excess_return"], errors="coerce").sum()),
            }
        )
    return pd.DataFrame(rows)


def summarize_monthly_delta(g0_monthly: pd.DataFrame, g1_monthly: pd.DataFrame) -> pd.DataFrame:
    """比较 G1 相对 G0 的月度退化/改善来源。"""
    cols = ["month_end", "year", "month", "strategy_return", "benchmark_return", "excess_return"]
    left = g0_monthly[cols].rename(
        columns={"strategy_return": "g0_strategy_return", "excess_return": "g0_excess_return"}
    )
    right = g1_monthly[cols].rename(
        columns={"strategy_return": "g1_strategy_return", "excess_return": "g1_excess_return"}
    )
    out = left.merge(right, on=["month_end", "year", "month", "benchmark_return"], how="inner")
    out["g1_minus_g0_strategy_return"] = out["g1_strategy_return"] - out["g0_strategy_return"]
    out["g1_minus_g0_excess_return"] = out["g1_excess_return"] - out["g0_excess_return"]
    return out.sort_values("month_end").reset_index(drop=True)


def summarize_selection_overlap(
    weights_by_group: dict[str, pd.DataFrame],
    asset_returns: pd.DataFrame,
) -> pd.DataFrame:
    """拆 G1 相对 G0 的换股收益：共同持仓、G1 独有、G0 独有。"""
    g0 = weights_by_group["G0"]
    g1 = weights_by_group["G1"]
    dates = sorted(set(g0.index) & set(g1.index))
    rows: list[dict[str, Any]] = []
    for i, dt in enumerate(dates[:-1]):
        next_dt = pd.Timestamp(dates[i + 1])
        dt = pd.Timestamp(dt)
        r0 = pd.to_numeric(g0.loc[dt], errors="coerce")
        r1 = pd.to_numeric(g1.loc[dt], errors="coerce")
        s0 = set(r0[r0 > 0].index.astype(str))
        s1 = set(r1[r1 > 0].index.astype(str))
        if not s0 or not s1:
            continue
        period = asset_returns[(asset_returns.index > dt) & (asset_returns.index <= next_dt)]
        if period.empty:
            continue
        fwd = (1.0 + period).prod(axis=0) - 1.0

        def _mean_for(symbols: set[str]) -> float:
            vals = pd.to_numeric(fwd.reindex(sorted(symbols)), errors="coerce").dropna()
            return float(vals.mean()) if not vals.empty else np.nan

        common = s0 & s1
        g1_only = s1 - s0
        g0_only = s0 - s1
        rows.append(
            {
                "trade_date": dt,
                "year": int(dt.year),
                "next_rebalance_date": next_dt,
                "g0_count": int(len(s0)),
                "g1_count": int(len(s1)),
                "overlap_count": int(len(common)),
                "overlap_ratio": float(len(common) / max(len(s0 | s1), 1)),
                "common_forward_return": _mean_for(common),
                "g1_only_count": int(len(g1_only)),
                "g1_only_forward_return": _mean_for(g1_only),
                "g0_only_count": int(len(g0_only)),
                "g0_only_forward_return": _mean_for(g0_only),
                "g1_only_minus_g0_only_forward_return": _mean_for(g1_only) - _mean_for(g0_only),
            }
        )
    return pd.DataFrame(rows)


def _params_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    params = payload.get("parameters") or {}
    return {
        "start": str(params.get("start", "2021-01-01")),
        "end": str(params.get("end", "")),
        "config": str(payload.get("config_source") or params.get("config_source") or "config.yaml.backtest"),
        "lookback_days": int(params.get("lookback_days", 260)),
        "min_hist_days": int(params.get("min_hist_days", 130)),
        "rebalance_rule": str(params.get("rebalance_rule", "M")),
        "top_k": int(params.get("top_k", 20)),
        "max_turnover": float(params.get("max_turnover", 1.0)),
        "portfolio_method": str(params.get("portfolio_method", "equal_weight")),
        "prepared_factors_cache": str(params.get("prepared_factors_cache") or ""),
        "benchmark_min_history_days": int(params.get("benchmark_min_history_days", 130)),
    }


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


def _fmt_pct(value: Any) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if not np.isfinite(v):
        return "N/A"
    return f"{v:.2%}"


def _markdown_table(df: pd.DataFrame, *, max_rows: int | None = None) -> str:
    """不依赖 pandas tabulate 的小型 Markdown 表格渲染。"""
    if df.empty:
        return "_无数据_"
    view = df.head(int(max_rows)).copy() if max_rows is not None else df.copy()

    def _cell(value: Any) -> str:
        if isinstance(value, pd.Timestamp):
            return value.strftime("%Y-%m-%d")
        if isinstance(value, (np.floating, float)):
            v = float(value)
            if not np.isfinite(v):
                return ""
            return f"{v:.6g}"
        if isinstance(value, (np.integer, int)):
            return str(int(value))
        text = "" if value is None else str(value)
        return text.replace("|", "\\|").replace("\n", " ")

    cols = [str(c) for c in view.columns]
    rows = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in view.iterrows():
        rows.append("| " + " | ".join(_cell(row[c]) for c in view.columns) + " |")
    if max_rows is not None and len(df) > max_rows:
        rows.append(f"\n_仅展示前 {max_rows} 行，共 {len(df)} 行。_")
    return "\n".join(rows)


def _build_doc(
    *,
    output_prefix: str,
    params: dict[str, Any],
    group_yearly: dict[str, pd.DataFrame],
    group_capture: dict[str, pd.DataFrame],
    group_exposure: dict[str, pd.DataFrame],
    group_rank: dict[str, pd.DataFrame],
    monthly_delta: pd.DataFrame,
    overlap: pd.DataFrame,
) -> str:
    generated_at = pd.Timestamp.utcnow().isoformat()
    yearly_md = "\n\n".join(f"### {group}\n\n{_markdown_table(df)}" for group, df in group_yearly.items() if not df.empty)
    capture_md = "\n\n".join(f"### {group}\n\n{_markdown_table(df)}" for group, df in group_capture.items() if not df.empty)
    exposure_md = "\n\n".join(
        f"### {group}\n\n{_markdown_table(df)}" for group, df in group_exposure.items() if not df.empty
    )
    rank_md = "\n\n".join(f"### {group}\n\n{_markdown_table(df)}" for group, df in group_rank.items() if not df.empty)
    delta_2022 = monthly_delta[monthly_delta["year"] == 2022].copy()
    delta_2022_sum = (
        float(pd.to_numeric(delta_2022["g1_minus_g0_excess_return"], errors="coerce").sum())
        if not delta_2022.empty
        else np.nan
    )
    worst_delta = monthly_delta.nsmallest(min(8, len(monthly_delta)), "g1_minus_g0_excess_return")
    worst_delta_md = (
        _markdown_table(
            worst_delta[["month_end", "g0_excess_return", "g1_excess_return", "g1_minus_g0_excess_return"]]
        )
        if not worst_delta.empty
        else "_无月度差异数据_"
    )
    overlap_summary = (
        overlap.groupby("year")
        .agg(
            rebalances=("trade_date", "count"),
            mean_overlap_ratio=("overlap_ratio", "mean"),
            mean_g1_only_minus_g0_only_forward_return=("g1_only_minus_g0_only_forward_return", "mean"),
        )
        .reset_index()
        if not overlap.empty
        else pd.DataFrame()
    )
    overlap_md = _markdown_table(overlap_summary) if not overlap_summary.empty else "_无换股收益数据_"

    g0_key = group_yearly.get("G0", pd.DataFrame())
    g1_key = group_yearly.get("G1", pd.DataFrame())
    g0_2021 = g0_key.loc[g0_key["year"] == 2021, "excess_sum"].iloc[0] if not g0_key.empty and (g0_key["year"] == 2021).any() else np.nan
    g1_2021 = g1_key.loc[g1_key["year"] == 2021, "excess_sum"].iloc[0] if not g1_key.empty and (g1_key["year"] == 2021).any() else np.nan

    return f"""# P1 G0/G1 Failure Diagnostics

- 生成时间：`{generated_at}`
- 固定口径：`sort_by=xgboost` / `top_k={params["top_k"]}` / `{params["rebalance_rule"]}` / `{params["portfolio_method"]}` / `tplus1_open`
- 回测区间：`{params["start"]}` ~ `{params["end"]}`
- 结果类型：`signal_diagnostic`，仅解释 P1 失效机制，不 promotion

## 结论摘要

1. `G0/G1` 的关键落后不是单个特征组偶然失效，`2021` 两组都大幅跑输：`G0={_fmt_pct(g0_2021)}`，`G1={_fmt_pct(g1_2021)}`。
2. `G1` 在 `2022` 相对 `G0` 的月度超额合计变化为 `{_fmt_pct(delta_2022_sum)}`；这解释了 full backtest 中 weekly KDJ 增量没有兑现。
3. 换股诊断显示 `G1-only` 相对 `G0-only` 的前向收益按年并不稳定，当前更像改变排序边界，而不是稳定补足上涨参与。
4. 本轮不改变主计划：继续先解释标签/目标/市场状态，不扩 `weekly_kdj` interaction 网格。

## 年份失效

{yearly_md or "_无年度诊断数据_"}

## 上涨月捕获率

{capture_md or "_无上涨月捕获数据_"}

## G1 vs G0 月度退化

{worst_delta_md}

## G1/G0 换股收益

{overlap_md}

## 持仓暴露 vs 基准

{exposure_md or "_无暴露数据_"}

## Top-K 与 21-40 桶

{rank_md or "_无排名桶数据_"}

## 本轮产物

- `data/results/{output_prefix}_summary.json`
- `data/results/{output_prefix}_monthly.csv`
- `data/results/{output_prefix}_yearly.csv`
- `data/results/{output_prefix}_capture.csv`
- `data/results/{output_prefix}_exposure_summary.csv`
- `data/results/{output_prefix}_rank_bucket_summary.csv`
- `data/results/{output_prefix}_selection_overlap.csv`
"""


def main() -> None:
    args = parse_args()
    g0_payload = _read_payload(args.g0_json)
    g1_payload = _read_payload(args.g1_json)
    params = _params_from_payload(g0_payload)
    if not params["end"]:
        params["end"] = str(pd.Timestamp.today().date())

    cfg, _ = load_config(params["config"])
    db_path = str(PROJECT_ROOT / cfg["paths"]["duckdb_path"])
    costs = transaction_cost_params_from_mapping(cfg.get("transaction_costs", {}))
    backtest_cfg = cfg.get("backtest", {}) or {}
    prefilter_cfg = cfg.get("prefilter", {}) or {}

    results_dir = _resolve_path(args.results_dir)
    docs_dir = _resolve_path(args.docs_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)

    print("[1/5] load daily + prepared factors", flush=True)
    daily_df = load_daily_from_duckdb(db_path, params["start"], params["end"], params["lookback_days"])
    cache_path = _resolve_path(params["prepared_factors_cache"])
    factors = pd.read_parquet(cache_path)
    factors["trade_date"] = pd.to_datetime(factors["trade_date"], errors="coerce").dt.normalize()
    factors["symbol"] = factors["symbol"].astype(str).str.zfill(6)
    factors = factors[
        (factors["trade_date"] >= pd.Timestamp(params["start"]))
        & (factors["trade_date"] <= pd.Timestamp(params["end"]))
    ].copy()

    print("[2/5] score and build weights", flush=True)
    specs = [_group_spec("G0", args.g0_json), _group_spec("G1", args.g1_json)]
    score_by_group: dict[str, pd.DataFrame] = {}
    weights_by_group: dict[str, pd.DataFrame] = {}
    for spec in specs:
        score = build_score(
            factors,
            {},
            sort_by="xgboost",
            tree_bundle_dir=str(_resolve_path(spec.bundle_dir)),
            tree_raw_features=spec.tree_features,
            tree_rsi_mode=str((g0_payload.get("parameters") or {}).get("tree_rsi_mode") or "level"),
        )
        weights = build_topk_weights(
            score_df=score,
            factor_df=factors,
            daily_df=daily_df,
            top_k=params["top_k"],
            rebalance_rule=params["rebalance_rule"],
            prefilter_cfg=prefilter_cfg,
            max_turnover=params["max_turnover"],
            portfolio_method=params["portfolio_method"],
        )
        weights = weights[weights.index >= pd.Timestamp(params["start"])]
        score_by_group[spec.group] = score
        weights_by_group[spec.group] = weights

    print("[3/5] run diagnostic backtests", flush=True)
    all_cols = sorted(set().union(*(set(w.columns.astype(str)) for w in weights_by_group.values())))
    open_returns = build_open_to_open_returns(daily_df, zero_if_limit_up_open=False)
    asset_returns = open_returns.reindex(columns=all_cols).fillna(0.0)
    asset_returns = asset_returns[
        (asset_returns.index >= pd.Timestamp(params["start"])) & (asset_returns.index <= pd.Timestamp(params["end"]))
    ]
    first_reb = min(w.index.min() for w in weights_by_group.values())
    asset_returns = asset_returns[asset_returns.index >= first_reb]
    bt_cost = BacktestConfig(
        cost_params=costs,
        execution_mode="tplus1_open",
        execution_lag=int(backtest_cfg.get("execution_lag", 1)),
        limit_up_mode=str(backtest_cfg.get("limit_up_mode", "idle")),
    )
    benchmark = build_market_ew_benchmark(
        daily_df,
        params["start"],
        params["end"],
        min_days=params["benchmark_min_history_days"],
    )

    monthly_frames: list[pd.DataFrame] = []
    yearly_frames: list[pd.DataFrame] = []
    capture_frames: list[pd.DataFrame] = []
    exposure_frames: list[pd.DataFrame] = []
    rank_frames: list[pd.DataFrame] = []
    group_yearly: dict[str, pd.DataFrame] = {}
    group_capture: dict[str, pd.DataFrame] = {}
    group_exposure: dict[str, pd.DataFrame] = {}
    group_rank: dict[str, pd.DataFrame] = {}
    group_monthly: dict[str, pd.DataFrame] = {}

    exposure_df = _prepare_exposure_frame(daily_df, factors)
    benchmark_symbols = set(
        daily_df.groupby("symbol")["trade_date"].count().loc[
            lambda s: s >= params["benchmark_min_history_days"]
        ].index.astype(str)
    )
    if not benchmark_symbols:
        benchmark_symbols = set(daily_df["symbol"].astype(str).unique())

    for group, weights in weights_by_group.items():
        w = weights.reindex(columns=all_cols, fill_value=0.0)
        if w.index.min() > asset_returns.index.min():
            seed = w.iloc[[0]].copy()
            seed.index = pd.DatetimeIndex([asset_returns.index.min()])
            w = pd.concat([seed, w], axis=0).sort_index()
        res = run_backtest(asset_returns, w, config=bt_cost)
        monthly, _ = summarize_monthly_excess(res.daily_returns, benchmark)
        monthly["group"] = group
        yearly = summarize_key_year_months(monthly, key_years=(2021, 2025, 2026))
        yearly["group"] = group
        capture = summarize_market_capture(monthly)
        capture["group"] = group
        _, exposure_summary = summarize_exposures(weights, exposure_df, benchmark_symbols)
        exposure_summary["group"] = group
        _, rank_summary, _, _ = build_rank_coverage_tables(score_by_group[group], asset_returns, params["rebalance_rule"])
        rank_summary["group"] = group

        monthly_frames.append(monthly)
        yearly_frames.append(yearly)
        capture_frames.append(capture)
        exposure_frames.append(exposure_summary)
        rank_frames.append(rank_summary)
        group_yearly[group] = yearly.drop(columns=["group"], errors="ignore")
        group_capture[group] = capture.drop(columns=["group"], errors="ignore")
        group_exposure[group] = exposure_summary.drop(columns=["group"], errors="ignore")
        group_rank[group] = rank_summary.drop(columns=["group"], errors="ignore")
        group_monthly[group] = monthly.drop(columns=["group"], errors="ignore")

    print("[4/5] compare G1 vs G0", flush=True)
    monthly_all = pd.concat(monthly_frames, ignore_index=True)
    yearly_all = pd.concat(yearly_frames, ignore_index=True)
    capture_all = pd.concat(capture_frames, ignore_index=True)
    exposure_all = pd.concat(exposure_frames, ignore_index=True)
    rank_all = pd.concat(rank_frames, ignore_index=True)
    monthly_delta = summarize_monthly_delta(group_monthly["G0"], group_monthly["G1"])
    monthly_delta["group"] = "G1_minus_G0"
    overlap = summarize_selection_overlap(weights_by_group, asset_returns)

    prefix = str(args.output_prefix).strip()
    monthly_path = results_dir / f"{prefix}_monthly.csv"
    yearly_path = results_dir / f"{prefix}_yearly.csv"
    capture_path = results_dir / f"{prefix}_capture.csv"
    exposure_path = results_dir / f"{prefix}_exposure_summary.csv"
    rank_path = results_dir / f"{prefix}_rank_bucket_summary.csv"
    overlap_path = results_dir / f"{prefix}_selection_overlap.csv"
    delta_path = results_dir / f"{prefix}_monthly_delta.csv"
    summary_path = results_dir / f"{prefix}_summary.json"
    doc_path = docs_dir / f"{prefix}.md"

    monthly_all.to_csv(monthly_path, index=False)
    yearly_all.to_csv(yearly_path, index=False)
    capture_all.to_csv(capture_path, index=False)
    exposure_all.to_csv(exposure_path, index=False)
    rank_all.to_csv(rank_path, index=False)
    overlap.to_csv(overlap_path, index=False)
    monthly_delta.to_csv(delta_path, index=False)

    payload = {
        "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
        "result_type": "signal_diagnostic",
        "research_topic": "p1_tree_groups",
        "research_config_id": str(g0_payload.get("research_config_id") or ""),
        "output_stem": prefix,
        "groups": [
            {
                "group": spec.group,
                "source_json": str(spec.result_json),
                "bundle_dir": spec.bundle_dir,
                "feature_count": len(spec.tree_features),
            }
            for spec in specs
        ],
        "artifacts": {
            "monthly_csv": str(monthly_path),
            "yearly_csv": str(yearly_path),
            "capture_csv": str(capture_path),
            "exposure_summary_csv": str(exposure_path),
            "rank_bucket_summary_csv": str(rank_path),
            "selection_overlap_csv": str(overlap_path),
            "monthly_delta_csv": str(delta_path),
            "doc": str(doc_path),
        },
        "params": params,
    }
    summary_path.write_text(json.dumps(_json_sanitize(payload), ensure_ascii=False, indent=2), encoding="utf-8")

    doc = _build_doc(
        output_prefix=prefix,
        params=params,
        group_yearly=group_yearly,
        group_capture=group_capture,
        group_exposure=group_exposure,
        group_rank=group_rank,
        monthly_delta=monthly_delta,
        overlap=overlap,
    )
    doc_path.write_text(doc, encoding="utf-8")
    print("[5/5] done", flush=True)
    print(json.dumps(_json_sanitize(payload), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
