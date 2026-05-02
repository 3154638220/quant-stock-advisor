"""复核 prefilter / universe 是否可升级为新的 S2 研究基线。"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
import time
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.research_identity import make_research_identity, slugify_token
from scripts.run_backtest_eval import (
    BacktestConfig,
    _attach_pit_fundamentals,
    _rebalance_dates,
    attach_universe_filter,
    build_asset_returns,
    build_market_ew_benchmark,
    build_open_to_open_returns,
    build_regime_weight_overrides,
    build_score,
    build_topk_weights,
    compare_full_vs_slices,
    compute_factors,
    contiguous_time_splits,
    load_config,
    load_daily_from_duckdb,
    normalize_weights,
    resolve_industry_cap_and_map,
    rolling_walk_forward_windows,
    run_backtest,
    transaction_cost_params_from_mapping,
    walk_forward_backtest,
)
from src.backtest.performance_panel import compute_performance_panel
from src.models.experiment import append_experiment_result
from src.models.research_contract import (
    ArtifactRef,
    DataSlice,
    ExperimentResult,
    build_result_id,
    config_snapshot,
    utc_now_iso,
    write_research_manifest,
)


@dataclass(frozen=True)
class Scenario:
    key: str
    label: str
    config_path: str


SCENARIOS: tuple[Scenario, ...] = (
    Scenario("v1", "V1_s2_baseline", "config.yaml.backtest.r1_s2_baseline"),
    Scenario("v2", "V2_s2_prefilter_off", "config.yaml.backtest.r5_s2_prefilter_off"),
    Scenario("v3", "V3_s2_prefilter_off_universe_on", "config.yaml.backtest.r7_s2_prefilter_off_universe_on"),
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="运行 S2 prefilter/universe 复核验证")
    p.add_argument("--start", default="2021-01-01", help="回测起始日期")
    p.add_argument("--end", default="", help="回测结束日期；为空时取今天")
    p.add_argument(
        "--output-prefix",
        default="prefilter_universe_validation_2026-04-19",
        help="输出文件前缀（写入 data/results 与 docs）",
    )
    p.add_argument("--lookback-days", type=int, default=260, help="因子热身回看交易日")
    p.add_argument("--min-hist-days", type=int, default=130, help="最少历史交易日")
    p.add_argument("--wf-train-window", type=int, default=252, help="滚动 WF 训练窗")
    p.add_argument("--wf-test-window", type=int, default=63, help="滚动 WF 测试窗")
    p.add_argument("--wf-step-window", type=int, default=63, help="滚动 WF 步长")
    p.add_argument("--wf-slice-splits", type=int, default=5, help="时间切片折数")
    p.add_argument("--wf-slice-min-train-days", type=int, default=252, help="时间切片最少训练窗")
    p.add_argument("--wf-slice-fixed-window", action="store_true", help="时间切片使用固定训练窗")
    return p.parse_args()


def _json_sanitize(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_sanitize(v) for v in obj]
    if isinstance(obj, tuple):
        return [_json_sanitize(v) for v in obj]
    if isinstance(obj, float):
        if obj != obj or obj in (float("inf"), float("-inf")):
            return None
        return obj
    if isinstance(obj, (np.floating,)):
        return _json_sanitize(float(obj))
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    return obj


def summarize_yearly(daily_returns: pd.Series, benchmark: pd.Series | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    strat = pd.to_numeric(daily_returns, errors="coerce").dropna().sort_index()
    if strat.empty:
        return pd.DataFrame(), {}
    yearly = strat.groupby(strat.index.year).apply(lambda r: float((1.0 + r).prod() - 1.0))
    rows: list[dict[str, Any]] = []
    for y, val in yearly.items():
        bench_val = np.nan
        excess = np.nan
        if benchmark is not None and not benchmark.empty:
            b = pd.to_numeric(benchmark, errors="coerce").dropna().sort_index()
            by = b.groupby(b.index.year).apply(lambda r: float((1.0 + r).prod() - 1.0))
            bench_val = float(by.get(y, np.nan))
            excess = float(val - bench_val) if np.isfinite(bench_val) else np.nan
        rows.append(
            {
                "year": int(y),
                "strategy_return": float(val),
                "benchmark_return": bench_val,
                "excess_return": excess,
            }
        )
    out = pd.DataFrame(rows)
    summary = {
        "positive_years": int((out["strategy_return"] > 0).sum()),
        "negative_years": int((out["strategy_return"] < 0).sum()),
        "median_yearly_return": float(out["strategy_return"].median()),
        "worst_year_return": float(out["strategy_return"].min()),
        "best_year_return": float(out["strategy_return"].max()),
        "median_yearly_excess_return": (
            float(out["excess_return"].dropna().median()) if out["excess_return"].notna().any() else np.nan
        ),
    }
    return out, summary


def summarize_subperiods(daily_returns: pd.Series) -> tuple[pd.DataFrame, dict[str, Any]]:
    strat = pd.to_numeric(daily_returns, errors="coerce").dropna().sort_index()
    if strat.empty:
        return pd.DataFrame(), {}
    periods = [
        ("2021-2022", "2021-01-01", "2022-12-31"),
        ("2023-2024", "2023-01-01", "2024-12-31"),
        ("2025-2026", "2025-01-01", "2026-12-31"),
    ]
    rows: list[dict[str, Any]] = []
    for label, start, end in periods:
        seg = strat[(strat.index >= pd.Timestamp(start)) & (strat.index <= pd.Timestamp(end))]
        if seg.empty:
            continue
        panel = compute_performance_panel(seg.to_numpy(dtype=np.float64), periods_per_year=252.0)
        rows.append(
            {
                "period": label,
                "start": str(pd.Timestamp(start).date()),
                "end": str(pd.Timestamp(end).date()),
                "n_days": int(len(seg)),
                "annualized_return": float(panel.annualized_return),
                "sharpe_ratio": float(panel.sharpe_ratio),
                "max_drawdown": float(panel.max_drawdown),
                "total_return": float(panel.total_return),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out, {}
    summary = {
        "subperiod_count": int(len(out)),
        "min_subperiod_ann_return": float(out["annualized_return"].min()),
        "median_subperiod_ann_return": float(out["annualized_return"].median()),
        "min_subperiod_sharpe": float(out["sharpe_ratio"].min()),
        "max_subperiod_drawdown": float(out["max_drawdown"].max()),
    }
    return out, summary


def _build_doc(summary_df: pd.DataFrame, yearly_df: pd.DataFrame, subperiod_df: pd.DataFrame, output_prefix: str) -> str:
    summary_md = summary_df.to_markdown(index=False)
    yearly_md = yearly_df.to_markdown(index=False) if not yearly_df.empty else "_无年度数据_"
    subperiod_md = subperiod_df.to_markdown(index=False) if not subperiod_df.empty else "_无子区间数据_"
    generated_at = pd.Timestamp.utcnow().isoformat()
    return f"""# S2 Prefilter / Universe 复核

- 生成时间：`{generated_at}`
- 固定口径：`S2 = vol_to_turnover` / `top_k=20` / `M` / `max_turnover=0.3`
- 对照矩阵：当前基线、`prefilter=false`、`prefilter=false + universe=true`

## 汇总

{summary_md}

## 年度表现

{yearly_md}

## 子区间表现

{subperiod_md}

## 本轮产物

- `data/results/{output_prefix}_summary.csv`
- `data/results/{output_prefix}_yearly.csv`
- `data/results/{output_prefix}_subperiods.csv`
- `data/results/{output_prefix}_v*.json`
"""


def main() -> None:
    args = parse_args()
    started_at = time.perf_counter()
    end_date = args.end or pd.Timestamp.today().strftime("%Y-%m-%d")
    output_prefix = str(args.output_prefix).strip()
    results_dir = PROJECT_ROOT / "data/results"
    docs_dir = PROJECT_ROOT / "docs"
    results_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] load base data: start={args.start} end={end_date}", flush=True)
    base_cfg, _ = load_config(SCENARIOS[0].config_path)
    db_path = str(PROJECT_ROOT / str(base_cfg["paths"]["duckdb_path"]))
    daily_df = load_daily_from_duckdb(db_path, args.start, end_date, args.lookback_days)
    print(f"  daily_df={daily_df.shape}", flush=True)
    factors = compute_factors(daily_df, min_hist_days=args.min_hist_days)
    print(f"  factors={factors.shape}", flush=True)
    needed_base_cols = [
        "symbol",
        "trade_date",
        "vol_to_turnover",
        "turnover_roll_mean",
        "price_position",
        "limit_move_hits_5d",
        "log_market_cap",
    ]
    factors = factors[[c for c in needed_base_cols if c in factors.columns]].copy()
    rebalance_dates = set(_rebalance_dates(factors["trade_date"].unique(), "M"))
    factors = factors[factors["trade_date"].isin(rebalance_dates)].copy()
    print(f"  factors@rebalance={factors.shape}", flush=True)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
        factors = _attach_pit_fundamentals(factors, db_path)
    print(f"  factors+pit={factors.shape}", flush=True)
    benchmark_market = build_market_ew_benchmark(daily_df, args.start, end_date, min_days=439)

    summary_rows: list[dict[str, Any]] = []
    yearly_frames: list[pd.DataFrame] = []
    subperiod_frames: list[pd.DataFrame] = []

    for idx, scenario in enumerate(SCENARIOS, start=1):
        print(f"[2/4] scenario {idx}/{len(SCENARIOS)}: {scenario.key} ({scenario.label})", flush=True)
        cfg, config_source = load_config(scenario.config_path)
        signals = cfg.get("signals", {})
        portfolio_cfg = cfg.get("portfolio", {})
        backtest_cfg = cfg.get("backtest", {})
        risk_cfg = cfg.get("risk", {}) or {}
        prefilter_cfg = cfg.get("prefilter", {}) or {}
        regime_cfg = cfg.get("regime", {}) or {}
        uf_cfg = cfg.get("universe_filter", {}) or {}

        ce_weights = normalize_weights(signals.get("composite_extended", {}))
        scenario_factors = attach_universe_filter(
            factors,
            daily_df,
            enabled=bool(uf_cfg.get("enabled", False)),
            min_amount_20d=float(uf_cfg.get("min_amount_20d", 50_000_000)),
            require_roe_ttm_positive=bool(uf_cfg.get("require_roe_ttm_positive", True)),
        )

        regime_overrides: dict[pd.Timestamp, dict[str, float]] = {}
        if bool(regime_cfg.get("enabled", True)):
            regime_overrides, _ = build_regime_weight_overrides(
                scenario_factors,
                daily_df,
                ce_weights,
                benchmark_symbol=str(risk_cfg.get("benchmark_symbol", "market_ew_proxy")),
                regime_cfg_raw=regime_cfg,
                market_ew_min_days=439,
            )

        score_df = build_score(
            scenario_factors,
            ce_weights,
            weights_by_date=regime_overrides if regime_overrides else None,
        )
        print(f"  score_df={score_df.shape}", flush=True)

        industry_cap_count, industry_map, _ = resolve_industry_cap_and_map(
            int(portfolio_cfg.get("industry_cap_count", 5)),
            "data/cache/industry_map.csv",
        )
        weights = build_topk_weights(
            score_df=score_df,
            factor_df=scenario_factors,
            daily_df=daily_df,
            top_k=int(signals.get("top_k", 20)),
            rebalance_rule=str(backtest_cfg.get("eval_rebalance_rule", "M")),
            prefilter_cfg=prefilter_cfg,
            max_turnover=float(portfolio_cfg.get("max_turnover", 1.0)),
            industry_map=industry_map,
            industry_cap_count=industry_cap_count,
            portfolio_method=str(portfolio_cfg.get("weight_method", "equal_weight")),
            cov_lookback_days=int(portfolio_cfg.get("cov_lookback_days", 252)),
            cov_ridge=float(portfolio_cfg.get("cov_ridge", 1e-6)),
            cov_shrinkage=str(portfolio_cfg.get("cov_shrinkage", "ledoit_wolf")).lower(),
            cov_ewma_halflife=float(portfolio_cfg.get("cov_ewma_halflife", 20.0)),
            risk_aversion=float(portfolio_cfg.get("risk_aversion", 1.0)),
        )
        weights = weights[weights.index >= pd.Timestamp(args.start)]
        print(f"  weights={weights.shape}", flush=True)

        target_cols = sorted({str(col).zfill(6) for col in weights.columns})
        execution_mode = str(backtest_cfg.get("execution_mode", "tplus1_open")).lower().strip()
        if execution_mode == "tplus1_open":
            open_returns = build_open_to_open_returns(daily_df, zero_if_limit_up_open=False)
            open_returns = open_returns.sort_index()
            open_returns.index = pd.to_datetime(open_returns.index)
            asset_returns = open_returns[
                (open_returns.index >= pd.Timestamp(args.start)) & (open_returns.index <= pd.Timestamp(end_date))
            ]
        else:
            asset_returns = build_asset_returns(daily_df, target_cols, args.start, end_date)
        asset_returns = asset_returns.reindex(columns=target_cols).fillna(0.0)
        weights = weights.reindex(columns=target_cols, fill_value=0.0)

        first_reb = weights.index.min()
        if not asset_returns.empty and weights.index.min() > asset_returns.index.min():
            seed = weights.iloc[[0]].copy()
            seed.index = pd.DatetimeIndex([asset_returns.index.min()])
            weights = pd.concat([seed, weights], axis=0)
            weights = weights[~weights.index.duplicated(keep="last")].sort_index()
        asset_returns = asset_returns[asset_returns.index >= first_reb]

        costs = transaction_cost_params_from_mapping(cfg.get("transaction_costs", {}))
        bt_no_cost = BacktestConfig(cost_params=None, execution_mode=execution_mode, execution_lag=1)
        bt_cost = BacktestConfig(cost_params=costs, execution_mode=execution_mode, execution_lag=1)
        res_nc = run_backtest(asset_returns, weights, config=bt_no_cost)
        res_wc = run_backtest(asset_returns, weights, config=bt_cost)

        rolling = rolling_walk_forward_windows(
            asset_returns.index,
            train_days=max(20, int(args.wf_train_window)),
            test_days=max(5, int(args.wf_test_window)),
            step_days=max(5, int(args.wf_step_window)),
        )
        _, wf_detail, wf_agg = walk_forward_backtest(asset_returns, weights, rolling, config=bt_cost, use_test_only=True)
        slices = contiguous_time_splits(
            asset_returns.index,
            n_splits=max(2, int(args.wf_slice_splits)),
            min_train_days=max(20, int(args.wf_slice_min_train_days)),
            expanding_window=not bool(args.wf_slice_fixed_window),
        )
        _, sp_detail, sp_agg = walk_forward_backtest(asset_returns, weights, slices, config=bt_cost, use_test_only=True)

        yearly_df, yearly_summary = summarize_yearly(res_wc.daily_returns, benchmark_market)
        if not yearly_df.empty:
            yearly_df.insert(0, "scenario", scenario.label)
            yearly_frames.append(yearly_df)

        subperiod_df, subperiod_summary = summarize_subperiods(res_wc.daily_returns)
        if not subperiod_df.empty:
            subperiod_df.insert(0, "scenario", scenario.label)
            subperiod_frames.append(subperiod_df)

        summary_rows.append(
            {
                "scenario": scenario.label,
                "annualized_return": float(res_wc.panel.annualized_return),
                "sharpe_ratio": float(res_wc.panel.sharpe_ratio),
                "max_drawdown": float(res_wc.panel.max_drawdown),
                "turnover_mean": float(res_wc.panel.turnover_mean),
                "rolling_oos_median_ann_return": float(wf_agg.get("median_ann_return", np.nan)),
                "slice_oos_median_ann_return": float(sp_agg.get("median_ann_return", np.nan)),
                "positive_years": yearly_summary.get("positive_years"),
                "negative_years": yearly_summary.get("negative_years"),
                "median_yearly_return": yearly_summary.get("median_yearly_return"),
                "worst_year_return": yearly_summary.get("worst_year_return"),
                "best_year_return": yearly_summary.get("best_year_return"),
                "median_yearly_excess_return": yearly_summary.get("median_yearly_excess_return"),
                "min_subperiod_ann_return": subperiod_summary.get("min_subperiod_ann_return"),
                "median_subperiod_ann_return": subperiod_summary.get("median_subperiod_ann_return"),
                "min_subperiod_sharpe": subperiod_summary.get("min_subperiod_sharpe"),
                "max_subperiod_drawdown": subperiod_summary.get("max_subperiod_drawdown"),
            }
        )

        payload = {
            "generated_at": pd.Timestamp.utcnow().isoformat(),
            "config_source": config_source,
            "parameters": {
                "start": args.start,
                "end": end_date,
                "top_k": int(signals.get("top_k", 20)),
                "rebalance_rule": str(backtest_cfg.get("eval_rebalance_rule", "M")),
                "max_turnover": float(portfolio_cfg.get("max_turnover", 1.0)),
                "execution_mode": execution_mode,
                "prefilter": prefilter_cfg,
                "universe_filter": uf_cfg,
            },
            "full_sample": {
                "no_cost": _json_sanitize(res_nc.panel.to_dict()),
                "with_cost": _json_sanitize(res_wc.panel.to_dict()),
            },
            "walk_forward_rolling": {
                "detail": _json_sanitize(wf_detail.to_dict(orient="records")),
                "agg": _json_sanitize(wf_agg),
            },
            "walk_forward_slices": {
                "detail": _json_sanitize(sp_detail.to_dict(orient="records")),
                "agg": _json_sanitize(sp_agg),
                "full_vs_slices": _json_sanitize(compare_full_vs_slices(res_wc.panel, sp_agg) if sp_agg else {}),
            },
            "yearly": _json_sanitize(yearly_df.to_dict(orient="records")),
            "yearly_summary": _json_sanitize(yearly_summary),
            "subperiods": _json_sanitize(subperiod_df.to_dict(orient="records")),
            "subperiod_summary": _json_sanitize(subperiod_summary),
        }
        report_path = results_dir / f"{output_prefix}_{scenario.key}.json"
        report_path.write_text(json.dumps(_json_sanitize(payload), ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[ok] {scenario.label}: {report_path.relative_to(PROJECT_ROOT)}", flush=True)

    summary_df = pd.DataFrame(summary_rows).sort_values("scenario").reset_index(drop=True)
    yearly_all_df = pd.concat(yearly_frames, ignore_index=True) if yearly_frames else pd.DataFrame()
    subperiod_all_df = pd.concat(subperiod_frames, ignore_index=True) if subperiod_frames else pd.DataFrame()

    summary_path = results_dir / f"{output_prefix}_summary.csv"
    yearly_path = results_dir / f"{output_prefix}_yearly.csv"
    subperiod_path = results_dir / f"{output_prefix}_subperiods.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    yearly_all_df.to_csv(yearly_path, index=False, encoding="utf-8-sig")
    subperiod_all_df.to_csv(subperiod_path, index=False, encoding="utf-8-sig")

    doc_path = docs_dir / f"{output_prefix}.md"
    doc_path.write_text(_build_doc(summary_df, yearly_all_df, subperiod_all_df, output_prefix), encoding="utf-8")
    print(f"[3/4] summary: {summary_path.relative_to(PROJECT_ROOT)}", flush=True)
    print(f"[3/4] yearly: {yearly_path.relative_to(PROJECT_ROOT)}", flush=True)
    print(f"[3/4] subperiods: {subperiod_path.relative_to(PROJECT_ROOT)}", flush=True)
    print(f"[3/4] doc: {doc_path.relative_to(PROJECT_ROOT)}", flush=True)

    # --- standard research contract ---
    duration_sec = round(time.perf_counter() - started_at, 6)

    def _project_relative(p: str | Path) -> str:
        return str(Path(p).resolve().relative_to(PROJECT_ROOT.resolve()))

    manifest_path = results_dir / f"{output_prefix}_manifest.json"
    scenario_keys = [s.key for s in SCENARIOS]
    identity = make_research_identity(
        result_type="prefilter_universe_validation",
        research_topic="prefilter_universe_validation",
        research_config_id=f"scenarios_{slugify_token('-'.join(scenario_keys[:4]))}_topk_{int(base_cfg.get('signals', {}).get('top_k', 20))}",
        output_stem=slugify_token(output_prefix),
    )
    data_slice = DataSlice(
        dataset_name="prefilter_universe_validation_backtest",
        source_tables=("a_share_daily",),
        date_start=args.start,
        date_end=end_date,
        asof_trade_date=end_date,
        signal_date_col="trade_date",
        symbol_col="symbol",
        candidate_pool_version="universe_filtered",
        rebalance_rule="M",
        execution_mode="tplus1_open",
        label_return_mode="open_to_open",
        feature_set_id=None,
        feature_columns=(),
        label_columns=(),
        pit_policy="signal_date_close_visible_only",
        config_path=str(SCENARIOS[0].config_path),
        extra={
            "scenario_keys": scenario_keys,
            "lookback_days": int(args.lookback_days),
            "min_hist_days": int(args.min_hist_days),
        },
    )
    artifact_refs = (
        ArtifactRef("summary_csv", _project_relative(summary_path), "csv", False, "场景对比汇总"),
        ArtifactRef("yearly_csv", _project_relative(yearly_path), "csv", False, "年度切片"),
        ArtifactRef("subperiods_csv", _project_relative(subperiod_path), "csv", False, "子时段切片"),
        ArtifactRef("report_md", _project_relative(doc_path), "md", False, "验证报告"),
        ArtifactRef("manifest_json", _project_relative(manifest_path), "json", False),
    ) + tuple(
        ArtifactRef(
            f"scenario_{slugify_token(s.key)}_json",
            _project_relative(results_dir / f"{output_prefix}_{s.key}.json"),
            "json",
            False,
            f"场景 {s.label} 回测详情",
        )
        for s in SCENARIOS
    )
    metrics = {
        "scenario_count": len(SCENARIOS),
        "summary_rows": int(len(summary_df)),
    }
    gates = {
        "data_gate": {
            "passed": bool(daily_df is not None and len(daily_df) > 0),
            "daily_rows": int(len(daily_df)),
        },
        "execution_gate": {
            "passed": bool(len(summary_rows) == len(SCENARIOS)),
            "expected": len(SCENARIOS),
            "completed": len(summary_rows),
        },
        "governance_gate": {
            "passed": True,
            "manifest_schema": "research_result_v1",
        },
    }
    config_info = config_snapshot(
        config_path=Path(str(SCENARIOS[0].config_path)) if not str(SCENARIOS[0].config_path).startswith("/") else Path(str(SCENARIOS[0].config_path)),
        resolved_config=None,
        sections=(),
    )
    config_info["config_path"] = str(SCENARIOS[0].config_path)
    result = ExperimentResult(
        result_id=build_result_id(identity, [data_slice], metrics),
        identity=identity,
        script_name=_project_relative(Path(__file__).resolve()),
        command=shlex.join([sys.executable, *sys.argv]),
        created_at=utc_now_iso(),
        duration_sec=duration_sec,
        seed=None,
        data_slices=(data_slice,),
        config=config_info,
        params={
            "cli": {k: str(v) for k, v in vars(args).items()},
            "scenarios": [s.__dict__ for s in SCENARIOS],
        },
        metrics=metrics,
        gates=gates,
        artifacts=artifact_refs,
        promotion={
            "production_eligible": False,
            "registry_status": "not_registered",
            "blocking_reasons": ["prefilter_universe_validation_is_diagnostic_research_only"],
        },
        notes=f"Prefilter/universe validation: {len(SCENARIOS)} scenarios compared.",
    )
    write_research_manifest(
        manifest_path,
        result,
        extra={
            "generated_at_utc": result.created_at,
            "scenarios": [s.__dict__ for s in SCENARIOS],
            "legacy_artifacts": [
                _project_relative(summary_path),
                _project_relative(yearly_path),
                _project_relative(subperiod_path),
                _project_relative(doc_path),
            ]
            + [_project_relative(results_dir / f"{output_prefix}_{s.key}.json") for s in SCENARIOS],
        },
    )
    append_experiment_result(results_dir.parent / "experiments", result)
    # --- end standard research contract ---

    print(f"[4/4] manifest: {manifest_path.relative_to(PROJECT_ROOT)}", flush=True)


if __name__ == "__main__":
    main()
