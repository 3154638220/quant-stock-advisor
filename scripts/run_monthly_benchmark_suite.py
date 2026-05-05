#!/usr/bin/env python3
"""Build the promoted monthly-selection benchmark suite.

将核心逻辑提取到 src/analysis/benchmark_suite.py，本脚本只做 CLI 解析 + 文件 I/O 编排。
"""

from __future__ import annotations

import argparse
import shlex
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.research_identity import make_research_identity, slugify_token
from src.analysis.benchmark_suite import (
    BenchmarkSpec,
    build_benchmark_doc,
    build_benchmark_suite,
    build_capacity_analysis,
    build_cost_sensitivity,
    build_limit_up_stress_comparison,
    load_promoted_monthly,
    parse_index_specs,
)
from src.models.experiment import append_experiment_result
from src.models.research_contract import (
    ArtifactRef, DataSlice, ExperimentResult,
    build_result_id, utc_now_iso, write_research_manifest,
)

DEFAULT_MODEL = "M8_regime_aware_fixed_policy__indcap3"
DEFAULT_POOL = "U1_liquid_tradable"
DEFAULT_TOP_K = 20


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="生成月度选股多基准对比报告")
    p.add_argument("--monthly-long", type=str,
                   default="data/results/monthly_selection_m8_concentration_regime_2026-05-01_monthly_long.csv")
    p.add_argument("--model", type=str, default=DEFAULT_MODEL)
    p.add_argument("--candidate-pool", type=str, default=DEFAULT_POOL)
    p.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    p.add_argument("--as-of-date", type=str, default="")
    p.add_argument("--results-dir", type=str, default="data/results")
    p.add_argument("--output-prefix", type=str, default="monthly_selection_benchmark_suite")
    p.add_argument("--index-csv", action="append", default=[],
                   help="可选指数日线 CSV，格式 name=path 或 name:symbol=path。")
    p.add_argument("--db-path", type=str, default="")
    p.add_argument("--daily-table", type=str, default="a_share_daily")
    p.add_argument("--stress-monthly-long", type=str, default="")
    p.add_argument("--stress-label", type=str, default="stress")
    p.add_argument("--multi-cost-monthly-long", action="append", default=[],
                   help="多成本档 monthly_long，格式 cost_bps=path。")
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
    index_specs = parse_index_specs(args.index_csv, ROOT)
    summary, relative, series, index_meta = build_benchmark_suite(monthly, index_specs)

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
                cost_monthly = load_promoted_monthly(cost_path, model=args.model, pool=args.candidate_pool, top_k=int(args.top_k))
                multi_cost_monthly[bps] = cost_monthly
            except Exception as exc:
                print(f"[benchmark] 加载失败 cost_bps={bps}: {exc}", file=sys.stderr)

    cost_sensitivity = build_cost_sensitivity(monthly, base_cost_bps=10.0,
                                               multi_cost_monthly=multi_cost_monthly if multi_cost_monthly else None)
    capacity = build_capacity_analysis(monthly, db_path=args.db_path, daily_table=args.daily_table, top_k=int(args.top_k))
    limit_up_vwap_comparison = pd.DataFrame()
    if args.stress_monthly_long:
        stress_path = _resolve_project_path(args.stress_monthly_long)
        if stress_path.exists():
            try:
                stress_monthly = load_promoted_monthly(stress_path, model=args.model, pool=args.candidate_pool, top_k=int(args.top_k))
                limit_up_vwap_comparison = build_limit_up_stress_comparison(
                    monthly, stress_monthly, base_label="baseline", stress_label=args.stress_label)
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
    artifact_paths = [_project_relative(p) for p in paths.values()]
    paths["doc"].write_text(
        build_benchmark_doc(
            monthly_path=monthly_path, model=args.model, pool=args.candidate_pool,
            top_k=int(args.top_k), summary=summary, relative=relative,
            index_meta=index_meta, cost_sensitivity=cost_sensitivity,
            capacity=capacity, limit_up_stress=limit_up_vwap_comparison,
            artifacts=artifact_paths, project_root=ROOT,
        ), encoding="utf-8",
    )

    # --- research contract ---
    identity = make_research_identity(
        result_type="monthly_selection_benchmark_suite",
        research_topic="monthly_selection_benchmark_suite",
        research_config_id=f"model_{slugify_token(args.model)}_pool_{slugify_token(args.candidate_pool)}_topk_{int(args.top_k)}",
        output_stem=output_stem, parent_result_id="",
    )
    signal_dates = pd.to_datetime(monthly["signal_date"], errors="coerce").dropna()
    data_slice = DataSlice(
        dataset_name="monthly_selection_monthly_long",
        source_tables=("monthly_selection_monthly_long",),
        date_start=str(signal_dates.min().date()) if not signal_dates.empty else "",
        date_end=str(signal_dates.max().date()) if not signal_dates.empty else "",
        asof_trade_date=as_of,
        signal_date_col="signal_date", symbol_col="symbol",
        candidate_pool_version=args.candidate_pool,
        rebalance_rule="M", execution_mode="tplus1_open",
        label_return_mode="open_to_open",
        feature_set_id=args.model, feature_columns=(), label_columns=(),
        pit_policy="derived_from_monthly_selection_research_outputs",
        config_path=None,
        extra={"monthly_long_path": _project_relative(monthly_path), "top_k": int(args.top_k), "index_input_count": len(index_specs)},
    )
    artifact_refs = (
        ArtifactRef("summary_csv", _project_relative(paths["summary"]), "csv"),
        ArtifactRef("relative_csv", _project_relative(paths["relative"]), "csv"),
        ArtifactRef("monthly_series_csv", _project_relative(paths["series"]), "csv"),
        ArtifactRef("cost_sensitivity_csv", _project_relative(paths["cost_sensitivity"]), "csv"),
        ArtifactRef("capacity_csv", _project_relative(paths["capacity"]), "csv"),
        ArtifactRef("limit_up_stress_csv", _project_relative(paths["limit_up_stress"]), "csv"),
        ArtifactRef("report_md", _project_relative(paths["doc"]), "md"),
        ArtifactRef("manifest_json", _project_relative(paths["manifest"]), "json"),
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
        "data_gate": {"passed": bool(len(monthly) > 0 and signal_dates.notna().any()),
                      "monthly_rows": int(len(monthly)), "date_start": data_slice.date_start, "date_end": data_slice.date_end},
        "benchmark_gate": {"passed": bool({"u1_candidate_pool_ew", "all_a_market_ew"}.issubset(set(summary["benchmark"].astype(str)))),
                           "index_input_count": len(index_specs)},
        "governance_gate": {"passed": True, "manifest_schema": "research_result_v1"},
    }
    result = ExperimentResult(
        result_id=build_result_id(identity, [data_slice], metrics),
        identity=identity, script_name=_project_relative(Path(__file__).resolve()),
        command=shlex.join([sys.executable, *sys.argv]),
        created_at=utc_now_iso(), duration_sec=round(time.perf_counter() - started_at, 6),
        seed=None, data_slices=(data_slice,),
        config={"config_path": "", "config_hash": None},
        params={"cli": {k: str(v) for k, v in vars(args).items()}},
        metrics=metrics, gates=gates, artifacts=artifact_refs,
        promotion={"production_eligible": False, "registry_status": "not_registered",
                   "blocking_reasons": ["benchmark_suite_is_diagnostic_research_only"]},
        notes="Monthly selection benchmark comparison; does not change promoted config.",
    )
    write_research_manifest(paths["manifest"], result, extra={
        "generated_at_utc": result.created_at, "legacy_result_type": "monthly_selection_benchmark_suite",
        "legacy_artifacts": artifact_paths, "monthly_long": _project_relative(monthly_path),
        "model": args.model, "candidate_pool": args.candidate_pool, "top_k": int(args.top_k),
    })
    append_experiment_result(results_dir.parent / "experiments", result)

    print(f"[benchmark-suite] summary={paths['summary']}")
    print(f"[benchmark-suite] relative={paths['relative']}")
    print(f"[benchmark-suite] doc={paths['doc']}")
    print(f"[benchmark-suite] manifest={paths['manifest']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
