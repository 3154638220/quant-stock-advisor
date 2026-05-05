#!/usr/bin/env python3
"""M7: 研究版月度 Top-K 推荐报告。消费 M2 canonical dataset，生成研究报告与推荐名单。"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.research_identity import make_research_identity, slugify_token
from src.data_fetcher.stock_name_cache import (
    attach_stock_names,
    ensure_stock_name_cache,
    load_stock_name_cache,
)
from src.pipeline.cli_helpers import (
    parse_int_list,
    parse_str_list,
    project_relative,
    resolve_loaded_config_path,
    resolve_project_path,
)
from src.pipeline.monthly_baselines import (
    load_baseline_dataset,
    model_n_jobs_token,
    normalize_model_n_jobs,
    summarize_candidate_pool_reject_reason,
    summarize_candidate_pool_width,
)
from src.pipeline.monthly_ltr import build_m6_feature_spec, summarize_ltr_feature_importance
from src.pipeline.monthly_multisource import M5RunConfig, attach_enabled_families
from src.pipeline.research_runner import finalize_research_contract
from src.reporting.markdown_report import json_sanitize
from src.reporting.monthly_report import (
    M7RunConfig,
    apply_m9_feature_coverage_policy,
    build_full_fit_report_scores,
    build_m7_doc,
    build_quality_payload,
    build_recommendation_table,
    filter_evidence,
    latest_evidence_stem,
    read_evidence,
    select_report_signal_date,
    summarize_m9_integrity,
    summarize_recommendation_industry_exposure,
    summarize_recommendation_risk,
    summarize_report_feature_coverage,
)
from src.research.gates import POOL_RULES
from src.settings import load_config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="生成月度选股 M7 研究版推荐报告")
    for flag, default, kw in [
        ("--config", None, {}), ("--dataset", "data/cache/monthly_selection_features.parquet", {}),
        ("--duckdb-path", "", {}), ("--output-prefix", "monthly_selection_m7_recommendation_report", {}),
        ("--results-dir", "", {}), ("--signal-date", "", {}),
        ("--top-k", "20,30", {}), ("--report-top-k", 20, {"type": int}),
        ("--candidate-pools", "U2_risk_sane", {}), ("--min-train-months", 24, {"type": int}),
        ("--min-train-rows", 500, {"type": int}), ("--max-fit-rows", 0, {"type": int}),
        ("--cost-bps", 10.0, {"type": float}), ("--random-seed", 42, {"type": int}),
        ("--availability-lag-days", 30, {"type": int}), ("--relevance-grades", 5, {"type": int}),
        ("--model-n-jobs", 0, {"type": int}), ("--min-core-feature-coverage", 0.30, {"type": float}),
        ("--stock-name-cache", "data/cache/a_share_stock_names.csv", {}),
        ("--families", "industry_breadth,fund_flow,fundamental", {}),
        ("--evidence-stem", "", {}),
    ]:
        p.add_argument(flag, type=kw.get("type", str), default=default)
    p.add_argument("--refresh-name-cache", action="store_true", default=False)
    p.add_argument("--name-cache-max-age-days", type=int, default=30)
    return p.parse_args()


def main() -> int:
    started_at = time.perf_counter()
    args = parse_args()
    loaded_config_path = resolve_loaded_config_path(args.config)
    cfg_raw = load_config(args.config)
    paths = cfg_raw.get("paths", {}) or {}
    config_source = project_relative(loaded_config_path) if loaded_config_path else "default_config_lookup"
    dataset_path = resolve_project_path(args.dataset)
    db_path = resolve_project_path(args.duckdb_path.strip() or str(paths.get("duckdb_path") or "data/market.duckdb"))
    results_dir = resolve_project_path(args.results_dir.strip() or str(paths.get("results_dir") or "data/results"))
    docs_dir = ROOT / "docs"
    experiments_dir = resolve_project_path(str(paths.get("experiments_dir") or "data/experiments"))
    for d in [results_dir, docs_dir, experiments_dir]:
        d.mkdir(parents=True, exist_ok=True)

    top_ks = parse_int_list(args.top_k)
    pools = tuple(parse_str_list(args.candidate_pools))
    enabled_families = parse_str_list(args.families)
    cfg = M7RunConfig(
        top_ks=tuple(top_ks), report_top_k=int(args.report_top_k),
        candidate_pools=pools, min_train_months=int(args.min_train_months),
        min_train_rows=int(args.min_train_rows), max_fit_rows=int(args.max_fit_rows),
        cost_bps=float(args.cost_bps), random_seed=int(args.random_seed),
        availability_lag_days=int(args.availability_lag_days),
        relevance_grades=int(args.relevance_grades),
        min_core_feature_coverage=float(args.min_core_feature_coverage),
        model_n_jobs=int(args.model_n_jobs),
    )
    output_stem = f"{slugify_token(args.output_prefix)}_{pd.Timestamp.now().strftime('%Y-%m-%d')}"
    research_config_id = (
        f"dataset_{slugify_token(dataset_path.stem)}"
        f"_families_{'-'.join(slugify_token(x) for x in ['price_volume_only', *enabled_families])}"
        f"_pools_{'-'.join(slugify_token(x) for x in pools)}"
        f"_topk_{'-'.join(str(x) for x in top_ks)}"
        f"_model_{slugify_token(cfg.model_name)}"
        f"_maxfit_{int(args.max_fit_rows)}_jobs_{slugify_token(model_n_jobs_token(args.model_n_jobs))}"
        f"_wf_{int(args.min_train_months)}m"
    )
    identity = make_research_identity(
        result_type="monthly_selection_m7_recommendation_report",
        research_topic="monthly_selection",
        research_config_id=research_config_id, output_stem=output_stem,
        canonical_config_name="monthly_selection_m7_recommendation_report_v1",
    )
    print(f"[monthly-m7] research_config_id={research_config_id}")

    dataset = load_baseline_dataset(dataset_path, candidate_pools=list(pools))
    stock_name_cache_path = resolve_project_path(args.stock_name_cache)
    ensure_stock_name_cache(
        stock_name_cache_path, force=bool(args.refresh_name_cache),
        max_age_days=int(args.name_cache_max_age_days),
        config_path=args.config, project_root=ROOT,
    )
    stock_names = load_stock_name_cache(stock_name_cache_path)
    dataset = attach_stock_names(dataset, stock_names)
    m5_cfg = M5RunConfig(
        top_ks=cfg.top_ks, candidate_pools=cfg.candidate_pools,
        min_train_months=cfg.min_train_months, min_train_rows=cfg.min_train_rows,
        max_fit_rows=cfg.max_fit_rows, cost_bps=cfg.cost_bps,
        random_seed=cfg.random_seed, availability_lag_days=cfg.availability_lag_days,
        model_n_jobs=cfg.model_n_jobs,
    )
    dataset = attach_enabled_families(dataset, db_path, m5_cfg, enabled_families)
    spec = build_m6_feature_spec(enabled_families)
    feature_coverage = summarize_report_feature_coverage(dataset, spec, candidate_pools=cfg.candidate_pools)
    active_feature_cols, feature_policy = apply_m9_feature_coverage_policy(
        dataset, spec, feature_coverage,
        candidate_pools=cfg.candidate_pools, min_core_coverage=cfg.min_core_feature_coverage,
    )
    report_signal_date = select_report_signal_date(
        dataset, candidate_pools=cfg.candidate_pools,
        requested=args.signal_date.strip() or None,
    )
    scores, raw_importance = build_full_fit_report_scores(
        dataset, spec, cfg, report_signal_date=report_signal_date,
        feature_cols=active_feature_cols,
    )
    feature_importance = summarize_ltr_feature_importance(raw_importance)

    evidence_stem = args.evidence_stem.strip() or latest_evidence_stem(results_dir)
    previous_holdings = read_evidence(results_dir, evidence_stem, "topk_holdings")
    recommendations, feature_contrib = build_recommendation_table(
        scores, dataset, feature_importance,
        feature_cols=[c for c in spec.feature_cols if c in dataset.columns],
        top_ks=top_ks, previous_holdings=previous_holdings,
    )
    leaderboard = filter_evidence(read_evidence(results_dir, evidence_stem, "leaderboard"), cfg)
    monthly_long = filter_evidence(read_evidence(results_dir, evidence_stem, "monthly_long"), cfg)
    rank_ic = filter_evidence(read_evidence(results_dir, evidence_stem, "rank_ic"), cfg)
    quantile_spread = filter_evidence(read_evidence(results_dir, evidence_stem, "quantile_spread"), cfg)
    year_slice = filter_evidence(read_evidence(results_dir, evidence_stem, "year_slice"), cfg)
    regime_slice = filter_evidence(read_evidence(results_dir, evidence_stem, "regime_slice"), cfg)
    industry_exposure = summarize_recommendation_industry_exposure(recommendations)
    risk_summary = summarize_recommendation_risk(recommendations)
    m9_integrity = summarize_m9_integrity(
        dataset=dataset, recommendations=recommendations,
        feature_coverage=feature_coverage, feature_policy=feature_policy,
        report_signal_date=report_signal_date, candidate_pools=cfg.candidate_pools,
    )
    candidate_width = summarize_candidate_pool_width(dataset)
    reject_reason = summarize_candidate_pool_reject_reason(dataset)
    quality = build_quality_payload(
        dataset=dataset, recommendations=recommendations,
        report_signal_date=report_signal_date, spec=spec, cfg=cfg,
        dataset_path=dataset_path, db_path=db_path,
        output_stem=output_stem, config_source=config_source,
        research_config_id=research_config_id, evidence_stem=evidence_stem,
        project_root=ROOT,
    )

    paths_out = {
        "summary_json": results_dir / f"{output_stem}_summary.json",
        "recommendations": results_dir / f"{output_stem}_recommendations.csv",
        "leaderboard": results_dir / f"{output_stem}_leaderboard.csv",
        "monthly_long": results_dir / f"{output_stem}_monthly_long.csv",
        "rank_ic": results_dir / f"{output_stem}_rank_ic.csv",
        "quantile_spread": results_dir / f"{output_stem}_quantile_spread.csv",
        "topk_holdings": results_dir / f"{output_stem}_topk_holdings.csv",
        "industry_exposure": results_dir / f"{output_stem}_industry_exposure.csv",
        "candidate_pool_width": results_dir / f"{output_stem}_candidate_pool_width.csv",
        "candidate_pool_reject_reason": results_dir / f"{output_stem}_candidate_pool_reject_reason.csv",
        "feature_importance": results_dir / f"{output_stem}_feature_importance.csv",
        "feature_contrib": results_dir / f"{output_stem}_feature_contrib.csv",
        "feature_coverage": results_dir / f"{output_stem}_feature_coverage.csv",
        "feature_policy": results_dir / f"{output_stem}_feature_policy.csv",
        "m9_integrity": results_dir / f"{output_stem}_m9_integrity.csv",
        "risk_summary": results_dir / f"{output_stem}_risk_summary.csv",
        "year_slice": results_dir / f"{output_stem}_year_slice.csv",
        "regime_slice": results_dir / f"{output_stem}_regime_slice.csv",
        "manifest": results_dir / f"{output_stem}_manifest.json",
        "doc": docs_dir / f"{output_stem}.md",
    }

    recommendations.to_csv(paths_out["recommendations"], index=False)
    recommendations.to_csv(paths_out["topk_holdings"], index=False)
    leaderboard.to_csv(paths_out["leaderboard"], index=False)
    monthly_long.to_csv(paths_out["monthly_long"], index=False)
    rank_ic.to_csv(paths_out["rank_ic"], index=False)
    quantile_spread.to_csv(paths_out["quantile_spread"], index=False)
    industry_exposure.to_csv(paths_out["industry_exposure"], index=False)
    candidate_width.to_csv(paths_out["candidate_pool_width"], index=False)
    reject_reason.to_csv(paths_out["candidate_pool_reject_reason"], index=False)
    feature_importance.to_csv(paths_out["feature_importance"], index=False)
    feature_contrib.to_csv(paths_out["feature_contrib"], index=False)
    feature_coverage.to_csv(paths_out["feature_coverage"], index=False)
    feature_policy.to_csv(paths_out["feature_policy"], index=False)
    m9_integrity.to_csv(paths_out["m9_integrity"], index=False)
    risk_summary.to_csv(paths_out["risk_summary"], index=False)
    year_slice.to_csv(paths_out["year_slice"], index=False)
    regime_slice.to_csv(paths_out["regime_slice"], index=False)

    summary_payload = {
        "quality": quality,
        "recommendations_top": recommendations[
            recommendations["top_k"] == int(cfg.report_top_k)
        ].head(int(cfg.report_top_k)).to_dict(orient="records")
        if not recommendations.empty else [],
        "historical_evidence_stem": evidence_stem,
        "stock_name_cache": project_relative(stock_name_cache_path),
        "active_feature_cols": active_feature_cols,
        "m9_integrity_pass": bool(m9_integrity["pass"].all()) if not m9_integrity.empty else False,
    }
    paths_out["summary_json"].write_text(
        json.dumps(json_sanitize(summary_payload), ensure_ascii=False, indent=2), encoding="utf-8",
    )
    artifact_paths_raw = [
        project_relative(p) for key, p in paths_out.items()
        if key not in {"manifest", "doc"}
    ]
    paths_out["doc"].write_text(
        build_m7_doc(
            quality=quality, recommendations=recommendations, leaderboard=leaderboard,
            risk_summary=risk_summary, industry_exposure=industry_exposure,
            feature_coverage=feature_coverage, feature_policy=feature_policy,
            m9_integrity=m9_integrity,
            artifacts=[*artifact_paths_raw, project_relative(paths_out["manifest"])],
        ), encoding="utf-8",
    )

    min_signal_date = str(dataset["signal_date"].min().date()) if not dataset.empty else ""
    max_signal_date = str(dataset["signal_date"].max().date()) if not dataset.empty else ""
    m9_pass = bool(m9_integrity["pass"].all()) if not m9_integrity.empty else False

    finalize_research_contract(
        identity=identity,
        script_path=project_relative(Path(__file__).resolve()),
        started_at=started_at,
        config_source=config_source, config_raw=cfg_raw,
        loaded_config_path=loaded_config_path,
        experiments_dir=experiments_dir,
        paths_out=paths_out, dataset_path=dataset_path,
        data_slice_kwargs=dict(
            dataset_name="monthly_selection_m7_recommendation",
            source_tables=(project_relative(dataset_path),),
            date_start=min_signal_date, date_end=max_signal_date,
            asof_trade_date=str(report_signal_date.date()),
            signal_date_col="signal_date", symbol_col="symbol",
            candidate_pool_version=",".join(pools),
            rebalance_rule="M", execution_mode="tplus1_open",
            label_return_mode="open_to_open",
            feature_set_id=spec.name,
            feature_columns=tuple(active_feature_cols),
            label_columns=(),
            pit_policy="target features are signal-date rows; full-fit model uses only labeled months before report_signal_date",
            config_path=config_source,
            extra={
                "dataset_path": project_relative(dataset_path),
                "candidate_pool_rules": {p: POOL_RULES.get(p, "") for p in pools},
                "top_ks": top_ks, "report_top_k": int(cfg.report_top_k),
                "enabled_families": enabled_families,
                "feature_spec_name": spec.name,
                "active_feature_count": len(active_feature_cols),
                "evidence_stem": evidence_stem,
            },
        ),
        metrics={
            "recommendation_rows": int(len(recommendations)),
            "recommendation_topk_groups": int(recommendations["top_k"].nunique()) if not recommendations.empty else 0,
            "candidate_pool_pass_rows": int(quality.get("target_candidate_pass_rows", 0)),
            "active_feature_count": len(active_feature_cols),
            "core_features": len([c for c in active_feature_cols if not c.startswith("is_missing_")]),
            "missing_flag_features": len([c for c in active_feature_cols if c.startswith("is_missing_")]),
            "evidence_stem_found": bool(evidence_stem),
            "stock_name_cache_rows": int(len(stock_names)),
        },
        gates={
            "data_gate": {
                "passed": bool(not dataset.empty and quality.get("target_candidate_pass_rows", 0) > 0),
                "checks": {
                    "dataset_not_empty": not dataset.empty,
                    "has_candidate_pool_pass": quality.get("target_candidate_pass_rows", 0) > 0,
                },
            },
            "m9_integrity_gate": {
                "passed": m9_pass,
                "integrity_checks": m9_integrity.to_dict(orient="records") if not m9_integrity.empty else [],
            },
            "governance_gate": {"passed": True, "manifest_schema": "research_result_v1"},
        },
        seed=int(cfg.random_seed),
        promotion_blocking=["m7_research_report_only_not_promotion_candidate"],
        notes="Monthly selection M7 recommendation report; full-fit XGBoost ranker on latest signal month; research-only output.",
        artifact_paths_raw=artifact_paths_raw,
        cli_args=vars(args),
        params_extra={
            "top_ks": list(cfg.top_ks), "report_top_k": int(cfg.report_top_k),
            "candidate_pools": list(cfg.candidate_pools),
            "min_train_months": cfg.min_train_months, "min_train_rows": cfg.min_train_rows,
            "max_fit_rows": cfg.max_fit_rows, "cost_bps": cfg.cost_bps,
            "availability_lag_days": cfg.availability_lag_days,
            "model_name": cfg.model_name,
            "min_core_feature_coverage": cfg.min_core_feature_coverage,
            "model_n_jobs": normalize_model_n_jobs(cfg.model_n_jobs),
        },
        manifest_extra={
            "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
            "result_type": "monthly_selection_m7_recommendation_report_manifest",
            "research_topic": identity.research_topic,
            "research_config_id": identity.research_config_id,
            "output_stem": identity.output_stem,
            "config_source": config_source,
            "dataset_path": project_relative(dataset_path),
            "dataset_version": "monthly_selection_features_v1",
            "candidate_pools": pools,
            "candidate_pool_rules": {p: POOL_RULES.get(p, "") for p in pools},
            "top_ks": top_ks, "report_top_k": int(cfg.report_top_k),
            "feature_spec": spec.name, "active_feature_cols": active_feature_cols,
            "pit_policy": "target features are signal-date rows; full-fit model uses only labeled months before report_signal_date",
            "historical_evidence_stem": evidence_stem,
            "legacy_artifacts": [*artifact_paths_raw, project_relative(paths_out["doc"])],
            **quality,
        },
    )

    # OOS auto-writer
    try:
        from src.monitoring.oos_auto_writer import record_oos_from_m7_report
        predicted_excess = float(monthly_long["excess_return"].mean()) if (
            not monthly_long.empty and "excess_return" in monthly_long.columns
        ) else None
        if predicted_excess is not None and np.isfinite(predicted_excess):
            oos_config_id = str(research_config_id).replace("m7_", "m8_") if research_config_id else (
                f"monthly_selection_u1_top{int(cfg.report_top_k)}_m7"
            )
            oos_holdings = []
            if not recommendations.empty:
                topk = recommendations[recommendations["top_k"] == int(cfg.report_top_k)]
                oos_holdings = topk["symbol"].astype(str).tolist()[:int(cfg.report_top_k)]
            oos_result = record_oos_from_m7_report(
                db_path=str(db_path), config_id=oos_config_id,
                signal_date=str(report_signal_date.date()),
                predicted_excess_monthly=predicted_excess,
                candidate_pool=",".join(pools) if pools else "U1_liquid_tradable",
                top_k=int(cfg.report_top_k), cost_bps=float(cfg.cost_bps),
                holdings=oos_holdings, num_holdings=len(oos_holdings),
            )
            if oos_result.predicted_written:
                print(f"[monthly-m7] OOS record written: config={oos_config_id} "
                      f"date={report_signal_date.date()} pred={predicted_excess:.4%}")
            if oos_result.realized_backfilled:
                print(f"[monthly-m7] OOS backfill: date={oos_result.backfilled_date} "
                      f"realized={oos_result.previous_prediction}")
    except Exception as oos_err:
        print(f"[monthly-m7] OOS record skipped (non-fatal): {oos_err}")

    print(f"[monthly-m7] report_signal_date={quality['report_signal_date']} recommendations={len(recommendations)}")
    print(f"[monthly-m7] recommendations={paths_out['recommendations']}")
    print(f"[monthly-m7] doc={paths_out['doc']}")
    print(f"[monthly-m7] manifest={paths_out['manifest']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
