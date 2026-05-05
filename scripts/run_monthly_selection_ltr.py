#!/usr/bin/env python3
"""M6: 月度选股 learning-to-rank 主模型。消费 M2 canonical dataset，训练截面排序模型。"""

from __future__ import annotations

import argparse, json, sys, time, warnings
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.research_identity import make_research_identity, slugify_token
from src.pipeline.cli_helpers import (
    parse_int_list, parse_str_list, project_relative, resolve_loaded_config_path, resolve_project_path,
)
from src.pipeline.monthly_baselines import (
    build_leaderboard, build_monthly_long, build_quantile_spread, build_rank_ic,
    build_realized_market_states, load_baseline_dataset, model_n_jobs_token,
    normalize_model_n_jobs, summarize_candidate_pool_reject_reason,
    summarize_candidate_pool_width, summarize_industry_exposure,
    summarize_regime_slice, summarize_year_slice, valid_pool_frame,
)
from src.pipeline.monthly_ltr import (
    M6RunConfig, build_ltr_doc, build_ltr_quality_payload, build_ltr_relevance,
    build_m6_feature_spec, build_walk_forward_ltr_scores, summarize_ltr_feature_importance,
)
from src.pipeline.monthly_multisource import (
    FeatureSpec, M5RunConfig, attach_enabled_families, summarize_feature_coverage_by_spec,
)
from src.pipeline.research_runner import finalize_research_contract
from src.reporting.markdown_report import json_sanitize
from src.research.gates import EXCESS_COL, LABEL_COL, POOL_RULES, TOP20_COL
from src.settings import load_config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="运行月度选股 M6 learning-to-rank 主模型")
    for flag, default, kw in [
        ("--config", None, {}), ("--dataset", "data/cache/monthly_selection_features.parquet", {}),
        ("--duckdb-path", "", {}), ("--output-prefix", "monthly_selection_m6_ltr", {}),
        ("--results-dir", "", {}), ("--top-k", "20,30,50", {}),
        ("--bucket-count", 5, {"type": int}), ("--candidate-pools", "U1_liquid_tradable,U2_risk_sane", {}),
        ("--min-train-months", 24, {"type": int}), ("--min-train-rows", 500, {"type": int}),
        ("--max-fit-rows", 0, {"type": int}), ("--cost-bps", 10.0, {"type": float}),
        ("--random-seed", 42, {"type": int}), ("--availability-lag-days", 30, {"type": int}),
        ("--relevance-grades", 5, {"type": int}), ("--model-n-jobs", 0, {"type": int}),
        ("--families", "industry_breadth,fund_flow,fundamental", {}),
        ("--ltr-models", "xgboost_rank_ndcg,xgboost_rank_pairwise,top20_calibrated,ranker_top20_ensemble", {}),
        ("--rebalance-rule", "", {}),
    ]:
        p.add_argument(flag, type=kw.get("type", str), default=default)
    p.add_argument("--skip-xgboost", action="store_true")
    p.add_argument("--use-industry-neutral-zscore", action="store_true", default=False)
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
    experiments_dir = resolve_project_path(str(paths.get("experiments_dir") or "data/experiments"))
    docs_dir = ROOT / "docs"
    for d in [results_dir, experiments_dir, docs_dir]:
        d.mkdir(parents=True, exist_ok=True)

    top_ks = parse_int_list(args.top_k)
    pools = parse_str_list(args.candidate_pools)
    enabled_families = parse_str_list(args.families)
    ltr_models = tuple(parse_str_list(args.ltr_models))
    cfg = M6RunConfig(
        top_ks=tuple(top_ks), candidate_pools=tuple(pools),
        bucket_count=int(args.bucket_count), min_train_months=int(args.min_train_months),
        min_train_rows=int(args.min_train_rows), max_fit_rows=int(args.max_fit_rows),
        cost_bps=float(args.cost_bps), random_seed=int(args.random_seed),
        availability_lag_days=int(args.availability_lag_days),
        relevance_grades=int(args.relevance_grades),
        model_n_jobs=int(args.model_n_jobs), ltr_models=ltr_models,
        use_industry_neutral_zscore=bool(args.use_industry_neutral_zscore),
    )
    output_stem = f"{slugify_token(args.output_prefix)}_{pd.Timestamp.now().strftime('%Y-%m-%d')}"
    research_config_id = (
        f"dataset_{slugify_token(dataset_path.stem)}"
        f"_families_{'-'.join(slugify_token(x) for x in ['price_volume_only', *enabled_families])}"
        f"_pools_{'-'.join(slugify_token(x) for x in pools)}"
        f"_topk_{'-'.join(str(x) for x in top_ks)}"
        f"_models_{'-'.join(slugify_token(x) for x in ltr_models)}"
        f"_grades_{int(args.relevance_grades)}_maxfit_{int(args.max_fit_rows)}"
        f"_jobs_{slugify_token(model_n_jobs_token(args.model_n_jobs))}"
        f"_wf_{int(args.min_train_months)}m_costbps_{slugify_token(args.cost_bps)}"
    )
    identity = make_research_identity(
        result_type="monthly_selection_m6_ltr",
        research_topic="monthly_selection_m6_ltr",
        research_config_id=research_config_id, output_stem=output_stem,
    )
    print(f"[monthly-m6] research_config_id={research_config_id}")

    dataset = load_baseline_dataset(dataset_path, candidate_pools=pools)
    m5_cfg = M5RunConfig(
        top_ks=cfg.top_ks, candidate_pools=cfg.candidate_pools,
        bucket_count=cfg.bucket_count, min_train_months=cfg.min_train_months,
        min_train_rows=cfg.min_train_rows, max_fit_rows=cfg.max_fit_rows,
        cost_bps=cfg.cost_bps, random_seed=cfg.random_seed,
        availability_lag_days=cfg.availability_lag_days,
        model_n_jobs=cfg.model_n_jobs,
        use_industry_neutral_zscore=cfg.use_industry_neutral_zscore,
    )
    dataset = attach_enabled_families(dataset, db_path, m5_cfg, enabled_families)
    spec = build_m6_feature_spec(enabled_families, use_industry_neutral_zscore=cfg.use_industry_neutral_zscore)
    feature_coverage = summarize_feature_coverage_by_spec(dataset, [spec])
    scores, raw_importance = build_walk_forward_ltr_scores(dataset, spec, cfg)
    if scores.empty:
        warnings.warn("M6 未生成任何 score；请检查训练窗、候选池或特征覆盖。", RuntimeWarning)
    rank_ic = build_rank_ic(scores)
    monthly_long, topk_holdings = build_monthly_long(scores, top_ks=top_ks, cost_bps=cfg.cost_bps)
    quantile_spread = build_quantile_spread(scores, bucket_count=cfg.bucket_count)
    market_states = build_realized_market_states(dataset)
    year_slice = summarize_year_slice(monthly_long)
    regime_slice = summarize_regime_slice(monthly_long, market_states)
    industry_exposure = summarize_industry_exposure(topk_holdings)
    candidate_width = summarize_candidate_pool_width(dataset)
    reject_reason = summarize_candidate_pool_reject_reason(dataset)
    feature_importance = summarize_ltr_feature_importance(raw_importance)
    leaderboard = build_leaderboard(monthly_long, rank_ic, quantile_spread, regime_slice)
    quality = build_ltr_quality_payload(
        dataset=dataset, scores=scores, spec=spec, cfg=cfg,
        dataset_path=dataset_path, db_path=db_path,
        output_stem=output_stem, config_source=config_source,
        research_config_id=research_config_id,
    )

    paths_out = {
        "summary_json": results_dir / f"{output_stem}_summary.json",
        "leaderboard": results_dir / f"{output_stem}_leaderboard.csv",
        "monthly_long": results_dir / f"{output_stem}_monthly_long.csv",
        "rank_ic": results_dir / f"{output_stem}_rank_ic.csv",
        "quantile_spread": results_dir / f"{output_stem}_quantile_spread.csv",
        "feature_coverage": results_dir / f"{output_stem}_feature_coverage.csv",
        "feature_importance": results_dir / f"{output_stem}_feature_importance.csv",
        "topk_holdings": results_dir / f"{output_stem}_topk_holdings.csv",
        "industry_exposure": results_dir / f"{output_stem}_industry_exposure.csv",
        "candidate_pool_width": results_dir / f"{output_stem}_candidate_pool_width.csv",
        "candidate_pool_reject_reason": results_dir / f"{output_stem}_candidate_pool_reject_reason.csv",
        "year_slice": results_dir / f"{output_stem}_year_slice.csv",
        "regime_slice": results_dir / f"{output_stem}_regime_slice.csv",
        "market_states": results_dir / f"{output_stem}_market_states.csv",
        "manifest": results_dir / f"{output_stem}_manifest.json",
        "doc": docs_dir / f"{output_stem}.md",
    }
    leaderboard.to_csv(paths_out["leaderboard"], index=False)
    monthly_long.to_csv(paths_out["monthly_long"], index=False)
    rank_ic.to_csv(paths_out["rank_ic"], index=False)
    quantile_spread.to_csv(paths_out["quantile_spread"], index=False)
    feature_coverage.to_csv(paths_out["feature_coverage"], index=False)
    feature_importance.to_csv(paths_out["feature_importance"], index=False)
    topk_holdings.to_csv(paths_out["topk_holdings"], index=False)
    industry_exposure.to_csv(paths_out["industry_exposure"], index=False)
    candidate_width.to_csv(paths_out["candidate_pool_width"], index=False)
    reject_reason.to_csv(paths_out["candidate_pool_reject_reason"], index=False)
    year_slice.to_csv(paths_out["year_slice"], index=False)
    regime_slice.to_csv(paths_out["regime_slice"], index=False)
    market_states.to_csv(paths_out["market_states"], index=False)

    summary_payload = {
        "quality": quality,
        "top_models_by_topk": leaderboard.sort_values(
            ["top_k", "topk_excess_after_cost_mean", "rank_ic_mean"],
            ascending=[True, False, False],
        ).groupby("top_k", as_index=False).head(5).to_dict(orient="records")
        if not leaderboard.empty else [],
    }
    paths_out["summary_json"].write_text(
        json.dumps(json_sanitize(summary_payload), ensure_ascii=False, indent=2), encoding="utf-8",
    )
    artifact_paths_raw = [
        project_relative(p) for key, p in paths_out.items()
        if key not in {"manifest", "doc"}
    ]
    paths_out["doc"].write_text(
        build_ltr_doc(
            quality=quality, leaderboard=leaderboard, feature_coverage=feature_coverage,
            year_slice=year_slice, regime_slice=regime_slice,
            artifacts=[*artifact_paths_raw, project_relative(paths_out["manifest"])],
        ), encoding="utf-8",
    )

    min_signal_date = str(quality.get("min_valid_signal_date") or "")
    max_signal_date = str(quality.get("max_valid_signal_date") or "")
    best_row = leaderboard.sort_values(
        ["topk_excess_after_cost_mean", "rank_ic_mean"], ascending=[False, False],
    ).iloc[0].to_dict() if not leaderboard.empty else {}
    rank_ic_obs = int(pd.to_numeric(rank_ic.get("rank_ic"), errors="coerce").notna().sum()) if not rank_ic.empty else 0
    best_after_cost = best_row.get("topk_excess_after_cost_mean")
    rebalance_rule = args.rebalance_rule or ("M" if dataset.empty or "rebalance_rule" not in dataset.columns else str(dataset["rebalance_rule"].iloc[0]).strip().upper() or "M")
    feature_columns = tuple(spec.feature_cols)

    finalize_research_contract(
        identity=identity,
        script_path=project_relative(Path(__file__).resolve()),
        started_at=started_at, config_source=config_source, config_raw=cfg_raw,
        loaded_config_path=loaded_config_path,
        experiments_dir=experiments_dir,
        paths_out=paths_out, dataset_path=dataset_path,
        data_slice_kwargs=dict(
            dataset_name="monthly_selection_m6_ltr",
            source_tables=(project_relative(dataset_path), project_relative(db_path)),
            date_start=min_signal_date, date_end=max_signal_date,
            asof_trade_date=max_signal_date or None,
            signal_date_col="signal_date", symbol_col="symbol",
            candidate_pool_version=",".join(pools),
            rebalance_rule=rebalance_rule, execution_mode="tplus1_open",
            label_return_mode="open_to_open",
            feature_set_id=spec.name,
            feature_columns=feature_columns,
            label_columns=(LABEL_COL, EXCESS_COL, TOP20_COL),
            pit_policy=quality["pit_policy"], config_path=config_source,
            extra={
                "dataset_path": project_relative(dataset_path),
                "duckdb_path": project_relative(db_path),
                "candidate_pool_rules": {p: POOL_RULES.get(p, "") for p in pools},
                "enabled_families": enabled_families,
                "feature_spec": quality["feature_spec"],
                "top_ks": top_ks, "bucket_count": int(args.bucket_count),
                "availability_lag_days": int(args.availability_lag_days),
                "relevance_grades": int(args.relevance_grades),
                "cv_policy": quality["cv_policy"],
            },
        ),
        metrics={
            "rows": int(quality["rows"]), "valid_rows": int(quality["valid_rows"]),
            "valid_signal_months": int(quality["valid_signal_months"]),
            "score_rows": int(len(scores)), "rank_ic_observations": rank_ic_obs,
            "monthly_long_rows": int(len(monthly_long)),
            "topk_holdings_rows": int(len(topk_holdings)),
            "feature_coverage_rows": int(len(feature_coverage)),
            "model_count": int(len(quality["models"])),
            "best_model": str(best_row.get("model") or ""),
            "best_candidate_pool_version": str(best_row.get("candidate_pool_version") or ""),
            "best_top_k": int(best_row["top_k"]) if best_row.get("top_k") is not None and pd.notna(best_row.get("top_k")) else None,
            "best_topk_excess_after_cost_mean": float(best_after_cost) if pd.notna(best_after_cost) else None,
            "best_rank_ic_mean": float(best_row["rank_ic_mean"]) if best_row.get("rank_ic_mean") is not None and pd.notna(best_row.get("rank_ic_mean")) else None,
        },
        gates={
            "data_gate": {
                "passed": bool(quality["valid_rows"] > 0 and quality["valid_signal_months"] > 0),
                "checks": {
                    "has_valid_rows": quality["valid_rows"] > 0,
                    "has_valid_signal_months": quality["valid_signal_months"] > 0,
                    "has_feature_coverage": len(feature_coverage) > 0,
                },
            },
            "rank_gate": {"passed": bool(rank_ic_obs > 0), "rank_ic_observations": rank_ic_obs},
            "spread_gate": {
                "passed": bool(not monthly_long.empty and not quantile_spread.empty),
                "monthly_rows": int(len(monthly_long)),
                "quantile_spread_rows": int(len(quantile_spread)),
            },
            "baseline_gate": {
                "passed": bool(best_after_cost is not None and pd.notna(best_after_cost) and float(best_after_cost) > 0.0),
                "best_topk_excess_after_cost_mean": float(best_after_cost) if pd.notna(best_after_cost) else None,
            },
            "year_gate": {"passed": bool(not year_slice.empty), "year_slice_rows": int(len(year_slice))},
            "regime_gate": {"passed": bool(not regime_slice.empty), "regime_slice_rows": int(len(regime_slice))},
            "governance_gate": {"passed": True, "manifest_schema": "research_result_v1"},
        },
        seed=int(args.random_seed),
        promotion_blocking=["m6_ltr_research_only_not_promotion_candidate"],
        notes="Monthly selection M6 LTR contract; ranking outputs are unchanged.",
        artifact_paths_raw=artifact_paths_raw,
        cli_args=vars(args),
        params_extra={
            "top_ks": list(cfg.top_ks), "candidate_pools": list(cfg.candidate_pools),
            "bucket_count": cfg.bucket_count, "min_train_months": cfg.min_train_months,
            "min_train_rows": cfg.min_train_rows, "max_fit_rows": cfg.max_fit_rows,
            "cost_bps": cfg.cost_bps, "availability_lag_days": cfg.availability_lag_days,
            "relevance_grades": cfg.relevance_grades, "ltr_models": list(cfg.ltr_models),
            "model_n_jobs": normalize_model_n_jobs(cfg.model_n_jobs),
            "use_industry_neutral_zscore": cfg.use_industry_neutral_zscore,
        },
    )

    print(f"[monthly-m6] valid_rows={quality['valid_rows']} valid_months={quality['valid_signal_months']}")
    print(f"[monthly-m6] leaderboard={paths_out['leaderboard']}")
    print(f"[monthly-m6] manifest={paths_out['manifest']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
