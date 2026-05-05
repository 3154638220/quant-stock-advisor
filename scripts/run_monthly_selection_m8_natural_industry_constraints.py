#!/usr/bin/env python3
"""M8: naturalized industry constraints for monthly selection — soft concentration penalties and risk-budget selection."""

from __future__ import annotations

import argparse, json, sys, time, warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.research_identity import make_research_identity, slugify_token
from src.pipeline.cli_helpers import (
    parse_float_list, parse_int_list, parse_str_list, project_relative,
    resolve_loaded_config_path, resolve_project_path,
)
from src.pipeline.m8_natural_industry import (
    LABEL_VARIANTS, M8NaturalRunConfig, build_gate_table, build_label_variant_scores,
    build_leaderboard, build_m8_natural_doc, build_monthly_from_scores,
    build_quality_payload, build_score_decomposition_scores, build_soft_optimizer_scores,
    build_soft_penalty_scores, copy_source_metric_for_optimizer, load_hardcap_baseline,
)
from src.pipeline.monthly_baselines import (
    build_quantile_spread, build_rank_ic, build_realized_market_states,
    load_baseline_dataset, model_n_jobs_token, normalize_model_n_jobs,
    summarize_candidate_pool_reject_reason, summarize_candidate_pool_width,
    summarize_industry_exposure, summarize_regime_slice, summarize_year_slice,
    valid_pool_frame,
)
from src.pipeline.monthly_concentration import summarize_industry_concentration
from src.pipeline.monthly_multisource import (
    M5RunConfig, attach_enabled_families, build_all_m5_scores,
    build_feature_specs, summarize_feature_coverage_by_spec,
    summarize_feature_importance as summarize_m5_feature_importance,
)
from src.pipeline.research_runner import finalize_research_contract
from src.reporting.markdown_report import json_sanitize
from src.research.gates import EXCESS_COL, INDUSTRY_EXCESS_COL, LABEL_COL, MARKET_COL, POOL_RULES, TOP20_COL
from src.settings import load_config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="运行月度选股 M8 行业约束自然化实验")
    for flag, default, kw in [
        ("--config", None, {}), ("--dataset", "data/cache/monthly_selection_features.parquet", {}),
        ("--duckdb-path", "", {}), ("--output-prefix", "monthly_selection_m8_natural_industry_constraints", {}),
        ("--as-of-date", "", {}), ("--results-dir", "", {}),
        ("--top-k", "20,30", {}), ("--candidate-pools", "U1_liquid_tradable,U2_risk_sane", {}),
        ("--bucket-count", 5, {"type": int}), ("--min-train-months", 24, {"type": int}),
        ("--min-train-rows", 500, {"type": int}), ("--max-fit-rows", 0, {"type": int}),
        ("--cost-bps", 10.0, {"type": float}), ("--random-seed", 42, {"type": int}),
        ("--availability-lag-days", 30, {"type": int}), ("--model-n-jobs", 0, {"type": int}),
        ("--families", "industry_breadth,fund_flow,fundamental", {}),
        ("--soft-gamma", "0.05,0.10,0.20,0.35,0.50,0.80", {}),
        ("--hardcap-leaderboard", "", {}), ("--hardcap-tolerance", 0.005, {"type": float}),
    ]:
        p.add_argument(flag, type=kw.get("type", str), default=default)
    return p.parse_args()


def resolve_hardcap_leaderboard(path_raw: str, *, as_of: str, results_dir: Path) -> Path | None:
    if path_raw.strip():
        p = resolve_project_path(path_raw.strip())
        return p if p.exists() else None
    preferred = results_dir / f"monthly_selection_m8_concentration_regime_{as_of}_leaderboard.csv"
    if preferred.exists():
        return preferred
    candidates = sorted(results_dir.glob("monthly_selection_m8_concentration_regime_*_leaderboard.csv"))
    return candidates[-1] if candidates else None


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
    as_of = args.as_of_date.strip() or pd.Timestamp.now().strftime("%Y-%m-%d")
    docs_dir = ROOT / "docs" / "reports" / as_of[:7]
    for d in [results_dir, experiments_dir, docs_dir]:
        d.mkdir(parents=True, exist_ok=True)

    top_ks = parse_int_list(args.top_k)
    pools = parse_str_list(args.candidate_pools)
    gammas = parse_float_list(args.soft_gamma)
    enabled_families = parse_str_list(args.families)
    cfg = M8NaturalRunConfig(
        top_ks=tuple(top_ks), candidate_pools=tuple(pools),
        bucket_count=int(args.bucket_count), min_train_months=int(args.min_train_months),
        min_train_rows=int(args.min_train_rows), max_fit_rows=int(args.max_fit_rows),
        cost_bps=float(args.cost_bps), random_seed=int(args.random_seed),
        availability_lag_days=int(args.availability_lag_days),
        model_n_jobs=int(args.model_n_jobs),
        soft_gammas=tuple(gammas), hardcap_tolerance=float(args.hardcap_tolerance),
    )
    output_stem = f"{slugify_token(args.output_prefix)}_{as_of}"
    research_config_id = (
        f"dataset_{slugify_token(dataset_path.stem)}"
        f"_families_{'-'.join(slugify_token(x) for x in ['price_volume_only', *enabled_families])}"
        f"_pools_{'-'.join(slugify_token(x) for x in pools)}"
        f"_topk_{'-'.join(str(x) for x in top_ks)}"
        f"_labels_{'-'.join(v.name for v in LABEL_VARIANTS)}"
        f"_softgamma_{slugify_token(args.soft_gamma)}"
        f"_maxfit_{int(args.max_fit_rows)}_jobs_{slugify_token(model_n_jobs_token(args.model_n_jobs))}"
        f"_wf_{int(args.min_train_months)}m_costbps_{slugify_token(args.cost_bps)}"
    )
    identity = make_research_identity(
        result_type="monthly_selection_m8_natural_industry_constraints",
        research_topic="monthly_selection_m8_natural_industry_constraints",
        research_config_id=research_config_id, output_stem=output_stem,
    )
    print(f"[monthly-m8-natural] research_config_id={research_config_id}", flush=True)

    dataset = load_baseline_dataset(dataset_path, candidate_pools=pools)
    m5_attach_cfg = M5RunConfig(
        top_ks=cfg.top_ks, candidate_pools=cfg.candidate_pools,
        min_train_months=cfg.min_train_months, min_train_rows=cfg.min_train_rows,
        max_fit_rows=cfg.max_fit_rows, cost_bps=cfg.cost_bps,
        random_seed=cfg.random_seed, include_xgboost=False,
        availability_lag_days=cfg.availability_lag_days,
        ml_models=("elasticnet", "extratrees"), model_n_jobs=cfg.model_n_jobs,
    )
    dataset = attach_enabled_families(dataset, db_path, m5_attach_cfg, enabled_families)
    feature_specs = build_feature_specs(enabled_families)[-1:]
    feature_coverage = summarize_feature_coverage_by_spec(dataset, feature_specs)

    label_scores, importance_raw = build_label_variant_scores(dataset, cfg=cfg, enabled_families=enabled_families)
    if label_scores.empty:
        warnings.warn("M8 natural 未生成 label variant score；请检查训练窗或特征覆盖。", RuntimeWarning)
    decomposition_scores, score_decomposition = build_score_decomposition_scores(label_scores)
    penalty_scores = build_soft_penalty_scores(label_scores, gammas=gammas)

    plain_scores = pd.concat([label_scores, decomposition_scores, penalty_scores], ignore_index=True, sort=False)
    plain_monthly, plain_holdings = build_monthly_from_scores(
        plain_scores, top_ks=top_ks, cost_bps=cfg.cost_bps, selection_policy="score_topk",
    )

    optimizer_monthly_frames: list[pd.DataFrame] = []
    optimizer_holding_frames: list[pd.DataFrame] = []
    optimizer_score_sets = build_soft_optimizer_scores(label_scores, gammas=gammas)
    for gamma, opt_scores in optimizer_score_sets:
        monthly, holdings = build_monthly_from_scores(
            opt_scores, top_ks=top_ks, cost_bps=cfg.cost_bps,
            selection_policy="soft_industry_risk_budget", soft_gamma=gamma,
        )
        if not monthly.empty:
            optimizer_monthly_frames.append(monthly)
        if not holdings.empty:
            optimizer_holding_frames.append(holdings)

    monthly_long = pd.concat([plain_monthly, *optimizer_monthly_frames], ignore_index=True, sort=False)
    topk_holdings = pd.concat([plain_holdings, *optimizer_holding_frames], ignore_index=True, sort=False)

    rank_ic_plain = build_rank_ic(plain_scores)
    quantile_spread_plain = build_quantile_spread(plain_scores, bucket_count=cfg.bucket_count)
    optimizer_monthly = (
        pd.concat(optimizer_monthly_frames, ignore_index=True, sort=False) if optimizer_monthly_frames else pd.DataFrame()
    )
    rank_ic = pd.concat(
        [rank_ic_plain, copy_source_metric_for_optimizer(rank_ic_plain, optimizer_monthly)],
        ignore_index=True, sort=False,
    )
    quantile_spread = pd.concat(
        [quantile_spread_plain, copy_source_metric_for_optimizer(quantile_spread_plain, optimizer_monthly)],
        ignore_index=True, sort=False,
    )
    market_states = build_realized_market_states(dataset)
    year_slice = summarize_year_slice(monthly_long)
    regime_slice = summarize_regime_slice(monthly_long, market_states)
    industry_exposure = summarize_industry_exposure(topk_holdings)
    industry_concentration = summarize_industry_concentration(topk_holdings)
    leaderboard = build_leaderboard(monthly_long, rank_ic, quantile_spread, regime_slice, industry_concentration)

    hardcap_path = resolve_hardcap_leaderboard(args.hardcap_leaderboard, as_of=as_of, results_dir=results_dir)
    hardcap_baseline = load_hardcap_baseline(hardcap_path)
    gate = build_gate_table(leaderboard, hardcap_baseline, tolerance=cfg.hardcap_tolerance)

    label_compare = leaderboard[leaderboard["score_family"].eq("label_compare")].copy()
    penalty_frontier = leaderboard[leaderboard["score_family"].eq("penalty_frontier")].copy()
    optimizer_compare = leaderboard[leaderboard["score_family"].eq("optimizer_compare")].copy()
    feature_importance = summarize_m5_feature_importance(importance_raw) if not importance_raw.empty else pd.DataFrame()
    candidate_width = summarize_candidate_pool_width(dataset)
    reject_reason = summarize_candidate_pool_reject_reason(dataset)
    quality = build_quality_payload(
        dataset=dataset, cfg=cfg, dataset_path=dataset_path, db_path=db_path,
        output_stem=output_stem, config_source=config_source,
        research_config_id=research_config_id, enabled_families=enabled_families,
        hardcap_path=hardcap_path,
    )

    paths_out = {
        "summary_json": results_dir / f"{output_stem}_summary.json",
        "leaderboard": results_dir / f"{output_stem}_leaderboard.csv",
        "label_compare": results_dir / f"{output_stem}_label_compare.csv",
        "penalty_frontier": results_dir / f"{output_stem}_penalty_frontier.csv",
        "score_decomposition": results_dir / f"{output_stem}_score_decomposition.csv",
        "optimizer_compare": results_dir / f"{output_stem}_optimizer_compare.csv",
        "monthly_long": results_dir / f"{output_stem}_monthly_long.csv",
        "industry_concentration": results_dir / f"{output_stem}_industry_concentration.csv",
        "industry_exposure": results_dir / f"{output_stem}_industry_exposure.csv",
        "regime_slice": results_dir / f"{output_stem}_regime_slice.csv",
        "year_slice": results_dir / f"{output_stem}_year_slice.csv",
        "topk_holdings": results_dir / f"{output_stem}_topk_holdings.csv",
        "gate": results_dir / f"{output_stem}_gate.csv",
        "rank_ic": results_dir / f"{output_stem}_rank_ic.csv",
        "quantile_spread": results_dir / f"{output_stem}_quantile_spread.csv",
        "feature_coverage": results_dir / f"{output_stem}_feature_coverage.csv",
        "feature_importance": results_dir / f"{output_stem}_feature_importance.csv",
        "candidate_pool_width": results_dir / f"{output_stem}_candidate_pool_width.csv",
        "candidate_pool_reject_reason": results_dir / f"{output_stem}_candidate_pool_reject_reason.csv",
        "manifest": results_dir / f"{output_stem}_manifest.json",
        "doc": docs_dir / f"{output_stem}.md",
    }
    leaderboard.to_csv(paths_out["leaderboard"], index=False)
    label_compare.to_csv(paths_out["label_compare"], index=False)
    penalty_frontier.to_csv(paths_out["penalty_frontier"], index=False)
    score_decomposition.to_csv(paths_out["score_decomposition"], index=False)
    optimizer_compare.to_csv(paths_out["optimizer_compare"], index=False)
    monthly_long.to_csv(paths_out["monthly_long"], index=False)
    industry_concentration.to_csv(paths_out["industry_concentration"], index=False)
    industry_exposure.to_csv(paths_out["industry_exposure"], index=False)
    regime_slice.to_csv(paths_out["regime_slice"], index=False)
    year_slice.to_csv(paths_out["year_slice"], index=False)
    topk_holdings.to_csv(paths_out["topk_holdings"], index=False)
    gate.to_csv(paths_out["gate"], index=False)
    rank_ic.to_csv(paths_out["rank_ic"], index=False)
    quantile_spread.to_csv(paths_out["quantile_spread"], index=False)
    feature_coverage.to_csv(paths_out["feature_coverage"], index=False)
    feature_importance.to_csv(paths_out["feature_importance"], index=False)
    candidate_width.to_csv(paths_out["candidate_pool_width"], index=False)
    reject_reason.to_csv(paths_out["candidate_pool_reject_reason"], index=False)

    summary_payload = {
        "quality": quality,
        "top_gate_pass": gate[gate["m8_natural_gate_pass"]].head(20).to_dict(orient="records") if not gate.empty else [],
        "top_models_by_topk": leaderboard.groupby("top_k", as_index=False).head(10).to_dict(orient="records")
        if not leaderboard.empty else [],
    }

    # Statistical tests
    statistical_section: dict[str, Any] = {}
    if not monthly_long.empty and "topk_excess_vs_market" in monthly_long.columns:
        try:
            from src.analysis.capacity_report import build_m10_statistical_section
            if not leaderboard.empty:
                best = leaderboard.sort_values(
                    ["topk_excess_after_cost_mean", "rank_ic_mean"], ascending=[False, False],
                ).iloc[0]
                filt = (
                    (monthly_long["model"] == best.get("model", ""))
                    & (monthly_long["candidate_pool_version"] == best.get("candidate_pool_version", ""))
                    & (monthly_long["top_k"] == int(best.get("top_k", 20)))
                    & (monthly_long["selection_policy"] == best.get("selection_policy", "score_topk"))
                )
                best_monthly = monthly_long[filt].sort_values("signal_date")
            else:
                best_monthly = monthly_long

            excess_arr = pd.to_numeric(best_monthly["topk_excess_vs_market"], errors="coerce").to_numpy()
            turnover_arr = pd.to_numeric(best_monthly.get("turnover_half_l1", pd.Series(0.0)), errors="coerce").to_numpy()
            rank_ic_monthly = None
            if not rank_ic.empty and "rank_ic" in rank_ic.columns:
                rc = rank_ic.sort_values("signal_date")
                rank_ic_monthly = pd.to_numeric(rc["rank_ic"], errors="coerce").to_numpy()
            months = best_monthly["signal_date"].astype(str).tolist() if "signal_date" in best_monthly.columns else None
            regime_labels = None
            if not regime_slice.empty and not best_monthly.empty:
                best_monthly_dates = pd.to_datetime(best_monthly["signal_date"])
                regime_map = {}
                for _, rr in regime_slice.iterrows():
                    regime_map[str(rr.get("period", ""))[:7]] = str(rr.get("regime", ""))
                regime_labels = [regime_map.get(str(d.date())[:7], "") for d in best_monthly_dates]
            statistical_section = build_m10_statistical_section(
                excess_arr, turnover_arr, rank_ic_monthly=rank_ic_monthly,
                months=months, regime_labels=regime_labels,
            )
        except Exception:
            statistical_section = {"error": "统计显著性计算失败"}

    summary_payload["statistical_tests"] = statistical_section
    paths_out["statistical_json"] = results_dir / f"{output_stem}_statistical_tests.json"
    paths_out["statistical_json"].write_text(
        json.dumps(json_sanitize(statistical_section), ensure_ascii=False, indent=2), encoding="utf-8",
    )
    paths_out["summary_json"].write_text(
        json.dumps(json_sanitize(summary_payload), ensure_ascii=False, indent=2), encoding="utf-8",
    )
    artifact_paths_raw = [
        project_relative(p) for key, p in paths_out.items()
        if key not in {"manifest", "doc"}
    ]
    paths_out["doc"].write_text(
        build_m8_natural_doc(
            quality=quality, leaderboard=leaderboard, label_compare=label_compare,
            penalty_frontier=penalty_frontier, score_decomposition=score_decomposition,
            optimizer_compare=optimizer_compare, gate=gate,
            year_slice=year_slice, regime_slice=regime_slice,
            artifacts=[*artifact_paths_raw, project_relative(paths_out["manifest"])],
        ), encoding="utf-8",
    )

    pass_count = int(gate["m8_natural_gate_pass"].sum()) if not gate.empty else 0
    min_signal_date = str(quality.get("min_valid_signal_date") or "")
    max_signal_date = str(quality.get("max_valid_signal_date") or "")
    best_row = leaderboard.sort_values(
        ["topk_excess_after_cost_mean", "rank_ic_mean"], ascending=[False, False],
    ).iloc[0].to_dict() if not leaderboard.empty else {}
    best_gate_row = gate.sort_values(
        ["m8_natural_gate_pass", "topk_excess_after_cost_mean"], ascending=[False, False],
    ).iloc[0].to_dict() if not gate.empty else {}
    rank_ic_observations = (
        int(pd.to_numeric(rank_ic.get("rank_ic"), errors="coerce").notna().sum()) if not rank_ic.empty else 0
    )
    best_after_cost = best_row.get("topk_excess_after_cost_mean")
    best_concentration_share = best_gate_row.get("max_industry_share_mean")
    source_tables = [project_relative(dataset_path), project_relative(db_path)]
    if hardcap_path is not None:
        source_tables.append(project_relative(hardcap_path))
    feature_columns = tuple(dict.fromkeys(col for spec in feature_specs for col in spec.feature_cols))

    finalize_research_contract(
        identity=identity,
        script_path=project_relative(Path(__file__).resolve()),
        started_at=started_at, config_source=config_source, config_raw=cfg_raw,
        loaded_config_path=loaded_config_path,
        experiments_dir=experiments_dir,
        paths_out=paths_out, dataset_path=dataset_path,
        data_slice_kwargs=dict(
            dataset_name="monthly_selection_m8_natural_industry_constraints",
            source_tables=tuple(source_tables),
            date_start=min_signal_date, date_end=max_signal_date,
            asof_trade_date=max_signal_date or None,
            signal_date_col="signal_date", symbol_col="symbol",
            candidate_pool_version=",".join(pools),
            rebalance_rule="M", execution_mode="tplus1_open",
            label_return_mode="open_to_open",
            feature_set_id="m8_natural_" + "_".join(["price_volume_only", *enabled_families]),
            feature_columns=feature_columns,
            label_columns=(LABEL_COL, EXCESS_COL, INDUSTRY_EXCESS_COL, MARKET_COL, TOP20_COL),
            pit_policy=quality["pit_policy"], config_path=config_source,
            extra={
                "dataset_path": project_relative(dataset_path),
                "duckdb_path": project_relative(db_path),
                "hardcap_leaderboard_path": project_relative(hardcap_path) if hardcap_path else "",
                "candidate_pool_rules": {p: POOL_RULES.get(p, "") for p in pools},
                "enabled_families": enabled_families,
                "top_ks": top_ks, "bucket_count": int(args.bucket_count),
                "availability_lag_days": int(args.availability_lag_days),
                "label_variants": [v.name for v in LABEL_VARIANTS],
                "soft_gammas": gammas, "hardcap_tolerance": float(args.hardcap_tolerance),
                "cv_policy": quality["cv_policy"],
                "naturalization_policy": quality["naturalization_policy"],
            },
        ),
        metrics={
            "rows": int(quality["rows"]), "valid_rows": int(quality["valid_rows"]),
            "valid_signal_months": int(quality["valid_signal_months"]),
            "label_score_rows": int(len(label_scores)),
            "decomposition_score_rows": int(len(decomposition_scores)),
            "penalty_score_rows": int(len(penalty_scores)),
            "monthly_long_rows": int(len(monthly_long)),
            "topk_holdings_rows": int(len(topk_holdings)),
            "rank_ic_observations": rank_ic_observations,
            "feature_coverage_rows": int(len(feature_coverage)),
            "feature_importance_rows": int(len(feature_importance)),
            "m8_natural_gate_rows": int(len(gate)),
            "m8_natural_gate_pass_count": pass_count,
            "model_count": int(leaderboard["model"].nunique()) if "model" in leaderboard.columns else 0,
            "best_model": str(best_row.get("model") or ""),
            "best_candidate_pool_version": str(best_row.get("candidate_pool_version") or ""),
            "best_top_k": int(best_row["top_k"]) if best_row.get("top_k") is not None and pd.notna(best_row.get("top_k")) else None,
            "best_topk_excess_after_cost_mean": float(best_after_cost) if pd.notna(best_after_cost) else None,
            "best_rank_ic_mean": float(best_row["rank_ic_mean"]) if best_row.get("rank_ic_mean") is not None and pd.notna(best_row.get("rank_ic_mean")) else None,
            "best_gate_model": str(best_gate_row.get("model") or ""),
            "best_gate_pass": bool(best_gate_row.get("m8_natural_gate_pass", False)),
            "best_max_industry_share_mean": float(best_concentration_share) if pd.notna(best_concentration_share) else None,
        },
        gates={
            "data_gate": {
                "passed": bool(quality["valid_rows"] > 0 and quality["valid_signal_months"] > 0 and len(label_scores) > 0),
                "checks": {
                    "has_valid_rows": quality["valid_rows"] > 0,
                    "has_valid_signal_months": quality["valid_signal_months"] > 0,
                    "has_label_scores": len(label_scores) > 0,
                    "has_feature_coverage": len(feature_coverage) > 0,
                },
            },
            "rank_gate": {"passed": bool(rank_ic_observations > 0), "rank_ic_observations": rank_ic_observations},
            "spread_gate": {
                "passed": bool(not monthly_long.empty and not quantile_spread.empty),
                "monthly_rows": int(len(monthly_long)), "quantile_spread_rows": int(len(quantile_spread)),
            },
            "year_gate": {"passed": bool(not year_slice.empty), "year_slice_rows": int(len(year_slice))},
            "regime_gate": {"passed": bool(not regime_slice.empty), "regime_slice_rows": int(len(regime_slice))},
            "concentration_gate": {
                "passed": bool(pass_count > 0),
                "m8_natural_gate_pass_count": pass_count, "gate_rows": int(len(gate)),
                "best_max_industry_share_mean": float(best_concentration_share) if pd.notna(best_concentration_share) else None,
            },
            "hardcap_comparison_gate": {
                "passed": bool(not hardcap_baseline.empty or not gate.empty),
                "hardcap_baseline_rows": int(len(hardcap_baseline)),
                "hardcap_leaderboard_path": project_relative(hardcap_path) if hardcap_path else "",
            },
            "governance_gate": {"passed": True, "manifest_schema": "research_result_v1"},
        },
        seed=int(args.random_seed),
        promotion_blocking=["m8_natural_constraints_requires_manual_promotion_package"],
        notes="Monthly selection M8 natural industry constraints contract; strategy outputs are unchanged.",
        artifact_paths_raw=artifact_paths_raw,
        cli_args=vars(args),
        params_extra={
            "top_ks": list(cfg.top_ks), "candidate_pools": list(cfg.candidate_pools),
            "bucket_count": cfg.bucket_count, "min_train_months": cfg.min_train_months,
            "min_train_rows": cfg.min_train_rows, "max_fit_rows": cfg.max_fit_rows,
            "cost_bps": cfg.cost_bps, "availability_lag_days": cfg.availability_lag_days,
            "soft_gammas": list(cfg.soft_gammas), "hardcap_tolerance": cfg.hardcap_tolerance,
            "enabled_families": enabled_families,
            "model_n_jobs": normalize_model_n_jobs(cfg.model_n_jobs),
        },
    )

    print(f"[monthly-m8-natural] valid_rows={quality['valid_rows']} valid_months={quality['valid_signal_months']}", flush=True)
    print(f"[monthly-m8-natural] gate_pass={pass_count}", flush=True)
    print(f"[monthly-m8-natural] leaderboard={paths_out['leaderboard']}", flush=True)
    print(f"[monthly-m8-natural] gate={paths_out['gate']}", flush=True)
    print(f"[monthly-m8-natural] manifest={paths_out['manifest']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
