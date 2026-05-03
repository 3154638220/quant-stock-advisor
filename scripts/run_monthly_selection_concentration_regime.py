#!/usr/bin/env python3
"""M8: 月度选股行业集中度约束与 lagged-regime 复核 —— 精简 CLI 入口。

核心算法与 M8 分析逻辑已迁移至 src/pipeline/monthly_concentration.py。
本文件仅保留 CLI 参数解析、路径解析与 main() 编排。
"""

from __future__ import annotations

import argparse, json, os, shlex, sys, time, warnings
from pathlib import Path
from typing import Any

import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.research_identity import make_research_identity, slugify_token
from scripts.run_monthly_selection_baselines import (
    EXCESS_COL, INDUSTRY_EXCESS_COL, LABEL_COL, MARKET_COL, POOL_RULES, TOP20_COL,
    _format_markdown_table,
    build_quantile_spread, build_rank_ic, build_realized_market_states,
    load_baseline_dataset, model_n_jobs_token, normalize_model_n_jobs,
    summarize_candidate_pool_reject_reason, summarize_candidate_pool_width,
    summarize_industry_exposure, summarize_regime_slice, summarize_year_slice, valid_pool_frame,
)
from scripts.run_monthly_selection_ltr import M6RunConfig, build_m6_feature_spec, build_walk_forward_ltr_scores, summarize_ltr_feature_importance
from scripts.run_monthly_selection_multisource import (
    M5RunConfig, attach_enabled_families, build_all_m5_scores, build_feature_specs,
    summarize_feature_coverage_by_spec, summarize_feature_importance as summarize_m5_feature_importance,
)
from src.models.experiment import append_experiment_result
from src.models.research_contract import (
    ArtifactRef, DataSlice, ExperimentResult, build_result_id, config_snapshot, utc_now_iso, write_research_manifest,
)
from src.settings import config_path_candidates, load_config, resolve_config_path
from src.pipeline.monthly_concentration import (
    M8RunConfig, TOPK_PRESET_DEFAULT, TOPK_PRESETS, M8_POLICY_MODEL,
    attach_trade_dates_to_scores, build_constrained_monthly,
    build_constrained_leaderboard, build_gate_table,
    build_lagged_state_frame, build_regime_policy_scores,
    resolve_topk_and_cap_grid, serialize_cap_grid,
    summarize_industry_concentration,
)


# ── CLI ───────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="M8 月度选股行业集中度与 lagged-regime 治理")
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--dataset", type=str, default="data/cache/monthly_selection_features.parquet")
    p.add_argument("--duckdb-path", type=str, default="")
    p.add_argument("--output-prefix", type=str, default="monthly_selection_m8_concentration_regime")
    p.add_argument("--as-of-date", type=str, default="")
    p.add_argument("--results-dir", type=str, default="")
    p.add_argument("--topk-preset", type=str, default=TOPK_PRESET_DEFAULT, choices=tuple(TOPK_PRESETS.keys()))
    p.add_argument("--top-k", type=str, default="")
    p.add_argument("--cap-grid", type=str, default="")
    p.add_argument("--candidate-pools", type=str, default="U1_liquid_tradable,U2_risk_sane")
    p.add_argument("--bucket-count", type=int, default=5)
    p.add_argument("--min-train-months", type=int, default=24)
    p.add_argument("--min-train-rows", type=int, default=500)
    p.add_argument("--max-fit-rows", type=int, default=0)
    p.add_argument("--cost-bps", type=float, default=10.0)
    p.add_argument("--random-seed", type=int, default=42)
    p.add_argument("--availability-lag-days", type=int, default=30)
    p.add_argument("--model-n-jobs", type=int, default=0)
    p.add_argument("--min-state-history-months", type=int, default=24)
    p.add_argument("--families", type=str, default="industry_breadth,fund_flow,fundamental")
    p.add_argument("--skip-m6", action="store_true")
    return p.parse_args()


def _resolve_project_path(raw: str | Path) -> Path:
    p = Path(raw); return p if p.is_absolute() else ROOT / p

def _project_relative(path: str | Path) -> str:
    try: return str(Path(path).resolve().relative_to(ROOT))
    except ValueError: return str(path)

def _resolve_loaded_config_path(config_arg: Path | None) -> Path | None:
    if config_arg is not None: return resolve_config_path(config_arg)
    candidates: list[Path] = []
    env_path = os.environ.get("QUANT_CONFIG", "").strip()
    if env_path: candidates.extend(config_path_candidates(env_path))
    candidates.extend([ROOT / "config.yaml", ROOT / "config.yaml.example"])
    for path in candidates:
        if path.exists(): return path
    return candidates[0] if candidates else None

def _parse_str_list(raw: str) -> list[str]:
    return [x.strip() for x in str(raw).split(",") if x.strip()]


# ── main ──────────────────────────────────────────────────────────────────

def main() -> int:
    started_at = time.perf_counter()
    args = parse_args()
    loaded_config_path = _resolve_loaded_config_path(args.config)
    cfg_raw = load_config(args.config)
    paths = cfg_raw.get("paths", {}) or {}
    config_source = _project_relative(loaded_config_path) if loaded_config_path else "default"
    dataset_path = _resolve_project_path(args.dataset)
    db_path = _resolve_project_path(args.duckdb_path.strip() or str(paths.get("duckdb_path") or "data/market.duckdb"))
    results_dir = _resolve_project_path(args.results_dir.strip() or str(paths.get("results_dir") or "data/results"))
    experiments_dir = _resolve_project_path(str(paths.get("experiments_dir") or "data/experiments"))
    as_of = args.as_of_date.strip() or pd.Timestamp.now().strftime("%Y-%m-%d")
    docs_dir = ROOT / "docs" / "reports" / as_of[:7]
    results_dir.mkdir(parents=True, exist_ok=True); docs_dir.mkdir(parents=True, exist_ok=True)

    top_ks, cap_grid = resolve_topk_and_cap_grid(preset=args.topk_preset, top_k_raw=args.top_k, cap_grid_raw=args.cap_grid)
    pools = _parse_str_list(args.candidate_pools)
    enabled_families = _parse_str_list(args.families)
    cfg = M8RunConfig(
        top_ks=tuple(top_ks), candidate_pools=tuple(pools),
        bucket_count=int(args.bucket_count), min_train_months=int(args.min_train_months),
        min_train_rows=int(args.min_train_rows), max_fit_rows=int(args.max_fit_rows),
        cost_bps=float(args.cost_bps), random_seed=int(args.random_seed),
        availability_lag_days=int(args.availability_lag_days),
        min_state_history_months=int(args.min_state_history_months),
        cap_grid=cap_grid, model_n_jobs=int(args.model_n_jobs),
    )
    output_stem = f"{slugify_token(args.output_prefix)}_{as_of}"
    research_config_id = (
        f"dataset_{slugify_token(dataset_path.stem)}_families_{'-'.join(slugify_token(x) for x in ['pv', *enabled_families])}"
        f"_pools_{'-'.join(slugify_token(x) for x in pools)}_topk_{'-'.join(str(x) for x in top_ks)}"
        f"_capgrid_{slugify_token(serialize_cap_grid(cap_grid))}"
    )
    identity = make_research_identity(
        result_type="monthly_selection_m8_concentration_regime",
        research_topic="monthly_selection_m8_concentration_regime",
        research_config_id=research_config_id, output_stem=output_stem,
    )
    research_config_id, output_stem = identity.research_config_id, identity.output_stem
    print(f"[monthly-m8] config_id={research_config_id}", flush=True)

    # ── 数据 & 特征 ──
    dataset = load_baseline_dataset(dataset_path, candidate_pools=pools)
    m5_cfg = M5RunConfig(
        top_ks=cfg.top_ks, candidate_pools=cfg.candidate_pools, bucket_count=cfg.bucket_count,
        min_train_months=cfg.min_train_months, min_train_rows=cfg.min_train_rows,
        max_fit_rows=cfg.max_fit_rows, cost_bps=cfg.cost_bps, random_seed=cfg.random_seed,
        include_xgboost=False, availability_lag_days=cfg.availability_lag_days,
        ml_models=("elasticnet", "extratrees"), model_n_jobs=cfg.model_n_jobs,
    )
    dataset = attach_enabled_families(dataset, db_path, m5_cfg, enabled_families)
    m5_specs = build_feature_specs(enabled_families)
    m5_spec = m5_specs[-1:]
    feature_coverage = summarize_feature_coverage_by_spec(dataset, m5_spec)
    m5_scores, m5_imp = build_all_m5_scores(dataset, m5_spec, m5_cfg)
    score_frames = [m5_scores] if not m5_scores.empty else []
    imp_frames = [summarize_m5_feature_importance(m5_imp)] if not m5_imp.empty else []

    include_m6 = not bool(args.skip_m6)
    if include_m6:
        m6_cfg = M6RunConfig(
            top_ks=cfg.top_ks, candidate_pools=cfg.candidate_pools, bucket_count=cfg.bucket_count,
            min_train_months=cfg.min_train_months, min_train_rows=cfg.min_train_rows,
            max_fit_rows=cfg.max_fit_rows, cost_bps=cfg.cost_bps, random_seed=cfg.random_seed,
            availability_lag_days=cfg.availability_lag_days, relevance_grades=5, model_n_jobs=cfg.model_n_jobs,
            ltr_models=("xgboost_rank_ndcg", "top20_calibrated"),
        )
        m6_scores, m6_imp = build_walk_forward_ltr_scores(dataset, build_m6_feature_spec(enabled_families), m6_cfg)
        if not m6_scores.empty: score_frames.append(m6_scores)
        if not m6_imp.empty: imp_frames.append(summarize_ltr_feature_importance(m6_imp))

    base_scores = pd.concat(score_frames, ignore_index=True) if score_frames else pd.DataFrame()
    if base_scores.empty: warnings.warn("M8 未生成 score", RuntimeWarning)
    base_scores = attach_trade_dates_to_scores(base_scores, dataset)
    lagged_states = build_lagged_state_frame(dataset, min_history_months=cfg.min_state_history_months)
    policy_scores = build_regime_policy_scores(base_scores, lagged_states)
    if not policy_scores.empty:
        base_scores = pd.concat([base_scores, policy_scores], ignore_index=True, sort=False)
    base_scores = attach_trade_dates_to_scores(base_scores, dataset)

    # ── 分析 ──
    rank_ic = build_rank_ic(base_scores)
    quantile_spread = build_quantile_spread(base_scores, bucket_count=cfg.bucket_count)
    signal_cal = sorted(pd.to_datetime(dataset["signal_date"], errors="coerce").dropna().dt.normalize().unique())
    monthly_long, topk_holdings = build_constrained_monthly(base_scores, top_ks=top_ks, cap_grid=cap_grid, cost_bps=cfg.cost_bps, signal_calendar=signal_cal)
    market_states = build_realized_market_states(dataset)
    industry_concentration = summarize_industry_concentration(topk_holdings)
    leaderboard = build_constrained_leaderboard(monthly_long, rank_ic, quantile_spread, summarize_regime_slice(monthly_long, market_states), industry_concentration)
    gate = build_gate_table(leaderboard)

    # ── 落盘 ──
    csvs = {
        "leaderboard": leaderboard, "monthly_long": monthly_long, "industry_concentration": industry_concentration,
        "industry_exposure": summarize_industry_exposure(topk_holdings),
        "regime_slice": summarize_regime_slice(monthly_long, market_states),
        "year_slice": summarize_year_slice(monthly_long),
        "topk_holdings": topk_holdings, "gate": gate, "rank_ic": rank_ic, "quantile_spread": quantile_spread,
        "lagged_states": lagged_states, "feature_coverage": feature_coverage,
        "feature_importance": pd.concat(imp_frames, ignore_index=True, sort=False) if imp_frames else pd.DataFrame(),
        "candidate_pool_width": summarize_candidate_pool_width(dataset),
        "candidate_pool_reject_reason": summarize_candidate_pool_reject_reason(dataset),
    }
    paths_out = {k: results_dir / f"{output_stem}_{k}.csv" for k in csvs}
    paths_out["doc"] = docs_dir / f"{output_stem}.md"
    paths_out["manifest"] = results_dir / f"{output_stem}_manifest.json"
    for k, df in csvs.items(): df.to_csv(paths_out[k], index=False)

    quality = {
        "research_config_id": research_config_id, "output_stem": output_stem, "config_source": config_source,
        "dataset_path": str(dataset_path.relative_to(ROOT)) if dataset_path.is_relative_to(ROOT) else str(dataset_path),
        "candidate_pools": list(pools), "top_ks": list(top_ks), "include_m6": include_m6,
        "feature_families": ["price_volume", *enabled_families],
        "pit_policy": "walk-forward ML uses only past signal months",
        "regime_policy": "strong_down=80%EN/20%ET; up/wide=60%ET/25%rank/15%top20; neutral=50/50",
        "valid_rows": int(len(valid_pool_frame(dataset))),
        "valid_signal_months": int(valid_pool_frame(dataset)["signal_date"].nunique()),
        "rows": int(len(dataset)), "base_models": sorted(base_scores["model"].unique().tolist()) if not base_scores.empty else [],
    }
    artifact_paths = [_project_relative(p) for k, p in paths_out.items() if k not in {"manifest", "doc"}]
    paths_out["doc"].write_text(f"""# M8 Concentration + Regime Report
- 生成时间：{pd.Timestamp.utcnow().isoformat()}
- 研究配置：`{research_config_id}`
- 有效月份：{quality['valid_signal_months']}
- Regime：{quality['regime_policy']}

## Leaderboard
{_format_markdown_table(leaderboard.sort_values(["top_k","candidate_pool_version","topk_excess_after_cost_mean","max_industry_share_mean"],ascending=[True,True,False,True]).head(60), max_rows=60)}

## Gate ({int(gate["m8_gate_pass"].astype(bool).sum()) if "m8_gate_pass" in gate.columns else 0}/{len(gate)} pass)
{_format_markdown_table(gate.sort_values(["top_k","candidate_pool_version","m8_gate_pass","topk_excess_after_cost_mean"],ascending=[True,True,False,False]).head(60), max_rows=60)}

## 产物
{chr(10).join(f'- `{x}`' for x in [*artifact_paths, _project_relative(paths_out["manifest"])])}
""", encoding="utf-8")

    # Manifest
    metrics = {"rows": quality["rows"], "valid_rows": quality["valid_rows"],
               "valid_signal_months": quality["valid_signal_months"],
               "m8_gate_pass_count": int(gate["m8_gate_pass"].astype(bool).sum()) if "m8_gate_pass" in gate.columns else 0}
    data_slice = DataSlice(
        dataset_name="monthly_selection_m8", source_tables=(_project_relative(dataset_path), _project_relative(db_path)),
        date_start=str(pd.to_datetime(dataset["signal_date"]).min().date()) if not dataset.empty else "",
        date_end=str(pd.to_datetime(dataset["signal_date"]).max().date()) if not dataset.empty else "",
        signal_date_col="signal_date", symbol_col="symbol", candidate_pool_version=",".join(pools),
        rebalance_rule="M", execution_mode="tplus1_open", label_return_mode="open_to_open",
        feature_set_id="m8_" + "_".join(["pv", *enabled_families]),
        feature_columns=tuple(dict.fromkeys([col for spec in m5_spec for col in spec.feature_cols])),
        label_columns=(LABEL_COL, EXCESS_COL, INDUSTRY_EXCESS_COL, MARKET_COL, TOP20_COL),
        pit_policy=quality["pit_policy"], config_path=config_source,
        extra={"top_ks": top_ks, "cap_grid": {str(k): list(v) for k, v in cap_grid.items()}, "include_m6": include_m6},
    )
    result = ExperimentResult(
        result_id=build_result_id(identity, [data_slice], metrics),
        identity=identity, script_name=_project_relative(Path(__file__).resolve()),
        command=shlex.join([sys.executable, *sys.argv]), created_at=utc_now_iso(),
        duration_sec=round(time.perf_counter() - started_at, 6), seed=int(args.random_seed),
        data_slices=(data_slice,),
        config=config_snapshot(config_path=loaded_config_path, resolved_config=cfg_raw,
                              sections=("paths","database","signals","portfolio","backtest","transaction_costs","prefilter","monthly_selection")),
        params={"cli": vars(args)}, metrics=metrics,
        gates={"data_gate": {"passed": quality["valid_rows"] > 0}},
        artifacts=(
            ArtifactRef("leaderboard_csv", _project_relative(paths_out["leaderboard"]), "csv", required_for_promotion=True),
            ArtifactRef("gate_csv", _project_relative(paths_out["gate"]), "csv", required_for_promotion=True),
            ArtifactRef("monthly_long_csv", _project_relative(paths_out["monthly_long"]), "csv"),
            ArtifactRef("manifest_json", _project_relative(paths_out["manifest"]), "json", required_for_promotion=True),
            ArtifactRef("report_md", _project_relative(paths_out["doc"]), "md", required_for_promotion=True),
        ),
        promotion={"production_eligible": False, "registry_status": "not_registered"},
    )
    write_research_manifest(paths_out["manifest"], result, extra={"generated_at_utc": result.created_at, **quality})
    append_experiment_result(experiments_dir, result)
    print(f"[monthly-m8] done valid_months={quality['valid_signal_months']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
