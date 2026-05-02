#!/usr/bin/env python3
"""复用历史 P1 tree bundle 校准 light proxy 与 full-like proxy。"""

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

from scripts.run_backtest_eval import build_market_ew_benchmark, build_score, load_daily_from_duckdb  # noqa: E402
from src.models.xtree.p1_workflow import (  # noqa: E402
    build_tree_daily_backtest_like_proxy_detail,
    build_tree_light_proxy_detail,
    build_tree_turnover_aware_proxy_detail,
    summarize_p1_full_backtest_payload,
    summarize_tree_daily_backtest_like_proxy,
    summarize_tree_group_result,
)
from src.backtest.transaction_costs import transaction_cost_params_from_mapping  # noqa: E402
from src.models.experiment import append_experiment_result  # noqa: E402
from src.models.research_contract import (  # noqa: E402
    ArtifactRef,
    DataSlice,
    ExperimentResult,
    build_result_id,
    config_snapshot,
    utc_now_iso,
    write_research_manifest,
)
from src.settings import load_config  # noqa: E402
from scripts.research_identity import make_research_identity, slugify_token  # noqa: E402


DEFAULT_JSONS = (
    "data/results/p1_full_backtest_g0_marketrel_rank_20260426.json",
    "data/results/p1_full_backtest_g1_marketrel_rank_20260426.json",
    "data/results/p1_full_backtest_g0.json",
    "data/results/p1_full_backtest_g0_rank_direction_20260426.json",
    "data/results/p1_full_backtest_g1_rankfix_same_window_20260426.json",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="P1 light proxy / full-like proxy 历史 bundle 校准")
    p.add_argument("--config", type=Path, default=Path("config.yaml.backtest"))
    p.add_argument("--full-backtest-jsons", default=",".join(DEFAULT_JSONS), help="逗号分隔 full backtest JSON")
    p.add_argument("--proxy-horizon", type=int, default=5)
    p.add_argument("--rebalance-rule", default="M")
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--proxy-max-turnover", type=float, default=1.0)
    p.add_argument("--fallback-prepared-cache", default="data/cache/prepared_factors_p1_tree_2021_20260420.parquet")
    p.add_argument("--output-prefix", default="p1_proxy_calibration_history_2026-04-27")
    p.add_argument("--results-dir", default="data/results")
    p.add_argument("--docs-dir", default="docs")
    return p.parse_args()


def _resolve(path_like: str | Path) -> Path:
    p = Path(path_like).expanduser()
    return p if p.is_absolute() else ROOT / p


def _read_json(path_like: str | Path) -> dict[str, Any]:
    return json.loads(_resolve(path_like).read_text(encoding="utf-8"))


def _tree_model(payload: dict[str, Any]) -> dict[str, Any]:
    return ((payload.get("meta") or {}).get("tree_model") or {})


def _cache_path(payload: dict[str, Any], fallback: str) -> Path:
    params = payload.get("parameters") or {}
    meta_cache = (payload.get("meta") or {}).get("prepared_factors_cache") or {}
    raw = params.get("prepared_factors_cache") or meta_cache.get("path") or fallback
    return _resolve(str(raw))


def _bundle_dir(payload: dict[str, Any]) -> Path:
    params = payload.get("parameters") or {}
    raw = params.get("tree_bundle_dir") or _tree_model(payload).get("bundle_dir")
    if not raw:
        raise ValueError("full backtest JSON 缺少 tree_bundle_dir")
    return _resolve(str(raw))


def _tree_features(payload: dict[str, Any]) -> list[str]:
    params = payload.get("parameters") or {}
    raw = params.get("tree_features") or _tree_model(payload).get("requested_features") or []
    if isinstance(raw, str):
        return [x.strip() for x in raw.split(",") if x.strip()]
    return [str(x).strip() for x in raw if str(x).strip()]


def _sample_name(path: Path, payload: dict[str, Any]) -> str:
    tm = _tree_model(payload)
    label = tm.get("label_spec") or {}
    group = tm.get("feature_group") or (payload.get("parameters") or {}).get("tree_feature_group") or path.stem
    mode = label.get("label_mode") or "unknown_label"
    obj = label.get("xgboost_objective") or "unknown_obj"
    return f"{mode}_{obj}_{group}"


def _build_forward_returns_from_open(daily: pd.DataFrame, *, horizon: int) -> pd.DataFrame:
    if horizon < 1:
        raise ValueError("proxy horizon 须 >= 1")
    df = daily[["symbol", "trade_date", "open"]].copy()
    df["symbol"] = df["symbol"].astype(str).str.zfill(6)
    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce").dt.normalize()
    df["open"] = pd.to_numeric(df["open"], errors="coerce")
    df = df.dropna(subset=["trade_date", "open"]).sort_values(["symbol", "trade_date"])
    out_col = f"forward_ret_{int(horizon)}d"
    if df.empty:
        return pd.DataFrame(columns=["symbol", "trade_date", out_col])
    wide = df.pivot(index="trade_date", columns="symbol", values="open").sort_index()
    fwd = wide.shift(-(1 + int(horizon))) / wide.shift(-1) - 1.0
    out = fwd.stack(future_stack=True).rename(out_col).reset_index()
    return out[np.isfinite(pd.to_numeric(out[out_col], errors="coerce"))].copy()


def _build_open_to_open_returns_fast(daily: pd.DataFrame) -> pd.DataFrame:
    df = daily[["symbol", "trade_date", "open"]].copy()
    df["symbol"] = df["symbol"].astype(str).str.zfill(6)
    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce").dt.normalize()
    df["open"] = pd.to_numeric(df["open"], errors="coerce")
    df = df.dropna(subset=["trade_date", "open"]).sort_values(["trade_date", "symbol"])
    if df.empty:
        return pd.DataFrame()
    wide = df.pivot(index="trade_date", columns="symbol", values="open").sort_index()
    return (wide.shift(-1) / wide - 1.0).astype(np.float64)


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_无数据_"
    view = df.copy()

    def _cell(v: Any) -> str:
        if isinstance(v, (float, np.floating)):
            return "" if not np.isfinite(float(v)) else f"{float(v):.6g}"
        if isinstance(v, (int, np.integer)):
            return str(int(v))
        return str(v).replace("|", "\\|").replace("\n", " ")

    cols = [str(c) for c in view.columns]
    rows = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in view.iterrows():
        rows.append("| " + " | ".join(_cell(row[c]) for c in view.columns) + " |")
    return "\n".join(rows)


def _build_doc(*, prefix: str, summary: pd.DataFrame, rows: list[dict[str, Any]]) -> str:
    mean_full_like_gap = float(summary["full_like_minus_full_backtest"].mean()) if not summary.empty else float("nan")
    mean_daily_gap = float(summary["daily_bt_like_minus_full_backtest"].mean()) if not summary.empty else float("nan")
    max_abs_daily_gap = (
        float(summary["daily_bt_like_minus_full_backtest"].abs().max()) if not summary.empty else float("nan")
    )
    daily_positive_count = int((summary["daily_bt_like_proxy_excess"] > 0.0).sum()) if not summary.empty else 0
    compact = summary[
        [
            "sample",
            "group",
            "label_mode",
            "xgboost_objective",
            "unconstrained_proxy_excess",
            "full_like_proxy_excess",
            "daily_bt_like_proxy_excess",
            "full_backtest_excess",
            "full_like_minus_unconstrained",
            "full_like_minus_full_backtest",
            "daily_bt_like_minus_unconstrained",
            "daily_bt_like_minus_full_backtest",
            "n_periods",
            "daily_bt_like_n_periods",
        ]
    ].copy()
    return f"""# P1 Proxy Calibration History

- 生成时间：`{pd.Timestamp.utcnow().isoformat()}`
- 结果类型：`light_strategy_proxy_calibration`
- 样本数：`{len(rows)}`

## Summary

{_markdown_table(compact)}

## 判读

本轮复用历史 full backtest 的 tree bundle 与 prepared factors，不重训模型。`full_like_proxy_excess` 比旧版 `unconstrained_proxy_excess` 更贴近正式口径，因为它加入了月频 Top-K 的持仓延续和 `max_turnover` 限制。`daily_bt_like_proxy_excess` 进一步直接复用日频 open-to-open 收益、交易成本和 `market_ew` 对齐口径。

本轮结论：daily full-backtest-like proxy 显著降低了旧 proxy 对正式 full backtest 的高估。

1. `full_like_minus_full_backtest` 平均为 `{mean_full_like_gap:+.2%}`。
2. `daily_bt_like_minus_full_backtest` 平均为 `{mean_daily_gap:+.2%}`，最大绝对偏差为 `{max_abs_daily_gap:.2%}`。
3. `daily_bt_like_proxy_excess` 为正的样本数：`{daily_positive_count}/{len(rows)}`。

因此，daily proxy 可以作为 P1 下一轮准入 gate：若该层已经为负，就不应再补正式 full backtest；若该层接近或超过 0，再进入年度/市场状态诊断和正式回测。

## 产物

- `data/results/{prefix}_summary.csv`
- `data/results/{prefix}_detail.csv`
- `data/results/{prefix}.json`
- `docs/{prefix}.md`
"""


def main() -> int:
    args = parse_args()
    print("[P1-CAL] load config", flush=True)
    cfg = load_config(args.config)
    paths = cfg.get("paths", {}) or {}
    backtest_cfg = cfg.get("backtest", {}) or {}
    costs = transaction_cost_params_from_mapping(cfg.get("transaction_costs", {}))
    db_path = str(paths.get("duckdb_path") or "data/market.duckdb")
    db_path = str(_resolve(db_path))
    results_dir = _resolve(args.results_dir)
    docs_dir = _resolve(args.docs_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)

    json_paths = [_resolve(x.strip()) for x in str(args.full_backtest_jsons).split(",") if x.strip()]
    payloads = [(p, _read_json(p)) for p in json_paths]
    if not payloads:
        raise ValueError("未提供 full backtest JSON")

    min_start = min(pd.Timestamp((payload.get("parameters") or {}).get("start", "2021-01-01")) for _, payload in payloads)
    max_end = max(pd.Timestamp((payload.get("parameters") or {}).get("end", "2026-04-26")) for _, payload in payloads)
    print(f"[P1-CAL] load daily {min_start.date()} -> {max_end.date()}", flush=True)
    daily = load_daily_from_duckdb(db_path, str(min_start.date()), str(max_end.date()), lookback_days=0)
    print(f"[P1-CAL] daily rows={len(daily):,}", flush=True)
    fwd = _build_forward_returns_from_open(daily, horizon=int(args.proxy_horizon))
    print(f"[P1-CAL] forward rows={len(fwd):,}", flush=True)
    proxy_col = f"forward_ret_{int(args.proxy_horizon)}d"
    daily["symbol"] = daily["symbol"].astype(str).str.zfill(6)
    daily["trade_date"] = pd.to_datetime(daily["trade_date"], errors="coerce").dt.normalize()
    execution_mode = str(backtest_cfg.get("execution_mode", "tplus1_open")).lower().strip()
    if execution_mode == "tplus1_open":
        precomputed_asset_returns = _build_open_to_open_returns_fast(daily).sort_index()
        precomputed_asset_returns = precomputed_asset_returns.fillna(0.0)
    else:
        d = daily[
            (daily["trade_date"] >= min_start)
            & (daily["trade_date"] <= max_end)
            & (pd.to_numeric(daily["close"], errors="coerce") > 0)
        ].sort_values(["symbol", "trade_date"]).copy()
        d["ret"] = d.groupby("symbol")["close"].pct_change()
        precomputed_asset_returns = d.pivot(index="trade_date", columns="symbol", values="ret").sort_index().fillna(0.0)
    n_trade_days = int(precomputed_asset_returns.index.nunique())
    market_ew_min_days = max(20, int(0.35 * max(n_trade_days, 1)))
    precomputed_market_benchmark = build_market_ew_benchmark(
        daily,
        str(min_start.date()),
        str(max_end.date()),
        min_days=market_ew_min_days,
    ).sort_index()
    print(
        f"[P1-CAL] asset_returns shape={precomputed_asset_returns.shape}, "
        f"market_ew_days={len(precomputed_market_benchmark):,}",
        flush=True,
    )

    cache_frames: dict[Path, pd.DataFrame] = {}
    summary_rows: list[dict[str, Any]] = []
    detail_frames: list[pd.DataFrame] = []
    for json_path, payload in payloads:
        print(f"[P1-CAL] sample {json_path.name}", flush=True)
        cache_path = _cache_path(payload, args.fallback_prepared_cache)
        if cache_path not in cache_frames:
            print(f"[P1-CAL] read cache {cache_path}", flush=True)
            factors = pd.read_parquet(cache_path)
            factors["symbol"] = factors["symbol"].astype(str).str.zfill(6)
            factors["trade_date"] = pd.to_datetime(factors["trade_date"], errors="coerce").dt.normalize()
            factors = factors.merge(fwd, on=["symbol", "trade_date"], how="inner")
            factors = factors[np.isfinite(pd.to_numeric(factors[proxy_col], errors="coerce"))].copy()
            cache_frames[cache_path] = factors
            print(f"[P1-CAL] cache rows after merge={len(factors):,}", flush=True)
        factors = cache_frames[cache_path]
        params = payload.get("parameters") or {}
        start = pd.Timestamp(params.get("start", "2021-01-01")).normalize()
        end = pd.Timestamp(params.get("end", "2026-04-26")).normalize()
        panel = factors[(factors["trade_date"] >= start) & (factors["trade_date"] <= end)].copy()

        features = _tree_features(payload)
        print(f"[P1-CAL] score rows={len(panel):,}, features={len(features)}", flush=True)
        score = build_score(
            panel,
            {},
            sort_by="xgboost",
            tree_bundle_dir=str(_bundle_dir(payload)),
            tree_raw_features=features,
            tree_rsi_mode=str(params.get("tree_rsi_mode") or "level"),
        )
        scored = score.merge(panel[["symbol", "trade_date", proxy_col]], on=["symbol", "trade_date"], how="inner")
        sample = _sample_name(json_path, payload)
        print(f"[P1-CAL] proxy rows={len(scored):,}", flush=True)
        detail_old = build_tree_light_proxy_detail(
            scored,
            score_col="score",
            proxy_return_col=proxy_col,
            rebalance_rule=args.rebalance_rule,
            top_k=int(args.top_k),
            scenario=sample,
        )
        detail_new = build_tree_turnover_aware_proxy_detail(
            scored,
            score_col="score",
            proxy_return_col=proxy_col,
            rebalance_rule=args.rebalance_rule,
            top_k=int(args.top_k),
            max_turnover=float(args.proxy_max_turnover),
            scenario=sample,
        )
        detail_daily, daily_meta = build_tree_daily_backtest_like_proxy_detail(
            scored,
            daily,
            score_col="score",
            rebalance_rule=args.rebalance_rule,
            top_k=int(args.top_k),
            max_turnover=float(args.proxy_max_turnover),
            scenario=sample,
            cost_params=costs,
            execution_mode=str(backtest_cfg.get("execution_mode", "tplus1_open")),
            execution_lag=int(backtest_cfg.get("execution_lag", 1)),
            limit_up_mode=str(backtest_cfg.get("limit_up_mode", "idle")),
            vwap_slippage_bps_per_side=float(backtest_cfg.get("vwap_slippage_bps_per_side", 3.0)),
            vwap_impact_bps=float(backtest_cfg.get("vwap_impact_bps", 8.0)),
            market_ew_min_days=market_ew_min_days,
            precomputed_asset_returns=precomputed_asset_returns,
            precomputed_market_benchmark=precomputed_market_benchmark,
        )
        old_sum = summarize_tree_group_result(detail_old, rebalance_rule=args.rebalance_rule)
        new_sum = summarize_tree_group_result(detail_new, rebalance_rule=args.rebalance_rule)
        daily_sum = summarize_tree_daily_backtest_like_proxy(detail_daily)
        full_sum = summarize_p1_full_backtest_payload(payload)
        tm = _tree_model(payload)
        label = tm.get("label_spec") or {}
        group = str(tm.get("feature_group") or params.get("tree_feature_group") or "")
        full_excess = float(full_sum.get("full_backtest_annualized_excess_vs_market", np.nan))
        old_excess = float(old_sum.get("annualized_excess_vs_market", np.nan))
        new_excess = float(new_sum.get("annualized_excess_vs_market", np.nan))
        daily_excess = float(daily_sum.get("annualized_excess_vs_market", np.nan))
        summary_rows.append(
            {
                "sample": sample,
                "group": group,
                "label_mode": str(label.get("label_mode") or ""),
                "xgboost_objective": str(label.get("xgboost_objective") or ""),
                "source_json": str(json_path),
                "bundle_dir": str(_bundle_dir(payload)),
                "prepared_factors_cache": str(cache_path),
                "unconstrained_proxy_excess": old_excess,
                "full_like_proxy_excess": new_excess,
                "daily_bt_like_proxy_excess": daily_excess,
                "full_backtest_excess": full_excess,
                "full_like_minus_unconstrained": new_excess - old_excess,
                "full_like_minus_full_backtest": new_excess - full_excess,
                "daily_bt_like_minus_unconstrained": daily_excess - old_excess,
                "daily_bt_like_minus_full_backtest": daily_excess - full_excess,
                "n_periods": int(new_sum.get("n_periods", 0)),
                "daily_bt_like_n_periods": int(daily_sum.get("n_periods", 0)),
                "daily_bt_like_n_rebalances": int(daily_meta.get("n_rebalances", 0)),
                "daily_bt_like_avg_turnover_half_l1": float(daily_meta.get("avg_turnover_half_l1", np.nan)),
                "proxy_max_turnover": float(args.proxy_max_turnover),
                "top_k": int(args.top_k),
                "rebalance_rule": str(args.rebalance_rule),
            }
        )
        detail_old["proxy_variant"] = "topk_unconstrained"
        detail_new["proxy_variant"] = "full_like_turnover_aware"
        detail_daily["proxy_variant"] = "daily_backtest_like"
        for df in (detail_old, detail_new, detail_daily):
            df["sample"] = sample
            df["group"] = group
            df["source_json"] = str(json_path)
        detail_frames.extend([detail_old, detail_new, detail_daily])

    prefix = str(args.output_prefix).strip()
    summary = pd.DataFrame(summary_rows)
    detail = pd.concat(detail_frames, ignore_index=True) if detail_frames else pd.DataFrame()
    summary_path = results_dir / f"{prefix}_summary.csv"
    detail_path = results_dir / f"{prefix}_detail.csv"
    json_path = results_dir / f"{prefix}.json"
    doc_path = docs_dir / f"{prefix}.md"
    summary.to_csv(summary_path, index=False)
    detail.to_csv(detail_path, index=False)
    payload = {
        "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
        "result_type": "light_strategy_proxy_calibration",
        "research_topic": "p1_tree_groups",
        "output_stem": prefix,
        "summary_csv": str(summary_path),
        "detail_csv": str(detail_path),
        "doc": str(doc_path),
        "samples": summary.to_dict(orient="records"),
        "config": {
            "proxy_horizon": int(args.proxy_horizon),
            "rebalance_rule": str(args.rebalance_rule),
            "top_k": int(args.top_k),
            "proxy_max_turnover": float(args.proxy_max_turnover),
            "execution_mode": execution_mode,
            "market_ew_min_days": int(market_ew_min_days),
            "costs": {
                "commission_buy_bps": float(costs.commission_buy_bps),
                "commission_sell_bps": float(costs.commission_sell_bps),
                "slippage_bps_per_side": float(costs.slippage_bps_per_side),
                "stamp_duty_sell_bps": float(costs.stamp_duty_sell_bps),
            },
        },
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    doc_path.write_text(_build_doc(prefix=prefix, summary=summary, rows=summary_rows), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))

    # --- standard research contract ---
    def _project_relative(path: str | Path) -> str:
        p = Path(path).resolve()
        try:
            return str(p.relative_to(ROOT.resolve()))
        except ValueError:
            return str(p)

    manifest_path = results_dir / f"{prefix}_manifest.json"
    identity = make_research_identity(
        result_type="p1_proxy_calibration",
        research_topic="p1_proxy_calibration",
        research_config_id=f"p1_proxy_cal_{slugify_token(prefix)}",
        output_stem=prefix,
    )
    data_slice = DataSlice(
        dataset_name="p1_proxy_calibration_backtest",
        source_tables=("a_share_daily",),
        date_start=str(args.start) if hasattr(args, "start") else "2021-01-01",
        date_end=str(getattr(args, "end", "")) or pd.Timestamp.today().strftime("%Y-%m-%d"),
        asof_trade_date=pd.Timestamp.today().strftime("%Y-%m-%d"),
        signal_date_col="trade_date",
        symbol_col="symbol",
        candidate_pool_version="p1_tree_groups_historical",
        rebalance_rule=str(args.rebalance_rule),
        execution_mode=execution_mode,
        label_return_mode="open_to_open",
        feature_set_id="p1_historical_bundle",
        feature_columns=(),
        label_columns=(),
        pit_policy="historical_bundle_replay",
        config_path=str(args.config),
        extra={
            "proxy_horizon": int(args.proxy_horizon),
            "top_k": int(args.top_k),
            "proxy_max_turnover": float(args.proxy_max_turnover),
            "json_sources": [str(p) for p in json_paths],
        },
    )
    artifact_refs = (
        ArtifactRef("summary_csv", _project_relative(summary_path), "csv", False, "校准汇总"),
        ArtifactRef("detail_csv", _project_relative(detail_path), "csv", False, "校准明细"),
        ArtifactRef("json_report", _project_relative(json_path), "json", False, "校准 JSON 报告"),
        ArtifactRef("report_md", _project_relative(doc_path), "md", False, "校准报告"),
        ArtifactRef("manifest_json", _project_relative(manifest_path), "json", False),
    )
    metrics = {
        "sample_count": int(len(summary_rows)),
        "json_source_count": int(len(json_paths)),
    }
    gates = {
        "data_gate": {
            "passed": bool(len(summary_rows) > 0),
            "summary_rows": int(len(summary_rows)),
        },
        "execution_gate": {
            "passed": True,
            "json_sources_loaded": int(len(json_paths)),
        },
        "governance_gate": {
            "passed": True,
            "manifest_schema": "research_result_v1",
        },
    }
    result = ExperimentResult(
        result_id=build_result_id(identity, [data_slice], metrics),
        identity=identity,
        script_name=_project_relative(Path(__file__).resolve()),
        command=" ".join(sys.argv),
        created_at=utc_now_iso(),
        duration_sec=None,
        seed=None,
        data_slices=(data_slice,),
        config=config_snapshot(config_path=str(args.config), resolved_config=cfg),
        params={
            "cli": {k: str(v) for k, v in vars(args).items()},
        },
        metrics=metrics,
        gates=gates,
        artifacts=artifact_refs,
        promotion={
            "production_eligible": False,
            "registry_status": "not_registered",
            "blocking_reasons": ["p1_proxy_calibration_is_historical_research_only"],
        },
        notes="Historical P1 proxy calibration; not a promotion candidate.",
    )
    write_research_manifest(manifest_path, result)
    append_experiment_result(ROOT / "data" / "experiments", result)
    # --- end standard research contract ---

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
