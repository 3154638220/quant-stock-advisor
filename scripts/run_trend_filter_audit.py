#!/usr/bin/env python3
"""Audit W7 trend-persistence filters as standalone candidate-pool gates."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.research_identity import make_research_identity, slugify_token
from src.pipeline.cli_helpers import (
    parse_str_list,
    project_relative,
    resolve_loaded_config_path,
    resolve_project_path,
)
from src.pipeline.monthly_baselines import load_baseline_dataset, valid_pool_frame
from src.pipeline.research_runner import finalize_research_contract
from src.pipeline.shared_loaders import attach_feature_families
from src.reporting.markdown_report import format_markdown_table, json_sanitize
from src.research.gates import EXCESS_COL, LABEL_COL, POOL_RULES, TOP20_COL
from src.settings import load_config


@dataclass(frozen=True)
class TrendFilterSpec:
    name: str
    description: str
    min_count: int = 60


FILTER_SPECS: tuple[TrendFilterSpec, ...] = (
    TrendFilterSpec("bull_state", "保留 EMA12/EMA26 多头状态股票"),
    TrendFilterSpec("bear_state_reverse", "保留 EMA12/EMA26 空头状态股票，用于检验反向/过热假设"),
    TrendFilterSpec("stable_q70", "保留距上次翻转天数位于当月同池前 30% 的股票"),
    TrendFilterSpec("stable_q80", "保留距上次翻转天数位于当月同池前 20% 的股票"),
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="W7 trend-persistence filter audit")
    p.add_argument("--config", type=str, default="")
    p.add_argument("--dataset", type=str, default="data/cache/monthly_selection_features.parquet")
    p.add_argument("--duckdb-path", type=str, default="")
    p.add_argument("--results-dir", type=str, default="")
    p.add_argument("--output-prefix", type=str, default="w7_trend_filter_audit")
    p.add_argument("--candidate-pools", type=str, default="U1_liquid_tradable,U2_risk_sane")
    return p.parse_args()


def _numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _filter_mask(part: pd.DataFrame, name: str) -> pd.Series:
    bull = _numeric(part["feature_trend_bull_state"])
    flip_days = _numeric(part["feature_trend_flip_days_ago"])
    if name == "bull_state":
        return bull.eq(1.0)
    if name == "bear_state_reverse":
        return bull.eq(0.0)
    if name in {"stable_q70", "stable_q80"}:
        q = 0.70 if name == "stable_q70" else 0.80
        threshold = flip_days.quantile(q)
        if pd.isna(threshold):
            return pd.Series(False, index=part.index)
        return flip_days.ge(float(threshold))
    raise ValueError(f"unknown trend filter: {name}")


def build_monthly_filter_audit(dataset: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    valid = valid_pool_frame(dataset)
    if valid.empty:
        return pd.DataFrame()

    for (signal_date, pool), part in valid.groupby(["signal_date", "candidate_pool_version"], sort=True):
        part = part.copy()
        excess = _numeric(part[EXCESS_COL])
        label = _numeric(part[LABEL_COL])
        top20 = _numeric(part[TOP20_COL]) if TOP20_COL in part.columns else pd.Series(np.nan, index=part.index)
        all_excess_mean = float(excess.mean())
        all_return_mean = float(label.mean())
        all_hit_rate = float((excess > 0).mean())
        all_top20_rate = float(top20.mean())

        for spec in FILTER_SPECS:
            mask = _filter_mask(part, spec.name).fillna(False)
            selected = part.loc[mask].copy()
            rejected = part.loc[~mask].copy()
            selected_excess = _numeric(selected[EXCESS_COL]) if not selected.empty else pd.Series(dtype=float)
            selected_label = _numeric(selected[LABEL_COL]) if not selected.empty else pd.Series(dtype=float)
            rejected_excess = _numeric(rejected[EXCESS_COL]) if not rejected.empty else pd.Series(dtype=float)
            selected_top20 = _numeric(selected[TOP20_COL]) if TOP20_COL in selected.columns else pd.Series(dtype=float)

            selected_excess_mean = float(selected_excess.mean()) if not selected.empty else np.nan
            rejected_excess_mean = float(rejected_excess.mean()) if not rejected.empty else np.nan
            rows.append(
                {
                    "signal_date": pd.Timestamp(signal_date).date().isoformat(),
                    "candidate_pool_version": pool,
                    "filter_name": spec.name,
                    "description": spec.description,
                    "before_count": int(len(part)),
                    "after_count": int(len(selected)),
                    "rejected_count": int(len(rejected)),
                    "retention_ratio": float(len(selected) / len(part)) if len(part) else np.nan,
                    "below_min_count": bool(len(selected) < spec.min_count),
                    "all_excess_mean": all_excess_mean,
                    "selected_excess_mean": selected_excess_mean,
                    "rejected_excess_mean": rejected_excess_mean,
                    "selected_minus_all_excess": selected_excess_mean - all_excess_mean,
                    "selected_minus_rejected_excess": selected_excess_mean - rejected_excess_mean,
                    "all_forward_return_mean": all_return_mean,
                    "selected_forward_return_mean": float(selected_label.mean()) if not selected.empty else np.nan,
                    "all_hit_rate": all_hit_rate,
                    "selected_hit_rate": float((selected_excess > 0).mean()) if not selected.empty else np.nan,
                    "all_top20_rate": all_top20_rate,
                    "selected_top20_rate": float(selected_top20.mean()) if not selected.empty else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _t_stat(values: pd.Series) -> float:
    vals = _numeric(values).dropna()
    if len(vals) < 2:
        return np.nan
    std = float(vals.std(ddof=1))
    if std == 0.0:
        return np.nan
    return float(vals.mean() / (std / np.sqrt(len(vals))))


def summarize_filter_audit(monthly: pd.DataFrame) -> pd.DataFrame:
    if monthly.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for (pool, name), part in monthly.groupby(["candidate_pool_version", "filter_name"], sort=True):
        rows.append(
            {
                "candidate_pool_version": pool,
                "filter_name": name,
                "months": int(part["signal_date"].nunique()),
                "mean_before_count": float(part["before_count"].mean()),
                "mean_after_count": float(part["after_count"].mean()),
                "median_after_count": float(part["after_count"].median()),
                "min_after_count": int(part["after_count"].min()),
                "months_below_60": int(part["below_min_count"].sum()),
                "mean_retention_ratio": float(part["retention_ratio"].mean()),
                "selected_excess_mean": float(_numeric(part["selected_excess_mean"]).mean()),
                "selected_minus_all_excess_mean": float(_numeric(part["selected_minus_all_excess"]).mean()),
                "selected_minus_all_excess_t": _t_stat(part["selected_minus_all_excess"]),
                "selected_minus_rejected_excess_mean": float(_numeric(part["selected_minus_rejected_excess"]).mean()),
                "selected_hit_rate_mean": float(_numeric(part["selected_hit_rate"]).mean()),
                "selected_top20_rate_mean": float(_numeric(part["selected_top20_rate"]).mean()),
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values(
        ["candidate_pool_version", "selected_minus_all_excess_mean", "selected_minus_rejected_excess_mean"],
        ascending=[True, False, False],
    )


def build_report(summary: pd.DataFrame, monthly: pd.DataFrame, artifacts: list[str]) -> str:
    generated_at = pd.Timestamp.utcnow().isoformat()
    gate_view = summary.copy()
    if not gate_view.empty:
        gate_view = gate_view[
            [
                "candidate_pool_version",
                "filter_name",
                "months",
                "mean_after_count",
                "min_after_count",
                "months_below_60",
                "mean_retention_ratio",
                "selected_minus_all_excess_mean",
                "selected_minus_all_excess_t",
                "selected_minus_rejected_excess_mean",
                "selected_hit_rate_mean",
                "selected_top20_rate_mean",
            ]
        ]
    conclusion = "无有效过滤诊断结果。"
    if not summary.empty:
        bull = summary[summary["filter_name"].eq("bull_state")]
        stable = summary[summary["filter_name"].isin(["stable_q70", "stable_q80"])]
        bull_best = float(_numeric(bull["selected_minus_all_excess_mean"]).max()) if not bull.empty else np.nan
        stable_best_t = float(_numeric(stable["selected_minus_all_excess_t"]).max()) if not stable.empty else np.nan
        if pd.notna(bull_best) and bull_best < 0 and pd.notna(stable_best_t) and stable_best_t < 2.0:
            conclusion = (
                "`bull_state` 过滤为负向；稳定性过滤虽有弱正增量，但统计强度不足，"
                "不支持进入 M8 Baseline Gate。"
            )
        else:
            conclusion = "需结合 Summary 表人工复核是否满足进入 M8 Baseline Gate 的证据强度。"
    monthly_tail = monthly.sort_values(["candidate_pool_version", "filter_name", "signal_date"]).tail(24)
    artifact_lines = "\n".join(f"- `{x}`" for x in artifacts)
    return f"""# W7 Trend Filter Audit

- 生成时间：`{generated_at}`
- 目的：复查 T3 未通过后的候选池过滤支线，不进入 T4 主线回测。

## Summary

{format_markdown_table(gate_view, max_rows=40)}

## Conclusion

{conclusion}

## Recent Monthly Rows

{format_markdown_table(monthly_tail, max_rows=24)}

## 口径

- 样本为 `candidate_pool_pass == True` 且存在次月 open-to-open 标签的月度截面。
- `selected_minus_all_excess_mean` 是过滤后股票的次月市场超额均值减同月同池全集均值。
- `selected_minus_rejected_excess_mean` 是过滤后股票均值减被过滤股票均值。
- 这是过滤信号的窄口径诊断，不包含 M8 组合优化、行业约束或交易成本。

## Artifacts

{artifact_lines}
"""


def main() -> int:
    started_at = time.perf_counter()
    args = parse_args()
    loaded_config_path = resolve_loaded_config_path(args.config or None)
    cfg_raw = load_config(args.config or None)
    paths = cfg_raw.get("paths", {}) or {}
    dataset_path = resolve_project_path(args.dataset)
    db_path = resolve_project_path(args.duckdb_path.strip() or str(paths.get("duckdb_path") or "data/market.duckdb"))
    results_dir = resolve_project_path(args.results_dir.strip() or str(paths.get("results_dir") or "data/results"))
    reports_dir = ROOT / "docs" / "reports" / "2026-05"
    experiments_dir = resolve_project_path(str(paths.get("experiments_dir") or "data/experiments"))
    for directory in [results_dir, reports_dir, experiments_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    pools = parse_str_list(args.candidate_pools)
    output_stem = f"{slugify_token(args.output_prefix)}_{pd.Timestamp.now().strftime('%Y-%m-%d')}"
    research_config_id = (
        f"dataset_{slugify_token(dataset_path.stem)}"
        f"_trend_filter_audit_pools_{'-'.join(slugify_token(x) for x in pools)}"
    )
    identity = make_research_identity(
        result_type="w7_trend_filter_audit",
        research_topic="w7_trend_filter_audit",
        research_config_id=research_config_id,
        output_stem=output_stem,
    )

    dataset = load_baseline_dataset(dataset_path, candidate_pools=pools)
    dataset = attach_feature_families(dataset, db_path, ["trend_persistence"])
    monthly = build_monthly_filter_audit(dataset)
    summary = summarize_filter_audit(monthly)

    paths_out = {
        "summary": results_dir / f"{output_stem}_summary.csv",
        "monthly": results_dir / f"{output_stem}_monthly.csv",
        "summary_json": results_dir / f"{output_stem}_summary.json",
        "doc": reports_dir / f"{output_stem}.md",
        "manifest": results_dir / f"{output_stem}_manifest.json",
    }
    summary.to_csv(paths_out["summary"], index=False)
    monthly.to_csv(paths_out["monthly"], index=False)
    payload = {
        "research_config_id": research_config_id,
        "output_stem": output_stem,
        "candidate_pools": pools,
        "summary": summary.to_dict(orient="records"),
    }
    paths_out["summary_json"].write_text(
        json.dumps(json_sanitize(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    artifact_paths_raw = [project_relative(p) for key, p in paths_out.items() if key not in {"manifest", "doc"}]
    paths_out["doc"].write_text(
        build_report(summary, monthly, [*artifact_paths_raw, project_relative(paths_out["manifest"])]),
        encoding="utf-8",
    )

    valid = valid_pool_frame(dataset)
    min_signal_date = str(valid["signal_date"].min().date()) if not valid.empty else ""
    max_signal_date = str(valid["signal_date"].max().date()) if not valid.empty else ""
    best = summary.sort_values("selected_minus_all_excess_mean", ascending=False).head(1)
    best_row = best.iloc[0].to_dict() if not best.empty else {}
    finalize_research_contract(
        identity=identity,
        script_path=project_relative(Path(__file__).resolve()),
        started_at=started_at,
        config_source=project_relative(loaded_config_path) if loaded_config_path else "default_config_lookup",
        config_raw=cfg_raw,
        loaded_config_path=loaded_config_path,
        experiments_dir=experiments_dir,
        paths_out=paths_out,
        dataset_path=dataset_path,
        data_slice_kwargs=dict(
            dataset_name="w7_trend_filter_audit",
            source_tables=(project_relative(dataset_path), project_relative(db_path)),
            date_start=min_signal_date,
            date_end=max_signal_date,
            asof_trade_date=max_signal_date or None,
            signal_date_col="signal_date",
            symbol_col="symbol",
            candidate_pool_version=",".join(pools),
            rebalance_rule="M",
            execution_mode="tplus1_open",
            label_return_mode="open_to_open",
            feature_set_id="trend_persistence_filter_audit",
            feature_columns=(
                "feature_trend_bull_state",
                "feature_trend_flip_days_ago",
            ),
            label_columns=(LABEL_COL, EXCESS_COL, TOP20_COL),
            pit_policy="trend factors use daily rows with trade_date <= signal_date",
            config_path=project_relative(loaded_config_path) if loaded_config_path else "default_config_lookup",
            extra={
                "candidate_pool_rules": {p: POOL_RULES.get(p, "") for p in pools},
                "filters": [spec.name for spec in FILTER_SPECS],
            },
        ),
        metrics={
            "rows": int(len(dataset)),
            "valid_rows": int(len(valid)),
            "monthly_rows": int(len(monthly)),
            "summary_rows": int(len(summary)),
            "best_filter": str(best_row.get("filter_name") or ""),
            "best_candidate_pool_version": str(best_row.get("candidate_pool_version") or ""),
            "best_selected_minus_all_excess_mean": (
                float(best_row["selected_minus_all_excess_mean"])
                if best_row.get("selected_minus_all_excess_mean") is not None
                and pd.notna(best_row.get("selected_minus_all_excess_mean"))
                else None
            ),
        },
        gates={
            "data_gate": {"passed": not valid.empty, "valid_rows": int(len(valid))},
            "filter_count_gate": {
                "passed": bool((summary["months_below_60"] == 0).all()) if not summary.empty else False,
                "min_after_count": int(summary["min_after_count"].min()) if not summary.empty else 0,
            },
            "standalone_delta_gate": {
                "passed": bool((summary["selected_minus_all_excess_mean"] > 0).any()) if not summary.empty else False,
                "best_selected_minus_all_excess_mean": (
                    float(best_row["selected_minus_all_excess_mean"])
                    if best_row.get("selected_minus_all_excess_mean") is not None
                    and pd.notna(best_row.get("selected_minus_all_excess_mean"))
                    else None
                ),
            },
        },
        seed=None,
        promotion_blocking=["w7_stage_a_failed_filter_audit_only"],
        notes="Standalone filter audit after W7 Stage A IC gate failure; not a production candidate.",
        artifact_paths_raw=artifact_paths_raw,
    )

    print(f"[trend-filter-audit] summary={paths_out['summary']}")
    print(f"[trend-filter-audit] monthly={paths_out['monthly']}")
    print(f"[trend-filter-audit] report={paths_out['doc']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
