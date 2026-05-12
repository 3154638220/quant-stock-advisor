#!/usr/bin/env python3
"""Audit W7 trend-persistence factors under original and reversed directions."""

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
from src.features.trend_persistence_factors import TREND_PERSISTENCE_FACTOR_COLS
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
class ReverseGate:
    min_ic_mean: float = 0.01
    min_ic_ir: float = 0.30
    min_positive_month_ratio: float = 0.55


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="W7 trend-persistence reverse-direction IC audit")
    p.add_argument("--config", type=str, default="")
    p.add_argument("--dataset", type=str, default="data/cache/monthly_selection_features.parquet")
    p.add_argument("--duckdb-path", type=str, default="")
    p.add_argument("--results-dir", type=str, default="")
    p.add_argument("--output-prefix", type=str, default="w7_trend_reverse_ic_audit")
    p.add_argument("--candidate-pools", type=str, default="U1_liquid_tradable,U2_risk_sane")
    p.add_argument("--label-col", type=str, default=EXCESS_COL)
    p.add_argument("--min-samples", type=int, default=30)
    p.add_argument("--bucket-count", type=int, default=5)
    return p.parse_args()


def _numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _rank_ic(x: pd.Series, y: pd.Series) -> float:
    m = x.notna() & y.notna()
    if int(m.sum()) < 3:
        return np.nan
    x_m = x.loc[m]
    y_m = y.loc[m]
    if x_m.nunique(dropna=True) < 2 or y_m.nunique(dropna=True) < 2:
        return np.nan
    return float(x_m.corr(y_m, method="spearman"))


def _t_stat(values: pd.Series) -> float:
    vals = _numeric(values).dropna()
    if len(vals) < 2:
        return np.nan
    std = float(vals.std(ddof=1))
    if std == 0.0:
        return np.nan
    return float(vals.mean() / (std / np.sqrt(len(vals))))


def _bucket_spread(score: pd.Series, label: pd.Series, bucket_count: int) -> float:
    s = pd.DataFrame({"score": _numeric(score), "label": _numeric(label)}).dropna()
    if len(s) < max(bucket_count * 2, 3):
        return np.nan
    if s["score"].nunique(dropna=True) < bucket_count:
        return np.nan
    try:
        s["_bucket"] = pd.qcut(s["score"], bucket_count, labels=False, duplicates="drop")
    except ValueError:
        return np.nan
    if s["_bucket"].nunique(dropna=True) < 2:
        return np.nan
    bucket_mean = s.groupby("_bucket", observed=True)["label"].mean()
    return float(bucket_mean.iloc[-1] - bucket_mean.iloc[0])


def build_monthly_reverse_ic(
    dataset: pd.DataFrame,
    *,
    factor_cols: tuple[str, ...] = TREND_PERSISTENCE_FACTOR_COLS,
    label_col: str = EXCESS_COL,
    min_samples: int = 30,
    bucket_count: int = 5,
) -> pd.DataFrame:
    """Build monthly IC rows for original and reversed trend factor directions."""
    valid = valid_pool_frame(dataset)
    if valid.empty or label_col not in valid.columns:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for (signal_date, pool), part in valid.groupby(["signal_date", "candidate_pool_version"], sort=True):
        label = _numeric(part[label_col])
        for factor in factor_cols:
            if factor not in part.columns:
                continue
            base = _numeric(part[factor])
            coverage = float(base.notna().mean()) if len(base) else np.nan
            if int((base.notna() & label.notna()).sum()) < int(min_samples):
                continue
            for direction_name, score in (("original", base), ("reversed", -base)):
                rows.append(
                    {
                        "signal_date": pd.Timestamp(signal_date).date().isoformat(),
                        "candidate_pool_version": pool,
                        "factor": factor,
                        "direction_variant": direction_name,
                        "rank_ic": _rank_ic(score, label),
                        "top_bottom_excess_spread": _bucket_spread(score, label, int(bucket_count)),
                        "sample_count": int((score.notna() & label.notna()).sum()),
                        "coverage": coverage,
                    }
                )
    return pd.DataFrame(rows)


def summarize_reverse_ic(monthly: pd.DataFrame, gate: ReverseGate = ReverseGate()) -> pd.DataFrame:
    if monthly.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for (pool, factor, direction), part in monthly.groupby(
        ["candidate_pool_version", "factor", "direction_variant"],
        sort=True,
    ):
        ic = _numeric(part["rank_ic"]).dropna()
        spread = _numeric(part["top_bottom_excess_spread"]).dropna()
        ic_std = float(ic.std(ddof=1)) if len(ic) >= 2 else np.nan
        ic_mean = float(ic.mean()) if len(ic) else np.nan
        ic_ir = float(ic_mean / ic_std) if pd.notna(ic_mean) and pd.notna(ic_std) and ic_std > 0 else np.nan
        positive_month_ratio = float((ic > 0).mean()) if len(ic) else np.nan
        passed_reverse_gate = (
            direction == "reversed"
            and pd.notna(ic_mean)
            and pd.notna(ic_ir)
            and pd.notna(positive_month_ratio)
            and ic_mean >= gate.min_ic_mean
            and ic_ir >= gate.min_ic_ir
            and positive_month_ratio >= gate.min_positive_month_ratio
        )
        rows.append(
            {
                "candidate_pool_version": pool,
                "factor": factor,
                "direction_variant": direction,
                "months": int(part["signal_date"].nunique()),
                "mean_sample_count": float(part["sample_count"].mean()),
                "mean_coverage": float(_numeric(part["coverage"]).mean()),
                "rank_ic_mean": ic_mean,
                "rank_ic_std": ic_std,
                "ic_ir": ic_ir,
                "positive_month_ratio": positive_month_ratio,
                "top_bottom_excess_spread_mean": float(spread.mean()) if len(spread) else np.nan,
                "top_bottom_excess_spread_t": _t_stat(spread),
                "reverse_gate_pass": bool(passed_reverse_gate),
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values(
        ["candidate_pool_version", "reverse_gate_pass", "rank_ic_mean", "ic_ir"],
        ascending=[True, False, False, False],
    )


def build_report(summary: pd.DataFrame, monthly: pd.DataFrame, artifacts: list[str]) -> str:
    generated_at = pd.Timestamp.utcnow().isoformat()
    gate_view = summary.copy()
    if not gate_view.empty:
        gate_view = gate_view[
            [
                "candidate_pool_version",
                "factor",
                "direction_variant",
                "months",
                "rank_ic_mean",
                "ic_ir",
                "positive_month_ratio",
                "top_bottom_excess_spread_mean",
                "top_bottom_excess_spread_t",
                "reverse_gate_pass",
            ]
        ]
    conclusion = "无有效反向 IC 诊断结果。"
    if not summary.empty:
        passed = summary[summary["reverse_gate_pass"].astype(bool)]
        if passed.empty:
            best_rev = summary[summary["direction_variant"].eq("reversed")].head(3)
            best_txt = ", ".join(
                f"{r.factor}@{r.candidate_pool_version}: IC={r.rank_ic_mean:.4f}, IR={r.ic_ir:.2f}"
                for r in best_rev.itertuples()
                if pd.notna(r.rank_ic_mean) and pd.notna(r.ic_ir)
            )
            conclusion = (
                "反向方向未通过离线门槛；可作为过热解释保留在研究记录中，"
                "但不支持注册为生产正向因子。"
                + (f" 最强反向项：{best_txt}。" if best_txt else "")
            )
        else:
            conclusion = (
                "存在反向方向通过离线门槛的趋势项；仍需独立 Stage A 复跑和 M8 Gate，"
                "不能直接推广到生产配置。"
            )
    monthly_tail = monthly.sort_values(["candidate_pool_version", "factor", "direction_variant", "signal_date"]).tail(24)
    artifact_lines = "\n".join(f"- `{x}`" for x in artifacts)
    return f"""# W7 Trend Reverse IC Audit

- 生成时间：`{generated_at}`
- 目的：在 W7 Stage A 失败后，离线验证“多头趋势状态更像短期过热信号”的反向假设。

## Summary

{format_markdown_table(gate_view, max_rows=40)}

## Conclusion

{conclusion}

## Recent Monthly Rows

{format_markdown_table(monthly_tail, max_rows=24)}

## 口径

- 样本为 `candidate_pool_pass == True` 且存在月度 open-to-open 标签的截面。
- `original` 使用因子原始方向；`reversed` 使用 `-factor`，用于检验过热/反向假设。
- `rank_ic` 使用每月截面 Spearman 相关；默认标签为 `{EXCESS_COL}`。
- `top_bottom_excess_spread` 是按方向分桶后最高桶均值减最低桶均值。
- 这是离线诊断，不改生产注册方向，不进入 M8 Baseline Gate。

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
        f"_trend_reverse_ic_pools_{'-'.join(slugify_token(x) for x in pools)}"
        f"_label_{slugify_token(args.label_col)}"
    )
    identity = make_research_identity(
        result_type="w7_trend_reverse_ic_audit",
        research_topic="w7_trend_reverse_ic_audit",
        research_config_id=research_config_id,
        output_stem=output_stem,
    )

    dataset = load_baseline_dataset(dataset_path, candidate_pools=pools)
    dataset = attach_feature_families(dataset, db_path, ["trend_persistence"])
    monthly = build_monthly_reverse_ic(
        dataset,
        label_col=args.label_col,
        min_samples=int(args.min_samples),
        bucket_count=int(args.bucket_count),
    )
    summary = summarize_reverse_ic(monthly)

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
        "label_col": args.label_col,
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
    reverse_summary = summary[summary["direction_variant"].eq("reversed")] if not summary.empty else pd.DataFrame()
    best = reverse_summary.sort_values(["rank_ic_mean", "ic_ir"], ascending=[False, False]).head(1)
    best_row = best.iloc[0].to_dict() if not best.empty else {}
    any_reverse_gate_pass = bool(summary["reverse_gate_pass"].any()) if not summary.empty else False
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
            dataset_name="w7_trend_reverse_ic_audit",
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
            feature_set_id="trend_persistence_reverse_ic_audit",
            feature_columns=TREND_PERSISTENCE_FACTOR_COLS,
            label_columns=(LABEL_COL, EXCESS_COL, TOP20_COL),
            pit_policy="trend factors use daily rows with trade_date <= signal_date",
            config_path=project_relative(loaded_config_path) if loaded_config_path else "default_config_lookup",
            extra={
                "candidate_pool_rules": {p: POOL_RULES.get(p, "") for p in pools},
                "label_col": args.label_col,
                "direction_variants": ["original", "reversed"],
            },
        ),
        metrics={
            "rows": int(len(dataset)),
            "valid_rows": int(len(valid)),
            "monthly_rows": int(len(monthly)),
            "summary_rows": int(len(summary)),
            "best_reversed_factor": str(best_row.get("factor") or ""),
            "best_reversed_candidate_pool_version": str(best_row.get("candidate_pool_version") or ""),
            "best_reversed_rank_ic_mean": (
                float(best_row["rank_ic_mean"])
                if best_row.get("rank_ic_mean") is not None and pd.notna(best_row.get("rank_ic_mean"))
                else None
            ),
        },
        gates={
            "data_gate": {"passed": not valid.empty, "valid_rows": int(len(valid))},
            "reverse_direction_gate": {
                "passed": any_reverse_gate_pass,
                "criteria": {
                    "rank_ic_mean_gte": ReverseGate.min_ic_mean,
                    "ic_ir_gte": ReverseGate.min_ic_ir,
                    "positive_month_ratio_gte": ReverseGate.min_positive_month_ratio,
                },
            },
        },
        seed=None,
        promotion_blocking=["w7_stage_a_failed_reverse_audit_only"],
        notes="Offline reverse-direction diagnostic after W7 Stage A failure; not a production candidate.",
        artifact_paths_raw=artifact_paths_raw,
    )

    print(f"[trend-reverse-ic-audit] summary={paths_out['summary']}")
    print(f"[trend-reverse-ic-audit] monthly={paths_out['monthly']}")
    print(f"[trend-reverse-ic-audit] report={paths_out['doc']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
