#!/usr/bin/env python3
"""M3：月度选股 oracle top-bucket 诊断。

输入 M2 canonical dataset：
``data/cache/monthly_selection_features.parquet``。

输出候选池 oracle 上限、特征分桶单调性、简单 baseline 与 oracle overlap、
realized market regime 切片和行业分布。
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.research_identity import make_research_identity, slugify_token
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
from src.settings import load_config, resolve_config_path

LABEL_COL = "label_forward_1m_o2o_return"
MARKET_COL = "label_market_ew_o2o_return"

FEATURE_SPECS: tuple[tuple[str, str, int], ...] = (
    ("momentum_20d", "feature_ret_20d_z", 1),
    ("momentum_60d", "feature_ret_60d_z", 1),
    ("short_momentum_5d", "feature_ret_5d_z", 1),
    ("low_vol_20d", "feature_realized_vol_20d_z", -1),
    ("liquidity_amount_20d", "feature_amount_20d_log_z", 1),
    ("turnover_20d", "feature_turnover_20d_z", 1),
    ("price_position_250d", "feature_price_position_250d_z", 1),
    ("low_limit_move_hits_20d", "feature_limit_move_hits_20d_z", -1),
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="运行月度选股 oracle top-bucket 诊断")
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--dataset", type=str, default="data/cache/monthly_selection_features.parquet")
    p.add_argument("--output-prefix", type=str, default="monthly_selection_oracle")
    p.add_argument("--results-dir", type=str, default="")
    p.add_argument("--top-k", type=str, default="20,30,50")
    p.add_argument("--bucket-count", type=int, default=5)
    p.add_argument("--candidate-pools", type=str, default="U0_all_tradable,U1_liquid_tradable,U2_risk_sane")
    return p.parse_args()


def _resolve_project_path(raw: str | Path) -> Path:
    p = Path(raw)
    return p if p.is_absolute() else ROOT / p


def _project_relative(path: str | Path) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(ROOT))
    except ValueError:
        return str(p)


def _parse_int_list(raw: str) -> list[int]:
    vals = []
    for item in str(raw).split(","):
        item = item.strip()
        if not item:
            continue
        vals.append(int(item))
    return sorted(set(vals))


def _parse_str_list(raw: str) -> list[str]:
    return [x.strip() for x in str(raw).split(",") if x.strip()]


def _json_sanitize(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_sanitize(v) for v in obj]
    if isinstance(obj, tuple):
        return [_json_sanitize(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        val = float(obj)
        return val if np.isfinite(val) else None
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    return obj


def _markdown_cell(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value).replace("|", "\\|").replace("\n", "<br>")


def _format_markdown_table(df: pd.DataFrame, *, max_rows: int = 30) -> str:
    if df.empty:
        return "_无记录_"
    view = df.head(max_rows).copy()
    header = "| " + " | ".join(str(c) for c in view.columns) + " |"
    sep = "| " + " | ".join("---" for _ in view.columns) + " |"
    rows = [
        "| " + " | ".join(_markdown_cell(row[col]) for col in view.columns) + " |"
        for _, row in view.iterrows()
    ]
    suffix = [f"\n_仅展示前 {max_rows} 行，共 {len(df)} 行。_"] if len(df) > max_rows else []
    return "\n".join([header, sep, *rows, *suffix])


def load_oracle_dataset(path: Path, *, candidate_pools: list[str] | None = None) -> pd.DataFrame:
    df = pd.read_parquet(path)
    required = {"signal_date", "symbol", "candidate_pool_version", "candidate_pool_pass", LABEL_COL}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"M2 dataset 缺少列: {missing}")
    out = df.copy()
    out["signal_date"] = pd.to_datetime(out["signal_date"], errors="coerce").dt.normalize()
    out["symbol"] = out["symbol"].astype(str).str.zfill(6)
    out[LABEL_COL] = pd.to_numeric(out[LABEL_COL], errors="coerce").replace([np.inf, -np.inf], np.nan)
    if MARKET_COL in out.columns:
        out[MARKET_COL] = pd.to_numeric(out[MARKET_COL], errors="coerce").replace([np.inf, -np.inf], np.nan)
    else:
        out[MARKET_COL] = np.nan
    if candidate_pools:
        out = out[out["candidate_pool_version"].isin(candidate_pools)].copy()
    return out


def valid_pool_frame(dataset: pd.DataFrame) -> pd.DataFrame:
    m = dataset["candidate_pool_pass"].astype(bool) & dataset[LABEL_COL].notna()
    return dataset.loc[m].copy()


def _top_by_return(part: pd.DataFrame, k: int) -> pd.DataFrame:
    return part.sort_values([LABEL_COL, "symbol"], ascending=[False, True]).head(int(k)).copy()


def build_oracle_topk_tables(
    dataset: pd.DataFrame,
    *,
    top_ks: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    valid = valid_pool_frame(dataset)
    monthly_rows: list[dict[str, Any]] = []
    holding_rows: list[dict[str, Any]] = []
    if valid.empty:
        return pd.DataFrame(), pd.DataFrame()
    for (pool, signal_date), part in valid.groupby(["candidate_pool_version", "signal_date"], sort=True):
        part = part.copy()
        pool_avg = float(part[LABEL_COL].mean())
        market_ret = float(part[MARKET_COL].dropna().iloc[0]) if part[MARKET_COL].notna().any() else np.nan
        for k in top_ks:
            top = _top_by_return(part, k)
            if top.empty:
                continue
            top_ret = pd.to_numeric(top[LABEL_COL], errors="coerce")
            monthly_rows.append(
                {
                    "signal_date": signal_date,
                    "candidate_pool_version": pool,
                    "top_k": int(k),
                    "candidate_pool_width": int(part["symbol"].nunique()),
                    "oracle_topk_count": int(len(top)),
                    "oracle_topk_mean_return": float(top_ret.mean()),
                    "oracle_topk_median_return": float(top_ret.median()),
                    "oracle_topk_min_return": float(top_ret.min()),
                    "candidate_pool_mean_return": pool_avg,
                    "market_ew_return": market_ret,
                    "oracle_topk_excess_vs_market": float(top_ret.mean() - market_ret)
                    if np.isfinite(market_ret)
                    else np.nan,
                    "oracle_topk_minus_pool_mean": float(top_ret.mean() - pool_avg),
                }
            )
            for rank, row in enumerate(top.itertuples(index=False), start=1):
                holding_rows.append(
                    {
                        "signal_date": signal_date,
                        "candidate_pool_version": pool,
                        "top_k": int(k),
                        "oracle_rank": int(rank),
                        "symbol": str(getattr(row, "symbol")).zfill(6),
                        "industry_level1": str(getattr(row, "industry_level1", "_UNKNOWN_") or "_UNKNOWN_"),
                        LABEL_COL: float(getattr(row, LABEL_COL)),
                    }
                )
    return pd.DataFrame(monthly_rows), pd.DataFrame(holding_rows)


def summarize_oracle_by_candidate_pool(monthly: pd.DataFrame) -> pd.DataFrame:
    if monthly.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for (pool, k), part in monthly.groupby(["candidate_pool_version", "top_k"], sort=True):
        excess = pd.to_numeric(part["oracle_topk_excess_vs_market"], errors="coerce")
        rows.append(
            {
                "candidate_pool_version": pool,
                "top_k": int(k),
                "months": int(part["signal_date"].nunique()),
                "median_candidate_pool_width": float(part["candidate_pool_width"].median()),
                "mean_oracle_topk_return": float(part["oracle_topk_mean_return"].mean()),
                "median_oracle_topk_return": float(part["oracle_topk_mean_return"].median()),
                "mean_oracle_excess_vs_market": float(excess.mean()),
                "median_oracle_excess_vs_market": float(excess.median()),
                "positive_oracle_excess_share": float((excess > 0).mean()) if excess.notna().any() else np.nan,
                "mean_oracle_minus_pool": float(part["oracle_topk_minus_pool_mean"].mean()),
            }
        )
    return pd.DataFrame(rows)


def summarize_candidate_pool_width(dataset: pd.DataFrame) -> pd.DataFrame:
    if dataset.empty:
        return pd.DataFrame()
    out = (
        dataset.groupby(["signal_date", "candidate_pool_version"], dropna=False)
        .agg(
            raw_universe_width=("symbol", "nunique"),
            candidate_pool_width=("candidate_pool_pass", "sum"),
            label_valid_count=(LABEL_COL, lambda s: pd.to_numeric(s, errors="coerce").notna().sum()),
        )
        .reset_index()
    )
    out["candidate_pool_pass_ratio"] = out["candidate_pool_width"] / out["raw_universe_width"].replace(0, np.nan)
    return out


def _safe_qcut(values: pd.Series, bucket_count: int) -> pd.Series:
    x = pd.to_numeric(values, errors="coerce")
    if x.notna().sum() < bucket_count or x.nunique(dropna=True) < 2:
        return pd.Series(pd.NA, index=values.index, dtype="Int64")
    try:
        b = pd.qcut(x, bucket_count, labels=False, duplicates="drop")
    except ValueError:
        return pd.Series(pd.NA, index=values.index, dtype="Int64")
    return (b + 1).astype("Int64")


def summarize_feature_bucket_monotonicity(
    dataset: pd.DataFrame,
    *,
    feature_specs: tuple[tuple[str, str, int], ...] = FEATURE_SPECS,
    bucket_count: int = 5,
) -> pd.DataFrame:
    valid = valid_pool_frame(dataset)
    rows: list[dict[str, Any]] = []
    for feature_name, col, direction in feature_specs:
        if col not in valid.columns:
            continue
        for (pool, signal_date), part in valid.groupby(["candidate_pool_version", "signal_date"], sort=True):
            tmp = part[["symbol", LABEL_COL, "label_future_top_20pct", col]].copy()
            tmp["_score"] = pd.to_numeric(tmp[col], errors="coerce") * int(direction)
            tmp["_bucket"] = _safe_qcut(tmp["_score"], bucket_count)
            tmp = tmp[tmp["_bucket"].notna()].copy()
            if tmp.empty:
                continue
            for bucket, bp in tmp.groupby("_bucket", sort=True):
                rows.append(
                    {
                        "signal_date": signal_date,
                        "candidate_pool_version": pool,
                        "feature": feature_name,
                        "feature_col": col,
                        "direction": int(direction),
                        "bucket": int(bucket),
                        "n": int(len(bp)),
                        "mean_forward_return": float(pd.to_numeric(bp[LABEL_COL], errors="coerce").mean()),
                        "median_forward_return": float(pd.to_numeric(bp[LABEL_COL], errors="coerce").median()),
                        "future_top20_share": float(pd.to_numeric(bp.get("label_future_top_20pct"), errors="coerce").mean()),
                    }
                )
    detail = pd.DataFrame(rows)
    if detail.empty:
        return detail
    agg = (
        detail.groupby(["candidate_pool_version", "feature", "feature_col", "direction", "bucket"], sort=True)
        .agg(
            months=("signal_date", "nunique"),
            n=("n", "sum"),
            mean_forward_return=("mean_forward_return", "mean"),
            median_forward_return=("median_forward_return", "median"),
            future_top20_share=("future_top20_share", "mean"),
        )
        .reset_index()
    )
    mono_rows: list[dict[str, Any]] = []
    for (pool, feature), part in agg.groupby(["candidate_pool_version", "feature"], sort=True):
        ordered = part.sort_values("bucket")
        ret_corr = (
            float(ordered["bucket"].corr(ordered["mean_forward_return"], method="spearman"))
            if len(ordered) >= 3
            else np.nan
        )
        hit_corr = (
            float(ordered["bucket"].corr(ordered["future_top20_share"], method="spearman"))
            if len(ordered) >= 3
            else np.nan
        )
        mono_rows.append(
            {
                "candidate_pool_version": pool,
                "feature": feature,
                "bucket_return_spearman": ret_corr,
                "bucket_top20_spearman": hit_corr,
                "bucket_count": int(ordered["bucket"].nunique()),
            }
        )
    mono = pd.DataFrame(mono_rows)
    return agg.merge(mono, on=["candidate_pool_version", "feature"], how="left")


def summarize_baseline_overlap(
    dataset: pd.DataFrame,
    *,
    top_ks: list[int],
    feature_specs: tuple[tuple[str, str, int], ...] = FEATURE_SPECS,
) -> pd.DataFrame:
    valid = valid_pool_frame(dataset)
    rows: list[dict[str, Any]] = []
    if valid.empty:
        return pd.DataFrame()
    for feature_name, col, direction in feature_specs:
        if col not in valid.columns:
            continue
        for (pool, signal_date), part in valid.groupby(["candidate_pool_version", "signal_date"], sort=True):
            part = part.copy()
            part["_baseline_score"] = pd.to_numeric(part[col], errors="coerce") * int(direction)
            part = part[np.isfinite(part["_baseline_score"])].copy()
            if part.empty:
                continue
            market_ret = float(part[MARKET_COL].dropna().iloc[0]) if part[MARKET_COL].notna().any() else np.nan
            oracle_top20 = set(part.loc[pd.to_numeric(part.get("label_future_top_20pct"), errors="coerce") == 1, "symbol"])
            for k in top_ks:
                oracle = _top_by_return(part, k)
                baseline = part.sort_values(["_baseline_score", "symbol"], ascending=[False, True]).head(k)
                oracle_symbols = set(oracle["symbol"].astype(str))
                baseline_symbols = set(baseline["symbol"].astype(str))
                overlap_count = len(oracle_symbols & baseline_symbols)
                top20_hits = len(baseline_symbols & oracle_top20)
                bret = pd.to_numeric(baseline[LABEL_COL], errors="coerce")
                rows.append(
                    {
                        "signal_date": signal_date,
                        "candidate_pool_version": pool,
                        "baseline": feature_name,
                        "feature_col": col,
                        "top_k": int(k),
                        "candidate_pool_width": int(part["symbol"].nunique()),
                        "oracle_topk_overlap_count": int(overlap_count),
                        "oracle_topk_overlap_share": float(overlap_count / max(len(oracle_symbols), 1)),
                        "oracle_top20_bucket_hit_count": int(top20_hits),
                        "oracle_top20_bucket_hit_share": float(top20_hits / max(len(baseline_symbols), 1)),
                        "baseline_topk_mean_return": float(bret.mean()),
                        "baseline_topk_excess_vs_market": float(bret.mean() - market_ret)
                        if np.isfinite(market_ret)
                        else np.nan,
                    }
                )
    detail = pd.DataFrame(rows)
    if detail.empty:
        return detail
    return (
        detail.groupby(["candidate_pool_version", "baseline", "feature_col", "top_k"], sort=True)
        .agg(
            months=("signal_date", "nunique"),
            median_candidate_pool_width=("candidate_pool_width", "median"),
            mean_oracle_topk_overlap_share=("oracle_topk_overlap_share", "mean"),
            median_oracle_topk_overlap_share=("oracle_topk_overlap_share", "median"),
            mean_oracle_top20_bucket_hit_share=("oracle_top20_bucket_hit_share", "mean"),
            mean_baseline_topk_return=("baseline_topk_mean_return", "mean"),
            mean_baseline_excess_vs_market=("baseline_topk_excess_vs_market", "mean"),
            positive_baseline_excess_share=("baseline_topk_excess_vs_market", lambda s: float((pd.to_numeric(s, errors="coerce") > 0).mean())),
        )
        .reset_index()
    )


def build_realized_market_states(dataset: pd.DataFrame) -> pd.DataFrame:
    base = dataset[dataset[LABEL_COL].notna()].copy()
    if base.empty:
        return pd.DataFrame(columns=["signal_date", "market_ew_return", "realized_market_state"])
    monthly = (
        base.groupby("signal_date", sort=True)[MARKET_COL]
        .first()
        .reset_index()
        .rename(columns={MARKET_COL: "market_ew_return"})
    )
    vals = pd.to_numeric(monthly["market_ew_return"], errors="coerce")
    lo = vals.quantile(0.20)
    hi = vals.quantile(0.80)
    monthly["realized_market_state"] = np.select(
        [vals <= lo, vals >= hi],
        ["strong_down", "strong_up"],
        default="neutral",
    )
    monthly["state_p20"] = float(lo) if np.isfinite(lo) else np.nan
    monthly["state_p80"] = float(hi) if np.isfinite(hi) else np.nan
    return monthly


def summarize_regime_oracle_capacity(monthly_oracle: pd.DataFrame, market_states: pd.DataFrame) -> pd.DataFrame:
    if monthly_oracle.empty or market_states.empty:
        return pd.DataFrame()
    states = market_states.rename(columns={"market_ew_return": "state_market_ew_return"})
    df = monthly_oracle.merge(states, on="signal_date", how="left")
    return (
        df.groupby(["candidate_pool_version", "top_k", "realized_market_state"], dropna=False, sort=True)
        .agg(
            months=("signal_date", "nunique"),
            mean_market_return=("state_market_ew_return", "mean"),
            mean_oracle_topk_return=("oracle_topk_mean_return", "mean"),
            median_oracle_topk_return=("oracle_topk_mean_return", "median"),
            mean_oracle_excess_vs_market=("oracle_topk_excess_vs_market", "mean"),
            median_oracle_excess_vs_market=("oracle_topk_excess_vs_market", "median"),
        )
        .reset_index()
    )


def summarize_industry_oracle_distribution(holdings: pd.DataFrame) -> pd.DataFrame:
    if holdings.empty:
        return pd.DataFrame()
    df = holdings.copy()
    total = (
        df.groupby(["signal_date", "candidate_pool_version", "top_k"], sort=True)["symbol"]
        .nunique()
        .rename("topk_count")
        .reset_index()
    )
    ind = (
        df.groupby(["signal_date", "candidate_pool_version", "top_k", "industry_level1"], dropna=False, sort=True)[
            "symbol"
        ]
        .nunique()
        .rename("industry_count")
        .reset_index()
        .merge(total, on=["signal_date", "candidate_pool_version", "top_k"], how="left")
    )
    ind["industry_share"] = ind["industry_count"] / ind["topk_count"].replace(0, np.nan)
    return ind


def build_summary_payload(
    *,
    oracle_summary: pd.DataFrame,
    feature_buckets: pd.DataFrame,
    baseline_overlap: pd.DataFrame,
    quality: dict[str, Any],
) -> dict[str, Any]:
    best_oracle = pd.DataFrame()
    if not oracle_summary.empty:
        best_oracle = oracle_summary.sort_values(
            ["top_k", "mean_oracle_excess_vs_market"],
            ascending=[True, False],
        ).groupby("top_k", as_index=False).head(3)
    best_features = pd.DataFrame()
    if not feature_buckets.empty:
        feature_summary = feature_buckets.drop_duplicates(
            ["candidate_pool_version", "feature", "bucket_return_spearman"]
        )
        best_features = feature_summary.sort_values("bucket_return_spearman", ascending=False).head(10)
    best_baselines = pd.DataFrame()
    if not baseline_overlap.empty:
        best_baselines = baseline_overlap.sort_values("mean_baseline_excess_vs_market", ascending=False).head(10)
    return {
        "quality": quality,
        "top_oracle_by_topk": best_oracle.to_dict(orient="records"),
        "top_feature_monotonicity": best_features.to_dict(orient="records"),
        "top_baseline_overlap": best_baselines.to_dict(orient="records"),
    }


def build_doc(
    *,
    quality: dict[str, Any],
    oracle_summary: pd.DataFrame,
    feature_buckets: pd.DataFrame,
    baseline_overlap: pd.DataFrame,
    regime_oracle: pd.DataFrame,
    industry_distribution: pd.DataFrame,
    artifacts: list[str],
) -> str:
    generated_at = pd.Timestamp.utcnow().isoformat()
    oracle_view = oracle_summary.sort_values(["top_k", "mean_oracle_excess_vs_market"], ascending=[True, False])
    feature_view = pd.DataFrame()
    if not feature_buckets.empty:
        feature_view = (
            feature_buckets.drop_duplicates(
                ["candidate_pool_version", "feature", "bucket_return_spearman", "bucket_top20_spearman"]
            )
            .sort_values("bucket_return_spearman", ascending=False)
            .head(20)
        )
    baseline_view = (
        baseline_overlap.sort_values("mean_baseline_excess_vs_market", ascending=False).head(20)
        if not baseline_overlap.empty
        else baseline_overlap
    )
    industry_view = pd.DataFrame()
    if not industry_distribution.empty:
        industry_view = (
            industry_distribution.groupby(["candidate_pool_version", "top_k", "industry_level1"], sort=True)
            .agg(mean_share=("industry_share", "mean"), months=("signal_date", "nunique"))
            .reset_index()
            .sort_values(["top_k", "candidate_pool_version", "mean_share"], ascending=[True, True, False])
            .head(30)
        )
    artifact_lines = "\n".join(f"- `{x}`" for x in artifacts)
    return f"""# Monthly Selection Oracle

- 生成时间：`{generated_at}`
- 结果类型：`monthly_selection_oracle`
- 研究主题：`{quality.get('research_topic', '')}`
- 研究配置：`{quality.get('research_config_id', '')}`
- 输出 stem：`{quality.get('output_stem', '')}`
- 数据集：`{quality.get('dataset_path', '')}`
- 有效标签月份：`{quality.get('valid_signal_months', 0)}`

## Oracle By Candidate Pool

{_format_markdown_table(oracle_view)}

## Feature Bucket Monotonicity

{_format_markdown_table(feature_view)}

## Baseline Overlap

{_format_markdown_table(baseline_view)}

## Interpretation Note

- 不应过度在意因子或模型是否命中事后 oracle Top-20。Oracle Top-20 是候选空间上限和可分性诊断，不是训练目标，也不是策略准入 gate。
- 可交易月度选股的核心目标是稳定改善推荐 Top-K 的收益分布，包括 `topk_excess_mean`、`topk_hit_rate`、`topk_minus_nextk`、分桶 spread、年度/市场状态稳定性和成本后表现。
- 一个模型可以几乎不命中每个月事后最强的 20 只股票，但只要它稳定避开差股票、提高 Top-K 平均收益并控制回撤/换手，就仍然是有效研究候选。
- 因此，`baseline_overlap` 应只用于理解“现有特征离 oracle 上限有多远”，不能用于否定能盈利的非 oracle-mimic 排序模型。

## Regime Oracle Capacity

{_format_markdown_table(regime_oracle)}

## Industry Oracle Distribution

{_format_markdown_table(industry_view)}

## 口径

- Oracle Top-K：每个 `signal_date`、每个候选池内，按未来 `label_forward_1m_o2o_return` 事后排序取 Top-K。
- Oracle overlap 只作诊断，不作为主评价指标；模型不需要命中每个月事后最强的 Top-20，只要稳定提高可交易 Top-K 收益分布即可。
- `realized_market_state` 使用同一持有期市场等权收益的全样本 20%/80% 分位切片，仅用于 oracle capacity 归因，不作为可交易信号。
- Baseline overlap 使用单因子截面排序与 oracle Top-K / future top 20% bucket 对比。
- 特征分桶使用已在 M2 中按月截面 winsorize/z-score 的特征列。

## 本轮产物

{artifact_lines}
"""


def main() -> int:
    started_at = time.perf_counter()
    args = parse_args()
    cfg = load_config(args.config)
    paths = cfg.get("paths", {}) or {}
    config_source = str(resolve_config_path(args.config)) if args.config is not None else "default_config_lookup"
    dataset_path = _resolve_project_path(args.dataset)
    results_dir_raw = args.results_dir.strip() or str(paths.get("results_dir") or "data/results")
    results_dir = _resolve_project_path(results_dir_raw)
    docs_dir = ROOT / "docs"
    results_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)
    experiments_dir = _resolve_project_path(
        str(paths.get("experiments_dir") or "data/experiments")
    )
    experiments_dir.mkdir(parents=True, exist_ok=True)

    top_ks = _parse_int_list(args.top_k)
    pools = _parse_str_list(args.candidate_pools)
    output_stem = f"{slugify_token(args.output_prefix)}_{pd.Timestamp.now().strftime('%Y-%m-%d')}"
    research_topic = "monthly_selection_oracle"
    research_config_id = (
        f"dataset_{slugify_token(dataset_path.stem)}"
        f"_pools_{'-'.join(slugify_token(x) for x in pools)}"
        f"_topk_{'-'.join(str(x) for x in top_ks)}"
        f"_buckets_{int(args.bucket_count)}"
    )

    identity = make_research_identity(
        result_type="monthly_selection_oracle",
        research_topic=research_topic,
        research_config_id=research_config_id,
        output_stem=output_stem,
        canonical_config_name="monthly_selection_oracle_v1",
    )
    loaded_config_path = resolve_config_path(args.config) if args.config is not None else None

    dataset = load_oracle_dataset(dataset_path, candidate_pools=pools)
    valid = valid_pool_frame(dataset)
    monthly_oracle, oracle_holdings = build_oracle_topk_tables(dataset, top_ks=top_ks)
    oracle_summary = summarize_oracle_by_candidate_pool(monthly_oracle)
    feature_buckets = summarize_feature_bucket_monotonicity(dataset, bucket_count=int(args.bucket_count))
    baseline_overlap = summarize_baseline_overlap(dataset, top_ks=top_ks)
    market_states = build_realized_market_states(dataset)
    regime_oracle = summarize_regime_oracle_capacity(monthly_oracle, market_states)
    industry_distribution = summarize_industry_oracle_distribution(oracle_holdings)
    candidate_width = summarize_candidate_pool_width(dataset)

    quality = {
        "result_type": "monthly_selection_oracle",
        "research_topic": research_topic,
        "research_config_id": research_config_id,
        "output_stem": output_stem,
        "config_source": config_source,
        "dataset_path": str(dataset_path.relative_to(ROOT)) if dataset_path.is_relative_to(ROOT) else str(dataset_path),
        "candidate_pools": pools,
        "top_ks": top_ks,
        "bucket_count": int(args.bucket_count),
        "rows": int(len(dataset)),
        "valid_rows": int(len(valid)),
        "valid_signal_months": int(valid["signal_date"].nunique()) if not valid.empty else 0,
        "min_valid_signal_date": str(valid["signal_date"].min().date()) if not valid.empty else "",
        "max_valid_signal_date": str(valid["signal_date"].max().date()) if not valid.empty else "",
    }

    paths_out = {
        "summary_json": results_dir / f"{output_stem}_summary.json",
        "oracle_topk_return_by_month": results_dir / f"{output_stem}_oracle_topk_return_by_month.csv",
        "oracle_topk_holdings": results_dir / f"{output_stem}_oracle_topk_holdings.csv",
        "oracle_topk_by_candidate_pool": results_dir / f"{output_stem}_oracle_topk_by_candidate_pool.csv",
        "feature_bucket_monotonicity": results_dir / f"{output_stem}_feature_bucket_monotonicity.csv",
        "baseline_overlap": results_dir / f"{output_stem}_baseline_overlap.csv",
        "regime_oracle_capacity": results_dir / f"{output_stem}_regime_oracle_capacity.csv",
        "industry_oracle_distribution": results_dir / f"{output_stem}_industry_oracle_distribution.csv",
        "candidate_pool_width": results_dir / f"{output_stem}_candidate_pool_width.csv",
        "market_states": results_dir / f"{output_stem}_market_states.csv",
        "manifest": results_dir / f"{output_stem}_manifest.json",
        "doc": docs_dir / f"{output_stem}.md",
    }

    monthly_oracle.to_csv(paths_out["oracle_topk_return_by_month"], index=False)
    oracle_holdings.to_csv(paths_out["oracle_topk_holdings"], index=False)
    oracle_summary.to_csv(paths_out["oracle_topk_by_candidate_pool"], index=False)
    feature_buckets.to_csv(paths_out["feature_bucket_monotonicity"], index=False)
    baseline_overlap.to_csv(paths_out["baseline_overlap"], index=False)
    regime_oracle.to_csv(paths_out["regime_oracle_capacity"], index=False)
    industry_distribution.to_csv(paths_out["industry_oracle_distribution"], index=False)
    candidate_width.to_csv(paths_out["candidate_pool_width"], index=False)
    market_states.to_csv(paths_out["market_states"], index=False)

    summary_payload = build_summary_payload(
        oracle_summary=oracle_summary,
        feature_buckets=feature_buckets,
        baseline_overlap=baseline_overlap,
        quality=quality,
    )
    paths_out["summary_json"].write_text(
        json.dumps(_json_sanitize(summary_payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    artifact_paths = [
        str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)
        for key, p in paths_out.items()
        if key not in {"manifest", "doc"}
    ]
    manifest = {
        "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
        **quality,
        "artifacts": [*artifact_paths, str(paths_out["doc"].relative_to(ROOT))],
    }
    paths_out["manifest"].write_text(
        json.dumps(_json_sanitize(manifest), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    paths_out["doc"].write_text(
        build_doc(
            quality=quality,
            oracle_summary=oracle_summary,
            feature_buckets=feature_buckets,
            baseline_overlap=baseline_overlap,
            regime_oracle=regime_oracle,
            industry_distribution=industry_distribution,
            artifacts=[*artifact_paths, str(paths_out["manifest"].relative_to(ROOT))],
        ),
        encoding="utf-8",
    )

    # --- research contract ---
    min_signal_date = str(dataset["signal_date"].min().date()) if not dataset.empty else ""
    max_signal_date = str(dataset["signal_date"].max().date()) if not dataset.empty else ""
    data_slice = DataSlice(
        dataset_name="monthly_selection_oracle",
        source_tables=(_project_relative(dataset_path),),
        date_start=min_signal_date,
        date_end=max_signal_date,
        asof_trade_date=max_signal_date or None,
        signal_date_col="signal_date",
        symbol_col="symbol",
        candidate_pool_version=",".join(pools),
        rebalance_rule="M",
        execution_mode="tplus1_open",
        label_return_mode="open_to_open",
        feature_set_id=None,
        feature_columns=tuple(col for _, col, _ in FEATURE_SPECS),
        label_columns=(LABEL_COL, MARKET_COL),
        pit_policy="oracle uses ex-post label_forward_1m_o2o_return for upper-bound diagnosis only",
        config_path=config_source,
        extra={
            "dataset_path": _project_relative(dataset_path),
            "top_ks": top_ks,
            "bucket_count": int(args.bucket_count),
        },
    )

    artifact_refs: list[ArtifactRef] = []
    for key, p in paths_out.items():
        if key == "manifest":
            artifact_refs.append(ArtifactRef("manifest_json", _project_relative(p), "json"))
        elif key == "doc":
            artifact_refs.append(ArtifactRef("report_md", _project_relative(p), "md"))
        else:
            artifact_refs.append(ArtifactRef(f"{key}_csv", _project_relative(p), "csv"))
    artifact_refs = tuple(artifact_refs)

    metrics = {
        "rows": int(quality["rows"]),
        "valid_rows": int(quality["valid_rows"]),
        "valid_signal_months": int(quality["valid_signal_months"]),
        "oracle_summary_rows": int(len(oracle_summary)),
        "feature_bucket_rows": int(len(feature_buckets)),
    }

    gates = {
        "data_gate": {
            "passed": bool(metrics["valid_rows"] > 0 and metrics["valid_signal_months"] > 0),
            "checks": {
                "has_valid_rows": metrics["valid_rows"] > 0,
                "has_valid_signal_months": metrics["valid_signal_months"] > 0,
            },
        },
        "governance_gate": {
            "passed": True,
            "manifest_schema": "research_result_v1",
        },
    }

    config_info = config_snapshot(
        config_path=loaded_config_path,
        resolved_config=cfg,
        sections=("paths", "database", "signals", "monthly_selection"),
    )
    config_info["config_path"] = config_source

    result = ExperimentResult(
        result_id=build_result_id(identity, [data_slice], metrics),
        identity=identity,
        script_name=_project_relative(Path(__file__).resolve()),
        command=shlex.join([sys.executable, *sys.argv]),
        created_at=utc_now_iso(),
        duration_sec=round(time.perf_counter() - started_at, 6),
        seed=None,
        data_slices=(data_slice,),
        config=config_info,
        params={
            "cli": vars(args),
            "overrides": {
                key: value
                for key, value in {
                    "dataset": args.dataset,
                    "results_dir": args.results_dir.strip(),
                    "top_k": args.top_k,
                    "candidate_pools": args.candidate_pools,
                    "bucket_count": args.bucket_count,
                }.items()
                if value
            },
        },
        metrics=metrics,
        gates=gates,
        artifacts=artifact_refs,
        promotion={
            "production_eligible": False,
            "registry_status": "not_registered",
            "blocking_reasons": ["oracle_diagnostic_only_not_promotion_candidate"],
        },
        notes="Monthly selection M3 oracle top-bucket diagnostic; uses ex-post labels for upper-bound estimation only.",
    )
    write_research_manifest(
        paths_out["manifest"],
        result,
        extra={
            "generated_at_utc": result.created_at,
            "result_type": "monthly_selection_oracle_manifest",
            "research_topic": identity.research_topic,
            "research_config_id": identity.research_config_id,
            "output_stem": identity.output_stem,
            "config_source": config_source,
            "dataset_path": _project_relative(dataset_path),
            "candidate_pools": pools,
            "top_ks": top_ks,
            "bucket_count": int(args.bucket_count),
            "legacy_artifacts": [*artifact_paths, str(paths_out["doc"].relative_to(ROOT))],
        },
    )
    append_experiment_result(experiments_dir, result)

    print(f"[monthly-oracle] valid_rows={quality['valid_rows']} valid_months={quality['valid_signal_months']}")
    print(f"[monthly-oracle] oracle_summary={paths_out['oracle_topk_by_candidate_pool']}")
    print(f"[monthly-oracle] doc={paths_out['doc']}")
    print(f"[monthly-oracle] manifest={paths_out['manifest']}")
    print(f"[monthly-oracle] research_index={experiments_dir / 'research_results.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
