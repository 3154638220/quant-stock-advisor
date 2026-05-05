"""M3 oracle analysis functions extracted from scripts/run_monthly_selection_oracle.py."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.pipeline.monthly_baselines import (
    _safe_qcut,
    valid_pool_frame,
)
from src.reporting.markdown_report import format_markdown_table
from src.research.gates import LABEL_COL, MARKET_COL

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


def build_oracle_doc(
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

{format_markdown_table(oracle_view)}

## Feature Bucket Monotonicity

{format_markdown_table(feature_view)}

## Baseline Overlap

{format_markdown_table(baseline_view)}

## Interpretation Note

- 不应过度在意因子或模型是否命中事后 oracle Top-20。Oracle Top-20 是候选空间上限和可分性诊断，不是训练目标，也不是策略准入 gate。
- 可交易月度选股的核心目标是稳定改善推荐 Top-K 的收益分布，包括 `topk_excess_mean`、`topk_hit_rate`、`topk_minus_nextk`、分桶 spread、年度/市场状态稳定性和成本后表现。
- 一个模型可以几乎不命中每个月事后最强的 20 只股票，但只要它稳定避开差股票、提高 Top-K 平均收益并控制回撤/换手，就仍然是有效研究候选。
- 因此，`baseline_overlap` 应只用于理解"现有特征离 oracle 上限有多远"，不能用于否定能盈利的非 oracle-mimic 排序模型。

## Regime Oracle Capacity

{format_markdown_table(regime_oracle)}

## Industry Oracle Distribution

{format_markdown_table(industry_view)}

## 口径

- Oracle Top-K：每个 `signal_date`、每个候选池内，按未来 `label_forward_1m_o2o_return` 事后排序取 Top-K。
- Oracle overlap 只作诊断，不作为主评价指标；模型不需要命中每个月事后最强的 Top-20，只要稳定提高可交易 Top-K 收益分布即可。
- `realized_market_state` 使用同一持有期市场等权收益的全样本 20%/80% 分位切片，仅用于 oracle capacity 归因，不作为可交易信号。
- Baseline overlap 使用单因子截面排序与 oracle Top-K / future top 20% bucket 对比。
- 特征分桶使用已在 M2 中按月截面 winsorize/z-score 的特征列。

## 本轮产物

{artifact_lines}
"""
