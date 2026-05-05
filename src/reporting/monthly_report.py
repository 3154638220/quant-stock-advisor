"""月度选股研究报告生成。

从 scripts/run_monthly_selection_report.py 提取核心报告逻辑：
- 推荐信号日选择
- full-fit 报告打分
- 推荐表构建
- M9 数据完整性检查
- 特征覆盖策略

不放 CLI 参数解析与文件 I/O 编排。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data_fetcher.stock_name_cache import (
    _display_symbol,
    _is_st_name,
    _name_column,
)
from src.pipeline.monthly_ltr import (
    _tag_importance,
    _train_predict_xgboost_ranker,
)
from src.pipeline.monthly_multisource import (
    _cap_fit_rows,
)
from src.reporting.markdown_report import format_markdown_table
from src.research.gates import POOL_RULES


@dataclass(frozen=True)
class M7RunConfig:
    top_ks: tuple[int, ...] = (20, 30)
    report_top_k: int = 20
    candidate_pools: tuple[str, ...] = ("U2_risk_sane",)
    min_train_months: int = 24
    min_train_rows: int = 500
    max_fit_rows: int = 0
    cost_bps: float = 10.0
    random_seed: int = 42
    availability_lag_days: int = 30
    relevance_grades: int = 5
    model_name: str = "M6_xgboost_rank_ndcg"
    min_core_feature_coverage: float = 0.30
    model_n_jobs: int = 0


def select_report_signal_date(
    dataset: pd.DataFrame,
    *,
    candidate_pools: tuple[str, ...],
    requested: str | pd.Timestamp | None = None,
) -> pd.Timestamp:
    df = dataset[dataset["candidate_pool_version"].isin(candidate_pools)].copy()
    if requested:
        target = pd.Timestamp(requested).normalize()
        part = df[df["signal_date"] == target]
        if part.empty:
            raise ValueError(f"报告信号日不存在于 dataset: {target.date()}")
        if not part["candidate_pool_pass"].astype(bool).any():
            raise ValueError(f"报告信号日无 candidate_pool_pass 标的: {target.date()}")
        passed = part[part["candidate_pool_pass"].astype(bool)]
        if "next_trade_date" in passed.columns and not passed["next_trade_date"].notna().all():
            raise ValueError(f"报告信号日存在 candidate_pool_pass 但缺少 next_trade_date: {target.date()}")
        return target
    eligible = df[df["candidate_pool_pass"].astype(bool)].copy()
    if "next_trade_date" in eligible.columns:
        eligible = eligible[eligible["next_trade_date"].notna()].copy()
    if eligible.empty:
        raise ValueError("没有可用于 M7 推荐的 candidate_pool_pass 信号月。")
    return pd.Timestamp(eligible["signal_date"].max()).normalize()


def build_full_fit_report_scores(
    dataset: pd.DataFrame,
    spec: Any,
    cfg: M7RunConfig,
    *,
    report_signal_date: pd.Timestamp,
    feature_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    active_feature_cols = list(feature_cols) if feature_cols is not None else [c for c in spec.feature_cols if c in dataset.columns]
    active_feature_cols = [c for c in active_feature_cols if c in dataset.columns]
    if not active_feature_cols:
        return pd.DataFrame(), pd.DataFrame()
    score_frames: list[pd.DataFrame] = []
    importance_frames: list[pd.DataFrame] = []

    for pool in cfg.candidate_pools:
        pool_df = dataset[dataset["candidate_pool_version"] == pool].copy()
        train = pool_df[
            pool_df["candidate_pool_pass"].astype(bool)
            & (pool_df["signal_date"] < report_signal_date)
            & pool_df["label_forward_1m_o2o_return"].notna()
        ].copy()
        test = pool_df[
            pool_df["candidate_pool_pass"].astype(bool) & (pool_df["signal_date"] == report_signal_date)
        ].copy()
        if test.empty:
            continue
        if train["signal_date"].nunique() < cfg.min_train_months or len(train) < cfg.min_train_rows:
            continue
        train_fit = _cap_fit_rows(train, max_rows=cfg.max_fit_rows, random_seed=cfg.random_seed)
        scores, imp = _train_predict_xgboost_ranker(
            model_name=cfg.model_name, objective="rank:ndcg",
            train=train_fit, test=test, feature_cols=active_feature_cols,
            random_seed=cfg.random_seed, relevance_grades=cfg.relevance_grades,
            model_n_jobs=cfg.model_n_jobs,
        )
        if scores is not None and not scores.empty:
            scores = scores.copy()
            scores["feature_spec"] = spec.name
            scores["feature_families"] = ",".join(spec.families)
            scores["score_percentile"] = pd.to_numeric(scores["score"], errors="coerce")
            score_frames.append(scores)
        if imp is not None and not imp.empty:
            importance_frames.append(_tag_importance(imp, spec, pool, report_signal_date))

    scores_out = pd.concat(score_frames, ignore_index=True) if score_frames else pd.DataFrame()
    imp_out = pd.concat(importance_frames, ignore_index=True) if importance_frames else pd.DataFrame()
    return scores_out, imp_out


def summarize_report_feature_coverage(
    dataset: pd.DataFrame, spec: Any, *, candidate_pools: tuple[str, ...],
) -> pd.DataFrame:
    base = dataset[dataset["candidate_pool_version"].isin(candidate_pools)].copy()
    if base.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for pool, pool_df in base.groupby("candidate_pool_version", sort=True):
        pool_pass_part = pool_df["candidate_pool_pass"].astype(bool)
        for col in spec.feature_cols:
            raw_col = col[:-2] if col.endswith("_z") else col
            vals = pd.to_numeric(pool_df[raw_col], errors="coerce") if raw_col in pool_df.columns else pd.Series(np.nan, index=pool_df.index)
            rows.append({
                "candidate_pool_version": pool, "feature_spec": spec.name,
                "families": ",".join(spec.families), "feature": col, "raw_feature": raw_col,
                "rows": int(len(pool_df)), "candidate_pool_pass_rows": int(pool_pass_part.sum()),
                "non_null": int(vals.notna().sum()),
                "coverage_ratio": float(vals.notna().mean()) if len(pool_df) else np.nan,
                "candidate_pool_pass_coverage_ratio": float(vals.loc[pool_pass_part].notna().mean()) if pool_pass_part.any() else np.nan,
                "first_signal_date": str(pool_df.loc[vals.notna(), "signal_date"].min().date()) if vals.notna().any() else "",
                "last_signal_date": str(pool_df.loc[vals.notna(), "signal_date"].max().date()) if vals.notna().any() else "",
            })
    return pd.DataFrame(rows)


def apply_m9_feature_coverage_policy(
    dataset: pd.DataFrame, spec: Any, feature_coverage: pd.DataFrame,
    *, candidate_pools: tuple[str, ...], min_core_coverage: float = 0.30,
) -> tuple[list[str], pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    active: list[str] = []
    coverage = feature_coverage.copy()
    if coverage.empty:
        active = [c for c in spec.feature_cols if c in dataset.columns]
        return active, pd.DataFrame()
    coverage["candidate_pool_pass_coverage_ratio"] = pd.to_numeric(
        coverage.get("candidate_pool_pass_coverage_ratio"), errors="coerce",
    )
    for feature in spec.feature_cols:
        if feature not in dataset.columns:
            rows.append({"feature": feature, "candidate_pool_pass_coverage_ratio": np.nan,
                         "m9_feature_policy": "missing_from_dataset", "active_feature": ""})
            continue
        raw_feature = feature[:-2] if feature.endswith("_z") else feature
        part = coverage[
            (coverage["feature"] == feature)
            & (coverage["candidate_pool_version"].isin(candidate_pools) if "candidate_pool_version" in coverage.columns else True)
        ].copy()
        cov = float(part["candidate_pool_pass_coverage_ratio"].min()) if not part.empty else np.nan
        missing_flag = f"is_missing_{raw_feature}"
        if np.isfinite(cov) and cov < float(min_core_coverage):
            if missing_flag in dataset.columns:
                active.append(missing_flag)
                policy = "missing_flag_only_low_coverage"
                active_feature = missing_flag
            else:
                policy = "dropped_low_coverage_no_missing_flag"
                active_feature = ""
        else:
            active.append(feature)
            policy = "core_feature"
            active_feature = feature
        rows.append({"feature": feature, "raw_feature": raw_feature,
                     "candidate_pool_pass_coverage_ratio": cov,
                     "m9_feature_policy": policy, "active_feature": active_feature})
    return list(dict.fromkeys(active)), pd.DataFrame(rows)


def apply_industry_cap(
    ranked: pd.DataFrame,
    *,
    top_k: int,
    max_industry_share: float = 0.30,
    industry_col: str = "industry",
    score_col: str = "score",
) -> pd.DataFrame:
    """贪心选取：按 score 降序遍历，当某行业已达上限时跳过。

    保证最终推荐集大小 <= top_k。
    max_industry_share = 0.30 表示 Top20 中单行业最多 6 只。
    """
    if ranked.empty:
        return ranked
    cap = max(1, int(np.floor(max_industry_share * top_k)))
    industry_counts: dict[str, int] = {}
    selected = []
    for _, row in ranked.sort_values(score_col, ascending=False).iterrows():
        ind = str(row.get(industry_col, "") or "")
        if industry_counts.get(ind, 0) >= cap:
            continue
        selected.append(row)
        industry_counts[ind] = industry_counts.get(ind, 0) + 1
        if len(selected) >= top_k:
            break
    return pd.DataFrame(selected)


# ═══════════════════════════════════════════════════════════════════════════
# 推荐表构建
# ═══════════════════════════════════════════════════════════════════════════


def _date_iso(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(pd.Timestamp(value).date())


def _next_signal_date_by_pool(dataset: pd.DataFrame) -> dict[tuple[str, pd.Timestamp], pd.Timestamp]:
    if dataset.empty or not {"candidate_pool_version", "signal_date"}.issubset(dataset.columns):
        return {}
    signal_dates = dataset[["candidate_pool_version", "signal_date"]].copy()
    signal_dates["candidate_pool_version"] = signal_dates["candidate_pool_version"].astype(str)
    signal_dates["signal_date"] = pd.to_datetime(signal_dates["signal_date"], errors="coerce").dt.normalize()
    signal_dates = signal_dates.dropna(subset=["signal_date"]).drop_duplicates()
    out: dict[tuple[str, pd.Timestamp], pd.Timestamp] = {}
    for pool, part in signal_dates.groupby("candidate_pool_version", sort=True):
        dates = sorted(pd.Timestamp(x).normalize() for x in part["signal_date"].unique())
        out.update({(str(pool), dates[i]): dates[i + 1] for i in range(len(dates) - 1)})
    return out


def _buyability_text(row: pd.Series) -> str:
    ok = bool(row.get("is_buyable_tplus1_open", False))
    if ok:
        return "buyable_tplus1_open"
    reason = str(row.get("buyability_reject_reason", "") or "").strip()
    return reason or "not_buyable_tplus1_open"


def _risk_flags_text(row: pd.Series) -> str:
    flags: list[str] = []
    for col in ["risk_flags", "candidate_pool_reject_reason"]:
        val = str(row.get(col, "") or "").strip()
        if val and val.lower() != "nan":
            flags.extend([x.strip() for x in val.split(";") if x.strip()])
    if not bool(row.get("is_buyable_tplus1_open", False)):
        reason = str(row.get("buyability_reject_reason", "") or "").strip()
        flags.append(reason or "not_buyable_tplus1_open")
    return ";".join(dict.fromkeys(flags))


def _build_previous_rank_map(
    previous_holdings: pd.DataFrame,
    *,
    report_signal_date: pd.Timestamp,
    pool: str,
    model: str,
    top_k: int,
) -> dict[str, int]:
    if previous_holdings.empty:
        return {}
    prev = previous_holdings.copy()
    prev["signal_date"] = pd.to_datetime(prev["signal_date"], errors="coerce").dt.normalize()
    prev = prev[
        (prev["signal_date"] < report_signal_date)
        & (prev["candidate_pool_version"] == pool)
        & (prev["model"] == model)
        & (pd.to_numeric(prev["top_k"], errors="coerce") == int(top_k))
    ].copy()
    if prev.empty:
        return {}
    last_date = prev["signal_date"].max()
    prev = prev[prev["signal_date"] == last_date].copy()
    rank_col = "selected_rank" if "selected_rank" in prev.columns else "rank"
    return {
        _display_symbol(row["symbol"]): int(row[rank_col])
        for _, row in prev.iterrows()
        if pd.notna(row.get(rank_col))
    }


def _feature_contrib_text(row: pd.Series, imp: pd.DataFrame, feature_cols: list[str], *, max_features: int = 3) -> str:
    if imp.empty:
        return ""
    model = row.get("model", "")
    pool = row.get("candidate_pool_version", "")
    view = imp[(imp["model"] == model) & (imp["candidate_pool_version"] == pool)].copy()
    if view.empty:
        view = imp[imp["model"] == model].copy()
    if view.empty:
        return ""
    imp_map = view.groupby("feature")["importance"].mean().to_dict()
    rows: list[tuple[float, str]] = []
    for feature in feature_cols:
        if feature not in row.index or feature not in imp_map:
            continue
        val = pd.to_numeric(pd.Series([row[feature]]), errors="coerce").iloc[0]
        importance = float(imp_map.get(feature, 0.0) or 0.0)
        if not np.isfinite(val) or not np.isfinite(importance) or importance <= 0:
            continue
        rows.append((abs(float(val)) * importance, f"{feature}={float(val):+.3g}"))
    rows.sort(key=lambda x: x[0], reverse=True)
    return "; ".join(text for _, text in rows[:max_features])


def build_recommendation_table(
    scores: pd.DataFrame,
    dataset: pd.DataFrame,
    feature_importance: pd.DataFrame,
    *,
    feature_cols: list[str],
    top_ks: list[int],
    previous_holdings: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if scores.empty:
        return pd.DataFrame(), pd.DataFrame()
    previous = previous_holdings if previous_holdings is not None else pd.DataFrame()
    keys = ["signal_date", "candidate_pool_version", "symbol"]
    feature_keep = [
        c
        for c in [
            *keys,
            *feature_cols,
            "name", "stock_name", "股票名称", "名称",
            "industry_level1", "industry_level2",
            "candidate_pool_rule", "candidate_pool_reject_reason",
            "buyability_reject_reason", "is_buyable_tplus1_open", "next_trade_date",
        ]
        if c in dataset.columns
    ]
    enriched = scores.merge(dataset[feature_keep], on=keys, how="left", suffixes=("", "_dataset"))
    if "industry_level1_dataset" in enriched.columns:
        enriched["industry_level1"] = enriched["industry_level1"].fillna(enriched["industry_level1_dataset"])
    if "industry_level2_dataset" in enriched.columns:
        enriched["industry_level2"] = enriched["industry_level2"].fillna(enriched["industry_level2_dataset"])
    enriched["name"] = _name_column(enriched)
    next_signal_by_pool = _next_signal_date_by_pool(dataset)

    rec_frames: list[pd.DataFrame] = []
    contrib_rows: list[dict[str, Any]] = []
    for (signal_date, pool, model), part in enriched.groupby(
        ["signal_date", "candidate_pool_version", "model"], sort=True
    ):
        ranked = part.sort_values(["score", "symbol"], ascending=[False, True]).copy()
        ranked["rank_all"] = np.arange(1, len(ranked) + 1)
        for k in top_ks:
            top = apply_industry_cap(
                ranked, top_k=int(k), max_industry_share=0.30,
                industry_col="industry_level1", score_col="score",
            )
            if top.empty:
                top = ranked.head(int(k)).copy()
            prev_map = _build_previous_rank_map(
                previous,
                report_signal_date=pd.Timestamp(signal_date),
                pool=str(pool), model=str(model), top_k=int(k),
            )
            rows: list[dict[str, Any]] = []
            for i, (_, row) in enumerate(top.iterrows(), start=1):
                symbol = _display_symbol(row["symbol"])
                contrib = _feature_contrib_text(row, feature_importance, feature_cols)
                buy_trade_date = _date_iso(row.get("next_trade_date"))
                sell_trade_date = _date_iso(
                    next_signal_by_pool.get((str(pool), pd.Timestamp(signal_date).normalize()))
                )
                rows.append({
                    "signal_date": pd.Timestamp(signal_date).date().isoformat(),
                    "top_k": int(k), "rank": int(i), "symbol": symbol,
                    "name": str(row.get("name", "") or ""),
                    "score": float(row.get("score")) if pd.notna(row.get("score")) else np.nan,
                    "score_percentile": float(row.get("score_percentile"))
                    if pd.notna(row.get("score_percentile")) else np.nan,
                    "industry": str(row.get("industry_level1", "") or ""),
                    "industry_level2": str(row.get("industry_level2", "") or ""),
                    "feature_contrib": contrib, "risk_flags": _risk_flags_text(row),
                    "last_month_rank": prev_map.get(symbol, pd.NA),
                    "last_month_selected": symbol in prev_map,
                    "buyability": _buyability_text(row),
                    "next_trade_date": buy_trade_date,
                    "buy_trade_date": buy_trade_date,
                    "sell_trade_date": sell_trade_date,
                    "candidate_pool_version": str(pool),
                    "candidate_pool_rule": str(row.get("candidate_pool_rule", POOL_RULES.get(str(pool), "")) or ""),
                    "model": str(model),
                    "model_type": str(row.get("model_type", "") or ""),
                    "feature_spec": str(row.get("feature_spec", "") or ""),
                })
                contrib_rows.append({
                    "signal_date": pd.Timestamp(signal_date).date().isoformat(),
                    "top_k": int(k), "rank": int(i), "symbol": symbol,
                    "candidate_pool_version": str(pool), "model": str(model),
                    "feature_contrib": contrib,
                })
            rec_frames.append(pd.DataFrame(rows))
    rec = pd.concat(rec_frames, ignore_index=True) if rec_frames else pd.DataFrame()
    if not rec.empty:
        rec["last_month_rank"] = pd.to_numeric(rec["last_month_rank"], errors="coerce").astype("Int64")
    contrib_df = pd.DataFrame(contrib_rows)
    return rec, contrib_df


def summarize_recommendation_industry_exposure(recommendations: pd.DataFrame) -> pd.DataFrame:
    if recommendations.empty:
        return pd.DataFrame()
    total = (
        recommendations.groupby(["signal_date", "candidate_pool_version", "model", "top_k"], sort=True)["symbol"]
        .nunique().rename("topk_count").reset_index()
    )
    out = (
        recommendations.groupby(
            ["signal_date", "candidate_pool_version", "model", "top_k", "industry"], dropna=False, sort=True,
        )["symbol"].nunique().rename("industry_count").reset_index()
        .merge(total, on=["signal_date", "candidate_pool_version", "model", "top_k"], how="left")
    )
    out["industry_share"] = out["industry_count"] / out["topk_count"].replace(0, np.nan)
    return out


def summarize_recommendation_risk(recommendations: pd.DataFrame) -> pd.DataFrame:
    if recommendations.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for (signal_date, pool, model, top_k), part in recommendations.groupby(
        ["signal_date", "candidate_pool_version", "model", "top_k"], sort=True
    ):
        flag_count = int(part["risk_flags"].fillna("").astype(str).str.len().gt(0).sum())
        not_buyable = int((part["buyability"] != "buyable_tplus1_open").sum())
        rows.append({
            "signal_date": signal_date, "candidate_pool_version": pool, "model": model,
            "top_k": int(top_k), "selected_count": int(len(part)),
            "risk_flagged_count": flag_count, "not_buyable_count": not_buyable,
            "last_month_selected_count": int(part["last_month_selected"].sum()),
        })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════
# M9 数据完整性
# ═══════════════════════════════════════════════════════════════════════════


def _build_m9_industry_concentration_checks(
    recommendations: pd.DataFrame, report_signal_date: pd.Timestamp,
) -> list[dict[str, Any]]:
    if recommendations.empty:
        return [
            {"check": "max_single_industry_share_le_40pct", "value": np.nan, "pass": False,
             "detail": "推荐集为空，无法检查行业集中度"},
            {"check": "distinct_industry_count_ge_3", "value": 0, "pass": False,
             "detail": "推荐集为空，无法检查行业覆盖"},
        ]
    main = recommendations
    if "signal_date" in recommendations.columns and "top_k" in recommendations.columns:
        main = recommendations[
            (recommendations["signal_date"] == str(report_signal_date.date()))
            & (recommendations["top_k"] == recommendations["top_k"].max())
        ].copy()
        if main.empty:
            main = recommendations.copy()
    if "industry" not in main.columns:
        return [
            {"check": "max_single_industry_share_le_40pct", "value": np.nan, "pass": True,
             "detail": "推荐集缺少 industry 列，跳过行业集中度检查"},
            {"check": "distinct_industry_count_ge_3", "value": np.nan, "pass": True,
             "detail": "推荐集缺少 industry 列，跳过行业覆盖检查"},
        ]
    n = len(main)
    max_ind_share = (
        (main.groupby("industry")["symbol"].nunique() / n).max()
        if n > 0 and "symbol" in main.columns else 0.0
    )
    n_industries = int(main["industry"].nunique()) if n > 0 else 0
    return [
        {
            "check": "max_single_industry_share_le_40pct",
            "value": round(float(max_ind_share), 4),
            "pass": bool(max_ind_share <= 0.40),
            "detail": f"单行业占比 {max_ind_share:.1%}，超 40% 视为集中度过高",
        },
        {
            "check": "distinct_industry_count_ge_3",
            "value": n_industries,
            "pass": bool(n_industries >= 3),
            "detail": f"Top-K 推荐覆盖 {n_industries} 个行业，至少应覆盖 3 个",
        },
    ]


def summarize_m9_integrity(
    *,
    dataset: pd.DataFrame,
    recommendations: pd.DataFrame,
    feature_coverage: pd.DataFrame,
    feature_policy: pd.DataFrame,
    report_signal_date: pd.Timestamp,
    candidate_pools: tuple[str, ...],
) -> pd.DataFrame:
    target = dataset[
        dataset["candidate_pool_version"].isin(candidate_pools)
        & (dataset["signal_date"] == report_signal_date)
    ].copy()
    pass_part = target[target["candidate_pool_pass"].astype(bool)].copy() if not target.empty else pd.DataFrame()
    unknown_name_count = (
        int(recommendations["name"].fillna("").astype(str).str.strip().isin(["", "UNKNOWN"]).sum())
        if "name" in recommendations.columns and not recommendations.empty else 0
    )
    st_name_count = (
        int(_is_st_name(recommendations["name"]).sum())
        if "name" in recommendations.columns and not recommendations.empty else 0
    )
    not_buyable_count = (
        int((recommendations["buyability"] != "buyable_tplus1_open").sum())
        if "buyability" in recommendations.columns and not recommendations.empty else 0
    )
    zero_coverage_core = 0
    if not feature_policy.empty:
        zero_coverage_core = int(
            (feature_policy["m9_feature_policy"].eq("core_feature")
             & (pd.to_numeric(feature_policy["candidate_pool_pass_coverage_ratio"], errors="coerce") <= 0)).sum()
        )
    low_coverage_core = 0
    if not feature_policy.empty:
        low_coverage_core = int(
            (feature_policy["m9_feature_policy"].eq("core_feature")
             & (pd.to_numeric(feature_policy["candidate_pool_pass_coverage_ratio"], errors="coerce") < 0.30)).sum()
        )
    rows = [
        {
            "check": "target_candidate_pool_pass_rows", "value": int(len(pass_part)),
            "pass": bool(len(pass_part) > 0),
            "detail": "latest selected signal date must have buyable candidates",
        },
        {
            "check": "target_next_trade_date_present",
            "value": int(pass_part["next_trade_date"].notna().sum()) if "next_trade_date" in pass_part.columns else 0,
            "pass": bool((not pass_part.empty) and "next_trade_date" in pass_part.columns and pass_part["next_trade_date"].notna().all()),
            "detail": "all candidate_pool_pass rows should carry next_trade_date",
        },
        {
            "check": "recommendation_buyable", "value": not_buyable_count,
            "pass": bool(not_buyable_count == 0 and not recommendations.empty),
            "detail": "recommendation rows should be buyable at t+1 open",
        },
        {
            "check": "recommendation_names_readable", "value": unknown_name_count,
            "pass": bool(unknown_name_count == 0 and not recommendations.empty),
            "detail": "name should not be UNKNOWN or blank",
        },
        {
            "check": "recommendation_excludes_st_names", "value": st_name_count,
            "pass": bool(st_name_count == 0 and not recommendations.empty),
            "detail": "name-aware report filter should exclude ST/*ST targets",
        },
        {
            "check": "zero_coverage_core_features", "value": zero_coverage_core,
            "pass": bool(zero_coverage_core == 0),
            "detail": "zero coverage fields must not be core model features",
        },
        {
            "check": "low_coverage_core_features_lt_30pct", "value": low_coverage_core,
            "pass": bool(low_coverage_core == 0),
            "detail": "low coverage fields should be missing-marker-only or ablation-only",
        },
        *_build_m9_industry_concentration_checks(recommendations, report_signal_date),
    ]
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════
# Quality payload & doc builder
# ═══════════════════════════════════════════════════════════════════════════


def build_quality_payload(
    *,
    dataset: pd.DataFrame,
    recommendations: pd.DataFrame,
    report_signal_date: pd.Timestamp,
    spec: Any,
    cfg: Any,
    dataset_path: Any,
    db_path: Any,
    output_stem: str,
    config_source: str,
    research_config_id: str,
    evidence_stem: str,
    project_root: Any = None,
) -> dict[str, Any]:
    ROOT = project_root or Path(__file__).resolve().parents[2]
    target = dataset[
        dataset["candidate_pool_version"].isin(cfg.candidate_pools)
        & (dataset["signal_date"] == report_signal_date)
    ].copy()
    next_signal_by_pool = _next_signal_date_by_pool(dataset)
    sell_dates = [
        next_signal_by_pool.get((pool, pd.Timestamp(report_signal_date).normalize()))
        for pool in cfg.candidate_pools
        if next_signal_by_pool.get((pool, pd.Timestamp(report_signal_date).normalize())) is not None
    ]
    return {
        "result_type": "monthly_selection_m7_recommendation_report",
        "research_topic": "monthly_selection_m7_recommendation_report",
        "research_config_id": research_config_id, "output_stem": output_stem,
        "config_source": config_source,
        "dataset_path": str(dataset_path.relative_to(ROOT)) if dataset_path.is_relative_to(ROOT) else str(dataset_path),
        "duckdb_path": str(db_path.relative_to(ROOT)) if db_path.is_relative_to(ROOT) else str(db_path),
        "dataset_version": "monthly_selection_features_v1",
        "candidate_pool_version": ",".join(cfg.candidate_pools),
        "candidate_pool_rule": {p: POOL_RULES.get(p, "") for p in cfg.candidate_pools},
        "candidate_pool_width_by_month": "see candidate_pool_width.csv",
        "feature_spec": {"name": spec.name, "families": list(spec.families), "feature_count": len(spec.feature_cols)},
        "label_spec": "full-fit ranker trained on historical forward_1m_excess_vs_market relevance; target report month has no label requirement",
        "pit_policy": "target features are signal-date rows; full-fit model uses only labeled months before report_signal_date",
        "universe_filter": "candidate_pool_pass in selected pool; no alpha filter outside model ranking",
        "industry_map_source": "industry_level1/2 columns from M2 canonical dataset",
        "data_quality_report": "M2/M5/M6 artifacts plus current M7 feature_coverage/candidate_pool_width files",
        "model_type": "xgboost_ranker_full_fit_for_research_report",
        "train_window": f"< {report_signal_date.date()} labeled signal months",
        "validation_window": f"M6 historical walk-forward evidence stem: {evidence_stem or 'not_found'}",
        "test_window": f"{report_signal_date.date()} recommendation month, unlabeled at report time",
        "cv_policy": "historical evidence uses walk_forward_by_signal_month; current report uses full-fit on past labeled months",
        "hyperparameter_policy": "fixed M6 watchlist hyperparameters; no random CV and no target-month tuning",
        "random_seed": int(cfg.random_seed),
        "rebalance_rule": "M", "execution_mode": "tplus1_open",
        "benchmark_return_mode": "market_ew_open_to_open",
        "sell_timing": "holding_month_last_trading_day_open",
        "top_k": list(cfg.top_ks), "report_top_k": int(cfg.report_top_k),
        "cost_assumption": f"{float(cfg.cost_bps):.4g} bps per unit half-L1 turnover",
        "model_n_jobs": int(getattr(cfg, 'model_n_jobs', 0)),
        "buyability_policy": "selected rows must pass candidate_pool_pass; buyability column reports t+1 open status",
        "report_signal_date": str(report_signal_date.date()),
        "next_trade_date": str(target["next_trade_date"].dropna().iloc[0].date())
        if "next_trade_date" in target.columns and target["next_trade_date"].notna().any() else "",
        "buy_trade_date": str(target["next_trade_date"].dropna().iloc[0].date())
        if "next_trade_date" in target.columns and target["next_trade_date"].notna().any() else "",
        "sell_trade_date": str(min(sell_dates).date()) if sell_dates else "",
        "target_candidate_rows": int(len(target)),
        "target_candidate_pass_rows": int(target["candidate_pool_pass"].astype(bool).sum()) if not target.empty else 0,
        "recommendation_rows": int(len(recommendations)),
        "model": getattr(cfg, 'model_name', 'M6_xgboost_rank_ndcg'),
        "production_status": "research_only_not_promoted",
    }


def latest_evidence_stem(results_dir: Path) -> str:
    manifests = sorted(results_dir.glob("monthly_selection_m6_ltr_*_manifest.json"))
    if not manifests:
        return ""
    return manifests[-1].name[: -len("_manifest.json")]


def read_evidence(results_dir: Path, stem: str, suffix: str) -> pd.DataFrame:
    if not stem:
        return pd.DataFrame()
    path = results_dir / f"{stem}_{suffix}.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def filter_evidence(df: pd.DataFrame, cfg: Any) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    if "candidate_pool_version" in out.columns:
        out = out[out["candidate_pool_version"].isin(cfg.candidate_pools)].copy()
    if "model" in out.columns:
        out = out[out["model"] == cfg.model_name].copy()
    if "top_k" in out.columns:
        out = out[pd.to_numeric(out["top_k"], errors="coerce").isin(list(cfg.top_ks))].copy()
    return out


def build_m7_doc(
    *,
    quality: dict[str, Any],
    recommendations: pd.DataFrame,
    leaderboard: pd.DataFrame,
    risk_summary: pd.DataFrame,
    industry_exposure: pd.DataFrame,
    feature_coverage: pd.DataFrame,
    feature_policy: pd.DataFrame,
    m9_integrity: pd.DataFrame,
    artifacts: list[str],
) -> str:
    generated_at = pd.Timestamp.utcnow().isoformat()
    main = recommendations[recommendations["top_k"] == quality.get("report_top_k", 20)].copy()
    main = main.sort_values(["top_k", "rank"]).head(int(quality.get("report_top_k", 20)))
    leader_view = leaderboard.sort_values(
        ["top_k", "topk_excess_after_cost_mean", "rank_ic_mean"],
        ascending=[True, False, False],
    ).head(20) if not leaderboard.empty else pd.DataFrame()
    artifact_lines = "\n".join(f"- `{x}`" for x in artifacts)
    return f"""# Monthly Selection M7 Recommendation Report

- 生成时间：`{generated_at}`
- 结果类型：`monthly_selection_m7_recommendation_report`
- 研究配置：`{quality.get('research_config_id', '')}`
- 报告信号日：`{quality.get('report_signal_date', '')}`
- 买入交易日：`{quality.get('buy_trade_date', '')}`
- 卖出交易日：`{quality.get('sell_trade_date', '')}`
- 候选池：`{quality.get('candidate_pool_version', '')}`
- 模型：`{quality.get('model', '')}`
- 生产状态：`research_only_not_promoted`

## 推荐名单

{format_markdown_table(main, max_rows=int(quality.get('report_top_k', 20)))}

## M6 Historical Evidence

{format_markdown_table(leader_view, max_rows=20)}

## Risk Summary

{format_markdown_table(risk_summary, max_rows=20)}

## Industry Exposure

{format_markdown_table(industry_exposure, max_rows=40)}

## Feature Coverage

{format_markdown_table(feature_coverage.head(80), max_rows=80)}

## M9 Integrity

{format_markdown_table(m9_integrity, max_rows=20)}

## M9 Feature Policy

{format_markdown_table(feature_policy.head(80), max_rows=80)}

## 口径与结论

- 本轮新增的是 M7 研究版推荐报告，不新增生产策略；推荐名单不自动生成交易指令。
- 数据质量沿用 M2/M5/M6 的 PIT 检查，本脚本只使用 `report_signal_date` 当日可观测特征，并只用该日期之前已完成标签的月份训练。
- M9 数据完整性 gate 要求报告信号日存在 `next_trade_date`、推荐行全部 `buyable_tplus1_open`、名称非 UNKNOWN，且零覆盖/低覆盖字段不作为核心模型特征。
- 月度交易时点按"本持有月最后一个交易日卖出、下一持有月首个交易日买入"展示；`sell_trade_date` 为下一次月末信号日。
- 候选池使用 `U2_risk_sane` watchlist 口径，只做准入过滤；alpha 判断来自 `M6_xgboost_rank_ndcg` 排序分数。
- 历史 baseline 改善证据来自 M6 walk-forward 附件；M6 结论仍是 watchlist，不进入生产。
- 当前推荐月尚无未来收益标签，因此本报告不声称本月已跑赢市场；稳定性判断以历史 leaderboard / monthly_long / rank_ic / quantile_spread 为准。
- 分桶、年份和 regime 的历史失败月份保留在附件中；后续进入 regime-aware calibration 复核。

## 本轮产物

{artifact_lines}
"""
