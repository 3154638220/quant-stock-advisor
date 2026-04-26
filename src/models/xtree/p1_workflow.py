"""P1 树模型研究编排：特征分组、面板增强与轻量代理汇总。"""

from __future__ import annotations

import json
import re
from collections import OrderedDict
from typing import Any, Iterable

import numpy as np
import pandas as pd

from scripts.light_strategy_proxy import infer_periods_per_year, summarize_signal_diagnostic
from src.features.tree_dataset import default_tree_factor_names

WEEKLY_KDJ_TREE_FEATURES: tuple[str, ...] = (
    "weekly_kdj_k",
    "weekly_kdj_d",
    "weekly_kdj_j",
    "weekly_kdj_oversold",
    "weekly_kdj_oversold_depth",
    "weekly_kdj_rebound",
)

FUND_FLOW_TREE_FEATURES: tuple[str, ...] = (
    "main_inflow_z_5d",
    "main_inflow_z_10d",
    "main_inflow_z_20d",
    "super_inflow_z_5d",
    "super_inflow_z_10d",
    "super_inflow_z_20d",
    "flow_divergence_5d",
    "flow_divergence_10d",
    "flow_divergence_20d",
    "main_inflow_streak",
)

SHAREHOLDER_TREE_FEATURES: tuple[str, ...] = (
    "holder_count_change_pct",
    "holder_change_rate_z",
    "holder_count_log_z",
    "holder_concentration_proxy",
)

WEEKLY_KDJ_INTERACTION_TREE_FEATURES: tuple[str, ...] = (
    "wk_j_contrarian_x_vol_to_turnover",
    "wk_j_contrarian_low_turnover_gate",
    "wk_j_contrarian_weak_momentum_gate",
    "wk_oversold_depth_low_vol_gate",
)

DEFAULT_P1_BENCHMARK_KEY_YEARS: tuple[int, ...] = (2021, 2025, 2026)


def baseline_tree_feature_names() -> tuple[str, ...]:
    """P1 的 G0 基线技术面：默认树模型因子剔除 weekly_kdj 家族。"""
    banned = set(WEEKLY_KDJ_TREE_FEATURES)
    return tuple(name for name in default_tree_factor_names() if name not in banned)


def p1_tree_feature_groups(
    *,
    include_interaction_groups: bool = False,
) -> "OrderedDict[str, tuple[str, ...]]":
    """P1 研究固定分组，保证 G0~G4 口径稳定可复现。"""
    g0 = baseline_tree_feature_names()
    groups: "OrderedDict[str, tuple[str, ...]]" = OrderedDict(
        [
            ("G0", g0),
            ("G1", g0 + WEEKLY_KDJ_TREE_FEATURES),
            ("G2", g0 + FUND_FLOW_TREE_FEATURES),
            ("G3", g0 + SHAREHOLDER_TREE_FEATURES),
            ("G4", g0 + WEEKLY_KDJ_TREE_FEATURES + FUND_FLOW_TREE_FEATURES),
        ]
    )
    if include_interaction_groups:
        groups["G5"] = g0 + WEEKLY_KDJ_TREE_FEATURES + WEEKLY_KDJ_INTERACTION_TREE_FEATURES
        groups["G6"] = g0 + WEEKLY_KDJ_INTERACTION_TREE_FEATURES
    return groups


def union_p1_tree_feature_names() -> tuple[str, ...]:
    """P1 runner 所需的全量特征并集。"""
    out: list[str] = []
    seen: set[str] = set()
    for names in p1_tree_feature_groups().values():
        for name in names:
            if name not in seen:
                seen.add(name)
                out.append(name)
    return tuple(out)


def panel_generation_feature_names() -> tuple[str, ...]:
    """
    供 ``long_factor_panel_from_daily`` 直接生成的原生树因子。

    资金流 / 股东人数列需要在面板生成后再 attach，因此这里仅保留 baseline + weekly_kdj。
    """
    g0 = baseline_tree_feature_names()
    out: list[str] = []
    seen: set[str] = set()
    for name in g0 + WEEKLY_KDJ_TREE_FEATURES:
        if name not in seen:
            seen.add(name)
            out.append(name)
    return tuple(out)


def _slugify_token(value: Any) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "na"


def build_p1_tree_research_config_id(
    *,
    rebalance_rule: str,
    top_k: int,
    label_horizons: Iterable[int],
    proxy_horizon: int,
    val_frac: float,
    label_mode: str = "",
    xgboost_objective: str = "",
) -> str:
    """把核心研究参数压成稳定 config id，便于结果/工件追踪。"""
    horizons = "-".join(str(int(x)) for x in label_horizons)
    rebalance = _slugify_token(rebalance_rule or "d")
    val_pct = int(round(float(val_frac) * 100))
    parts = [
        f"rb_{rebalance}",
        f"top{int(top_k)}",
        f"lh_{horizons}",
        f"px_{int(proxy_horizon)}",
        f"val{val_pct}",
    ]
    if label_mode:
        parts.append(f"lbl_{_slugify_token(label_mode)}")
    if xgboost_objective:
        parts.append(f"obj_{_slugify_token(xgboost_objective)}")
    return "_".join(parts)


def build_p1_training_label(
    panel: pd.DataFrame,
    *,
    label_columns: Iterable[str],
    label_weights: Iterable[float],
    label_mode: str = "rank_fusion",
    date_col: str = "trade_date",
    out_col: str = "forward_ret_fused",
) -> tuple[pd.DataFrame, str, dict[str, Any]]:
    """
    构造 P1 树模型训练标签。

    ``rank_fusion`` 复用既有截面 rank 融合；``raw_fusion`` 直接融合原始收益；
    ``market_relative`` / ``benchmark_relative`` 先按日扣掉截面等权收益，再融合。
    当前 full backtest 以 ``market_ew_proxy`` 为 benchmark，因此 benchmark-relative
    在训练面板内采用同日截面等权前瞻收益作为可复现代理。
    """
    label_cols = [str(c) for c in label_columns]
    weights = [float(w) for w in label_weights]
    if len(label_cols) != len(weights):
        raise ValueError("label_columns 与 label_weights 长度不一致")
    if not label_cols:
        raise ValueError("label_columns 不能为空")
    missing = [c for c in label_cols if c not in panel.columns]
    if missing:
        raise ValueError(f"缺少标签列: {missing}")
    if date_col not in panel.columns:
        raise ValueError(f"缺少日期列: {date_col}")

    w = np.asarray(weights, dtype=np.float64)
    if not np.isfinite(w).all() or np.sum(np.abs(w)) <= 1e-12:
        raise ValueError("label_weights 非法")
    w = w / np.sum(np.abs(w))

    mode = _slugify_token(label_mode or "rank_fusion")
    if mode not in {"rank_fusion", "raw_fusion", "market_relative", "benchmark_relative"}:
        raise ValueError("label_mode 须为 rank_fusion/raw_fusion/market_relative/benchmark_relative")

    out = panel.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce").dt.normalize()
    score = np.zeros(len(out), dtype=np.float64)
    valid = out[date_col].notna().to_numpy(dtype=bool)
    component_cols: list[str] = []
    for col, wi in zip(label_cols, w):
        vals = pd.to_numeric(out[col], errors="coerce")
        if mode == "rank_fusion":
            comp = vals.groupby(out[date_col], sort=False).rank(method="average", pct=True) - 0.5
        elif mode in {"market_relative", "benchmark_relative"}:
            comp = vals - vals.groupby(out[date_col], sort=False).transform("mean")
        else:
            comp = vals
        comp_np = comp.to_numpy(dtype=np.float64)
        score += wi * comp_np
        valid &= np.isfinite(comp_np)
        component_cols.append(col)

    out[out_col] = np.where(valid, score, np.nan)
    meta = {
        "label_mode": mode,
        "label_scope": "cross_section_relative" if mode == "rank_fusion" else mode,
        "label_component_columns": ",".join(component_cols),
        "label_weights_normalized": ",".join(f"{x:.8g}" for x in w),
        "label_market_proxy": "same_date_cross_section_equal_weight" if mode in {"market_relative", "benchmark_relative"} else "",
    }
    return out[np.isfinite(out[out_col])].copy(), out_col, meta


def build_p1_tree_output_stem(
    *,
    out_tag: str,
    research_config_id: str,
    generated_at: Any,
) -> str:
    """生成 P1 树模型研究输出 stem，统一命名口径。"""
    ts = pd.Timestamp(generated_at).strftime("%Y%m%d_%H%M%S")
    return f"{_slugify_token(out_tag)}_{research_config_id}_{ts}"


def attach_p1_experimental_features(
    panel: pd.DataFrame,
    *,
    db_path: str,
    fund_flow_table: str = "a_share_fund_flow",
    shareholder_table: str = "a_share_shareholder",
    shareholder_lag_days: int = 30,
) -> pd.DataFrame:
    """在树模型面板上接入 P1 新数据链路。"""
    from src.features.fund_flow_factors import attach_fund_flow
    from src.features.shareholder_factors import attach_shareholder_factors

    out = attach_fund_flow(panel, db_path, table_name=fund_flow_table)
    out = attach_shareholder_factors(
        out,
        db_path,
        table_name=shareholder_table,
        availability_lag_days=shareholder_lag_days,
    )
    return out


def _cross_section_z(df: pd.DataFrame, col: str, *, date_col: str = "trade_date") -> pd.Series:
    vals = pd.to_numeric(df[col], errors="coerce")
    mean = vals.groupby(df[date_col], sort=False).transform("mean")
    std = vals.groupby(df[date_col], sort=False).transform("std")
    return (vals - mean) / std.replace(0.0, np.nan)


def attach_weekly_kdj_interaction_features(
    panel: pd.DataFrame,
    *,
    low_turnover_quantile: float = 0.4,
    weak_momentum_quantile: float = 0.6,
    low_vol_quantile: float = 0.7,
) -> pd.DataFrame:
    """
    构造 P1 的 weekly_kdj gated / interaction 特征。

    原始证据显示 ``weekly_kdj_j`` 低位更偏正向，因此这里统一使用
    ``-weekly_kdj_j`` 作为 contrarian 强度，再用低换手、非强动量、非高波动
    三类条件做门控。
    """
    required = {
        "trade_date",
        "weekly_kdj_j",
        "weekly_kdj_oversold_depth",
        "vol_to_turnover",
        "turnover_roll_mean",
        "momentum",
        "realized_vol",
    }
    missing = sorted(required - set(panel.columns))
    if missing:
        raise ValueError(f"缺少 weekly_kdj interaction 依赖列: {missing}")

    out = panel.copy()
    out["trade_date"] = pd.to_datetime(out["trade_date"], errors="coerce").dt.normalize()
    contrarian = -pd.to_numeric(out["weekly_kdj_j"], errors="coerce")
    out["wk_j_contrarian_x_vol_to_turnover"] = contrarian * _cross_section_z(out, "vol_to_turnover")

    turnover = pd.to_numeric(out["turnover_roll_mean"], errors="coerce")
    momentum = pd.to_numeric(out["momentum"], errors="coerce")
    realized_vol = pd.to_numeric(out["realized_vol"], errors="coerce")
    low_turnover_cut = turnover.groupby(out["trade_date"], sort=False).transform(
        lambda s: s.quantile(float(low_turnover_quantile))
    )
    weak_momentum_cut = momentum.groupby(out["trade_date"], sort=False).transform(
        lambda s: s.quantile(float(weak_momentum_quantile))
    )
    low_vol_cut = realized_vol.groupby(out["trade_date"], sort=False).transform(
        lambda s: s.quantile(float(low_vol_quantile))
    )

    out["wk_j_contrarian_low_turnover_gate"] = contrarian.where(turnover <= low_turnover_cut, 0.0)
    out["wk_j_contrarian_weak_momentum_gate"] = contrarian.where(momentum <= weak_momentum_cut, 0.0)
    out["wk_oversold_depth_low_vol_gate"] = pd.to_numeric(
        out["weekly_kdj_oversold_depth"], errors="coerce"
    ).where(realized_vol <= low_vol_cut, 0.0)
    return out


def resolve_available_feature_names(
    panel: pd.DataFrame,
    feature_names: Iterable[str],
) -> tuple[list[str], list[str]]:
    """过滤掉面板中不存在或全为空的列。"""
    available: list[str] = []
    missing: list[str] = []
    for name in feature_names:
        if name not in panel.columns:
            missing.append(name)
            continue
        ser = pd.to_numeric(panel[name], errors="coerce")
        if ser.notna().any():
            available.append(name)
        else:
            missing.append(name)
    return available, missing


def select_rebalance_dates(
    dates: Iterable[Any],
    *,
    rebalance_rule: str,
) -> pd.DataFrame:
    """按调仓频率选出每个 period 的最终调仓日。"""
    idx = pd.DatetimeIndex(pd.to_datetime(list(dates), errors="coerce")).dropna().sort_values().unique()
    if len(idx) == 0:
        return pd.DataFrame(columns=["trade_date", "period"])

    rule = str(rebalance_rule).strip().upper() or "D"
    multiple = ""
    unit = rule
    while unit and unit[0].isdigit():
        multiple += unit[0]
        unit = unit[1:]
    step = max(int(multiple), 1) if multiple else 1

    if unit in {"M", "ME"}:
        base_period = pd.Series(idx.to_period("M").astype(str), index=idx)
    elif unit.startswith("W"):
        weekly_freq = unit if "-" in unit else "W-FRI"
        base_period = pd.Series(idx.to_period(weekly_freq).astype(str), index=idx)
    elif unit in {"Q", "QE"}:
        base_period = pd.Series(idx.to_period("Q").astype(str), index=idx)
    else:
        base_period = pd.Series(idx.strftime("%Y-%m-%d"), index=idx)

    ordered_periods = pd.Index(pd.unique(base_period))
    period_to_gid = {period: i // step for i, period in enumerate(ordered_periods)}
    grouped = (
        pd.DataFrame(
            {
                "trade_date": idx,
                "base_period": base_period.to_numpy(),
                "group_id": base_period.map(period_to_gid).to_numpy(),
            }
        )
        .groupby("group_id", sort=True)
        .agg(trade_date=("trade_date", "last"), period=("base_period", "last"))
        .reset_index(drop=True)
    )
    grouped["trade_date"] = pd.to_datetime(grouped["trade_date"]).dt.normalize()
    return grouped


def build_tree_light_proxy_detail(
    scored_panel: pd.DataFrame,
    *,
    score_col: str,
    proxy_return_col: str,
    rebalance_rule: str,
    top_k: int,
    scenario: str,
) -> pd.DataFrame:
    """
    用验证集预测分数构造 benchmark-first 轻量代理。

    口径：
    - 每个 rebalance period 只取最后一个交易日做一次截面选股
    - `strategy_return` = Top-K 在该日的平均前瞻收益
    - `benchmark_return` = 同日全截面的平均前瞻收益
    """
    if top_k < 1:
        raise ValueError("top_k 须 >= 1")
    if score_col not in scored_panel.columns:
        raise ValueError(f"缺少 score_col={score_col}")
    if proxy_return_col not in scored_panel.columns:
        raise ValueError(f"缺少 proxy_return_col={proxy_return_col}")

    df = scored_panel.copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["trade_date"]).copy()
    if df.empty:
        return pd.DataFrame(
            columns=[
                "period",
                "trade_date",
                "strategy_return",
                "benchmark_return",
                "scenario",
                "excess_return",
                "benchmark_up",
                "strategy_up",
                "beat_benchmark",
            ]
        )

    chosen = select_rebalance_dates(df["trade_date"].unique(), rebalance_rule=rebalance_rule)
    rows: list[dict[str, Any]] = []
    for row in chosen.itertuples(index=False):
        day = df[df["trade_date"] == row.trade_date].copy()
        if day.empty:
            continue
        day[score_col] = pd.to_numeric(day[score_col], errors="coerce")
        day[proxy_return_col] = pd.to_numeric(day[proxy_return_col], errors="coerce")
        day = day[np.isfinite(day[score_col]) & np.isfinite(day[proxy_return_col])].copy()
        if day.empty:
            continue

        top = day.nlargest(top_k, columns=score_col, keep="all").head(top_k)
        strategy_ret = float(pd.to_numeric(top[proxy_return_col], errors="coerce").mean())
        benchmark_ret = float(pd.to_numeric(day[proxy_return_col], errors="coerce").mean())
        excess_ret = strategy_ret - benchmark_ret
        rows.append(
            {
                "period": row.period,
                "trade_date": pd.Timestamp(row.trade_date),
                "strategy_return": strategy_ret,
                "benchmark_return": benchmark_ret,
                "scenario": scenario,
                "excess_return": excess_ret,
                "benchmark_up": benchmark_ret > 0.0,
                "strategy_up": strategy_ret > 0.0,
                "beat_benchmark": excess_ret > 0.0,
            }
        )

    return pd.DataFrame(rows)


def summarize_tree_group_result(
    detail_df: pd.DataFrame,
    *,
    rebalance_rule: str,
) -> dict[str, float]:
    """统一输出树模型 P1 轻量代理摘要。"""
    periods_per_year = infer_periods_per_year(rebalance_rule)
    summary = summarize_signal_diagnostic(detail_df, periods_per_year=periods_per_year)
    summary["periods_per_year"] = float(periods_per_year)
    return summary


def summarize_p1_full_backtest_payload(
    payload: dict[str, Any],
    *,
    benchmark_key_years: Iterable[int] | None = None,
) -> dict[str, float | str]:
    """
    从 ``run_backtest_eval`` 的 JSON 抽取 P1 promotion gate 需要的核心字段。

    优先读取 benchmark-first 指标；如果当前 JSON 尚未提供 rolling/slice 的超额口径，
    对应字段保留为 ``NaN``，避免误把缺失当成通过。
    """
    years = tuple(int(y) for y in (benchmark_key_years or DEFAULT_P1_BENCHMARK_KEY_YEARS))
    full_sample = payload.get("full_sample", {}) or {}
    with_cost = full_sample.get("with_cost", {}) or {}
    excess_vs_market = full_sample.get("excess_vs_market", {}) or {}
    params = payload.get("parameters", {}) or {}
    meta = payload.get("meta", {}) or {}
    tree_model = meta.get("tree_model", {}) or {}
    prepared_cache = meta.get("prepared_factors_cache", {}) or {}
    rolling_agg = ((payload.get("walk_forward_rolling", {}) or {}).get("agg", {}) or {})
    slice_agg = ((payload.get("walk_forward_slices", {}) or {}).get("agg", {}) or {})
    rolling_excess = ((payload.get("walk_forward_rolling", {}) or {}).get("excess_vs_market", {}) or {})
    slice_excess = ((payload.get("walk_forward_slices", {}) or {}).get("excess_vs_market", {}) or {})
    yearly_rows = payload.get("yearly", []) or []

    yearly_df = pd.DataFrame(yearly_rows)
    if not yearly_df.empty:
        yearly_df["year"] = pd.to_numeric(yearly_df["year"], errors="coerce")
        yearly_df["excess"] = pd.to_numeric(yearly_df["excess"], errors="coerce")
        yearly_excess = yearly_df["excess"].to_numpy(dtype=np.float64)
        valid_yearly = yearly_excess[np.isfinite(yearly_excess)]
        yearly_excess_median = float(np.median(valid_yearly)) if valid_yearly.size else float("nan")
        key_df = yearly_df[yearly_df["year"].isin(years)].copy()
        key_vals = pd.to_numeric(key_df["excess"], errors="coerce").to_numpy(dtype=np.float64)
        key_vals = key_vals[np.isfinite(key_vals)]
        key_year_mean = float(np.mean(key_vals)) if key_vals.size else float("nan")
    else:
        yearly_excess_median = float("nan")
        key_year_mean = float("nan")

    return {
        "full_backtest_annualized_excess_vs_market": float(excess_vs_market.get("annualized_return", np.nan)),
        "full_backtest_sharpe_ratio": float(with_cost.get("sharpe_ratio", np.nan)),
        "full_backtest_annualized_return": float(with_cost.get("annualized_return", np.nan)),
        "full_backtest_max_drawdown": float(with_cost.get("max_drawdown", np.nan)),
        "rolling_oos_median_ann_return": float(rolling_agg.get("median_ann_return", np.nan)),
        "slice_oos_median_ann_return": float(slice_agg.get("median_ann_return", np.nan)),
        "rolling_oos_median_ann_excess_vs_market": float(
            rolling_agg.get("median_ann_excess_vs_market", rolling_excess.get("median_ann_excess_return", np.nan))
        ),
        "slice_oos_median_ann_excess_vs_market": float(
            slice_agg.get("median_ann_excess_vs_market", slice_excess.get("median_ann_excess_return", np.nan))
        ),
        "yearly_excess_median_vs_market": yearly_excess_median,
        "key_year_excess_mean_vs_market": key_year_mean,
        "benchmark_key_years": ",".join(str(y) for y in years),
        "full_backtest_research_topic": str(payload.get("research_topic", "")),
        "full_backtest_research_config_id": str(payload.get("research_config_id", "")),
        "full_backtest_output_stem": str(payload.get("output_stem", "")),
        "full_backtest_tree_bundle_dir": str(tree_model.get("bundle_dir") or params.get("tree_bundle_dir") or ""),
        "full_backtest_tree_feature_group": str(
            tree_model.get("feature_group") or params.get("tree_feature_group") or ""
        ),
        "full_backtest_tree_label_spec": json.dumps(
            tree_model.get("label_spec") or {}, ensure_ascii=False, sort_keys=True
        ),
        "full_backtest_tree_score_auto_flipped": bool(tree_model.get("tree_score_auto_flipped"))
        if tree_model.get("tree_score_auto_flipped") is not None
        else "",
        "full_backtest_prepared_factors_cache": str(
            prepared_cache.get("path") or params.get("prepared_factors_cache") or ""
        ),
        "full_backtest_prepared_cache_schema_version": str(
            prepared_cache.get("schema_version")
            or (params.get("prepared_factors_cache_meta") or {}).get("prepared_factors_schema_version")
            or ""
        ),
    }


def _non_degrade(series: pd.Series) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce")
    return vals.notna() & (vals >= 0.0)


def summarize_tree_score_direction(metrics: dict[str, Any]) -> dict[str, float | bool]:
    """
    对齐 ``predict_xgboost_tree`` 的自动翻转语义，给 P1 summary 显式展示有效方向。

    推理端优先看 ``val_rank_ic``，缺失时再看 ``train_rank_ic``；若该值为负，会把
    ``tree_score`` 乘以 -1。因此 P1 比较表需要同时保留 raw IC 与 effective IC。
    """
    chosen = np.nan
    for key in ("val_rank_ic", "train_rank_ic"):
        try:
            val = float(metrics.get(key, np.nan))
        except (TypeError, ValueError):
            continue
        if np.isfinite(val):
            chosen = val
            break
    auto_flipped = bool(np.isfinite(chosen) and chosen < 0.0)
    return {
        "tree_score_auto_flipped": auto_flipped,
        "effective_val_rank_ic": float(abs(chosen)) if np.isfinite(chosen) else float("nan"),
    }


def summarize_p1_label_diagnostics(
    panel: pd.DataFrame,
    *,
    target_column: str,
    label_columns: Iterable[str],
    label_weights: Iterable[float],
    proxy_return_col: str,
    date_col: str = "trade_date",
) -> dict[str, Any]:
    """诊断 P1 树模型标签口径，尤其是目标列与 light proxy 收益列的方向一致性。"""
    label_cols = [str(c) for c in label_columns]
    weights = [float(w) for w in label_weights]
    out: dict[str, Any] = {
        "target_column": target_column,
        "label_columns": ",".join(label_cols),
        "label_weights": ",".join(f"{w:.8g}" for w in weights),
        "proxy_return_col": proxy_return_col,
        "target_higher_is_better": True,
    }
    required = {date_col, target_column, proxy_return_col}
    missing = sorted(required - set(panel.columns))
    if missing:
        out.update(
            {
                "label_diagnostic_status": "missing_columns",
                "label_diagnostic_missing_columns": ",".join(missing),
            }
        )
        return out

    df = panel[[date_col, target_column, proxy_return_col]].copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce").dt.normalize()
    df[target_column] = pd.to_numeric(df[target_column], errors="coerce")
    df[proxy_return_col] = pd.to_numeric(df[proxy_return_col], errors="coerce")
    valid = df.dropna(subset=[date_col, target_column, proxy_return_col]).copy()
    out["label_row_count"] = int(len(panel))
    out["label_valid_row_count"] = int(len(valid))
    out["label_missing_rate"] = float(1.0 - (len(valid) / len(panel))) if len(panel) else float("nan")
    out["label_date_count"] = int(valid[date_col].nunique()) if not valid.empty else 0
    if valid.empty:
        out["label_diagnostic_status"] = "empty"
        return out

    out["target_mean"] = float(valid[target_column].mean())
    out["target_std"] = float(valid[target_column].std())
    out["proxy_return_mean"] = float(valid[proxy_return_col].mean())
    out["proxy_return_std"] = float(valid[proxy_return_col].std())

    per_day_corr: list[float] = []
    for _, g in valid.groupby(date_col, sort=False):
        if len(g) < 3:
            continue
        corr = g[target_column].rank(method="average").corr(
            g[proxy_return_col].rank(method="average")
        )
        if pd.notna(corr) and np.isfinite(float(corr)):
            per_day_corr.append(float(corr))
    out["target_proxy_rank_corr_mean"] = float(np.mean(per_day_corr)) if per_day_corr else float("nan")
    out["target_proxy_rank_corr_negative_rate"] = (
        float(np.mean(np.asarray(per_day_corr) < 0.0)) if per_day_corr else float("nan")
    )
    out["target_proxy_rank_corr_days"] = int(len(per_day_corr))
    out["label_diagnostic_status"] = "ok"
    return out


def build_tree_direction_diagnostic_table(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """汇总 P1 各组 raw Rank IC、自动翻转和目标方向是否稳定。"""
    if not rows:
        return pd.DataFrame()
    cols = [
        "group",
        "val_rank_ic",
        "train_rank_ic",
        "tree_score_auto_flipped",
        "effective_val_rank_ic",
        "annualized_excess_vs_market",
    ]
    df = pd.DataFrame(rows).copy()
    present = [c for c in cols if c in df.columns]
    out = df[present].copy()
    raw = pd.to_numeric(out.get("val_rank_ic"), errors="coerce")
    out["raw_val_rank_ic_negative"] = raw < 0.0
    if "tree_score_auto_flipped" in out.columns:
        auto_flipped = out["tree_score_auto_flipped"].astype(bool)
    else:
        auto_flipped = pd.Series(False, index=out.index)
    out["needs_direction_diagnosis"] = out["raw_val_rank_ic_negative"] | auto_flipped
    return out


def build_group_comparison_table(
    rows: list[dict[str, Any]],
    *,
    baseline_group: str = "G0",
) -> pd.DataFrame:
    """把各组结果整理成一张可直接落盘/写文档的比较表。"""
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).copy()
    out = out.sort_values("group", kind="mergesort").reset_index(drop=True)
    base = out.loc[out["group"] == baseline_group]
    if base.empty:
        out["delta_vs_baseline_val_rank_ic"] = np.nan
        out["delta_vs_baseline_proxy_excess"] = np.nan
        return out

    val_ic_col = "effective_val_rank_ic" if "effective_val_rank_ic" in out.columns else "val_rank_ic"
    base_val_ic = float(pd.to_numeric(base[val_ic_col], errors="coerce").iloc[0])
    base_proxy = float(pd.to_numeric(base["annualized_excess_vs_market"], errors="coerce").iloc[0])
    out["delta_vs_baseline_val_rank_ic"] = pd.to_numeric(out[val_ic_col], errors="coerce") - base_val_ic
    out["delta_vs_baseline_proxy_excess"] = (
        pd.to_numeric(out["annualized_excess_vs_market"], errors="coerce") - base_proxy
    )
    delta_pairs = {
        "full_backtest_annualized_excess_vs_market": "delta_vs_baseline_full_backtest_excess",
        "rolling_oos_median_ann_return": "delta_vs_baseline_rolling_oos_ann_return",
        "slice_oos_median_ann_return": "delta_vs_baseline_slice_oos_ann_return",
        "rolling_oos_median_ann_excess_vs_market": "delta_vs_baseline_rolling_oos_excess",
        "slice_oos_median_ann_excess_vs_market": "delta_vs_baseline_slice_oos_excess",
        "yearly_excess_median_vs_market": "delta_vs_baseline_yearly_excess_median",
        "key_year_excess_mean_vs_market": "delta_vs_baseline_key_year_excess_mean",
    }
    for col, delta_col in delta_pairs.items():
        if col not in out.columns:
            continue
        base_val = float(pd.to_numeric(base[col], errors="coerce").iloc[0])
        out[delta_col] = pd.to_numeric(out[col], errors="coerce") - base_val

    if "delta_vs_baseline_full_backtest_excess" in out.columns:
        out["pass_p1_val_rank_ic_gate"] = _non_degrade(out["delta_vs_baseline_val_rank_ic"])
        out["pass_p1_proxy_gate"] = _non_degrade(out["delta_vs_baseline_proxy_excess"])
        out["pass_p1_full_backtest_gate"] = _non_degrade(out["delta_vs_baseline_full_backtest_excess"])
        if "delta_vs_baseline_key_year_excess_mean" in out.columns:
            out["pass_p1_key_year_gate"] = _non_degrade(out["delta_vs_baseline_key_year_excess_mean"])
        else:
            out["pass_p1_key_year_gate"] = False
        if "delta_vs_baseline_rolling_oos_excess" in out.columns and out["delta_vs_baseline_rolling_oos_excess"].notna().any():
            out["pass_p1_rolling_oos_gate"] = _non_degrade(out["delta_vs_baseline_rolling_oos_excess"])
        elif "delta_vs_baseline_rolling_oos_ann_return" in out.columns:
            out["pass_p1_rolling_oos_gate"] = _non_degrade(out["delta_vs_baseline_rolling_oos_ann_return"])
        else:
            out["pass_p1_rolling_oos_gate"] = False
        if "delta_vs_baseline_slice_oos_excess" in out.columns and out["delta_vs_baseline_slice_oos_excess"].notna().any():
            out["pass_p1_slice_oos_gate"] = _non_degrade(out["delta_vs_baseline_slice_oos_excess"])
        elif "delta_vs_baseline_slice_oos_ann_return" in out.columns:
            out["pass_p1_slice_oos_gate"] = _non_degrade(out["delta_vs_baseline_slice_oos_ann_return"])
        else:
            out["pass_p1_slice_oos_gate"] = False
        gate_cols = [
            "pass_p1_val_rank_ic_gate",
            "pass_p1_rolling_oos_gate",
            "pass_p1_slice_oos_gate",
            "pass_p1_key_year_gate",
        ]
        out["pass_p1_promotion_gate"] = out[gate_cols].all(axis=1)
        out.loc[out["group"] == baseline_group, "pass_p1_promotion_gate"] = True
    return out
