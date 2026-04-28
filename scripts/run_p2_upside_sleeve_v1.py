"""R2 Day 3-5：Upside sleeve v1（规则 / 线性 composite，不训练模型）。

固定口径：
    top_k=20 / rebalance_rule=M / equal_weight / max_turnover=1.0 / tplus1_open
    universe_filter=cfg.universe_filter / prefilter=cfg.prefilter（默认 disabled）

候选（plan §3 Day 3-5）：
    BASELINE: S2 = vol_to_turnover（defensive sleeve 基线）
    A: rel_strength_20d + amount_expansion_5_60         （机制 1+2 同向）
    B: rel_strength_60d + turnover_expansion_5_60       （中长期版本）
    C: limit_up_hits_20d + tail_strength_20d            （可交易路径版）

每个候选都过 daily-proxy-first：使用 ``build_tree_daily_backtest_like_proxy_detail``
直接复用回测引擎 + open-to-open 收益 + market_ew 对齐。Gate 由 plan §2 R0：

    < 0%      → reject
    0% ~ +3%  → gray_zone
    >= +3%    → full_backtest_candidate

输出落到：
    data/results/{prefix}_*.csv / .json
    docs/{prefix}.md
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_backtest_eval import (
    _attach_pit_fundamentals,
    attach_universe_filter,
    build_market_ew_open_to_open_benchmark,
    build_open_to_open_returns,
    compute_factors,
    load_config,
    load_daily_from_duckdb,
    transaction_cost_params_from_mapping,
)
from scripts.run_p1_strong_up_attribution import (
    REGIME_ORDER,
    _compound_return,
    _json_sanitize,
    build_groups_per_rebalance,
    build_switch_quality,
    classify_regimes,
    compute_breadth,
    compute_r1_extra_features,
    summarize_breadth_capture,
    summarize_regime_capture,
    summarize_switch_by_regime,
)
from src.models.xtree.p1_workflow import (
    build_tree_daily_backtest_like_proxy_detail,
    summarize_tree_daily_backtest_like_proxy,
)


# 候选定义：(candidate_id, score_name, components, direction)
# direction 取值 +1/-1：score 越高越好则 +1。所有上行成分都希望越高越多参与，所以 +1。
# S2 = vol_to_turnover 越高越好（按 plan 默认基线）。
CANDIDATES: list[dict[str, Any]] = [
    {
        "id": "BASELINE_S2",
        "label": "S2 vol_to_turnover (defensive baseline)",
        "components": [("vol_to_turnover", +1.0)],
    },
    {
        "id": "UPSIDE_A_relstr20_amtexp",
        "label": "rel_strength_20d + amount_expansion_5_60",
        "components": [
            ("rel_strength_20d", +1.0),
            ("amount_expansion_5_60", +1.0),
        ],
    },
    {
        "id": "UPSIDE_B_relstr60_tnexp",
        "label": "rel_strength_60d + turnover_expansion_5_60",
        "components": [
            ("rel_strength_60d", +1.0),
            ("turnover_expansion_5_60", +1.0),
        ],
    },
    {
        "id": "UPSIDE_C_limitup_tail",
        "label": "limit_up_hits_20d + tail_strength_20d",
        "components": [
            ("limit_up_hits_20d", +1.0),
            ("tail_strength_20d", +1.0),
        ],
    },
]

GATE_REJECT = 0.0
GATE_FULL_BACKTEST = 0.03


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="R2 Day 3-5 upside sleeve v1 daily-proxy first")
    p.add_argument("--config", default="config.yaml.backtest")
    p.add_argument("--start", default="2021-01-01")
    p.add_argument("--end", default="")
    p.add_argument("--lookback-days", type=int, default=320)
    p.add_argument("--min-hist-days", type=int, default=130)
    p.add_argument("--output-prefix", default="p2_upside_sleeve_v1_2026-04-27")
    return p.parse_args()


def _add_cs_rank(panel: pd.DataFrame, col: str, out_col: str) -> pd.DataFrame:
    """每个 trade_date 内对 col 的 percentile rank（0~1，nan 透传）。"""
    s = pd.to_numeric(panel[col], errors="coerce")
    panel[out_col] = s.groupby(panel["trade_date"]).rank(pct=True, method="average")
    return panel


def build_candidate_scores(panel: pd.DataFrame) -> pd.DataFrame:
    """对所有候选成分，先做 cross-sectional rank，再按 direction 加权汇总。"""
    df = panel.copy()
    needed = sorted({c for cand in CANDIDATES for c, _ in cand["components"]})
    for col in needed:
        if col not in df.columns:
            raise KeyError(f"feature panel 缺少列: {col}")
        df = _add_cs_rank(df, col, f"_rk_{col}")
    for cand in CANDIDATES:
        rk_cols = [f"_rk_{c}" for c, _ in cand["components"]]
        signs = np.array([d for _, d in cand["components"]], dtype=float)
        ranks = df[rk_cols].to_numpy(dtype=float)
        # nan 行：组内任一缺失则该候选分数为 nan，避免被部分成分主导
        score = (ranks * signs).sum(axis=1)
        score = np.where(np.isnan(ranks).any(axis=1), np.nan, score)
        df[f"score__{cand['id']}"] = score
    return df


def _filter_universe(panel: pd.DataFrame) -> pd.DataFrame:
    if "_universe_eligible" not in panel.columns:
        return panel
    return panel[panel["_universe_eligible"].astype(bool)].copy()


def _run_one_candidate(
    *,
    cand: dict[str, Any],
    panel: pd.DataFrame,
    daily_df: pd.DataFrame,
    asset_returns: pd.DataFrame,
    bench_daily: pd.Series,
    cost_params: Any,
    top_k: int,
    rebalance_rule: str,
    max_turnover: float,
    monthly_regime_factory,
    benchmark_symbols: set[str],
    bench_min: int,
) -> dict[str, Any]:
    score_col = f"score__{cand['id']}"
    investable = panel.dropna(subset=[score_col]).copy()
    investable = investable[["symbol", "trade_date", score_col]]
    detail, meta = build_tree_daily_backtest_like_proxy_detail(
        investable,
        daily_df,
        score_col=score_col,
        rebalance_rule=rebalance_rule,
        top_k=top_k,
        max_turnover=max_turnover,
        scenario=cand["id"],
        cost_params=cost_params,
        execution_mode="tplus1_open",
        execution_lag=1,
        limit_up_mode="redistribute",
        market_ew_min_days=bench_min,
        precomputed_asset_returns=asset_returns,
        precomputed_market_benchmark=bench_daily,
    )
    summary = summarize_tree_daily_backtest_like_proxy(detail)

    # monthly 聚合（按月底 ME 复利）
    if detail.empty:
        monthly = pd.DataFrame(columns=["month_end", "strategy_return", "benchmark_return", "excess_return"])
    else:
        d = detail.copy()
        d["trade_date"] = pd.to_datetime(d["trade_date"]).dt.normalize()
        d = d.set_index("trade_date").sort_index()
        m = pd.DataFrame(
            {
                "strategy_return": d["strategy_return"].resample("ME").apply(_compound_return),
                "benchmark_return": d["benchmark_return"].resample("ME").apply(_compound_return),
            }
        )
        m["excess_return"] = m["strategy_return"] - m["benchmark_return"]
        monthly = m.dropna(how="all").reset_index(names="month_end")

    monthly_regime = monthly_regime_factory(monthly)

    # group / switch quality
    score_df_for_groups = investable.rename(columns={score_col: "score"})
    # 用每月最后一个交易日作为调仓日（select_rebalance_dates 已经在 weight_matrix 里处理；
    # 这里 switch_quality 只需要每个调仓日的 top20，可以用 detail 中可识别的调仓点近似）
    rebalance_actual = sorted(
        pd.to_datetime(detail["trade_date"]).dt.normalize().unique().tolist()
        if not detail.empty
        else []
    )
    # 真实调仓日：detail 中 turnover_half_l1 不为 NaN 的行
    if not detail.empty and "turnover_half_l1" in detail.columns:
        mask_rb = detail["turnover_half_l1"].notna()
        rebalance_actual = sorted(
            pd.to_datetime(detail.loc[mask_rb, "trade_date"]).dt.normalize().unique().tolist()
        )
    rebalance_groups = build_groups_per_rebalance(score_df_for_groups, rebalance_actual)
    switch_df = build_switch_quality(rebalance_groups, monthly_regime, asset_returns)
    switch_by_regime = summarize_switch_by_regime(switch_df)
    regime_capture = summarize_regime_capture(monthly_regime)
    breadth_capture = summarize_breadth_capture(monthly_regime)

    year_rows: list[dict[str, Any]] = []
    for year in (2021, 2025, 2026):
        for regime in REGIME_ORDER:
            part = monthly_regime[
                (monthly_regime["month_end"].dt.year == year) & (monthly_regime["regime"] == regime)
            ]
            if part.empty:
                continue
            year_rows.append(
                {
                    "year": year,
                    "regime": regime,
                    "months": int(len(part)),
                    "median_excess_return": float(part["excess_return"].median()),
                    "positive_excess_share": float((part["excess_return"] > 0).mean()),
                }
            )
    year_capture = pd.DataFrame(year_rows)

    return {
        "candidate": cand,
        "summary": summary,
        "meta": meta,
        "monthly": monthly_regime,
        "regime_capture": regime_capture,
        "breadth_capture": breadth_capture,
        "year_capture": year_capture,
        "switch_by_regime": switch_by_regime,
        "switch_detail": switch_df,
    }


def _gate_decision(daily_excess: float) -> str:
    if not np.isfinite(daily_excess):
        return "reject"
    if daily_excess < GATE_REJECT:
        return "reject"
    if daily_excess < GATE_FULL_BACKTEST:
        return "gray_zone"
    return "full_backtest_candidate"


def _build_leaderboard(results: list[dict[str, Any]], baseline_id: str) -> pd.DataFrame:
    base = next(r for r in results if r["candidate"]["id"] == baseline_id)

    def _strong_up_row(rc: pd.DataFrame) -> dict[str, float]:
        sub = rc[rc["regime"] == "strong_up"]
        if sub.empty:
            return {"strong_up_median_excess": np.nan, "strong_up_positive_share": np.nan, "strong_up_capture": np.nan}
        r = sub.iloc[0]
        return {
            "strong_up_median_excess": float(r["median_excess_return"]),
            "strong_up_positive_share": float(r["positive_excess_share"]),
            "strong_up_capture": float(r["capture_ratio"]),
        }

    def _strong_down_row(rc: pd.DataFrame) -> dict[str, float]:
        sub = rc[rc["regime"] == "strong_down"]
        if sub.empty:
            return {"strong_down_median_excess": np.nan, "strong_down_positive_share": np.nan}
        r = sub.iloc[0]
        return {
            "strong_down_median_excess": float(r["median_excess_return"]),
            "strong_down_positive_share": float(r["positive_excess_share"]),
        }

    def _switch_strong_up(sw: pd.DataFrame) -> dict[str, float]:
        sub = sw[sw["regime"] == "strong_up"] if not sw.empty else pd.DataFrame()
        if sub.empty:
            return {"strong_up_switch_in_minus_out": np.nan, "strong_up_topk_minus_next": np.nan}
        r = sub.iloc[0]
        return {
            "strong_up_switch_in_minus_out": float(r["mean_switch_in_minus_out"]),
            "strong_up_topk_minus_next": float(r["mean_topk_minus_next"]),
        }

    base_su = _strong_up_row(base["regime_capture"])
    base_sd = _strong_down_row(base["regime_capture"])
    base_proxy = float(base["summary"].get("annualized_excess_vs_market", np.nan))
    base_turnover = float(base["meta"].get("avg_turnover_half_l1", np.nan))

    rows: list[dict[str, Any]] = []
    for r in results:
        s = r["summary"]
        m = r["meta"]
        proxy = float(s.get("annualized_excess_vs_market", np.nan))
        su = _strong_up_row(r["regime_capture"])
        sd = _strong_down_row(r["regime_capture"])
        sw = _switch_strong_up(r["switch_by_regime"])
        rows.append(
            {
                "candidate_id": r["candidate"]["id"],
                "label": r["candidate"]["label"],
                "daily_proxy_annualized_excess_vs_market": proxy,
                "delta_vs_baseline_proxy": proxy - base_proxy,
                "gate_decision": _gate_decision(proxy),
                "strong_up_median_excess": su["strong_up_median_excess"],
                "delta_vs_baseline_strong_up_median_excess": su["strong_up_median_excess"] - base_su["strong_up_median_excess"],
                "strong_up_positive_share": su["strong_up_positive_share"],
                "delta_vs_baseline_strong_up_positive_share": su["strong_up_positive_share"] - base_su["strong_up_positive_share"],
                "strong_up_capture": su["strong_up_capture"],
                "strong_down_median_excess": sd["strong_down_median_excess"],
                "delta_vs_baseline_strong_down_median_excess": sd["strong_down_median_excess"] - base_sd["strong_down_median_excess"],
                "strong_up_switch_in_minus_out": sw["strong_up_switch_in_minus_out"],
                "strong_up_topk_minus_next": sw["strong_up_topk_minus_next"],
                "avg_turnover_half_l1": float(m.get("avg_turnover_half_l1", np.nan)),
                "delta_vs_baseline_turnover": float(m.get("avg_turnover_half_l1", np.nan)) - base_turnover,
                "n_periods": int(s.get("n_periods", 0)),
                "n_rebalances": int(m.get("n_rebalances", 0)),
                "primary_result_type": "daily_bt_like_proxy",
                "primary_decision_metric": "daily_bt_like_proxy_annualized_excess_vs_market",
                "p1_experiment_mode": "daily_proxy_first",
                "legacy_proxy_decision_role": "diagnostic_only",
            }
        )
    out = pd.DataFrame(rows)
    out["_sort"] = out["candidate_id"].apply(lambda x: 0 if x == baseline_id else 1)
    out["_proxy_sort"] = -pd.to_numeric(out["daily_proxy_annualized_excess_vs_market"], errors="coerce")
    out = out.sort_values(["_sort", "_proxy_sort"]).drop(columns=["_sort", "_proxy_sort"]).reset_index(drop=True)
    return out


def _accept_summary(row: pd.Series, baseline: pd.Series) -> dict[str, Any]:
    """对照 plan §2 R2 验收清单逐条判定（baseline 自身不评定）。"""
    if row["candidate_id"] == baseline["candidate_id"]:
        return {"status": "baseline", "checks": {}}
    checks: dict[str, str] = {}
    proxy = row["daily_proxy_annualized_excess_vs_market"]
    checks["daily_proxy_>=_0"] = "pass" if np.isfinite(proxy) and proxy >= 0 else "fail"
    su_med_delta = row["delta_vs_baseline_strong_up_median_excess"]
    checks["strong_up_median_excess_improves"] = (
        "pass" if np.isfinite(su_med_delta) and su_med_delta > 0 else "fail"
    )
    su_pos_delta = row["delta_vs_baseline_strong_up_positive_share"]
    checks["strong_up_positive_share_improves"] = (
        "pass" if np.isfinite(su_pos_delta) and su_pos_delta > 0 else "fail"
    )
    sd_med_delta = row["delta_vs_baseline_strong_down_median_excess"]
    # 不明显劣化：放宽至 -2pct 以内
    checks["strong_down_not_materially_worse"] = (
        "pass" if np.isfinite(sd_med_delta) and sd_med_delta >= -0.02 else "fail"
    )
    overall = "pass" if all(v == "pass" for v in checks.values()) else "fail"
    return {"status": overall, "checks": checks}


def _build_doc(
    *,
    config_source: str,
    params: dict[str, Any],
    leaderboard: pd.DataFrame,
    accept_map: dict[str, dict[str, Any]],
    regime_long: pd.DataFrame,
    breadth_long: pd.DataFrame,
    year_long: pd.DataFrame,
    switch_long: pd.DataFrame,
    output_prefix: str,
) -> str:
    lines: list[str] = []
    lines.append("# R2 Upside Sleeve v1 (Day 3-5)\n")
    lines.append(f"- 生成时间：`{pd.Timestamp.utcnow().isoformat()}`")
    lines.append(f"- 配置快照：`{config_source}`")
    lines.append(
        "- 固定口径：`top_k=20` / `M` / `equal_weight` / `max_turnover=1.0` / `tplus1_open` / "
        "`universe_filter=on` / `prefilter=off`"
    )
    lines.append("- `p1_experiment_mode`: `daily_proxy_first`")
    lines.append("- `legacy_proxy_decision_role`: `diagnostic_only`")
    lines.append("- `primary_decision_metric`: `daily_bt_like_proxy_annualized_excess_vs_market`")
    lines.append("- Gate：`<0%`→reject / `0%~+3%`→gray_zone / `>=+3%`→full_backtest_candidate")
    lines.append("")

    lines.append("## 1. Leaderboard\n")
    cols_show = [
        "candidate_id",
        "label",
        "daily_proxy_annualized_excess_vs_market",
        "delta_vs_baseline_proxy",
        "gate_decision",
        "strong_up_median_excess",
        "delta_vs_baseline_strong_up_median_excess",
        "strong_up_positive_share",
        "delta_vs_baseline_strong_up_positive_share",
        "strong_down_median_excess",
        "delta_vs_baseline_strong_down_median_excess",
        "strong_up_switch_in_minus_out",
        "avg_turnover_half_l1",
        "delta_vs_baseline_turnover",
        "n_rebalances",
    ]
    cols_show = [c for c in cols_show if c in leaderboard.columns]
    lines.append(leaderboard[cols_show].to_markdown(index=False, floatfmt=".4f"))
    lines.append("")

    lines.append("## 2. R2 验收（对 BASELINE 的相对改善）\n")
    accept_rows = []
    for cid, info in accept_map.items():
        row = {"candidate_id": cid, "status": info["status"]}
        row.update(info.get("checks", {}))
        accept_rows.append(row)
    if accept_rows:
        lines.append(pd.DataFrame(accept_rows).to_markdown(index=False))
    lines.append("")

    lines.append("## 3. Regime 切片（candidate × regime）\n")
    lines.append(regime_long.to_markdown(index=False, floatfmt=".4f"))
    lines.append("")

    lines.append("## 4. Breadth 切片（candidate × breadth）\n")
    lines.append(breadth_long.to_markdown(index=False, floatfmt=".4f"))
    lines.append("")

    lines.append("## 5. 关键年份 strong_up（2021/2025/2026）\n")
    if year_long.empty:
        lines.append("_无数据_")
    else:
        sub = year_long[year_long["regime"] == "strong_up"]
        if sub.empty:
            lines.append("_strong_up 无样本_")
        else:
            lines.append(sub.to_markdown(index=False, floatfmt=".4f"))
    lines.append("")

    lines.append("## 6. Switch quality（candidate × regime）\n")
    if switch_long.empty:
        lines.append("_无 switch 样本_")
    else:
        lines.append(switch_long.to_markdown(index=False, floatfmt=".4f"))
    lines.append("")

    lines.append("## 7. 结论 / 下一步\n")
    base_id = leaderboard.iloc[0]["candidate_id"]
    promoted = [cid for cid, info in accept_map.items() if info["status"] == "pass"]
    grayzone = [
        r["candidate_id"]
        for _, r in leaderboard.iterrows()
        if r["candidate_id"] != base_id and r["gate_decision"] == "gray_zone"
    ]
    rejected = [
        r["candidate_id"]
        for _, r in leaderboard.iterrows()
        if r["candidate_id"] != base_id and r["gate_decision"] == "reject"
    ]
    if promoted:
        lines.append(f"- 通过 R2 验收的候选：{', '.join(promoted)}。可进入 Day 6-7 双袖套权重规则；")
    else:
        lines.append("- **没有候选同时满足 R2 验收四条**。Day 6-7 双袖套不能直接接 v1 score。")
    if grayzone:
        lines.append(f"- 进入 daily proxy gray_zone（0%~+3%）：{', '.join(grayzone)}（仅归档诊断，不补 full backtest）。")
    if rejected:
        lines.append(f"- daily proxy <0%（reject）：{', '.join(rejected)}（停止该候选）。")
    lines.append(
        "- 后续若所有候选都未通过 R2 验收，下一轮应在 Day 6-7 之前补 1-2 个新组合（如 `relative_strength + industry_breadth`、"
        "`tradable_breakout + turnover_expansion`），保持每轮 ≤3。"
    )
    lines.append("")

    lines.append("## 8. 产出文件\n")
    for suf in [
        "leaderboard.csv",
        "regime_long.csv",
        "breadth_long.csv",
        "year_long.csv",
        "switch_long.csv",
        "monthly_long.csv",
        "summary.json",
    ]:
        lines.append(f"- `data/results/{output_prefix}_{suf}`")
    lines.append("")

    lines.append("## 9. 配置参数\n")
    for k, v in params.items():
        lines.append(f"- `{k}`: `{v}`")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    cfg, config_source = load_config(args.config)
    paths_cfg = cfg.get("paths", {}) or {}
    db_path_raw = paths_cfg.get("duckdb_path") or paths_cfg.get("database_path") or "data/market.duckdb"
    db_path = str(db_path_raw if Path(db_path_raw).is_absolute() else PROJECT_ROOT / db_path_raw)
    end_date = args.end or str(paths_cfg.get("asof_trade_date") or pd.Timestamp.today().strftime("%Y-%m-%d"))

    backtest_cfg = cfg.get("backtest", {}) or {}
    portfolio_cfg = cfg.get("portfolio", {}) or {}
    signals_cfg = cfg.get("signals", {}) or {}
    prefilter_cfg = cfg.get("prefilter", {}) or {}
    uf_cfg = cfg.get("universe_filter", {}) or {}
    risk_cfg = cfg.get("risk", {}) or {}

    top_k = int(signals_cfg.get("top_k", 20))
    rebalance_rule = str(backtest_cfg.get("eval_rebalance_rule", "M"))
    max_turnover = float(portfolio_cfg.get("max_turnover", 1.0))

    print(f"[1/6] load daily {args.start}->{end_date}", flush=True)
    daily_df = load_daily_from_duckdb(db_path, args.start, end_date, args.lookback_days)

    print("[2/6] compute factors + R1 extras + universe filter", flush=True)
    factors = compute_factors(daily_df, min_hist_days=args.min_hist_days)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
        factors = _attach_pit_fundamentals(factors, db_path)
    factors = attach_universe_filter(
        factors,
        daily_df,
        enabled=bool(uf_cfg.get("enabled", False)),
        min_amount_20d=float(uf_cfg.get("min_amount_20d", 50_000_000)),
        require_roe_ttm_positive=bool(uf_cfg.get("require_roe_ttm_positive", True)),
    )
    extras = compute_r1_extra_features(daily_df)
    panel = factors.merge(extras, on=["symbol", "trade_date"], how="left")
    panel = panel[panel["trade_date"] >= pd.Timestamp(args.start)].copy()
    panel = _filter_universe(panel)
    print(f"  panel(after filter)={panel.shape}", flush=True)

    print("[3/6] build candidate scores (cross-sectional rank)", flush=True)
    panel = build_candidate_scores(panel)

    print("[4/6] precompute open-to-open returns + market_ew benchmark", flush=True)
    open_returns = build_open_to_open_returns(daily_df, zero_if_limit_up_open=False).sort_index()
    n_trade_days = int(open_returns.index.nunique())
    bench_min = max(60, int(0.35 * max(n_trade_days, 1)))
    bench_daily = build_market_ew_open_to_open_benchmark(daily_df, args.start, end_date, min_days=bench_min)
    sym_counts = daily_df.groupby("symbol")["trade_date"].count()
    benchmark_symbols = set(sym_counts[sym_counts >= bench_min].index.astype(str))

    breadth_series = compute_breadth(daily_df, benchmark_symbols)

    # Lock regime thresholds against the BASELINE_S2 monthly series (与 R1 同口径)：
    # 直接复用 daily proxy 跑 baseline 后的 monthly 来推导 regime 切片，
    # 但所有候选都用同一份 monthly_regime（来自 baseline）以保证可比。
    # 然而 plan 的 regime 是 by 基准月收益分位，对所有候选完全相同；只 strategy_return 不同。
    # 因此我们用一个 factory：传入候选 monthly（含 strategy/benchmark/excess），
    # benchmark_return 在所有候选间一致 -> regime 切片相同。

    cost_params = transaction_cost_params_from_mapping(cfg.get("transaction_costs", {}))

    # Pre-load asset_returns reindexed to all sym in panel
    sym_universe = sorted(set(panel["symbol"].astype(str).str.zfill(6).unique()))
    asset_returns = open_returns.reindex(columns=sym_universe).fillna(0.0)

    # 我们在 baseline 第一次跑完后，把 baseline 的 benchmark_return 当作 regime 阈值锚点，
    # 对所有候选使用同一阈值（基于第一个已跑出的候选 monthly 的 benchmark_return）。
    baseline_thresholds: dict[str, dict[str, float]] = {}

    def make_monthly_regime_factory():
        def _factory(monthly: pd.DataFrame) -> pd.DataFrame:
            if monthly.empty:
                cols = ["month_end", "strategy_return", "benchmark_return", "excess_return", "regime", "breadth"]
                return pd.DataFrame(columns=cols)
            mr = classify_regimes(monthly, breadth_series)
            if "p20" not in baseline_thresholds:
                baseline_thresholds["regime"] = mr.attrs["regime_thresholds"]
                baseline_thresholds["breadth"] = mr.attrs["breadth_thresholds"]
            return mr
        return _factory

    factory = make_monthly_regime_factory()

    print(f"[5/6] run daily proxy for {len(CANDIDATES)} candidates", flush=True)
    results: list[dict[str, Any]] = []
    for cand in CANDIDATES:
        print(f"  -> {cand['id']}", flush=True)
        res = _run_one_candidate(
            cand=cand,
            panel=panel,
            daily_df=daily_df,
            asset_returns=asset_returns,
            bench_daily=bench_daily,
            cost_params=cost_params,
            top_k=top_k,
            rebalance_rule=rebalance_rule,
            max_turnover=max_turnover,
            monthly_regime_factory=factory,
            benchmark_symbols=benchmark_symbols,
            bench_min=bench_min,
        )
        results.append(res)

    print("[6/6] aggregate + write outputs", flush=True)
    leaderboard = _build_leaderboard(results, baseline_id="BASELINE_S2")
    base_row = leaderboard[leaderboard["candidate_id"] == "BASELINE_S2"].iloc[0]
    accept_map: dict[str, dict[str, Any]] = {}
    for _, row in leaderboard.iterrows():
        accept_map[row["candidate_id"]] = _accept_summary(row, base_row)

    # long format helpers
    def _long_with_id(df: pd.DataFrame, cid: str) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()
        out = df.copy()
        out.insert(0, "candidate_id", cid)
        return out

    regime_long = pd.concat([_long_with_id(r["regime_capture"], r["candidate"]["id"]) for r in results], ignore_index=True)
    breadth_long = pd.concat([_long_with_id(r["breadth_capture"], r["candidate"]["id"]) for r in results], ignore_index=True)
    year_long = pd.concat([_long_with_id(r["year_capture"], r["candidate"]["id"]) for r in results], ignore_index=True)
    switch_long = pd.concat([_long_with_id(r["switch_by_regime"], r["candidate"]["id"]) for r in results], ignore_index=True)
    monthly_long = pd.concat([_long_with_id(r["monthly"], r["candidate"]["id"]) for r in results], ignore_index=True)

    results_dir = PROJECT_ROOT / "data" / "results"
    docs_dir = PROJECT_ROOT / "docs"
    results_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.output_prefix

    leaderboard.to_csv(results_dir / f"{prefix}_leaderboard.csv", index=False, encoding="utf-8-sig")
    regime_long.to_csv(results_dir / f"{prefix}_regime_long.csv", index=False, encoding="utf-8-sig")
    breadth_long.to_csv(results_dir / f"{prefix}_breadth_long.csv", index=False, encoding="utf-8-sig")
    year_long.to_csv(results_dir / f"{prefix}_year_long.csv", index=False, encoding="utf-8-sig")
    switch_long.to_csv(results_dir / f"{prefix}_switch_long.csv", index=False, encoding="utf-8-sig")
    monthly_long.to_csv(results_dir / f"{prefix}_monthly_long.csv", index=False, encoding="utf-8-sig")

    params = {
        "start": args.start,
        "end": end_date,
        "top_k": top_k,
        "rebalance_rule": rebalance_rule,
        "portfolio_method": "equal_weight",
        "max_turnover": max_turnover,
        "execution_mode": "tplus1_open",
        "prefilter": prefilter_cfg,
        "universe_filter": uf_cfg,
        "benchmark_symbol": str(risk_cfg.get("benchmark_symbol", "market_ew_proxy")),
        "benchmark_min_history_days": bench_min,
        "config_source": config_source,
        "p1_experiment_mode": "daily_proxy_first",
        "legacy_proxy_decision_role": "diagnostic_only",
        "primary_decision_metric": "daily_bt_like_proxy_annualized_excess_vs_market",
        "gate_thresholds": {"reject": GATE_REJECT, "full_backtest": GATE_FULL_BACKTEST},
    }
    summary = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "config_source": config_source,
        "parameters": params,
        "regime_thresholds": baseline_thresholds.get("regime"),
        "breadth_thresholds": baseline_thresholds.get("breadth"),
        "leaderboard": leaderboard.to_dict(orient="records"),
        "accept": accept_map,
        "candidates": [
            {
                "id": r["candidate"]["id"],
                "label": r["candidate"]["label"],
                "components": r["candidate"]["components"],
                "summary": r["summary"],
                "meta": r["meta"],
            }
            for r in results
        ],
    }
    with open(results_dir / f"{prefix}_summary.json", "w", encoding="utf-8") as f:
        json.dump(_json_sanitize(summary), f, ensure_ascii=False, indent=2)

    doc_text = _build_doc(
        config_source=config_source,
        params=params,
        leaderboard=leaderboard,
        accept_map=accept_map,
        regime_long=regime_long,
        breadth_long=breadth_long,
        year_long=year_long,
        switch_long=switch_long,
        output_prefix=prefix,
    )
    (docs_dir / f"{prefix}.md").write_text(doc_text, encoding="utf-8")
    print(f"  doc -> {docs_dir / f'{prefix}.md'}", flush=True)


if __name__ == "__main__":
    main()
