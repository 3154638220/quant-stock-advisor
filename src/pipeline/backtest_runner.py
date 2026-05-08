"""回测执行核心管线（编排层）。

本模块只保留高层编排与回测特有的业务逻辑：
- 数据加载 & PIT 基本面 attach
- Regime 动态调权
- Top-K 权重构建

因子计算、缓存、截面打分分别移至：
- :mod:`src.pipeline.factor_computer`
- :mod:`src.pipeline.factor_cache`
- :mod:`src.pipeline.score_builder`
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import duckdb
import numpy as np
import pandas as pd

from src.backtest.engine import build_open_to_open_returns as _build_open_to_open_returns
from src.features.fund_flow_factors import attach_fund_flow
from src.features.fundamental_factors import pit_safe_fundamental_rows, preprocess_fundamental_cross_section
from src.features.shareholder_factors import attach_shareholder_factors
from src.market.regime import (
    MARKET_EW_PROXY,
    classify_regime,
    get_regime_weights,
    regime_config_from_mapping,
)
from src.pipeline.factor_cache import (
    load_prepared_factors_cache,
    write_prepared_factors_cache,
)
from src.pipeline.factor_computer import compute_factors
from src.pipeline.monthly_dataset import read_daily_from_duckdb
from src.pipeline.shared_loaders import DataLoader, DataLoaderConfig
from src.portfolio.covariance import mean_cov_returns_from_daily_long
from src.portfolio.weights import build_portfolio_weights
from src.research.gates import apply_prefilter, attach_universe_filter

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Re-export public API from sub-modules for backward compatibility
from src.pipeline.factor_computer import (  # noqa: F401, E402
    PREPARED_FACTORS_REQUIRED_COLUMNS,
    PREPARED_FACTORS_SCHEMA_VERSION,
)
from src.pipeline.score_builder import (  # noqa: F401, E402
    apply_p1_factor_policy,
    build_ic_weights_from_monitor,
    build_score,
    build_weights_by_date,
    load_factor_ic_summary,
    load_ic_weights_by_date,
    normalize_weights,
)

# ── 日线加载 ─────────────────────────────────────────────────────────────

def load_daily_from_duckdb(
    db_path: str, start: str, end: str, lookback_days: int
) -> pd.DataFrame:
    with duckdb.connect(db_path, read_only=True) as con:
        return read_daily_from_duckdb(
            con,
            table="a_share_daily",
            start=start,
            end=end,
            min_history_days=lookback_days,
            price_position_lookback=lookback_days,
        )


# ── PIT 基本面 attach ────────────────────────────────────────────────────

def _attach_pit_fundamentals(factors: pd.DataFrame, db_path: str) -> pd.DataFrame:
    """按公告日 merge_asof，将基本面快照对齐到每个交易日（PIT）。"""
    out = factors.copy(deep=False)
    con = duckdb.connect(db_path, read_only=True)
    want_cols = [
        "symbol", "report_period", "announcement_date",
        "pe_ttm", "pb", "ev_ebitda", "roe_ttm", "net_profit_yoy",
        "gross_margin_change", "debt_to_assets_change", "ocf_to_net_profit",
        "ocf_to_asset", "gross_margin_delta", "asset_turnover",
        "net_margin_stability", "northbound_net_inflow", "margin_buy_ratio", "source",
    ]
    try:
        exists = con.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'a_share_fundamental'"
        ).fetchone()
        if not exists or int(exists[0]) <= 0:
            return out
        info = con.execute("PRAGMA table_info('a_share_fundamental')").fetchall()
        have = {str(r[1]) for r in info}
        sel = [c for c in want_cols if c in have]
        fund = con.execute(f"SELECT {', '.join(sel)} FROM a_share_fundamental").df()
        for c in want_cols:
            if c not in fund.columns:
                fund[c] = "" if c == "source" else np.nan
    finally:
        con.close()
    if fund.empty:
        return out

    out["symbol"] = out["symbol"].astype(str).str.zfill(6)
    out["trade_date"] = pd.to_datetime(out["trade_date"], errors="coerce").dt.normalize()
    out = out.dropna(subset=["trade_date"])
    fund["symbol"] = fund["symbol"].astype(str).str.zfill(6)
    fund["announcement_date"] = (
        pd.to_datetime(fund["announcement_date"], errors="coerce").dt.normalize()
    )
    fund["report_period"] = pd.to_datetime(fund["report_period"], errors="coerce")
    fund = fund.dropna(subset=["announcement_date"])
    fund = fund[pit_safe_fundamental_rows(fund)].copy()
    if fund.empty:
        return out
    fund = fund.sort_values(["symbol", "announcement_date", "report_period"], na_position="last", kind="mergesort")
    fund = fund.drop_duplicates(["symbol", "announcement_date"], keep="last")
    fund = fund.drop(columns=["report_period", "source"], errors="ignore")

    out = out.sort_values(["trade_date", "symbol"], kind="mergesort").reset_index(drop=True)
    fund = fund.sort_values(["announcement_date", "symbol"], kind="mergesort").reset_index(drop=True)
    chunked: list[pd.DataFrame] = []
    for _, chunk in out.groupby(pd.Grouper(key="trade_date", freq="31D"), sort=True):
        if chunk.empty:
            continue
        chunk = chunk.sort_values(["trade_date", "symbol"], kind="mergesort").reset_index(drop=True)
        chunk_end = pd.Timestamp(chunk["trade_date"].max())
        chunk_symbols = chunk["symbol"].astype(str).unique().tolist()
        fund_chunk = fund[
            (fund["announcement_date"] <= chunk_end)
            & fund["symbol"].astype(str).isin(chunk_symbols)
        ].copy()
        if fund_chunk.empty:
            merged = chunk.copy()
            for c in want_cols:
                if c not in merged.columns:
                    merged[c] = np.nan
        else:
            fund_chunk = fund_chunk.sort_values(["announcement_date", "symbol"], kind="mergesort").reset_index(drop=True)
            merged = pd.merge_asof(
                chunk, fund_chunk,
                left_on="trade_date", right_on="announcement_date",
                by="symbol", direction="backward", allow_exact_matches=True,
            )
        merged = preprocess_fundamental_cross_section(
            merged, date_col="trade_date", size_col="log_market_cap", neutralize=True,
        )
        chunked.append(merged)
    if not chunked:
        return out
    return pd.concat(chunked, ignore_index=True)


# ── 因子准备管线 ─────────────────────────────────────────────────────────

def prepare_factors_for_backtest(
    daily_df: pd.DataFrame,
    *,
    min_hist_days: int,
    db_path: str,
    results_dir: Path,
    universe_filter_cfg: dict[str, Any],
    cache_path: Path | None = None,
    refresh_cache: bool = False,
    cache_meta: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, bool]:
    expected_meta = dict(cache_meta or {})
    if cache_path is not None and not refresh_cache:
        cached = load_prepared_factors_cache(cache_path, expected_meta)
        if cached is not None:
            return cached, True

    factors = compute_factors(daily_df, min_hist_days=min_hist_days)
    loader = DataLoader(
        db_path,
        config=DataLoaderConfig(strict_unknown_families=True),
        registry={
            "fundamental": lambda frame: _attach_pit_fundamentals(frame, db_path),
            "fund_flow": lambda frame: attach_fund_flow(frame, db_path),
            "shareholder": lambda frame: attach_shareholder_factors(frame, db_path),
        },
        family_order=("fundamental", "fund_flow", "shareholder"),
    )
    factors = loader.attach(factors, ("fundamental", "fund_flow", "shareholder"))
    factors = attach_universe_filter(
        factors, daily_df,
        enabled=bool(universe_filter_cfg.get("enabled", False)),
        min_amount_20d=float(universe_filter_cfg.get("min_amount_20d", 50_000_000)),
        require_roe_ttm_positive=bool(universe_filter_cfg.get("require_roe_ttm_positive", True)),
    )
    if cache_path is not None:
        write_prepared_factors_cache(cache_path, factors, expected_meta)
    return factors, False


# ── 行业映射 ─────────────────────────────────────────────────────────────

def load_industry_map(industry_map_csv: str) -> Dict[str, str]:
    p = Path(industry_map_csv)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    if not p.exists():
        return {}
    tab = pd.read_csv(
        p,
        encoding="utf-8-sig",
        converters={
            "symbol": lambda v: str(v).strip(),
            "代码": lambda v: str(v).strip(),
            "industry": lambda v: str(v).strip(),
            "行业": lambda v: str(v).strip(),
        },
    )
    sym_col = "symbol" if "symbol" in tab.columns else ("代码" if "代码" in tab.columns else None)
    ind_col = "industry" if "industry" in tab.columns else ("行业" if "行业" in tab.columns else None)
    if sym_col is None or ind_col is None:
        return {}
    tab = tab[[sym_col, ind_col]].copy()
    tab[sym_col] = tab[sym_col].astype(str).str.zfill(6)
    tab[ind_col] = tab[ind_col].astype(str).str.strip()
    tab = tab[(tab[sym_col].str.len() == 6) & (tab[ind_col] != "")]
    tab = tab.drop_duplicates(subset=[sym_col], keep="last")
    return dict(zip(tab[sym_col], tab[ind_col]))


# ── Regime 动态调权 ──────────────────────────────────────────────────────

def build_market_ew_benchmark(
    daily_df: pd.DataFrame, start: str, end: str, min_days: int = 500
) -> pd.Series:
    """构建全市场等权日收益基准（close-to-close）。"""
    df = daily_df[
        (daily_df["trade_date"] >= pd.Timestamp(start))
        & (daily_df["trade_date"] <= pd.Timestamp(end))
        & (daily_df["close"] > 0)
    ].copy()
    sym_cnt = df.groupby("symbol")["trade_date"].count()
    good = sym_cnt[sym_cnt >= min_days].index
    if len(good) == 0:
        good = sym_cnt.index
    df = df[df["symbol"].isin(good)].sort_values(["symbol", "trade_date"])
    df["ret"] = df.groupby("symbol")["close"].pct_change()
    cs = df.dropna(subset=["ret"]).groupby("trade_date")["ret"].mean()
    cs.index = pd.to_datetime(cs.index)
    return cs


def build_regime_weight_overrides(
    factors: pd.DataFrame,
    daily_df: pd.DataFrame,
    base_weights: Dict[str, float],
    benchmark_symbol: str,
    regime_cfg_raw: dict,
    *,
    market_ew_min_days: int = 130,
) -> Tuple[Dict[pd.Timestamp, Dict[str, float]], pd.DataFrame]:
    cfg = regime_config_from_mapping(regime_cfg_raw)
    sym_key = str(benchmark_symbol).strip()
    if sym_key.lower() == MARKET_EW_PROXY.lower():
        fac_start = pd.to_datetime(factors["trade_date"]).min()
        fac_end = pd.to_datetime(factors["trade_date"]).max()
        bench_s = build_market_ew_benchmark(
            daily_df, str(fac_start.date()), str(fac_end.date()),
            min_days=int(market_ew_min_days),
        )
        if bench_s.empty:
            return {}, pd.DataFrame(columns=["trade_date", "regime", "short_return", "vol_ann"])
    else:
        bench = daily_df[daily_df["symbol"].astype(str).str.zfill(6) == sym_key.zfill(6)].copy()
        if bench.empty:
            return {}, pd.DataFrame(columns=["trade_date", "regime", "short_return", "vol_ann"])
        bench = bench.sort_values("trade_date")
        bench["ret"] = pd.to_numeric(bench["close"], errors="coerce").pct_change()
        bench = bench.dropna(subset=["ret"])
        bench_s = pd.Series(
            bench["ret"].to_numpy(dtype=np.float64), index=pd.to_datetime(bench["trade_date"])
        ).sort_index()

    overrides: Dict[pd.Timestamp, Dict[str, float]] = {}
    rows: list[dict[str, Any]] = []
    for dt in sorted(pd.to_datetime(factors["trade_date"]).unique()):
        regime, result = classify_regime(bench_s, dt, cfg=cfg)
        overrides[pd.Timestamp(dt)] = get_regime_weights(base_weights, regime, cfg=cfg, regime_result=result)
        rows.append({
            "trade_date": pd.Timestamp(dt),
            "regime": regime,
            "short_return": float(result.short_return),
            "vol_ann": float(result.realized_vol_ann),
        })
    return overrides, pd.DataFrame(rows)


# ── Top-K 权重构建 ───────────────────────────────────────────────────────

def _rebalance_dates(all_dates: Iterable[pd.Timestamp], rule: str) -> list[pd.Timestamp]:
    from src.pipeline.monthly_dataset import select_month_end_signal_dates
    return select_month_end_signal_dates(list(all_dates), rebalance_rule=rule)


def _pick_topk_with_industry_cap(
    day_df: pd.DataFrame,
    *,
    top_k: int,
    industry_map: Dict[str, str] | None,
    industry_cap_count: int | None,
) -> pd.DataFrame:
    ranked = day_df.nlargest(top_k * 5, "score")[["symbol", "score"]].copy()
    ranked["symbol"] = ranked["symbol"].astype(str).str.zfill(6)
    if not industry_map or not industry_cap_count or industry_cap_count <= 0:
        return ranked.nlargest(top_k, "score")

    cap = int(industry_cap_count)
    picked: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for _, r in ranked.sort_values("score", ascending=False).iterrows():
        sym = str(r["symbol"]).zfill(6)
        ind = industry_map.get(sym, "_UNKNOWN_")
        cur = counts.get(ind, 0)
        if ind == "_UNKNOWN_" or cur < cap:
            picked.append({"symbol": sym, "score": float(r["score"])})
            counts[ind] = cur + 1
        if len(picked) >= top_k:
            break
    return pd.DataFrame(picked).nlargest(top_k, "score")


def _select_topk_with_holding_buffer(
    day_df: pd.DataFrame,
    *,
    top_k: int,
    entry_top_k: int,
    hold_buffer_top_k: int,
    prev_holdings: set[str],
    industry_map: Dict[str, str] | None = None,
    industry_cap_count: int | None = None,
) -> pd.DataFrame:
    hold = day_df[day_df["symbol"].astype(str).isin(prev_holdings)].copy()
    non_hold = day_df[~day_df["symbol"].astype(str).isin(prev_holdings)].copy()

    result = _pick_topk_with_industry_cap(
        hold, top_k=min(hold_buffer_top_k, len(hold)),
        industry_map=industry_map, industry_cap_count=industry_cap_count,
    )
    remain = top_k - len(result)
    if remain > 0 and not non_hold.empty:
        non_picked = _pick_topk_with_industry_cap(
            non_hold, top_k=remain,
            industry_map=industry_map, industry_cap_count=industry_cap_count,
        )
        result = pd.concat([result, non_picked], ignore_index=True)
    return result.nlargest(top_k, "score")


def build_topk_weights(
    score_df: pd.DataFrame,
    factor_df: pd.DataFrame,
    daily_df: pd.DataFrame,
    top_k: int,
    rebalance_rule: str,
    prefilter_cfg: dict,
    max_turnover: float,
    entry_top_k: int | None = None,
    hold_buffer_top_k: int | None = None,
    top_tier_count: int | None = None,
    top_tier_weight_share: float | None = None,
    industry_map: Dict[str, str] | None = None,
    industry_cap_count: int | None = None,
    portfolio_method: str = "equal_weight",
    cov_lookback_days: int = 60,
    cov_ridge: float = 1e-6,
    cov_shrinkage: str = "ledoit_wolf",
    cov_ewma_halflife: float = 20.0,
    risk_aversion: float = 1.0,
    return_details: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    score_df = score_df.copy()
    factor_df = factor_df.copy()
    score_df["trade_date"] = pd.to_datetime(score_df["trade_date"])
    factor_df["trade_date"] = pd.to_datetime(factor_df["trade_date"])
    fac_today = factor_df[
        ["symbol", "trade_date", "turnover_roll_mean", "price_position", "limit_move_hits_5d"]
    ].copy()

    rd_list = _rebalance_dates(score_df["trade_date"].unique(), rebalance_rule)
    rows = []
    diag_rows: list[dict[str, Any]] = []
    prev_holdings: set[str] = set()
    entry_top_k = int(max(1, int(entry_top_k if entry_top_k is not None else top_k)))
    hold_buffer_top_k = int(max(top_k, int(hold_buffer_top_k if hold_buffer_top_k is not None else top_k)))

    pf_enabled = bool(prefilter_cfg.get("enabled", True))
    limit_move_max = int(prefilter_cfg.get("limit_move_max", 2))
    turnover_low_pct = float(prefilter_cfg.get("turnover_low_pct", 0.10))
    turnover_high_pct = float(prefilter_cfg.get("turnover_high_pct", 0.98))
    price_position_high_pct = float(prefilter_cfg.get("price_position_high_pct", 0.90))

    for rd in rd_list:
        day_s = score_df[score_df["trade_date"] == rd].copy()
        if day_s.empty:
            continue
        day_s = day_s.merge(
            fac_today[fac_today["trade_date"] == rd],
            on=["symbol", "trade_date"], how="left",
        )
        filtered = apply_prefilter(
            day_s, top_k,
            enabled=pf_enabled,
            limit_move_max=limit_move_max,
            turnover_low_pct=turnover_low_pct,
            turnover_high_pct=turnover_high_pct,
            price_position_high_pct=price_position_high_pct,
        )

        if hold_buffer_top_k > top_k:
            topk = _select_topk_with_holding_buffer(
                filtered, top_k=top_k, entry_top_k=entry_top_k,
                hold_buffer_top_k=hold_buffer_top_k,
                prev_holdings=prev_holdings,
                industry_map=industry_map, industry_cap_count=industry_cap_count,
            )
        else:
            topk = _pick_topk_with_industry_cap(
                filtered, top_k=top_k,
                industry_map=industry_map, industry_cap_count=industry_cap_count,
            )
        if topk.empty:
            continue

        if prev_holdings and max_turnover < 1.0 and len(topk) >= top_k:
            selected = topk["symbol"].astype(str).tolist()
            selected_set = set(selected)
            required_overlap = int(np.ceil(top_k * (1.0 - max_turnover)))
            overlap = len(selected_set & prev_holdings)
            if overlap < required_overlap:
                need = required_overlap - overlap
                cand_prev = filtered[filtered["symbol"].astype(str).isin(prev_holdings)]
                cand_prev = cand_prev[~cand_prev["symbol"].astype(str).isin(selected_set)]
                cand_prev = cand_prev.nlargest(need, "score")[["symbol", "score"]]
                if not cand_prev.empty:
                    drop_cnt = min(len(cand_prev), len(topk))
                    non_overlap = topk[~topk["symbol"].astype(str).isin(prev_holdings)]
                    if len(non_overlap) >= drop_cnt:
                        to_drop = non_overlap.nsmallest(drop_cnt, "score").index
                    else:
                        to_drop = topk.nsmallest(drop_cnt, "score").index
                    topk = topk.drop(index=to_drop)
                    topk = pd.concat([topk, cand_prev], ignore_index=True)
                    topk = topk.drop_duplicates(subset=["symbol"], keep="first").nlargest(top_k, "score")

        topk = topk.sort_values("score", ascending=False).reset_index(drop=True)
        pm = str(portfolio_method).lower().strip()

        # 权重分配
        if pm in ("", "equal", "equal_weight"):
            ww = np.ones(len(topk), dtype=np.float64) / float(len(topk))
            diag_rows.append({
                "trade_date": pd.Timestamp(rd), "portfolio_method": "equal_weight",
                "n_assets": int(len(topk)), "effective_n": float(len(topk)),
                "weight_std": float(np.std(ww)), "max_weight": float(np.max(ww)),
                "is_equal_like": True, "solver_success": True, "fallback_reason": "",
                "post_constraint_l1_shift": 0.0,
            })
        elif pm in ("tiered_equal_weight", "two_tier_equal_weight"):
            ww, weight_diag = build_portfolio_weights(
                topk, weight_method=pm, score_col="score",
                top_tier_count=top_tier_count, top_tier_weight_share=top_tier_weight_share,
                max_single_weight=1.0, max_industry_weight=None, industry_col=None,
                prev_weights_aligned=None, max_turnover=1.0, return_diagnostics=True,
            )
            final_w_diag = dict(weight_diag.get("post_constraints", {}))
            diag_rows.append({
                "trade_date": pd.Timestamp(rd), "portfolio_method": pm,
                "n_assets": int(len(topk)), "effective_n": final_w_diag.get("effective_n"),
                "weight_std": final_w_diag.get("weight_std"),
                "max_weight": final_w_diag.get("max_weight"),
                "l1_diff_vs_equal": final_w_diag.get("l1_diff_vs_reference"),
                "is_equal_like": final_w_diag.get("is_close_to_reference"),
                "solver_success": True, "fallback_reason": "",
                "post_constraint_l1_shift": weight_diag.get("post_constraint_l1_shift"),
            })
        else:
            syms_topk = topk["symbol"].astype(str).str.zfill(6).tolist()
            mu_arr, cov_mtx = mean_cov_returns_from_daily_long(
                daily_df, syms_topk, asof=rd,
                lookback_days=int(cov_lookback_days),
                ridge=float(cov_ridge),
                shrinkage=str(cov_shrinkage).lower(),
                ewma_halflife=float(cov_ewma_halflife),
            )
            method_map = {"risk_parity": "risk_parity", "min_variance": "min_variance", "mean_variance": "mean_variance"}
            m = method_map.get(pm, "equal")
            exp_ret = mu_arr if m == "mean_variance" else None
            ww, weight_diag = build_portfolio_weights(
                topk, weight_method=m, score_col="score",
                max_single_weight=1.0, max_industry_weight=None, industry_col=None,
                prev_weights_aligned=None, max_turnover=1.0,
                cov_matrix=cov_mtx if m != "equal" else None,
                expected_returns=exp_ret, risk_aversion=float(risk_aversion),
                turnover_cost_model=None, return_diagnostics=True,
            )
            opt_diag = dict(weight_diag.get("optimizer", {}))
            cov_diag = dict(opt_diag.get("covariance", {}))
            final_w_diag = dict(weight_diag.get("post_constraints", {}))
            diag_rows.append({
                "trade_date": pd.Timestamp(rd), "portfolio_method": pm,
                "n_assets": int(len(topk)), "effective_n": final_w_diag.get("effective_n"),
                "weight_std": final_w_diag.get("weight_std"),
                "max_weight": final_w_diag.get("max_weight"),
                "diag_share": cov_diag.get("diag_share"),
                "condition_number": cov_diag.get("condition_number"),
                "l1_diff_vs_equal": final_w_diag.get("l1_diff_vs_reference"),
                "is_equal_like": final_w_diag.get("is_close_to_reference"),
                "solver_success": opt_diag.get("solver_success"),
                "fallback_reason": opt_diag.get("fallback_reason"),
                "post_constraint_l1_shift": weight_diag.get("post_constraint_l1_shift"),
            })

        for i, r in topk.iterrows():
            rows.append({"trade_date": rd, "symbol": str(r["symbol"]).zfill(6), "weight": float(ww[i])})
        prev_holdings = set(topk["symbol"].astype(str).tolist())

    if not rows:
        raise RuntimeError("未生成任何调仓权重")
    w_long = pd.DataFrame(rows)
    w_wide = w_long.pivot(index="trade_date", columns="symbol", values="weight").fillna(0.0)
    w_wide.index = pd.to_datetime(w_wide.index)
    if not return_details:
        return w_wide
    diag_detail = pd.DataFrame(diag_rows).sort_values("trade_date").reset_index(drop=True)

    def _summarize_diag(detail: pd.DataFrame, *, method: str) -> dict[str, Any]:
        if detail.empty:
            return {"portfolio_method": str(method), "n_rebalances": 0}

        def _sm(col: str) -> float | None:
            if col not in detail.columns:
                return None
            vals = pd.to_numeric(detail[col], errors="coerce").dropna()
            return float(vals.mean()) if not vals.empty else None

        def _smed(col: str) -> float | None:
            if col not in detail.columns:
                return None
            vals = pd.to_numeric(detail[col], errors="coerce").dropna()
            return float(vals.median()) if not vals.empty else None

        return {
            "portfolio_method": str(method),
            "n_rebalances": int(len(detail)),
            "mean_weight_std": _sm("weight_std"),
            "median_effective_n": _smed("effective_n"),
            "mean_diag_share": _sm("diag_share"),
            "median_condition_number": _smed("condition_number"),
            "mean_l1_diff_vs_equal": _sm("l1_diff_vs_equal"),
            "equal_like_ratio": (
                float(pd.to_numeric(detail["is_equal_like"], errors="coerce").fillna(0.0).mean())
                if "is_equal_like" in detail.columns else None
            ),
            "solver_success_ratio": (
                float(pd.to_numeric(detail["solver_success"], errors="coerce").fillna(0.0).mean())
                if "solver_success" in detail.columns else None
            ),
        }

    diag_summary = _summarize_diag(diag_detail, method=portfolio_method)
    return w_wide, diag_detail, diag_summary


# ── 基准 & 收益矩阵 ─────────────────────────────────────────────────────

def build_market_ew_open_to_open_benchmark(
    daily_df: pd.DataFrame,
    start: str,
    end: str,
    min_days: int = 500,
) -> pd.Series:
    """构建全市场等权 open-to-open 日收益基准。"""
    df = daily_df[
        (daily_df["trade_date"] >= pd.Timestamp(start))
        & (daily_df["trade_date"] <= pd.Timestamp(end))
        & (pd.to_numeric(daily_df["open"], errors="coerce") > 0)
    ].copy()
    if df.empty:
        return pd.Series(dtype=np.float64)
    df["symbol"] = df["symbol"].astype(str).str.zfill(6)
    sym_cnt = df.groupby("symbol")["trade_date"].count()
    good = sym_cnt[sym_cnt >= min_days].index
    if len(good) == 0:
        good = sym_cnt.index
    open_returns = _build_open_to_open_returns(df[df["symbol"].isin(good)], zero_if_limit_up_open=False)
    cs = open_returns.mean(axis=1, skipna=True).dropna()
    cs.index = pd.to_datetime(cs.index)
    return cs.astype(np.float64)


def build_symbol_benchmark(daily_df: pd.DataFrame, symbol: str, start: str, end: str) -> pd.Series:
    """构建单标的 close-to-close 日收益基准。"""
    sym = str(symbol).zfill(6)
    df = daily_df[
        (daily_df["trade_date"] >= pd.Timestamp(start))
        & (daily_df["trade_date"] <= pd.Timestamp(end))
        & (daily_df["symbol"] == sym)
        & (daily_df["close"] > 0)
    ].copy()
    if df.empty:
        return pd.Series(dtype=np.float64)
    df = df.sort_values("trade_date")
    ret = pd.to_numeric(df["close"], errors="coerce").pct_change()
    ret.index = pd.to_datetime(df["trade_date"])
    return ret.dropna().astype(np.float64)


def build_asset_returns(daily_df: pd.DataFrame, symbols: Iterable[str], start: str, end: str) -> pd.DataFrame:
    """从日线构建标的 close-to-close 收益矩阵。"""
    syms = set(str(s).zfill(6) for s in symbols)
    d = daily_df[
        daily_df["symbol"].isin(syms)
        & (daily_df["trade_date"] >= pd.Timestamp(start))
        & (daily_df["trade_date"] <= pd.Timestamp(end))
        & (daily_df["close"] > 0)
    ].copy()
    d = d.sort_values(["symbol", "trade_date"])
    d["ret"] = d.groupby("symbol")["close"].pct_change()
    d = d.dropna(subset=["ret"])
    returns = d.pivot(index="trade_date", columns="symbol", values="ret").sort_index()
    returns.index = pd.to_datetime(returns.index)
    return returns.fillna(0.0)
