#!/usr/bin/env python3
"""W6: Oracle Gap 分解诊断 — Level 1 单因子条件 IC 分析。

分析目标：
  Level 1（因子层 gap）：
    - 对每个单因子计算条件 IC：P(top20 | 该因子 top quartile)
    - 找出"能预测 oracle top-20"最强的单因子
    - 找出与 oracle top-20 相关性最高的特征组合（稀疏回归）

用法::

    python scripts/run_oracle_gap_analysis.py --db data/market.duckdb --output data/results/oracle_gap_analysis.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Tuple

import duckdb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

_LOG = logging.getLogger(__name__)


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="W6 Oracle Gap Level 1 Analysis")
    p.add_argument("--db", default="data/market.duckdb", help="DuckDB 路径")
    p.add_argument("--output", default="data/results/oracle_gap_analysis.csv", help="输出 CSV 路径")
    p.add_argument("--min-samples", type=int, default=30, help="每个截面最低样本数")
    p.add_argument("--top-k", type=int, default=20, help="Oracle top-K 数量")
    return p.parse_args(argv)


def load_factor_data(db_path: str) -> pd.DataFrame:
    """从 prepared_factors 视图加载因子数据（含 U1 候选池标记和 forward return）。"""
    con = duckdb.connect(db_path, read_only=True)
    try:
        # 尝试从 prepared_factors 视图加载（由 pipeline 维护）
        tables = [r[0] for r in con.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema='main'"
        ).fetchall()]

        if "prepared_factors" in tables:
            df = con.execute("SELECT * FROM prepared_factors").df()
        elif "a_share_daily" in tables:
            # 兜底：从日线表构建基础价量因子
            _LOG.warning("prepared_factors 视图不存在，使用 a_share_daily 构建基础因子")
            df = _build_basic_factors_from_daily(con)
        else:
            _LOG.error("无可用的因子数据表")
            return pd.DataFrame()
    finally:
        con.close()

    if df.empty:
        return df

    # Normalize: prepared_factors uses "signal_date", a_share_daily uses "trade_date"
    if "signal_date" in df.columns and "trade_date" not in df.columns:
        df["trade_date"] = pd.to_datetime(df["signal_date"], errors="coerce").dt.normalize()
    elif "trade_date" in df.columns:
        df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce").dt.normalize()
    df["symbol"] = df["symbol"].astype(str).str.zfill(6)

    return df


def _build_basic_factors_from_daily(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """从日线数据构建基础价量因子（兜底方案）。"""
    raw = con.execute("""
        SELECT symbol, trade_date, close, volume, amount, turnover, pct_chg
        FROM a_share_daily
        ORDER BY symbol, trade_date
    """).df()
    if raw.empty:
        return raw

    raw["trade_date"] = pd.to_datetime(raw["trade_date"]).dt.normalize()
    raw["symbol"] = raw["symbol"].astype(str).str.zfill(6)

    for c in ["close", "volume", "amount", "turnover", "pct_chg"]:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")

    # 计算基础因子
    def _compute_stock_factors(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("trade_date")
        g["ret_5d"] = g["close"].pct_change(5)
        g["ret_20d"] = g["close"].pct_change(20)
        g["ret_60d"] = g["close"].pct_change(60)
        g["realized_vol_20d"] = g["pct_chg"].rolling(20).std()
        g["amount_20d_log"] = np.log(g["amount"].rolling(20).mean() + 1)
        g["turnover_20d"] = g["turnover"].rolling(20).mean()
        g["price_position_250d"] = (g["close"] - g["close"].rolling(250).min()) / (
            g["close"].rolling(250).max() - g["close"].rolling(250).min() + 1e-12
        )
        return g

    raw = raw.groupby("symbol", group_keys=False).apply(_compute_stock_factors)
    return raw


def compute_oracle_topk(
    df: pd.DataFrame,
    *,
    forward_col: str = "ret_20d",
    top_k: int = 20,
) -> pd.DataFrame:
    """计算 Oracle top-K 标签。

    对每个 trade_date 截面，forward_col 最高的 top_k 只股票标记为 oracle_top=1。
    """
    df = df.copy()
    df["_forward"] = pd.to_numeric(df.get(forward_col, pd.Series(dtype=float)), errors="coerce")

    def _rank(x):
        r = x.rank(ascending=False, method="first")
        return (r <= top_k).astype(int)

    df["oracle_top"] = df.groupby("trade_date")["_forward"].transform(_rank)
    return df


def compute_single_factor_conditional_ic(
    df: pd.DataFrame,
    factor_cols: list[str],
    *,
    min_samples: int = 30,
) -> pd.DataFrame:
    """Level 1: 单因子条件 IC 分析。

    对每个因子：
    1. 计算截面 Rank IC（因子 vs oracle_top 标签）
    2. 计算条件概率 P(oracle_top=1 | 因子在 top quartile)
    3. 计算条件概率提升（相对无条件概率）

    Returns
    -------
    pd.DataFrame with columns: factor, rank_ic_mean, rank_ic_std, ic_ir,
                                 cond_prob_top_quartile, prob_lift, n_valid_months
    """
    results = []
    unconditional_prob = df["oracle_top"].mean()

    for col in factor_cols:
        if col not in df.columns:
            continue

        valid = df[[col, "oracle_top", "trade_date"]].dropna()
        if valid.empty:
            continue

        ic_vals = []
        cond_probs = []

        for dt, g in valid.groupby("trade_date"):
            if len(g) < min_samples:
                continue
            # Rank IC
            ic, _ = spearmanr(g[col], g["oracle_top"])
            if np.isfinite(ic):
                ic_vals.append(ic)

            # 条件概率：因子在 top quartile
            threshold = g[col].quantile(0.75)
            top_q = g[g[col] >= threshold]
            if len(top_q) >= 5:
                cond_prob = top_q["oracle_top"].mean()
                cond_probs.append(cond_prob)

        if len(ic_vals) < 3:
            continue

        ic_mean = float(np.mean(ic_vals))
        ic_std = float(np.std(ic_vals, ddof=1))
        ic_ir = ic_mean / ic_std if ic_std > 0 else 0.0
        cond_prob_mean = float(np.mean(cond_probs)) if cond_probs else unconditional_prob
        prob_lift = cond_prob_mean - unconditional_prob

        results.append({
            "factor": col,
            "rank_ic_mean": ic_mean,
            "rank_ic_std": ic_std,
            "ic_ir": ic_ir,
            "cond_prob_top_quartile": cond_prob_mean,
            "unconditional_prob": unconditional_prob,
            "prob_lift": prob_lift,
            "n_valid_months": len(ic_vals),
        })

    return pd.DataFrame(results).sort_values("ic_ir", ascending=False)


def compute_factor_correlation_with_oracle(
    df: pd.DataFrame,
    factor_cols: list[str],
) -> pd.DataFrame:
    """计算各因子与 oracle_top 标签的 Spearman 相关性（截面均值）。

    Returns
    -------
    pd.DataFrame with columns: factor, oracle_corr_mean, oracle_corr_std
    """
    results = []
    for col in factor_cols:
        if col not in df.columns:
            continue
        valid = df[[col, "oracle_top", "trade_date"]].dropna()
        if valid.empty:
            continue

        corrs = []
        for dt, g in valid.groupby("trade_date"):
            if len(g) < 30:
                continue
            c, _ = spearmanr(g[col], g["oracle_top"])
            if np.isfinite(c):
                corrs.append(c)

        if len(corrs) < 3:
            continue

        results.append({
            "factor": col,
            "oracle_corr_mean": float(np.mean(corrs)),
            "oracle_corr_std": float(np.std(corrs, ddof=1)),
            "n_months": len(corrs),
        })

    return pd.DataFrame(results).sort_values("oracle_corr_mean", ascending=False)


def find_top_oracle_predictors(
    df: pd.DataFrame,
    factor_cols: list[str],
    *,
    top_n: int = 10,
    min_samples: int = 30,
) -> pd.DataFrame:
    """综合评估：找出最能预测 oracle top-20 的单因子。

    综合指标 = rank_ic_ir * 0.5 + prob_lift_rank * 0.5
    """
    ic_df = compute_single_factor_conditional_ic(df, factor_cols, min_samples=min_samples)
    if ic_df.empty:
        return ic_df

    ic_df["ic_ir_rank"] = ic_df["ic_ir"].rank(ascending=False)
    ic_df["prob_lift_rank"] = ic_df["prob_lift"].rank(ascending=False)
    ic_df["composite_score"] = (ic_df["ic_ir_rank"] + ic_df["prob_lift_rank"]) / 2
    ic_df = ic_df.sort_values("composite_score")

    return ic_df.head(top_n)


def run_oracle_gap_analysis(
    db_path: str,
    *,
    output_path: str = "data/results/oracle_gap_analysis.csv",
    min_samples: int = 30,
    top_k: int = 20,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """执行完整的 Oracle Gap Level 1 分析。

    Returns
    -------
    (single_factor_ic, oracle_predictors)
    """
    _LOG.info("加载因子数据...")
    df = load_factor_data(db_path)
    if df.empty:
        _LOG.error("因子数据为空，无法执行分析")
        return pd.DataFrame(), pd.DataFrame()

    _LOG.info("计算 Oracle top-%d 标签...", top_k)
    # Auto-detect forward return column: prepared_factors uses label_*, daily uses ret_20d
    if "label_forward_1m_o2o_return" in df.columns:
        forward_col = "label_forward_1m_o2o_return"
    elif "feature_ret_20d" in df.columns:
        forward_col = "feature_ret_20d"
    else:
        forward_col = "ret_20d"
    _LOG.info("Using forward column: %s", forward_col)
    df = compute_oracle_topk(df, top_k=top_k, forward_col=forward_col)

    # 识别因子列（排除非因子列）
    exclude_prefixes = ("symbol", "trade_date", "oracle_", "_", "close", "open",
                        "high", "low", "volume", "amount", "pct_chg", "change",
                        "turnover", "amplitude", "report_period", "announcement",
                        "source", "fetched_at")
    factor_cols = [
        c for c in df.columns
        if not any(c.startswith(p) for p in exclude_prefixes)
        and df[c].dtype in ("float64", "float32", "int64", "int32")
    ]

    _LOG.info("识别到 %d 个候选因子列", len(factor_cols))

    _LOG.info("Level 1: 单因子条件 IC 分析...")
    single_factor_ic = compute_single_factor_conditional_ic(
        df, factor_cols, min_samples=min_samples
    )

    _LOG.info("寻找 Top oracle predictors...")
    oracle_predictors = find_top_oracle_predictors(
        df, factor_cols, min_samples=min_samples
    )

    # 输出
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    single_factor_ic.to_csv(output_path, index=False)
    _LOG.info("结果已写入 %s", output_path)

    # 打印摘要
    print("\n" + "=" * 70)
    print("Oracle Gap Level 1 — Top-10 Oracle Predictors")
    print("=" * 70)
    if not oracle_predictors.empty:
        cols = ["factor", "rank_ic_mean", "ic_ir", "prob_lift", "composite_score"]
        print(oracle_predictors[cols].to_string(index=False))

    print("\n无条件 P(oracle_top) = %.4f" % float(df["oracle_top"].mean()))
    print("分析完成，共 %d 个因子，%d 个月份截面" % (
        len(factor_cols), df["trade_date"].nunique()
    ))
    print("=" * 70)

    return single_factor_ic, oracle_predictors


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    try:
        run_oracle_gap_analysis(
            args.db,
            output_path=args.output,
            min_samples=args.min_samples,
            top_k=args.top_k,
        )
    except Exception as e:
        _LOG.error("分析失败: %s", e)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
