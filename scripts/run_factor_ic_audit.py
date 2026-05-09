#!/usr/bin/env python3
"""W5: 因子 IC 衰减审计 & 因子正交化治理。

分析目标：
  1. 输出每个特征的 IC 时序、IC IR、IC 衰减曲线（lag 0/5/10/20 交易日）
  2. 输出特征间 Spearman 相关矩阵（截面均值）
  3. 标记 IC IR < 0.2 的弱因子（候选降级）
  4. 标记两两相关 > 0.7 的冗余对（候选剔除）
  5. 可选：PCA 正交化降维保留 95% 方差，对比降维前后 Rank IC

用法::

    python scripts/run_factor_ic_audit.py --db data/market.duckdb --output-dir data/results/factor_audit/
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import duckdb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

_LOG = logging.getLogger(__name__)

# ── 参数解析 ──────────────────────────────────────────────────────────────

def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="W5 Factor IC Decay Audit")
    p.add_argument("--db", default="data/market.duckdb", help="DuckDB 路径")
    p.add_argument("--output-dir", default="data/results/factor_audit", help="输出目录")
    p.add_argument("--min-samples", type=int, default=30, help="每个截面最低样本数")
    p.add_argument("--decay-lags", nargs="+", type=int, default=[0, 5, 10, 20],
                   help="IC 衰减测试的 lag 交易日数")
    p.add_argument("--weak-ic-threshold", type=float, default=0.2,
                   help="IC IR 低于此值的因子标记为弱因子")
    p.add_argument("--redundant-cor-threshold", type=float, default=0.7,
                   help="两两相关高于此值的因子对标记为冗余")
    p.add_argument("--no-pca", action="store_true", help="跳过 PCA 正交化对比")
    p.add_argument("--pca-variance", type=float, default=0.95, help="PCA 保留方差比例")
    return p.parse_args(argv)


# ── 数据加载 ──────────────────────────────────────────────────────────────

def load_factor_data(db_path: str, forward_col: str = "ret_20d") -> pd.DataFrame:
    """从 prepared_factors 视图或日线表加载因子数据。"""
    con = duckdb.connect(db_path, read_only=True)
    try:
        tables = [r[0] for r in con.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema='main'"
        ).fetchall()]

        if "prepared_factors" in tables:
            df = con.execute("SELECT * FROM prepared_factors").df()
        elif "a_share_daily" in tables:
            _LOG.warning("prepared_factors 视图不存在，从 a_share_daily 构建基础因子")
            df = _build_basic_factors(con)
        else:
            _LOG.error("无可用的因子数据表")
            return pd.DataFrame()
    finally:
        con.close()

    if df.empty:
        return df

    # Normalize date column: prepared_factors uses "signal_date", a_share_daily uses "trade_date"
    if "signal_date" in df.columns and "trade_date" not in df.columns:
        df["trade_date"] = pd.to_datetime(df["signal_date"], errors="coerce").dt.normalize()
    elif "trade_date" in df.columns:
        df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce").dt.normalize()
    df["symbol"] = df["symbol"].astype(str).str.zfill(6)
    df = df.sort_values(["symbol", "trade_date"]).reset_index(drop=True)
    return df


def _build_basic_factors(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    raw = con.execute("""
        SELECT symbol, trade_date, close, volume, amount, turnover, pct_chg
        FROM a_share_daily ORDER BY symbol, trade_date
    """).df()
    if raw.empty:
        return raw

    raw["trade_date"] = pd.to_datetime(raw["trade_date"]).dt.normalize()
    raw["symbol"] = raw["symbol"].astype(str).str.zfill(6)
    for c in ["close", "volume", "amount", "turnover", "pct_chg"]:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")

    def _compute(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("trade_date")
        g["feature_ret_5d"] = g["close"].pct_change(5)
        g["feature_ret_20d"] = g["close"].pct_change(20)
        g["feature_ret_60d"] = g["close"].pct_change(60)
        g["feature_realized_vol_20d"] = g["pct_chg"].rolling(20, min_periods=10).std()
        g["feature_amount_20d_log"] = np.log(g["amount"].rolling(20, min_periods=10).mean() + 1)
        g["feature_turnover_20d"] = g["turnover"].rolling(20, min_periods=10).mean()
        g["feature_price_position_250d"] = (g["close"] - g["close"].rolling(250, min_periods=60).min()) / (
            g["close"].rolling(250, min_periods=60).max() - g["close"].rolling(250, min_periods=60).min() + 1e-12
        )
        return g

    return raw.groupby("symbol", group_keys=False).apply(_compute)


# ── 因子列识别 ────────────────────────────────────────────────────────────

def identify_factor_cols(df: pd.DataFrame) -> list[str]:
    exclude = {"symbol", "trade_date", "close", "open", "high", "low",
               "volume", "amount", "turnover", "pct_chg", "change",
               "amplitude", "amplitude_pct", "report_period", "announcement_date",
               "source", "fetched_at", "history_days", "market_cap", "log_market_cap",
               "next_trade_date", "signal_date"}
    exclude_prefixes = ("_", "oracle_", "label_")
    cols = [
        c for c in df.columns
        if c not in exclude
        and not any(c.startswith(p) for p in exclude_prefixes)
        and df[c].dtype in ("float64", "float32", "int64", "int32")
    ]
    # label 列不应参与因子审计，否则 spearmanr 可能返回相关矩阵。
    return [c for c in cols if c != "feature_ret_20d"]


def _extract_spearman_scalar(x: pd.Series, y: pd.Series) -> float | None:
    corr, _ = spearmanr(x, y)
    # scipy 在输入退化或同列时可能返回 2x2 矩阵
    if isinstance(corr, np.ndarray):
        if corr.size >= 4:
            corr = corr[0, 1]
        elif corr.size == 1:
            corr = corr.item()
        else:
            return None
    try:
        corr_f = float(corr)
    except Exception:
        return None
    if not np.isfinite(corr_f):
        return None
    return corr_f


# ── 1. IC 时间序列 & IR ──────────────────────────────────────────────────

def compute_factor_ic_ts(
    df: pd.DataFrame,
    factor_cols: list[str],
    *,
    label_col: str = "feature_ret_20d",
    min_samples: int = 30,
) -> pd.DataFrame:
    """计算每个因子的月度 IC 时间序列和 IC IR。

    Returns
    -------
    pd.DataFrame with columns: factor, ic_mean, ic_std, ic_ir,
                                 pos_ratio, n_months
    """
    results = []
    for col in factor_cols:
        if col not in df.columns:
            continue
        valid = df[[col, label_col, "trade_date"]].dropna()
        if valid.empty:
            continue

        ic_vals = []
        for dt, g in valid.groupby("trade_date"):
            if len(g) < min_samples:
                continue
            ic = _extract_spearman_scalar(g[col], g[label_col])
            if ic is not None:
                ic_vals.append(ic)

        if len(ic_vals) < 3:
            continue

        ic_arr = np.array(ic_vals)
        results.append({
            "factor": col,
            "ic_mean": float(np.mean(ic_arr)),
            "ic_std": float(np.std(ic_arr, ddof=1)),
            "ic_ir": float(np.mean(ic_arr) / np.std(ic_arr, ddof=1)) if np.std(ic_arr, ddof=1) > 0 else 0.0,
            "pos_ratio": float((ic_arr > 0).mean()),
            "n_months": len(ic_vals),
        })

    return pd.DataFrame(results).sort_values("ic_ir", ascending=False)


# ── 2. IC 衰减曲线 ───────────────────────────────────────────────────────

def compute_ic_decay_curve(
    df: pd.DataFrame,
    factor_cols: list[str],
    *,
    forward_col: str = "feature_ret_20d",
    decay_lags: list[int] = (0, 5, 10, 20),
    min_samples: int = 30,
) -> pd.DataFrame:
    """计算各因子在不同 lag 下的 IC 衰减曲线。

    对每个 lag，用因子值预测 lag 个交易日后的 forward return，
    计算 Rank IC 均值。

    Returns
    -------
    pd.DataFrame with columns: factor, ic_lag0, ic_lag5, ..., ic_decay_pct
    """
    df = df.sort_values(["symbol", "trade_date"]).reset_index(drop=True)
    if forward_col not in df.columns:
        return pd.DataFrame(columns=["factor"])
    results = []

    for col in factor_cols:
        if col not in df.columns:
            continue
        row = {"factor": col}
        decay_ics = []

        for lag in decay_lags:
            valid = df[[col, forward_col, "trade_date", "symbol"]].dropna(subset=[col]).copy()
            # Build shifted forward return: for each symbol, shift forward_col by -lag
            valid["_forward"] = valid.groupby("symbol")[forward_col].shift(-lag)
            valid = valid.dropna(subset=["_forward"])

            ic_vals = []
            for dt, g in valid.groupby("trade_date"):
                if len(g) < min_samples:
                    continue
                ic = _extract_spearman_scalar(g[col], g["_forward"])
                if ic is not None:
                    ic_vals.append(ic)

            ic_label = f"ic_lag{lag}"
            if len(ic_vals) >= 3:
                ic_mean = float(np.mean(ic_vals))
                row[ic_label] = ic_mean
                decay_ics.append(ic_mean)
            else:
                row[ic_label] = np.nan

        # 衰减比例 = (lag_max - lag_0) / lag_0，负值表示衰减
        if decay_ics and decay_ics[0] != 0:
            row["ic_decay_pct"] = float((decay_ics[-1] - decay_ics[0]) / abs(decay_ics[0]) * 100)
        else:
            row["ic_decay_pct"] = np.nan

        results.append(row)

    return pd.DataFrame(results).sort_values("ic_lag0", ascending=False, na_position="last")


# ── 3. 特征间 Spearman 相关矩阵 ──────────────────────────────────────────

def compute_factor_correlation_matrix(
    df: pd.DataFrame,
    factor_cols: list[str],
    *,
    min_samples: int = 30,
) -> pd.DataFrame:
    """计算特征间截面 Spearman 相关的均值矩阵。

    对每个截面日期计算因子间 Spearman 相关，取时间序列均值。

    Returns
    -------
    pd.DataFrame: 因子 × 因子 的相关矩阵
    """
    available = [c for c in factor_cols if c in df.columns]
    if len(available) < 2:
        return pd.DataFrame()

    # 累积相关矩阵
    corr_sum = np.zeros((len(available), len(available)))
    n_dates = 0

    for dt, g in df.groupby("trade_date"):
        sub = g[available].dropna()
        if len(sub) < min_samples:
            continue
        corr = sub.corr(method="spearman").values
        corr_sum += np.nan_to_num(corr, nan=0.0)
        n_dates += 1

    if n_dates == 0:
        return pd.DataFrame(index=available, columns=available)

    corr_avg = corr_sum / n_dates
    return pd.DataFrame(corr_avg, index=available, columns=available)


# ── 4. 弱因子标记 ────────────────────────────────────────────────────────

def flag_weak_factors(ic_summary: pd.DataFrame, *, threshold: float = 0.2) -> pd.DataFrame:
    """标记 IC IR 低于阈值的弱因子。"""
    out = ic_summary.copy()
    out["is_weak"] = out["ic_ir"].abs() < threshold
    out["recommendation"] = out["is_weak"].apply(
        lambda w: "DEMOTE (IC IR < %.1f)" % threshold if w else "KEEP"
    )
    return out


# ── 5. 冗余因子对标记 ────────────────────────────────────────────────────

def flag_redundant_pairs(
    corr_matrix: pd.DataFrame,
    *,
    threshold: float = 0.7,
) -> pd.DataFrame:
    """标记两两相关高于阈值的冗余因子对。

    Returns
    -------
    pd.DataFrame with columns: factor_a, factor_b, corr, recommended_drop
    """
    factors = list(corr_matrix.columns)
    pairs = []
    seen = set()

    for i, fa in enumerate(factors):
        for j, fb in enumerate(factors):
            if i >= j:
                continue
            corr_val = corr_matrix.iloc[i, j]
            if pd.isna(corr_val):
                continue
            if abs(corr_val) >= threshold:
                pair_key = tuple(sorted([fa, fb]))
                if pair_key in seen:
                    continue
                seen.add(pair_key)
                # 推荐保留 IC IR 更高的因子
                pairs.append({
                    "factor_a": fa,
                    "factor_b": fb,
                    "corr": round(corr_val, 4),
                })

    if not pairs:
        return pd.DataFrame(columns=["factor_a", "factor_b", "corr"])
    return pd.DataFrame(pairs).sort_values("corr", ascending=False, key=abs)


# ── 6. PCA 正交化对比 ────────────────────────────────────────────────────

def pca_orthogonalize_and_compare(
    df: pd.DataFrame,
    factor_cols: list[str],
    *,
    variance_ratio: float = 0.95,
    label_col: str = "feature_ret_20d",
    min_samples: int = 30,
) -> dict:
    """PCA 正交化降维，对比降维前后的 Rank IC。

    Returns
    -------
    dict with keys: n_original, n_pca_components, ic_original, ic_pca
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    available = [c for c in factor_cols if c in df.columns]
    if len(available) < 3:
        return {"n_original": len(available), "error": "too few factors"}

    # 填充缺失值为截面中位数
    df_filled = df[available + ["trade_date", "symbol", label_col]].copy()
    for col in available:
        df_filled[col] = df_filled.groupby("trade_date")[col].transform(
            lambda s: s.fillna(s.median())
        )

    df_filled = df_filled.dropna(subset=available + [label_col])
    if df_filled.empty:
        return {"n_original": len(available), "error": "no valid rows after fillna"}

    # Original IC
    orig_ic = _mean_rank_ic(df_filled, available[0], label_col, min_samples)
    # Average IC across all factors
    orig_ics = []
    for col in available:
        ic = _mean_rank_ic(df_filled, col, label_col, min_samples)
        if ic is not None:
            orig_ics.append(abs(ic))

    # PCA
    X_all = df_filled[available].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)

    pca = PCA(n_components=variance_ratio)
    X_pca = pca.fit_transform(X_scaled)

    # PCA component IC: compute IC of each PC vs label
    pca_ics = []
    for pc_idx in range(X_pca.shape[1]):
        pc_col = f"_pc{pc_idx}"
        df_filled[pc_col] = X_pca[:, pc_idx]
        ic = _mean_rank_ic(df_filled, pc_col, label_col, min_samples)
        if ic is not None:
            pca_ics.append(abs(ic))

    return {
        "n_original": len(available),
        "n_pca_components": X_pca.shape[1],
        "variance_explained": float(pca.explained_variance_ratio_.sum()),
        "ic_original_mean": float(np.mean(orig_ics)) if orig_ics else np.nan,
        "ic_original_max": float(np.max(orig_ics)) if orig_ics else np.nan,
        "ic_pca_mean": float(np.mean(pca_ics)) if pca_ics else np.nan,
        "ic_pca_max": float(np.max(pca_ics)) if pca_ics else np.nan,
    }


def _mean_rank_ic(df, col, label_col, min_samples):
    ics = []
    for dt, g in df.groupby("trade_date"):
        if len(g) < min_samples:
            continue
        valid = g[[col, label_col]].dropna()
        if len(valid) < min_samples:
            continue
        ic = _extract_spearman_scalar(valid[col], valid[label_col])
        if ic is not None:
            ics.append(ic)
    return float(np.mean(ics)) if ics else None


# ── 主流程 ────────────────────────────────────────────────────────────────

def run_factor_ic_audit(
    db_path: str,
    *,
    output_dir: str = "data/results/factor_audit",
    min_samples: int = 30,
    decay_lags: list[int] = (0, 5, 10, 20),
    weak_ic_threshold: float = 0.2,
    redundant_cor_threshold: float = 0.7,
    run_pca: bool = True,
    pca_variance: float = 0.95,
) -> dict:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    _LOG.info("加载因子数据...")
    df = load_factor_data(db_path)
    if df.empty:
        _LOG.error("因子数据为空")
        return {}

    factor_cols = identify_factor_cols(df)
    _LOG.info("识别到 %d 个因子列", len(factor_cols))

    # 1. IC 时间序列 & IR
    _LOG.info("Step 1: IC 时间序列 & IR...")
    ic_summary = compute_factor_ic_ts(df, factor_cols, min_samples=min_samples)
    ic_summary.to_csv(out / "ic_summary.csv", index=False)

    # 2. IC 衰减曲线
    _LOG.info("Step 2: IC 衰减曲线 (lags=%s)...", decay_lags)
    ic_decay = compute_ic_decay_curve(df, factor_cols, decay_lags=decay_lags, min_samples=min_samples)
    ic_decay.to_csv(out / "ic_decay.csv", index=False)

    # 3. 特征间相关矩阵
    _LOG.info("Step 3: 特征间 Spearman 相关矩阵...")
    corr_matrix = compute_factor_correlation_matrix(df, factor_cols, min_samples=min_samples)
    corr_matrix.to_csv(out / "factor_correlation_matrix.csv")

    # 4. 弱因子标记
    _LOG.info("Step 4: 弱因子标记 (threshold=%.1f)...", weak_ic_threshold)
    weak_factors = flag_weak_factors(ic_summary, threshold=weak_ic_threshold)
    weak_factors.to_csv(out / "weak_factors.csv", index=False)

    # 5. 冗余因子对
    _LOG.info("Step 5: 冗余因子对 (threshold=%.1f)...", redundant_cor_threshold)
    redundant_pairs = flag_redundant_pairs(corr_matrix, threshold=redundant_cor_threshold)
    redundant_pairs.to_csv(out / "redundant_pairs.csv", index=False)

    # 6. PCA 正交化对比
    pca_result = {}
    if run_pca:
        _LOG.info("Step 6: PCA 正交化对比 (variance=%.2f)...", pca_variance)
        try:
            pca_result = pca_orthogonalize_and_compare(
                df, factor_cols, variance_ratio=pca_variance, min_samples=min_samples
            )
            pd.DataFrame([pca_result]).to_csv(out / "pca_comparison.csv", index=False)
        except Exception as e:
            _LOG.warning("PCA 分析失败: %s", e)
            pca_result = {"error": str(e)}

    # ── 打印摘要 ──
    print("\n" + "=" * 70)
    print("W5 因子 IC 衰减审计 — 摘要")
    print("=" * 70)

    print(f"\n总因子数: {len(factor_cols)}")
    print(f"弱因子 (IC IR < {weak_ic_threshold}): {weak_factors['is_weak'].sum() if not weak_factors.empty else 0}")
    print(f"冗余因子对 (corr > {redundant_cor_threshold}): {len(redundant_pairs)}")

    print("\n--- Top-10 最强因子 (IC IR) ---")
    if not ic_summary.empty:
        cols = ["factor", "ic_mean", "ic_ir", "pos_ratio", "n_months"]
        print(ic_summary[cols].head(10).to_string(index=False))

    print("\n--- IC 衰减 Top-5 (lag0 最强) ---")
    if not ic_decay.empty:
        dec_cols = ["factor"] + [f"ic_lag{l}" for l in decay_lags] + ["ic_decay_pct"]
        print(ic_decay[dec_cols].head(5).to_string(index=False))

    if run_pca and "error" not in pca_result:
        print("\n--- PCA 正交化对比 ---")
        print(f"  原始因子: {pca_result.get('n_original')} → PCA 成分: {pca_result.get('n_pca_components')}")
        print(f"  方差保留: {pca_result.get('variance_explained', 0):.1%}")
        print(f"  |IC| 均值: {pca_result.get('ic_original_mean', np.nan):.4f} → {pca_result.get('ic_pca_mean', np.nan):.4f}")
        print(f"  |IC| 最大: {pca_result.get('ic_original_max', np.nan):.4f} → {pca_result.get('ic_pca_max', np.nan):.4f}")

    print(f"\n输出目录: {out.resolve()}")
    print("=" * 70)

    return {
        "ic_summary": ic_summary,
        "ic_decay": ic_decay,
        "corr_matrix": corr_matrix,
        "weak_factors": weak_factors,
        "redundant_pairs": redundant_pairs,
        "pca_result": pca_result,
    }


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    try:
        run_factor_ic_audit(
            args.db,
            output_dir=args.output_dir,
            min_samples=args.min_samples,
            decay_lags=list(args.decay_lags),
            weak_ic_threshold=args.weak_ic_threshold,
            redundant_cor_threshold=args.redundant_cor_threshold,
            run_pca=not args.no_pca,
            pca_variance=args.pca_variance,
        )
    except Exception as e:
        _LOG.error("审计失败: %s", e, exc_info=True)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
