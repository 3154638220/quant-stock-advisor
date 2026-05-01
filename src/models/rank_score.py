"""
截面组合打分：将多路因子（如动量、RSI）标准化后线性加权，供排序与后续 ML 替换。

不依赖 sklearn，便于在无额外依赖环境下跑通；后续可换为学习到的权重或神经网络。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd


def cross_section_zscore(x: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    """
    对一维截面向量做 z-score；``nan`` 不参与均值与标准差，对应位置输出 ``nan``。
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("x 须为一维")
    valid = np.isfinite(x)
    if not valid.any():
        return np.full_like(x, np.nan, dtype=np.float64)
    m = np.nanmean(x)
    s = np.nanstd(x)
    if not np.isfinite(s) or s < eps:
        out = np.full_like(x, np.nan, dtype=np.float64)
        out[valid] = 0.0
        return out
    return (x - m) / s


def composite_linear_score(
    momentum: np.ndarray,
    rsi: np.ndarray,
    *,
    w_momentum: float = 0.65,
    w_rsi: float = 0.35,
    rsi_mode: str = "level",
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    对最近一根 K 线的动量、RSI 做截面 z-score 后加权求和。

    Parameters
    ----------
    momentum, rsi
        同长度一维数组，可含 ``nan``。
    w_momentum, w_rsi
        非负权重；会在内部按和归一化。
    rsi_mode
        ``level``：高 RSI 与高 z(RSI) 同向参与（偏趋势延续语义）；
        ``mean_revert``：使用 ``-(RSI-50)`` 的 z-score（偏均值回归语义）。

    Returns
    -------
    score : ndarray
        组合得分，越大越靠前；全无效时为 ``nan``。
    debug : DataFrame
        含 ``z_momentum``、``z_rsi`` 等便于排查。
    """
    mom = np.asarray(momentum, dtype=np.float64).ravel()
    r = np.asarray(rsi, dtype=np.float64).ravel()
    if mom.shape != r.shape:
        raise ValueError("momentum 与 rsi 长度须一致")

    w_m = max(0.0, float(w_momentum))
    w_r = max(0.0, float(w_rsi))
    s = w_m + w_r
    if s <= 0:
        raise ValueError("权重和须为正")
    w_m, w_r = w_m / s, w_r / s

    z_m = cross_section_zscore(mom)
    mode = str(rsi_mode).lower()
    if mode == "level":
        z_r = cross_section_zscore(r)
    elif mode == "mean_revert":
        z_r = cross_section_zscore(50.0 - r)
    else:
        raise ValueError(f"未知 rsi_mode: {rsi_mode!r}（期望 level | mean_revert）")

    score = w_m * z_m + w_r * z_r
    debug = pd.DataFrame(
        {
            "z_momentum": z_m,
            "z_rsi": z_r,
            "composite_score": score,
        }
    )
    return score, debug


def _normalize_factor_weights(weights: Mapping[str, float]) -> Dict[str, float]:
    """按绝对值之和归一化，支持负权重（反向偏好）。"""
    w = {k: float(v) for k, v in weights.items() if abs(float(v)) > 1e-15}
    s = sum(abs(v) for v in w.values())
    if s <= 0:
        raise ValueError("因子权重和须非零")
    return {k: v / s for k, v in w.items()}


def cross_section_z_columns(
    df: pd.DataFrame,
    raw_names: Sequence[str],
    *,
    rsi_mode: str = "level",
) -> pd.DataFrame:
    """
    在**同一截面**（通常同一 ``trade_date``）上对原始因子列做 z-score，追加 ``z_<name>`` 列。
    """
    out = df.copy()
    for name in raw_names:
        if name not in out.columns:
            raise ValueError(f"缺少因子列: {name!r}")
        raw = pd.to_numeric(out[name], errors="coerce").to_numpy(dtype=np.float64)
        if name == "rsi":
            mode = str(rsi_mode).lower()
            if mode == "level":
                z = cross_section_zscore(raw)
            elif mode == "mean_revert":
                z = cross_section_zscore(50.0 - raw)
            else:
                raise ValueError(f"未知 rsi_mode: {rsi_mode!r}（期望 level | mean_revert）")
        else:
            z = cross_section_zscore(raw)
        out[f"z_{name}"] = z
    return out


def apply_cross_section_z_by_date(
    df: pd.DataFrame,
    *,
    date_col: str = "trade_date",
    raw_names: Sequence[str],
    rsi_mode: str = "level",
) -> pd.DataFrame:
    """按 ``trade_date`` 分组，组内分别做截面 z-score（训练集构造与推理单日截面一致）。"""
    parts: list[pd.DataFrame] = []
    for _, sub in df.groupby(date_col, sort=True):
        parts.append(cross_section_z_columns(sub, raw_names, rsi_mode=rsi_mode))
    return pd.concat(parts, ignore_index=True)


def composite_extended_linear_score(
    df: pd.DataFrame,
    *,
    weights: Mapping[str, float],
    weights_override: Optional[Mapping[str, float]] = None,
    rsi_mode: str = "level",
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    多因子截面 z-score 线性组合：键为 ``df`` 列名，含 ``momentum``、``rsi`` 及阶段一扩展列
    （如 ``atr``、``realized_vol``、``turnover_roll_mean``、``vol_ret_corr``、``short_reversal``），
    以及 P1 扩展列（如 ``pe_ttm``、``roe_ttm``、资金流与股东户数因子）。
    """
    effective_weights = dict(weights)
    if weights_override:
        # 动态权重仅覆盖同名因子；其余回退静态配置，避免单日缺失导致不可用。
        for k, v in weights_override.items():
            if k in effective_weights:
                effective_weights[k] = float(v)
    w_norm = _normalize_factor_weights(effective_weights)
    score = np.zeros(len(df), dtype=np.float64)
    debug_cols: Dict[str, np.ndarray] = {}
    for name, wc in w_norm.items():
        if name not in df.columns:
            raise ValueError(f"composite_extended 缺少列: {name!r}")
        raw = pd.to_numeric(df[name], errors="coerce").to_numpy(dtype=np.float64)
        if name == "rsi":
            mode = str(rsi_mode).lower()
            if mode == "level":
                z = cross_section_zscore(raw)
            elif mode == "mean_revert":
                z = cross_section_zscore(50.0 - raw)
            else:
                raise ValueError(f"未知 rsi_mode: {rsi_mode!r}")
        else:
            z = cross_section_zscore(raw)
        debug_cols[f"z_{name}"] = z
        score = score + wc * z
    dbg = pd.DataFrame(debug_cols)
    dbg["composite_extended_score"] = score
    return score, dbg


def sort_key_for_dataframe(
    df: pd.DataFrame,
    *,
    sort_by: str = "momentum",
    w_momentum: float = 0.65,
    w_rsi: float = 0.35,
    rsi_mode: str = "level",
    composite_extended_weights: Optional[Mapping[str, float]] = None,
    composite_extended_weights_override: Optional[Mapping[str, float]] = None,
    tree_bundle_dir: Optional[Union[str, Path]] = None,
    tree_raw_features: Optional[Sequence[str]] = None,
    tree_rsi_mode: Optional[str] = None,
    deep_sequence_bundle_dir: Optional[Union[str, Path]] = None,
    deep_sequence_long_df: Optional[pd.DataFrame] = None,
    deep_sequence_map_location: str = "cpu",
) -> pd.DataFrame:
    """
    在含 ``momentum``、``rsi`` 列的表上追加 ``composite_score``（若 ``sort_by=='composite'``）
    或仅按 ``sort_by`` 排序。返回新 DataFrame，含 ``rank`` 重算。
    """
    out = df.copy()
    if sort_by == "composite":
        _, dbg = composite_linear_score(
            out["momentum"].to_numpy(),
            out["rsi"].to_numpy(),
            w_momentum=w_momentum,
            w_rsi=w_rsi,
            rsi_mode=rsi_mode,
        )
        out["z_momentum"] = dbg["z_momentum"].to_numpy()
        out["z_rsi"] = dbg["z_rsi"].to_numpy()
        out["composite_score"] = dbg["composite_score"].to_numpy()
        out = out.sort_values("composite_score", ascending=False, na_position="last")
    elif sort_by == "composite_extended":
        cw = composite_extended_weights or {}
        if not cw:
            raise ValueError("sort_by=composite_extended 须在配置中提供 signals.composite_extended 权重")
        _, dbg = composite_extended_linear_score(
            out,
            weights=cw,
            weights_override=composite_extended_weights_override,
            rsi_mode=rsi_mode,
        )
        for c in dbg.columns:
            out[c] = dbg[c].to_numpy()
        out = out.sort_values(
            "composite_extended_score", ascending=False, na_position="last"
        )
    elif sort_by == "xgboost":
        from src.models.inference import predict_xgboost_tree

        if not tree_bundle_dir:
            raise ValueError("sort_by=xgboost 须在配置中设置 signals.tree_model.bundle_dir")
        raw = list(tree_raw_features or [])
        if not raw:
            raise ValueError("sort_by=xgboost 须在 signals.tree_model.features 中指定因子列")
        trsi = str(tree_rsi_mode or rsi_mode).lower()
        if trsi not in ("level", "mean_revert"):
            trsi = "level"
        dbg = cross_section_z_columns(out, raw, rsi_mode=trsi)
        for c in dbg.columns:
            if c.startswith("z_"):
                out[c] = dbg[c].to_numpy()
        out["tree_score"] = predict_xgboost_tree(tree_bundle_dir, out)
        out = out.sort_values("tree_score", ascending=False, na_position="last")
    elif sort_by == "deep_sequence":
        from src.models.inference import predict_timeseries_bundle_last

        if not deep_sequence_bundle_dir:
            raise ValueError("sort_by=deep_sequence 须在配置中设置 signals.deep_sequence.bundle_dir")
        if deep_sequence_long_df is None or deep_sequence_long_df.empty:
            raise ValueError("sort_by=deep_sequence 需要 OHLCV 长表 deep_sequence_long_df")
        pred = predict_timeseries_bundle_last(
            deep_sequence_bundle_dir,
            deep_sequence_long_df,
            map_location=deep_sequence_map_location,
        )
        if pred.empty:
            raise ValueError("deep_sequence 推理无有效预测（检查 seq_len 与数据）")
        pred = pred.rename(columns={"pred": "deep_sequence_score"})
        out = out.merge(
            pred[["symbol", "deep_sequence_score"]],
            on="symbol",
            how="left",
        )
        out = out.sort_values("deep_sequence_score", ascending=False, na_position="last")
    elif sort_by == "momentum":
        out["z_momentum"] = cross_section_zscore(
            pd.to_numeric(out["momentum"], errors="coerce").to_numpy(dtype=np.float64)
        )
        out = out.sort_values("momentum", ascending=False, na_position="last")
    elif sort_by == "rsi":
        r_raw = pd.to_numeric(out["rsi"], errors="coerce").to_numpy(dtype=np.float64)
        mode = str(rsi_mode).lower()
        if mode == "level":
            out["z_rsi"] = cross_section_zscore(r_raw)
        elif mode == "mean_revert":
            out["z_rsi"] = cross_section_zscore(50.0 - r_raw)
        else:
            out["z_rsi"] = cross_section_zscore(r_raw)
        out = out.sort_values("rsi", ascending=False, na_position="last")
    else:
        raise ValueError(
            f"未知 sort_by: {sort_by!r}（期望 momentum | rsi | composite | composite_extended | xgboost | deep_sequence）"
        )

    out = out.reset_index(drop=True)
    out.insert(0, "rank", range(1, len(out) + 1))
    return out
