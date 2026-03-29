"""从工件目录加载模型并按 ``inference_config.json`` 推理（与训练脚本解耦）。"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import torch

from src.models.artifacts import load_normalizer_stats
from src.models.baseline.train import load_baseline_bundle
from src.models.timeseries.ohlcv_norm import normalize_ohlcv_anchor
from src.models.timeseries.train import load_timeseries_bundle


def predict_baseline_bundle(
    bundle_dir: Union[str, Path],
    df: pd.DataFrame,
) -> np.ndarray:
    """对行样本逐行回归预测（与训练时 ``feature_columns`` 对齐）。"""
    root = Path(bundle_dir)
    model, inf, _ = load_baseline_bundle(root)
    X = df[inf.feature_columns].to_numpy(dtype=np.float64)
    norm_path = root / "normalizer.json"
    if norm_path.exists():
        mean, scale = load_normalizer_stats(norm_path)
        X = (X - mean) / scale
    pred = model.predict(X)
    return np.asarray(pred, dtype=np.float64)


def predict_xgboost_tree(
    bundle_dir: Union[str, Path],
    df: pd.DataFrame,
) -> np.ndarray:
    """
    截面 XGBoost 预测：``df`` 须已含 ``inference_config.json`` 中的 ``z_*`` 特征列
    （与 ``rank_score.cross_section_z_columns`` 一致）。
    """
    root = Path(bundle_dir)
    model, inf, _ = load_baseline_bundle(root)
    missing = [c for c in inf.feature_columns if c not in df.columns]
    if missing:
        raise ValueError(f"推理缺少列（请先截面 z-score）: {missing[:8]}")
    X = df[inf.feature_columns].to_numpy(dtype=np.float64)
    pred = model.predict(X)
    return np.asarray(pred, dtype=np.float64)


def predict_timeseries_bundle_last(
    bundle_dir: Union[str, Path],
    df: pd.DataFrame,
    *,
    map_location: str = "cpu",
) -> pd.DataFrame:
    """
    对每个 ``symbol`` 使用最后 ``seq_len`` 根 K 线窗口预测下一目标（与训练时序列构造一致）。
    返回列：``symbol``, ``trade_date``（窗口末日）, ``pred``。
    """
    root = Path(bundle_dir)
    model, inf, _, extra = load_timeseries_bundle(root, map_location=map_location)
    seq_len = int(inf.seq_len or extra.get("seq_len", 0))
    if seq_len <= 0:
        raise ValueError("bundle 缺少 seq_len")

    device = map_location if map_location else "cpu"
    model.eval()
    model.to(device)

    sub = df.copy()
    sub[inf.date_col] = pd.to_datetime(sub[inf.date_col]).dt.normalize()
    sub = sub.sort_values([inf.symbol_col, inf.date_col])
    rows = []
    for sym, g in sub.groupby(inf.symbol_col, sort=False):
        g = g.reset_index(drop=True)
        if len(g) < seq_len:
            continue
        mat = g[inf.feature_columns].to_numpy(dtype=np.float64)
        t = len(g) - 1
        window = mat[t - seq_len + 1 : t + 1]
        if not np.isfinite(window).all():
            continue
        nm = str(extra.get("normalize_mode") or inf.normalize or "none").lower()
        if nm == "ohlcv_anchor":
            window = normalize_ohlcv_anchor(window)
        x = torch.from_numpy(window[None, ...]).float().to(device)
        with torch.no_grad():
            logits = float(model(x).cpu().numpy().ravel()[0])
        tt = str((extra.get("target_task") or inf.extra.get("target_task") or "regression")).lower()
        if tt == "binary_up":
            p = float(1.0 / (1.0 + np.exp(-np.clip(logits, -50.0, 50.0))))
        else:
            p = logits
        rows.append(
            {
                inf.symbol_col: sym,
                inf.date_col: g[inf.date_col].iloc[t],
                "pred": p,
            }
        )
    return pd.DataFrame(rows)
