"""
XGBoost 截面学习：特征为逐日截面 Z-score 后的因子；标签为前瞻收益。

默认使用 ``XGBRanker``（pairwise 排序），与荐股 Top-K 目标一致；可选 ``XGBRegressor`` 回归。

工件格式与 ``src.models.baseline.train`` 对齐：``bundle.json``、``inference_config.json``、``model.joblib``。
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from src.models.artifacts import (
    BundleMetadata,
    InferenceConfig,
    save_bundle_metadata,
    save_inference_config,
    save_sklearn_model,
    utc_now_iso,
)
from src.models.data_slice import combined_data_fingerprint, hash_slice_spec, normalize_slice_spec, seed_everything
from src.models.experiment import append_experiment_csv, append_experiment_jsonl, build_experiment_record
from src.models.rank_score import apply_cross_section_z_by_date

try:
    from xgboost import XGBRanker, XGBRegressor
except ImportError:  # pragma: no cover
    XGBRanker = None  # type: ignore
    XGBRegressor = None  # type: ignore


@dataclass
class XGBoostTrainResult:
    bundle_dir: Path
    metrics: Dict[str, float]
    data_slice_hash: str
    content_hash: str


def _require_xgb():
    if XGBRegressor is None or XGBRanker is None:
        raise RuntimeError("需要安装 xgboost：pip install xgboost")


def _integer_relevance_labels(y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """
    XGBoost 2.x 的 ``rank:pairwise`` 在校验集上要求标签为非负整数相关度；
    组内按前瞻收益降序赋 dense rank（收益越高，整数越大）。
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    groups = np.asarray(groups, dtype=np.int32).ravel()
    out = np.zeros(len(y), dtype=np.int32)
    pos = 0
    for g in groups:
        g = int(g)
        if g <= 0:
            continue
        sl = y[pos : pos + g]
        rk = pd.Series(sl).rank(method="dense", ascending=False).astype(np.int32).to_numpy()
        out[pos : pos + g] = rk
        pos += g
    return out


def _mean_rank_ic_per_group(pred: np.ndarray, y: np.ndarray, groups: np.ndarray) -> float:
    """按组（交易日）计算 Spearman 秩相关的截面均值。"""
    pos = 0
    ics: List[float] = []
    for g in groups:
        g = int(g)
        if g < 2:
            pos += g
            continue
        p = pred[pos : pos + g]
        yy = y[pos : pos + g]
        pos += g
        rp = pd.Series(p).rank(method="average").to_numpy(dtype=np.float64)
        ry = pd.Series(yy).rank(method="average").to_numpy(dtype=np.float64)
        if np.std(rp) < 1e-15 or np.std(ry) < 1e-15:
            continue
        ic = float(np.corrcoef(rp, ry)[0, 1])
        if np.isfinite(ic):
            ics.append(ic)
    return float(np.mean(ics)) if ics else float("nan")


def _prepare_xy_z(
    df: pd.DataFrame,
    *,
    raw_feature_names: Sequence[str],
    target_column: str,
    rsi_mode: str,
    date_col: str = "trade_date",
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """按日截面 z 后拼接 X、y；丢弃标签无效行。"""
    zdf = apply_cross_section_z_by_date(
        df,
        date_col=date_col,
        raw_names=raw_feature_names,
        rsi_mode=rsi_mode,
    )
    z_cols = [f"z_{n}" for n in raw_feature_names]
    X = zdf[z_cols].to_numpy(dtype=np.float64)
    y = zdf[target_column].to_numpy(dtype=np.float64)
    m = np.isfinite(y)
    for j in range(X.shape[1]):
        m &= np.isfinite(X[:, j])
    return X[m], y[m], zdf


def _prepare_xy_z_grouped(
    df: pd.DataFrame,
    *,
    raw_feature_names: Sequence[str],
    target_column: str,
    rsi_mode: str,
    date_col: str = "trade_date",
    min_group_size: int = 2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """
    截面 z 后按 ``date_col`` 排序；``y`` 为相关性标签（通常为前瞻收益，越大越好）。
    ``groups`` 为与 ``X`` 行顺序一致的各组长度（每个交易日一组）。
    """
    zdf = apply_cross_section_z_by_date(
        df,
        date_col=date_col,
        raw_names=raw_feature_names,
        rsi_mode=rsi_mode,
    )
    z_cols = [f"z_{n}" for n in raw_feature_names]
    zdf = zdf.sort_values(date_col, kind="mergesort")
    X = zdf[z_cols].to_numpy(dtype=np.float64)
    y = zdf[target_column].to_numpy(dtype=np.float64)
    m = np.isfinite(y)
    for j in range(X.shape[1]):
        m &= np.isfinite(X[:, j])
    zdf = zdf.loc[zdf.index[m]].copy()
    X = zdf[z_cols].to_numpy(dtype=np.float64)
    y = zdf[target_column].to_numpy(dtype=np.float64)
    sizes = zdf.groupby(date_col, sort=False).size().to_numpy(dtype=np.int32)
    if min_group_size > 1:
        keep = zdf.groupby(date_col, sort=False)[target_column].transform("count") >= min_group_size
        zdf = zdf.loc[keep].copy()
        X = zdf[z_cols].to_numpy(dtype=np.float64)
        y = zdf[target_column].to_numpy(dtype=np.float64)
        sizes = zdf.groupby(date_col, sort=False).size().to_numpy(dtype=np.int32)
    return X, y, sizes, zdf


def train_xgboost_panel(
    df: pd.DataFrame,
    *,
    raw_feature_names: Sequence[str],
    target_column: str,
    rsi_mode: str = "level",
    xgboost_objective: Literal["rank", "regression"] = "rank",
    model_version: str = "1.0.0",
    feature_version: str = "tree_z_v1",
    training_seed: int = 42,
    val_frac: float = 0.2,
    date_col: str = "trade_date",
    slice_spec: Optional[dict] = None,
    xgb_params: Optional[Dict[str, Any]] = None,
    out_root: Union[str, Path] = "data/models",
    log_experiments: bool = True,
    experiments_dir: Union[str, Path] = "data/experiments",
) -> XGBoostTrainResult:
    """
    在含 ``trade_date``、原始因子列与 ``target_column`` 的面板上训练 XGBoost。

    - ``xgboost_objective='rank'``：``XGBRanker``（pairwise），直接优化截面相对排序。
    - ``xgboost_objective='regression'``：``XGBRegressor``（MSE 前瞻收益）。

    验证集为时间切分（后 ``val_frac`` 比例的交易日），避免随机切分泄漏。
    """
    _require_xgb()
    t0 = time.perf_counter()
    seed_everything(training_seed)
    xgb_params = dict(xgb_params or {})
    use_rank = str(xgboost_objective).lower() == "rank"

    if df.empty or len(df) < 20:
        raise ValueError("训练样本过少")

    spec = slice_spec or normalize_slice_spec()
    data_slice_hash = hash_slice_spec(spec)
    z_cols = [f"z_{n}" for n in raw_feature_names]
    fp = combined_data_fingerprint(
        df,
        slice_spec=spec,
        content_columns=list(raw_feature_names) + [target_column, date_col],
    )
    content_hash = fp["content_hash"]

    dates = sorted(pd.to_datetime(df[date_col]).dt.normalize().unique())
    if len(dates) < 5:
        raise ValueError("交易日过少，无法时间切分")

    cut_idx = max(1, int(len(dates) * (1.0 - float(val_frac))))
    cutoff = dates[cut_idx - 1]
    dnorm = pd.to_datetime(df[date_col]).dt.normalize()
    train_df = df[dnorm <= cutoff].copy()
    val_df = df[dnorm > cutoff].copy()
    if len(train_df) < 10 or len(val_df) < 5:
        raise ValueError("时间切分后训练/验证样本过少")

    if use_rank:
        X_tr, y_tr, g_tr, _ = _prepare_xy_z_grouped(
            train_df,
            raw_feature_names=raw_feature_names,
            target_column=target_column,
            rsi_mode=rsi_mode,
            date_col=date_col,
            min_group_size=2,
        )
        X_va, y_va, g_va, _ = _prepare_xy_z_grouped(
            val_df,
            raw_feature_names=raw_feature_names,
            target_column=target_column,
            rsi_mode=rsi_mode,
            date_col=date_col,
            min_group_size=2,
        )
    else:
        X_tr, y_tr, _ = _prepare_xy_z(
            train_df,
            raw_feature_names=raw_feature_names,
            target_column=target_column,
            rsi_mode=rsi_mode,
            date_col=date_col,
        )
        X_va, y_va, _ = _prepare_xy_z(
            val_df,
            raw_feature_names=raw_feature_names,
            target_column=target_column,
            rsi_mode=rsi_mode,
            date_col=date_col,
        )
        g_tr = np.array([], dtype=np.int32)
        g_va = np.array([], dtype=np.int32)

    if len(X_tr) < 10:
        raise ValueError("有效训练样本过少")

    default_params: Dict[str, Any] = {
        "n_estimators": 300,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_lambda": 2.0,
        "reg_alpha": 0.1,
        "random_state": training_seed,
        "n_jobs": -1,
        "tree_method": "hist",
    }
    if use_rank:
        default_params["objective"] = "rank:pairwise"
        # XGBoost 3.x：指数 NDCG 增益要求相关度等级 ≤31；dense rank 按日可远超该上限
        default_params.setdefault("ndcg_exp_gain", False)
    default_params.update({k: v for k, v in xgb_params.items() if k != "use_gpu"})

    if use_rank:
        y_tr_fit = _integer_relevance_labels(y_tr, g_tr)
        y_va_fit = _integer_relevance_labels(y_va, g_va)
        est = XGBRanker(**default_params)
        try:
            est.fit(
                X_tr,
                y_tr_fit,
                group=g_tr,
                eval_set=[(X_va, y_va_fit)],
                eval_group=[g_va],
            )
        except TypeError:
            est.fit(X_tr, y_tr_fit, group=g_tr)
    else:
        reg_params = {k: v for k, v in default_params.items() if k != "objective"}
        est = XGBRegressor(**reg_params)
        try:
            est.fit(X_tr, y_tr, eval_set=[(X_va, y_va)])
        except TypeError:
            est.fit(X_tr, y_tr)

    pred_tr = est.predict(X_tr)
    pred_va = est.predict(X_va)
    if use_rank:
        metrics = {
            "train_rank_ic": _mean_rank_ic_per_group(pred_tr, y_tr, g_tr),
            "val_rank_ic": _mean_rank_ic_per_group(pred_va, y_va, g_va),
            "train_mse": float(np.mean((y_tr - pred_tr) ** 2)),
            "val_mse": float(np.mean((y_va - pred_va) ** 2)),
        }
    else:
        metrics = {
            "train_mse": float(np.mean((y_tr - pred_tr) ** 2)),
            "val_mse": float(np.mean((y_va - pred_va) ** 2)),
            "train_mae": float(np.mean(np.abs(y_tr - pred_tr))),
            "val_mae": float(np.mean(np.abs(y_va - pred_va))),
        }

    run_id = uuid.uuid4().hex[:12]
    bundle_dir = Path(out_root) / f"xgboost_panel_{run_id}"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    save_sklearn_model(est, bundle_dir)

    inf = InferenceConfig(
        feature_columns=list(z_cols),
        target_column=target_column,
        normalize="none",
        extra={
            "backend": "xgboost",
            "raw_feature_names": list(raw_feature_names),
            "cross_section_z": True,
            "rsi_mode": rsi_mode,
            "date_col": date_col,
            "val_frac": val_frac,
            "xgboost_objective": "rank" if use_rank else "regression",
        },
    )
    save_inference_config(bundle_dir, inf)

    meta = BundleMetadata(
        model_version=model_version,
        feature_version=feature_version,
        model_type="xgboost_panel",
        backend="sklearn",
        training_seed=training_seed,
        data_slice_hash=data_slice_hash,
        content_hash=content_hash,
        created_at=utc_now_iso(),
        metrics=metrics,
        params={
            **default_params,
            "val_frac": val_frac,
            "rsi_mode": rsi_mode,
            "xgboost_objective": "rank" if use_rank else "regression",
        },
    )
    save_bundle_metadata(bundle_dir, meta)

    duration = time.perf_counter() - t0
    if log_experiments:
        rec = build_experiment_record(
            run_id=run_id,
            model_type=meta.model_type,
            duration_sec=duration,
            seed=training_seed,
            data_slice_hash=data_slice_hash,
            content_hash=content_hash,
            params=meta.params,
            metrics=metrics,
            bundle_dir=bundle_dir,
        )
        append_experiment_jsonl(experiments_dir, rec)
        append_experiment_csv(experiments_dir, rec)

    return XGBoostTrainResult(
        bundle_dir=bundle_dir,
        metrics=metrics,
        data_slice_hash=data_slice_hash,
        content_hash=content_hash,
    )
