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
from sklearn.model_selection import TimeSeriesSplit

from src.models.artifacts import (
    BundleMetadata,
    InferenceConfig,
    list_bundle_dirs,
    load_metric_from_bundle,
    publish_bundle_with_history,
    save_bundle_metadata,
    save_inference_config,
    save_sklearn_model,
    should_promote_by_quantile,
    utc_now_iso,
)
from src.models.data_slice import (
    combined_data_fingerprint,
    hash_slice_spec,
    normalize_slice_spec,
    seed_everything,
)
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


def _fit_rank_and_eval_rank_ic(
    *,
    params: Dict[str, Any],
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    g_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    g_va: np.ndarray,
) -> float:
    """训练 ranker 并返回验证集按组 Rank IC。"""
    y_tr_fit = _integer_relevance_labels(y_tr, g_tr)
    y_va_fit = _integer_relevance_labels(y_va, g_va)
    est = XGBRanker(**params)
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
    pred_va = est.predict(X_va)
    return _mean_rank_ic_per_group(pred_va, y_va, g_va)


def _fit_reg_and_eval_mse(
    *,
    params: Dict[str, Any],
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
) -> float:
    """训练 regressor 并返回验证集 MSE。"""
    est = XGBRegressor(**params)
    try:
        est.fit(X_tr, y_tr, eval_set=[(X_va, y_va)])
    except TypeError:
        est.fit(X_tr, y_tr)
    pred_va = est.predict(X_va)
    return float(np.mean((y_va - pred_va) ** 2))


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
    label_spec: Optional[Dict[str, Any]] = None,
    enforce_metric_guard: bool = True,
    guard_metric_key: str = "val_rank_ic",
    guard_quantile: float = 0.25,
    guard_min_history: int = 4,
    min_rank_ic_to_publish: float = 0.03,
    time_cv_splits: int = 3,
    keep_recent_versions: int = 8,
    publish_bundle_dir: Optional[Union[str, Path]] = None,
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
    label_spec = dict(label_spec or {})
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

    cv_metrics: Dict[str, float] = {}
    n_cv = int(max(0, time_cv_splits))
    if n_cv >= 2:
        train_dates = np.sort(pd.to_datetime(train_df[date_col]).dt.normalize().unique())
        if len(train_dates) >= n_cv + 1:
            cv_values: List[float] = []
            tscv = TimeSeriesSplit(n_splits=n_cv)
            for tr_idx_date, va_idx_date in tscv.split(train_dates):
                tr_dates = set(train_dates[tr_idx_date])
                va_dates = set(train_dates[va_idx_date])
                fold_tr = train_df[pd.to_datetime(train_df[date_col]).dt.normalize().isin(tr_dates)]
                fold_va = train_df[pd.to_datetime(train_df[date_col]).dt.normalize().isin(va_dates)]
                if fold_tr.empty or fold_va.empty:
                    continue
                if use_rank:
                    X_ftr, y_ftr, g_ftr, _ = _prepare_xy_z_grouped(
                        fold_tr,
                        raw_feature_names=raw_feature_names,
                        target_column=target_column,
                        rsi_mode=rsi_mode,
                        date_col=date_col,
                        min_group_size=2,
                    )
                    X_fva, y_fva, g_fva, _ = _prepare_xy_z_grouped(
                        fold_va,
                        raw_feature_names=raw_feature_names,
                        target_column=target_column,
                        rsi_mode=rsi_mode,
                        date_col=date_col,
                        min_group_size=2,
                    )
                    if len(X_ftr) < 10 or len(X_fva) < 5:
                        continue
                    v = _fit_rank_and_eval_rank_ic(
                        params=default_params,
                        X_tr=X_ftr,
                        y_tr=y_ftr,
                        g_tr=g_ftr,
                        X_va=X_fva,
                        y_va=y_fva,
                        g_va=g_fva,
                    )
                else:
                    X_ftr, y_ftr, _ = _prepare_xy_z(
                        fold_tr,
                        raw_feature_names=raw_feature_names,
                        target_column=target_column,
                        rsi_mode=rsi_mode,
                        date_col=date_col,
                    )
                    X_fva, y_fva, _ = _prepare_xy_z(
                        fold_va,
                        raw_feature_names=raw_feature_names,
                        target_column=target_column,
                        rsi_mode=rsi_mode,
                        date_col=date_col,
                    )
                    if len(X_ftr) < 10 or len(X_fva) < 5:
                        continue
                    reg_params = {k: v for k, v in default_params.items() if k != "objective"}
                    v = _fit_reg_and_eval_mse(
                        params=reg_params,
                        X_tr=X_ftr,
                        y_tr=y_ftr,
                        X_va=X_fva,
                        y_va=y_fva,
                    )
                if np.isfinite(v):
                    cv_values.append(float(v))
            if cv_values:
                arr = np.asarray(cv_values, dtype=np.float64)
                if use_rank:
                    cv_metrics = {
                        "cv_rank_ic_mean": float(np.nanmean(arr)),
                        "cv_rank_ic_std": float(np.nanstd(arr)),
                        "cv_rank_ic_min": float(np.nanmin(arr)),
                        "cv_folds": float(len(arr)),
                    }
                else:
                    cv_metrics = {
                        "cv_val_mse_mean": float(np.nanmean(arr)),
                        "cv_val_mse_std": float(np.nanstd(arr)),
                        "cv_val_mse_min": float(np.nanmin(arr)),
                        "cv_folds": float(len(arr)),
                    }

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
        metrics.update(cv_metrics)
    else:
        metrics = {
            "train_mse": float(np.mean((y_tr - pred_tr) ** 2)),
            "val_mse": float(np.mean((y_va - pred_va) ** 2)),
            "train_mae": float(np.mean(np.abs(y_tr - pred_tr))),
            "val_mae": float(np.mean(np.abs(y_va - pred_va))),
        }
        metrics.update(cv_metrics)

    guard_decision: Dict[str, Any] = {}
    if use_rank:
        val_rank_ic = float(metrics.get("val_rank_ic", np.nan))
        if not np.isfinite(val_rank_ic) or val_rank_ic < float(min_rank_ic_to_publish):
            raise RuntimeError(
                f"候选模型 val_rank_ic={val_rank_ic:.6f} 低于最低门槛 {float(min_rank_ic_to_publish):.6f}，拒绝发布。"
            )
    if use_rank and enforce_metric_guard:
        hist_dirs = list_bundle_dirs(out_root, prefix="xgboost_panel_")
        history_vals: List[float] = []
        for d in hist_dirs:
            v = load_metric_from_bundle(d, guard_metric_key)
            if np.isfinite(v):
                history_vals.append(v)
        cand = float(metrics.get(guard_metric_key, np.nan))
        ok, qv = should_promote_by_quantile(
            candidate_metric=cand,
            history_metrics=history_vals,
            quantile=guard_quantile,
            min_history=guard_min_history,
        )
        guard_decision = {
            "metric_key": guard_metric_key,
            "candidate_metric": cand,
            "history_count": len(history_vals),
            "quantile": float(guard_quantile),
            "threshold": qv,
            "passed": bool(ok),
        }
        if not ok:
            raise RuntimeError(
                f"候选模型 {guard_metric_key}={cand:.6f} 低于历史 P{int(guard_quantile * 100)} 阈值 {qv:.6f}，拒绝替换。"
            )

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
            "label_spec": label_spec,
            "time_cv_splits": int(time_cv_splits),
            "min_rank_ic_to_publish": float(min_rank_ic_to_publish),
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
            "label_spec": label_spec,
            "time_cv_splits": int(time_cv_splits),
            "min_rank_ic_to_publish": float(min_rank_ic_to_publish),
            "metric_guard": guard_decision,
        },
    )
    save_bundle_metadata(bundle_dir, meta)

    if keep_recent_versions > 0:
        all_dirs = list_bundle_dirs(out_root, prefix="xgboost_panel_")
        if len(all_dirs) > keep_recent_versions:
            for d in all_dirs[: len(all_dirs) - keep_recent_versions]:
                try:
                    import shutil

                    shutil.rmtree(d, ignore_errors=True)
                except Exception:
                    pass

    if publish_bundle_dir:
        publish_dir = Path(publish_bundle_dir)
        if not publish_dir.is_absolute():
            publish_dir = Path(out_root) / publish_dir
        history_root = Path(out_root) / "xgboost_panel_history"
        publish_meta = publish_bundle_with_history(
            source_bundle_dir=bundle_dir,
            publish_dir=publish_dir,
            history_root=history_root,
            keep_recent=keep_recent_versions,
        )
        meta.params["publish"] = publish_meta
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
