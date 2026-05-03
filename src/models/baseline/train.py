"""线性 / 树模型训练，输出可复现工件与指标。"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from src.models.artifacts import (
    BundleMetadata,
    InferenceConfig,
    load_bundle_metadata,
    load_inference_config,
    load_sklearn_model,
    save_bundle_metadata,
    save_inference_config,
    save_normalizer_stats,
    save_sklearn_model,
    utc_now_iso,
)
from src.models.data_slice import (
    combined_data_fingerprint,
    hash_slice_spec,
    normalize_slice_spec,
    seed_everything,
)
from src.models.experiment import append_experiment_csv, append_experiment_jsonl, build_experiment_record

BaselineKind = Literal["ridge", "elasticnet", "random_forest"]


@dataclass
class BaselineTrainResult:
    bundle_dir: Path
    metrics: Dict[str, float]
    data_slice_hash: str
    content_hash: str


def _make_estimator(kind: BaselineKind, seed: int, params: Dict[str, Any]) -> Any:
    if kind == "ridge":
        return Ridge(
            alpha=float(params.get("alpha", 1.0)),
            random_state=seed,
        )
    if kind == "elasticnet":
        return ElasticNet(
            alpha=float(params.get("alpha", 0.001)),
            l1_ratio=float(params.get("l1_ratio", 0.5)),
            random_state=seed,
            max_iter=int(params.get("max_iter", 5000)),
        )
    if kind == "random_forest":
        return RandomForestRegressor(
            n_estimators=int(params.get("n_estimators", 100)),
            max_depth=params.get("max_depth"),
            min_samples_leaf=int(params.get("min_samples_leaf", 1)),
            random_state=seed,
            n_jobs=int(params.get("n_jobs", -1)),
        )
    raise ValueError(f"未知 kind: {kind}")


def _prepare_xy(
    df: pd.DataFrame,
    feature_columns: Sequence[str],
    target_column: str,
) -> Tuple[np.ndarray, np.ndarray]:
    X = df[list(feature_columns)].to_numpy(dtype=np.float64)
    y = df[target_column].to_numpy(dtype=np.float64)
    mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    return X[mask], y[mask]


def train_baseline(
    df: pd.DataFrame,
    *,
    kind: BaselineKind,
    feature_columns: Sequence[str],
    target_column: str,
    model_version: str = "1.0.0",
    feature_version: str = "v1",
    training_seed: int = 42,
    test_size: float = 0.2,
    normalize: str = "none",
    slice_spec: Optional[dict] = None,
    extra_params: Optional[Dict[str, Any]] = None,
    out_root: Union[str, Path] = "data/models",
    log_experiments: bool = True,
    experiments_dir: Union[str, Path] = "data/experiments",
) -> BaselineTrainResult:
    """
    训练基线模型并写入 ``out_root/<run_id>/``。

    Parameters
    ----------
    slice_spec
        若给定，写入 ``bundle.json`` 的 ``data_slice_hash``（与数据内容哈希共同锁定切片）。
    """
    t0 = time.perf_counter()
    seed_everything(training_seed)
    extra_params = extra_params or {}

    spec = slice_spec or normalize_slice_spec()
    data_slice_hash = hash_slice_spec(spec)
    fp = combined_data_fingerprint(df, slice_spec=spec, content_columns=list(feature_columns) + [target_column])
    content_hash = fp["content_hash"]

    X, y = _prepare_xy(df, feature_columns, target_column)
    if len(X) < 4:
        raise ValueError("有效样本过少，无法训练")

    # 时序划分：按行顺序切割，避免随机划分引入前视偏差
    n = len(X)
    cutoff = max(1, int(n * (1.0 - test_size)))
    X_train, X_test = X[:cutoff], X[cutoff:]
    y_train, y_test = y[:cutoff], y[cutoff:]

    normalizer: Optional[Tuple[np.ndarray, np.ndarray]] = None
    if normalize == "standard":
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        normalizer = (scaler.mean_.astype(np.float64), scaler.scale_.astype(np.float64))

    est = _make_estimator(kind, training_seed, extra_params)
    est.fit(X_train, y_train)
    pred_tr = est.predict(X_train)
    pred_te = est.predict(X_test)
    metrics = {
        "train_mse": float(mean_squared_error(y_train, pred_tr)),
        "test_mse": float(mean_squared_error(y_test, pred_te)),
        "train_r2": float(r2_score(y_train, pred_tr)),
        "test_r2": float(r2_score(y_test, pred_te)),
    }

    run_id = uuid.uuid4().hex[:12]
    bundle_dir = Path(out_root) / f"baseline_{kind}_{run_id}"
    bundle_dir.mkdir(parents=True, exist_ok=True)

    save_sklearn_model(est, bundle_dir)

    inf = InferenceConfig(
        feature_columns=list(feature_columns),
        target_column=target_column,
        normalize=normalize,
        extra={"kind": kind, "test_size": test_size},
    )
    save_inference_config(bundle_dir, inf)

    if normalizer is not None:
        save_normalizer_stats(bundle_dir, normalizer[0], normalizer[1])

    meta = BundleMetadata(
        model_version=model_version,
        feature_version=feature_version,
        model_type=f"baseline_{kind}",
        backend="sklearn",
        training_seed=training_seed,
        data_slice_hash=data_slice_hash,
        content_hash=content_hash,
        created_at=utc_now_iso(),
        metrics=metrics,
        params={"kind": kind, **extra_params, "test_size": test_size, "normalize": normalize},
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

    return BaselineTrainResult(
        bundle_dir=bundle_dir,
        metrics=metrics,
        data_slice_hash=data_slice_hash,
        content_hash=content_hash,
    )


def load_baseline_bundle(bundle_dir: Union[str, Path]) -> Tuple[Any, InferenceConfig, BundleMetadata]:
    root = Path(bundle_dir)
    meta = load_bundle_metadata(root)
    inf = load_inference_config(root / "inference_config.json")
    model = load_sklearn_model(root / "model.joblib")
    return model, inf, meta
