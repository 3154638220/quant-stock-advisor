"""W3: LightGBM 基线训练器 — lambdarank / regression / binary 分类。

与现有 ExtraTrees/ElasticNet baseline 并列对比，使用相同的 walk-forward CV 切分。
验收 gate：Rank IC delta vs ExtraTrees >= 0（统计显著 p < 0.1）。

使用方式::

    from src.models.baseline.lightgbm_ranker import train_lgbm_ranker
    result = train_lgbm_ranker(df, feature_columns=cols, target_column="excess_forward_1m")
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
    utc_now_iso,
)
from src.models.data_slice import (
    combined_data_fingerprint,
    hash_slice_spec,
    normalize_slice_spec,
    seed_everything,
)
from src.models.experiment import append_experiment_csv, append_experiment_jsonl, build_experiment_record

LGBMRankerKind = Literal["lgbm_regression", "lgbm_lambdarank", "lgbm_binary_top20"]


@dataclass
class LGBMRankerResult:
    bundle_dir: Path
    metrics: Dict[str, float]
    data_slice_hash: str
    content_hash: str
    feature_importance: pd.DataFrame


def _check_lightgbm() -> None:
    try:
        import lightgbm as lgb
        _ = lgb
    except ImportError:
        raise ImportError(
            "LightGBM 未安装。请执行: pip install lightgbm"
        )


def _make_lgbm_dataset(
    X: np.ndarray, y: np.ndarray, group: Optional[np.ndarray] = None
):
    """构造 LightGBM Dataset，支持 ranking group。"""
    import lightgbm as lgb

    if group is not None:
        return lgb.Dataset(X, label=y, group=group)
    return lgb.Dataset(X, label=y)


def _prepare_xy(
    df: pd.DataFrame,
    feature_columns: Sequence[str],
    target_column: str,
) -> Tuple[np.ndarray, np.ndarray]:
    X = df[list(feature_columns)].to_numpy(dtype=np.float64)
    y = df[target_column].to_numpy(dtype=np.float64)
    mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    return X[mask], y[mask]


def _prepare_ranking_xy(
    df: pd.DataFrame,
    feature_columns: Sequence[str],
    target_column: str,
    date_col: str = "trade_date",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """准备 ranking 数据，按日期分组。

    Returns
    -------
    X, y, group : 其中 group[i] = 第 i 组（日期）的样本数
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce").dt.normalize()
    df = df.sort_values(date_col)

    X_list, y_list, groups = [], [], []
    for _, g in df.groupby(date_col, sort=True):
        X_g = g[list(feature_columns)].to_numpy(dtype=np.float64)
        y_g = g[target_column].to_numpy(dtype=np.float64)
        mask = np.isfinite(X_g).all(axis=1) & np.isfinite(y_g)
        X_g, y_g = X_g[mask], y_g[mask]
        if len(X_g) < 5:
            continue
        X_list.append(X_g)
        y_list.append(y_g)
        groups.append(len(X_g))

    if not X_list:
        return np.empty((0, len(feature_columns))), np.empty((0,)), np.array([], dtype=np.int32)

    return np.vstack(X_list), np.concatenate(y_list), np.array(groups, dtype=np.int32)


def _params_for_kind(
    kind: LGBMRankerKind,
    seed: int,
    extra_params: Dict[str, Any],
) -> Dict[str, Any]:
    """返回 LightGBM 参数字典。"""
    base = {
        "seed": seed,
        "feature_fraction": float(extra_params.get("feature_fraction", 0.8)),
        "bagging_fraction": float(extra_params.get("bagging_fraction", 0.8)),
        "bagging_freq": int(extra_params.get("bagging_freq", 1)),
        "num_leaves": int(extra_params.get("num_leaves", 63)),
        "min_child_samples": int(extra_params.get("min_child_samples", 100)),
        "learning_rate": float(extra_params.get("learning_rate", 0.05)),
        "n_estimators": int(extra_params.get("n_estimators", 500)),
        "verbose": -1,
        "num_threads": int(extra_params.get("num_threads", -1)),
    }

    if kind == "lgbm_regression":
        return {**base, "objective": "regression", "metric": "l2"}
    elif kind == "lgbm_lambdarank":
        return {
            **base,
            "objective": "lambdarank",
            "metric": "ndcg",
            "ndcg_eval_at": [20],
            "num_leaves": int(extra_params.get("num_leaves", 31)),
        }
    elif kind == "lgbm_binary_top20":
        return {
            **base,
            "objective": "binary",
            "metric": "auc",
        }
    raise ValueError(f"未知 kind: {kind}")


def _spearman_rank_ic(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """计算 Spearman Rank IC。"""
    from scipy.stats import spearmanr
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 5:
        return 0.0
    corr, _ = spearmanr(y_true[mask], y_pred[mask])
    return float(corr) if np.isfinite(corr) else 0.0


def _topk_hit_rate(y_true: np.ndarray, y_pred: np.ndarray, k: int = 20) -> float:
    """Top-K 命中率：预测的 top-k 中真实 top-k 的占比。"""
    n = len(y_true)
    if n < k:
        return 0.0
    true_top = set(np.argsort(y_true)[-k:])
    pred_top = set(np.argsort(y_pred)[-k:])
    return float(len(true_top & pred_top) / k)


def train_lgbm_ranker(
    df: pd.DataFrame,
    *,
    kind: LGBMRankerKind,
    feature_columns: Sequence[str],
    target_column: str,
    date_col: str = "trade_date",
    model_version: str = "1.0.0",
    feature_version: str = "v1",
    training_seed: int = 42,
    test_size: float = 0.2,
    slice_spec: Optional[dict] = None,
    extra_params: Optional[Dict[str, Any]] = None,
    out_root: Union[str, Path] = "data/models",
    log_experiments: bool = True,
    experiments_dir: Union[str, Path] = "data/experiments",
) -> LGBMRankerResult:
    """训练 LightGBM 排序/回归模型并写入 bundle。

    Parameters
    ----------
    kind : str
        "lgbm_regression" / "lgbm_lambdarank" / "lgbm_binary_top20"
    feature_columns : list[str]
        特征列名列表。
    target_column : str
        目标列名（excess_forward_1m 等）。
    date_col : str
        日期列名，用于时序切分和 ranking group。
    test_size : float
        验证集比例（按时间顺序切分尾部）。
    """
    _check_lightgbm()
    import lightgbm as lgb

    t0 = time.perf_counter()
    seed_everything(training_seed)
    extra_params = extra_params or {}

    spec = slice_spec or normalize_slice_spec()
    data_slice_hash = hash_slice_spec(spec)
    fp = combined_data_fingerprint(
        df, slice_spec=spec, content_columns=list(feature_columns) + [target_column]
    )
    content_hash = fp["content_hash"]

    # ── 时序切分 ──
    df_sorted = df.sort_values(date_col).reset_index(drop=True)
    n = len(df_sorted)
    cutoff = max(1, int(n * (1.0 - test_size)))

    params = _params_for_kind(kind, training_seed, extra_params)

    if kind == "lgbm_lambdarank":
        X_train, y_train, g_train = _prepare_ranking_xy(
            df_sorted.iloc[:cutoff], feature_columns, target_column, date_col
        )
        X_test, y_test, g_test = _prepare_ranking_xy(
            df_sorted.iloc[cutoff:], feature_columns, target_column, date_col
        )

        if len(X_train) == 0 or len(X_test) == 0:
            raise ValueError("有效 ranking 样本过少，无法训练")

        dtrain = _make_lgbm_dataset(X_train, y_train, group=g_train)
        dval = _make_lgbm_dataset(X_test, y_test, group=g_test)

        model = lgb.train(
            params,
            dtrain,
            valid_sets=[dtrain, dval],
            valid_names=["train", "val"],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=0),
            ],
        )
        pred_tr = model.predict(X_train)
        pred_te = model.predict(X_test)

    elif kind == "lgbm_regression":
        X_train, y_train = _prepare_xy(df_sorted.iloc[:cutoff], feature_columns, target_column)
        X_test, y_test = _prepare_xy(df_sorted.iloc[cutoff:], feature_columns, target_column)

        if len(X_train) < 4 or len(X_test) < 4:
            raise ValueError("有效回归样本过少，无法训练")

        dtrain = _make_lgbm_dataset(X_train, y_train)
        dval = _make_lgbm_dataset(X_test, y_test)
        model = lgb.train(
            params,
            dtrain,
            valid_sets=[dtrain, dval],
            valid_names=["train", "val"],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=0),
            ],
        )
        pred_tr = model.predict(X_train)
        pred_te = model.predict(X_test)

    elif kind == "lgbm_binary_top20":
        # 构造 Top-20% 二分类标签
        df_labeled = df_sorted.copy()
        df_labeled["_top20_label"] = (
            df_labeled.groupby(date_col)[target_column].transform(
                lambda x: (x.rank(pct=True) >= 0.80).astype(int)
            )
        ).fillna(0).astype(int)

        X_train, y_train = _prepare_xy(
            df_labeled.iloc[:cutoff], feature_columns, "_top20_label"
        )
        X_test, y_test = _prepare_xy(
            df_labeled.iloc[cutoff:], feature_columns, "_top20_label"
        )

        if len(X_train) < 4 or len(X_test) < 4:
            raise ValueError("有效二分类样本过少，无法训练")

        dtrain = _make_lgbm_dataset(X_train, y_train)
        dval = _make_lgbm_dataset(X_test, y_test)
        model = lgb.train(
            params,
            dtrain,
            valid_sets=[dtrain, dval],
            valid_names=["train", "val"],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=0),
            ],
        )
        pred_tr = model.predict(X_train)
        pred_te = model.predict(X_test)

    else:
        raise ValueError(f"未知 kind: {kind}")

    # ── 指标计算 ──
    from sklearn.metrics import mean_squared_error, r2_score

    if kind == "lgbm_binary_top20":
        from sklearn.metrics import roc_auc_score
        metrics = {
            "train_auc": float(roc_auc_score(y_train, pred_tr)) if len(np.unique(y_train)) > 1 else 0.5,
            "test_auc": float(roc_auc_score(y_test, pred_te)) if len(np.unique(y_test)) > 1 else 0.5,
            "train_rank_ic": _spearman_rank_ic(y_train, pred_tr),
            "test_rank_ic": _spearman_rank_ic(y_test, pred_te),
        }
    else:
        metrics = {
            "train_mse": float(mean_squared_error(y_train, pred_tr)),
            "test_mse": float(mean_squared_error(y_test, pred_te)),
            "train_r2": float(r2_score(y_train, pred_tr)) if len(y_train) > 2 else 0.0,
            "test_r2": float(r2_score(y_test, pred_te)) if len(y_test) > 2 else 0.0,
            "train_rank_ic": _spearman_rank_ic(y_train, pred_tr),
            "test_rank_ic": _spearman_rank_ic(y_test, pred_te),
        }

    # ── 特征重要性 ──
    importance_df = pd.DataFrame({
        "feature": list(feature_columns),
        "gain": model.feature_importance(importance_type="gain"),
    }).sort_values("gain", ascending=False)

    # ── 保存 bundle ──
    run_id = uuid.uuid4().hex[:12]
    bundle_dir = Path(out_root) / f"lgbm_{kind}_{run_id}"
    bundle_dir.mkdir(parents=True, exist_ok=True)

    model_file = bundle_dir / "model.txt"
    model.save_model(str(model_file))

    inf = InferenceConfig(
        feature_columns=list(feature_columns),
        target_column=target_column,
        normalize="none",
        extra={"kind": kind, "test_size": test_size, "lgbm_params": params},
    )
    save_inference_config(bundle_dir, inf)

    meta = BundleMetadata(
        model_version=model_version,
        feature_version=feature_version,
        model_type=f"lgbm_{kind}",
        backend="lightgbm",
        training_seed=training_seed,
        data_slice_hash=data_slice_hash,
        content_hash=content_hash,
        created_at=utc_now_iso(),
        metrics=metrics,
        params={"kind": kind, **extra_params, "test_size": test_size},
    )
    save_bundle_metadata(bundle_dir, meta)

    # ── 实验记录 ──
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

    return LGBMRankerResult(
        bundle_dir=bundle_dir,
        metrics=metrics,
        data_slice_hash=data_slice_hash,
        content_hash=content_hash,
        feature_importance=importance_df,
    )


def load_lgbm_bundle(bundle_dir: Union[str, Path]) -> Tuple[Any, InferenceConfig, BundleMetadata]:
    """加载 LightGBM bundle。"""
    import lightgbm as lgb
    from src.models.artifacts import load_bundle_metadata, load_inference_config

    root = Path(bundle_dir)
    meta = load_bundle_metadata(root)
    inf = load_inference_config(root / "inference_config.json")
    model = lgb.Booster(model_file=str(root / "model.txt"))
    return model, inf, meta
