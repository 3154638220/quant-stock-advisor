"""面板数据 → 序列张量，训练 LSTM/TCN 并写出工件。"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.artifacts import (
    BundleMetadata,
    InferenceConfig,
    load_bundle_metadata,
    load_inference_config,
    load_torch_state,
    save_bundle_metadata,
    save_inference_config,
    save_torch_state,
    utc_now_iso,
)
from src.models.data_slice import (
    combined_data_fingerprint,
    hash_slice_spec,
    normalize_slice_spec,
    seed_everything,
)
from src.models.experiment import append_experiment_csv, append_experiment_jsonl, build_experiment_record
from src.models.timeseries.lstm_tcn import ArchKind, build_timeseries_model
from src.models.timeseries.ohlcv_norm import normalize_ohlcv_anchor

TimeseriesTargetTask = Literal["regression", "binary_up"]
TimeseriesKind = ArchKind

NormalizeMode = Literal["none", "ohlcv_anchor"]


@dataclass
class TimeseriesTrainResult:
    bundle_dir: Path
    metrics: Dict[str, float]
    data_slice_hash: str
    content_hash: str


def build_panel_sequences(
    df: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    target_column: str,
    seq_len: int,
    symbol_col: str = "symbol",
    date_col: str = "trade_date",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    按标的分组、按日期排序，滑动窗口生成 ``X: (N, seq_len, F)``、``y: (N,)``。
    窗口末行对齐目标 ``target_column``。
    """
    need = {symbol_col, date_col, target_column, *feature_columns}
    if not need.issubset(df.columns):
        raise ValueError(f"缺少列: {need - set(df.columns)}")

    sub = df[list(need)].copy()
    sub[date_col] = pd.to_datetime(sub[date_col]).dt.normalize()
    sub = sub.sort_values([symbol_col, date_col])
    feats = list(feature_columns)

    xs: List[np.ndarray] = []
    ys: List[float] = []
    for _, g in sub.groupby(symbol_col, sort=False):
        g = g.reset_index(drop=True)
        if len(g) < seq_len:
            continue
        mat = g[feats].to_numpy(dtype=np.float64)
        targ = g[target_column].to_numpy(dtype=np.float64)
        for t in range(seq_len - 1, len(g)):
            window = mat[t - seq_len + 1 : t + 1]
            yv = targ[t]
            if not np.isfinite(window).all() or not np.isfinite(yv):
                continue
            xs.append(window)
            ys.append(float(yv))

    if not xs:
        raise ValueError("无有效序列样本（检查 seq_len、缺失值）")

    X = np.stack(xs, axis=0)
    y = np.asarray(ys, dtype=np.float64)
    return X, y


def build_panel_sequences_with_dates(
    df: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    target_column: str,
    seq_len: int,
    symbol_col: str = "symbol",
    date_col: str = "trade_date",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回序列样本及其窗口末日（用于 walk-forward OOS 切分）。"""
    need = {symbol_col, date_col, target_column, *feature_columns}
    if not need.issubset(df.columns):
        raise ValueError(f"缺少列: {need - set(df.columns)}")
    sub = df[list(need)].copy()
    sub[date_col] = pd.to_datetime(sub[date_col]).dt.normalize()
    sub = sub.sort_values([symbol_col, date_col])

    xs: List[np.ndarray] = []
    ys: List[float] = []
    ds: List[np.datetime64] = []
    feats = list(feature_columns)
    for _, g in sub.groupby(symbol_col, sort=False):
        g = g.reset_index(drop=True)
        if len(g) < seq_len:
            continue
        mat = g[feats].to_numpy(dtype=np.float64)
        targ = g[target_column].to_numpy(dtype=np.float64)
        dts = pd.to_datetime(g[date_col]).to_numpy(dtype="datetime64[ns]")
        for t in range(seq_len - 1, len(g)):
            window = mat[t - seq_len + 1 : t + 1]
            yv = targ[t]
            if not np.isfinite(window).all() or not np.isfinite(yv):
                continue
            xs.append(window)
            ys.append(float(yv))
            ds.append(np.datetime64(dts[t]))
    if not xs:
        raise ValueError("无有效序列样本（检查 seq_len、缺失值）")
    return np.stack(xs, axis=0), np.asarray(ys, dtype=np.float64), np.asarray(ds, dtype="datetime64[ns]")


def _mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


def _r2(a: np.ndarray, b: np.ndarray) -> float:
    ss_res = float(np.sum((a - b) ** 2))
    ss_tot = float(np.sum((a - np.mean(a)) ** 2))
    return float(1.0 - ss_res / ss_tot) if ss_tot > 1e-18 else 0.0


def _binary_acc(y_true: np.ndarray, logits: np.ndarray) -> float:
    p = 1.0 / (1.0 + np.exp(-np.clip(logits, -50.0, 50.0)))
    pb = (p >= 0.5).astype(np.float64)
    return float(np.mean(pb == y_true))


def _set_torch_deterministic(seed: int) -> torch.Generator:
    seed_everything(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    g = torch.Generator()
    g.manual_seed(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except (RuntimeError, TypeError):
        pass
    return g


def _walk_forward_oos_metrics(
    df: pd.DataFrame,
    *,
    kind: TimeseriesKind,
    feature_columns: Sequence[str],
    target_column: str,
    seq_len: int,
    training_seed: int,
    device: str,
    hidden: int,
    num_layers: int,
    kernel: int,
    num_blocks: int,
    dropout: float,
    d_model: int,
    nhead: int,
    num_encoder_layers: int,
    dim_feedforward: int,
    max_seq_len: int,
    batch_size: int,
    lr: float,
    normalize_mode: NormalizeMode,
    target_task: TimeseriesTargetTask,
    train_days: int,
    test_days: int,
    step_days: int,
    epochs: int,
    min_folds: int,
) -> Dict[str, float]:
    X, y, end_dates = build_panel_sequences_with_dates(
        df,
        feature_columns=feature_columns,
        target_column=target_column,
        seq_len=seq_len,
    )
    if normalize_mode == "ohlcv_anchor":
        X = normalize_ohlcv_anchor(X)
    if str(target_task).lower() == "binary_up":
        y = (y > 0.0).astype(np.float64)

    unique_dates = np.sort(np.unique(end_dates))
    if len(unique_dates) < (train_days + test_days):
        return {}

    use_binary = str(target_task).lower() == "binary_up"
    metrics: List[float] = []
    fold_cnt = 0
    start = 0
    while start + train_days + test_days <= len(unique_dates):
        tr_start = unique_dates[start]
        tr_end = unique_dates[start + train_days - 1]
        te_start = unique_dates[start + train_days]
        te_end = unique_dates[start + train_days + test_days - 1]

        tr_m = (end_dates >= tr_start) & (end_dates <= tr_end)
        te_m = (end_dates >= te_start) & (end_dates <= te_end)
        if int(np.sum(tr_m)) < 16 or int(np.sum(te_m)) < 4:
            start += step_days
            continue

        X_tr = torch.from_numpy(X[tr_m]).float()
        y_tr = torch.from_numpy(y[tr_m]).float()
        X_te = torch.from_numpy(X[te_m]).float()
        y_te = torch.from_numpy(y[te_m]).float()
        n_feat = X_tr.shape[2]

        gen = _set_torch_deterministic(training_seed + fold_cnt + 1)
        model = build_timeseries_model(
            kind,
            n_feat,
            hidden=hidden,
            num_layers=num_layers,
            kernel=kernel,
            num_blocks=num_blocks,
            dropout=dropout,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            max_seq_len=max(seq_len, max_seq_len),
        ).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn: nn.Module = nn.BCEWithLogitsLoss() if use_binary else nn.MSELoss()
        ds_tr = TensorDataset(X_tr, y_tr)
        dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, generator=gen)
        model.train()
        for _ in range(max(1, int(epochs))):
            for xb, yb in dl_tr:
                xb = xb.to(device)
                yb = yb.to(device)
                opt.zero_grad(set_to_none=True)
                pred = model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                opt.step()
        model.eval()
        with torch.no_grad():
            pred_te = model(X_te.to(device)).cpu().numpy()
        y_te_np = y_te.numpy()
        if use_binary:
            metrics.append(_binary_acc(y_te_np, pred_te))
        else:
            metrics.append(_r2(y_te_np, pred_te))
        fold_cnt += 1
        start += step_days

    if fold_cnt < int(min_folds):
        return {}
    metric_key = "wf_test_acc_mean" if use_binary else "wf_test_r2_mean"
    return {
        "wf_folds": float(fold_cnt),
        metric_key: float(np.mean(metrics)),
    }


def train_timeseries(
    df: pd.DataFrame,
    *,
    kind: TimeseriesKind,
    feature_columns: Sequence[str],
    target_column: str,
    seq_len: int = 20,
    model_version: str = "1.0.0",
    feature_version: str = "v1",
    training_seed: int = 42,
    test_size: float = 0.2,
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: Optional[str] = None,
    hidden: int = 64,
    num_layers: int = 2,
    kernel: int = 3,
    num_blocks: int = 3,
    dropout: float = 0.1,
    normalize_mode: NormalizeMode = "none",
    d_model: int = 64,
    nhead: int = 4,
    num_encoder_layers: int = 2,
    dim_feedforward: int = 128,
    max_seq_len: int = 128,
    time_val_split: bool = False,
    target_task: TimeseriesTargetTask = "regression",
    walk_forward_oos: bool = False,
    wf_train_days: int = 252,
    wf_test_days: int = 63,
    wf_step_days: int = 63,
    wf_epochs: int = 8,
    wf_min_folds: int = 2,
    slice_spec: Optional[dict] = None,
    out_root: Union[str, Path] = "data/models",
    log_experiments: bool = True,
    experiments_dir: Union[str, Path] = "data/experiments",
) -> TimeseriesTrainResult:
    t0 = time.perf_counter()
    gen = _set_torch_deterministic(training_seed)

    spec = slice_spec or normalize_slice_spec()
    data_slice_hash = hash_slice_spec(spec)
    cols = list(feature_columns) + [target_column]
    fp = combined_data_fingerprint(df, slice_spec=spec, content_columns=cols)
    content_hash = fp["content_hash"]

    date_col = "trade_date"
    if time_val_split:
        sub_dates = pd.to_datetime(df[date_col]).dt.normalize()
        dates_u = sorted(sub_dates.unique())
        if len(dates_u) < 4:
            raise ValueError("交易日过少，无法按时间划分验证集")
        cut = max(1, int(len(dates_u) * (1.0 - float(test_size))))
        cutoff = dates_u[cut - 1]
        tr_df = df[sub_dates <= cutoff].copy()
        va_df = df[sub_dates > cutoff].copy()
        X_np, y_np = build_panel_sequences(
            tr_df,
            feature_columns=feature_columns,
            target_column=target_column,
            seq_len=seq_len,
            date_col=date_col,
        )
        X_te_np, y_te_np = build_panel_sequences(
            va_df,
            feature_columns=feature_columns,
            target_column=target_column,
            seq_len=seq_len,
            date_col=date_col,
        )
        if normalize_mode == "ohlcv_anchor":
            X_np = normalize_ohlcv_anchor(X_np)
            X_te_np = normalize_ohlcv_anchor(X_te_np)
        n_feat = X_np.shape[2]
        if len(X_np) < 4 or len(X_te_np) < 1:
            raise ValueError("时间划分后序列样本过少")
        tr_idx = np.arange(len(X_np))
        te_idx = np.arange(len(X_te_np))
        X_tr = torch.from_numpy(X_np).float()
        y_tr = torch.from_numpy(y_np).float()
        X_te = torch.from_numpy(X_te_np).float()
        y_te = torch.from_numpy(y_te_np).float()
        if str(target_task).lower() == "binary_up":
            y_tr = (y_tr > 0.0).float()
            y_te = (y_te > 0.0).float()
    else:
        X_np, y_np = build_panel_sequences(
            df,
            feature_columns=feature_columns,
            target_column=target_column,
            seq_len=seq_len,
        )
        if normalize_mode == "ohlcv_anchor":
            X_np = normalize_ohlcv_anchor(X_np)
        n_feat = X_np.shape[2]
        n = len(X_np)
        if n < 8:
            raise ValueError("有效序列样本过少")

        idx = np.arange(n)
        rng = np.random.default_rng(training_seed)
        rng.shuffle(idx)
        n_test = max(1, int(n * test_size))
        te_idx = idx[:n_test]
        tr_idx = idx[n_test:]
        if len(tr_idx) < 1:
            raise ValueError("训练集为空")

        X_tr = torch.from_numpy(X_np[tr_idx]).float()
        y_tr = torch.from_numpy(y_np[tr_idx]).float()
        X_te = torch.from_numpy(X_np[te_idx]).float()
        y_te = torch.from_numpy(y_np[te_idx]).float()
        if str(target_task).lower() == "binary_up":
            y_tr = (y_tr > 0.0).float()
            y_te = (y_te > 0.0).float()

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = build_timeseries_model(
        kind,
        n_feat,
        hidden=hidden,
        num_layers=num_layers,
        kernel=kernel,
        num_blocks=num_blocks,
        dropout=dropout,
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        dim_feedforward=dim_feedforward,
        max_seq_len=max(seq_len, max_seq_len),
    ).to(dev)

    ds_tr = TensorDataset(X_tr, y_tr)
    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, generator=gen)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    use_binary = str(target_task).lower() == "binary_up"
    loss_fn: nn.Module = nn.BCEWithLogitsLoss() if use_binary else nn.MSELoss()

    model.train()
    for _ in range(epochs):
        for xb, yb in dl_tr:
            xb = xb.to(dev)
            yb = yb.to(dev)
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        pred_tr = model(X_tr.to(dev)).cpu().numpy()
        pred_te = model(X_te.to(dev)).cpu().numpy()
    y_tr_np = y_tr.numpy()
    y_te_np = y_te.numpy()

    if use_binary:
        metrics = {
            "train_acc": _binary_acc(y_tr_np, pred_tr),
            "test_acc": _binary_acc(y_te_np, pred_te),
            "train_mse": _mse(y_tr_np, pred_tr),
            "test_mse": _mse(y_te_np, pred_te),
        }
    else:
        metrics = {
            "train_mse": _mse(y_tr_np, pred_tr),
            "test_mse": _mse(y_te_np, pred_te),
            "train_r2": _r2(y_tr_np, pred_tr),
            "test_r2": _r2(y_te_np, pred_te),
        }

    if walk_forward_oos:
        try:
            wf = _walk_forward_oos_metrics(
                df,
                kind=kind,
                feature_columns=feature_columns,
                target_column=target_column,
                seq_len=seq_len,
                training_seed=training_seed,
                device=dev,
                hidden=hidden,
                num_layers=num_layers,
                kernel=kernel,
                num_blocks=num_blocks,
                dropout=dropout,
                d_model=d_model,
                nhead=nhead,
                num_encoder_layers=num_encoder_layers,
                dim_feedforward=dim_feedforward,
                max_seq_len=max_seq_len,
                batch_size=batch_size,
                lr=lr,
                normalize_mode=normalize_mode,
                target_task=target_task,
                train_days=wf_train_days,
                test_days=wf_test_days,
                step_days=wf_step_days,
                epochs=wf_epochs,
                min_folds=wf_min_folds,
            )
            metrics.update(wf)
        except Exception:
            metrics["wf_folds"] = 0.0

    run_id = uuid.uuid4().hex[:12]
    bundle_dir = Path(out_root) / f"ts_{kind}_{run_id}"
    bundle_dir.mkdir(parents=True, exist_ok=True)

    save_torch_state(
        model,
        bundle_dir,
        extra_state={
            "arch": kind,
            "n_features": n_feat,
            "seq_len": seq_len,
            "hidden": hidden,
            "num_layers": num_layers,
            "kernel": kernel,
            "num_blocks": num_blocks,
            "dropout": dropout,
            "normalize_mode": normalize_mode,
            "d_model": d_model,
            "nhead": nhead,
            "num_encoder_layers": num_encoder_layers,
            "dim_feedforward": dim_feedforward,
            "max_seq_len": max(seq_len, max_seq_len),
            "target_task": str(target_task),
        },
    )

    inf = InferenceConfig(
        feature_columns=list(feature_columns),
        target_column=target_column,
        seq_len=seq_len,
        normalize=normalize_mode,
        extra={
            "arch": kind,
            "test_size": test_size,
            "epochs": epochs,
            "normalize_mode": normalize_mode,
            "target_task": str(target_task),
            "time_val_split": bool(time_val_split),
            "walk_forward_oos": bool(walk_forward_oos),
        },
    )
    save_inference_config(bundle_dir, inf)

    meta = BundleMetadata(
        model_version=model_version,
        feature_version=feature_version,
        model_type=f"timeseries_{kind}",
        backend="torch",
        training_seed=training_seed,
        data_slice_hash=data_slice_hash,
        content_hash=content_hash,
        created_at=utc_now_iso(),
        metrics=metrics,
        params={
            "kind": kind,
            "seq_len": seq_len,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "hidden": hidden,
            "test_size": test_size,
            "normalize_mode": normalize_mode,
            "time_val_split": time_val_split,
            "target_task": str(target_task),
            "walk_forward_oos": bool(walk_forward_oos),
            "wf_train_days": int(wf_train_days),
            "wf_test_days": int(wf_test_days),
            "wf_step_days": int(wf_step_days),
            "wf_epochs": int(wf_epochs),
            "wf_min_folds": int(wf_min_folds),
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

    return TimeseriesTrainResult(
        bundle_dir=bundle_dir,
        metrics=metrics,
        data_slice_hash=data_slice_hash,
        content_hash=content_hash,
    )


def load_timeseries_bundle(
    bundle_dir: Union[str, Path],
    *,
    map_location: Optional[str] = None,
) -> Tuple[nn.Module, InferenceConfig, BundleMetadata, dict[str, Any]]:
    root = Path(bundle_dir)
    meta = load_bundle_metadata(root)
    inf = load_inference_config(root / "inference_config.json")
    payload = load_torch_state(root / "model.pt", map_location=map_location)
    extra = payload.get("extra") or {}
    kind = str(extra.get("arch", "lstm"))
    n_features = int(extra["n_features"])
    seq_len_stored = int(extra.get("seq_len", inf.seq_len or 20))
    model = build_timeseries_model(
        kind,  # type: ignore[arg-type]
        n_features,
        hidden=int(extra.get("hidden", 64)),
        num_layers=int(extra.get("num_layers", 2)),
        kernel=int(extra.get("kernel", 3)),
        num_blocks=int(extra.get("num_blocks", 3)),
        dropout=float(extra.get("dropout", 0.1)),
        d_model=int(extra.get("d_model", 64)),
        nhead=int(extra.get("nhead", 4)),
        num_encoder_layers=int(extra.get("num_encoder_layers", 2)),
        dim_feedforward=int(extra.get("dim_feedforward", 128)),
        max_seq_len=int(extra.get("max_seq_len", max(seq_len_stored, 128))),
    )
    model.load_state_dict(payload["state_dict"], strict=False)
    return model, inf, meta, extra
