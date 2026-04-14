"""训练与推理解耦：保存模型版本、特征版本、推理配置与元数据。"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Union

import numpy as np

try:
    import joblib
except ImportError:  # pragma: no cover
    joblib = None  # type: ignore

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore


ModelBackend = Literal["sklearn", "torch"]


@dataclass
class InferenceConfig:
    """推理侧只依赖该配置 + 已训练权重，不依赖训练脚本。"""

    feature_columns: list[str]
    target_column: str
    symbol_col: str = "symbol"
    date_col: str = "trade_date"
    # baseline: 无序列；timeseries: 序列长度与特征维
    seq_len: Optional[int] = None
    normalize: str = "none"  # none | standard（按训练集统计量，存于 artifacts）
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json_dict(cls, d: dict[str, Any]) -> "InferenceConfig":
        return cls(
            feature_columns=list(d["feature_columns"]),
            target_column=str(d["target_column"]),
            symbol_col=str(d.get("symbol_col", "symbol")),
            date_col=str(d.get("date_col", "trade_date")),
            seq_len=d.get("seq_len"),
            normalize=str(d.get("normalize", "none")),
            extra=dict(d.get("extra") or {}),
        )


@dataclass
class BundleMetadata:
    model_version: str
    feature_version: str
    model_type: str
    backend: ModelBackend
    training_seed: int
    data_slice_hash: str
    content_hash: str
    created_at: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    params: Dict[str, Any] = field(default_factory=dict)

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json_dict(cls, d: dict[str, Any]) -> "BundleMetadata":
        return cls(
            model_version=str(d["model_version"]),
            feature_version=str(d["feature_version"]),
            model_type=str(d["model_type"]),
            backend=d["backend"],  # type: ignore[arg-type]
            training_seed=int(d["training_seed"]),
            data_slice_hash=str(d["data_slice_hash"]),
            content_hash=str(d["content_hash"]),
            created_at=str(d["created_at"]),
            metrics=dict(d.get("metrics") or {}),
            params=dict(d.get("params") or {}),
        )


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def utc_now_iso() -> str:
    """UTC ISO8601 时间戳（用于 bundle 元数据与实验记录）。"""
    return _utc_now_iso()


def ensure_bundle_dir(path: Union[str, Path]) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_inference_config(
    out_dir: Union[str, Path],
    cfg: InferenceConfig,
    *,
    filename: str = "inference_config.json",
) -> Path:
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    fp = d / filename
    fp.write_text(json.dumps(cfg.to_json_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return fp


def load_inference_config(path: Union[str, Path]) -> InferenceConfig:
    p = Path(path)
    raw = json.loads(p.read_text(encoding="utf-8"))
    return InferenceConfig.from_json_dict(raw)


def save_bundle_metadata(
    out_dir: Union[str, Path],
    meta: BundleMetadata,
    *,
    filename: str = "bundle.json",
) -> Path:
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    fp = d / filename
    fp.write_text(json.dumps(meta.to_json_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return fp


def load_bundle_metadata(out_dir: Union[str, Path], *, filename: str = "bundle.json") -> BundleMetadata:
    fp = Path(out_dir) / filename
    raw = json.loads(fp.read_text(encoding="utf-8"))
    return BundleMetadata.from_json_dict(raw)


def save_sklearn_model(
    estimator: Any,
    out_dir: Union[str, Path],
    *,
    filename: str = "model.joblib",
) -> Path:
    if joblib is None:
        raise RuntimeError("需要 joblib（随 scikit-learn 安装）")
    d = ensure_bundle_dir(out_dir)
    fp = d / filename
    joblib.dump(estimator, fp)
    return fp


def load_sklearn_model(path: Union[str, Path]) -> Any:
    if joblib is None:
        raise RuntimeError("需要 joblib（随 scikit-learn 安装）")
    return joblib.load(Path(path))


def save_torch_state(
    module: Any,
    out_dir: Union[str, Path],
    *,
    filename: str = "model.pt",
    extra_state: Optional[dict[str, Any]] = None,
) -> Path:
    if torch is None:
        raise RuntimeError("需要 PyTorch")
    d = ensure_bundle_dir(out_dir)
    fp = d / filename
    payload: dict[str, Any] = {"state_dict": module.state_dict()}
    if extra_state:
        payload["extra"] = extra_state
    torch.save(payload, fp)
    return fp


def load_torch_state(
    path: Union[str, Path],
    map_location: Optional[str] = None,
) -> dict[str, Any]:
    if torch is None:
        raise RuntimeError("需要 PyTorch")
    p = Path(path)
    loc = map_location or "cpu"
    try:
        return torch.load(p, map_location=loc, weights_only=False)  # type: ignore[call-arg]
    except TypeError:
        return torch.load(p, map_location=loc)


def save_normalizer_stats(
    out_dir: Union[str, Path],
    mean: np.ndarray,
    scale: np.ndarray,
    *,
    filename: str = "normalizer.json",
) -> Path:
    d = ensure_bundle_dir(out_dir)
    fp = d / filename
    data = {
        "mean": mean.astype(np.float64).tolist(),
        "scale": np.maximum(scale.astype(np.float64), 1e-12).tolist(),
    }
    fp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return fp


def load_normalizer_stats(path: Union[str, Path]) -> tuple[np.ndarray, np.ndarray]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return np.asarray(raw["mean"], dtype=np.float64), np.asarray(raw["scale"], dtype=np.float64)


def write_full_bundle(
    out_dir: Union[str, Path],
    *,
    model: Any,
    inference: InferenceConfig,
    meta: BundleMetadata,
    backend: ModelBackend,
    normalizer_path: Optional[Path] = None,
) -> Path:
    """
    写出完整训练工件目录：``bundle.json``、``inference_config.json``、模型文件。
    """
    root = ensure_bundle_dir(out_dir)
    save_inference_config(root, inference)
    save_bundle_metadata(root, meta)
    if backend == "sklearn":
        save_sklearn_model(model, root)
    elif backend == "torch":
        save_torch_state(model, root)
    else:
        raise ValueError(f"未知 backend: {backend}")
    if normalizer_path is not None:
        # 已写入目标目录时仅记录
        pass
    return root


def list_bundle_dirs(
    root_dir: Union[str, Path],
    *,
    prefix: str,
) -> list[Path]:
    root = Path(root_dir)
    if not root.exists():
        return []
    out: list[Path] = []
    for p in root.iterdir():
        if p.is_dir() and p.name.startswith(prefix):
            out.append(p)
    out.sort(key=lambda x: x.stat().st_mtime)
    return out


def load_metric_from_bundle(
    bundle_dir: Union[str, Path],
    metric_key: str,
) -> float:
    meta = load_bundle_metadata(bundle_dir)
    raw = (meta.metrics or {}).get(metric_key)
    if raw is None:
        return float("nan")
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return float("nan")
    return val if np.isfinite(val) else float("nan")


def should_promote_by_quantile(
    *,
    candidate_metric: float,
    history_metrics: list[float],
    quantile: float = 0.25,
    min_history: int = 4,
) -> tuple[bool, float]:
    vals = np.asarray([v for v in history_metrics if np.isfinite(v)], dtype=np.float64)
    if not np.isfinite(candidate_metric):
        return False, float("nan")
    if len(vals) < int(min_history):
        return True, float("nan")
    qv = float(np.quantile(vals, float(quantile)))
    return bool(candidate_metric >= qv), qv


def publish_bundle_with_history(
    *,
    source_bundle_dir: Union[str, Path],
    publish_dir: Union[str, Path],
    history_root: Union[str, Path],
    keep_recent: int = 8,
) -> Dict[str, Any]:
    src = Path(source_bundle_dir)
    if not src.is_dir():
        raise ValueError(f"source_bundle_dir 不存在: {src}")
    pub = Path(publish_dir)
    hist_root = Path(history_root)
    hist_root.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    version_name = f"{stamp}_{src.name}"
    version_dir = hist_root / version_name
    shutil.copytree(src, version_dir)

    previous_name = None
    if pub.exists():
        prev_meta = hist_root / "active_bundle.json"
        if prev_meta.exists():
            try:
                previous_name = json.loads(prev_meta.read_text(encoding="utf-8")).get("active_version")
            except Exception:
                previous_name = None
        if pub.is_dir():
            shutil.rmtree(pub)
        else:
            pub.unlink()
    shutil.copytree(src, pub)

    meta = {
        "active_version": version_name,
        "previous_version": previous_name,
        "source_bundle_dir": str(src),
        "publish_dir": str(pub),
        "history_root": str(hist_root),
    }
    (hist_root / "active_bundle.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    versions = sorted([p for p in hist_root.iterdir() if p.is_dir()], key=lambda x: x.name)
    if keep_recent > 0 and len(versions) > keep_recent:
        for p in versions[: len(versions) - keep_recent]:
            shutil.rmtree(p, ignore_errors=True)
    return meta
