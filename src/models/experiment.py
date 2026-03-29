"""最小实验追踪：本地 CSV + JSONL（参数、指标、时间）。"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Union


def _utc_ts() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def ensure_experiment_dir(base: Union[str, Path]) -> Path:
    p = Path(base)
    p.mkdir(parents=True, exist_ok=True)
    return p


def append_experiment_jsonl(
    base_dir: Union[str, Path],
    record: Dict[str, Any],
    *,
    filename: str = "experiments.jsonl",
) -> Path:
    """追加一行 JSON（每行一条实验）。"""
    d = ensure_experiment_dir(base_dir)
    fp = d / filename
    line = json.dumps(record, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    with fp.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    return fp


def append_experiment_csv(
    base_dir: Union[str, Path],
    row: Dict[str, Any],
    *,
    filename: str = "experiments.csv",
    fieldnames: Optional[list[str]] = None,
) -> Path:
    """追加 CSV 行；若文件不存在则写表头。"""
    d = ensure_experiment_dir(base_dir)
    fp = d / filename
    defaults = [
        "timestamp",
        "run_id",
        "model_type",
        "duration_sec",
        "seed",
        "data_slice_hash",
        "content_hash",
        "params_json",
        "metrics_json",
        "bundle_dir",
    ]
    keys = fieldnames if fieldnames is not None else defaults
    if not fp.exists():
        with fp.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
    with fp.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writerow({k: row.get(k, "") for k in keys})
    return fp


def build_experiment_record(
    *,
    run_id: str,
    model_type: str,
    duration_sec: float,
    seed: int,
    data_slice_hash: str,
    content_hash: str,
    params: Dict[str, Any],
    metrics: Dict[str, Any],
    bundle_dir: Union[str, Path],
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    rec: Dict[str, Any] = {
        "timestamp": _utc_ts(),
        "run_id": run_id,
        "model_type": model_type,
        "duration_sec": round(float(duration_sec), 6),
        "seed": seed,
        "data_slice_hash": data_slice_hash,
        "content_hash": content_hash,
        "params_json": json.dumps(params, ensure_ascii=False, sort_keys=True),
        "metrics_json": json.dumps(metrics, ensure_ascii=False, sort_keys=True),
        "bundle_dir": str(bundle_dir),
    }
    if extra:
        rec.update(extra)
    return rec
