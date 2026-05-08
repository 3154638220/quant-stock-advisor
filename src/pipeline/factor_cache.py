"""因子缓存管理模块。

提供 Parquet 序列化 + JSON 元数据 sidecar 的因子缓存读写，
支持 schema 版本校验以自动判定缓存失效。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.pipeline.factor_computer import PREPARED_FACTORS_SCHEMA_VERSION, PREPARED_FACTORS_REQUIRED_COLUMNS


def _json_sanitize(obj: Any) -> Any:
    from src.reporting.markdown_report import json_sanitize as _js
    return _js(obj)


def _factor_cache_meta_path(cache_path: Path) -> Path:
    return cache_path.with_suffix(cache_path.suffix + ".meta.json")


def _prepared_factors_cache_expected_meta(
    *,
    start_date: str,
    end_date: str,
    lookback_days: int,
    min_hist_days: int,
    db_path: str,
    results_dir: str,
    universe_filter_cfg: dict[str, Any],
) -> dict[str, Any]:
    return {
        "start_date": str(start_date),
        "end_date": str(end_date),
        "lookback_days": int(lookback_days),
        "min_hist_days": int(min_hist_days),
        "db_path": str(db_path),
        "results_dir": str(results_dir),
        "universe_filter_cfg": _json_sanitize(universe_filter_cfg),
        "cache_format_version": 2,
        "prepared_factors_schema_version": PREPARED_FACTORS_SCHEMA_VERSION,
        "required_columns": list(PREPARED_FACTORS_REQUIRED_COLUMNS),
    }


def load_prepared_factors_cache(cache_path: Path, expected_meta: dict[str, Any]) -> pd.DataFrame | None:
    meta_path = _factor_cache_meta_path(cache_path)
    if not cache_path.exists() or not meta_path.exists():
        return None
    try:
        actual_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if actual_meta != expected_meta:
        return None
    try:
        cached = pd.read_parquet(cache_path)
    except Exception:
        return None
    required_columns = {str(col) for col in expected_meta.get("required_columns", []) if str(col).strip()}
    if required_columns:
        cached_columns = {str(col) for col in cached.columns}
        if not required_columns.issubset(cached_columns):
            return None
    if "trade_date" in cached.columns:
        cached["trade_date"] = pd.to_datetime(cached["trade_date"], errors="coerce")
    if "symbol" in cached.columns:
        cached["symbol"] = cached["symbol"].astype(str).str.zfill(6)
    return cached


def write_prepared_factors_cache(cache_path: Path, factors: pd.DataFrame, meta: dict[str, Any]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    factors.to_parquet(cache_path, index=False)
    _factor_cache_meta_path(cache_path).write_text(
        json.dumps(_json_sanitize(meta), ensure_ascii=False, indent=2), encoding="utf-8",
    )
