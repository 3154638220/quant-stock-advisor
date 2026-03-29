"""数据切片指纹：同一 DataFrame 切片在相同语义下得到稳定哈希，便于复现实验。"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable, Optional, Sequence, Union

import numpy as np
import pandas as pd


def normalize_slice_spec(
    *,
    symbols: Optional[Sequence[str]] = None,
    date_start: Optional[Union[str, pd.Timestamp]] = None,
    date_end: Optional[Union[str, pd.Timestamp]] = None,
    extra: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """构造可 JSON 序列化、键有序的切片描述。"""
    spec: dict[str, Any] = {}
    if symbols is not None:
        spec["symbols"] = sorted(str(s) for s in symbols)
    if date_start is not None:
        spec["date_start"] = str(pd.Timestamp(date_start).normalize())
    if date_end is not None:
        spec["date_end"] = str(pd.Timestamp(date_end).normalize())
    if extra:
        spec["extra"] = dict(sorted(extra.items()))
    return spec


def hash_slice_spec(spec: dict[str, Any]) -> str:
    """对切片描述做 SHA256（十六进制）。"""
    payload = json.dumps(spec, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def hash_dataframe_content(
    df: pd.DataFrame,
    *,
    columns: Optional[Sequence[str]] = None,
) -> str:
    """
    对选定列的逐行内容做稳定哈希（用于「同一数据切片」强一致验收）。

    使用 pandas 的 hash_pandas_object（含 index），避免浮点打印顺序问题。
    """
    if df.empty:
        return hashlib.sha256(b"empty").hexdigest()
    cols = list(columns) if columns is not None else list(df.columns)
    sub = df[cols]
    h = pd.util.hash_pandas_object(sub, index=True)
    return hashlib.sha256(h.values.tobytes()).hexdigest()


def combined_data_fingerprint(
    df: pd.DataFrame,
    *,
    slice_spec: Optional[dict[str, Any]] = None,
    content_columns: Optional[Sequence[str]] = None,
) -> dict[str, str]:
    """
    返回 ``slice_hash``（若给定 spec）与 ``content_hash``（数据内容）。
    """
    out: dict[str, str] = {
        "content_hash": hash_dataframe_content(df, columns=content_columns),
        "n_rows": str(len(df)),
    }
    if slice_spec is not None:
        out["slice_hash"] = hash_slice_spec(slice_spec)
    return out


def apply_time_symbol_filter(
    df: pd.DataFrame,
    *,
    symbol_col: str = "symbol",
    date_col: str = "trade_date",
    symbols: Optional[Iterable[str]] = None,
    date_start: Optional[Union[str, pd.Timestamp]] = None,
    date_end: Optional[Union[str, pd.Timestamp]] = None,
) -> pd.DataFrame:
    """按标的与时间区间过滤；列存在时生效。"""
    out = df.copy()
    if symbol_col in out.columns and symbols is not None:
        sym_set = set()
        for s in symbols:
            ss = str(s)
            sym_set.add(ss.zfill(6) if ss.isdigit() and len(ss) <= 6 else ss)
        out = out[out[symbol_col].astype(str).isin(sym_set)]
    if date_col in out.columns:
        out[date_col] = pd.to_datetime(out[date_col]).dt.normalize()
        if date_start is not None:
            out = out[out[date_col] >= pd.Timestamp(date_start).normalize()]
        if date_end is not None:
            out = out[out[date_col] <= pd.Timestamp(date_end).normalize()]
    return out.sort_values([symbol_col, date_col] if symbol_col in out.columns else [date_col]).reset_index(drop=True)


def seed_everything(seed: int) -> None:
    """统一随机种子（Python / NumPy；Torch 在各自 trainer 中设置）。"""
    import random

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
