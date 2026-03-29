"""本地结果落盘（CSV/Parquet），无云端推送。"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Optional, Union

import pandas as pd


def ensure_dir(path: Union[str, Path]) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def find_latest_recommendation_csv(
    results_dir: Union[str, Path],
    *,
    prefix: str = "recommend",
) -> Path:
    """
    返回 ``results_dir`` 下按文件名排序的最后一份 ``{prefix}_YYYY-MM-DD.csv``。
    """
    d = Path(results_dir)
    files = sorted(d.glob(f"{prefix}_*.csv"))
    if not files:
        raise FileNotFoundError(f"目录中无 {prefix}_*.csv: {d}")
    return files[-1]


def save_recommendation_csv(
    df: pd.DataFrame,
    *,
    results_dir: Union[str, Path],
    asof: Optional[Union[date, datetime, str]] = None,
    prefix: str = "recommend",
) -> Path:
    """
    将推荐池写入 ``{results_dir}/{prefix}_YYYY-MM-DD.csv``。

    Parameters
    ----------
    df : DataFrame
        至少含排名与代码等列，由上游流水线组装。
    asof : date or str, optional
        文件名中的日期；默认当天（本地时区）。
    prefix : str
        文件名前缀。
    """
    d = asof
    if d is None:
        d = datetime.now().date()
    elif isinstance(d, datetime):
        d = d.date()
    elif isinstance(d, str):
        d = pd.Timestamp(d).date()

    out_dir = ensure_dir(results_dir)
    stem = f"{prefix}_{d.isoformat()}"
    path = out_dir / f"{stem}.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def save_recommendation_parquet(
    df: pd.DataFrame,
    *,
    results_dir: Union[str, Path],
    asof: Optional[Union[date, datetime, str]] = None,
    prefix: str = "recommend",
) -> Path:
    """与 ``save_recommendation_csv`` 同名规则，扩展名为 ``.parquet``。"""
    d = asof
    if d is None:
        d = datetime.now().date()
    elif isinstance(d, datetime):
        d = d.date()
    elif isinstance(d, str):
        d = pd.Timestamp(d).date()

    out_dir = ensure_dir(results_dir)
    stem = f"{prefix}_{d.isoformat()}"
    path = out_dir / f"{stem}.parquet"
    df.to_parquet(path, index=False)
    return path
