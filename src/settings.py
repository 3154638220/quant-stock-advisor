"""项目根路径与全局配置加载。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import yaml


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    path = config_path or project_root() / "config.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_asof_trade_end(paths: Optional[Dict[str, Any]] = None) -> pd.Timestamp:
    """
    全市场日线截面使用的统一交易日上界。

    ``paths["asof_trade_date"]`` 为非空字符串时解析（如 ``2026-03-27``），
    否则为运行当日 ``normalize()``。
    """
    paths = paths or {}
    raw = paths.get("asof_trade_date")
    if raw is None:
        return pd.Timestamp.now().normalize()
    s = str(raw).strip()
    if not s:
        return pd.Timestamp.now().normalize()
    return pd.Timestamp(s).normalize()


def has_explicit_asof_trade_date(paths: Optional[Dict[str, Any]] = None) -> bool:
    """是否配置了非空的 ``asof_trade_date``（用于输出标签与 CLI 覆盖判断）。"""
    paths = paths or {}
    raw = paths.get("asof_trade_date")
    return bool(raw is not None and str(raw).strip())
