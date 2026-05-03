"""Markdown 报告生成：共享的格式化、序列化与文档构建工具。

从 scripts/run_monthly_selection_baselines.py 和
scripts/run_monthly_selection_dataset.py 中提取。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ── 项目根（用于相对路径格式化） ──────────────────────────────────────────

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def set_project_root(root: str | Path) -> None:
    """允许外部设置项目根路径。"""
    global _PROJECT_ROOT
    _PROJECT_ROOT = Path(root)


def project_relative(path: str | Path) -> str:
    """将绝对路径转为相对于项目根的字符串。"""
    p = Path(path)
    try:
        return str(p.resolve().relative_to(_PROJECT_ROOT))
    except ValueError:
        return str(p)


# ── JSON 安全序列化 ──────────────────────────────────────────────────────

def json_sanitize(obj: Any) -> Any:
    """递归转换 numpy/pandas/Path 类型为 JSON 兼容值。"""
    if isinstance(obj, dict):
        return {str(k): json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, set)):
        return [json_sanitize(v) for v in obj]
    if isinstance(obj, tuple):
        return [json_sanitize(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        val = float(obj)
        return val if np.isfinite(val) else None
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    if obj is None:
        return None
    if isinstance(obj, (str, int, bool)):
        return obj
    return str(obj)


# ── Markdown 表格格式化 ──────────────────────────────────────────────────

def _markdown_cell(value: object) -> str:
    """将单个值转为 Markdown 表格单元格文本。"""
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value).replace("|", "\\|").replace("\n", "<br>")


def format_markdown_table(df: pd.DataFrame, *, max_rows: int = 30) -> str:
    """将 DataFrame 转为 Markdown 表格字符串。"""
    if df.empty:
        return "_无记录_"
    view = df.head(max_rows).copy()
    cols = [str(c) for c in view.columns]
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    rows = [
        "| " + " | ".join(_markdown_cell(row[col]) for col in view.columns) + " |"
        for _, row in view.iterrows()
    ]
    suffix = (
        [f"\n_仅展示前 {max_rows} 行，共 {len(df)} 行。_"]
        if len(df) > max_rows
        else []
    )
    return "\n".join([header, sep, *rows, *suffix])


# ── 数值格式化 ───────────────────────────────────────────────────────────

def fmt_pct(v: float, d: int = 2) -> str:
    """将小数转为百分比字符串。"""
    if not np.isfinite(v):
        return "N/A"
    return f"{v * 100:+.{d}f}%"


def fmt_num(v: float, d: int = 3) -> str:
    """将数值转为带符号的字符串。"""
    if not np.isfinite(v):
        return "N/A"
    return f"{v:+.{d}f}"


# ── 研究文档构建（通用模板） ──────────────────────────────────────────────

def build_research_doc(
    *,
    title: str,
    sections: dict[str, str],
    generated_at: str | None = None,
) -> str:
    """通用研究文档构建器，按 sections 顺序拼接 Markdown。"""
    ts = generated_at or pd.Timestamp.utcnow().isoformat()
    lines = [f"# {title}", "", f"- 生成时间：`{ts}`", ""]
    for heading, body in sections.items():
        lines.append(f"## {heading}")
        lines.append("")
        lines.append(body)
        lines.append("")
    return "\n".join(lines)
