#!/usr/bin/env python3
"""
从 recommend_*.csv 生成本地 HTML 报告：热力图（截面 z）、条形贡献、中文推荐理由。

用法（项目根目录，conda 环境 quant-system）::

  python scripts/recommend_report.py --csv data/results/recommend_2026-03-27.csv
  python scripts/recommend_report.py --latest
"""

from __future__ import annotations

import argparse
import html
import math
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.recommend_explain import (
    FACTOR_LABELS,
    build_recommend_reason_column,
    normalize_composite_weights,
)
from src.notify import find_latest_recommendation_csv


def _z_heatmap_color(z: float, zmin: float = -2.5, zmax: float = 2.5) -> str:
    """z 分映射到蓝-白-红背景（低-中-高）。"""
    if not math.isfinite(z):
        return "#f0f0f0"
    t = (z - zmin) / (zmax - zmin) if zmax > zmin else 0.5
    t = max(0.0, min(1.0, t))
    # 蓝 rgb(59,130,246) -> 白 -> 红 rgb(239,68,68)
    if t < 0.5:
        u = t * 2
        r = int(59 + (255 - 59) * u)
        g = int(130 + (255 - 130) * u)
        b = int(246 + (255 - 246) * u)
    else:
        u = (t - 0.5) * 2
        r = int(255 + (239 - 255) * u)
        g = int(255 + (68 - 255) * u)
        b = int(255 + (68 - 255) * u)
    return f"rgb({r},{g},{b})"


def _load_config(path: Optional[Path]) -> dict:
    p = path or (ROOT / "config.yaml")
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_html(
    df: pd.DataFrame,
    *,
    sort_by: str,
    rsi_mode: str,
    comp_ext: dict,
    w_mom: float,
    w_rsi: float,
    top_detail: int,
    title: str,
) -> str:
    sym_col = "symbol" if "symbol" in df.columns else "代码"
    z_cols = [c for c in df.columns if str(c).startswith("z_")]
    z_cols.sort()

    # 缺失 recommend_reason 时补算
    if "recommend_reason" not in df.columns or df["recommend_reason"].isna().all():
        df = df.copy()
        df["recommend_reason"] = build_recommend_reason_column(
            df,
            sort_by=sort_by,
            rsi_mode=rsi_mode,
            composite_extended_weights=comp_ext if sort_by == "composite_extended" else None,
            w_momentum=w_mom,
            w_rsi=w_rsi,
        )

    asof = ""
    if "asof_trade_date" in df.columns and len(df):
        asof = str(df["asof_trade_date"].iloc[0])

    rows_html = []
    for i in range(min(top_detail, len(df))):
        row = df.iloc[i]
        sym = html.escape(str(row.get(sym_col, "")))
        rk = int(row["rank"]) if "rank" in row and pd.notna(row["rank"]) else i + 1
        reason = html.escape(str(row.get("recommend_reason", "")))

        contrib_rows = ""
        bar_html = ""
        if sort_by == "composite_extended" and comp_ext:
            wn = normalize_composite_weights(comp_ext)
            bars = []
            for name, wc in sorted(wn.items(), key=lambda x: -abs(x[1])):
                zc = f"z_{name}"
                if zc not in row.index:
                    continue
                zv = row[zc]
                if zv is None or pd.isna(zv):
                    continue
                c = wc * float(zv)
                lbl = FACTOR_LABELS.get(name, name)
                w_pct = abs(wc) * 100
                contrib_rows += (
                    f"<tr><td>{html.escape(lbl)}</td>"
                    f"<td>{float(zv):.3f}</td>"
                    f"<td>{w_pct:.1f}%</td>"
                    f"<td>{c:+.4f}</td></tr>"
                )
                # 条形：贡献绝对值归一化到 0–100%
                bars.append((lbl, c))
            max_abs = max((abs(b[1]) for b in bars), default=1.0) or 1.0
            for lbl, c in sorted(bars, key=lambda x: -abs(x[1]))[:8]:
                pct = min(100.0, abs(c) / max_abs * 100.0)
                color = "#2563eb" if c >= 0 else "#dc2626"
                bar_html += (
                    f'<div class="barline"><span class="blabel">{html.escape(lbl)}</span>'
                    f'<span class="bbar"><i style="width:{pct:.1f}%;background:{color}"></i></span>'
                    f'<span class="bval">{c:+.4f}</span></div>'
                )
        else:
            if "z_momentum" in row.index and pd.notna(row.get("z_momentum")):
                zm = float(row["z_momentum"])
                pct = min(100.0, abs(zm) / 3.0 * 100.0)
                bar_html += (
                    f'<div class="barline"><span class="blabel">动量 z</span>'
                    f'<span class="bbar"><i style="width:{pct:.1f}%;background:#2563eb"></i></span>'
                    f'<span class="bval">{zm:+.3f}</span></div>'
                )
            if "z_rsi" in row.index and pd.notna(row.get("z_rsi")):
                zr = float(row["z_rsi"])
                pct = min(100.0, abs(zr) / 3.0 * 100.0)
                bar_html += (
                    f'<div class="barline"><span class="blabel">RSI z</span>'
                    f'<span class="bbar"><i style="width:{pct:.1f}%;background:#7c3aed"></i></span>'
                    f'<span class="bval">{zr:+.3f}</span></div>'
                )

        heat_html = ""
        if z_cols:
            heat_html = "<table class='heat'><thead><tr><th>因子</th><th>z</th></tr></thead><tbody>"
            for zc in z_cols:
                zv = row[zc]
                if zv is None or pd.isna(zv):
                    continue
                zf = float(zv)
                name = zc[2:]
                lbl = FACTOR_LABELS.get(name, name)
                bg = _z_heatmap_color(zf)
                heat_html += (
                    f"<tr><td>{html.escape(lbl)}</td>"
                    f"<td style='background:{bg};font-weight:600'>{zf:.3f}</td></tr>"
                )
            heat_html += "</tbody></table>"

        block = f"""
        <article class="card">
          <header><span class="rank">#{rk}</span> <span class="sym">{sym}</span></header>
          <p class="reason">{reason}</p>
          <div class="viz">{bar_html}</div>
          {heat_html}
          {"<table class='contrib'><thead><tr><th>因子</th><th>z</th><th>|权重|</th><th>w×z</th></tr></thead><tbody>" + contrib_rows + "</tbody></table>" if contrib_rows else ""}
        </article>
        """
        rows_html.append(block)

    # 全表热力：前 top_detail 行 × z 列
    matrix_html = ""
    if z_cols and len(df):
        sub = df.head(top_detail)
        matrix_html = "<h2>截面 z 分一览（前若干标的）</h2><table class='matrix'><thead><tr><th>排名</th><th>代码</th>"
        for zc in z_cols:
            name = zc[2:]
            matrix_html += f"<th>{html.escape(FACTOR_LABELS.get(name, name))}</th>"
        matrix_html += "</tr></thead><tbody>"
        for i in range(len(sub)):
            row = sub.iloc[i]
            rk = int(row["rank"]) if "rank" in row and pd.notna(row["rank"]) else i + 1
            sym = html.escape(str(row.get(sym_col, "")))
            matrix_html += f"<tr><td>{rk}</td><td>{sym}</td>"
            for zc in z_cols:
                zv = row[zc]
                if zv is None or pd.isna(zv):
                    matrix_html += "<td>—</td>"
                else:
                    zf = float(zv)
                    bg = _z_heatmap_color(zf)
                    matrix_html += f"<td style='background:{bg}'>{zf:.2f}</td>"
            matrix_html += "</tr>"
        matrix_html += "</tbody></table>"

    styles = """
    :root { font-family: "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif; color: #1e293b; }
    body { max-width: 960px; margin: 2rem auto; padding: 0 1rem; background: #f8fafc; }
    h1 { font-size: 1.35rem; font-weight: 700; }
    .meta { color: #64748b; font-size: 0.9rem; margin-bottom: 1.5rem; }
    .card { background: #fff; border-radius: 12px; padding: 1rem 1.25rem; margin-bottom: 1rem;
            box-shadow: 0 1px 3px rgba(0,0,0,.08); }
    .card header { font-weight: 600; margin-bottom: 0.5rem; }
    .rank { color: #64748b; margin-right: 0.5rem; }
    .sym { font-size: 1.1rem; }
    .reason { font-size: 0.92rem; line-height: 1.55; margin: 0.6rem 0; }
    .barline { display: flex; align-items: center; gap: 8px; margin: 4px 0; font-size: 0.85rem; }
    .blabel { width: 7rem; flex-shrink: 0; }
    .bbar { flex: 1; height: 10px; background: #e2e8f0; border-radius: 4px; overflow: hidden; }
    .bbar i { display: block; height: 100%; border-radius: 4px; }
    .bval { width: 4.5rem; text-align: right; font-variant-numeric: tabular-nums; }
    .heat { font-size: 0.85rem; margin-top: 0.75rem; border-collapse: collapse; }
    .heat th, .heat td { padding: 4px 8px; border-bottom: 1px solid #e2e8f0; }
    .contrib { width: 100%; font-size: 0.8rem; margin-top: 0.5rem; border-collapse: collapse; }
    .contrib th, .contrib td { padding: 4px 6px; border: 1px solid #e2e8f0; }
    .matrix { width: 100%; border-collapse: collapse; font-size: 0.78rem; margin: 1rem 0; }
    .matrix th, .matrix td { padding: 6px 8px; border: 1px solid #e2e8f0; text-align: center; }
    .matrix th { background: #f1f5f9; position: sticky; top: 0; }
    """

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{html.escape(title)}</title>
<style>{styles}</style>
</head>
<body>
<h1>{html.escape(title)}</h1>
<p class="meta">基准日 {html.escape(asof)} · 排序方式 <code>{html.escape(sort_by)}</code></p>
{matrix_html}
<h2>逐只说明（前 {top_detail} 只）</h2>
{"".join(rows_html)}
<p class="meta">蓝→白→红表示 z 分由低到高；条形为加权贡献或单因子 z 的相对幅度。</p>
</body>
</html>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="荐股 CSV 可视化 HTML 报告")
    parser.add_argument("--csv", type=Path, default=None, help="recommend_*.csv 路径")
    parser.add_argument("--latest", action="store_true", help="使用 results_dir 下最新 recommend_*.csv")
    parser.add_argument("--config", type=Path, default=None, help="默认项目根 config.yaml")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="输出 HTML 路径；默认与 CSV 同目录同名 .html",
    )
    parser.add_argument("--top", type=int, default=20, help="逐只说明与矩阵行数上限")
    args = parser.parse_args()

    cfg = _load_config(args.config)
    paths = cfg.get("paths", {}) or {}
    results_dir = paths.get("results_dir", "data/results")
    if not Path(results_dir).is_absolute():
        results_dir = ROOT / results_dir

    if args.latest:
        csv_path = find_latest_recommendation_csv(results_dir)
    elif args.csv:
        csv_path = Path(args.csv)
        if not csv_path.is_absolute():
            csv_path = ROOT / csv_path
    else:
        parser.error("请指定 --csv 或 --latest")

    if not csv_path.is_file():
        print(f"文件不存在: {csv_path}", file=sys.stderr)
        return 1

    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    sig = cfg.get("signals", {}) or {}
    comp = sig.get("composite") or {}
    sort_by = str(df["sort_by"].iloc[0]).lower() if "sort_by" in df.columns and len(df) else str(
        sig.get("sort_by", "xgboost")
    ).lower()
    rsi_mode = str(comp.get("rsi_mode", "level")).lower()
    comp_ext = sig.get("composite_extended") or {}
    w_mom = float(comp.get("w_momentum", 0.65))
    w_rsi = float(comp.get("w_rsi", 0.35))

    out_path = args.out
    if out_path is None:
        out_path = csv_path.with_suffix(".html")
    else:
        out_path = Path(out_path)
        if not out_path.is_absolute():
            out_path = ROOT / out_path

    title = f"荐股可视化 · {csv_path.name}"
    html_doc = build_html(
        df,
        sort_by=sort_by,
        rsi_mode=rsi_mode,
        comp_ext=comp_ext,
        w_mom=w_mom,
        w_rsi=w_rsi,
        top_detail=int(args.top),
        title=title,
    )
    out_path.write_text(html_doc, encoding="utf-8")
    print(f"已写入: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
