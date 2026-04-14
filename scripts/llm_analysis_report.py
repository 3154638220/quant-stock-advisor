#!/usr/bin/env python3
"""
将 llm_attention_*.json 转为可读 Markdown 报告 + 自包含 HTML。

用法（项目根目录）::

  python scripts/llm_analysis_report.py --latest
  python scripts/llm_analysis_report.py --json data/results/llm_attention_2026-03-27.json
  python scripts/llm_analysis_report.py --json ... --no-html
  python scripts/llm_analysis_report.py --json ... --no-md
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


CATALYST_ZH = {
    "policy": "政策",
    "merger": "并购重组",
    "product": "产品/技术",
    "regulatory": "监管",
    "earnings": "业绩",
    "industry": "行业事件",
    "management": "管理层",
    "contract": "合同/订单",
    "other": "其他",
    "market_movement": "市场波动",
}


def _find_latest_json(results_dir: Path) -> Path | None:
    files = sorted(results_dir.glob("llm_attention_*.json"), reverse=True)
    return files[0] if files else None


def _sentiment_zh(s: str) -> str:
    return {"bullish": "偏多", "bearish": "偏空", "neutral": "中性"}.get(
        (s or "").lower(), s or "—"
    )


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------


def build_markdown(data: dict[str, Any]) -> str:
    trade_date = data.get("trade_date", "")
    model = data.get("model", "")
    gen = data.get("generated_at", "")
    macro = data.get("market_macro") or {}
    stocks = data.get("attention_stocks") or []
    news_driven = [s for s in stocks if s.get("is_news_driven")]
    total = data.get("total_scanned", len(stocks))

    lines: list[str] = [
        f"# 新闻驱动关注度扫描 · {trade_date}",
        "",
        f"- **模型**：{model}",
        f"- **生成时间**：{gen}",
        f"- **扫描量**：飙升榜 {total} 只 → 新闻驱动 {len(news_driven)} 只",
        "",
    ]

    if macro and macro.get("summary"):
        ms = macro.get("market_sentiment", "neutral")
        lines.extend(
            [
                "## 市场整体",
                "",
                f"- **情绪**：{_sentiment_zh(str(ms))}（score={macro.get('market_score', 0)}）",
                f"- **摘要**：{macro.get('summary', '—')}",
            ]
        )
        tb = macro.get("top_bullish") or []
        td = macro.get("top_bearish") or []
        if tb:
            lines.append(f"- **利好**：{'；'.join(str(x) for x in tb[:5])}")
        if td:
            lines.append(f"- **利空**：{'；'.join(str(x) for x in td[:5])}")
        hs = macro.get("hot_sectors") or []
        if hs:
            lines.append(f"- **热点板块**：{' / '.join(str(x) for x in hs[:6])}")
        lines.append("")

    # 新闻驱动股票一览
    lines.extend(
        [
            "## 新闻驱动关注度飙升股票",
            "",
            "| 代码 | 名称 | 显著性 | 类别 | 催化剂 | 排名变化 | 涨跌幅 |",
            "| --- | --- | ---: | --- | --- | ---: | ---: |",
        ]
    )

    for s in sorted(news_driven, key=lambda x: x.get("significance", 0), reverse=True):
        sym = s.get("symbol", "")
        name = s.get("name", "")
        sig = s.get("significance", 0)
        cat = CATALYST_ZH.get(s.get("catalyst_category", ""), s.get("catalyst_category", ""))
        catalyst = str(s.get("news_catalyst", "—")).replace("|", "\\|")
        rk_chg = s.get("attention_rank_change", 0)
        pct = s.get("price_change_pct", 0)
        lines.append(
            f"| {sym} | {name} | {sig} | {cat} | {catalyst} | {rk_chg:+d} | {pct:+.2f}% |"
        )

    # 详情
    lines.extend(["", "## 详情", ""])
    for s in sorted(news_driven, key=lambda x: x.get("significance", 0), reverse=True):
        sym = s.get("symbol", "")
        name = s.get("name", "")
        lines.append(f"### {sym} {name}")
        lines.append("")
        lines.append(
            f"- **催化剂**：{s.get('news_catalyst', '—')}"
        )
        lines.append(
            f"- **类别**：{CATALYST_ZH.get(s.get('catalyst_category', ''), '—')} "
            f"| **显著性**：{s.get('significance', 0)}/10"
        )
        lines.append(f"- **摘要**：{s.get('news_summary', '—')}")
        lines.append(
            f"- **排名**：第{s.get('attention_rank', '?')}名 "
            f"（变化 {s.get('attention_rank_change', 0):+d}） "
            f"| 涨跌幅 {s.get('price_change_pct', 0):+.2f}%"
        )
        headlines = s.get("key_headlines") or []
        if headlines:
            lines.append("- **主要新闻**：")
            for h in headlines[:5]:
                lines.append(f"  - {h}")
        lines.append("")

    # 被过滤掉的（非新闻驱动）简要列出
    non_driven = [s for s in stocks if not s.get("is_news_driven")]
    if non_driven:
        lines.extend(["## 市场波动驱动（已过滤）", ""])
        for s in non_driven[:20]:
            lines.append(
                f"- {s.get('symbol', '')} {s.get('name', '')} — "
                f"{s.get('news_summary', '—')}"
            )
        if len(non_driven) > 20:
            lines.append(f"- *…及其他 {len(non_driven) - 20} 只*")
        lines.append("")

    lines.extend(
        [
            "---",
            "*由 `scripts/llm_analysis_report.py` 自 `llm_attention_*.json` 生成*",
        ]
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------


def build_html(data: dict[str, Any]) -> str:
    trade_date = html.escape(str(data.get("trade_date", "")))
    model = html.escape(str(data.get("model", "")))
    gen = html.escape(str(data.get("generated_at", "")))
    macro = data.get("market_macro") or {}
    stocks = data.get("attention_stocks") or []
    news_driven = sorted(
        [s for s in stocks if s.get("is_news_driven")],
        key=lambda x: x.get("significance", 0),
        reverse=True,
    )
    non_driven = [s for s in stocks if not s.get("is_news_driven")]
    total = data.get("total_scanned", len(stocks))

    ms = str(macro.get("market_sentiment", "neutral"))

    # Chart data
    categories: dict[str, int] = {}
    for s in news_driven:
        cat = CATALYST_ZH.get(s.get("catalyst_category", "other"), "其他")
        categories[cat] = categories.get(cat, 0) + 1

    sig_labels = [f"{s.get('symbol', '')} {s.get('name', '')}" for s in news_driven]
    sig_values = [s.get("significance", 0) for s in news_driven]

    payload = json.dumps(
        {
            "categories": categories,
            "sigLabels": sig_labels,
            "sigValues": sig_values,
            "total": total,
            "newsDriven": len(news_driven),
            "marketDriven": len(non_driven),
        },
        ensure_ascii=False,
    )

    # Cards
    cards: list[str] = []
    for s in news_driven:
        sym = html.escape(str(s.get("symbol", "")))
        name = html.escape(str(s.get("name", "")))
        cat = html.escape(CATALYST_ZH.get(s.get("catalyst_category", ""), "—"))
        catalyst = html.escape(str(s.get("news_catalyst", "—")))
        summary = html.escape(str(s.get("news_summary", "—")))
        sig = s.get("significance", 0)
        rk = s.get("attention_rank", 0)
        rk_chg = s.get("attention_rank_change", 0)
        pct = s.get("price_change_pct", 0.0)
        headlines = s.get("key_headlines") or []

        hl_items = "".join(
            f"<li>{html.escape(h[:80])}</li>" for h in headlines[:5]
        )
        sig_bar = f'<div class="sig-bar"><div class="sig-fill" style="width:{sig*10}%"></div><span>{sig}/10</span></div>'

        cards.append(
            f"""<article class="card">
  <header>
    <span class="sym">{sym}</span> <span class="nm">{name}</span>
    <span class="badge">{cat}</span>
    <span class="pct {'pos' if pct >= 0 else 'neg'}">{pct:+.2f}%</span>
  </header>
  <p class="catalyst">{catalyst}</p>
  <p class="summary">{summary}</p>
  <div class="meta-row">
    <span>排名 #{rk}（{rk_chg:+d}）</span>
    {sig_bar}
  </div>
  {'<ul class="headlines">' + hl_items + '</ul>' if hl_items else ''}
</article>"""
        )

    macro_block = ""
    if macro and macro.get("summary"):
        mlabel = html.escape(_sentiment_zh(ms))
        mscore = macro.get("market_score", 0)
        macro_block = f"""<section class="macro">
    <p class="lbl">市场整体 · {mlabel}（score={mscore}）</p>
    <p>{html.escape(str(macro.get('summary', '—')))}</p>
  </section>"""

    styles = """
    :root { --bg:#0f1419; --card:#1a2332; --text:#e7ecf3; --muted:#94a3b8;
            --acc:#38bdf8; --good:#34d399; --bad:#f87171; --orange:#fb923c; }
    * { box-sizing: border-box; }
    body { font-family: "Segoe UI","PingFang SC","Microsoft YaHei",sans-serif;
           background: var(--bg); color: var(--text); margin: 0; padding: 1.5rem; line-height: 1.55; }
    .wrap { max-width: 960px; margin: 0 auto; }
    h1 { font-size: 1.4rem; font-weight: 700; margin: 0 0 0.2rem; }
    h2 { font-size: 1.1rem; color: var(--acc); margin: 1.5rem 0 0.8rem; }
    .meta { color: var(--muted); font-size: 0.88rem; margin-bottom: 0.8rem; }
    .stats { display: flex; gap: 1rem; margin-bottom: 1.2rem; flex-wrap: wrap; }
    .stat { background: var(--card); border-radius: 10px; padding: 0.8rem 1.2rem;
            border: 1px solid rgba(255,255,255,.06); min-width: 140px; }
    .stat .num { font-size: 1.6rem; font-weight: 700; color: var(--acc); }
    .stat .lbl { font-size: 0.8rem; color: var(--muted); }
    .macro { background: var(--card); border-radius: 12px; padding: 1rem 1.15rem;
             margin-bottom: 1.2rem; border: 1px solid rgba(255,255,255,.06); }
    .macro .lbl { color: var(--acc); font-weight: 600; }
    .charts { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-bottom: 1.2rem; }
    @media (max-width: 720px) { .charts { grid-template-columns: 1fr; } }
    .ch { background: var(--card); border-radius: 12px; padding: 0.75rem;
          border: 1px solid rgba(255,255,255,.06); height: 280px; }
    .ch.wide { grid-column: 1 / -1; }
    .card { background: var(--card); border-radius: 12px; padding: 1rem 1.15rem;
            margin-bottom: 0.8rem; border: 1px solid rgba(255,255,255,.06); }
    .card header { font-weight: 600; margin-bottom: 0.4rem; display: flex;
                   flex-wrap: wrap; align-items: center; gap: 0.5rem; }
    .sym { font-size: 1.1rem; color: var(--acc); }
    .nm { color: var(--muted); font-weight: 400; font-size: 0.95rem; }
    .badge { font-size: 0.72rem; padding: 2px 8px; border-radius: 999px;
             font-weight: 600; background: rgba(251,191,36,.2); color: var(--orange); }
    .pct { font-size: 0.85rem; font-weight: 600; }
    .pct.pos { color: var(--bad); }
    .pct.neg { color: var(--good); }
    .catalyst { font-size: 0.95rem; margin: 0.3rem 0; font-weight: 500; }
    .summary { font-size: 0.88rem; color: var(--muted); margin: 0 0 0.5rem; }
    .meta-row { display: flex; align-items: center; gap: 1rem; font-size: 0.82rem; color: var(--muted); }
    .sig-bar { flex: 1; max-width: 160px; height: 8px; background: rgba(255,255,255,.08);
               border-radius: 4px; position: relative; overflow: visible; display: flex; align-items: center; }
    .sig-fill { height: 100%; border-radius: 4px; background: linear-gradient(90deg, var(--acc), var(--good)); }
    .sig-bar span { margin-left: 6px; white-space: nowrap; }
    .headlines { font-size: 0.82rem; color: #cbd5e1; padding-left: 1.2rem; margin: 0.5rem 0 0; }
    .headlines li { margin-bottom: 2px; }
    footer { margin-top: 2rem; font-size: 0.78rem; color: var(--muted); text-align: center; }
    """

    chart_blocks = [
        '<div class="ch"><canvas id="cCat"></canvas></div>',
        '<div class="ch"><canvas id="cSig"></canvas></div>',
    ]
    charts_html = f'<div class="charts">{"".join(chart_blocks)}</div>'

    script = f"""
    const D = {payload};
    // Category pie
    const catK = Object.keys(D.categories);
    const catV = catK.map(k => D.categories[k]);
    const palette = ['rgba(56,189,248,.75)','rgba(52,211,153,.75)','rgba(251,191,36,.75)',
                     'rgba(248,113,113,.75)','rgba(167,139,250,.75)','rgba(148,163,184,.55)',
                     'rgba(236,72,153,.7)','rgba(45,212,191,.7)'];
    new Chart(document.getElementById('cCat'), {{
      type: 'doughnut',
      data: {{ labels: catK, datasets: [{{ data: catV, backgroundColor: palette.slice(0, catK.length) }}] }},
      options: {{ responsive: true, maintainAspectRatio: false,
        plugins: {{ title: {{ display: true, text: '催化剂类别分布', color: '#e7ecf3' }},
                   legend: {{ position: 'bottom', labels: {{ color: '#cbd5e1' }} }} }} }}
    }});
    // Significance bar
    new Chart(document.getElementById('cSig'), {{
      type: 'bar',
      data: {{
        labels: D.sigLabels,
        datasets: [{{ label: '显著性', data: D.sigValues,
          backgroundColor: D.sigValues.map(v => v >= 7 ? 'rgba(52,211,153,.7)' : v >= 4 ? 'rgba(251,191,36,.7)' : 'rgba(148,163,184,.5)') }}]
      }},
      options: {{ responsive: true, maintainAspectRatio: false, indexAxis: 'y',
        plugins: {{ title: {{ display: true, text: '新闻显著性评分', color: '#e7ecf3' }}, legend: {{ display: false }} }},
        scales: {{
          x: {{ min: 0, max: 10, ticks: {{ color: '#94a3b8' }}, grid: {{ color: 'rgba(255,255,255,.06)' }} }},
          y: {{ ticks: {{ color: '#94a3b8', font: {{ size: 11 }} }}, grid: {{ display: false }} }}
        }}
      }}
    }});
"""

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>新闻驱动关注度扫描 · {trade_date}</title>
  <style>{styles}</style>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
</head>
<body>
<div class="wrap">
  <h1>新闻驱动关注度扫描</h1>
  <p class="meta">日期 {trade_date} · 模型 {model} · 生成 {gen}</p>

  <div class="stats">
    <div class="stat"><div class="num">{total}</div><div class="lbl">扫描飙升股</div></div>
    <div class="stat"><div class="num" style="color:var(--good)">{len(news_driven)}</div><div class="lbl">新闻驱动</div></div>
    <div class="stat"><div class="num" style="color:var(--muted)">{len(non_driven)}</div><div class="lbl">市场波动</div></div>
  </div>

  {macro_block}
  {charts_html if news_driven else ''}

  <h2>新闻驱动标的</h2>
  {"".join(cards) if cards else "<p style='color:var(--muted)'>本次扫描无新闻驱动标的</p>"}

  <footer>本地生成 · 仅供参考，不构成投资建议</footer>
</div>
<script>
{script if news_driven else ''}
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description="从 llm_attention_*.json 生成 Markdown / HTML 报告"
    )
    ap.add_argument("--json", type=Path, default=None, help="llm_attention_*.json 路径")
    ap.add_argument(
        "--latest", action="store_true", help="使用 data/results 下最新 llm_attention_*.json"
    )
    ap.add_argument("--results-dir", type=Path, default=None, help="结果目录")
    ap.add_argument("--md-out", type=Path, default=None, help="Markdown 输出路径")
    ap.add_argument("--html-out", type=Path, default=None, help="HTML 输出路径")
    ap.add_argument("--no-md", action="store_true", help="不生成 Markdown")
    ap.add_argument("--no-html", action="store_true", help="不生成 HTML")
    args = ap.parse_args()

    from src.settings import load_config

    cfg = load_config()
    results_dir = Path(
        args.results_dir or cfg.get("paths", {}).get("results_dir", "data/results")
    )
    results_dir = results_dir if results_dir.is_absolute() else ROOT / results_dir

    json_path = args.json
    if args.latest or json_path is None:
        json_path = _find_latest_json(results_dir)
        if not json_path:
            print("未找到 llm_attention_*.json", file=sys.stderr)
            sys.exit(1)

    json_path = json_path if json_path.is_absolute() else ROOT / json_path
    if not json_path.is_file():
        print(f"文件不存在: {json_path}", file=sys.stderr)
        sys.exit(1)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    stem = json_path.stem
    date_part = stem.replace("llm_attention_", "") if stem.startswith("llm_attention_") else stem

    md_path = args.md_out
    if md_path is None and not args.no_md:
        md_path = results_dir / f"llm_report_{date_part}.md"
    html_path = args.html_out
    if html_path is None and not args.no_html:
        html_path = results_dir / f"llm_report_{date_part}.html"

    if md_path:
        md_path = md_path if md_path.is_absolute() else ROOT / md_path
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(build_markdown(data), encoding="utf-8")
        print(f"Markdown: {md_path}")

    if html_path:
        html_path = html_path if html_path.is_absolute() else ROOT / html_path
        html_path.parent.mkdir(parents=True, exist_ok=True)
        html_path.write_text(build_html(data), encoding="utf-8")
        print(f"HTML: {html_path}")


if __name__ == "__main__":
    main()
