#!/usr/bin/env python3
"""G2: 生成自包含 HTML 仪表盘，可视化月度选股策略表现。

用法:
    python scripts/generate_dashboard.py
    python scripts/generate_dashboard.py --results-dir data/results
    python scripts/generate_dashboard.py --output docs/reports/dashboard.html
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _find_latest_results(results_dir: Path) -> dict[str, Path]:
    """Auto-discover latest leaderboard, monthly_long, rank_ic, and concentration CSVs."""
    found: dict[str, Path] = {}
    patterns = {
        "leaderboard": "*leaderboard*.csv",
        "monthly_long": "*monthly_long*.csv",
        "rank_ic": "*rank_ic*.csv",
        "industry_concentration": "*industry_concentration*.csv",
        "topk_holdings": "*topk_holdings*.csv",
    }
    for key, pattern in patterns.items():
        files = sorted(results_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
        if files:
            found[key] = files[0]
    return found


def _build_dashboard_html(
    files: dict[str, Path],
    *,
    title: str = "Quant Stock Advisor Dashboard",
) -> str:
    """生成自包含 HTML 报告。"""
    import pandas as pd
    import plotly.graph_objects as go
    import plotly.io as pio
    has_data = False
    fig_components: list[str] = []

    # ── (a) 月度超额 bar chart ──
    if "monthly_long" in files:
        df = pd.read_csv(files["monthly_long"])
        if not df.empty and "signal_date" in df.columns and "topk_excess_after_cost" in df.columns:
            has_data = True
            df["signal_date"] = pd.to_datetime(df["signal_date"])
            # Best performing model per month (exclude NaN excess)
            valid = df[df["topk_excess_after_cost"].notna()]
            if valid.empty:
                valid = df
            best_idx = valid.groupby("signal_date")["topk_excess_after_cost"].idxmax().dropna()
            best = valid.loc[best_idx.astype(int)].sort_values("signal_date")
            best_valid = best[best["topk_excess_after_cost"].notna()]

            fig1 = go.Figure()
            fig1.add_trace(go.Bar(
                x=best_valid["signal_date"],
                y=best_valid["topk_excess_after_cost"] * 100,
                marker_color=[
                    "green" if v > 0 else "red"
                    for v in best_valid["topk_excess_after_cost"]
                ],
                text=[f"{v*100:.2f}%" for v in best_valid["topk_excess_after_cost"]],
                textposition="outside",
                name="月度净超额",
            ))
            fig1.update_layout(
                title="月度净超额（最优模型，after-cost）",
                yaxis_title="净超额 (%)",
                xaxis_title="信号月",
                height=400,
                margin=dict(t=40, b=40),
            )
            fig_components.append(pio.to_html(fig1, full_html=False, include_plotlyjs="cdn"))

    # ── (b) 因子滚动 IC 折线图 ──
    if "rank_ic" in files:
        df_ic = pd.read_csv(files["rank_ic"])
        if not df_ic.empty and "rank_ic" in df_ic.columns and "signal_date" in df_ic.columns:
            has_data = True
            df_ic["signal_date"] = pd.to_datetime(df_ic["signal_date"])

            fig2 = go.Figure()
            for model_name in df_ic["model"].unique()[:5]:  # top 5 models
                sub = df_ic[df_ic["model"] == model_name].sort_values("signal_date")
                sub["rolling_6m"] = sub["rank_ic"].rolling(6, min_periods=3).mean()
                fig2.add_trace(go.Scatter(
                    x=sub["signal_date"],
                    y=sub["rolling_6m"],
                    mode="lines",
                    name=model_name[:60],
                ))
            fig2.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
            fig2.update_layout(
                title="因子 Rank IC（6 月滚动均值）",
                yaxis_title="Rank IC",
                xaxis_title="信号月",
                height=400,
                margin=dict(t=40, b=40),
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            fig_components.append(pio.to_html(fig2, full_html=False, include_plotlyjs="cdn"))

    # ── (c) 候选池大小历史 ──
    if "monthly_long" in files:
        df_pool = pd.read_csv(files["monthly_long"])
        if not df_pool.empty and "candidate_pool_width" in df_pool.columns:
            has_data = True
            df_pool["signal_date"] = pd.to_datetime(df_pool["signal_date"])
            pool_by_date = df_pool.groupby("signal_date")["candidate_pool_width"].first().sort_index()

            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(
                x=pool_by_date.index,
                y=pool_by_date.values,
                mode="lines+markers",
                fill="tozeroy",
                fillcolor="rgba(0,100,200,0.1)",
                line_color="steelblue",
                name="候选池宽度",
            ))
            fig3.update_layout(
                title="候选池宽度历史",
                yaxis_title="标的数",
                xaxis_title="信号月",
                height=300,
                margin=dict(t=40, b=40),
            )
            fig_components.append(pio.to_html(fig3, full_html=False, include_plotlyjs="cdn"))

    # ── (d) 行业集中度趋势 ──
    if "industry_concentration" in files:
        df_ind = pd.read_csv(files["industry_concentration"])
        if not df_ind.empty and "signal_date" in df_ind.columns and "max_industry_share" in df_ind.columns:
            has_data = True
            df_ind["signal_date"] = pd.to_datetime(df_ind["signal_date"])
            df_ind = df_ind.sort_values("signal_date")

            # Select top model by concentration_pass rate
            models = df_ind.groupby("model")["concentration_pass"].mean().sort_values(ascending=False)
            top_model = models.index[0] if not models.empty else df_ind["model"].iloc[0]
            sub = df_ind[df_ind["model"] == top_model]

            fig4 = go.Figure()
            fig4.add_trace(go.Scatter(
                x=sub["signal_date"],
                y=sub["max_industry_share"],
                mode="lines+markers",
                line_color="coral",
                name="最大行业集中度",
            ))
            # Color points by pass/fail
            passed = sub[sub["concentration_pass"]]
            failed = sub[~sub["concentration_pass"]]
            if not passed.empty:
                fig4.add_trace(go.Scatter(
                    x=passed["signal_date"], y=passed["max_industry_share"],
                    mode="markers", marker_color="green", marker_size=8,
                    name="集中度通过",
                ))
            if not failed.empty:
                fig4.add_trace(go.Scatter(
                    x=failed["signal_date"], y=failed["max_industry_share"],
                    mode="markers", marker_color="red", marker_size=8,
                    name="集中度超标",
                ))
            fig4.update_layout(
                title=f"行业集中度趋势 (top model: {top_model[:60]})",
                yaxis_title="最大行业权重",
                xaxis_title="信号月",
                height=350,
                margin=dict(t=40, b=40),
                yaxis=dict(range=[0, 1.05]),
            )
            fig_components.append(pio.to_html(fig4, full_html=False, include_plotlyjs="cdn"))

    # ── Assembly ──
    if not has_data:
        return "<html><body><h2>无可用仪表盘数据</h2><p>请先运行月度选股管线生成结果 CSV。</p></body></html>"

    sections = "\n".join(f'<div class="chart">{c}</div>' for c in fig_components)

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
  body {{ font-family: -apple-system, 'Segoe UI', Roboto, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }}
  h1 {{ color: #333; margin-bottom: 5px; }}
  .subtitle {{ color: #666; font-size: 0.9em; margin-bottom: 20px; }}
  .chart {{ background: white; border-radius: 8px; padding: 15px; margin-bottom: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
  .footer {{ text-align: center; color: #999; font-size: 0.8em; margin-top: 40px; }}
  iframe {{ width: 100%; border: none; }}
</style>
</head>
<body>
<h1>{title}</h1>
<div class="subtitle">生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')} | 数据源: {', '.join(f.name for f in files.values())}</div>
{sections}
<div class="footer">Quant Stock Advisor · Auto-generated dashboard</div>
</body>
</html>"""


def main() -> int:
    import pandas as pd

    parser = argparse.ArgumentParser(description="生成自包含 HTML 仪表盘")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "data" / "results", help="结果 CSV 目录")
    parser.add_argument("--output", type=Path, default=None, help="输出 HTML 路径（默认: docs/reports/dashboard_YYYY-MM-DD.html）")
    parser.add_argument("--title", type=str, default="Quant Stock Advisor Dashboard", help="仪表盘标题")
    args = parser.parse_args()

    results_dir = args.results_dir
    if not results_dir.exists():
        print(f"[ERROR] 结果目录不存在: {results_dir}")
        return 1

    files = _find_latest_results(results_dir)
    if not files:
        print(f"[WARN] 未在 {results_dir} 中找到结果 CSV 文件")
        return 1

    print(f"[dashboard] 发现 {len(files)} 个数据源:")
    for k, v in files.items():
        print(f"  {k}: {v.name}")

    html = _build_dashboard_html(files, title=args.title)

    output = args.output or (ROOT / "docs" / "reports" / f"dashboard_{pd.Timestamp.now().strftime('%Y%m%d')}.html")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    print(f"[dashboard] 仪表盘已生成: {output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
