#!/usr/bin/env python3
"""为 M8 Regime-Aware 混合打分 + indcap3 硬上限策略生成每月 Top20 选股 Markdown 报告。

数据来源：M8 行业集中度约束回测的 topk_holdings.csv。
策略口径：M8_regime_aware_fixed_policy__indcap3 + U1_liquid_tradable + Top20。

对每个月输出每只股票的代码、名称、当月收益，以及 Top20 等权当月收益。
同时生成一张汇总表。

输出目录：docs/reports/m8_indcap3_monthly/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

# ── 常量 ─────────────────────────────────────────────────────────────────

MODEL_NAME = "M8_regime_aware_fixed_policy__indcap3"
CANDIDATE_POOL = "U1_liquid_tradable"
TOP_K = 20
LABEL_COL = "label_forward_1m_o2o_return"

DEFAULT_HOLDINGS_CSV = ROOT / "data" / "results" / "monthly_selection_m8_concentration_regime_pitfix_2026_05_03_2026_05_03_topk_holdings.csv"
STOCK_NAMES_PATH = ROOT / "data" / "cache" / "a_share_stock_names.csv"
OUTPUT_DIR = ROOT / "docs" / "reports" / "m8_indcap3_monthly"


def load_stock_names(path: Path) -> pd.DataFrame:
    """加载股票代码→名称映射。"""
    df = pd.read_csv(path, dtype={"symbol": str})
    df["symbol"] = df["symbol"].astype(str).str.extract(r"(\d{1,6})", expand=False).fillna("").str.zfill(6)
    df["name"] = df["name"].astype(str).str.strip()
    return df[["symbol", "name"]].drop_duplicates("symbol", keep="first")


def generate_monthly_reports(
    holdings_csv: Path,
    names: pd.DataFrame,
    output_dir: Path,
) -> None:
    """从 M8 topk_holdings CSV 读取已应用 indcap3 的选股结果，生成每月 Markdown 报告。"""
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[m8-report] 读取 holdings: {holdings_csv}")
    df = pd.read_csv(holdings_csv, dtype={"symbol": str})
    print(f"[m8-report] 总行数: {len(df)}, 列: {list(df.columns)}")

    # 筛选目标策略
    mask = (
        (df["model"] == MODEL_NAME)
        & (df["candidate_pool_version"] == CANDIDATE_POOL)
        & (df["top_k"] == TOP_K)
    )
    selected = df[mask].copy()
    print(f"[m8-report] 策略 {MODEL_NAME} + {CANDIDATE_POOL} + Top{TOP_K}: {len(selected)} 行")

    if selected.empty:
        print(f"[ERROR] 未找到策略数据")
        return

    # 标准化股票代码
    selected["symbol_norm"] = (
        selected["symbol"].astype(str).str.extract(r"(\d{1,6})", expand=False).fillna("").str.zfill(6)
    )

    # 合并股票名称
    selected = selected.merge(names, left_on="symbol_norm", right_on="symbol", how="left", suffixes=("", "_name"))
    selected["name"] = selected["name"].fillna("").astype(str).str.strip()

    # 按信号日排序
    months = sorted(selected["signal_date"].dropna().unique())
    print(f"[m8-report] 共 {len(months)} 个月份待生成报告")
    print()

    all_summaries: list[dict] = []

    for signal_date in months:
        part = selected[selected["signal_date"] == signal_date].copy()
        # 按 selected_rank 排序
        part = part.sort_values("selected_rank")

        month_str = pd.Timestamp(signal_date).strftime("%Y-%m")

        # 获取买卖日期
        buy_date = str(part["buy_trade_date"].dropna().iloc[0]) if part["buy_trade_date"].notna().any() else ""
        sell_date = str(part["sell_trade_date"].dropna().iloc[0]) if part["sell_trade_date"].notna().any() else ""

        # 计算等权收益
        returns = pd.to_numeric(part[LABEL_COL], errors="coerce")
        ew_return = float(returns.mean()) if returns.notna().any() else np.nan

        # ── 构建 Markdown ──
        lines: list[str] = []
        lines.append(f"# M8 Regime-Aware + indcap3 月度 Top20 - {month_str}")
        lines.append("")
        lines.append(f"- 持有月份：`{month_str}`")
        lines.append(f"- 信号日：`{pd.Timestamp(signal_date).strftime('%Y-%m-%d')}`")
        lines.append(f"- 买入日：`{buy_date}`" if buy_date else "")
        lines.append(f"- 卖出日：`{sell_date}`" if sell_date else "")
        lines.append(f"- 候选池：`{CANDIDATE_POOL}`")
        lines.append(f"- 选股模型：`{MODEL_NAME}`")
        lines.append(f"- 选股规则：`industry_names_cap`，同一级行业上限：`3`")
        ew_str = f"{ew_return * 100:.2f}%" if np.isfinite(ew_return) else "N/A"
        lines.append(f"- Top20 等权当月收益：`{ew_str}`")
        lines.append("")

        # 表头
        lines.append("| 排名 | 股票代码 | 股票名称 | 一级行业 | 模型分数 | 当月收益 |")
        lines.append("| -- | ------ | ----- | ---- | -------- | ------ |")

        for _, row in part.iterrows():
            rank = int(row.get("selected_rank", 0))
            code = str(row.get("symbol_norm", row.get("symbol", "")))
            name = str(row.get("name", ""))
            industry = str(row.get("industry_level1", "")) if pd.notna(row.get("industry_level1")) else ""
            score = float(row.get("score", np.nan))
            ret = float(row.get(LABEL_COL, np.nan))

            score_str = f"{score:.4f}" if np.isfinite(score) else "-"
            ret_str = f"{ret * 100:.2f}%" if np.isfinite(ret) else "-"

            lines.append(f"| {rank} | {code} | {name} | {industry} | {score_str} | {ret_str} |")

        lines.append("")

        # 写文件
        filename = f"m8_regime_aware_indcap3_top20_{month_str}.md"
        filepath = output_dir / filename
        filepath.write_text("\n".join(lines), encoding="utf-8")
        print(f"  [{month_str}] → {filepath.name}  等权收益: {ew_str}")

        all_summaries.append({
            "month": month_str,
            "signal_date": str(pd.Timestamp(signal_date).date()),
            "buy_date": buy_date,
            "sell_date": sell_date,
            "ew_return": ew_return,
            "selected_count": len(part),
        })

    # ── 汇总表 ──
    if not all_summaries:
        print("\n[WARN] 无任何报告生成")
        return

    summary_lines: list[str] = []
    summary_lines.append("# M8 Regime-Aware + indcap3 月度收益汇总")
    summary_lines.append("")
    summary_lines.append(f"- 策略模型：`{MODEL_NAME}`")
    summary_lines.append(f"- 候选池：`{CANDIDATE_POOL}`")
    summary_lines.append(f"- TopK：`{TOP_K}`")
    summary_lines.append("")
    summary_lines.append("| 月份 | 信号日 | 买入日 | 卖出日 | 选股数 | Top20 等权当月收益 |")
    summary_lines.append("| -- | -- | -- | -- | -- | -- |")

    for s in all_summaries:
        ret_str = f"{s['ew_return'] * 100:.2f}%" if np.isfinite(s['ew_return']) else "N/A"
        summary_lines.append(
            f"| {s['month']} | {s['signal_date']} | {s['buy_date']} | {s['sell_date']} | {s['selected_count']} | {ret_str} |"
        )

    # 汇总统计
    valid_rets = [s["ew_return"] for s in all_summaries if np.isfinite(s["ew_return"])]
    if valid_rets:
        mean_ret = np.mean(valid_rets)
        median_ret = np.median(valid_rets)
        hit_rate = np.mean([r > 0 for r in valid_rets])
        total_ret = np.prod([1 + r for r in valid_rets]) - 1
        summary_lines.append("")
        summary_lines.append("### 统计摘要")
        summary_lines.append("")
        summary_lines.append(f"- 总月份数：{len(all_summaries)}")
        summary_lines.append(f"- 月均等权收益：{mean_ret * 100:.2f}%")
        summary_lines.append(f"- 中位数等权收益：{median_ret * 100:.2f}%")
        summary_lines.append(f"- 累计收益：{total_ret * 100:.2f}%")
        summary_lines.append(f"- 正收益月份占比：{hit_rate * 100:.1f}%")

    summary_path = output_dir / "m8_regime_aware_indcap3_summary.md"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
    print(f"\n汇总报告 → {summary_path}")


def run() -> None:
    parser = argparse.ArgumentParser(description="生成 M8 Regime-Aware + indcap3 月度 Top20 Markdown 报告")
    parser.add_argument("--holdings-csv", type=str, default=str(DEFAULT_HOLDINGS_CSV),
                        help="M8 topk_holdings.csv 路径")
    parser.add_argument("--stock-names", type=str, default=str(STOCK_NAMES_PATH),
                        help="股票名称映射 CSV")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR),
                        help="输出目录")
    parser.add_argument("--model", type=str, default=MODEL_NAME,
                        help="策略模型名")
    parser.add_argument("--pool", type=str, default=CANDIDATE_POOL,
                        help="候选池")
    parser.add_argument("--top-k", type=int, default=TOP_K,
                        help="TopK")
    args = parser.parse_args()

    holdings_csv = Path(args.holdings_csv)
    stock_names_path = Path(args.stock_names)
    output_dir = Path(args.output_dir)

    if not holdings_csv.exists():
        print(f"[ERROR] holdings 文件不存在: {holdings_csv}")
        return

    print(f"[m8-report] holdings: {holdings_csv}")
    print(f"[m8-report] 股票名称: {stock_names_path}")
    print(f"[m8-report] 输出目录: {output_dir}")
    print(f"[m8-report] 策略: {args.model} + {args.pool} + Top{args.top_k}")
    print()

    names = load_stock_names(stock_names_path)
    print(f"[m8-report] 股票名称映射: {len(names)} 条")
    print()

    generate_monthly_reports(
        holdings_csv=holdings_csv,
        names=names,
        output_dir=output_dir,
    )

    print("\n✅ 完成！")


if __name__ == "__main__":
    run()
