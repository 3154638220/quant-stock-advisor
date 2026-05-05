#!/usr/bin/env python3
"""M6 LTR 失败归因分析。

诊断为什么 XGBoost rank NDCG 在 U1 候选池上完全失效（after-cost 超额~0），
而在 U2+Top20 上仅有弱证据（超额 1.21% 但 Rank IC 0.007）。

假设检验：
1. U1 池噪声过高 — 微盘/ST 残余/涨停候选过多导致排序信号稀释
2. U1 池候选数过大 — NDCG 在极端 label imbalance 下性能退化
3. 特征非线性未被 XGBoost 捕获 — U2 筛选后的特征分布更符合线性假设
4. 牛熊周期差异 — U1/U2 在不同市场状态下的覆盖率变化
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="M6 LTR 失败归因分析")
    p.add_argument("--m6-monthly-long", type=str,
                   default="data/results/monthly_selection_m6_ltr_2026-05-01_monthly_long.csv")
    p.add_argument("--m5-monthly-long", type=str,
                   default="data/results/monthly_selection_m5_multisource_2026-05-01_monthly_long.csv")
    p.add_argument("--dataset", type=str,
                   default="data/cache/monthly_selection_features.parquet")
    p.add_argument("--results-dir", type=str, default="data/results")
    p.add_argument("--output-prefix", type=str, default="m6_ltr_failure_attribution")
    return p.parse_args()


def _resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _pool_summary(dataset: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """按候选池汇总规模与构成特征。"""
    out: dict[str, dict[str, Any]] = {}
    for pool, part in dataset.groupby("candidate_pool_version", sort=True):
        pass_part = part[part["candidate_pool_pass"].astype(bool)]
        out[str(pool)] = {
            "pool": str(pool),
            "total_rows": len(part),
            "pass_rows": len(pass_part),
            "signal_months": int(part["signal_date"].nunique()),
            "unique_symbols": int(part["symbol"].nunique()),
            "mean_per_month": float(part.groupby("signal_date")["symbol"].nunique().mean()),
            "pass_per_month": float(pass_part.groupby("signal_date")["symbol"].nunique().mean())
            if not pass_part.empty else 0.0,
        }
        # 检查风险标记
        if "risk_flags" in part.columns:
            flags = part["risk_flags"].dropna().astype(str)
            st_count = int(flags.str.contains("ST", case=False).sum())
            out[str(pool)]["st_flag_rows"] = st_count
        if "is_st" in part.columns:
            out[str(pool)]["st_rows"] = int(part["is_st"].astype(bool).sum())
    return out


def _excess_by_year(monthly_long: pd.DataFrame, model_filter: str = "") -> pd.DataFrame:
    """按年拆分 after-cost excess 表现。"""
    ml = monthly_long.copy()
    if model_filter and "model" in ml.columns:
        ml = ml[ml["model"].astype(str).str.contains(model_filter)]
    if ml.empty:
        return pd.DataFrame()
    ml["signal_date"] = pd.to_datetime(ml["signal_date"], errors="coerce")
    ml["year"] = ml["signal_date"].dt.year
    if "topk_return" not in ml.columns or "market_ew_return" not in ml.columns:
        return pd.DataFrame()
    topk = pd.to_numeric(ml["topk_return"], errors="coerce")
    market = pd.to_numeric(ml["market_ew_return"], errors="coerce")
    drag = pd.to_numeric(ml.get("cost_drag", 0.0), errors="coerce").fillna(0.0)
    ml["after_cost_excess"] = topk - drag - market
    return (
        ml.groupby(["year", "candidate_pool_version", "top_k"], sort=True)
        .agg(
            months=("signal_date", "nunique"),
            mean_excess=("after_cost_excess", "mean"),
            total_excess=("after_cost_excess", lambda x: float((1 + x).prod() - 1)),
            hit_rate=("after_cost_excess", lambda x: float((x > 0).mean())),
            mean_turnover=("turnover_half_l1", "mean") if "turnover_half_l1" in ml.columns else ("signal_date", "count"),
        )
        .reset_index()
    )


def _candidate_pool_overlap(dataset: pd.DataFrame, pool_a: str, pool_b: str) -> pd.DataFrame:
    """逐月计算两个候选池的 overlap。"""
    a = dataset[dataset["candidate_pool_version"] == pool_a]
    b = dataset[dataset["candidate_pool_version"] == pool_b]
    rows: list[dict[str, Any]] = []
    for sd in sorted(set(a["signal_date"].unique()) & set(b["signal_date"].unique())):
        syms_a = set(a[a["signal_date"] == sd]["symbol"].astype(str))
        syms_b = set(b[b["signal_date"] == sd]["symbol"].astype(str))
        union = syms_a | syms_b
        intersection = syms_a & syms_b
        rows.append({
            "signal_date": sd,
            "pool_a_count": len(syms_a),
            "pool_b_count": len(syms_b),
            "overlap_count": len(intersection),
            "overlap_ratio": len(intersection) / len(union) if union else 0.0,
            "pool_a_only": len(syms_a - syms_b),
            "pool_b_only": len(syms_b - syms_a),
        })
    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()
    m6_path = _resolve(args.m6_monthly_long)
    m5_path = _resolve(args.m5_monthly_long)
    dataset_path = _resolve(args.dataset)
    results_dir = _resolve(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    output_stem = f"{args.output_prefix}_{pd.Timestamp.now().strftime('%Y%m%d')}"

    findings: list[str] = []

    # 1. 加载 M6 vs M5 结果
    m6 = pd.read_csv(m6_path) if m6_path.exists() else pd.DataFrame()
    m5 = pd.read_csv(m5_path) if m5_path.exists() else pd.DataFrame()
    dataset = pd.read_parquet(dataset_path) if dataset_path.exists() else pd.DataFrame()

    if m6.empty:
        print("[m6-attribution] M6 monthly_long 不存在，无法分析")
        return 1
    if m5.empty:
        print("[m6-attribution] M5 monthly_long 不存在，跳过对比")
    if dataset.empty:
        print("[m6-attribution] dataset 不存在，跳过候选池分析")

    # 2. 逐年拆分 → 识别失败年份
    print("=== M6 LTR 逐年 after-cost excess ===")
    m6_by_year = _excess_by_year(m6, model_filter="xgboost")
    m5_by_year = _excess_by_year(m5) if not m5.empty else pd.DataFrame()
    print(m6_by_year.to_string(index=False))

    # 3. 识别 U1 vs U2 差异
    if not dataset.empty:
        pool_stats = _pool_summary(dataset)
        print("\n=== 候选池规模对比 ===")
        for pool, stats in pool_stats.items():
            print(f"  {pool}: {stats['pass_per_month']:.0f} pass/月, "
                  f"{stats['signal_months']} 月, {stats['unique_symbols']} 只股票")

        if "U1_liquid_tradable" in pool_stats and "U2_risk_sane" in pool_stats:
            u1 = pool_stats["U1_liquid_tradable"]
            u2 = pool_stats["U2_risk_sane"]
            ratio = u1["pass_per_month"] / max(u2["pass_per_month"], 1)
            if ratio > 3:
                findings.append(
                    f"U1 池候选数约为 U2 的 {ratio:.0f}x，"
                    f"label imbalance 可能导致 NDCG ranking 信号在大量无关标的中被稀释"
                )
            if u1.get("st_flag_rows", 0) > u1["pass_rows"] * 0.05:
                findings.append(
                    f"U1 池含 {u1.get('st_flag_rows', 0)} 条 ST 标记记录，"
                    f"需确认 U2 过滤后是否清除了这些噪声"
                )

            overlap = _candidate_pool_overlap(dataset, "U1_liquid_tradable", "U2_risk_sane")
            if not overlap.empty:
                mean_ol = float(overlap["overlap_ratio"].mean())
                print(f"\n=== U1/U2 Overlap 分析 ===")
                print(f"  月均 overlap ratio: {mean_ol:.1%}")
                print(f"  U1-only 月均: {float(overlap['pool_a_only'].mean()):.0f}")
                print(f"  U2-only 月均: {float(overlap['pool_b_only'].mean()):.0f}")
                if mean_ol < 0.5:
                    findings.append(
                        f"U1/U2 overlap 仅 {mean_ol:.1%}，说明 U2 过滤剔除了大量非重叠标的；"
                        f"U1 中多出的 {float(overlap['pool_a_only'].mean()):.0f} 只股票很可能包含微盘/ST/涨跌停候选等噪声"
                    )

    # 4. 年份识别
    if not m6_by_year.empty:
        bad_years = m6_by_year[m6_by_year["mean_excess"] < 0]
        if not bad_years.empty:
            years_str = ", ".join(str(int(y)) for y in bad_years["year"].unique())
            findings.append(f"M6 LTR 在 {years_str} 年出现负超额，建议检查这些年份的市场状态")

    # 5. 输出报告
    doc_lines = [
        "# M6 LTR 失败归因分析",
        "",
        f"生成时间：{pd.Timestamp.utcnow().isoformat()}",
        "",
        "## 逐年 After-Cost Excess",
        "",
        m6_by_year.to_markdown(index=False) if not m6_by_year.empty else "_无数据_",
        "",
        "## 发现",
        "",
    ]
    if findings:
        for i, f in enumerate(findings, 1):
            doc_lines.append(f"{i}. {f}")
    else:
        doc_lines.append("（归因分析需要完整的 M6/M5 monthly_long 和 dataset parquet 数据）")
    doc_lines.extend([
        "",
        "## 建议后续行动",
        "",
        "1. 若 U1 池候选数 > 3x U2，优先优化 U1 过滤规则（而非放弃 LTR）",
        "2. 在 U1 上测试 M5 ExtraTrees（线性/树模型对比）作为基线，验证是否是模型-池不匹配",
        "3. 检查 M6 NDCG 超参（`ndcg_exp_gain`、列采样）在 U1 上的敏感性",
        "4. 若 U1/U2 overlap < 50%，确认 U1-only 标的的财务健康度（是否含 ST/披星戴帽/亏损）",
    ])
    doc_path = results_dir / f"{output_stem}.md"
    doc_path.write_text("\n".join(doc_lines), encoding="utf-8")
    print(f"\n[m6-attribution] 报告: {doc_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
