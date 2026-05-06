#!/usr/bin/env python3
"""Oracle 条件 IC Gate — 新数据准入评估脚本。

比较两个 M5 月度选股结果（基线 vs +新特征家族），
计算三条准入门槛：
1. 全截面增量 Rank IC delta > 0.002
2. Oracle 条件 IC 均值 > 0.03
3. Oracle Top-20 overlap 提升 >= 2pp

用法:
    python scripts/run_oracle_ic_gate.py \
        --baseline data/results/m5_baseline_monthly_long.csv \
        --candidate data/results/m5_margin_delta_monthly_long.csv \
        --family margin_trading \
        --output-dir data/results
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Oracle 条件 IC Gate 准入评估")
    p.add_argument("--baseline", type=str, required=True,
                   help="基线 M5 monthly_long CSV")
    p.add_argument("--candidate", type=str, required=True,
                   help="候选 M5 monthly_long CSV (基线 + 新特征家族)")
    p.add_argument("--family", type=str, required=True,
                   help="新数据家族名称")
    p.add_argument("--label-col", type=str, default="label_f1m",
                   help="未来收益列名")
    p.add_argument("--pool", type=str, default="U1_liquid_tradable",
                   help="候选池")
    p.add_argument("--top-k", type=int, default=20,
                   help="Top-K")
    p.add_argument("--output-dir", type=str, default="data/results",
                   help="输出目录")
    p.add_argument("--output-prefix", type=str, default="",
                   help="输出文件名前缀")
    return p.parse_args()


def _resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def load_monthly_long(path: Path, pool: str, top_k: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "candidate_pool_version" in df.columns:
        df = df[df["candidate_pool_version"].astype(str) == pool]
    if "top_k" in df.columns:
        df = df[df["top_k"].astype(int) == top_k]
    return df


def main() -> int:
    args = parse_args()
    from src.features.oracle_ic_gate import (
        evaluate_newdata_admission,
        build_oracle_ic_gate_doc,
    )

    baseline_path = _resolve(args.baseline)
    candidate_path = _resolve(args.candidate)
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prefix = args.output_prefix or f"oracle_ic_gate_{args.family}"

    bl = load_monthly_long(baseline_path, args.pool, args.top_k)
    ca = load_monthly_long(candidate_path, args.pool, args.top_k)

    if bl.empty:
        print(f"[oracle-gate] 基线数据为空: {baseline_path}", file=sys.stderr)
        return 1
    if ca.empty:
        print(f"[oracle-gate] 候选数据为空: {candidate_path}", file=sys.stderr)
        return 1

    # Merge on signal_date to get aligned predictions
    merge_keys = ["signal_date"]
    merge_cols = merge_keys + ["score_mean", args.label_col, "topk_return", "market_ew_return"]

    bl_sub = bl[merge_cols].rename(columns={"score_mean": "score_baseline", args.label_col: args.label_col})
    ca_sub = ca[merge_cols].rename(columns={"score_mean": "score_candidate"})

    merged = bl_sub.merge(ca_sub, on="signal_date", how="inner", suffixes=("", "_cand"))
    if merged.empty:
        print("[oracle-gate] 两个月度文件无重叠 signal_date", file=sys.stderr)
        return 1

    labels = pd.to_numeric(merged[args.label_col], errors="coerce").values

    # Oracle top-k mask: top 20% by label within each signal_date
    oracle_topk_mask = np.zeros(len(merged), dtype=bool)
    for sd, idx in merged.groupby("signal_date").groups.items():
        sub_labels = pd.to_numeric(merged.loc[idx, args.label_col], errors="coerce")
        threshold = sub_labels.quantile(0.80)
        oracle_topk_mask[idx] = (sub_labels >= threshold).values

    baseline_scores = pd.to_numeric(merged["score_baseline"], errors="coerce").values
    candidate_scores = pd.to_numeric(merged["score_candidate"], errors="coerce").values

    valid = (
        np.isfinite(baseline_scores)
        & np.isfinite(candidate_scores)
        & np.isfinite(labels)
    )
    if valid.sum() < 10:
        print(f"[oracle-gate] 有效样本不足: {valid.sum()}", file=sys.stderr)
        return 1

    # Oracle overlap: top-20 by score that are in oracle top-20% by label
    def compute_oracle_overlap(scores_2d, oracle_mask):
        overlap_ratios = []
        for sd, idx in merged.groupby("signal_date").groups.items():
            m = oracle_mask[idx]
            if m.sum() == 0:
                overlap_ratios.append(0.0)
                continue
            s = scores_2d[idx]
            top_k = max(1, m.sum())
            top_k_idx = np.argsort(s)[-top_k:]
            overlap = m[top_k_idx].sum() / top_k
            overlap_ratios.append(overlap)
        return float(np.mean(overlap_ratios)) if overlap_ratios else 0.0

    baseline_ol = compute_oracle_overlap(baseline_scores[valid], oracle_topk_mask[valid])
    candidate_ol = compute_oracle_overlap(candidate_scores[valid], oracle_topk_mask[valid])

    gate = evaluate_newdata_admission(
        baseline_scores[valid],
        candidate_scores[valid],
        labels[valid],
        oracle_topk_mask[valid],
        baseline_oracle_overlap=baseline_ol,
        candidate_oracle_overlap=candidate_ol,
    )

    # Print summary
    print(f"\n=== Oracle IC Gate: {args.family} ===")
    print(f"  全截面 Rank IC delta: {gate.rank_ic_delta:+.4f} (门槛 > 0.002)")
    print(f"  Oracle 条件 IC: {gate.candidate_oracle_cond_ic:.4f} (门槛 > 0.03)")
    print(f"  Oracle Overlap delta: {gate.oracle_overlap_delta:+.4f} (门槛 >= 0.02)")
    print(f"  结论: {'✅ 通过' if gate.passed else '❌ 未通过'}")
    print(f"  建议: {gate.recommendation}")

    # Write Markdown doc
    doc = build_oracle_ic_gate_doc(gate, family_name=args.family)
    doc_path = output_dir / f"{prefix}_{pd.Timestamp.now().strftime('%Y%m%d')}.md"
    doc_path.write_text(doc, encoding="utf-8")
    print(f"\n[oracle-gate] 报告: {doc_path}")

    # Write JSON
    json_path = output_dir / f"{prefix}_{pd.Timestamp.now().strftime('%Y%m%d')}.json"
    json_payload = {
        "family": args.family,
        "passed": gate.passed,
        "baseline_rank_ic": gate.baseline_rank_ic,
        "candidate_rank_ic": gate.candidate_rank_ic,
        "rank_ic_delta": gate.rank_ic_delta,
        "baseline_oracle_cond_ic": gate.baseline_oracle_cond_ic,
        "candidate_oracle_cond_ic": gate.candidate_oracle_cond_ic,
        "baseline_oracle_overlap": gate.baseline_oracle_overlap,
        "candidate_oracle_overlap": gate.candidate_oracle_overlap,
        "oracle_overlap_delta": gate.oracle_overlap_delta,
        "gate_checks": gate.gate_checks,
        "recommendation": gate.recommendation,
    }
    json_path.write_text(json.dumps(json_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[oracle-gate] JSON: {json_path}")

    return 0 if gate.passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
