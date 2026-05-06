"""Oracle 条件 IC Gate — 新数据准入的量化门槛。

衡量新因子/特征家族在"真正好的股票"（Oracle Top-K）子集内的辨別力，
而非仅依赖全截面 Rank IC。

三条准入门槛（全部通过才进入 M5 主线）：
1. 全截面增量 Rank IC delta > 0.002
2. Oracle 条件 IC 均值 > 0.03
3. Oracle Top-20 overlap 提升 >= 2pp

用法::

    from src.features.oracle_ic_gate import oracle_conditional_ic, evaluate_newdata_admission

    gate = evaluate_newdata_admission(
        scores_baseline, scores_candidate, labels,
        oracle_topk_mask, baseline_overlap, candidate_overlap,
    )
    print(gate["passed"], gate["summary"])
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


# ── 阈值常量 ──────────────────────────────────────────────────────────────────

DELTA_RANK_IC_MIN = 0.002       # 全截面 Rank IC 增量最低要求
ORACLE_COND_IC_MEAN_MIN = 0.03  # Oracle 条件 IC 均值最低要求
ORACLE_OVERLAP_DELTA_MIN = 0.02  # Oracle overlap 提升最低要求 (2pp)


@dataclass
class OracleICGateResult:
    """Oracle 条件 IC Gate 评估结果。"""
    passed: bool
    baseline_rank_ic: float = 0.0
    candidate_rank_ic: float = 0.0
    rank_ic_delta: float = 0.0
    baseline_oracle_cond_ic: float = 0.0
    candidate_oracle_cond_ic: float = 0.0
    baseline_oracle_overlap: float = 0.0
    candidate_oracle_overlap: float = 0.0
    oracle_overlap_delta: float = 0.0
    gate_checks: dict[str, bool] = field(default_factory=dict)
    summary: str = ""
    recommendation: str = ""


def _safe_spearmanr(x, y):
    """安全的 Spearman rank correlation，处理 NaN 和常数输入。"""
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 10:
        return 0.0, 1.0
    x_valid = x[mask]
    y_valid = y[mask]
    if np.std(x_valid) < 1e-12 or np.std(y_valid) < 1e-12:
        return 0.0, 1.0
    r, p = spearmanr(x_valid, y_valid)
    return float(r) if np.isfinite(r) else 0.0, float(p) if np.isfinite(p) else 1.0


def oracle_conditional_ic(
    scores: np.ndarray,
    labels: np.ndarray,
    oracle_topk_mask: np.ndarray,
    *,
    top_k: int = 20,
) -> dict[str, float]:
    """在 Oracle Top-K 子集内计算 factor score 与未来收益的 Rank IC。

    衡量：这个因子对"真正好的股票"有多强的辨别力。

    Parameters
    ----------
    scores: 因子得分（截面内排序值或原始值）
    labels: 未来月收益
    oracle_topk_mask: bool 数组，True 表示该行属于 Oracle Top-K
    top_k: Oracle Top-K 的 k 值（仅用于日志）

    Returns
    -------
    dict with keys: oracle_cond_ic, oracle_cond_n, oracle_cond_p,
                    full_ic, full_n, oracle_mask_ratio
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=float)
    oracle_topk_mask = np.asarray(oracle_topk_mask, dtype=bool)

    # Full-sample IC
    full_ic, full_p = _safe_spearmanr(scores, labels)

    # Oracle conditional IC: only within oracle top-k subset
    oracle_mask = oracle_topk_mask.copy()
    oracle_scores = scores[oracle_mask]
    oracle_labels = labels[oracle_mask]
    oracle_ic, oracle_p = _safe_spearmanr(oracle_scores, oracle_labels)

    return {
        "oracle_cond_ic": oracle_ic,
        "oracle_cond_n": int(oracle_mask.sum()),
        "oracle_cond_p": oracle_p,
        "full_ic": full_ic,
        "full_n": int((np.isfinite(scores) & np.isfinite(labels)).sum()),
        "oracle_mask_ratio": float(oracle_mask.mean()) if len(oracle_mask) > 0 else 0.0,
    }


def compute_monthly_oracle_cond_ic(
    dataset: pd.DataFrame,
    score_col: str,
    label_col: str = "label_f1m",
    oracle_topk_col: str = "label_future_top_20pct",
) -> pd.DataFrame:
    """按 signal_date 分组计算 Oracle 条件 IC。

    Returns DataFrame with columns: signal_date, oracle_cond_ic, full_ic, oracle_n
    """
    rows = []
    for sd, part in dataset.groupby("signal_date", sort=True):
        part = part.copy()
        scores = pd.to_numeric(part[score_col], errors="coerce").values
        labels = pd.to_numeric(part[label_col], errors="coerce").values

        oracle_mask = np.zeros(len(part), dtype=bool)
        if oracle_topk_col in part.columns:
            oracle_mask = pd.to_numeric(part[oracle_topk_col], errors="coerce").fillna(0).values == 1

        result = oracle_conditional_ic(scores, labels, oracle_mask)
        result["signal_date"] = sd
        rows.append(result)

    return pd.DataFrame(rows)


def evaluate_newdata_admission(
    baseline_scores: np.ndarray,
    candidate_scores: np.ndarray,
    labels: np.ndarray,
    oracle_topk_mask: np.ndarray,
    baseline_oracle_overlap: float = 0.0,
    candidate_oracle_overlap: float = 0.0,
    *,
    delta_ic_min: float = DELTA_RANK_IC_MIN,
    oracle_cond_ic_min: float = ORACLE_COND_IC_MEAN_MIN,
    overlap_delta_min: float = ORACLE_OVERLAP_DELTA_MIN,
) -> OracleICGateResult:
    """评估新数据特征是否符合 M5 主线准入标准。

    Parameters
    ----------
    baseline_scores: 基线模型的预测得分
    candidate_scores: 加入新特征后的预测得分
    labels: 未来月收益
    oracle_topk_mask: Oracle Top-20% 标记
    baseline_oracle_overlap: 基线模型的 oracle overlap
    candidate_oracle_overlap: 加入新特征后的 oracle overlap
    """
    # 1. Full-sample Rank IC comparison
    baseline_ic, _ = _safe_spearmanr(baseline_scores, labels)
    candidate_ic, _ = _safe_spearmanr(candidate_scores, labels)
    ic_delta = candidate_ic - baseline_ic

    # 2. Oracle conditional IC
    baseline_cond = oracle_conditional_ic(baseline_scores, labels, oracle_topk_mask)
    candidate_cond = oracle_conditional_ic(candidate_scores, labels, oracle_topk_mask)

    # 3. Oracle overlap delta
    overlap_delta = candidate_oracle_overlap - baseline_oracle_overlap

    checks = {
        "rank_ic_delta": bool(ic_delta > delta_ic_min),
        "oracle_cond_ic": bool(candidate_cond["oracle_cond_ic"] > oracle_cond_ic_min),
        "oracle_overlap_delta": bool(overlap_delta >= overlap_delta_min),
    }

    passed = all(checks.values())

    # Build summary
    reasons = []
    if not checks["rank_ic_delta"]:
        reasons.append(f"Rank IC delta {ic_delta:.4f} ≤ {delta_ic_min}")
    if not checks["oracle_cond_ic"]:
        reasons.append(f"Oracle cond IC {candidate_cond['oracle_cond_ic']:.4f} ≤ {oracle_cond_ic_min}")
    if not checks["oracle_overlap_delta"]:
        reasons.append(f"Oracle overlap delta {overlap_delta:.4f} < {overlap_delta_min}")

    if passed:
        recommendation = "准入：新数据通过全部三条门槛，进入 M5 主线特征集。"
    elif sum(checks.values()) >= 2:
        recommendation = "观察：两条通过，可作为辅助特征，等待更多数据后重新评估。"
    else:
        recommendation = "拒绝：未通过足够门槛，仅保留数据链路，不入 M5。"

    return OracleICGateResult(
        passed=passed,
        baseline_rank_ic=baseline_ic,
        candidate_rank_ic=candidate_ic,
        rank_ic_delta=ic_delta,
        baseline_oracle_cond_ic=baseline_cond["oracle_cond_ic"],
        candidate_oracle_cond_ic=candidate_cond["oracle_cond_ic"],
        baseline_oracle_overlap=baseline_oracle_overlap,
        candidate_oracle_overlap=candidate_oracle_overlap,
        oracle_overlap_delta=overlap_delta,
        gate_checks=checks,
        summary="；".join(reasons) if reasons else "全部通过",
        recommendation=recommendation,
    )


def build_oracle_ic_gate_doc(gate: OracleICGateResult, family_name: str = "") -> str:
    """生成 Oracle 条件 IC Gate 的 Markdown 报告节。"""
    passed_icon = "✅" if gate.passed else "❌"
    header = f"## Oracle IC Gate: {family_name}" if family_name else "## Oracle IC Gate"

    lines = [
        header,
        "",
        f"**结果**: {passed_icon} {'通过' if gate.passed else '未通过'}",
        "",
        "| 指标 | 基线 | +新特征 | Delta | 门槛 | 通过 |",
        "|---|---|---|---|---|---|",
        f"| 全截面 Rank IC | {gate.baseline_rank_ic:.4f} | {gate.candidate_rank_ic:.4f} | {gate.rank_ic_delta:+.4f} | > {DELTA_RANK_IC_MIN} | {'✅' if gate.gate_checks.get('rank_ic_delta') else '❌'} |",
        f"| Oracle 条件 IC | {gate.baseline_oracle_cond_ic:.4f} | {gate.candidate_oracle_cond_ic:.4f} | — | > {ORACLE_COND_IC_MEAN_MIN} | {'✅' if gate.gate_checks.get('oracle_cond_ic') else '❌'} |",
        f"| Oracle Overlap | {gate.baseline_oracle_overlap:.4f} | {gate.candidate_oracle_overlap:.4f} | {gate.oracle_overlap_delta:+.4f} | >= {ORACLE_OVERLAP_DELTA_MIN} | {'✅' if gate.gate_checks.get('oracle_overlap_delta') else '❌'} |",
        "",
        f"**建议**: {gate.recommendation}",
        "",
    ]
    return "\n".join(lines)
