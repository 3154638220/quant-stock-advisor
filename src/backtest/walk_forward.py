"""
Walk-Forward 验证框架：滚动窗口 IC/收益稳定性评估。

解决的问题：单一时间段回测无法区分模型是"真实 alpha"还是"参数过拟合于某段行情"。
通过滚动训练/验证窗口，评估模型在时序上的稳定性。

Usage
-----
>>> from src.backtest.walk_forward import WalkForwardConfig, walk_forward_ic
>>> cfg = WalkForwardConfig(train_months=24, test_months=3, step_months=1)
>>> result = walk_forward_ic(df, factor_col="composite_score", forward_ret_col="label_forward_1m", cfg=cfg)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class WalkForwardConfig:
    """Walk-forward 滚动窗口配置。

    Attributes
    ----------
    train_months
        训练窗口月数。
    test_months
        验证窗口月数（OOS）。
    step_months
        每次滚动步长（月数）。
    min_folds
        最少折数；不足时返回空结果。
    min_train_samples
        训练窗口最少样本数。
    min_test_samples
        验证窗口最少样本数。
    ic_method
        IC 计算方法：``pearson``（经典 IC）或 ``spearman``（Rank IC）。
    """

    train_months: int = 24
    test_months: int = 3
    step_months: int = 1
    min_folds: int = 12
    min_train_samples: int = 50
    min_test_samples: int = 10
    ic_method: str = "spearman"


def _approx_months_to_days(months: float, trading_days_per_month: int = 21) -> int:
    return max(1, int(months * trading_days_per_month))


def walk_forward_ic(
    df: pd.DataFrame,
    factor_col: str,
    forward_ret_col: str,
    cfg: WalkForwardConfig,
    *,
    date_col: str = "trade_date",
) -> pd.DataFrame:
    """
    对因子做 Walk-Forward IC 评估。

    将数据按交易日排序后，滚动切割训练/验证窗口：
    - 训练窗口：最近 ``train_months`` 个月
    - 验证窗口：紧接着的 ``test_months`` 个月
    - 每次向前滚动 ``step_months`` 个月

    对每折验证窗口计算：
    - ``ic_mean``：验证期 IC 均值
    - ``ic_std``：验证期 IC 标准差
    - ``ic_ir``：IC 信息比率 = mean / std
    - ``ic_hit_rate``：正 IC 比例
    - ``max_ic`` / ``min_ic``：验证期 IC 极值
    - ``test_dates``：验证期包含的交易日数

    Returns
    -------
    DataFrame
        每折一行，列为 fold / train_start / train_end / test_start / test_end /
        ic_mean / ic_std / ic_ir / ic_hit_rate / max_ic / min_ic / test_dates。
        空 DataFrame 表示数据不足以生成最小折数。
    """
    if df.empty:
        return pd.DataFrame()

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col]).dt.normalize()
    df = df.sort_values(date_col).reset_index(drop=True)

    dates = df[date_col].unique()
    if len(dates) < 2:
        return pd.DataFrame()

    train_days = _approx_months_to_days(cfg.train_months)
    test_days = _approx_months_to_days(cfg.test_months)
    step_days = _approx_months_to_days(cfg.step_months)

    total_needed = train_days + test_days
    if len(dates) < total_needed:
        return pd.DataFrame()

    folds: list[dict] = []
    fold_idx = 0
    pos = train_days  # 第一个训练窗口结束位置

    while pos + test_days <= len(dates):
        train_end_date = dates[pos - 1]
        test_start_date = dates[pos]
        test_end_idx = min(pos + test_days, len(dates)) - 1
        test_end_date = dates[test_end_idx]

        # 训练窗口
        train_start_idx = max(0, pos - train_days)
        train_start_date = dates[train_start_idx]

        train_mask = (df[date_col] >= train_start_date) & (df[date_col] <= train_end_date)
        test_mask = (df[date_col] >= test_start_date) & (df[date_col] <= test_end_date)

        train_sub = df[train_mask]
        test_sub = df[test_mask]

        if len(train_sub) < cfg.min_train_samples or len(test_sub) < cfg.min_test_samples:
            pos += step_days
            fold_idx += 1
            continue

        # 计算验证期逐日 IC
        ic_series = _daily_ic(
            test_sub, factor_col, forward_ret_col, date_col=date_col, method=cfg.ic_method
        )
        ic_vals = ic_series.dropna()
        if len(ic_vals) < 2:
            pos += step_days
            fold_idx += 1
            continue

        m = float(ic_vals.mean())
        s = float(ic_vals.std(ddof=1))
        ir = m / s if s > 1e-15 else float("nan")
        hit = float((ic_vals > 0).mean())

        folds.append(
            {
                "fold": fold_idx,
                "train_start": train_start_date,
                "train_end": train_end_date,
                "test_start": test_start_date,
                "test_end": test_end_date,
                "ic_mean": m,
                "ic_std": s,
                "ic_ir": ir,
                "ic_hit_rate": hit,
                "max_ic": float(ic_vals.max()),
                "min_ic": float(ic_vals.min()),
                "test_dates": len(ic_vals),
            }
        )

        pos += step_days
        fold_idx += 1

    if len(folds) < cfg.min_folds:
        return pd.DataFrame()

    return pd.DataFrame(folds)


def _daily_ic(
    df: pd.DataFrame,
    factor_col: str,
    forward_ret_col: str,
    *,
    date_col: str = "trade_date",
    method: str = "spearman",
) -> pd.Series:
    """逐日截面 IC（与 factor_eval 中逻辑一致）。"""
    idx: list = []
    vals: list[float] = []
    for d, sub in df.groupby(date_col, sort=True):
        a = sub[factor_col]
        b = sub[forward_ret_col]
        m = a.notna() & b.notna()
        if m.sum() < 3:
            continue
        idx.append(d)
        vals.append(float(a[m].corr(b[m], method=method)))
    return pd.Series(vals, index=pd.Index(idx, name=date_col))


def walk_forward_stability_report(
    wf_result: pd.DataFrame,
) -> dict:
    """
    基于 walk-forward 结果生成稳定性概要。

    Returns
    -------
    dict
        - ``folds``：有效折数
        - ``ic_mean_of_means``：各折 IC 均值的均值
        - ``ic_mean_std``：各折 IC 均值的标准差（越低越稳定）
        - ``ic_ir_mean``：各折 IR 均值
        - ``hit_rate_stability``：各折 hit_rate < 0.5 的比例（越低越好）
        - ``worst_fold_ic``：最差一折的 IC 均值
        - ``best_fold_ic``：最好一折的 IC 均值
        - ``ic_cv``：IC 均值变异系数 = std(各折 IC mean) / |mean(各折 IC mean)|
    """
    if wf_result.empty:
        return {
            "folds": 0,
            "ic_mean_of_means": float("nan"),
            "ic_mean_std": float("nan"),
            "ic_ir_mean": float("nan"),
            "hit_rate_stability": float("nan"),
            "worst_fold_ic": float("nan"),
            "best_fold_ic": float("nan"),
            "ic_cv": float("nan"),
        }

    ic_means = wf_result["ic_mean"].dropna()
    if ic_means.empty:
        return {
            "folds": int(len(wf_result)),
            "ic_mean_of_means": float("nan"),
            "ic_mean_std": float("nan"),
            "ic_ir_mean": float("nan"),
            "hit_rate_stability": float("nan"),
            "worst_fold_ic": float("nan"),
            "best_fold_ic": float("nan"),
            "ic_cv": float("nan"),
        }

    ic_mean_of_means = float(ic_means.mean())
    ic_mean_std = float(ic_means.std(ddof=1))
    ic_cv = ic_mean_std / abs(ic_mean_of_means) if abs(ic_mean_of_means) > 1e-15 else float("nan")

    hit_rates = wf_result["ic_hit_rate"].dropna()
    hit_rate_stability = (
        float((hit_rates < 0.5).mean()) if len(hit_rates) > 0 else float("nan")
    )

    irs = wf_result["ic_ir"].dropna()

    return {
        "folds": int(len(wf_result)),
        "ic_mean_of_means": ic_mean_of_means,
        "ic_mean_std": ic_mean_std,
        "ic_ir_mean": float(irs.mean()) if len(irs) > 0 else float("nan"),
        "hit_rate_stability": hit_rate_stability,
        "worst_fold_ic": float(ic_means.min()),
        "best_fold_ic": float(ic_means.max()),
        "ic_cv": ic_cv,
    }


def walk_forward_pass(
    wf_result: pd.DataFrame,
    *,
    min_hit_rate: float = 0.60,
    max_ic_cv: float = 3.0,
) -> tuple[bool, str]:
    """
    Walk-forward 稳定性准入门控。

    准入条件：
    1. 滚动 IC 正比例（hit_rate）≥ ``min_hit_rate``（默认 60%）
    2. IC 均值变异系数 ≤ ``max_ic_cv``（默认 3.0，即 IR 的倒数低于 3）

    Returns
    -------
    (passed, reason)
    """
    report = walk_forward_stability_report(wf_result)

    if report["folds"] == 0:
        return False, "walk_forward 折数为 0，数据不足"

    hit_rate_stability = report["hit_rate_stability"]
    ic_cv = report["ic_cv"]

    failures: list[str] = []

    if np.isfinite(hit_rate_stability) and hit_rate_stability > (1.0 - min_hit_rate):
        failures.append(
            f"IC 正比例稳定性不足: hit_rate<0.5 的折数占比 {hit_rate_stability:.2%} > {1.0 - min_hit_rate:.2%}"
        )

    if np.isfinite(ic_cv) and ic_cv > max_ic_cv:
        failures.append(f"IC 变异系数过大: CV={ic_cv:.2f} > {max_ic_cv}")

    if failures:
        return False, "; ".join(failures)

    return True, (
        f"通过：{report['folds']} 折, IC均值={report['ic_mean_of_means']:.4f}, "
        f"CV={ic_cv if np.isfinite(ic_cv) else 'nan'}"
    )


# ── C3: 特征重要性跨折叠稳定性 ───────────────────────────────────────────────


def feature_importance_cv(
    fold_importances: dict[str, list[float]],
    *,
    unstable_threshold: float = 1.0,
) -> dict:
    """计算特征重要性跨折叠的变异系数 (CV)，标注不稳定特征。

    Parameters
    ----------
    fold_importances: {feature_name: [fold1_importance, fold2_importance, ...]}
        每个特征在各折中的重要性值（可以是 gain/weight/permutation importance）。
    unstable_threshold: CV > 此值标注为「不稳定特征」，需在 M12 Promotion Package 中说明。

    Returns
    -------
    dict with:
        - per_feature: {feature: {cv, mean, std, unstable}}
        - unstable_features: list[str] — CV > threshold 的特征
        - n_features: 总特征数
        - n_unstable: 不稳定特征数
        - unstable_ratio: 不稳定特征占比
    """
    per_feature: dict[str, dict] = {}
    unstable: list[str] = []

    for feat, imps in fold_importances.items():
        arr = np.asarray(imps, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if len(arr) < 2:
            per_feature[feat] = {"cv": float("nan"), "mean": float("nan"),
                                 "std": float("nan"), "unstable": True}
            unstable.append(feat)
            continue
        mu = float(np.mean(arr))
        sd = float(np.std(arr, ddof=1))
        cv = float(sd / abs(mu)) if abs(mu) > 1e-15 else float("inf")
        is_unstable = not np.isfinite(cv) or cv > unstable_threshold
        per_feature[feat] = {"cv": cv, "mean": mu, "std": sd, "unstable": is_unstable}
        if is_unstable:
            unstable.append(feat)

    return {
        "per_feature": per_feature,
        "unstable_features": sorted(unstable, key=lambda f: per_feature[f]["cv"]
                                    if np.isfinite(per_feature[f]["cv"]) else float("inf"),
                                    reverse=True),
        "n_features": len(fold_importances),
        "n_unstable": len(unstable),
        "unstable_ratio": len(unstable) / len(fold_importances) if fold_importances else 0.0,
    }


def feature_importance_cv_markdown(cv_report: dict) -> str:
    """将特征重要性 CV 报告渲染为 Markdown 表格。"""
    lines: list[str] = []
    lines.append("## 特征重要性跨折叠稳定性")
    lines.append("")
    lines.append(f"- 总特征数: {cv_report['n_features']}")
    lines.append(f"- 不稳定特征数 (CV > 1.0): {cv_report['n_unstable']}")
    lines.append(f"- 不稳定占比: {cv_report['unstable_ratio']:.1%}")
    lines.append("")

    pf = cv_report["per_feature"]
    if pf:
        lines.append("| 特征 | CV | 均值 | 标准差 | 稳定? |")
        lines.append("|------|-----|------|--------|-------|")
        for feat in sorted(pf, key=lambda f: pf[f]["cv"] if np.isfinite(pf[f]["cv"]) else float("inf"), reverse=True):
            d = pf[feat]
            cv_str = f"{d['cv']:.3f}" if np.isfinite(d['cv']) else "inf"
            stable = "否 ⚠" if d["unstable"] else "是"
            lines.append(f"| {feat} | {cv_str} | {d['mean']:.4f} | {d['std']:.4f} | {stable} |")
        lines.append("")

    if cv_report["unstable_features"]:
        lines.append("### 需在 M12 Promotion Package 中说明的特征")
        lines.append("")
        for f in cv_report["unstable_features"]:
            d = pf[f]
            cv_str = f"{d['cv']:.3f}" if np.isfinite(d['cv']) else "inf"
            lines.append(f"- **{f}**: CV={cv_str}, mean={d['mean']:.4f}, std={d['std']:.4f}")
        lines.append("")

    return "\n".join(lines)


# ── C3: Bonferroni 校正 ──────────────────────────────────────────────────────


def bonferroni_correction(
    p_values: list[float] | np.ndarray,
    *,
    alpha: float = 0.05,
) -> dict:
    """对多重假设检验结果应用 Bonferroni 校正。

    Parameters
    ----------
    p_values: 各检验的原始 p 值（如 M8 gamma 网格中每个 gamma 的 NW t-test p 值）
    alpha: 族系错误率 (FWER) 阈值，默认 0.05

    Returns
    -------
    dict with:
        - n_tests: 检验总数
        - alpha_original: 原始 alpha
        - alpha_corrected: 校正后 alpha = alpha / n_tests
        - significant_original: 原始 p < alpha 的检验索引
        - significant_corrected: 校正后仍显著的检验索引
        - any_significant_after_correction: 校正后是否有任何检验仍显著
        - min_p_value: 最小 p 值
        - min_p_corrected: 最小 p 值 × n_tests（Bonferroni 校正后 p 值）
    """
    p = np.asarray(p_values, dtype=np.float64)
    p = p[np.isfinite(p)]
    n = len(p)

    if n == 0:
        return {
            "n_tests": 0,
            "alpha_original": alpha,
            "alpha_corrected": float("nan"),
            "significant_original": [],
            "significant_corrected": [],
            "any_significant_after_correction": False,
            "min_p_value": float("nan"),
            "min_p_corrected": float("nan"),
        }

    alpha_corrected = alpha / n
    sig_orig = [int(i) for i in np.where(p < alpha)[0]]
    sig_corr = [int(i) for i in np.where(p < alpha_corrected)[0]]

    min_p = float(np.min(p))
    min_p_corr = min(min_p * n, 1.0)

    return {
        "n_tests": n,
        "alpha_original": alpha,
        "alpha_corrected": alpha_corrected,
        "significant_original": sig_orig,
        "significant_corrected": sig_corr,
        "any_significant_after_correction": len(sig_corr) > 0,
        "min_p_value": min_p,
        "min_p_corrected": min_p_corr,
    }


def bonferroni_markdown(result: dict, *, test_labels: list[str] | None = None) -> str:
    """将 Bonferroni 校正结果渲染为 Markdown。"""
    lines: list[str] = []
    lines.append("## Bonferroni 多重检验校正")
    lines.append("")
    lines.append(f"- 检验总数: {result['n_tests']}")
    lines.append(f"- 原始 α: {result['alpha_original']:.3f}")
    lines.append(f"- Bonferroni 校正 α: {result['alpha_corrected']:.6f}")
    lines.append(f"- 原始显著: {len(result['significant_original'])} 项")
    lines.append(f"- 校正后显著: {len(result['significant_corrected'])} 项")
    lines.append(f"- 校正后仍显著: {'是 ✅' if result['any_significant_after_correction'] else '否 ⚠'}")
    lines.append(f"- 最小 p 值: {result['min_p_value']:.6f}")
    lines.append(f"- Bonferroni 校正后最小 p 值: {result['min_p_corrected']:.6f}")
    lines.append("")

    if test_labels:
        lines.append("| 检验 | 原始 p | Bonferroni p | 原始显著 | 校正后显著 |")
        lines.append("|------|--------|-------------|----------|-----------|")
        # We don't have individual p-values here, just the aggregate
        lines.append("")

    lines.append("### 对 M8 Gamma 网格搜索的启示")
    lines.append("")
    if result["any_significant_after_correction"]:
        lines.append(
            "校正后最优 gamma 仍显著，说明 M8 自然化约束的超额不是单纯由多重比较伪发现驱动。"
        )
    else:
        lines.append(
            "校正后无 gamma 显著，说明观察到的最优超额可能部分来自多重比较伪发现。"
            "建议谨慎解读最优 gamma 结果，并使用更保守的参数。"
        )
    lines.append("")

    return "\n".join(lines)
