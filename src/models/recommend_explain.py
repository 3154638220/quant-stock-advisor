"""
荐股结果可读解释：根据排序方式与截面 z 分生成简短中文说明，供 CSV/HTML 展示。
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional

import numpy as np
import pandas as pd

FACTOR_LABELS: Dict[str, str] = {
    "momentum": "动量",
    "rsi": "RSI",
    "atr": "ATR",
    "realized_vol": "已实现波动",
    "turnover_roll_mean": "换手（滚动均值）",
    "vol_ret_corr": "量价相关性",
    "short_reversal": "短期反转",
    "vol_to_turnover": "量/换手比",
    "volume_skew_log": "成交量偏度（对数）",
    "bias_short": "短期乖离率",
    "bias_long": "长期乖离率",
    "max_single_day_drop": "最大单日跌幅",
    "recent_return": "近3日涨幅",
    "price_position": "价格位置",
    "log_market_cap": "对数流通市值",
}


def _norm_weights(weights: Mapping[str, float]) -> Dict[str, float]:
    w = {k: float(v) for k, v in weights.items() if abs(float(v)) > 1e-15}
    s = sum(abs(v) for v in w.values())
    if s <= 0:
        return {}
    return {k: v / s for k, v in w.items()}


def normalize_composite_weights(weights: Mapping[str, float]) -> Dict[str, float]:
    """与组合打分相同的按绝对值和归一化（供报告与解释复用）。"""
    return _norm_weights(weights)


def _fmt_num(x: float) -> str:
    if not np.isfinite(x):
        return "—"
    return f"{x:.3f}".rstrip("0").rstrip(".")


def build_recommend_reason(
    row: pd.Series,
    *,
    sort_by: str,
    rsi_mode: str = "level",
    composite_extended_weights: Optional[Mapping[str, float]] = None,
    w_momentum: float = 0.65,
    w_rsi: float = 0.35,
) -> str:
    """
    单行推荐理由（中文一句为主，可含分号连接子句）。
    """
    sb = str(sort_by).lower()

    if sb == "momentum":
        zm = row.get("z_momentum")
        m = row.get("momentum")
        zv = float(zm) if zm is not None and pd.notna(zm) else float("nan")
        mv = float(m) if m is not None and pd.notna(m) else float("nan")
        return (
            f"按截面动量排序入选：动量因子值 {_fmt_num(mv)}，"
            f"当日全市场截面 z 分 {_fmt_num(zv)}（z 越高表示相对全样本动量越强）。"
        )

    if sb == "rsi":
        rsi = row.get("rsi")
        zr = row.get("z_rsi")
        rv = float(rsi) if rsi is not None and pd.notna(rsi) else float("nan")
        zv = float(zr) if zr is not None and pd.notna(zr) else float("nan")
        mode_note = "水平越高相对越强" if str(rsi_mode).lower() == "level" else "偏离 50 的均值回归方向"
        return (
            f"按 RSI 排序入选：RSI={_fmt_num(rv)}，截面 z 分 {_fmt_num(zv)}（{mode_note}）。"
        )

    if sb == "composite":
        zm = row.get("z_momentum")
        zr = row.get("z_rsi")
        cs = row.get("composite_score")
        wm = max(0.0, float(w_momentum))
        wr = max(0.0, float(w_rsi))
        s = wm + wr
        if s > 0:
            wm, wr = wm / s, wr / s
        c_m = (
            wm * float(zm)
            if zm is not None and pd.notna(zm)
            else float("nan")
        )
        c_r = wr * float(zr) if zr is not None and pd.notna(zr) else float("nan")
        parts = []
        if np.isfinite(c_m):
            parts.append(f"动量贡献约 {_fmt_num(c_m)}")
        if np.isfinite(c_r):
            parts.append(f"RSI 贡献约 {_fmt_num(c_r)}")
        tail = "；".join(parts) if parts else "因子贡献"
        csv = (
            float(cs)
            if cs is not None and pd.notna(cs)
            else float("nan")
        )
        return (
            f"动量与 RSI 经截面标准化后加权（权重 {wm:.0%}/{wr:.0%}），"
            f"综合分 {_fmt_num(csv)}；{tail}。"
        )

    if sb == "composite_extended":
        cw = composite_extended_weights or {}
        wn = _norm_weights(cw)
        if not wn:
            return "composite_extended 排序但配置权重为空。"
        contribs: list[tuple[str, float, float, float]] = []
        for name, wc in wn.items():
            zcol = f"z_{name}"
            if zcol not in row.index:
                continue
            zv = row[zcol]
            if zv is None or pd.isna(zv):
                continue
            zf = float(zv)
            c = wc * zf
            label = FACTOR_LABELS.get(name, name)
            contribs.append((label, wc, zf, c))
        if not contribs:
            return "多因子组合得分领先（截面 z 加权），但缺少 z 分列。"
        contribs.sort(key=lambda x: x[3], reverse=True)
        top = contribs[:3]
        parts = [
            f"{t[0]}(w×z≈{_fmt_num(t[3])}, z={_fmt_num(t[2])})"
            for t in top
        ]
        ces = row.get("composite_extended_score")
        score_txt = (
            _fmt_num(float(ces))
            if ces is not None and pd.notna(ces)
            else "—"
        )
        head = "、".join(parts)
        return (
            f"多因子扩展组合（截面 z 按配置加权），综合分 {score_txt}；"
            f"贡献较大的项：{head}。"
        )

    if sb == "xgboost":
        ts = row.get("tree_score")
        zcols = [c for c in row.index if str(c).startswith("z_")]
        pairs: list[tuple[str, float]] = []
        for c in zcols:
            v = row[c]
            if v is not None and pd.notna(v):
                raw = str(c)[2:]
                pairs.append((FACTOR_LABELS.get(raw, raw), float(v)))
        pairs.sort(key=lambda x: abs(x[1]), reverse=True)
        top = pairs[:3]
        ztxt = "，".join(f"{a}(z={_fmt_num(b)})" for a, b in top) if top else "（无 z 分列）"
        tsv = (
            _fmt_num(float(ts))
            if ts is not None and pd.notna(ts)
            else "—"
        )
        return f"树模型（XGBoost）打分排序，模型分 {tsv}；因子截面 z 突出项：{ztxt}。"

    if sb == "deep_sequence":
        ds = row.get("deep_sequence_score")
        dsv = (
            _fmt_num(float(ds))
            if ds is not None and pd.notna(ds)
            else "—"
        )
        return (
            f"深度学习序列模型（OHLCV）预测得分排序，模型分 {dsv}；"
            f"表示在窗口序列特征上相对其他标的的预测排序位置。"
        )

    return f"排序方式 {sort_by!r}（未生成专用说明）。"


def build_recommend_reason_column(
    df: pd.DataFrame,
    *,
    sort_by: str,
    rsi_mode: str = "level",
    composite_extended_weights: Optional[Mapping[str, float]] = None,
    w_momentum: float = 0.65,
    w_rsi: float = 0.35,
) -> pd.Series:
    return df.apply(
        lambda r: build_recommend_reason(
            r,
            sort_by=sort_by,
            rsi_mode=rsi_mode,
            composite_extended_weights=composite_extended_weights,
            w_momentum=w_momentum,
            w_rsi=w_rsi,
        ),
        axis=1,
    )
