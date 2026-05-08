"""截面打分与权重构建模块。

包含：
- 多因子截面打分（经典加权 / XGBoost）
- IC 动态权重（滚动 ICIR → 权重）
- P1 因子策略（IC 规则过滤/归零/反转）
- 权重归一化
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np
import pandas as pd

from src.features.ic_monitor import ICMonitor
from src.models.rank_score import sort_key_for_dataframe
from src.pipeline.factor_computer import _zscore_clip


# ── 截面打分 ─────────────────────────────────────────────────────────────

def build_score(
    factors: pd.DataFrame,
    weights: Dict[str, float],
    *,
    weights_by_date: Dict[pd.Timestamp, Dict[str, float]] | None = None,
    universe_eligible_col: str | None = "_universe_eligible",
    sort_by: str = "composite_extended",
    tree_bundle_dir: str | None = None,
    tree_raw_features: Iterable[str] | None = None,
    tree_rsi_mode: str = "level",
) -> pd.DataFrame:
    mode = str(sort_by).lower().strip()
    fac_cols = [c for c in weights.keys() if c in factors.columns]
    rows = []
    for dt, g in factors.groupby("trade_date"):
        if universe_eligible_col and universe_eligible_col in g.columns:
            g = g.loc[g[universe_eligible_col].to_numpy(dtype=bool)]
        if mode == "xgboost":
            raw = [str(c) for c in (tree_raw_features or [])]
            if not raw:
                raise ValueError("sort_by=xgboost 需要 tree_raw_features")
            g = g.dropna(subset=raw, how="all").copy()
        else:
            g = g.dropna(subset=fac_cols, how="all").copy()
        if len(g) < 10:
            continue
        if mode == "xgboost":
            ranked = sort_key_for_dataframe(
                g, sort_by="xgboost",
                tree_bundle_dir=tree_bundle_dir,
                tree_raw_features=list(tree_raw_features or []),
                tree_rsi_mode=tree_rsi_mode,
            )
            rows.append(pd.DataFrame({
                "symbol": ranked["symbol"].values,
                "trade_date": dt,
                "score": pd.to_numeric(ranked["tree_score"], errors="coerce").to_numpy(dtype=np.float64),
            }))
            continue
        effective_weights = weights_by_date.get(pd.Timestamp(dt), weights) if weights_by_date else weights
        active_cols = []
        for fc in fac_cols:
            col = pd.to_numeric(g[fc], errors="coerce")
            m = col.notna() & np.isfinite(col)
            if m.sum() >= 5 and abs(float(effective_weights.get(fc, 0.0))) > 1e-15:
                active_cols.append(fc)
        if not active_cols:
            continue
        abs_sum = float(sum(abs(float(effective_weights.get(fc, 0.0))) for fc in active_cols))
        if abs_sum <= 1e-15:
            continue
        score = pd.Series(0.0, index=g.index)
        for fc in fac_cols:
            if fc not in active_cols:
                continue
            col = pd.to_numeric(g[fc], errors="coerce")
            m = col.notna() & np.isfinite(col)
            if m.sum() < 5:
                continue
            w = float(effective_weights.get(fc, 0.0)) / abs_sum
            score[m] += _zscore_clip(col[m]) * w
        rows.append(pd.DataFrame({"symbol": g["symbol"].values, "trade_date": dt, "score": score.values}))
    if not rows:
        raise RuntimeError("得分构建为空")
    return pd.concat(rows, ignore_index=True).dropna(subset=["score"])


# ── 权重归一化 ───────────────────────────────────────────────────────────

def normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    """归一化 composite_extended 权重：零权重因子剔除后按绝对值归一。"""
    cleaned = {str(k): float(v) for k, v in weights.items() if abs(float(v)) > 1e-12}
    s = sum(abs(v) for v in cleaned.values())
    if s <= 0:
        raise ValueError("composite_extended 权重和为 0")
    return {k: v / s for k, v in cleaned.items()}


# ── P1 因子策略 ──────────────────────────────────────────────────────────

def load_factor_ic_summary(ic_report_path: str) -> pd.DataFrame:
    """从 CSV 或 JSON 加载因子 IC 汇总表。"""
    import json as _json

    p = Path(ic_report_path).expanduser()
    if not p.is_absolute():
        return pd.DataFrame()
    if not p.exists():
        return pd.DataFrame()
    if p.suffix.lower() == ".csv":
        tab = pd.read_csv(p, encoding="utf-8-sig")
    elif p.suffix.lower() == ".json":
        payload = _json.loads(p.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and isinstance(payload.get("summary"), list):
            tab = pd.DataFrame(payload["summary"])
        else:
            tab = pd.DataFrame(payload)
    else:
        return pd.DataFrame()
    need = {"factor", "horizon_key", "ic_mean"}
    if not need.issubset(set(tab.columns)):
        return pd.DataFrame()
    tab["factor"] = tab["factor"].astype(str)
    tab["horizon_key"] = tab["horizon_key"].astype(str)
    tab["ic_mean"] = pd.to_numeric(tab["ic_mean"], errors="coerce")
    return tab.dropna(subset=["factor", "horizon_key", "ic_mean"]).copy()


def _normalize_clip(weights: Dict[str, float], clip_abs_weight: float) -> Dict[str, float]:
    if not weights:
        return {}
    c = float(max(1e-6, clip_abs_weight))
    clipped = {k: float(np.clip(v, -c, c)) for k, v in weights.items()}
    s = float(sum(abs(v) for v in clipped.values()))
    if s <= 1e-12:
        return {}
    return {k: v / s for k, v in clipped.items()}


def _icir_from_history(ic_hist: pd.Series, half_life: float) -> float:
    ser = pd.to_numeric(ic_hist, errors="coerce").dropna()
    if ser.empty:
        return float("nan")
    n = len(ser)
    decay = float(np.exp(np.log(0.5) / max(float(half_life), 1.0)))
    w = decay ** np.arange(n - 1, -1, -1, dtype=np.float64)
    w = w / np.sum(w)
    x = ser.to_numpy(dtype=np.float64)
    mu = float(np.sum(w * x))
    var = float(np.sum(w * (x - mu) ** 2))
    sd = float(np.sqrt(max(var, 0.0)))
    if sd < 1e-12:
        return float("nan")
    return mu / sd


def _rolling_icir_series(
    ic_ser: pd.Series,
    *,
    window: int,
    min_obs: int,
    half_life: float,
) -> pd.Series:
    values = pd.to_numeric(ic_ser, errors="coerce").to_numpy(dtype=np.float64)
    out = np.full(len(values), np.nan, dtype=np.float64)
    history: list[float] = []
    for idx, val in enumerate(values):
        if np.isfinite(val):
            history.append(float(val))
        if len(history) < int(min_obs):
            continue
        hist = pd.Series(history[-int(window):], dtype=np.float64)
        icir = _icir_from_history(hist, half_life=half_life)
        if np.isfinite(icir):
            out[idx] = float(icir)
    return pd.Series(out, index=ic_ser.index, dtype=np.float64)


def build_weights_by_date(
    ic_df: pd.DataFrame,
    *,
    window: int,
    min_obs: int,
    half_life: float,
    clip_abs_weight: float,
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    if ic_df.empty:
        return out
    tab = ic_df.copy()
    tab["trade_date"] = pd.to_datetime(tab["trade_date"]).dt.normalize()
    tab["factor"] = tab["factor"].astype(str)
    wide = (
        tab.pivot_table(index="trade_date", columns="factor", values="ic", aggfunc="last")
        .sort_index()
        .sort_index(axis=1)
    )
    if wide.empty:
        return out
    icir_wide = pd.DataFrame(index=wide.index)
    for fac in wide.columns:
        icir_wide[str(fac)] = _rolling_icir_series(
            wide[fac],
            window=window,
            min_obs=min_obs,
            half_life=half_life,
        )
    for dt, row in icir_wide.iterrows():
        raw = {str(fac): float(val) for fac, val in row.items() if np.isfinite(float(val))}
        normed = _normalize_clip(raw, clip_abs_weight=clip_abs_weight)
        if normed:
            out[pd.Timestamp(dt).strftime("%Y-%m-%d")] = normed
    return out


def apply_p1_factor_policy(
    base_weights: Dict[str, float],
    ic_summary: pd.DataFrame,
    *,
    remove_if_t1_and_t21_negative: bool = True,
    zero_if_abs_t1_below: float = 0.0,
    flip_if_t1_negative_and_t21_above: float = 0.005,
) -> tuple[Dict[str, float], pd.DataFrame]:
    """按 P1 IC 规则过滤/归零/反转因子权重。"""
    if not base_weights or ic_summary.empty:
        return dict(base_weights), pd.DataFrame(columns=["factor", "action", "ic_mean_t1", "ic_mean_t21"])

    piv = (
        ic_summary.pivot_table(index="factor", columns="horizon_key", values="ic_mean", aggfunc="first")
        .rename_axis(None, axis=1)
        .reset_index()
    )
    if "tplus1_open_1d" not in piv.columns:
        piv["tplus1_open_1d"] = np.nan
    if "close_21d" not in piv.columns:
        piv["close_21d"] = np.nan

    piv = piv.set_index("factor")
    updated = dict(base_weights)
    rows: list[dict[str, Any]] = []
    for fac, cur_w_raw in base_weights.items():
        row = piv.loc[fac] if fac in piv.index else None
        t1 = float(row["tplus1_open_1d"]) if row is not None and pd.notna(row["tplus1_open_1d"]) else float("nan")
        t21 = float(row["close_21d"]) if row is not None and pd.notna(row["close_21d"]) else float("nan")
        action = "keep"
        cur_w = float(cur_w_raw)
        if not np.isfinite(t1) and not np.isfinite(t21):
            updated[fac] = 0.0
            action = "remove"
        elif (
            bool(remove_if_t1_and_t21_negative)
            and np.isfinite(t1)
            and np.isfinite(t21)
            and t1 < 0.0
            and t21 < 0.0
        ):
            updated[fac] = 0.0
            action = "remove"
        elif np.isfinite(t1) and abs(t1) < float(zero_if_abs_t1_below):
            updated[fac] = 0.0
            action = "zero"
        elif (
            np.isfinite(t1)
            and np.isfinite(t21)
            and t1 < 0.0
            and t21 > float(flip_if_t1_negative_and_t21_above)
            and abs(cur_w) > 1e-12
        ):
            updated[fac] = -cur_w
            action = "flip"
        rows.append(
            {
                "factor": fac,
                "action": action,
                "ic_mean_t1": t1,
                "ic_mean_t21": t21,
            }
        )
    try:
        normalized = normalize_weights(updated)
    except ValueError:
        normalized = dict(base_weights)
    return normalized, pd.DataFrame(rows).sort_values(["action", "factor"]).reset_index(drop=True)


# ── IC 动态权重 ──────────────────────────────────────────────────────────

def load_ic_weights_by_date(ic_weights_json: str) -> Dict[pd.Timestamp, Dict[str, float]]:
    """从 JSON 文件加载按日期的 IC 动态权重。"""
    import json as _json

    p = Path(ic_weights_json).expanduser()
    if not p.exists():
        return {}
    try:
        payload = _json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}
    rows = payload.get("weights_by_date")
    if not isinstance(rows, dict):
        return {}
    out: Dict[pd.Timestamp, Dict[str, float]] = {}
    for k, v in rows.items():
        if not isinstance(v, dict):
            continue
        dt = pd.to_datetime(k, errors="coerce")
        if pd.isna(dt):
            continue
        out[pd.Timestamp(dt).normalize()] = {str(f): float(w) for f, w in v.items()}
    return out


def build_ic_weights_from_monitor(
    monitor_path: str,
    *,
    window: int,
    min_obs: int,
    half_life: float,
    clip_abs_weight: float,
) -> Dict[pd.Timestamp, Dict[str, float]]:
    """从 IC monitor 日志构建按日期的动态权重。"""
    p = Path(monitor_path).expanduser()
    if not p.exists():
        return {}
    mon = ICMonitor(p)
    ic_df = mon.load_dataframe()
    if ic_df.empty:
        return {}
    raw_out = build_weights_by_date(
        ic_df,
        window=window,
        min_obs=min_obs,
        half_life=half_life,
        clip_abs_weight=clip_abs_weight,
    )
    return {
        pd.Timestamp(dt).normalize(): {str(f): float(w) for f, w in weights.items()}
        for dt, weights in raw_out.items()
    }
