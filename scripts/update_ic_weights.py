#!/usr/bin/env python3
"""
根据 IC 监控文件生成动态因子权重（P2-A）。

输出 JSON：
- weights: 最新交易日可用权重
- weights_by_date: 历史逐日权重（用于无前视回测）
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features.ic_monitor import ICMonitor


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
    # 近期样本更高权重：对均值与方差都使用同一组 EW 权重。
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
        raw = {
            str(fac): float(val)
            for fac, val in row.items()
            if np.isfinite(float(val))
        }
        normed = _normalize_clip(raw, clip_abs_weight=clip_abs_weight)
        if normed:
            out[pd.Timestamp(dt).strftime("%Y-%m-%d")] = normed
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="根据 ic_monitor.json 生成动态 ICIR 权重")
    p.add_argument("--monitor-path", default="data/logs/ic_monitor.json", help="IC 监控 JSON 路径")
    p.add_argument("--output-path", default="data/cache/ic_weights.json", help="输出权重 JSON 路径")
    p.add_argument("--window", type=int, default=60, help="ICIR 统计窗口（交易日）")
    p.add_argument("--min-obs", type=int, default=20, help="最小样本数")
    p.add_argument("--half-life", type=float, default=20.0, help="指数衰减半衰期")
    p.add_argument("--clip-abs-weight", type=float, default=0.25, help="单因子绝对权重裁剪上限")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    monitor_path = Path(args.monitor_path).expanduser()
    output_path = Path(args.output_path).expanduser()
    mon = ICMonitor(monitor_path)
    ic_df = mon.load_dataframe()
    if ic_df.empty:
        print(f"[WARN] 无 IC 记录，跳过输出: {monitor_path}")
        return 1
    weights_by_date = build_weights_by_date(
        ic_df,
        window=int(args.window),
        min_obs=int(args.min_obs),
        half_life=float(args.half_life),
        clip_abs_weight=float(args.clip_abs_weight),
    )
    latest_weights = {}
    if weights_by_date:
        latest_date = sorted(weights_by_date.keys())[-1]
        latest_weights = weights_by_date[latest_date]
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "monitor_path": str(monitor_path),
        "window": int(args.window),
        "min_obs": int(args.min_obs),
        "half_life": float(args.half_life),
        "clip_abs_weight": float(args.clip_abs_weight),
        "weights": latest_weights,
        "weights_by_date": weights_by_date,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"[OK] IC 动态权重已写入: {output_path} | date_count={len(weights_by_date)} | "
        f"latest_factor_count={len(latest_weights)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
