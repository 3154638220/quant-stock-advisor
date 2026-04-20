#!/usr/bin/env python3
"""
阶段三评估：对深度序列模型在独立测试窗口计算截面 Rank IC。

用法（项目根目录，conda activate quant-system）::

    python scripts/train/eval_deep_sequence.py --config config.yaml --test-days 63
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_fetcher import DuckDBManager, list_default_universe_symbols
from src.features.tree_dataset import long_ohlcv_history_from_daily
from src.models.timeseries.ohlcv_norm import normalize_ohlcv_anchor
from src.models.timeseries.train import build_panel_sequences_with_dates, load_timeseries_bundle
from src.settings import load_config, resolve_asof_trade_end


def _mean_rank_ic_by_date(pred: np.ndarray, y: np.ndarray, dates: np.ndarray) -> dict:
    vals: list[float] = []
    for d in np.sort(np.unique(dates)):
        m = dates == d
        if int(np.sum(m)) < 2:
            continue
        p = np.asarray(pred[m], dtype=np.float64)
        yy = np.asarray(y[m], dtype=np.float64)
        if not np.isfinite(p).all() or not np.isfinite(yy).all():
            continue
        rp = pd.Series(p).rank(method="average").to_numpy(dtype=np.float64)
        ry = pd.Series(yy).rank(method="average").to_numpy(dtype=np.float64)
        if np.std(rp) < 1e-15 or np.std(ry) < 1e-15:
            continue
        ic = float(np.corrcoef(rp, ry)[0, 1])
        if np.isfinite(ic):
            vals.append(ic)
    arr = np.asarray(vals, dtype=np.float64)
    if arr.size == 0:
        return {
            "rank_ic_mean": float("nan"),
            "rank_ic_median": float("nan"),
            "rank_ic_p25": float("nan"),
            "rank_ic_p75": float("nan"),
            "n_dates": 0,
        }
    return {
        "rank_ic_mean": float(np.nanmean(arr)),
        "rank_ic_median": float(np.nanmedian(arr)),
        "rank_ic_p25": float(np.nanquantile(arr, 0.25)),
        "rank_ic_p75": float(np.nanquantile(arr, 0.75)),
        "n_dates": int(arr.size),
    }


def main() -> int:
    p = argparse.ArgumentParser(description="评估 deep_sequence 工件在独立测试窗口的 Rank IC")
    p.add_argument("--config", type=Path, default=None, help="默认项目根 config.yaml")
    p.add_argument("--bundle-dir", type=str, default="", help="覆盖 config.signals.deep_sequence.bundle_dir")
    p.add_argument("--max-symbols", type=int, default=None, help="仅评估前 N 只标的（调试用）")
    p.add_argument("--history-days", type=int, default=700, help="回溯交易日长度（需覆盖 train+test）")
    p.add_argument("--horizon", type=int, default=None, help="标签 horizon；默认 config features.eval_forward_horizon")
    p.add_argument("--test-days", type=int, default=63, help="独立测试窗口交易日数")
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--map-location", type=str, default="", help="cpu/cuda；默认从配置推断")
    p.add_argument("--min-rank-ic", type=float, default=0.0, help="通过阈值（默认 >0）")
    p.add_argument("--json-out", type=Path, default=None, help="可选：把评估结果写入 JSON")
    args = p.parse_args()

    cfg = load_config(args.config)
    paths = cfg.get("paths", {}) or {}
    feat = cfg.get("features", {}) or {}
    deep_cfg = (cfg.get("signals", {}) or {}).get("deep_sequence", {}) or {}

    bundle_s = args.bundle_dir.strip() or str(deep_cfg.get("bundle_dir", "")).strip()
    if not bundle_s:
        print("缺少 deep_sequence.bundle_dir", file=sys.stderr)
        return 1
    bundle = Path(bundle_s)
    if not bundle.is_absolute():
        bundle = ROOT / bundle
    if not bundle.is_dir():
        print(f"工件目录不存在: {bundle}", file=sys.stderr)
        return 1

    if args.map_location.strip():
        map_loc = args.map_location.strip().lower()
    else:
        map_loc = str(deep_cfg.get("map_location", "cpu")).lower()
        if map_loc not in ("cpu", "cuda"):
            map_loc = "cpu"
    if map_loc == "cuda" and not torch.cuda.is_available():
        map_loc = "cpu"

    model, inf, meta, extra = load_timeseries_bundle(bundle, map_location=map_loc)
    seq_len = int(inf.seq_len or extra.get("seq_len", 30))
    horizon = int(args.horizon if args.horizon is not None else feat.get("eval_forward_horizon", 5))
    target_col = str(inf.target_column or f"forward_ret_{horizon}d")

    symbols = list_default_universe_symbols(max_symbols=args.max_symbols, config_path=args.config)
    if not symbols:
        print("无可评估标的", file=sys.stderr)
        return 1

    end = resolve_asof_trade_end(paths)
    start = end - pd.offsets.BDay(int(args.history_days))
    with DuckDBManager(config_path=args.config) as db:
        daily = db.read_daily_frame(symbols=symbols, start=start, end=end)
    if daily.empty:
        print("日线数据为空，请先更新 DuckDB", file=sys.stderr)
        return 1

    panel = long_ohlcv_history_from_daily(
        daily,
        horizon=horizon,
        min_valid_days=max(20, seq_len + 2),
    )
    if panel.empty:
        print("构建 OHLCV 面板为空", file=sys.stderr)
        return 1

    X, y, end_dates = build_panel_sequences_with_dates(
        panel,
        feature_columns=list(inf.feature_columns),
        target_column=target_col,
        seq_len=seq_len,
    )
    if str(inf.normalize or "").lower() == "ohlcv_anchor":
        X = normalize_ohlcv_anchor(X)

    dates_u = np.sort(np.unique(end_dates))
    if len(dates_u) < int(args.test_days):
        print("测试窗口不足，请增大 --history-days 或减小 --test-days", file=sys.stderr)
        return 1
    test_start = dates_u[-int(args.test_days)]
    test_mask = end_dates >= test_start
    if int(np.sum(test_mask)) < 10:
        print("独立测试样本过少", file=sys.stderr)
        return 1

    X_te = torch.from_numpy(X[test_mask]).float()
    y_te = np.asarray(y[test_mask], dtype=np.float64)
    d_te = end_dates[test_mask]
    if str(extra.get("target_task", inf.extra.get("target_task", "regression"))).lower() == "binary_up":
        y_te = (y_te > 0.0).astype(np.float64)

    model.eval()
    model.to(map_loc)
    preds: list[np.ndarray] = []
    bs = max(1, int(args.batch_size))
    with torch.no_grad():
        for i in range(0, len(X_te), bs):
            xb = X_te[i : i + bs].to(map_loc)
            pb = model(xb).detach().cpu().numpy()
            preds.append(pb)
    pred = np.concatenate(preds, axis=0).ravel().astype(np.float64)

    ric = _mean_rank_ic_by_date(pred, y_te, d_te)
    payload = {
        "bundle_dir": str(bundle.resolve()),
        "model_type": meta.model_type,
        "seq_len": int(seq_len),
        "horizon": int(horizon),
        "test_days": int(args.test_days),
        "n_samples": int(len(pred)),
        **ric,
    }

    if args.json_out is not None:
        out_path = args.json_out
        if not out_path.is_absolute():
            out_path = ROOT / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(payload, ensure_ascii=False, indent=2))

    rank_ic_mean = float(payload.get("rank_ic_mean", float("nan")))
    if not np.isfinite(rank_ic_mean) or rank_ic_mean <= float(args.min_rank_ic):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
