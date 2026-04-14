#!/usr/bin/env python3
"""
阶段三：DuckDB 日线 → OHLCV 序列（默认 30 日窗口）+ 锚定归一化，训练 GRU / LSTM / TCN / Transformer。

用法（项目根目录，conda activate quant-system）::

    python scripts/train/train_deep_sequence.py --config config.yaml --kind gru --max-symbols 400

训练完成后将 ``signals.deep_sequence.bundle_dir`` 指向输出的 ``data/models/ts_*_*/``，
``signals.sort_by`` 设为 ``deep_sequence``，并与 ``deep_sequence.seq_len`` 对齐。

Jetson 可加 ``--device cuda``；验证集默认按**交易日时间**切分（``--time-val-split``，默认开启）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_fetcher import DuckDBManager, list_default_universe_symbols
from src.features.tree_dataset import long_ohlcv_history_from_daily
from src.models.data_slice import normalize_slice_spec
from src.models.timeseries.ohlcv_norm import OHLCV_COLUMNS
from src.models.timeseries.train import train_timeseries
from src.settings import load_config, resolve_asof_trade_end


def main() -> int:
    p = argparse.ArgumentParser(description="阶段三：OHLCV 序列深度学习（GRU/LSTM/TCN/Transformer）")
    p.add_argument("--config", type=Path, default=None, help="默认项目根 config.yaml")
    p.add_argument("--max-symbols", type=int, default=None)
    p.add_argument("--symbols", type=str, default=None, help="逗号分隔 6 位代码")
    p.add_argument("--horizon", type=int, default=None, help="前瞻交易日；默认 config features.eval_forward_horizon")
    p.add_argument("--seq-len", type=int, default=None, help="序列长度；默认 config signals.deep_sequence.seq_len 或 30")
    p.add_argument(
        "--kind",
        choices=("lstm", "gru", "tcn", "transformer"),
        default=None,
        help="架构；默认 config signals.deep_sequence.kind 或 gru",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test-size", type=float, default=0.2, help="验证比例（时间切分或随机切分）")
    p.add_argument(
        "--time-val-split",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="按交易日时间切分验证集（推荐）；关闭则随机切分",
    )
    p.add_argument(
        "--walk-forward-oos",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="训练后执行 walk-forward OOS 校验（生产推荐开启）",
    )
    p.add_argument("--wf-train-days", type=int, default=252)
    p.add_argument("--wf-test-days", type=int, default=63)
    p.add_argument("--wf-step-days", type=int, default=63)
    p.add_argument("--wf-epochs", type=int, default=8)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default="", help="cuda 或 cpu，默认自动")
    p.add_argument("--hidden", type=int, default=32, help="隐藏维（默认较小以降低过拟合）")
    p.add_argument("--num-layers", type=int, default=1, help="RNN/Transformer 编码层数")
    p.add_argument("--dropout", type=float, default=0.3, help="层间与输出头前 Dropout")
    p.add_argument(
        "--target-task",
        choices=("regression", "binary_up"),
        default="regression",
        help="regression：MSE 前瞻收益；binary_up：涨跌二分类（BCE）",
    )
    p.add_argument("--d-model", type=int, default=32, help="Transformer d_model（默认与较小 hidden 一致）")
    p.add_argument("--nhead", type=int, default=4)
    p.add_argument("--num-encoder-layers", type=int, default=1)
    p.add_argument("--dim-feedforward", type=int, default=64)
    p.add_argument("--num-blocks", type=int, default=2, help="TCN 残差块数")
    p.add_argument("--model-version", default="1.0.0")
    p.add_argument("--feature-version", default="v1")
    p.add_argument("--out-root", type=str, default="")
    p.add_argument("--experiments-dir", type=str, default="")
    args = p.parse_args()

    cfg = load_config(args.config)
    paths = cfg.get("paths", {}) or {}
    feat = cfg.get("features", {})
    gpu_cfg = cfg.get("gpu", {})
    sig = cfg.get("signals", {})
    deep_sig = sig.get("deep_sequence") or {}

    lookback = int(feat.get("lookback_trading_days", 160))
    min_valid = int(feat.get("min_valid_days", 30))
    horizon = int(args.horizon if args.horizon is not None else feat.get("eval_forward_horizon", 5))
    seq_len = int(
        args.seq_len
        if args.seq_len is not None
        else deep_sig.get("seq_len", 30)
    )
    kind = str(
        args.kind
        if args.kind is not None
        else deep_sig.get("kind", "gru")
    ).lower()
    if kind not in ("lstm", "gru", "tcn", "transformer"):
        print(f"未知 kind: {kind}", file=sys.stderr)
        return 1

    device_str = str(gpu_cfg.get("device", "cpu")).lower()
    if args.device.strip():
        device_str = args.device.strip().lower()
    if device_str == "cuda" and not torch.cuda.is_available():
        device_str = "cpu"

    dtype_str = str(gpu_cfg.get("dtype", "float32")).lower()
    torch_dtype = torch.float32 if dtype_str in ("float32", "fp32") else torch.float64

    if args.symbols:
        symbols = [s.strip().zfill(6) for s in args.symbols.split(",") if s.strip()]
    else:
        symbols = list_default_universe_symbols(
            max_symbols=args.max_symbols,
            config_path=args.config,
        )
    if not symbols:
        print("无标的列表", file=sys.stderr)
        return 1

    end = resolve_asof_trade_end(paths)
    start = end - pd.offsets.BDay(lookback + 40)

    with DuckDBManager(config_path=args.config) as db:
        df = db.read_daily_frame(symbols=symbols, start=start, end=end)

    if df.empty:
        print("日线为空", file=sys.stderr)
        return 1

    panel = long_ohlcv_history_from_daily(
        df,
        horizon=horizon,
        min_valid_days=min_valid,
        device=device_str,
        dtype=torch_dtype,
    )
    if panel.empty or len(panel) < 200:
        print(f"OHLCV 面板样本过少: n={len(panel)}", file=sys.stderr)
        return 1

    y_col = f"forward_ret_{horizon}d"
    spec = normalize_slice_spec(
        symbols=symbols,
        date_start=str(panel["trade_date"].min().date()),
        date_end=str(panel["trade_date"].max().date()),
        extra={"horizon": horizon, "seq_len": seq_len, "kind": kind, "n_rows": len(panel)},
    )

    out_root = args.out_root.strip() or paths.get("models_dir", "data/models")
    if not Path(out_root).is_absolute():
        out_root = str(ROOT / out_root)
    exp_dir = args.experiments_dir.strip() or paths.get("experiments_dir", "data/experiments")
    if not Path(exp_dir).is_absolute():
        exp_dir = str(ROOT / exp_dir)

    dev = device_str if device_str in ("cuda", "cpu") else None

    tt = str(deep_sig.get("target_task", args.target_task)).lower()
    if tt not in ("regression", "binary_up"):
        tt = "regression"

    res = train_timeseries(
        panel,
        kind=kind,  # type: ignore[arg-type]
        feature_columns=list(OHLCV_COLUMNS),
        target_column=y_col,
        seq_len=seq_len,
        model_version=args.model_version,
        feature_version=args.feature_version,
        training_seed=args.seed,
        test_size=args.test_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=dev,
        hidden=args.hidden,
        num_layers=args.num_layers,
        kernel=3,
        num_blocks=args.num_blocks,
        dropout=args.dropout,
        normalize_mode="ohlcv_anchor",
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers,
        dim_feedforward=args.dim_feedforward,
        max_seq_len=max(seq_len, 128),
        time_val_split=bool(args.time_val_split),
        target_task=tt,  # type: ignore[arg-type]
        walk_forward_oos=bool(args.walk_forward_oos),
        wf_train_days=int(args.wf_train_days),
        wf_test_days=int(args.wf_test_days),
        wf_step_days=int(args.wf_step_days),
        wf_epochs=int(args.wf_epochs),
        slice_spec=spec,
        out_root=out_root,
        experiments_dir=exp_dir,
    )
    print(
        json.dumps(
            {
                "bundle_dir": str(res.bundle_dir),
                "metrics": res.metrics,
                "n_samples": len(panel),
                "seq_len": seq_len,
                "kind": kind,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
