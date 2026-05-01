#!/usr/bin/env python3
"""
时序模型训练入口：LSTM 或 TCN。

面板长表须含 ``symbol``、``trade_date``、特征列与目标列。

用法::

    python scripts/train/train_timeseries.py --csv data/panel.csv --kind lstm \\
      --features ret5,vol20 --target fwd5 --seq-len 10 --seed 42
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.data_slice import apply_time_symbol_filter, normalize_slice_spec
from src.models.timeseries.train import train_timeseries


def _parse_features(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def main() -> int:
    p = argparse.ArgumentParser(description="训练时序模型（LSTM / TCN）")
    p.add_argument("--csv", required=True)
    p.add_argument("--kind", choices=("lstm", "gru", "tcn", "transformer"), default="lstm")
    p.add_argument("--features", required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--seq-len", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument(
        "--time-val-split",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否按时间切分验证集（生产候选应开启，避免未来泄漏）",
    )
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default="", help="cuda 或 cpu，默认自动")
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--normalize", choices=("none", "ohlcv_anchor"), default="none")
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--nhead", type=int, default=4)
    p.add_argument("--num-encoder-layers", type=int, default=2)
    p.add_argument("--dim-feedforward", type=int, default=128)
    p.add_argument("--max-seq-len", type=int, default=128)
    p.add_argument(
        "--walk-forward-oos",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="训练后执行 walk-forward OOS 校验（更耗时）",
    )
    p.add_argument("--wf-train-days", type=int, default=252)
    p.add_argument("--wf-test-days", type=int, default=63)
    p.add_argument("--wf-step-days", type=int, default=63)
    p.add_argument("--wf-epochs", type=int, default=8)
    p.add_argument("--model-version", default="1.0.0")
    p.add_argument("--feature-version", default="v1")
    p.add_argument("--out-root", default="data/models")
    p.add_argument("--experiments-dir", default="data/experiments")
    p.add_argument("--symbols", default="")
    p.add_argument("--date-start", default="")
    p.add_argument("--date-end", default="")
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    feats = _parse_features(args.features)
    syms = [s.strip() for s in args.symbols.split(",") if s.strip()] or None
    d0 = args.date_start.strip() or None
    d1 = args.date_end.strip() or None
    df = apply_time_symbol_filter(df, symbols=syms, date_start=d0, date_end=d1)

    spec = normalize_slice_spec(
        symbols=syms,
        date_start=d0,
        date_end=d1,
        extra={"csv": str(Path(args.csv).resolve())},
    )
    dev = args.device.strip() or None

    res = train_timeseries(
        df,
        kind=args.kind,
        feature_columns=feats,
        target_column=args.target,
        seq_len=args.seq_len,
        model_version=args.model_version,
        feature_version=args.feature_version,
        training_seed=args.seed,
        test_size=args.test_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=dev,
        hidden=args.hidden,
        normalize_mode=args.normalize,
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers,
        dim_feedforward=args.dim_feedforward,
        max_seq_len=args.max_seq_len,
        time_val_split=bool(args.time_val_split),
        walk_forward_oos=bool(args.walk_forward_oos),
        wf_train_days=int(args.wf_train_days),
        wf_test_days=int(args.wf_test_days),
        wf_step_days=int(args.wf_step_days),
        wf_epochs=int(args.wf_epochs),
        slice_spec=spec,
        out_root=args.out_root,
        experiments_dir=args.experiments_dir,
    )
    print(json.dumps({"bundle_dir": str(res.bundle_dir), "metrics": res.metrics}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
