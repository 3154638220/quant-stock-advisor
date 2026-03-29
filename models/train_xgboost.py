#!/usr/bin/env python3
"""
阶段二：用 DuckDB 历史日线构建 (symbol, date) 面板，截面 Z-score 特征 + 前瞻收益标签，训练 XGBoost。

默认使用 ``XGBRanker``（pairwise 排序，与 Top-K 荐股一致）；``--objective regression`` 可切回 ``XGBRegressor``。

用法（项目根目录，conda activate quant-system）::

    python models/train_xgboost.py --config config.yaml --max-symbols 400

训练完成后将 ``signals.tree_model.bundle_dir`` 指向输出的 ``data/models/xgboost_panel_<id>/``，
并将 ``signals.sort_by`` 设为 ``xgboost``。

Jetson 上可加 ``--gpu`` 使用 ``device=cuda``（须已安装支持 CUDA 的 xgboost）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_fetcher import DuckDBManager, list_default_universe_symbols
from src.features.tree_dataset import default_tree_factor_names, long_factor_panel_from_daily
from src.models.data_slice import normalize_slice_spec
from src.models.xtree.train import train_xgboost_panel
from src.settings import load_config, resolve_asof_trade_end


def _parse_features(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def main() -> int:
    p = argparse.ArgumentParser(description="训练截面 XGBoost（Z-score 因子 → 前瞻收益）")
    p.add_argument("--config", type=Path, default=None, help="默认项目根 config.yaml")
    p.add_argument("--max-symbols", type=int, default=None)
    p.add_argument("--symbols", type=str, default=None, help="逗号分隔 6 位代码")
    p.add_argument("--horizon", type=int, default=None, help="前瞻交易日数；默认 config features.eval_forward_horizon")
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--features",
        type=str,
        default="",
        help=(
            "逗号分隔因子列名，优先级高于 config 的 signals.tree_model.features。"
            "留空时：若配置了 tree_model.features 则用配置，否则用 default_tree_factor_names() "
            "（含 vol_to_turnover、volume_skew_log 等）。"
        ),
    )
    p.add_argument("--gpu", action="store_true", help="XGBoost device=cuda（需 GPU 版 xgboost）")
    p.add_argument(
        "--objective",
        choices=("rank", "regression"),
        default="rank",
        help="rank：XGBRanker（pairwise，与 Top-K 一致）；regression：XGBRegressor（MSE）",
    )
    p.add_argument("--out-root", type=str, default="data/models")
    p.add_argument("--experiments-dir", type=str, default="data/experiments")
    p.add_argument("--extra-xgb", type=str, default="{}", help="JSON，覆盖 n_estimators、max_depth 等")
    p.add_argument(
        "--label-transform",
        choices=("raw", "sharpe", "calmar", "truncate"),
        default=None,
        help="标签转换：raw=原始收益；sharpe=夏普比率；calmar=卡玛比率；truncate=截断极端值。默认从 config.label.transform 读取",
    )
    p.add_argument("--truncate-q", type=float, default=None, help="截断分位数（仅 truncate 模式）")
    p.add_argument(
        "--orthogonalize",
        action="store_true",
        default=None,
        help="训练前对因子截面做正交化（Löwdin）；默认从 config.orthogonalize.enabled 读取",
    )
    p.add_argument(
        "--orthogonalize-method",
        choices=("symmetric", "gram_schmidt"),
        default=None,
        help="正交化方法；默认 config.orthogonalize.method",
    )
    args = p.parse_args()

    cfg = load_config(args.config)
    paths = cfg.get("paths", {}) or {}
    feat = cfg.get("features", {})
    gpu_cfg = cfg.get("gpu", {})
    sig = cfg.get("signals", {})
    tree_sig = sig.get("tree_model") or {}

    lookback = int(feat.get("lookback_trading_days", 160))
    min_valid = int(feat.get("min_valid_days", 30))
    mom_w = int(feat.get("momentum_window", 10))
    rsi_p = int(feat.get("rsi_period", 14))
    atr_p = int(feat.get("atr_period", 14))
    vol_w = int(feat.get("vol_window", 20))
    to_w = int(feat.get("turnover_window", 20))
    vp_w = int(feat.get("vp_corr_window", 20))
    rev_w = int(feat.get("reversal_window", 5))
    horizon = int(args.horizon if args.horizon is not None else feat.get("eval_forward_horizon", 5))
    bias_ws = int(feat.get("bias_window_short", 20))
    bias_wl = int(feat.get("bias_window_long", 60))
    max_drop_w = int(feat.get("max_drop_window", 20))
    recent_ret_w = int(feat.get("recent_return_window", 3))
    pp_w = int(feat.get("price_position_window", 250))

    label_cfg = cfg.get("label") or {}
    label_transform = args.label_transform or str(label_cfg.get("transform", "raw")).lower()
    truncate_q = args.truncate_q if args.truncate_q is not None else float(label_cfg.get("truncate_quantile", 0.98))

    orth_cfg = cfg.get("orthogonalize") or {}
    orthogonalize = args.orthogonalize if args.orthogonalize is not None else bool(orth_cfg.get("enabled", False))
    orthogonalize_method = args.orthogonalize_method or str(orth_cfg.get("method", "symmetric")).lower()

    feat_cfg = cfg.get("features", {})
    tail_w = int(feat_cfg.get("tail_window", 10))
    vpt_w = int(feat_cfg.get("vpt_window", 20))
    range_skew_w = int(feat_cfg.get("range_skew_window", 20))

    device_str = str(gpu_cfg.get("device", "cpu")).lower()
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
    extra_lookback = max(40, pp_w - lookback + 40)
    start = end - pd.offsets.BDay(lookback + extra_lookback)

    with DuckDBManager(config_path=args.config) as db:
        df = db.read_daily_frame(symbols=symbols, start=start, end=end)

    if df.empty:
        print("日线为空", file=sys.stderr)
        return 1

    if args.features.strip():
        raw_names = _parse_features(args.features)
    elif tree_sig.get("features"):
        raw_names = [str(x) for x in tree_sig["features"]]
    else:
        raw_names = list(default_tree_factor_names())

    panel = long_factor_panel_from_daily(
        df,
        horizon=horizon,
        min_valid_days=min_valid,
        momentum_window=mom_w,
        rsi_period=rsi_p,
        atr_period=atr_p,
        vol_window=vol_w,
        turnover_window=to_w,
        vp_corr_window=vp_w,
        reversal_window=rev_w,
        device=device_str,
        dtype=torch_dtype,
        factor_names=raw_names,
        label_transform=label_transform,
        label_truncate_quantile=truncate_q,
        bias_window_short=bias_ws,
        bias_window_long=bias_wl,
        max_drop_window=max_drop_w,
        recent_return_window=recent_ret_w,
        price_position_window=pp_w,
        tail_window=tail_w,
        vpt_window=vpt_w,
        range_skew_window=range_skew_w,
        orthogonalize=orthogonalize,
        orthogonalize_method=orthogonalize_method,
    )
    if panel.empty or len(panel) < 50:
        print(f"面板样本过少: n={len(panel)}", file=sys.stderr)
        return 1

    y_col = f"forward_ret_{horizon}d"
    spec = normalize_slice_spec(
        symbols=symbols,
        date_start=str(panel["trade_date"].min().date()),
        date_end=str(panel["trade_date"].max().date()),
        extra={"horizon": horizon, "n_rows": len(panel)},
    )

    xgb_extra = json.loads(args.extra_xgb)
    if args.gpu:
        xgb_extra["device"] = "cuda"
        xgb_extra["use_gpu"] = True

    rsi_train = str(tree_sig.get("rsi_mode") or sig.get("composite", {}).get("rsi_mode", "level"))

    out_root = paths.get("models_dir", args.out_root)
    if not Path(out_root).is_absolute():
        out_root = str(ROOT / out_root)
    exp_dir = paths.get("experiments_dir", args.experiments_dir)
    if not Path(exp_dir).is_absolute():
        exp_dir = str(ROOT / exp_dir)

    res = train_xgboost_panel(
        panel,
        raw_feature_names=raw_names,
        target_column=y_col,
        rsi_mode=rsi_train,
        xgboost_objective=str(args.objective),
        training_seed=args.seed,
        val_frac=args.val_frac,
        slice_spec=spec,
        xgb_params=xgb_extra,
        out_root=out_root,
        experiments_dir=exp_dir,
    )
    print(
        json.dumps(
            {"bundle_dir": str(res.bundle_dir), "metrics": res.metrics, "n_samples": len(panel)},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
