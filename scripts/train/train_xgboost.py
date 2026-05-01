#!/usr/bin/env python3
"""
阶段二：用 DuckDB 历史日线构建 (symbol, date) 面板，截面 Z-score 特征 + 前瞻收益标签，训练 XGBoost。

默认使用 ``XGBRanker``（pairwise 排序，与 Top-K 选股一致）；``--objective regression`` 可切回 ``XGBRegressor``。

用法（项目根目录，conda activate quant-system）::

    python scripts/train/train_xgboost.py --config config.yaml --max-symbols 400

训练完成后将 ``signals.tree_model.bundle_dir`` 指向输出的 ``data/models/xgboost_panel_<id>/``，
并将 ``signals.sort_by`` 设为 ``xgboost``。

Jetson 上可加 ``--gpu`` 使用 ``device=cuda``（须已安装支持 CUDA 的 xgboost）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_fetcher import DuckDBManager, list_default_universe_symbols
from src.features.tree_dataset import default_tree_factor_names, long_factor_panel_from_daily
from src.models.data_slice import normalize_slice_spec
from src.models.xtree.train import train_xgboost_panel
from src.settings import load_config, resolve_asof_trade_end


def _parse_features(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def _parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _parse_float_list(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _cross_section_rank_fusion(
    df: pd.DataFrame,
    *,
    label_columns: Sequence[str],
    weights: Sequence[float],
    date_col: str = "trade_date",
    out_col: str = "forward_ret_fused",
) -> pd.DataFrame:
    if len(label_columns) != len(weights):
        raise ValueError("label_columns 与 weights 长度不一致")
    if not label_columns:
        raise ValueError("label_columns 不能为空")
    w = np.asarray(weights, dtype=np.float64)
    if not np.isfinite(w).all() or np.sum(np.abs(w)) <= 1e-12:
        raise ValueError("weights 非法或全为 0")
    w = w / np.sum(np.abs(w))
    out = df.copy()
    score = np.zeros(len(out), dtype=np.float64)
    valid = np.ones(len(out), dtype=bool)
    for col, wi in zip(label_columns, w):
        rk = out.groupby(date_col, sort=False)[col].rank(method="average", pct=True)
        rk = rk.to_numpy(dtype=np.float64) - 0.5
        score += wi * rk
        valid &= np.isfinite(rk)
    out[out_col] = np.where(valid, score, np.nan)
    out = out[np.isfinite(out[out_col])].copy()
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="训练截面 XGBoost（Z-score 因子 → 前瞻收益）")
    p.add_argument("--config", type=Path, default=None, help="默认项目根 config.yaml")
    p.add_argument("--max-symbols", type=int, default=None)
    p.add_argument("--symbols", type=str, default=None, help="逗号分隔 6 位代码")
    p.add_argument("--horizon", type=int, default=None, help="前瞻交易日数；默认 config features.eval_forward_horizon")
    p.add_argument(
        "--label-horizons",
        type=str,
        default="",
        help="多窗口标签融合，如 5,10,20；为空时仅使用 --horizon",
    )
    p.add_argument(
        "--label-weights",
        type=str,
        default="",
        help="多窗口标签融合权重，如 0.5,0.3,0.2；为空时等权",
    )
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
    p.add_argument(
        "--publish-dir",
        type=str,
        default="",
        help="训练通过门禁后发布到固定目录（默认取 config.signals.tree_model.bundle_dir）",
    )
    p.add_argument("--keep-versions", type=int, default=8, help="保留最近 N 个版本（含历史发布快照）")
    p.add_argument("--guard-quantile", type=float, default=0.25, help="门禁分位阈值（默认 P25）")
    p.add_argument("--guard-min-history", type=int, default=4, help="触发门禁所需历史样本数")
    p.add_argument(
        "--min-val-rank-ic",
        type=float,
        default=None,
        help="发布最低门槛：验证集 Rank IC 需 >= 该值（默认读取 config.signals.tree_model.rank_ic_guard.min_rank_ic 或 0.03）",
    )
    p.add_argument(
        "--time-cv-splits",
        type=int,
        default=None,
        help="严格时间序列 CV 折数（TimeSeriesSplit，默认读取 config.signals.tree_model.time_cv_splits 或 3）",
    )
    p.add_argument(
        "--disable-guard",
        action="store_true",
        help="关闭 Rank IC 分位门禁（不建议生产使用）",
    )
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
    tree_label_cfg = tree_sig.get("labels") or {}
    guard_cfg = tree_sig.get("rank_ic_guard") or {}

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
    if args.label_horizons.strip():
        label_horizons = _parse_int_list(args.label_horizons)
    elif tree_label_cfg.get("horizons"):
        label_horizons = [int(x) for x in tree_label_cfg.get("horizons")]
    else:
        label_horizons = [horizon]
    if len(set(label_horizons)) != len(label_horizons):
        raise ValueError("label_horizons 不允许重复")
    if any(h <= 0 for h in label_horizons):
        raise ValueError("label_horizons 必须为正整数")
    if args.label_weights.strip():
        label_weights = _parse_float_list(args.label_weights)
        if len(label_weights) != len(label_horizons):
            raise ValueError("label_weights 数量必须与 label_horizons 一致")
    elif tree_label_cfg.get("weights"):
        label_weights = [float(x) for x in tree_label_cfg.get("weights")]
        if len(label_weights) != len(label_horizons):
            raise ValueError("config tree_model.labels.weights 数量必须与 horizons 一致")
    else:
        label_weights = [1.0 / len(label_horizons)] * len(label_horizons)
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

    panel_parts: list[pd.DataFrame] = []
    y_columns: list[str] = []
    for h in label_horizons:
        p_h = long_factor_panel_from_daily(
            df,
            horizon=h,
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
        if p_h.empty:
            print(f"horizon={h} 生成面板为空", file=sys.stderr)
            return 1
        y_h = f"forward_ret_{h}d"
        y_columns.append(y_h)
        keep_cols = ["symbol", "trade_date", *raw_names, y_h]
        panel_parts.append(p_h[keep_cols].copy())

    panel = panel_parts[0]
    for p_h in panel_parts[1:]:
        y_h = [c for c in p_h.columns if c.startswith("forward_ret_")][0]
        panel = panel.merge(
            p_h[["symbol", "trade_date", y_h]],
            on=["symbol", "trade_date"],
            how="inner",
        )

    if panel.empty or len(panel) < 50:
        print(f"面板样本过少: n={len(panel)}", file=sys.stderr)
        return 1

    if len(y_columns) == 1:
        y_col = y_columns[0]
    else:
        panel = _cross_section_rank_fusion(
            panel,
            label_columns=y_columns,
            weights=label_weights,
            date_col="trade_date",
            out_col="forward_ret_fused",
        )
        y_col = "forward_ret_fused"
    if panel.empty:
        print("标签融合后无有效样本", file=sys.stderr)
        return 1

    label_spec = {
        "horizons": list(label_horizons),
        "weights": list(label_weights),
        "label_transform": label_transform,
        "truncate_quantile": float(truncate_q),
        "target_column": y_col,
        "scope": "cross_section_relative",
        "fusion_mode": "single_horizon" if len(y_columns) == 1 else "rank_fusion",
    }
    spec = normalize_slice_spec(
        symbols=symbols,
        date_start=str(panel["trade_date"].min().date()),
        date_end=str(panel["trade_date"].max().date()),
        extra={"horizons": label_horizons, "label_weights": label_weights, "n_rows": len(panel)},
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
    publish_dir = args.publish_dir.strip() or str(tree_sig.get("bundle_dir") or "").strip()
    if publish_dir and not Path(publish_dir).is_absolute():
        publish_dir = str(ROOT / publish_dir)
    keep_versions = int(tree_sig.get("keep_versions", args.keep_versions))
    guard_enabled = bool(guard_cfg.get("enabled", True)) and not bool(args.disable_guard)
    guard_quantile = float(guard_cfg.get("quantile", args.guard_quantile))
    guard_min_history = int(guard_cfg.get("min_history", args.guard_min_history))
    min_val_rank_ic = float(
        args.min_val_rank_ic
        if args.min_val_rank_ic is not None
        else guard_cfg.get("min_rank_ic", 0.03)
    )
    time_cv_splits = int(
        args.time_cv_splits
        if args.time_cv_splits is not None
        else tree_sig.get("time_cv_splits", 3)
    )

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
        label_spec=label_spec,
        enforce_metric_guard=guard_enabled,
        guard_metric_key="val_rank_ic",
        guard_quantile=guard_quantile,
        guard_min_history=guard_min_history,
        min_rank_ic_to_publish=min_val_rank_ic,
        time_cv_splits=time_cv_splits,
        keep_recent_versions=keep_versions,
        publish_bundle_dir=publish_dir or None,
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
