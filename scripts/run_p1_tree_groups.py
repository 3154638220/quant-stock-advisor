#!/usr/bin/env python3
"""P1：运行树模型分组 A/B，并以 daily proxy first 作为准入口径。"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.xtree.p1_workflow import (
    FUND_FLOW_TREE_FEATURES,
    attach_p1_experimental_features,
    attach_weekly_kdj_interaction_features,
    build_daily_proxy_first_leaderboard,
    build_group_comparison_table,
    build_p1_monthly_investable_label,
    build_p1_daily_proxy_first_report,
    build_p1_training_label,
    build_p1_tree_output_stem,
    build_p1_tree_research_config_id,
    build_tree_daily_backtest_like_proxy_detail,
    build_tree_direction_diagnostic_table,
    build_tree_light_proxy_detail,
    build_tree_topk_boundary_diagnostic,
    build_tree_turnover_aware_proxy_detail,
    p1_tree_feature_groups,
    panel_generation_feature_names,
    resolve_available_feature_names,
    summarize_p1_full_backtest_payload,
    summarize_p1_label_diagnostics,
    summarize_tree_daily_backtest_like_proxy,
    summarize_tree_daily_proxy_state_slices,
    summarize_tree_group_result,
    summarize_tree_score_direction,
)


def _parse_features(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def _parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _parse_float_list(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _parse_group_list(s: str) -> list[str]:
    return [x.strip().upper() for x in s.split(",") if x.strip()]


def resolve_label_horizons_and_weights(
    *,
    cli_label_horizons: str,
    cli_label_weights: str,
    default_horizon: int,
    label_cfg: dict,
) -> tuple[list[int], list[float]]:
    """解析多窗口标签；CLI horizons 优先时，未显式给 weights 则按 CLI 窗口等权。"""
    cli_horizons_set = bool(cli_label_horizons.strip())
    if cli_horizons_set:
        label_horizons = _parse_int_list(cli_label_horizons)
    elif label_cfg.get("horizons"):
        label_horizons = [int(x) for x in label_cfg.get("horizons")]
    else:
        label_horizons = [int(default_horizon)]

    if not label_horizons:
        raise ValueError("label_horizons 不能为空")

    if cli_label_weights.strip():
        label_weights = _parse_float_list(cli_label_weights)
    elif label_cfg.get("weights") and not cli_horizons_set:
        label_weights = [float(x) for x in label_cfg.get("weights")]
    else:
        label_weights = [1.0 / len(label_horizons)] * len(label_horizons)

    if len(label_weights) != len(label_horizons):
        raise ValueError("label_weights 数量必须与 label_horizons 一致")
    return label_horizons, label_weights


def _cross_section_rank_fusion(
    df: pd.DataFrame,
    *,
    label_columns: list[str],
    weights: list[float],
    date_col: str = "trade_date",
    out_col: str = "forward_ret_fused",
) -> pd.DataFrame:
    if len(label_columns) != len(weights):
        raise ValueError("label_columns 与 weights 长度不一致")
    w = np.asarray(weights, dtype=np.float64)
    if not len(w) or not np.isfinite(w).all() or np.sum(np.abs(w)) <= 1e-12:
        raise ValueError("weights 非法")
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
    return out[np.isfinite(out[out_col])].copy()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="P1 树模型分组实验：daily-proxy-first")
    p.add_argument("--config", type=Path, default=None, help="默认项目根 config.yaml")
    p.add_argument("--max-symbols", type=int, default=None)
    p.add_argument("--symbols", type=str, default=None, help="逗号分隔 6 位代码")
    p.add_argument("--history-start", type=str, default="", help="读日线历史起点 YYYY-MM-DD；默认按配置 lookback 回推")
    p.add_argument("--sample-start", type=str, default="", help="训练/验证样本起点 YYYY-MM-DD；用于保留前置特征 warmup")
    p.add_argument("--horizon", type=int, default=None, help="默认前瞻交易日数")
    p.add_argument("--label-horizons", type=str, default="", help="多窗口标签融合，如 5,10,20")
    p.add_argument("--label-weights", type=str, default="", help="多窗口标签融合权重，如 0.5,0.3,0.2")
    p.add_argument(
        "--label-transform",
        choices=("raw", "sharpe", "calmar", "truncate"),
        default="raw",
        help="前瞻标签变换：raw 为收益，sharpe/calmar 惩罚持有路径波动或回撤，truncate 截断极端收益",
    )
    p.add_argument(
        "--label-truncate-quantile",
        type=float,
        default=0.98,
        help="label-transform=truncate 时的截断分位数",
    )
    p.add_argument(
        "--label-mode",
        choices=(
            "rank_fusion",
            "top_bucket_rank_fusion",
            "raw_fusion",
            "market_relative",
            "benchmark_relative",
            "up_capture_market_relative",
            "monthly_investable",
            "monthly_investable_market_relative",
            "monthly_investable_up_capture_market_relative",
        ),
        default="rank_fusion",
        help="训练标签口径：截面 rank 融合、Top/Bottom bucket rank、原始收益融合、市场相对、上涨参与相对、benchmark 相对或月频可投资收益",
    )
    p.add_argument("--proxy-horizon", type=int, default=None, help="light proxy 使用的 forward_ret_h 列")
    p.add_argument(
        "--xgboost-objective",
        choices=("rank", "regression"),
        default="rank",
        help="P1 训练目标函数：rank 使用 XGBRanker，regression 使用 XGBRegressor",
    )
    p.add_argument("--rebalance-rule", type=str, default="M", help="light proxy 调仓频率")
    p.add_argument("--top-k", type=int, default=20, help="light proxy Top-K")
    p.add_argument(
        "--proxy-max-turnover",
        type=float,
        default=1.0,
        help="full-like/daily proxy 的 half-L1 换手上限；默认 1.0 表示允许全买全卖",
    )
    p.add_argument(
        "--daily-proxy-admission-threshold",
        type=float,
        default=0.0,
        help="daily full-backtest-like proxy 硬停止阈值；低于该值时停止候选",
    )
    p.add_argument(
        "--daily-proxy-full-backtest-threshold",
        type=float,
        default=0.03,
        help="daily proxy 进入正式 full backtest 的安全边际阈值；默认 3pct，介于硬停止线和该阈值之间只归档诊断",
    )
    p.add_argument(
        "--disable-daily-proxy-admission-gate",
        action="store_true",
        help="关闭 daily-proxy-first 拦截；仅用于复现历史失败样本或校准",
    )
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--out-tag", type=str, default="p1_tree_groups")
    p.add_argument("--out-root", type=str, default="data/models")
    p.add_argument("--results-dir", type=str, default="data/results")
    p.add_argument("--experiments-dir", type=str, default="data/experiments")
    p.add_argument("--extra-xgb", type=str, default="{}")
    p.add_argument("--groups", type=str, default="", help="仅运行指定分组，如 G2,G4；默认全部 G0~G4")
    p.add_argument(
        "--include-interaction-groups",
        action="store_true",
        help="额外运行 weekly_kdj gated/interaction 分组 G5/G6；默认保持 G0~G4 既定口径",
    )
    p.add_argument("--run-full-backtest", action="store_true", help="训练后对每个分组调用正式 run_backtest_eval")
    p.add_argument("--backtest-config", type=str, default="config.yaml.backtest", help="full backtest 配置路径")
    p.add_argument("--backtest-start", type=str, default="2021-01-01")
    p.add_argument("--backtest-end", type=str, default="")
    p.add_argument("--backtest-lookback-days", type=int, default=260)
    p.add_argument("--backtest-min-hist-days", type=int, default=130)
    p.add_argument("--backtest-max-turnover", type=float, default=1.0)
    p.add_argument("--backtest-top-k", type=int, default=20)
    p.add_argument("--backtest-portfolio-method", type=str, default="equal_weight")
    p.add_argument("--backtest-prepared-factors-cache", type=str, default="")
    p.add_argument("--backtest-no-regime", action="store_true")
    p.add_argument(
        "--min-val-rank-ic",
        type=float,
        default=-1.0,
        help="研究模式默认不设发布门槛，允许负值仅做对比",
    )
    p.add_argument("--time-cv-splits", type=int, default=3)
    p.add_argument("--orthogonalize", action="store_true", default=None)
    p.add_argument(
        "--orthogonalize-method",
        choices=("symmetric", "gram_schmidt"),
        default=None,
    )
    return p.parse_args()


def _run_full_backtest_for_group(
    *,
    config_path: str,
    bundle_dir: Path,
    tree_features: list[str],
    tree_rsi_mode: str,
    research_config_id: str,
    output_stem: str,
    results_dir: str,
    group_name: str,
    start: str,
    end: str,
    lookback_days: int,
    min_hist_days: int,
    rebalance_rule: str,
    top_k: int,
    max_turnover: float,
    portfolio_method: str,
    prepared_factors_cache: str,
    no_regime: bool,
) -> dict[str, object]:
    json_report = Path(results_dir) / f"p1_full_backtest_{group_name.lower()}.json"
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_backtest_eval.py"),
        "--config",
        config_path,
        "--start",
        start,
        "--lookback-days",
        str(int(lookback_days)),
        "--min-hist-days",
        str(int(min_hist_days)),
        "--rebalance-rule",
        rebalance_rule,
        "--top-k",
        str(int(top_k)),
        "--max-turnover",
        str(float(max_turnover)),
        "--portfolio-method",
        portfolio_method,
        "--sort-by",
        "xgboost",
        "--tree-bundle-dir",
        str(bundle_dir),
        "--tree-features",
        ",".join(tree_features),
        "--tree-rsi-mode",
        tree_rsi_mode,
        "--tree-feature-group",
        group_name,
        "--research-topic",
        "p1_tree_groups",
        "--research-config-id",
        research_config_id,
        "--output-stem",
        f"{output_stem}_full_backtest_{group_name.lower()}",
        "--canonical-config",
        "p1_tree_full_backtest",
        "--json-report",
        str(json_report),
    ]
    if end.strip():
        cmd.extend(["--end", end.strip()])
    if prepared_factors_cache.strip():
        cmd.extend(["--prepared-factors-cache", prepared_factors_cache.strip()])
    if no_regime:
        cmd.append("--no-regime")

    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
    if proc.returncode != 0:
        raise RuntimeError(
            f"{group_name} full backtest 失败，exit={proc.returncode}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    payload = json.loads(json_report.read_text(encoding="utf-8"))
    out = summarize_p1_full_backtest_payload(payload)
    out["full_backtest_json"] = str(json_report)
    out["full_backtest_research_topic"] = str(payload.get("research_topic", ""))
    out["full_backtest_research_config_id"] = str(payload.get("research_config_id", ""))
    out["full_backtest_output_stem"] = str(payload.get("output_stem", ""))
    return out


def main() -> int:
    from src.backtest.transaction_costs import (
        cost_params_dict_for_logging,
        transaction_cost_params_from_mapping,
    )
    from src.data_fetcher import DuckDBManager, list_default_universe_symbols
    from src.features.tree_dataset import long_factor_panel_from_daily
    from src.models.data_slice import normalize_slice_spec
    from src.models.inference import predict_xgboost_tree
    from src.models.rank_score import apply_cross_section_z_by_date
    from src.models.xtree.train import train_xgboost_panel
    from src.settings import load_config, resolve_asof_trade_end

    args = parse_args()
    daily_proxy_admission_threshold = float(args.daily_proxy_admission_threshold)
    daily_proxy_full_backtest_threshold = max(
        daily_proxy_admission_threshold,
        float(args.daily_proxy_full_backtest_threshold),
    )
    cfg = load_config(args.config)
    config_source = str(args.config or "config.yaml")
    paths = cfg.get("paths", {}) or {}
    feat = cfg.get("features", {}) or {}
    gpu_cfg = cfg.get("gpu", {}) or {}
    sig = cfg.get("signals", {}) or {}
    tree_sig = sig.get("tree_model") or {}
    label_cfg = tree_sig.get("labels") or {}
    backtest_cfg = cfg.get("backtest", {}) or {}
    costs = transaction_cost_params_from_mapping(cfg.get("transaction_costs", {}))

    if args.symbols:
        symbols = [s.strip().zfill(6) for s in args.symbols.split(",") if s.strip()]
    else:
        symbols = list_default_universe_symbols(max_symbols=args.max_symbols, config_path=args.config)
    if not symbols:
        print("无标的列表", file=sys.stderr)
        return 1

    lookback = int(feat.get("lookback_trading_days", 160))
    min_valid = int(feat.get("min_valid_days", 30))
    mom_w = int(feat.get("momentum_window", 10))
    rsi_p = int(feat.get("rsi_period", 14))
    atr_p = int(feat.get("atr_period", 14))
    vol_w = int(feat.get("vol_window", 20))
    to_w = int(feat.get("turnover_window", 20))
    vp_w = int(feat.get("vp_corr_window", 20))
    rev_w = int(feat.get("reversal_window", 5))
    bias_ws = int(feat.get("bias_window_short", 20))
    bias_wl = int(feat.get("bias_window_long", 60))
    max_drop_w = int(feat.get("max_drop_window", 20))
    recent_ret_w = int(feat.get("recent_return_window", 3))
    pp_w = int(feat.get("price_position_window", 250))
    tail_w = int(feat.get("tail_window", 10))
    vpt_w = int(feat.get("vpt_window", 20))
    range_skew_w = int(feat.get("range_skew_window", 20))

    horizon = int(args.horizon if args.horizon is not None else feat.get("eval_forward_horizon", 5))
    label_horizons, label_weights = resolve_label_horizons_and_weights(
        cli_label_horizons=args.label_horizons,
        cli_label_weights=args.label_weights,
        default_horizon=horizon,
        label_cfg=label_cfg,
    )
    proxy_horizon = int(args.proxy_horizon if args.proxy_horizon is not None else label_horizons[0])

    orth_cfg = cfg.get("orthogonalize") or {}
    orthogonalize = args.orthogonalize if args.orthogonalize is not None else bool(orth_cfg.get("enabled", False))
    orthogonalize_method = args.orthogonalize_method or str(orth_cfg.get("method", "symmetric")).lower()

    device_str = str(gpu_cfg.get("device", "cpu")).lower()
    if args.gpu:
        device_str = "cuda"
    if device_str == "cuda" and not torch.cuda.is_available():
        device_str = "cpu"
    dtype_str = str(gpu_cfg.get("dtype", "float32")).lower()
    torch_dtype = torch.float32 if dtype_str in ("float32", "fp32") else torch.float64

    end = resolve_asof_trade_end(paths)
    # P1 训练端需要给长窗口特征（如 price_position_window=250）和多窗口标签留出更宽缓冲，
    # 否则当 as-of 晚于库中最后交易日时，容易把可用样本全部挤掉。
    extra_lookback = max(120, pp_w - lookback + 120)
    if args.history_start.strip():
        start = pd.Timestamp(args.history_start.strip()).normalize()
    else:
        start = end - pd.offsets.BDay(lookback + extra_lookback)
    with DuckDBManager(config_path=args.config) as db:
        daily_df = db.read_daily_frame(symbols=symbols, start=start, end=end)
    if daily_df.empty:
        print("日线为空", file=sys.stderr)
        return 1

    technical_names = panel_generation_feature_names()
    panel_parts: list[pd.DataFrame] = []
    y_columns: list[str] = []
    label_transform = str(args.label_transform).lower().strip()
    label_truncate_quantile = float(args.label_truncate_quantile)
    transformed_label_columns: list[str] = []
    for h in label_horizons:
        panel_h = long_factor_panel_from_daily(
            daily_df,
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
            factor_names=technical_names,
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
        if panel_h.empty:
            print(f"horizon={h} 生成面板为空", file=sys.stderr)
            return 1
        y_h = f"forward_ret_{h}d"
        keep_cols = ["symbol", "trade_date", *technical_names, y_h]
        if label_transform != "raw":
            label_panel_h = long_factor_panel_from_daily(
                daily_df,
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
                factor_names=technical_names,
                label_transform=label_transform,
                label_truncate_quantile=label_truncate_quantile,
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
            if label_panel_h.empty:
                print(f"horizon={h}, label_transform={label_transform} 生成标签面板为空", file=sys.stderr)
                return 1
            y_label = f"forward_{label_transform}_{h}d"
            label_panel_h = label_panel_h[["symbol", "trade_date", y_h]].rename(columns={y_h: y_label})
            panel_h = panel_h.merge(label_panel_h, on=["symbol", "trade_date"], how="inner")
            keep_cols.append(y_label)
            transformed_label_columns.append(y_label)
        y_columns.append(y_h)
        panel_parts.append(panel_h[keep_cols].copy())

    panel = panel_parts[0]
    for panel_h in panel_parts[1:]:
        forward_cols = [c for c in panel_h.columns if c.startswith("forward_ret_") or c.startswith("forward_")]
        panel = panel.merge(panel_h[["symbol", "trade_date", *forward_cols]], on=["symbol", "trade_date"], how="inner")
    if panel.empty:
        print("合并多标签后面板为空", file=sys.stderr)
        return 1

    if args.sample_start.strip():
        sample_start = pd.Timestamp(args.sample_start.strip()).normalize()
        panel = panel[pd.to_datetime(panel["trade_date"], errors="coerce").dt.normalize() >= sample_start].copy()
        if panel.empty:
            print(f"按 sample_start={sample_start.date()} 过滤后面板为空", file=sys.stderr)
            return 1

    db_path = str((paths.get("duckdb_path") or "data/market.duckdb")).strip()
    if not Path(db_path).is_absolute():
        db_path = str(ROOT / db_path)
    panel = attach_p1_experimental_features(panel, db_path=db_path)
    panel = attach_weekly_kdj_interaction_features(panel)
    fund_flow_non_null = int(panel[list(FUND_FLOW_TREE_FEATURES)].notna().sum().sum())
    if fund_flow_non_null == 0:
        print(
            "[P1] 警告: fund_flow 特征当前全为空；G2/G4 将退化为不含资金流特征的结果。"
            " 请先补齐 a_share_fund_flow 原始表后再解释 G2/G4。",
            file=sys.stderr,
        )

    label_mode = str(args.label_mode)
    execution_mode = str(backtest_cfg.get("execution_mode", "tplus1_open"))
    effective_y_columns = transformed_label_columns if label_transform != "raw" else y_columns
    if label_mode.startswith("monthly_investable"):
        if label_transform != "raw":
            raise ValueError("monthly_investable 标签已自行按调仓期构造，不能叠加 --label-transform")
        panel, target_column, label_meta = build_p1_monthly_investable_label(
            panel,
            daily_df,
            rebalance_rule=args.rebalance_rule,
            execution_mode=execution_mode,
            label_mode=label_mode,
            date_col="trade_date",
            out_col="forward_ret_investable",
        )
    elif len(y_columns) == 1 and label_mode == "raw_fusion":
        target_column = effective_y_columns[0]
        label_meta = {
            "label_mode": "raw_fusion",
            "label_scope": "raw_return" if label_transform == "raw" else f"{label_transform}_path_quality",
            "label_component_columns": ",".join(effective_y_columns),
            "label_weights_normalized": "1",
            "label_market_proxy": "",
            "label_transform": label_transform,
            "label_truncate_quantile": label_truncate_quantile if label_transform == "truncate" else "",
        }
    else:
        panel, target_column, label_meta = build_p1_training_label(
            panel,
            label_columns=effective_y_columns,
            label_weights=label_weights,
            label_mode=label_mode,
            date_col="trade_date",
            out_col="forward_ret_fused",
        )
        label_meta["label_transform"] = label_transform
        label_meta["label_truncate_quantile"] = label_truncate_quantile if label_transform == "truncate" else ""
    if panel.empty:
        print("标签融合后无有效样本", file=sys.stderr)
        return 1

    proxy_col = f"forward_ret_{proxy_horizon}d"
    if proxy_col not in panel.columns:
        print(f"缺少 proxy_horizon 对应列: {proxy_col}", file=sys.stderr)
        return 1
    diagnostic_label_columns = [target_column] if label_mode.startswith("monthly_investable") else effective_y_columns
    diagnostic_label_weights = [1.0] if label_mode.startswith("monthly_investable") else label_weights
    label_diagnostics = summarize_p1_label_diagnostics(
        panel,
        target_column=target_column,
        label_columns=diagnostic_label_columns,
        label_weights=diagnostic_label_weights,
        proxy_return_col=proxy_col,
    )

    out_root = paths.get("models_dir", args.out_root)
    results_dir = paths.get("results_dir", args.results_dir)
    exp_dir = paths.get("experiments_dir", args.experiments_dir)
    out_root = str((ROOT / out_root) if not Path(out_root).is_absolute() else Path(out_root))
    results_dir = str((ROOT / results_dir) if not Path(results_dir).is_absolute() else Path(results_dir))
    exp_dir = str((ROOT / exp_dir) if not Path(exp_dir).is_absolute() else Path(exp_dir))
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    spec = normalize_slice_spec(
        symbols=symbols,
        date_start=str(panel["trade_date"].min().date()),
        date_end=str(panel["trade_date"].max().date()),
        extra={
            "label_horizons": label_horizons,
            "label_weights": label_weights,
            "proxy_horizon": proxy_horizon,
            "n_rows": int(len(panel)),
        },
    )
    xgb_extra = json.loads(args.extra_xgb)
    if device_str == "cuda":
        xgb_extra["device"] = "cuda"
        xgb_extra["use_gpu"] = True
    rsi_mode = str(tree_sig.get("rsi_mode") or sig.get("composite", {}).get("rsi_mode", "level"))

    run_rows: list[dict[str, object]] = []
    detail_frames: list[pd.DataFrame] = []
    boundary_frames: list[pd.DataFrame] = []
    group_defs = p1_tree_feature_groups(include_interaction_groups=bool(args.include_interaction_groups))
    if args.groups.strip():
        requested_groups = _parse_group_list(args.groups)
        unknown_groups = [g for g in requested_groups if g not in group_defs]
        if unknown_groups:
            raise ValueError(f"未知 groups: {unknown_groups}")
        group_defs = type(group_defs)((g, group_defs[g]) for g in requested_groups)
    research_config_id = build_p1_tree_research_config_id(
        rebalance_rule=args.rebalance_rule,
        top_k=int(args.top_k),
        label_horizons=label_horizons,
        label_weights=label_weights,
        proxy_horizon=int(proxy_horizon),
        val_frac=float(args.val_frac),
        label_mode=label_mode,
        label_transform=label_transform,
        xgboost_objective=str(args.xgboost_objective),
    )
    generated_at = pd.Timestamp.utcnow()
    output_stem = build_p1_tree_output_stem(
        out_tag=args.out_tag,
        research_config_id=research_config_id,
        generated_at=generated_at,
    )
    bundle_manifest_rows: list[dict[str, object]] = []

    for group_name, requested_features in group_defs.items():
        print(f"[P1] 开始训练 {group_name} ...", flush=True)
        available_features, missing_features = resolve_available_feature_names(panel, requested_features)
        if len(available_features) < 2:
            raise RuntimeError(f"{group_name} 可用特征不足 2 个，无法训练: {missing_features}")

        res = train_xgboost_panel(
            panel,
            raw_feature_names=available_features,
            target_column=target_column,
            rsi_mode=rsi_mode,
            xgboost_objective=str(args.xgboost_objective),
            training_seed=args.seed,
            val_frac=args.val_frac,
            slice_spec=spec,
            xgb_params=xgb_extra,
            label_spec={
                "horizons": list(label_horizons),
                "weights": list(label_weights),
                "target_column": target_column,
                "proxy_horizon": int(proxy_horizon),
                "scope": str(label_meta.get("label_scope") or label_mode),
                "label_mode": label_mode,
                "label_market_proxy": str(label_meta.get("label_market_proxy") or ""),
                "label_transform": label_transform,
                "label_truncate_quantile": label_truncate_quantile if label_transform == "truncate" else "",
                "xgboost_objective": str(args.xgboost_objective),
                "research_topic": "p1_tree_groups",
                "research_config_id": research_config_id,
                "research_group": group_name,
                "bundle_label": f"p1_tree_{group_name.lower()}_{research_config_id}",
            },
            enforce_metric_guard=False,
            min_rank_ic_to_publish=float(args.min_val_rank_ic),
            time_cv_splits=int(args.time_cv_splits),
            keep_recent_versions=0,
            publish_bundle_dir=None,
            out_root=out_root,
            log_experiments=False,
            experiments_dir=exp_dir,
        )

        dates = sorted(pd.to_datetime(panel["trade_date"]).dt.normalize().unique())
        cut_idx = max(1, int(len(dates) * (1.0 - float(args.val_frac))))
        cutoff = dates[cut_idx - 1]
        val_panel = panel[pd.to_datetime(panel["trade_date"]).dt.normalize() > cutoff].copy()
        z_val = apply_cross_section_z_by_date(
            val_panel,
            date_col="trade_date",
            raw_names=available_features,
            rsi_mode=rsi_mode,
        )
        z_val["tree_score"] = predict_xgboost_tree(res.bundle_dir, z_val)
        detail_df = build_tree_light_proxy_detail(
            z_val,
            score_col="tree_score",
            proxy_return_col=proxy_col,
            rebalance_rule=args.rebalance_rule,
            top_k=int(args.top_k),
            scenario=group_name,
        )
        full_like_detail_df = build_tree_turnover_aware_proxy_detail(
            z_val,
            score_col="tree_score",
            proxy_return_col=proxy_col,
            rebalance_rule=args.rebalance_rule,
            top_k=int(args.top_k),
            max_turnover=float(args.proxy_max_turnover),
            scenario=group_name,
        )
        proxy_summary = summarize_tree_group_result(detail_df, rebalance_rule=args.rebalance_rule)
        full_like_proxy_summary = summarize_tree_group_result(
            full_like_detail_df,
            rebalance_rule=args.rebalance_rule,
        )
        daily_bt_like_detail_df, daily_bt_like_meta = build_tree_daily_backtest_like_proxy_detail(
            z_val,
            daily_df,
            score_col="tree_score",
            rebalance_rule=args.rebalance_rule,
            top_k=int(args.top_k),
            max_turnover=float(args.proxy_max_turnover),
            scenario=group_name,
            cost_params=costs,
            execution_mode=str(backtest_cfg.get("execution_mode", "tplus1_open")),
            execution_lag=int(backtest_cfg.get("execution_lag", 1)),
            limit_up_mode=str(backtest_cfg.get("limit_up_mode", "idle")),
            vwap_slippage_bps_per_side=float(backtest_cfg.get("vwap_slippage_bps_per_side", 3.0)),
            vwap_impact_bps=float(backtest_cfg.get("vwap_impact_bps", 8.0)),
        )
        daily_bt_like_summary = summarize_tree_daily_backtest_like_proxy(daily_bt_like_detail_df)
        daily_bt_like_excess = float(daily_bt_like_summary.get("annualized_excess_vs_market", np.nan))
        pass_daily_proxy_admission = bool(
            np.isfinite(daily_bt_like_excess)
            and daily_bt_like_excess >= daily_proxy_admission_threshold
        )
        pass_daily_proxy_full_backtest = bool(
            np.isfinite(daily_bt_like_excess)
            and daily_bt_like_excess >= daily_proxy_full_backtest_threshold
        )
        boundary_detail_df, boundary_summary = build_tree_topk_boundary_diagnostic(
            z_val,
            daily_df,
            score_col="tree_score",
            rebalance_rule=args.rebalance_rule,
            top_k=int(args.top_k),
            execution_mode=execution_mode,
            scenario=group_name,
        )
        bundle_label = f"p1_tree_{group_name.lower()}_{research_config_id}"
        detail_df["proxy_variant"] = "topk_unconstrained"
        detail_df["group"] = group_name
        detail_df["result_type"] = "legacy_light_strategy_proxy"
        detail_df["research_topic"] = "p1_tree_groups"
        detail_df["research_config_id"] = research_config_id
        detail_df["output_stem"] = output_stem
        detail_frames.append(detail_df)
        full_like_detail_df["proxy_variant"] = "full_like_turnover_aware"
        full_like_detail_df["group"] = group_name
        full_like_detail_df["result_type"] = "legacy_light_strategy_proxy"
        full_like_detail_df["research_topic"] = "p1_tree_groups"
        full_like_detail_df["research_config_id"] = research_config_id
        full_like_detail_df["output_stem"] = output_stem
        detail_frames.append(full_like_detail_df)
        daily_bt_like_detail_df["proxy_variant"] = "daily_backtest_like"
        daily_bt_like_detail_df["group"] = group_name
        daily_bt_like_detail_df["result_type"] = "daily_bt_like_proxy"
        daily_bt_like_detail_df["research_topic"] = "p1_tree_groups"
        daily_bt_like_detail_df["research_config_id"] = research_config_id
        daily_bt_like_detail_df["output_stem"] = output_stem
        detail_frames.append(daily_bt_like_detail_df)
        boundary_detail_df["group"] = group_name
        boundary_detail_df["result_type"] = "topk_boundary_diagnostic"
        boundary_detail_df["research_topic"] = "p1_tree_groups"
        boundary_detail_df["research_config_id"] = research_config_id
        boundary_detail_df["output_stem"] = output_stem
        boundary_frames.append(boundary_detail_df)

        run_rows.append(
            {
                "group": group_name,
                "result_type": "daily_bt_like_proxy",
                "primary_result_type": "daily_bt_like_proxy",
                "primary_decision_metric": "daily_bt_like_proxy_annualized_excess_vs_market",
                "legacy_proxy_decision_role": "diagnostic_only",
                "p1_experiment_mode": "daily_proxy_first",
                "config_source": config_source,
                "benchmark_symbol": "market_ew_proxy",
                "research_topic": "p1_tree_groups",
                "research_config_id": research_config_id,
                "output_stem": output_stem,
                "bundle_label": bundle_label,
                "bundle_dir": str(res.bundle_dir),
                "training_window_start": str(pd.to_datetime(panel["trade_date"]).min().date()),
                "training_window_end": str(pd.to_datetime(panel["trade_date"]).max().date()),
                "validation_cutoff": str(pd.Timestamp(cutoff).date()),
                "feature_count": int(len(available_features)),
                "features": ",".join(available_features),
                "missing_features": ",".join(missing_features),
                "xgboost_objective": str(args.xgboost_objective),
                **label_meta,
                "val_rank_ic": float(res.metrics.get("val_rank_ic", np.nan)),
                "train_rank_ic": float(res.metrics.get("train_rank_ic", np.nan)),
                **summarize_tree_score_direction(res.metrics),
                "val_mse": float(res.metrics.get("val_mse", np.nan)),
                "annualized_excess_vs_market": float(proxy_summary.get("annualized_excess_vs_market", np.nan)),
                "legacy_unconstrained_proxy_annualized_excess_vs_market": float(
                    proxy_summary.get("annualized_excess_vs_market", np.nan)
                ),
                "strategy_annualized_return": float(proxy_summary.get("strategy_annualized_return", np.nan)),
                "benchmark_annualized_return": float(proxy_summary.get("benchmark_annualized_return", np.nan)),
                "strategy_sharpe_ratio": float(proxy_summary.get("strategy_sharpe_ratio", np.nan)),
                "strategy_max_drawdown": float(proxy_summary.get("strategy_max_drawdown", np.nan)),
                "period_beat_rate": float(proxy_summary.get("period_beat_rate", np.nan)),
                "n_periods": int(proxy_summary.get("n_periods", 0)),
                "periods_per_year": float(proxy_summary.get("periods_per_year", np.nan)),
                "full_like_proxy_annualized_excess_vs_market": float(
                    full_like_proxy_summary.get("annualized_excess_vs_market", np.nan)
                ),
                "legacy_full_like_proxy_annualized_excess_vs_market": float(
                    full_like_proxy_summary.get("annualized_excess_vs_market", np.nan)
                ),
                "full_like_proxy_strategy_annualized_return": float(
                    full_like_proxy_summary.get("strategy_annualized_return", np.nan)
                ),
                "full_like_proxy_benchmark_annualized_return": float(
                    full_like_proxy_summary.get("benchmark_annualized_return", np.nan)
                ),
                "full_like_proxy_strategy_sharpe_ratio": float(
                    full_like_proxy_summary.get("strategy_sharpe_ratio", np.nan)
                ),
                "full_like_proxy_strategy_max_drawdown": float(
                    full_like_proxy_summary.get("strategy_max_drawdown", np.nan)
                ),
                "full_like_proxy_period_beat_rate": float(full_like_proxy_summary.get("period_beat_rate", np.nan)),
                "full_like_proxy_n_periods": int(full_like_proxy_summary.get("n_periods", 0)),
                "full_like_proxy_max_turnover": float(args.proxy_max_turnover),
                "proxy_gap_full_like_minus_unconstrained": float(
                    full_like_proxy_summary.get("annualized_excess_vs_market", np.nan)
                    - proxy_summary.get("annualized_excess_vs_market", np.nan)
                ),
                "daily_bt_like_proxy_annualized_excess_vs_market": daily_bt_like_excess,
                "daily_bt_like_proxy_strategy_annualized_return": float(
                    daily_bt_like_summary.get("strategy_annualized_return", np.nan)
                ),
                "daily_bt_like_proxy_benchmark_annualized_return": float(
                    daily_bt_like_summary.get("benchmark_annualized_return", np.nan)
                ),
                "daily_bt_like_proxy_strategy_sharpe_ratio": float(
                    daily_bt_like_summary.get("strategy_sharpe_ratio", np.nan)
                ),
                "daily_bt_like_proxy_strategy_max_drawdown": float(
                    daily_bt_like_summary.get("strategy_max_drawdown", np.nan)
                ),
                "daily_bt_like_proxy_period_beat_rate": float(daily_bt_like_summary.get("period_beat_rate", np.nan)),
                "daily_bt_like_proxy_n_periods": int(daily_bt_like_summary.get("n_periods", 0)),
                "daily_bt_like_proxy_n_rebalances": int(daily_bt_like_meta.get("n_rebalances", 0)),
                "daily_bt_like_proxy_avg_turnover_half_l1": float(
                    daily_bt_like_meta.get("avg_turnover_half_l1", np.nan)
                ),
                "proxy_gap_daily_bt_like_minus_unconstrained": float(
                    daily_bt_like_summary.get("annualized_excess_vs_market", np.nan)
                    - proxy_summary.get("annualized_excess_vs_market", np.nan)
                ),
                "daily_proxy_admission_threshold": daily_proxy_admission_threshold,
                "daily_proxy_full_backtest_threshold": daily_proxy_full_backtest_threshold,
                "pass_p1_daily_proxy_admission_gate": pass_daily_proxy_admission,
                "pass_p1_daily_proxy_full_backtest_gate": pass_daily_proxy_full_backtest,
                **boundary_summary,
            }
        )
        bundle_manifest_rows.append(
            {
                "group": group_name,
                "result_type": "xgboost_tree_bundle",
                "p1_experiment_mode": "daily_proxy_first",
                "config_source": config_source,
                "research_topic": "p1_tree_groups",
                "output_stem": output_stem,
                "bundle_label": bundle_label,
                "bundle_dir": str(res.bundle_dir),
                "research_config_id": research_config_id,
                "label_mode": label_mode,
                "label_scope": str(label_meta.get("label_scope") or label_mode),
                "label_horizons": ",".join(str(x) for x in label_horizons),
                "label_weights": ",".join(f"{float(x):.8g}" for x in label_weights),
                "label_transform": label_transform,
                "label_truncate_quantile": label_truncate_quantile if label_transform == "truncate" else "",
                "xgboost_objective": str(args.xgboost_objective),
                "rebalance_rule": args.rebalance_rule,
                "top_k": int(args.top_k),
                "execution_mode": execution_mode,
                "daily_proxy_admission_threshold": daily_proxy_admission_threshold,
                "daily_proxy_full_backtest_threshold": daily_proxy_full_backtest_threshold,
                "daily_proxy_max_turnover": float(args.proxy_max_turnover),
                "backtest_config": str(args.backtest_config),
                "backtest_start": str(args.backtest_start),
                "backtest_end": str(args.backtest_end),
                "backtest_top_k": int(args.backtest_top_k),
                "backtest_max_turnover": float(args.backtest_max_turnover),
                "backtest_portfolio_method": str(args.backtest_portfolio_method),
                "backtest_prepared_factors_cache": str(args.backtest_prepared_factors_cache),
                "transaction_costs": json.dumps(cost_params_dict_for_logging(costs), ensure_ascii=False, sort_keys=True),
                "feature_count": int(len(available_features)),
                "features": ",".join(available_features),
                "missing_features": ",".join(missing_features),
            }
        )
        print(
            "[P1] 完成训练 {group}: val_rank_ic={val_ic:.6f}, "
            "proxy_excess={proxy:.2%}, full_like_proxy={full_like:.2%}, "
            "daily_bt_like={daily_bt_like:.2%}, daily_full_bt_gate={daily_gate}, auto_flip={auto_flip}".format(
                group=group_name,
                val_ic=float(res.metrics.get("val_rank_ic", np.nan)),
                proxy=float(proxy_summary.get("annualized_excess_vs_market", np.nan)),
                full_like=float(full_like_proxy_summary.get("annualized_excess_vs_market", np.nan)),
                daily_bt_like=daily_bt_like_excess,
                daily_gate=pass_daily_proxy_full_backtest,
                auto_flip=bool(summarize_tree_score_direction(res.metrics)["tree_score_auto_flipped"]),
            ),
            flush=True,
        )
        if args.run_full_backtest:
            if (not args.disable_daily_proxy_admission_gate) and (not pass_daily_proxy_admission):
                run_rows[-1].update(
                    {
                        "full_backtest_skipped_reason": (
                            "daily_bt_like_proxy_below_admission_threshold:"
                            f"{daily_bt_like_excess:.6g}<"
                            f"{daily_proxy_admission_threshold:.6g}"
                        ),
                    }
                )
                print(
                    "[P1] 跳过 {group} 正式回测: daily_bt_like={daily_bt_like:.2%} < threshold={threshold:.2%}".format(
                        group=group_name,
                        daily_bt_like=daily_bt_like_excess,
                        threshold=daily_proxy_admission_threshold,
                    ),
                    flush=True,
                )
                continue
            if (not args.disable_daily_proxy_admission_gate) and (not pass_daily_proxy_full_backtest):
                run_rows[-1].update(
                    {
                        "full_backtest_skipped_reason": (
                            "daily_bt_like_proxy_gray_zone_below_full_backtest_threshold:"
                            f"{daily_bt_like_excess:.6g}<"
                            f"{daily_proxy_full_backtest_threshold:.6g}"
                        ),
                    }
                )
                print(
                    "[P1] 跳过 {group} 正式回测: daily_bt_like={daily_bt_like:.2%} "
                    "< full_backtest_threshold={threshold:.2%} (gray zone)".format(
                        group=group_name,
                        daily_bt_like=daily_bt_like_excess,
                        threshold=daily_proxy_full_backtest_threshold,
                    ),
                    flush=True,
                )
                continue
            backtest_metrics = _run_full_backtest_for_group(
                config_path=str(args.backtest_config),
                bundle_dir=res.bundle_dir,
                tree_features=available_features,
                tree_rsi_mode=rsi_mode,
                research_config_id=research_config_id,
                output_stem=output_stem,
                results_dir=results_dir,
                group_name=group_name,
                start=args.backtest_start,
                end=args.backtest_end,
                lookback_days=args.backtest_lookback_days,
                min_hist_days=args.backtest_min_hist_days,
                rebalance_rule=args.rebalance_rule,
                top_k=args.backtest_top_k,
                max_turnover=args.backtest_max_turnover,
                portfolio_method=args.backtest_portfolio_method,
                prepared_factors_cache=args.backtest_prepared_factors_cache,
                no_regime=bool(args.backtest_no_regime),
            )
            run_rows[-1].update(backtest_metrics)

    summary_df = build_group_comparison_table(run_rows, baseline_group="G0")
    daily_leaderboard_df = build_daily_proxy_first_leaderboard(summary_df)
    direction_diag_df = build_tree_direction_diagnostic_table(run_rows)
    detail_all = pd.concat(detail_frames, ignore_index=True) if detail_frames else pd.DataFrame()
    boundary_all = pd.concat(boundary_frames, ignore_index=True) if boundary_frames else pd.DataFrame()
    state_monthly_df, state_summary_df = summarize_tree_daily_proxy_state_slices(
        detail_all,
        daily_df,
        execution_mode=execution_mode,
    )
    bundle_manifest_df = pd.DataFrame(bundle_manifest_rows)

    summary_path = Path(results_dir) / f"{output_stem}_summary.csv"
    detail_path = Path(results_dir) / f"{output_stem}_detail.csv"
    boundary_path = Path(results_dir) / f"{output_stem}_topk_boundary.csv"
    state_monthly_path = Path(results_dir) / f"{output_stem}_daily_proxy_monthly_state.csv"
    state_summary_path = Path(results_dir) / f"{output_stem}_daily_proxy_state_summary.csv"
    daily_leaderboard_path = Path(results_dir) / f"{output_stem}_daily_proxy_leaderboard.csv"
    bundle_manifest_path = Path(results_dir) / f"{output_stem}_bundle_manifest.csv"
    direction_diag_path = Path(results_dir) / f"{output_stem}_direction_diagnostic.csv"
    report_path = Path(results_dir) / f"{output_stem}_report.md"
    json_path = Path(results_dir) / f"{output_stem}.json"
    summary_df.to_csv(summary_path, index=False)
    detail_all.to_csv(detail_path, index=False)
    boundary_all.to_csv(boundary_path, index=False)
    state_monthly_df.to_csv(state_monthly_path, index=False)
    state_summary_df.to_csv(state_summary_path, index=False)
    daily_leaderboard_df.to_csv(daily_leaderboard_path, index=False)
    bundle_manifest_df.to_csv(bundle_manifest_path, index=False)
    direction_diag_df.to_csv(direction_diag_path, index=False)
    payload = {
        "generated_at_utc": generated_at.isoformat(),
        "research_topic": "p1_tree_groups",
        "research_config_id": research_config_id,
        "output_stem": output_stem,
        "summary_csv": str(summary_path),
        "detail_csv": str(detail_path),
        "topk_boundary_csv": str(boundary_path),
        "daily_proxy_monthly_state_csv": str(state_monthly_path),
        "daily_proxy_state_summary_csv": str(state_summary_path),
        "daily_proxy_leaderboard_csv": str(daily_leaderboard_path),
        "bundle_manifest_csv": str(bundle_manifest_path),
        "direction_diagnostic_csv": str(direction_diag_path),
        "report_md": str(report_path),
        "bundle_manifest": bundle_manifest_df.to_dict(orient="records"),
        "label_diagnostics": label_diagnostics,
        "label_meta": label_meta,
        "xgboost_objective": str(args.xgboost_objective),
        "direction_diagnostics": direction_diag_df.to_dict(orient="records"),
        "daily_proxy_state_slices": state_summary_df.to_dict(orient="records"),
        "daily_proxy_first_leaderboard": daily_leaderboard_df.to_dict(orient="records"),
        "groups": summary_df.to_dict(orient="records"),
        "config": {
            "p1_experiment_mode": "daily_proxy_first",
            "legacy_proxy_decision_role": "diagnostic_only",
            "config_source": config_source,
            "benchmark_symbol": "market_ew_proxy",
            "label_horizons": label_horizons,
            "label_weights": label_weights,
            "label_transform": label_transform,
            "label_truncate_quantile": label_truncate_quantile if label_transform == "truncate" else "",
            "label_spec": {
                "label_mode": label_mode,
                "label_scope": str(label_meta.get("label_scope") or label_mode),
                "label_horizons": label_horizons,
                "label_weights": label_weights,
                "label_component_columns": str(label_meta.get("label_component_columns") or ""),
                "label_weights_normalized": str(label_meta.get("label_weights_normalized") or ""),
                "label_market_proxy": str(label_meta.get("label_market_proxy") or ""),
                "label_transform": label_transform,
                "label_truncate_quantile": label_truncate_quantile if label_transform == "truncate" else "",
                "target_column": target_column,
                "proxy_horizon": int(proxy_horizon),
            },
            "proxy_horizon": proxy_horizon,
            "rebalance_rule": args.rebalance_rule,
            "top_k": int(args.top_k),
            "portfolio_method": str(args.backtest_portfolio_method),
            "execution_mode": execution_mode,
            "proxy_max_turnover": float(args.proxy_max_turnover),
            "backtest_config": str(args.backtest_config),
            "backtest_start": str(args.backtest_start),
            "backtest_end": str(args.backtest_end),
            "backtest_top_k": int(args.backtest_top_k),
            "backtest_max_turnover": float(args.backtest_max_turnover),
            "backtest_portfolio_method": str(args.backtest_portfolio_method),
            "backtest_prepared_factors_cache": str(args.backtest_prepared_factors_cache),
            "transaction_costs": cost_params_dict_for_logging(costs),
            "daily_proxy_admission_threshold": daily_proxy_admission_threshold,
            "daily_proxy_full_backtest_threshold": daily_proxy_full_backtest_threshold,
            "daily_proxy_admission_gate_enabled": not bool(args.disable_daily_proxy_admission_gate),
            "val_frac": float(args.val_frac),
        },
    }
    report_text = build_p1_daily_proxy_first_report(
        summary_df=summary_df,
        daily_leaderboard_df=daily_leaderboard_df,
        state_summary_df=state_summary_df,
        boundary_df=boundary_all,
        payload=payload,
    )
    report_path.write_text(report_text, encoding="utf-8")
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
