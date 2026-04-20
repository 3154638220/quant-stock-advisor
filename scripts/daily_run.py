#!/usr/bin/env python3
"""
收盘后流水线：增量更新 DuckDB → GPU 动量/RSI → 本地推荐 CSV。
子命令 eval：对已有推荐 CSV 做前向收益标注。

荐股侧仅做打分排序 + 可选「次日可买性」过滤（停牌、一字涨停），**不**维护 T+1 仓位状态机；
持仓与 T+1 卖出约束由使用者自行管理。

主入口（项目根目录，须使用 conda 环境 quant-system）::

    conda activate quant-system
    python scripts/daily_run.py --max-symbols 100
    python scripts/daily_run.py run --max-symbols 100
    python scripts/daily_run.py --skip-fetch
    python scripts/daily_run.py eval --latest
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_fetcher import DuckDBManager, list_default_universe_symbols
from src.data_fetcher.fundamental_client import FundamentalClient
from src.cli.eval_recommend import run_eval_recommend
from src.features.fundamental_factors import preprocess_fundamental_cross_section
from src.features.ic_monitor import ICMonitor
from src.features.neutralize import neutralize_size_industry_regression
from src.features.panel import pivot_close_wide, pivot_field_aligned_to_close, wide_close_to_numpy
from src.features.tensor_alpha import compute_momentum_rsi_torch
from src.features.tensor_base_factors import compute_base_factor_bundle
from src.features.tree_dataset import long_ohlcv_last_window_table
from src.logging_config import get_logger, setup_app_logging
from src.market.regime import (
    classify_regime,
    get_benchmark_returns_from_db,
    get_regime_weights,
    regime_config_from_mapping,
)
from src.market.tradability import filter_recommend_tradable_next_day, prefilter_stock_pool
from src.models.rank_score import sort_key_for_dataframe
from src.models.recommend_explain import build_recommend_reason_column
from src.notify import save_recommendation_csv
from src.portfolio.covariance import mean_cov_returns_from_wide
from src.portfolio.weights import build_portfolio_weights, portfolio_config_from_mapping
from src.settings import (
    has_explicit_asof_trade_date,
    load_config,
    resolve_asof_trade_end,
)
from scripts.run_backtest_eval import apply_p1_factor_policy, load_factor_ic_summary

EPILOG = """
子命令说明
  run（默认）
    拉取日线（除非 --skip-fetch）、计算因子并写出推荐 CSV。

  eval
    对已有 recommend_*.csv 做前向收益标注，不写因子、不拉取行情。

常用示例
  python scripts/daily_run.py --max-symbols 100
  python scripts/daily_run.py --symbols 600519,000001 --skip-fetch --top-k 10
  python scripts/daily_run.py eval --latest --horizon 5
  python scripts/daily_run.py eval --csv data/results/recommend_2025-03-20.csv

  python scripts/recommend_report.py --latest
  python scripts/recommend_report.py --csv data/results/recommend_2026-03-27.csv
"""


def _normalize_argv(argv: list[str]) -> list[str]:
    """未写子命令时默认 run，兼容旧用法：python daily_run.py --max-symbols 50。"""
    if not argv:
        return ["run"]
    # 顶层 -h/--help 展示含子命令说明的主帮助，勿自动注入 run
    if argv[0] in ("-h", "--help"):
        return argv
    if argv[0] in ("eval", "run", "daily"):
        return argv
    return ["run"] + argv


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="A 股日线增量 + 动量/RSI 推荐池（本地文件输出）；子命令 eval 为事后评估",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=EPILOG,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser(
        "run",
        aliases=("daily",),
        help="增量拉取（可选）+ 因子 + 推荐 CSV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="等价于省略子命令：python scripts/daily_run.py --max-symbols 50",
    )
    run_p.add_argument(
        "--max-symbols",
        type=int,
        default=None,
        help="仅处理全市场列表前 N 只（调试用）；默认按 AkShare 全市场或配置",
    )
    run_p.add_argument(
        "--symbols",
        type=str,
        default=None,
        help="逗号分隔 6 位代码，如 600519,000001；指定时忽略全市场列表与 --max-symbols",
    )
    run_p.add_argument(
        "--skip-fetch",
        action="store_true",
        help="不调用 AkShare，只读 DuckDB 已有日线做信号与输出",
    )
    run_p.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="推荐池长度；默认取 config.yaml 中 signals.top_k",
    )
    run_p.add_argument(
        "--config",
        type=Path,
        default=None,
        help="配置文件路径；默认按 QUANT_CONFIG -> config.yaml -> config.yaml.example 顺序查找",
    )
    run_p.add_argument(
        "--sort-by",
        type=str,
        default=None,
        choices=("momentum", "rsi", "composite", "composite_extended", "xgboost", "deep_sequence"),
        help="排序键；默认 config signals.sort_by（建议默认 composite_extended；xgboost=树模型；deep_sequence=阶段三 OHLCV 序列模型，需 deep_sequence.bundle_dir）",
    )
    run_p.add_argument(
        "--asof-date",
        type=str,
        default=None,
        help="覆盖 config paths.asof_trade_date，统一交易日上界，如 2026-03-27",
    )

    eval_p = sub.add_parser(
        "eval",
        help="对推荐 CSV 标注前向收益（读 DuckDB，不拉行情）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    g = eval_p.add_mutually_exclusive_group()
    g.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="推荐结果 CSV 路径（与 --latest 二选一）",
    )
    g.add_argument(
        "--latest",
        action="store_true",
        help="使用 results_dir 下最新的 recommend_*.csv",
    )
    eval_p.add_argument(
        "--horizon",
        type=int,
        default=5,
        help="前向交易日数量（默认 5）",
    )
    eval_p.add_argument(
        "--settlement",
        type=str,
        default="tplus1_open",
        choices=("tplus1_open", "close_to_close"),
        help="前向收益口径：tplus1_open（默认，与 T+1 次日开盘买入对齐）或 close_to_close（旧收盘口径）",
    )
    eval_p.add_argument(
        "--no-portfolio",
        action="store_true",
        help="不计算组合权重与交易成本（仅保留单票前向收益）",
    )
    eval_p.add_argument(
        "--prev-weights",
        type=Path,
        default=None,
        help="上一期标的权重 CSV（symbol,weight），用于换手约束与换手摩擦近似",
    )
    eval_p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="评估结果 CSV；默认 results_dir/eval_<原文件名>",
    )
    eval_p.add_argument(
        "--json-summary",
        action="store_true",
        help="将汇总指标打印为一行 JSON（便于脚本解析）",
    )
    eval_p.add_argument(
        "--config",
        type=Path,
        default=None,
        help="配置文件路径；默认按 QUANT_CONFIG -> config.yaml -> config.yaml.example 顺序查找",
    )

    return parser


def main_run(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    paths = cfg.get("paths", {}) or {}
    log_cfg = cfg.get("logging", {}) or {}
    feat = cfg.get("features", {})
    sig = cfg.get("signals", {})
    fund_cfg = cfg.get("fundamental", {}) or {}
    portfolio_cfg = portfolio_config_from_mapping(cfg.get("portfolio") or {})
    comp = sig.get("composite") or {}
    comp_ext = sig.get("composite_extended") or {}
    p1_cfg = sig.get("p1_factor_filter") or {}
    ic_cfg = sig.get("ic_weighting") or {}
    tree_m = sig.get("tree_model") or {}
    deep_m = sig.get("deep_sequence") or {}
    gpu = cfg.get("gpu", {})

    top_k = args.top_k if args.top_k is not None else int(sig.get("top_k", 10))
    # P0 约定：生产默认排序键与回测保持一致，回退值使用 composite_extended。
    sort_by = args.sort_by if args.sort_by is not None else str(sig.get("sort_by", "composite_extended")).lower()
    w_mom = float(comp.get("w_momentum", 0.65))
    w_rsi = float(comp.get("w_rsi", 0.35))
    rsi_mode = str(comp.get("rsi_mode", "level")).lower()
    if rsi_mode not in ("level", "mean_revert"):
        rsi_mode = "level"
    effective_comp_ext = dict(comp_ext) if comp_ext else {}
    p1_filter_enabled = bool(p1_cfg.get("enabled", False))
    if sort_by == "composite_extended" and effective_comp_ext and p1_filter_enabled:
        p1_ic_report_path = str(p1_cfg.get("ic_report_path", "")).strip()
        ic_summary = load_factor_ic_summary(p1_ic_report_path)
        if ic_summary.empty:
            log.warning("P1 因子过滤已启用，但 IC 报告为空或不可读：%s；保留静态权重。", p1_ic_report_path)
        else:
            effective_comp_ext, p1_actions = apply_p1_factor_policy(
                effective_comp_ext,
                ic_summary,
                remove_if_t1_and_t21_negative=bool(p1_cfg.get("remove_if_t1_and_t21_negative", True)),
                zero_if_abs_t1_below=float(p1_cfg.get("zero_if_abs_t1_below", 0.0)),
                flip_if_t1_negative_and_t21_above=float(
                    p1_cfg.get("flip_if_t1_negative_and_t21_above", 0.005)
                ),
            )
            if not p1_actions.empty:
                vc = p1_actions["action"].value_counts().to_dict()
                log.info(
                    "P1 因子 IC 规则已应用：keep=%d, zero=%d, flip=%d, remove=%d",
                    int(vc.get("keep", 0)),
                    int(vc.get("zero", 0)),
                    int(vc.get("flip", 0)),
                    int(vc.get("remove", 0)),
                )
    lookback = int(feat.get("lookback_trading_days", 160))
    min_valid = int(feat.get("min_valid_days", 30))
    fund_auto_update = bool(fund_cfg.get("auto_update", True))
    fund_auto_update_max_symbols = int(fund_cfg.get("auto_update_max_symbols", 0) or 0)
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
    neutralize_enabled = bool(feat.get("neutralize", True))

    prefilter = cfg.get("prefilter") or {}
    regime_cfg_raw = cfg.get("regime") or {}
    regime_enabled = bool(regime_cfg_raw.get("enabled", True))
    regime_cfg = regime_config_from_mapping(regime_cfg_raw)
    prefilter_enabled = bool(prefilter.get("enabled", True))

    device_str = gpu.get("device", "cuda")
    dtype_str = str(gpu.get("dtype", "float32")).lower()
    torch_dtype = torch.float32 if dtype_str in ("float32", "fp32") else torch.float64
    if device_str == "cuda" and not torch.cuda.is_available():
        device_str = "cpu"

    results_dir = paths.get("results_dir", "data/results")
    if not Path(results_dir).is_absolute():
        results_dir = ROOT / results_dir

    def _load_llm_attention_factor(trade_date_obj) -> pd.DataFrame:
        """
        读取当日 LLM 关注度 CSV，构造 ``llm_sentiment_z``。
        文件缺失或无可用列时返回空表。
        """
        p = Path(results_dir) / f"llm_attention_{trade_date_obj.isoformat()}.csv"
        if not p.exists():
            return pd.DataFrame(columns=["symbol", "llm_sentiment_z"])
        try:
            tab = pd.read_csv(p, encoding="utf-8-sig")
        except Exception as e:  # noqa: BLE001
            log.warning("读取 LLM 关注度 CSV 失败（跳过）: %s", e)
            return pd.DataFrame(columns=["symbol", "llm_sentiment_z"])
        if tab.empty:
            return pd.DataFrame(columns=["symbol", "llm_sentiment_z"])

        sym_col = "symbol" if "symbol" in tab.columns else ("代码" if "代码" in tab.columns else None)
        if sym_col is None:
            return pd.DataFrame(columns=["symbol", "llm_sentiment_z"])
        sig_col = "significance" if "significance" in tab.columns else None
        rank_col = "attention_rank_change" if "attention_rank_change" in tab.columns else None

        score = pd.Series(0.0, index=tab.index, dtype=float)
        if sig_col is not None:
            score = score + pd.to_numeric(tab[sig_col], errors="coerce").fillna(0.0)
        if rank_col is not None:
            score = score + pd.to_numeric(tab[rank_col], errors="coerce").fillna(0.0) * 0.5
        sd = float(score.std(ddof=0))
        if abs(sd) < 1e-12:
            z = pd.Series(0.0, index=score.index, dtype=float)
        else:
            z = (score - float(score.mean())) / sd

        out_llm = pd.DataFrame(
            {
                "symbol": tab[sym_col]
                .astype(str)
                .str.extract(r"(\d{6})", expand=False)
                .fillna("")
                .str.zfill(6),
                "llm_sentiment_z": z.astype(float),
            }
        )
        out_llm = out_llm[out_llm["symbol"].str.len() == 6]
        out_llm = out_llm.drop_duplicates(subset=["symbol"], keep="last")
        return out_llm

    def _load_ic_weight_override(trade_date_obj) -> Optional[dict[str, float]]:
        """
        P2-A：读取滚动 ICIR 动态权重（若可用）。
        支持两种输入：
        1) update_ic_weights.py 生成的 JSON（含 weights 字段）
        2) 直接读取 ic_monitor.json 并按窗口统计最新 roll_ir（回退方案）
        """
        if not bool(ic_cfg.get("enabled", False)):
            return None
        clip_abs = float(ic_cfg.get("clip_abs_weight", 0.25))
        min_obs = int(ic_cfg.get("min_obs", 20))
        window = int(ic_cfg.get("window", 60))
        override_path = Path(str(ic_cfg.get("weights_path", "data/cache/ic_weights.json")))
        if not override_path.is_absolute():
            override_path = ROOT / override_path
        if override_path.exists():
            try:
                payload = json.loads(override_path.read_text(encoding="utf-8"))
                tab = payload.get("weights_by_date")
                if isinstance(tab, dict):
                    key = pd.Timestamp(trade_date_obj).strftime("%Y-%m-%d")
                    if key in tab and isinstance(tab[key], dict):
                        return {str(k): float(v) for k, v in tab[key].items()}
                if isinstance(payload.get("weights"), dict):
                    return {str(k): float(v) for k, v in payload["weights"].items()}
            except Exception as e_json:  # noqa: BLE001
                log.warning("读取 IC 动态权重文件失败（回退静态）: %s", e_json)

        mon_path = Path(str(ic_cfg.get("monitor_path", "data/logs/ic_monitor.json")))
        if not mon_path.is_absolute():
            mon_path = ROOT / mon_path
        if not mon_path.exists():
            return None
        try:
            mon = ICMonitor(mon_path)
            st = mon.rolling_ic_stats(window=window)
            if st.empty:
                return None
            st = st[st["trade_date"] <= pd.Timestamp(trade_date_obj)]
            if st.empty:
                return None
            latest = st.sort_values("trade_date").groupby("factor", as_index=False).tail(1)
            latest = latest[pd.to_numeric(latest["roll_ir"], errors="coerce").notna()]
            if latest.empty:
                return None
            out_w: dict[str, float] = {}
            for _, r in latest.iterrows():
                if int((st["factor"] == r["factor"]).sum()) < min_obs:
                    continue
                v = float(r["roll_ir"])
                v = max(-clip_abs, min(clip_abs, v))
                out_w[str(r["factor"])] = v
            if not out_w:
                return None
            s_abs = float(sum(abs(v) for v in out_w.values()))
            if s_abs <= 1e-12:
                return None
            return {k: v / s_abs for k, v in out_w.items()}
        except Exception as e_mon:  # noqa: BLE001
            log.warning("从 IC 监控构建动态权重失败（回退静态）: %s", e_mon)
            return None

    logs_dir = paths.get("logs_dir", "data/logs")
    if not Path(logs_dir).is_absolute():
        logs_dir = ROOT / logs_dir
    setup_app_logging(
        logs_dir,
        name="daily_run",
        log_format=str(log_cfg.get("format", "json")),
    )
    log = get_logger("daily_run")

    if args.symbols:
        symbols = [s.strip().zfill(6) for s in args.symbols.split(",") if s.strip()]
    else:
        symbols = list_default_universe_symbols(
            max_symbols=args.max_symbols,
            config_path=args.config,
        )

    if not symbols:
        log.error("无可用标的列表，退出。")
        return 1

    end = resolve_asof_trade_end(paths)
    if getattr(args, "asof_date", None):
        end = pd.Timestamp(str(args.asof_date).strip()).normalize()
    start = end - pd.offsets.BDay(lookback + 20)
    log.info("统一交易日上界: %s", end.date())

    with DuckDBManager(config_path=args.config) as db:
        if not args.skip_fetch:
            counts = db.incremental_update_many(
                symbols,
                end_date=end.strftime("%Y%m%d"),
            )
            n_written = sum(r.rows_written for r in counts.values())
            n_fail = sum(1 for r in counts.values() if r.fetch_failed)
            log.info(
                "增量写入总行数: %s | 拉取失败标的数: %s | run_id=%s",
                n_written,
                n_fail,
                db.last_fetch_run_id,
            )
            qrep = db.quality_report()
            log.info("数据质量: %s", qrep.summary())
            if not qrep.ok:
                log.warning(
                    "质量未通过: %s",
                    "; ".join(qrep.notes) if qrep.notes else qrep.summary(),
                )
            elif qrep.ohlc_invalid_rows or qrep.large_gap_rows:
                log.info(
                    "数据特征统计（源站/长假/停牌等，非拉取失败）: ohlc_invalid=%s, large_gaps=%s",
                    qrep.ohlc_invalid_rows,
                    qrep.large_gap_rows,
                )

        df = db.read_daily_frame(
            symbols=symbols,
            start=start,
            end=end,
        )

    if df.empty:
        log.error("日线查询结果为空；请先拉取数据或检查库路径。")
        return 1

    if prefilter_enabled:
        kept_syms, pf_stats = prefilter_stock_pool(
            df,
            end.date() if hasattr(end, "date") else end,
            limit_move_lookback=int(prefilter.get("limit_move_lookback", 5)),
            limit_move_max=int(prefilter.get("limit_move_max", 2)),
            turnover_low_pct=float(prefilter.get("turnover_low_pct", 0.10)),
            turnover_high_pct=float(prefilter.get("turnover_high_pct", 0.98)),
            price_position_high_pct=float(prefilter.get("price_position_high_pct", 0.90)),
            price_position_lookback=int(prefilter.get("price_position_lookback", 250)),
            log=log,
        )
        if kept_syms:
            df = df[df["symbol"].astype(str).str.zfill(6).isin(kept_syms)].copy()
        else:
            log.warning("预过滤后无剩余标的，跳过过滤继续全量处理。")

    try:
        wide, sym_list, _dates = pivot_close_wide(df, min_valid_days=min_valid)
    except ValueError as e:
        log.error("透视失败: %s", e)
        return 1

    close_np = wide_close_to_numpy(wide)
    dev = torch.device(device_str)
    close_t = torch.from_numpy(close_np).to(device=dev, dtype=torch_dtype)
    mom, rsi = compute_momentum_rsi_torch(
        close_np,
        momentum_window=mom_w,
        rsi_period=rsi_p,
        device=device_str,
        dtype=torch_dtype,
    )
    last_m = mom[:, -1].detach().cpu().numpy()
    last_r = rsi[:, -1].detach().cpu().numpy()

    wide_h = pivot_field_aligned_to_close(df, "high", wide)
    wide_l = pivot_field_aligned_to_close(df, "low", wide)
    wide_vol = pivot_field_aligned_to_close(df, "volume", wide)
    wide_to = pivot_field_aligned_to_close(df, "turnover", wide)
    wide_open = pivot_field_aligned_to_close(df, "open", wide)
    high_np = wide_h.to_numpy(dtype=np.float64)
    low_np = wide_l.to_numpy(dtype=np.float64)
    vol_np = wide_vol.to_numpy(dtype=np.float64)
    to_np = wide_to.to_numpy(dtype=np.float64)
    open_np = wide_open.to_numpy(dtype=np.float64)
    high_t = torch.from_numpy(high_np).to(device=dev, dtype=torch_dtype)
    low_t = torch.from_numpy(low_np).to(device=dev, dtype=torch_dtype)
    vol_t = torch.from_numpy(vol_np).to(device=dev, dtype=torch_dtype)
    to_t = torch.from_numpy(to_np).to(device=dev, dtype=torch_dtype)
    open_t = torch.from_numpy(open_np).to(device=dev, dtype=torch_dtype)

    bundle = compute_base_factor_bundle(
        close_t,
        volume=vol_t,
        turnover=to_t,
        high=high_t,
        low=low_t,
        open_px=open_t,
        vol_window=vol_w,
        turnover_window=to_w,
        vp_corr_window=vp_w,
        reversal_window=rev_w,
        atr_period=atr_p,
        annualize_vol=False,
        bias_window_short=bias_ws,
        bias_window_long=bias_wl,
        max_drop_window=max_drop_w,
        recent_return_window=recent_ret_w,
        price_position_window=pp_w,
        tail_window=tail_w,
        vpt_window=vpt_w,
        range_skew_window=range_skew_w,
        include_intraday=True,
    )

    def _last1(name: str) -> np.ndarray:
        t = bundle.get(name)
        if t is None:
            return np.full(len(sym_list), np.nan, dtype=np.float64)
        return t[:, -1].detach().cpu().numpy()

    trade_date = wide.columns[-1]
    if hasattr(trade_date, "date"):
        panel_last = trade_date.date()
    else:
        panel_last = pd.Timestamp(trade_date).date()

    if pd.Timestamp(trade_date).normalize() != end.normalize():
        log.warning(
            "面板最后交易日 %s 早于统一上界 %s，请确认数据已拉全。",
            panel_last,
            end.date(),
        )

    if getattr(args, "asof_date", None) or has_explicit_asof_trade_date(paths):
        asof_date = end.date()
    else:
        asof_date = panel_last

    out = pd.DataFrame(
        {
            "symbol": sym_list,
            "momentum": last_m,
            "rsi": last_r,
            "atr": _last1("atr"),
            "realized_vol": _last1("realized_vol"),
            "turnover_roll_mean": _last1("turnover_roll_mean"),
            "vol_ret_corr": _last1("vol_ret_corr"),
            "short_reversal": _last1("short_reversal"),
            "vol_to_turnover": _last1("vol_to_turnover"),
            "volume_skew_log": _last1("volume_skew_log"),
            "bias_short": _last1("bias_short"),
            "bias_long": _last1("bias_long"),
            "max_single_day_drop": _last1("max_single_day_drop"),
            "recent_return": _last1("recent_return"),
            "price_position": _last1("price_position"),
            "log_market_cap": _last1("log_market_cap"),
            # K 线结构高频降频代理因子
            "intraday_range": _last1("intraday_range"),
            "upper_shadow_ratio": _last1("upper_shadow_ratio"),
            "lower_shadow_ratio": _last1("lower_shadow_ratio"),
            "close_open_return": _last1("close_open_return"),
            "overnight_gap": _last1("overnight_gap"),
            "tail_strength": _last1("tail_strength"),
            "volume_price_trend": _last1("volume_price_trend"),
            "intraday_range_skew": _last1("intraday_range_skew"),
        }
    )
    out = out.replace([float("inf"), float("-inf")], pd.NA)
    out = out.dropna(subset=["momentum"])
    out["symbol"] = out["symbol"].astype(str).str.zfill(6)

    # P1-A：按公告日对齐基本面快照（point-in-time）
    try:
        fund_symbols = out["symbol"].astype(str).str.zfill(6).drop_duplicates().tolist()
        with FundamentalClient(config_path=args.config) as fc:
            if fund_auto_update and fund_symbols:
                update_symbols = (
                    fund_symbols[:fund_auto_update_max_symbols]
                    if fund_auto_update_max_symbols > 0
                    else fund_symbols
                )
                try:
                    n_upsert = fc.update_symbols(update_symbols)
                    log.info(
                        "P1 基本面增量写入完成：symbols=%d, upsert_rows=%d",
                        len(update_symbols),
                        n_upsert,
                    )
                except Exception as e_fund_upd:  # noqa: BLE001
                    log.warning("基本面增量写入失败（继续使用已落库快照）: %s", e_fund_upd)
            elif not fund_auto_update:
                log.info("P1 基本面增量写入已关闭（fundamental.auto_update=false）。")
            fund = fc.load_point_in_time(
                asof_date=asof_date,
                symbols=fund_symbols,
            )
        if not fund.empty:
            out = out.merge(fund, on="symbol", how="left")
            out["trade_date"] = pd.Timestamp(asof_date)
            out = preprocess_fundamental_cross_section(
                out,
                date_col="trade_date",
                size_col="log_market_cap",
                industry_col=portfolio_cfg.get("industry_col"),
                neutralize=neutralize_enabled,
            )
            out = out.drop(columns=["trade_date"], errors="ignore")
            log.info("P1 基本面因子已接入：%d 只标的具备 PIT 基本面快照", int(fund["symbol"].nunique()))
        else:
            log.info("P1 基本面因子：fundamental 表暂无可用快照，本次回退技术面因子。")
    except Exception as e_fund:  # noqa: BLE001
        log.warning("加载 PIT 基本面因子失败（回退技术面）: %s", e_fund)

    # P1-C：接入 LLM 关注度情绪因子（若当日结果文件存在）
    llm_fac = _load_llm_attention_factor(asof_date)
    if not llm_fac.empty:
        out = out.merge(llm_fac, on="symbol", how="left")
        log.info("P1 LLM 因子已接入：%d 只标的含 llm_sentiment_z", int(llm_fac["symbol"].nunique()))
    elif "llm_sentiment_z" not in out.columns:
        out["llm_sentiment_z"] = np.nan

    # ——— 市值中性化：对截面因子去除市值暴露 ———
    # A 股市值效应极强，不中性化的因子 IC 很可能大部分来自市值暴露。
    # 仅对打分参与列做截面回归残差，log_market_cap 本身保留原值。
    _FACTORS_TO_NEUTRALIZE = [
        "momentum", "rsi", "atr", "realized_vol", "turnover_roll_mean",
        "vol_ret_corr", "short_reversal", "vol_to_turnover", "volume_skew_log",
        "bias_short", "bias_long", "max_single_day_drop", "recent_return",
        "price_position", "intraday_range", "upper_shadow_ratio",
        "lower_shadow_ratio", "close_open_return", "overnight_gap",
        "tail_strength", "volume_price_trend", "intraday_range_skew",
        # P1 扩展：基本面 / 资金流 / LLM 情绪
        "pe_ttm", "pb", "ev_ebitda", "roe_ttm", "net_profit_yoy",
        "gross_margin_change", "debt_to_assets_change", "ocf_to_net_profit",
        "northbound_net_inflow", "margin_buy_ratio", "llm_sentiment_z",
    ]
    if neutralize_enabled and "log_market_cap" in out.columns:
        _industry_col = str(portfolio_cfg.get("industry_col") or "").strip() or None
        _tmp = out.copy()
        _tmp["_nd"] = "_"  # 单截面虚拟日期列
        _neutralized = 0
        for _fc in _FACTORS_TO_NEUTRALIZE:
            if _fc not in _tmp.columns:
                continue
            _neut_col = f"{_fc}_neut"
            try:
                _tmp = neutralize_size_industry_regression(
                    _tmp,
                    _fc,
                    size_col="log_market_cap",
                    industry_col=_industry_col or "__no_industry__",
                    date_col="_nd",
                    suffix="_neut",
                )
                if _neut_col in _tmp.columns:
                    out[_fc] = _tmp[_neut_col].values
                    _neutralized += 1
            except Exception as _e:
                log.warning("中性化因子 %s 失败（跳过）: %s", _fc, _e)
        log.info(
            "截面市值中性化：已处理 %d 个因子列（industry_col=%s）",
            _neutralized,
            _industry_col or "未配置",
        )
    elif neutralize_enabled:
        log.debug("neutralize=true 但缺少 log_market_cap 列，跳过中性化。")

    tree_bundle: Optional[Path] = None
    tree_feat_list: Optional[List[str]] = None
    tree_rsi_mode: Optional[str] = None
    deep_bundle: Optional[Path] = None
    deep_long_df: Optional[pd.DataFrame] = None
    deep_map_loc: str = "cpu"
    if sort_by == "composite":
        out = out.dropna(subset=["rsi"])
    elif sort_by == "composite_extended":
        ext_w = {k: float(v) for k, v in effective_comp_ext.items() if isinstance(k, str)}
        need_all = [
            k
            for k, v in ext_w.items()
            if abs(v) > 1e-15 and k in out.columns
        ]
        if not need_all:
            log.error("composite_extended 权重全为 0 或无效，退出。")
            return 1
        # P1 起部分新因子（基本面/情绪）可能缺历史或当日覆盖，按覆盖率动态启用，避免全量 dropna。
        need = []
        dropped_sparse: list[str] = []
        for c in need_all:
            cov = float(pd.to_numeric(out[c], errors="coerce").notna().mean())
            if cov >= 0.20:
                need.append(c)
            else:
                dropped_sparse.append(c)
        if dropped_sparse:
            log.info("composite_extended 稀疏因子暂不参与（覆盖率<20%%）: %s", dropped_sparse)
        if not need:
            log.error("composite_extended 无可用因子列（覆盖率不足），退出。")
            return 1
        out = out.dropna(subset=need, how="all")
    elif sort_by == "xgboost":
        bundle_s = str(tree_m.get("bundle_dir", "")).strip()
        if not bundle_s:
            log.error("sort_by=xgboost 须在 config signals.tree_model.bundle_dir 指定训练工件目录。")
            return 1
        bundle_path = Path(bundle_s)
        if not bundle_path.is_absolute():
            bundle_path = ROOT / bundle_path
        if not bundle_path.is_dir():
            log.error("树模型工件不存在或不是目录: %s", bundle_path)
            return 1
        feats_cfg = tree_m.get("features")
        if feats_cfg:
            tree_feats = [str(x) for x in feats_cfg]
        else:
            tree_feats = [
                k
                for k, v in comp_ext.items()
                if isinstance(k, str) and abs(float(v)) > 1e-15 and k in out.columns
            ]
        if not tree_feats:
            tree_feats = [
                c
                for c in out.columns
                if c
                in (
                    "momentum",
                    "rsi",
                    "atr",
                    "realized_vol",
                    "turnover_roll_mean",
                    "vol_ret_corr",
                    "short_reversal",
                    "vol_to_turnover",
                    "volume_skew_log",
                )
            ]
        miss = [c for c in tree_feats if c not in out.columns]
        if miss:
            log.error("xgboost 排序缺少因子列: %s", miss)
            return 1
        out = out.dropna(subset=tree_feats)
        trsi = str(tree_m.get("rsi_mode") or rsi_mode).lower()
        if trsi not in ("level", "mean_revert"):
            trsi = str(rsi_mode).lower()
        tree_bundle = bundle_path
        tree_feat_list = tree_feats
        tree_rsi_mode = trsi
    elif sort_by == "deep_sequence":
        bundle_s = str(deep_m.get("bundle_dir", "")).strip()
        if not bundle_s:
            log.error("sort_by=deep_sequence 须在 config signals.deep_sequence.bundle_dir 指定训练工件目录。")
            return 1
        bundle_path = Path(bundle_s)
        if not bundle_path.is_absolute():
            bundle_path = ROOT / bundle_path
        if not bundle_path.is_dir():
            log.error("deep_sequence 工件不存在或不是目录: %s", bundle_path)
            return 1
        deep_bundle = bundle_path
        seq_ds = int(deep_m.get("seq_len", 30))
        try:
            deep_long_df = long_ohlcv_last_window_table(
                sym_list,
                list(wide.columns),
                wide_open=wide_open,
                wide_high=wide_h,
                wide_low=wide_l,
                wide_close=wide,
                wide_volume=wide_vol,
                seq_len=seq_ds,
            )
        except ValueError as e:
            log.error("构造 OHLCV 序列表失败: %s", e)
            return 1
        if deep_long_df.empty:
            log.error("deep_sequence 无有效 OHLCV 行。")
            return 1
        map_s = str(deep_m.get("map_location", "")).strip().lower()
        if map_s in ("cuda", "gpu") and device_str == "cuda":
            deep_map_loc = "cuda"
        else:
            deep_map_loc = "cpu"

    # ——— Regime Switch：动态权重调整 ———
    effective_w_mom = w_mom
    effective_w_rsi = w_rsi
    regime_label = "oscillation"
    if regime_enabled and sort_by in ("composite", "composite_extended"):
        try:
            benchmark_sym = str((cfg.get("risk") or {}).get("benchmark_symbol", "510300"))
            with DuckDBManager(config_path=args.config) as db_regime:
                bench_rets = get_benchmark_returns_from_db(
                    db_regime,
                    benchmark_sym,
                    lookback_days=int(regime_cfg_raw.get("long_window", 60)) + 10,
                    asof_date=asof_date,
                )
            regime_label, regime_res = classify_regime(bench_rets, asof_date, cfg=regime_cfg)
            log.info(
                "大盘状态: %s | 短期收益=%.2f%% | 年化波动率=%.1f%%",
                regime_label.upper(),
                regime_res.short_return * 100,
                regime_res.realized_vol_ann * 100,
            )
            if sort_by == "composite_extended" and effective_comp_ext:
                effective_comp_ext = get_regime_weights(
                    effective_comp_ext,
                    regime_label,
                    cfg=regime_cfg,
                    regime_result=regime_res,
                )
                log.info("Regime 权重调整后（%s）: %s", regime_label, effective_comp_ext)
            elif sort_by == "composite":
                # 动态调整动量/RSI 权重
                base_w = {"momentum": w_mom, "rsi": w_rsi}
                adj_w = get_regime_weights(
                    base_w,
                    regime_label,
                    cfg=regime_cfg,
                    regime_result=regime_res,
                )
                effective_w_mom = float(adj_w.get("momentum", w_mom))
                effective_w_rsi = float(adj_w.get("rsi", w_rsi))
                s = effective_w_mom + effective_w_rsi
                if s > 1e-10:
                    effective_w_mom /= s
                    effective_w_rsi /= s
                log.info(
                    "Regime 调整后 composite 权重 —— momentum=%.3f, rsi=%.3f",
                    effective_w_mom, effective_w_rsi,
                )
        except Exception as e_regime:
            log.warning("Regime Switch 失败（使用原始权重）: %s", e_regime)

    ic_weights_override = None
    if sort_by == "composite_extended":
        ic_weights_override = _load_ic_weight_override(asof_date)
        if ic_weights_override:
            log.info("P2 IC 动态权重已启用：%d 个因子覆盖静态权重", len(ic_weights_override))

    out = sort_key_for_dataframe(
        out,
        sort_by=sort_by,
        w_momentum=effective_w_mom,
        w_rsi=effective_w_rsi,
        rsi_mode=rsi_mode,
        composite_extended_weights=effective_comp_ext if sort_by == "composite_extended" else None,
        composite_extended_weights_override=ic_weights_override if sort_by == "composite_extended" else None,
        tree_bundle_dir=tree_bundle,
        tree_raw_features=tree_feat_list,
        tree_rsi_mode=tree_rsi_mode,
        deep_sequence_bundle_dir=deep_bundle,
        deep_sequence_long_df=deep_long_df,
        deep_sequence_map_location=deep_map_loc,
    )
    out["regime"] = regime_label
    out.insert(2, "asof_trade_date", asof_date.isoformat())
    pre_pool = min(len(out), max(top_k * 5, top_k + 20))
    out = out.iloc[:pre_pool].copy()
    out, n_drop = filter_recommend_tradable_next_day(
        out,
        df,
        asof_date=asof_date,
        symbol_col="symbol",
        log=log,
    )
    if n_drop:
        log.info("次日可买性过滤剔除 %s 只（停牌/开盘一字涨停）", n_drop)
    out = out.head(top_k)

    wm = str(portfolio_cfg["weight_method"]).lower()
    cov_methods = ("risk_parity", "min_variance", "mean_variance")
    cov_mtx = None
    exp_ret = None
    if wm in cov_methods:
        lb = int(portfolio_cfg.get("cov_lookback_days", 60))
        ridge = float(portfolio_cfg.get("cov_ridge", 1e-6))
        shr = str(portfolio_cfg.get("cov_shrinkage", "ledoit_wolf")).lower()
        if shr not in ("ledoit_wolf", "sample", "ewma", "industry_factor"):
            shr = "ledoit_wolf"
        syms_out = out["symbol"].astype(str).str.zfill(6).tolist()
        ind_labels = None
        ind_col_cfg = portfolio_cfg.get("industry_col")
        if shr == "industry_factor" and isinstance(ind_col_cfg, str) and ind_col_cfg in out.columns:
            ind_labels = out[ind_col_cfg].astype(str).fillna("_NA_").tolist()
        mu_arr, cov_mtx = mean_cov_returns_from_wide(
            wide,
            syms_out,
            lookback_days=lb,
            ridge=ridge,
            shrinkage=shr,  # type: ignore[arg-type]
            ewma_halflife=float(portfolio_cfg.get("cov_ewma_halflife", 20.0)),
            industry_labels=ind_labels,
        )
        if wm == "mean_variance":
            exp_ret = mu_arr

    w = build_portfolio_weights(
        out,
        weight_method=portfolio_cfg["weight_method"],
        score_col=portfolio_cfg["score_col"],
        max_single_weight=portfolio_cfg["max_single_weight"],
        max_industry_weight=portfolio_cfg.get("max_industry_weight"),
        industry_col=portfolio_cfg.get("industry_col"),
        prev_weights_aligned=None,
        max_turnover=portfolio_cfg["max_turnover"],
        cov_matrix=cov_mtx,
        expected_returns=exp_ret,
        risk_aversion=float(portfolio_cfg.get("risk_aversion", 1.0)),
        turnover_cost_model=portfolio_cfg.get("turnover_cost_model"),
    )
    out["weight"] = w
    out["sort_by"] = sort_by
    out["recommend_reason"] = build_recommend_reason_column(
        out,
        sort_by=sort_by,
        rsi_mode=rsi_mode,
        composite_extended_weights=effective_comp_ext if sort_by == "composite_extended" else None,
        w_momentum=w_mom,
        w_rsi=w_rsi,
    )

    path = save_recommendation_csv(
        out,
        results_dir=results_dir,
        asof=asof_date,
        prefix="recommend",
    )
    log.info("已写入: %s", path)
    return 0


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args(_normalize_argv(sys.argv[1:]))
    if args.command == "eval":
        return run_eval_recommend(args, root=ROOT)
    return main_run(args)


if __name__ == "__main__":
    raise SystemExit(main())
