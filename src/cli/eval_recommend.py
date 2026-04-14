"""推荐 CSV 前向收益标注（由 scripts/daily_run.py eval 子命令调用）。"""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pandas as pd

from ..backtest import attach_forward_returns, summarize_forward_returns
from ..backtest.portfolio_eval import summarize_portfolio_eval
from ..backtest.risk_metrics import risk_config_from_mapping
from ..backtest.transaction_costs import (
    TransactionCostParams,
    transaction_cost_params_from_mapping,
)
from ..data_fetcher import DuckDBManager
from ..logging_config import get_logger, setup_app_logging
from ..notify import find_latest_recommendation_csv
from ..settings import load_config


def run_eval_recommend(args: Namespace, *, root: Path) -> int:
    cfg = load_config(args.config)
    paths = cfg.get("paths", {})
    log_cfg = cfg.get("logging", {}) or {}
    portfolio_cfg = cfg.get("portfolio") or {}
    cost_cfg = cfg.get("transaction_costs") or {}
    risk_cfg = cfg.get("risk") or {}

    results_dir = paths.get("results_dir", "data/results")
    if not Path(results_dir).is_absolute():
        results_dir = root / results_dir

    logs_dir = paths.get("logs_dir", "data/logs")
    if not Path(logs_dir).is_absolute():
        logs_dir = root / logs_dir
    setup_app_logging(
        logs_dir,
        name="eval_recommend",
        log_format=str(log_cfg.get("format", "json")),
    )
    log = get_logger("eval_recommend")

    if args.latest:
        csv_path = find_latest_recommendation_csv(results_dir)
    elif args.csv:
        csv_path = Path(args.csv)
        if not csv_path.is_absolute():
            csv_path = root / csv_path
    else:
        log.error("请指定 --latest 或 --csv")
        return 1

    if not csv_path.exists():
        log.error("文件不存在: %s", csv_path)
        return 1

    rec_df = pd.read_csv(
        csv_path,
        encoding="utf-8-sig",
        converters={
            "symbol": lambda v: str(v).strip(),
            "代码": lambda v: str(v).strip(),
        },
    )
    if rec_df.empty:
        log.error("推荐 CSV 为空")
        return 1

    sym_col = "symbol" if "symbol" in rec_df.columns else "代码"
    if sym_col not in rec_df.columns:
        log.error("推荐表缺少 symbol/代码 列")
        return 1
    rec_df[sym_col] = rec_df[sym_col].astype(str).str.extract(r"(\d{1,6})", expand=False).fillna("").str.zfill(6)

    symbols = rec_df[sym_col].astype(str).str.zfill(6).unique().tolist()
    asof_c = "asof_trade_date" if "asof_trade_date" in rec_df.columns else None
    if asof_c is None:
        log.error("推荐表缺少 asof_trade_date 列")
        return 1

    asof_series = pd.to_datetime(rec_df[asof_c], errors="coerce")
    start = (asof_series.min() - pd.Timedelta(days=60)).normalize()
    end = pd.Timestamp.now().normalize()

    risk_m = risk_config_from_mapping(risk_cfg)
    bench = str(risk_m.get("benchmark_symbol", "510300"))
    symbols_fetch = sorted(set(symbols + [bench]))

    with DuckDBManager(config_path=args.config) as db:
        daily_df = db.read_daily_frame(symbols=symbols_fetch, start=start, end=end)

    if daily_df.empty:
        log.error("日线查询为空；请先运行 daily_run 拉取数据。")
        return 1

    settlement = getattr(args, "settlement", "tplus1_open")
    out_df = attach_forward_returns(
        rec_df,
        daily_df,
        horizon_days=args.horizon,
        symbol_col=sym_col,
        asof_col=asof_c,
        settlement=settlement,
    )
    fwd_col = f"forward_ret_{args.horizon}d"
    summary_row = summarize_forward_returns(out_df, forward_col=fwd_col)

    asof_ts = pd.to_datetime(asof_series.min(), errors="coerce").normalize()
    if pd.isna(asof_ts):
        log.error("asof_trade_date 无法解析")
        return 1

    portfolio_summary = None
    if not getattr(args, "no_portfolio", False):
        cost_params: TransactionCostParams = transaction_cost_params_from_mapping(
            cost_cfg
        )
        prev_path = getattr(args, "prev_weights", None)
        out_df, portfolio_summary = summarize_portfolio_eval(
            out_df,
            forward_col=fwd_col,
            daily_df=daily_df,
            portfolio_cfg=portfolio_cfg,
            cost_params=cost_params,
            risk_cfg=risk_m,
            asof=asof_ts,
            prev_weights_path=str(prev_path) if prev_path else None,
        )

    if args.out:
        out_path = Path(args.out)
        if not out_path.is_absolute():
            out_path = root / out_path
    else:
        out_path = results_dir / f"eval_{csv_path.name}"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    log.info("已写入: %s", out_path)

    out_payload = {
        "per_name": summary_row,
        "portfolio": portfolio_summary,
    }
    if args.json_summary:
        print(json.dumps(out_payload, ensure_ascii=False, default=str))
    else:
        log.info(
            "前向收益汇总 horizon=%s: n=%s n_valid=%s mean=%.6f median=%.6f",
            args.horizon,
            summary_row["n"],
            summary_row["n_valid"],
            summary_row["mean"],
            summary_row["median"],
        )
        if portfolio_summary:
            log.info(
                "组合(无成本) portfolio_gross_ret=%.6f | 组合(含交易成本) portfolio_net_ret_long_hold=%.6f",
                float(portfolio_summary.get("portfolio_gross_ret", float("nan"))),
                float(portfolio_summary.get("portfolio_net_ret_long_hold", float("nan"))),
            )
    return 0
