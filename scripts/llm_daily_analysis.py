#!/usr/bin/env python3
"""
本地 LLM 日常分析脚本：读取推荐 CSV，用 Ollama（qwen2.5）提炼：
  1. 当日市场整体利好/利空（宏观快讯摘要）
  2. 各推荐股票的新闻情绪（利好/利空要点）
  3. 各推荐股票的财务报表关键指标

输出：
  data/results/llm_analysis_<date>.json   ← 完整结构化结果
  data/results/llm_enriched_<date>.csv    ← 推荐 CSV + LLM 列

用法::

    # 分析最新推荐 CSV（默认）
    python scripts/llm_daily_analysis.py

    # 指定 CSV 文件
    python scripts/llm_daily_analysis.py --csv data/results/recommend_2026-03-27.csv

    # 只分析前 N 只股票
    python scripts/llm_daily_analysis.py --top-k 5

    # 仅做市场宏观分析，跳过个股
    python scripts/llm_daily_analysis.py --market-only

    # 指定模型（默认 qwen2.5:7b）
    python scripts/llm_daily_analysis.py --model qwen2.5:3b

    # 仅做财务分析（跳过新闻）
    python scripts/llm_daily_analysis.py --no-news

    # 仅做新闻分析（跳过财务）
    python scripts/llm_daily_analysis.py --no-financial
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.llm import FinancialAnalyzer, NewsAnalyzer, OllamaClient
from src.logging_config import get_logger, setup_app_logging
from src.settings import load_config

_LOG = get_logger(__name__)


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------


def _find_latest_recommend_csv(results_dir: Path) -> Optional[Path]:
    """找 results_dir 中最新的 recommend_*.csv。"""
    csvs = sorted(results_dir.glob("recommend_*.csv"), reverse=True)
    return csvs[0] if csvs else None


def _extract_trade_date_from_csv_path(csv_path: Path) -> str:
    """从文件名 recommend_YYYY-MM-DD.csv 提取日期，失败则用今天。"""
    stem = csv_path.stem  # recommend_2026-03-27
    parts = stem.split("_", 1)
    if len(parts) == 2:
        return parts[1]
    return date.today().isoformat()


def _load_recommend_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, dtype={"symbol": str})
    if "symbol" not in df.columns:
        raise ValueError(f"推荐 CSV 缺少 symbol 列: {csv_path}")
    return df


def _get_name_col(df: pd.DataFrame) -> Optional[str]:
    for c in ["name", "股票名称", "stock_name"]:
        if c in df.columns:
            return c
    return None


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


def run_llm_analysis(args: argparse.Namespace) -> None:
    cfg = load_config()
    llm_cfg = cfg.get("llm", {})

    model = args.model or llm_cfg.get("model", "qwen2.5:7b")
    timeout = (
        float(args.timeout)
        if args.timeout is not None
        else float(llm_cfg.get("timeout_sec", 120.0))
    )
    max_retries = llm_cfg.get("max_retries", 2)
    base_url = llm_cfg.get("base_url", "http://localhost:11434")
    ollama_options = llm_cfg.get("ollama_options")
    if ollama_options is not None and not isinstance(ollama_options, dict):
        ollama_options = None
    top_k = args.top_k or llm_cfg.get("default_top_k", 10)
    results_dir = Path(cfg.get("paths", {}).get("results_dir", "data/results"))
    results_dir.mkdir(parents=True, exist_ok=True)

    # ---- 确定推荐 CSV ----
    if args.csv:
        csv_path = Path(args.csv)
    else:
        csv_path = _find_latest_recommend_csv(results_dir)
    if csv_path is None or not csv_path.exists():
        _LOG.error("未找到推荐 CSV，请先运行 daily_run.py 或用 --csv 指定。")
        sys.exit(1)

    trade_date = _extract_trade_date_from_csv_path(csv_path)
    output_json = results_dir / f"llm_analysis_{trade_date}.json"
    output_csv = results_dir / f"llm_enriched_{trade_date}.csv"

    _LOG.info("=== LLM 日常分析 | 日期: %s | 模型: %s ===", trade_date, model)
    _LOG.info("读取推荐 CSV: %s", csv_path)

    # ---- 初始化客户端 ----
    client = OllamaClient(
        model=model,
        timeout=timeout,
        max_retries=max_retries,
        base_url=base_url,
        ollama_options=ollama_options,
    )
    if not client.is_available():
        _LOG.error("Ollama 服务不可达或模型 %s 未加载，请先运行: ollama run %s", model, model)
        sys.exit(1)

    # ---- 加载推荐表 ----
    rec_df = _load_recommend_csv(csv_path)
    name_col = _get_name_col(rec_df)

    # 选取 top-k
    if top_k and top_k < len(rec_df):
        stocks_df = rec_df.head(top_k).copy()
    else:
        stocks_df = rec_df.copy()

    symbol_list: list[tuple[str, str]] = [
        (str(row["symbol"]), str(row[name_col]) if name_col else str(row["symbol"]))
        for _, row in stocks_df.iterrows()
    ]

    analysis_result: dict = {
        "trade_date": trade_date,
        "model": model,
        "ollama_options": ollama_options or {},
        "generated_at": datetime.now().isoformat(),
        "market_macro": {},
        "stocks_news": [],
        "stocks_financial": [],
    }

    # ---- 1. 市场宏观分析 ----
    _LOG.info("--- 步骤 1/3：市场宏观利好/利空分析 ---")
    news_analyzer = NewsAnalyzer(
        client=client,
        max_news_per_stock=llm_cfg.get("max_news_per_stock", 10),
        max_market_news=llm_cfg.get("max_market_news", 20),
        sleep_between_stocks=llm_cfg.get("sleep_between_stocks_sec", 1.0),
    )
    macro = news_analyzer.analyze_market(trade_date)
    analysis_result["market_macro"] = macro.to_dict()
    _LOG.info(
        "市场情绪: %s (score=%d) | %s",
        macro.market_sentiment,
        macro.market_score,
        macro.summary,
    )
    if macro.top_bullish:
        _LOG.info("核心利好: %s", " / ".join(macro.top_bullish[:3]))
    if macro.top_bearish:
        _LOG.info("核心利空: %s", " / ".join(macro.top_bearish[:3]))

    if args.market_only:
        _save_results(analysis_result, output_json, rec_df, stocks_df, output_csv)
        return

    # ---- 2. 个股新闻情绪分析 ----
    news_rows: list[dict] = []
    if not args.no_news:
        _LOG.info("--- 步骤 2/3：个股新闻情绪分析（%d 只）---", len(symbol_list))
        sentiments = news_analyzer.batch_analyze_stocks(symbol_list, trade_date)
        for s in sentiments:
            analysis_result["stocks_news"].append(s.to_dict())
            news_rows.append(s.to_dict())
            label = {"bullish": "利好", "bearish": "利空", "neutral": "中性"}.get(s.sentiment, s.sentiment)
            _LOG.info("  %s %s → %s (score=%d) %s", s.symbol, s.name, label, s.score, s.summary)
    else:
        _LOG.info("--- 步骤 2/3：跳过个股新闻（--no-news）---")

    # ---- 3. 个股财务分析 ----
    fin_rows: list[dict] = []
    if not args.no_financial:
        _LOG.info("--- 步骤 3/3：个股财务关键指标提取（%d 只）---", len(symbol_list))
        fin_analyzer = FinancialAnalyzer(
            client=client,
            sleep_between_stocks=llm_cfg.get("sleep_between_financial_sec", 1.5),
        )
        financials = fin_analyzer.batch_analyze_stocks(symbol_list)
        for f in financials:
            analysis_result["stocks_financial"].append(f.to_dict())
            fin_rows.append(f.to_dict())
            _LOG.info(
                "  %s %s → %s | ROE=%.1f%% | 营收增速=%.1f%%",
                f.symbol,
                f.name,
                f.rating,
                (f.roe or 0),
                (f.revenue_growth_yoy or 0),
            )
    else:
        _LOG.info("--- 步骤 3/3：跳过财务分析（--no-financial）---")

    # ---- 保存结果 ----
    _save_results(analysis_result, output_json, rec_df, stocks_df, output_csv, news_rows, fin_rows)


def _save_results(
    analysis_result: dict,
    output_json: Path,
    rec_df: pd.DataFrame,
    stocks_df: pd.DataFrame,
    output_csv: Path,
    news_rows: list[dict] | None = None,
    fin_rows: list[dict] | None = None,
) -> None:
    # 保存 JSON
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(analysis_result, f, ensure_ascii=False, indent=2)
    _LOG.info("完整分析结果已保存: %s", output_json)

    # 合并到推荐 CSV
    enriched = stocks_df.copy()
    enriched["symbol"] = enriched["symbol"].astype(str)

    if news_rows:
        news_df = pd.DataFrame(news_rows)
        news_df["symbol"] = news_df["symbol"].astype(str)
        news_cols = [c for c in news_df.columns if c != "name"]
        enriched = enriched.merge(news_df[news_cols], on="symbol", how="left")

    if fin_rows:
        fin_df = pd.DataFrame(fin_rows)
        fin_df["symbol"] = fin_df["symbol"].astype(str)
        fin_cols = [c for c in fin_df.columns if c != "name"]
        enriched = enriched.merge(fin_df[fin_cols], on="symbol", how="left")

    enriched.to_csv(output_csv, index=False, encoding="utf-8-sig")
    _LOG.info("增强推荐 CSV 已保存: %s", output_csv)

    # 打印终端摘要
    _print_summary(analysis_result, enriched)


def _print_summary(analysis_result: dict, enriched: pd.DataFrame) -> None:
    macro = analysis_result.get("market_macro", {})
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  LLM 分析摘要 | {analysis_result.get('trade_date', '')} | {analysis_result.get('model', '')}")
    print(sep)

    sentiment_label = {"bullish": "🟢 利好", "bearish": "🔴 利空", "neutral": "⚪ 中性"}.get(
        macro.get("market_sentiment", "neutral"), "⚪"
    )
    print(f"\n【市场整体】{sentiment_label}  score={macro.get('market_score', 0)}")
    print(f"  结论: {macro.get('summary', '-')}")
    if macro.get("top_bullish"):
        print(f"  核心利好: {' / '.join(macro['top_bullish'][:3])}")
    if macro.get("top_bearish"):
        print(f"  核心利空: {' / '.join(macro['top_bearish'][:3])}")
    if macro.get("hot_sectors"):
        print(f"  热点板块: {' / '.join(macro['hot_sectors'][:4])}")

    # 个股摘要表
    disp_cols = ["symbol"]
    for c in ["name", "llm_sentiment", "llm_news_score", "llm_news_summary", "fin_rating", "fin_summary"]:
        if c in enriched.columns:
            disp_cols.append(c)

    if len(disp_cols) > 1:
        print(f"\n【个股分析摘要】（前 {min(len(enriched), 20)} 只）")
        print(enriched[disp_cols].head(20).to_string(index=False))

    print(f"\n{sep}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="本地 LLM 新闻情绪 + 财务报表分析（基于 Ollama）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--csv", default=None, help="指定推荐 CSV 路径；默认自动找最新 recommend_*.csv")
    p.add_argument("--top-k", type=int, default=None, help="只分析排名前 K 只股票（默认 config llm.default_top_k 或 10）")
    p.add_argument(
        "--model",
        default=None,
        help="Ollama 模型名，如 qwen2.5:7b / qwen2.5:14b / qwen2.5:32b（默认 config 中配置）",
    )
    p.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="单次请求超时（秒），覆盖 config 的 llm.timeout_sec；大模型建议 300~600",
    )
    p.add_argument("--market-only", action="store_true", help="只做市场宏观分析，跳过个股")
    p.add_argument("--no-news", action="store_true", help="跳过个股新闻情绪分析")
    p.add_argument("--no-financial", action="store_true", help="跳过个股财务分析")
    p.add_argument("--date", default=None, help="手动指定分析日期 YYYY-MM-DD（影响快讯拉取；默认从 CSV 文件名提取）")
    return p


def main() -> None:
    cfg = load_config()
    logs_dir = Path(cfg.get("paths", {}).get("logs_dir", "data/logs"))
    logs_dir.mkdir(parents=True, exist_ok=True)
    setup_app_logging(logs_dir, name="llm_analysis")
    parser = _build_parser()
    args = parser.parse_args()
    run_llm_analysis(args)


if __name__ == "__main__":
    main()
