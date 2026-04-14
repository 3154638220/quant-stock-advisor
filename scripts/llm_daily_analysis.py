#!/usr/bin/env python3
"""
新闻驱动关注度扫描脚本：从东方财富关注度飙升榜出发，用本地 LLM 筛选出
因真实新闻事件（而非市场波动）导致关注度大幅提升的股票。

工作流：
  1. 从飙升榜/人气榜获取关注度飙升的股票列表
  2. 逐只拉取近期新闻 → LLM 判定是否为新闻事件驱动
  3. （可选）对新闻驱动的标的做财务指标提取

输出：
  data/results/llm_attention_<date>.json  ← 完整扫描结果
  data/results/llm_attention_<date>.csv   ← 新闻驱动股票列表

用法::

    # 默认：扫描飙升榜前 50 只，LLM 过滤
    python scripts/llm_daily_analysis.py

    # 扫描前 30 只
    python scripts/llm_daily_analysis.py --top-k 30

    # 同时为新闻驱动的标的做财务分析
    python scripts/llm_daily_analysis.py --with-financial

    # 指定模型（默认 qwen2.5:32b，见 config.yaml）
    python scripts/llm_daily_analysis.py --model qwen2.5:3b

    # 也做市场宏观分析
    python scripts/llm_daily_analysis.py --with-macro
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.llm import AttentionScanner, FinancialAnalyzer, NewsAnalyzer, OllamaClient
from src.logging_config import get_logger, setup_app_logging
from src.settings import load_config

_LOG = get_logger(__name__)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


def run_attention_scan(args: argparse.Namespace) -> None:
    cfg = load_config()
    llm_cfg = cfg.get("llm", {})
    attn_cfg = llm_cfg.get("attention", {})

    model = args.model or llm_cfg.get("model", "qwen2.5:32b")
    timeout = (
        float(args.timeout)
        if args.timeout is not None
        else float(llm_cfg.get("timeout_sec", 400.0))
    )
    max_retries = llm_cfg.get("max_retries", 2)
    base_url = llm_cfg.get("base_url", "http://localhost:11434")
    ollama_options = llm_cfg.get("ollama_options")
    if ollama_options is not None and not isinstance(ollama_options, dict):
        ollama_options = None

    top_k = args.top_k or attn_cfg.get("max_surge_stocks", 50)
    news_lookback_hours = int(attn_cfg.get("news_lookback_hours", 48))
    results_dir = Path(cfg.get("paths", {}).get("results_dir", "data/results"))
    results_dir.mkdir(parents=True, exist_ok=True)

    trade_date = args.date or date.today().isoformat()
    output_json = results_dir / f"llm_attention_{trade_date}.json"
    output_csv = results_dir / f"llm_attention_{trade_date}.csv"

    if args.analysis_time:
        try:
            analysis_reference_time = datetime.fromisoformat(args.analysis_time)
        except ValueError:
            _LOG.error("--analysis-time 格式错误，应为 ISO 时间，例如 2026-03-30T15:30:00")
            sys.exit(2)
    else:
        analysis_reference_time = datetime.now()

    _LOG.info("=== 新闻驱动关注度扫描 | 日期: %s | 模型: %s ===", trade_date, model)
    _LOG.info(
        "分析基准时刻: %s | 新闻窗口: 近 %d 小时",
        analysis_reference_time.strftime("%Y-%m-%d %H:%M:%S"),
        news_lookback_hours,
    )

    # ---- 初始化客户端 ----
    client = OllamaClient(
        model=model,
        timeout=timeout,
        max_retries=max_retries,
        base_url=base_url,
        ollama_options=ollama_options,
    )
    if not client.is_available():
        _LOG.error(
            "Ollama 服务不可达或模型 %s 未加载，请先运行: ollama run %s",
            model,
            model,
        )
        sys.exit(1)

    analysis_result: dict = {
        "trade_date": trade_date,
        "model": model,
        "generated_at": datetime.now().isoformat(),
        "market_macro": {},
        "attention_stocks": [],
    }

    # ---- 可选：市场宏观分析 ----
    if args.with_macro:
        _LOG.info("--- 市场宏观分析 ---")
        news_analyzer = NewsAnalyzer(
            client=client,
            max_market_news=llm_cfg.get("max_market_news", 20),
            config=cfg,
        )
        macro = news_analyzer.analyze_market(trade_date)
        analysis_result["market_macro"] = macro.to_dict()
        _LOG.info(
            "市场情绪: %s (score=%d) | %s",
            macro.market_sentiment,
            macro.market_score,
            macro.summary,
        )

    # ---- 核心：关注度飙升扫描 ----
    _LOG.info("--- 关注度飙升扫描（前 %d 只）---", top_k)
    scanner = AttentionScanner(
        client=client,
        max_surge_stocks=top_k,
        max_news_per_stock=attn_cfg.get("max_news_per_stock", 15),
        sleep_between_stocks=attn_cfg.get(
            "sleep_between_stocks_sec",
            llm_cfg.get("sleep_between_stocks_sec", 1.0),
        ),
        news_lookback_hours=news_lookback_hours,
        analysis_reference_time=analysis_reference_time,
        config=cfg,
    )
    all_stocks = scanner.scan()
    news_driven = [s for s in all_stocks if s.is_news_driven]

    analysis_result["attention_stocks"] = [s.to_dict() for s in all_stocks]
    analysis_result["news_driven_count"] = len(news_driven)
    analysis_result["total_scanned"] = len(all_stocks)

    # ---- 可选：对新闻驱动标的做财务分析 ----
    if args.with_financial and news_driven:
        _LOG.info("--- 新闻驱动标的财务分析（%d 只）---", len(news_driven))
        fin_analyzer = FinancialAnalyzer(
            client=client,
            sleep_between_stocks=llm_cfg.get("sleep_between_financial_sec", 1.5),
            config=cfg,
        )
        symbol_list = [(s.symbol, s.name) for s in news_driven]
        financials = fin_analyzer.batch_analyze_stocks(symbol_list)
        analysis_result["financials"] = [f.to_dict() for f in financials]

    # ---- 保存 ----
    _save_results(analysis_result, output_json, output_csv, all_stocks, news_driven)


def _save_results(
    analysis_result: dict,
    output_json: Path,
    output_csv: Path,
    all_stocks: list,
    news_driven: list,
) -> None:
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(analysis_result, f, ensure_ascii=False, indent=2)
    _LOG.info("完整结果已保存: %s", output_json)

    if news_driven:
        rows = []
        for s in news_driven:
            d = s.to_dict()
            d["key_headlines"] = " | ".join(d.get("key_headlines", []))
            rows.append(d)
        df = pd.DataFrame(rows)
        df.to_csv(output_csv, index=False, encoding="utf-8-sig")
        _LOG.info("新闻驱动股票 CSV: %s (%d 只)", output_csv, len(news_driven))
    else:
        _LOG.info("无新闻驱动股票，未生成 CSV")

    _print_summary(analysis_result, all_stocks, news_driven)


def _print_summary(
    analysis_result: dict,
    all_stocks: list,
    news_driven: list,
) -> None:
    sep = "=" * 64
    print(f"\n{sep}")
    print(
        f"  新闻驱动关注度扫描 | {analysis_result.get('trade_date', '')} "
        f"| {analysis_result.get('model', '')}"
    )
    print(sep)

    macro = analysis_result.get("market_macro", {})
    if macro and macro.get("summary"):
        sentiment_label = {
            "bullish": "🟢 偏多",
            "bearish": "🔴 偏空",
            "neutral": "⚪ 中性",
        }.get(macro.get("market_sentiment", "neutral"), "⚪")
        print(f"\n【市场整体】{sentiment_label}  score={macro.get('market_score', 0)}")
        print(f"  {macro.get('summary', '-')}")

    total = len(all_stocks)
    driven = len(news_driven)
    print(f"\n【扫描结果】共扫描 {total} 只飙升股 → {driven} 只新闻驱动")

    if news_driven:
        cat_zh = {
            "policy": "政策",
            "merger": "并购重组",
            "product": "产品/技术",
            "regulatory": "监管",
            "earnings": "业绩",
            "industry": "行业事件",
            "management": "管理层",
            "contract": "合同/订单",
            "other": "其他",
            "market_movement": "市场波动",
        }
        print(f"\n{'代码':>8s}  {'名称':<8s}  显著性  类别        催化剂")
        print("-" * 64)
        for s in news_driven:
            cat = cat_zh.get(s.catalyst_category, s.catalyst_category)
            print(
                f"{s.symbol:>8s}  {s.name:<8s}  "
                f"{s.significance:>4d}    {cat:<10s}  {s.news_catalyst[:40]}"
            )

    print(f"\n{sep}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="新闻驱动关注度扫描：发现因新闻事件导致关注度飙升的股票",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="扫描飙升榜前 K 只（默认 config llm.attention.max_surge_stocks 或 50）",
    )
    p.add_argument(
        "--model",
        default=None,
        help="Ollama 模型名（默认 config llm.model）",
    )
    p.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="单次请求超时（秒）",
    )
    p.add_argument(
        "--date",
        default=None,
        help="分析日期 YYYY-MM-DD（默认今天）",
    )
    p.add_argument(
        "--analysis-time",
        default=None,
        help="分析基准时刻（ISO 格式，例如 2026-03-30T15:30:00；默认当前时间）",
    )
    p.add_argument(
        "--with-macro",
        action="store_true",
        help="同时做市场宏观利好/利空分析",
    )
    p.add_argument(
        "--with-financial",
        action="store_true",
        help="对新闻驱动标的补充财务指标提取",
    )
    return p


def main() -> None:
    cfg = load_config()
    logs_dir = Path(cfg.get("paths", {}).get("logs_dir", "data/logs"))
    logs_dir.mkdir(parents=True, exist_ok=True)
    setup_app_logging(
        logs_dir,
        name="llm_attention",
        log_format=str((cfg.get("logging") or {}).get("format", "json")),
    )
    parser = _build_parser()
    args = parser.parse_args()
    run_attention_scan(args)


if __name__ == "__main__":
    main()
