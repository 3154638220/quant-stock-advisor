"""新闻利好/利空分析：通过 AkShare 拉取个股与市场快讯，用本地 LLM 提炼。"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import pandas as pd

from ..data_fetcher.akshare_resilience import fetch_dataframe_with_cache, resolve_cache_ttl_seconds
from ..settings import load_config
from .client import OllamaClient
from .prompts import (
    MARKET_MACRO_SYSTEM,
    MARKET_MACRO_USER,
    NEWS_SENTIMENT_SYSTEM,
    NEWS_SENTIMENT_USER,
)

_LOG = logging.getLogger(__name__)


@dataclass
class StockNewsSentiment:
    """单只股票的新闻情绪分析结果。"""

    symbol: str
    name: str
    analysis_date: str
    sentiment: str = "neutral"          # bullish / bearish / neutral
    score: int = 0                       # -5 到 +5
    bullish_points: list[str] = field(default_factory=list)
    bearish_points: list[str] = field(default_factory=list)
    summary: str = ""
    news_count: int = 0
    error: str = ""

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "name": self.name,
            "analysis_date": self.analysis_date,
            "llm_sentiment": self.sentiment,
            "llm_news_score": self.score,
            "llm_bullish": " | ".join(self.bullish_points),
            "llm_bearish": " | ".join(self.bearish_points),
            "llm_news_summary": self.summary,
            "llm_news_count": self.news_count,
            "llm_error": self.error,
        }


@dataclass
class MarketMacroSentiment:
    """当日市场整体情绪结果。"""

    analysis_date: str
    market_sentiment: str = "neutral"
    market_score: int = 0
    top_bullish: list[str] = field(default_factory=list)
    top_bearish: list[str] = field(default_factory=list)
    hot_sectors: list[str] = field(default_factory=list)
    risk_sectors: list[str] = field(default_factory=list)
    summary: str = ""
    error: str = ""

    def to_dict(self) -> dict:
        return {
            "analysis_date": self.analysis_date,
            "market_sentiment": self.market_sentiment,
            "market_score": self.market_score,
            "top_bullish": self.top_bullish,
            "top_bearish": self.top_bearish,
            "hot_sectors": self.hot_sectors,
            "risk_sectors": self.risk_sectors,
            "summary": self.summary,
        }


class NewsAnalyzer:
    """利用 AkShare 拉取新闻 + 本地 LLM 分析利好/利空。"""

    def __init__(
        self,
        client: OllamaClient,
        max_news_per_stock: int = 10,
        max_market_news: int = 20,
        sleep_between_stocks: float = 1.0,
        config: dict | None = None,
    ) -> None:
        self.client = client
        self.max_news_per_stock = max_news_per_stock
        self.max_market_news = max_market_news
        self.sleep_between_stocks = sleep_between_stocks
        self.config = config or load_config()
        ak_cfg = self.config.get("akshare", {}) if isinstance(self.config, dict) else {}
        self._akshare_retries = max(1, int(ak_cfg.get("api_call_retries", 2)))
        self._akshare_timeout = float(ak_cfg.get("request_timeout_sec", 10.0))

    # ------------------------------------------------------------------
    # 数据拉取
    # ------------------------------------------------------------------

    def _fetch_stock_news(self, symbol: str) -> pd.DataFrame:
        """从东方财富拉取个股最新新闻，返回 DataFrame；失败时返回空表。"""
        try:
            import akshare as ak

            df = fetch_dataframe_with_cache(
                [("stock_news_em", lambda: ak.stock_news_em(symbol=symbol))],
                cache_key=f"news_analyzer_stock_news_{symbol}",
                cache_ttl_sec=resolve_cache_ttl_seconds("news", self.config),
                retries=self._akshare_retries,
                timeout_sec=self._akshare_timeout,
                cfg=self.config,
                accept_empty=True,
            )
            return df.head(self.max_news_per_stock) if not df.empty else df
        except Exception as exc:
            _LOG.warning("拉取 %s 个股新闻失败: %s", symbol, exc)
            return pd.DataFrame()

    def _fetch_market_news(self, trade_date: str) -> pd.DataFrame:
        """拉取当日百度财经快讯，返回 DataFrame；失败时返回空表。"""
        try:
            import akshare as ak
            # trade_date 格式 YYYY-MM-DD → YYYYMMDD
            date_compact = trade_date.replace("-", "")
            df = fetch_dataframe_with_cache(
                [("news_economic_baidu", lambda: ak.news_economic_baidu(date=date_compact))],
                cache_key=f"news_analyzer_market_news_{date_compact}",
                cache_ttl_sec=resolve_cache_ttl_seconds("news", self.config),
                retries=self._akshare_retries,
                timeout_sec=self._akshare_timeout,
                cfg=self.config,
                accept_empty=True,
            )
            return df.head(self.max_market_news) if not df.empty else df
        except Exception as exc:
            _LOG.warning("拉取市场快讯失败 (%s): %s", trade_date, exc)
            return pd.DataFrame()

    def _fetch_stock_notices(self, symbol: str, trade_date: str) -> pd.DataFrame:
        """拉取个股公告摘要，补充新闻来源。"""
        try:
            import akshare as ak
            date_compact = trade_date.replace("-", "")
            df = fetch_dataframe_with_cache(
                [(
                    "stock_notice_report",
                    lambda: ak.stock_notice_report(symbol=symbol, date=date_compact),
                )],
                cache_key=f"news_analyzer_notice_{symbol}_{date_compact}",
                cache_ttl_sec=resolve_cache_ttl_seconds("news", self.config),
                retries=self._akshare_retries,
                timeout_sec=self._akshare_timeout,
                cfg=self.config,
                accept_empty=True,
            )
            return df.head(5) if not df.empty else df
        except Exception as exc:
            _LOG.debug("拉取 %s 公告失败 (可能当日无公告): %s", symbol, exc)
            return pd.DataFrame()

    # ------------------------------------------------------------------
    # 文本格式化
    # ------------------------------------------------------------------

    @staticmethod
    def _format_stock_news(news_df: pd.DataFrame, notice_df: pd.DataFrame) -> str:
        """将新闻 DataFrame 转为 LLM 输入文本。"""
        lines: list[str] = []
        # 新闻
        title_col = next((c for c in ["新闻标题", "title", "标题"] if c in news_df.columns), None)
        content_col = next((c for c in ["新闻内容", "content", "摘要"] if c in news_df.columns), None)
        date_col = next((c for c in ["发布时间", "datetime", "日期", "date"] if c in news_df.columns), None)

        for _, row in news_df.iterrows():
            parts = []
            if date_col:
                parts.append(f"[{row[date_col]}]")
            if title_col:
                parts.append(str(row[title_col])[:100])
            if content_col:
                parts.append(str(row[content_col])[:150])
            if parts:
                lines.append(" ".join(parts))

        # 公告
        if not notice_df.empty:
            ann_col = next((c for c in ["公告标题", "title", "标题"] if c in notice_df.columns), None)
            if ann_col:
                lines.append("\n【公告】")
                for _, row in notice_df.iterrows():
                    lines.append(f"  - {str(row[ann_col])[:100]}")

        return "\n".join(lines) if lines else "暂无新闻数据"

    @staticmethod
    def _format_market_news(news_df: pd.DataFrame) -> str:
        """将市场快讯 DataFrame 转为 LLM 输入文本。"""
        if news_df.empty:
            return "暂无市场快讯"
        title_col = next((c for c in ["新闻标题", "title", "标题", "content"] if c in news_df.columns), None)
        date_col = next((c for c in ["发布时间", "datetime", "日期", "date"] if c in news_df.columns), None)
        lines = []
        for _, row in news_df.iterrows():
            parts = []
            if date_col:
                parts.append(f"[{row[date_col]}]")
            if title_col:
                parts.append(str(row[title_col])[:120])
            if parts:
                lines.append(" ".join(parts))
        return "\n".join(lines) if lines else "暂无市场快讯"

    # ------------------------------------------------------------------
    # 分析接口
    # ------------------------------------------------------------------

    def analyze_stock(
        self,
        symbol: str,
        name: str,
        trade_date: str,
    ) -> StockNewsSentiment:
        """分析单只股票当日新闻情绪。"""
        result = StockNewsSentiment(symbol=symbol, name=name, analysis_date=trade_date)

        news_df = self._fetch_stock_news(symbol)
        notice_df = self._fetch_stock_notices(symbol, trade_date)
        result.news_count = len(news_df) + len(notice_df)

        if result.news_count == 0:
            result.summary = "无新闻数据"
            return result

        news_text = self._format_stock_news(news_df, notice_df)
        messages = [
            {"role": "system", "content": NEWS_SENTIMENT_SYSTEM},
            {
                "role": "user",
                "content": NEWS_SENTIMENT_USER.format(
                    symbol=symbol,
                    name=name,
                    date=trade_date,
                    news_text=news_text,
                ),
            },
        ]

        try:
            parsed = self.client.chat_json(messages)
            result.sentiment = parsed.get("sentiment", "neutral")
            result.score = int(parsed.get("score", 0))
            result.bullish_points = parsed.get("bullish_points", [])
            result.bearish_points = parsed.get("bearish_points", [])
            result.summary = parsed.get("summary", "")
        except Exception as exc:
            result.error = str(exc)
            _LOG.error("分析 %s 新闻情绪失败: %s", symbol, exc)

        return result

    def analyze_market(self, trade_date: str) -> MarketMacroSentiment:
        """分析当日市场整体宏观情绪。"""
        result = MarketMacroSentiment(analysis_date=trade_date)
        news_df = self._fetch_market_news(trade_date)
        news_text = self._format_market_news(news_df)

        messages = [
            {"role": "system", "content": MARKET_MACRO_SYSTEM},
            {
                "role": "user",
                "content": MARKET_MACRO_USER.format(date=trade_date, news_text=news_text),
            },
        ]

        try:
            parsed = self.client.chat_json(messages)
            result.market_sentiment = parsed.get("market_sentiment", "neutral")
            result.market_score = int(parsed.get("market_score", 0))
            result.top_bullish = parsed.get("top_bullish", [])
            result.top_bearish = parsed.get("top_bearish", [])
            result.hot_sectors = parsed.get("hot_sectors", [])
            result.risk_sectors = parsed.get("risk_sectors", [])
            result.summary = parsed.get("summary", "")
        except Exception as exc:
            result.error = str(exc)
            _LOG.error("分析市场宏观情绪失败: %s", exc)

        return result

    def batch_analyze_stocks(
        self,
        symbols: list[tuple[str, str]],
        trade_date: str,
    ) -> list[StockNewsSentiment]:
        """批量分析股票列表，symbols 为 [(code, name), ...]。"""
        results: list[StockNewsSentiment] = []
        total = len(symbols)
        for i, (symbol, name) in enumerate(symbols):
            _LOG.info("LLM 新闻分析 [%d/%d] %s %s", i + 1, total, symbol, name)
            r = self.analyze_stock(symbol, name, trade_date)
            results.append(r)
            if i < total - 1:
                time.sleep(self.sleep_between_stocks)
        return results
