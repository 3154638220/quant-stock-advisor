"""关注度异常检测：发现因新闻事件驱动（而非市场波动）导致关注度飙升的股票。

工作流：
  1. 从东方财富飙升榜/人气榜获取关注度飙升的股票列表
  2. 对每只飙升股票拉取近期新闻
  3. 用本地 LLM 判断关注度飙升是否由真实新闻事件驱动
  4. 输出经过滤的新闻驱动型高关注度股票
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import pandas as pd

from ..data_fetcher.akshare_resilience import fetch_dataframe_with_cache, resolve_cache_ttl_seconds
from ..settings import load_config
from .client import OllamaClient
from .prompts import ATTENTION_FILTER_SYSTEM, ATTENTION_FILTER_USER

_LOG = logging.getLogger(__name__)


@dataclass
class AttentionStock:
    """关注度飙升股票的分析结果。"""

    symbol: str
    name: str
    rank: int = 0
    rank_change: int = 0
    latest_price: float = 0.0
    price_change_pct: float = 0.0

    is_news_driven: bool = False
    news_catalyst: str = ""
    catalyst_category: str = ""
    significance: int = 0
    news_summary: str = ""
    news_count: int = 0
    key_headlines: list[str] = field(default_factory=list)
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "name": self.name,
            "attention_rank": self.rank,
            "attention_rank_change": self.rank_change,
            "latest_price": self.latest_price,
            "price_change_pct": self.price_change_pct,
            "is_news_driven": self.is_news_driven,
            "news_catalyst": self.news_catalyst,
            "catalyst_category": self.catalyst_category,
            "significance": self.significance,
            "news_summary": self.news_summary,
            "news_count": self.news_count,
            "key_headlines": self.key_headlines,
            "error": self.error,
        }


def _col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """从候选列名中找到 df 中第一个存在的列。"""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _safe_int(val: Any, default: int = 0) -> int:
    try:
        return int(val)
    except (ValueError, TypeError):
        return default


def _safe_float(val: Any, default: float = 0.0) -> float:
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


class AttentionScanner:
    """扫描关注度飙升股票，用 LLM 过滤出新闻事件驱动的标的。"""

    def __init__(
        self,
        client: OllamaClient,
        max_surge_stocks: int = 50,
        max_news_per_stock: int = 15,
        sleep_between_stocks: float = 1.0,
        news_lookback_hours: int = 48,
        analysis_reference_time: datetime | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.client = client
        self.max_surge_stocks = max_surge_stocks
        self.max_news_per_stock = max_news_per_stock
        self.sleep_between_stocks = sleep_between_stocks
        self.news_lookback_hours = max(1, int(news_lookback_hours))
        self.analysis_reference_time = analysis_reference_time or datetime.now()
        self.config = config or load_config()
        ak_cfg = self.config.get("akshare", {}) if isinstance(self.config, dict) else {}
        self._akshare_retries = max(1, int(ak_cfg.get("api_call_retries", 2)))
        self._akshare_timeout = float(ak_cfg.get("request_timeout_sec", 10.0))

    # ------------------------------------------------------------------
    # 关注度数据拉取
    # ------------------------------------------------------------------

    def fetch_hot_surge(self) -> pd.DataFrame:
        """东方财富飙升榜：关注度上升最快的股票。"""
        try:
            import akshare as ak

            df = fetch_dataframe_with_cache(
                [("stock_hot_up_em", lambda: ak.stock_hot_up_em())],
                cache_key="llm_stock_hot_up_em",
                cache_ttl_sec=resolve_cache_ttl_seconds("hot_list", self.config),
                retries=self._akshare_retries,
                timeout_sec=self._akshare_timeout,
                cfg=self.config,
            )
            if df is not None and not df.empty:
                _LOG.info("飙升榜获取成功: %d 只", len(df))
                return df.head(self.max_surge_stocks)
        except Exception as exc:
            _LOG.warning("获取飙升榜失败: %s", exc)
        return pd.DataFrame()

    def fetch_hot_rank(self) -> pd.DataFrame:
        """东方财富人气榜：当前最受关注的股票。"""
        try:
            import akshare as ak

            df = fetch_dataframe_with_cache(
                [("stock_hot_rank_em", lambda: ak.stock_hot_rank_em())],
                cache_key="llm_stock_hot_rank_em",
                cache_ttl_sec=resolve_cache_ttl_seconds("hot_list", self.config),
                retries=self._akshare_retries,
                timeout_sec=self._akshare_timeout,
                cfg=self.config,
            )
            if df is not None and not df.empty:
                _LOG.info("人气榜获取成功: %d 只", len(df))
                return df.head(self.max_surge_stocks)
        except Exception as exc:
            _LOG.warning("获取人气榜失败: %s", exc)
        return pd.DataFrame()

    def _fetch_stock_news(self, symbol: str) -> pd.DataFrame:
        """拉取个股最新新闻。"""
        try:
            import akshare as ak

            df = fetch_dataframe_with_cache(
                [("stock_news_em", lambda: ak.stock_news_em(symbol=symbol))],
                cache_key=f"llm_stock_news_{symbol}",
                cache_ttl_sec=resolve_cache_ttl_seconds("news", self.config),
                retries=self._akshare_retries,
                timeout_sec=self._akshare_timeout,
                cfg=self.config,
                accept_empty=True,
            )
            if df is not None and not df.empty:
                return df
        except Exception as exc:
            _LOG.warning("拉取 %s 新闻失败: %s", symbol, exc)
        return pd.DataFrame()

    # ------------------------------------------------------------------
    # 格式化
    # ------------------------------------------------------------------

    @staticmethod
    def _format_news(news_df: pd.DataFrame) -> tuple[str, list[str], str]:
        """将新闻 DataFrame 转为 LLM 输入文本，同时返回标题列表和时间精度提示。"""
        if news_df.empty:
            return "暂无新闻", [], "无可用新闻时间信息"

        title_col = _col(news_df, ["新闻标题", "title", "标题"])
        content_col = _col(news_df, ["新闻内容", "content", "摘要"])
        date_col = _col(news_df, ["发布时间", "datetime", "日期", "date"])

        lines: list[str] = []
        headlines: list[str] = []
        has_time_precision = False
        for _, row in news_df.iterrows():
            parts: list[str] = []
            if date_col:
                raw_ts = str(row[date_col])
                parts.append(f"[{raw_ts}]")
                if ":" in raw_ts:
                    has_time_precision = True
            if title_col:
                title = str(row[title_col])[:120]
                parts.append(title)
                headlines.append(title)
            if content_col:
                parts.append(str(row[content_col])[:200])
            if parts:
                lines.append(" ".join(parts))

        time_precision = (
            "新闻时间包含小时分钟，时序可部分判定"
            if has_time_precision
            else "新闻时间仅到日期或格式不完整，时序判定存在歧义"
        )
        return ("\n".join(lines) if lines else "暂无新闻"), headlines, time_precision

    def _slice_recent_news(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """按基准时刻截取近 N 小时新闻，并按时间倒序。"""
        if news_df.empty:
            return news_df

        date_col = _col(news_df, ["发布时间", "datetime", "日期", "date"])
        if not date_col:
            return news_df.head(self.max_news_per_stock).copy()

        df = news_df.copy()
        parsed_dt = pd.to_datetime(df[date_col], errors="coerce")
        # 统一转为无时区时间，避免与本地基准时刻比较时报错。
        if hasattr(parsed_dt.dt, "tz") and parsed_dt.dt.tz is not None:
            parsed_dt = parsed_dt.dt.tz_localize(None)
        df = df.assign(__parsed_dt=parsed_dt)

        # 仅保留可解析时间，避免旧新闻噪音误导模型。
        df = df[df["__parsed_dt"].notna()]
        if df.empty:
            return news_df.head(self.max_news_per_stock).copy()

        start_time = self.analysis_reference_time - timedelta(hours=self.news_lookback_hours)
        df = df[(df["__parsed_dt"] <= self.analysis_reference_time) & (df["__parsed_dt"] >= start_time)]
        if df.empty:
            # 回退到最新少量新闻，避免完全无上下文。
            df = news_df.copy()
            df = df.assign(__parsed_dt=pd.to_datetime(df[date_col], errors="coerce"))
            df = df.sort_values(by="__parsed_dt", ascending=False, na_position="last")
            return df.head(min(5, self.max_news_per_stock)).drop(columns="__parsed_dt", errors="ignore")

        df = df.sort_values(by="__parsed_dt", ascending=False, na_position="last")
        return df.head(self.max_news_per_stock).drop(columns="__parsed_dt", errors="ignore")

    @staticmethod
    def _parse_surge_df(df: pd.DataFrame) -> list[dict[str, Any]]:
        """将飙升/人气 DataFrame 解析为标准化字典列表。"""
        if df.empty:
            return []

        sym_col = _col(df, ["代码", "股票代码", "symbol", "code"])
        name_c = _col(df, ["股票名称", "名称", "name"])
        rank_c = _col(df, ["当前排名", "排名", "rank", "序号"])
        chg_c = _col(df, ["排名变动", "排名变化", "rank_change"])
        price_c = _col(df, ["最新价", "latest_price", "收盘价"])
        pct_c = _col(df, ["涨跌幅", "price_change_pct", "涨幅"])

        items: list[dict[str, Any]] = []
        for _, row in df.iterrows():
            sym = str(row[sym_col]).strip() if sym_col else ""
            if not sym:
                continue
            items.append(
                {
                    "symbol": sym,
                    "name": str(row[name_c]).strip() if name_c else sym,
                    "rank": _safe_int(row[rank_c]) if rank_c else 0,
                    "rank_change": _safe_int(row[chg_c]) if chg_c else 0,
                    "price": _safe_float(row[price_c]) if price_c else 0.0,
                    "price_change_pct": _safe_float(row[pct_c]) if pct_c else 0.0,
                }
            )
        return items

    # ------------------------------------------------------------------
    # LLM 分类
    # ------------------------------------------------------------------

    def _classify_attention(
        self,
        symbol: str,
        name: str,
        news_text: str,
        price_change_pct: float,
        news_time_precision: str,
    ) -> dict[str, Any]:
        """用 LLM 判断关注度飙升是否由新闻事件驱动。"""
        messages = [
            {"role": "system", "content": ATTENTION_FILTER_SYSTEM},
            {
                "role": "user",
                "content": ATTENTION_FILTER_USER.format(
                    symbol=symbol,
                    name=name,
                    price_change_pct=f"{price_change_pct:+.2f}%",
                    analysis_reference_time=self.analysis_reference_time.strftime("%Y-%m-%d %H:%M:%S"),
                    lookback_hours=self.news_lookback_hours,
                    news_time_precision=news_time_precision,
                    news_text=news_text,
                ),
            },
        ]
        try:
            return self.client.chat_json(messages)
        except Exception as exc:
            _LOG.error("LLM 分类 %s 失败: %s", symbol, exc)
            return {}

    # ------------------------------------------------------------------
    # 主流程
    # ------------------------------------------------------------------

    def scan(self) -> list[AttentionStock]:
        """
        扫描关注度飙升股票，逐只用 LLM 判断是否为新闻事件驱动。

        返回全部扫描结果（每项 ``is_news_driven`` 标记是否通过筛选）。
        """
        _LOG.info("=== 步骤 1: 获取关注度飙升数据 ===")
        surge_df = self.fetch_hot_surge()
        source = "飙升榜"

        if surge_df.empty:
            _LOG.warning("飙升榜为空，回退到人气榜")
            surge_df = self.fetch_hot_rank()
            source = "人气榜"

        if surge_df.empty:
            _LOG.error("无法获取任何关注度数据，终止扫描")
            return []

        _LOG.info("数据源: %s | 列名: %s", source, list(surge_df.columns))
        stocks = self._parse_surge_df(surge_df)
        _LOG.info("解析出 %d 只关注度飙升股票", len(stocks))

        _LOG.info("=== 步骤 2: 逐只拉取新闻 + LLM 判定 ===")
        results: list[AttentionStock] = []

        for i, s in enumerate(stocks):
            symbol = s["symbol"]
            name = s["name"]
            _LOG.info(
                "扫描 [%d/%d] %s %s (排名=%s 变化=%s)",
                i + 1,
                len(stocks),
                symbol,
                name,
                s.get("rank", "?"),
                s.get("rank_change", "?"),
            )

            item = AttentionStock(
                symbol=symbol,
                name=name,
                rank=s.get("rank", 0),
                rank_change=s.get("rank_change", 0),
                latest_price=s.get("price", 0.0),
                price_change_pct=s.get("price_change_pct", 0.0),
            )

            news_df = self._fetch_stock_news(symbol)
            news_df = self._slice_recent_news(news_df)
            item.news_count = len(news_df)

            if item.news_count == 0:
                item.error = "无新闻数据"
                results.append(item)
                if i < len(stocks) - 1:
                    time.sleep(self.sleep_between_stocks * 0.3)
                continue

            news_text, headlines, news_time_precision = self._format_news(news_df)
            item.key_headlines = headlines[:5]

            parsed = self._classify_attention(
                symbol, name, news_text, item.price_change_pct, news_time_precision
            )

            if parsed:
                item.is_news_driven = bool(parsed.get("is_news_driven", False))
                item.news_catalyst = parsed.get("news_catalyst", "")
                item.catalyst_category = parsed.get("catalyst_category", "")
                item.significance = _safe_int(parsed.get("significance", 0))
                item.news_summary = parsed.get("news_summary", "")

            results.append(item)

            tag = "新闻驱动" if item.is_news_driven else "市场波动"
            _LOG.info(
                "  → %s | %s | 显著性=%d | %s",
                tag,
                item.catalyst_category or "-",
                item.significance,
                item.news_summary or item.news_catalyst or "-",
            )

            if i < len(stocks) - 1:
                time.sleep(self.sleep_between_stocks)

        news_driven = [r for r in results if r.is_news_driven]
        news_driven.sort(key=lambda x: x.significance, reverse=True)

        _LOG.info(
            "=== 扫描完成: 共 %d 只 → 新闻驱动 %d 只 ===",
            len(results),
            len(news_driven),
        )
        return results
