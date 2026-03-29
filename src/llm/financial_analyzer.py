"""财务报表关键数据提取：通过 AkShare 拉取财务摘要，用本地 LLM 解读。"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from .client import OllamaClient
from .prompts import FINANCIAL_EXTRACT_SYSTEM, FINANCIAL_EXTRACT_USER

_LOG = logging.getLogger(__name__)


@dataclass
class FinancialSummary:
    """单只股票财务分析结果。"""

    symbol: str
    name: str
    period: str = ""
    revenue_growth_yoy: Optional[float] = None
    profit_growth_yoy: Optional[float] = None
    gross_margin: Optional[float] = None
    net_margin: Optional[float] = None
    roe: Optional[float] = None
    debt_ratio: Optional[float] = None
    pe_ttm: Optional[float] = None
    highlights: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    rating: str = "hold"
    summary: str = ""
    error: str = ""

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "name": self.name,
            "fin_period": self.period,
            "fin_revenue_growth_yoy": self.revenue_growth_yoy,
            "fin_profit_growth_yoy": self.profit_growth_yoy,
            "fin_gross_margin": self.gross_margin,
            "fin_net_margin": self.net_margin,
            "fin_roe": self.roe,
            "fin_debt_ratio": self.debt_ratio,
            "fin_pe_ttm": self.pe_ttm,
            "fin_highlights": " | ".join(self.highlights),
            "fin_risks": " | ".join(self.risks),
            "fin_rating": self.rating,
            "fin_summary": self.summary,
            "fin_error": self.error,
        }


class FinancialAnalyzer:
    """拉取财务数据并用 LLM 提取关键指标。"""

    def __init__(
        self,
        client: OllamaClient,
        sleep_between_stocks: float = 1.5,
    ) -> None:
        self.client = client
        self.sleep_between_stocks = sleep_between_stocks

    # ------------------------------------------------------------------
    # 数据拉取
    # ------------------------------------------------------------------

    def _fetch_financial_abstract(self, symbol: str) -> pd.DataFrame:
        """拉取财务摘要（新浪接口），包含最近几个报告期的核心指标。"""
        try:
            import akshare as ak
            df = ak.stock_financial_abstract(symbol=symbol)
            return df
        except Exception as exc:
            _LOG.warning("拉取 %s 财务摘要失败: %s", symbol, exc)
            return pd.DataFrame()

    def _fetch_financial_indicator(self, symbol: str) -> pd.DataFrame:
        """拉取财务分析指标（含 ROE/毛利率/净利率等）。"""
        try:
            import akshare as ak
            df = ak.stock_financial_analysis_indicator(symbol=symbol, start_year="2022")
            return df.head(8) if not df.empty else df
        except Exception as exc:
            _LOG.warning("拉取 %s 财务指标失败: %s", symbol, exc)
            return pd.DataFrame()

    def _fetch_profit_sheet(self, symbol: str) -> pd.DataFrame:
        """拉取利润表（东方财富）。"""
        try:
            import akshare as ak
            df = ak.stock_profit_sheet_by_report_em(symbol=symbol)
            return df.head(4) if not df.empty else df
        except Exception as exc:
            _LOG.debug("拉取 %s 利润表失败: %s", symbol, exc)
            return pd.DataFrame()

    # ------------------------------------------------------------------
    # 文本格式化
    # ------------------------------------------------------------------

    @staticmethod
    def _df_to_text(df: pd.DataFrame, max_rows: int = 6, label: str = "") -> str:
        """将 DataFrame 转为紧凑文本供 LLM 输入。"""
        if df.empty:
            return ""
        lines = []
        if label:
            lines.append(f"【{label}】")
        # 转置宽表为更易读的格式
        subset = df.head(max_rows)
        for col in subset.columns:
            vals = subset[col].tolist()
            vals_str = " | ".join(str(v)[:20] for v in vals)
            lines.append(f"  {col}: {vals_str}")
        return "\n".join(lines)

    def _build_financial_text(
        self,
        abstract_df: pd.DataFrame,
        indicator_df: pd.DataFrame,
        profit_df: pd.DataFrame,
    ) -> tuple[str, str]:
        """组合多源财务数据为文本，同时提取最新报告期。"""
        parts = []
        period = ""

        if not abstract_df.empty:
            # 尝试找报告期列
            date_col = next(
                (c for c in ["报告期", "date", "period", "报表日期"] if c in abstract_df.columns), None
            )
            if date_col:
                period = str(abstract_df.iloc[0][date_col]) if not abstract_df.empty else ""
            parts.append(self._df_to_text(abstract_df, max_rows=4, label="财务摘要"))

        if not indicator_df.empty:
            parts.append(self._df_to_text(indicator_df, max_rows=4, label="财务指标"))

        if not profit_df.empty:
            parts.append(self._df_to_text(profit_df, max_rows=4, label="利润表"))

        text = "\n\n".join(p for p in parts if p)
        if not text:
            text = "暂无财务数据"
        return text, period

    # ------------------------------------------------------------------
    # 分析接口
    # ------------------------------------------------------------------

    def analyze_stock(self, symbol: str, name: str) -> FinancialSummary:
        """对单只股票做财务数据 LLM 分析。"""
        result = FinancialSummary(symbol=symbol, name=name)

        abstract_df = self._fetch_financial_abstract(symbol)
        indicator_df = self._fetch_financial_indicator(symbol)
        profit_df = self._fetch_profit_sheet(symbol)

        financial_text, period = self._build_financial_text(abstract_df, indicator_df, profit_df)
        result.period = period

        if financial_text == "暂无财务数据":
            result.summary = "无财务数据"
            return result

        messages = [
            {"role": "system", "content": FINANCIAL_EXTRACT_SYSTEM},
            {
                "role": "user",
                "content": FINANCIAL_EXTRACT_USER.format(
                    symbol=symbol,
                    name=name,
                    period=period or "最新",
                    financial_text=financial_text,
                ),
            },
        ]

        try:
            parsed = self.client.chat_json(messages)
            result.revenue_growth_yoy = _to_float(parsed.get("revenue_growth_yoy"))
            result.profit_growth_yoy = _to_float(parsed.get("profit_growth_yoy"))
            result.gross_margin = _to_float(parsed.get("gross_margin"))
            result.net_margin = _to_float(parsed.get("net_margin"))
            result.roe = _to_float(parsed.get("roe"))
            result.debt_ratio = _to_float(parsed.get("debt_ratio"))
            result.pe_ttm = _to_float(parsed.get("pe_ttm"))
            result.highlights = parsed.get("highlights", [])
            result.risks = parsed.get("risks", [])
            result.rating = parsed.get("rating", "hold")
            result.summary = parsed.get("summary", "")
        except Exception as exc:
            result.error = str(exc)
            _LOG.error("分析 %s 财务数据失败: %s", symbol, exc)

        return result

    def batch_analyze_stocks(
        self,
        symbols: list[tuple[str, str]],
    ) -> list[FinancialSummary]:
        """批量分析，symbols 为 [(code, name), ...]。"""
        results: list[FinancialSummary] = []
        total = len(symbols)
        for i, (symbol, name) in enumerate(symbols):
            _LOG.info("LLM 财务分析 [%d/%d] %s %s", i + 1, total, symbol, name)
            r = self.analyze_stock(symbol, name)
            results.append(r)
            if i < total - 1:
                time.sleep(self.sleep_between_stocks)
        return results


def _to_float(val) -> Optional[float]:
    """安全转 float，None/null 返回 None。"""
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None
