"""数据源适配器抽象层。

将 AkShare 封装为可替换的实现，预留 TuShare/Wind 适配器接口。
通过配置选择数据源：

.. code-block:: yaml

    data_sources:
      daily: akshare          # akshare | tushare | wind
      fundamental: akshare
      fund_flow: akshare
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable, Optional, Union

import pandas as pd

from ..settings import load_config, project_root


@dataclass(frozen=True)
class FetchResult:
    """单次拉取结果：成功时 data 非空，失败时 error 记录原因。"""

    symbol: str
    data: pd.DataFrame
    source: str
    fetch_date: date
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None


# ── 日线数据源 ──────────────────────────────────────────────────────────────


class DailyDataSource(ABC):
    """日线 OHLCV 数据源抽象。"""

    @abstractmethod
    def fetch(self, symbol: str, *, start_date: str, end_date: str) -> FetchResult:
        """拉取单标的日线数据。"""
        ...

    @abstractmethod
    def fetch_many(
        self, symbols: Iterable[str], *, start_date: str, end_date: str
    ) -> list[FetchResult]:
        """批量拉取日线数据。"""
        ...

    @abstractmethod
    def list_symbols(self) -> pd.DataFrame:
        """返回可交易标的列表（symbol, name, list_date）。"""
        ...


class AkShareDailyDataSource(DailyDataSource):
    """AkShare 日线数据源实现。"""

    def __init__(self, *, adjust: str = "qfq", timeout_sec: float = 10.0) -> None:
        self._adjust = adjust
        self._timeout_sec = timeout_sec

    def fetch(self, symbol: str, *, start_date: str, end_date: str) -> FetchResult:
        from .akshare_client import fetch_a_share_daily

        try:
            df = fetch_a_share_daily(
                symbol,
                start_date=start_date,
                end_date=end_date,
                adjust=self._adjust,
                request_timeout_sec=self._timeout_sec,
            )
            return FetchResult(
                symbol=symbol,
                data=df,
                source="akshare",
                fetch_date=date.today(),
            )
        except Exception as exc:
            return FetchResult(
                symbol=symbol,
                data=pd.DataFrame(),
                source="akshare",
                fetch_date=date.today(),
                error=str(exc),
            )

    def fetch_many(
        self, symbols: Iterable[str], *, start_date: str, end_date: str
    ) -> list[FetchResult]:
        from .akshare_client import fetch_many_daily

        symbols = list(symbols)
        dfs = fetch_many_daily(symbols, start_date=start_date, end_date=end_date)
        results: list[FetchResult] = []
        for sym, df in zip(symbols, dfs):
            error = None
            if df is None or df.empty:
                error = "empty or fetch failed"
                df = pd.DataFrame()
            results.append(
                FetchResult(
                    symbol=sym,
                    data=df,
                    source="akshare",
                    fetch_date=date.today(),
                    error=error,
                )
            )
        return results

    def list_symbols(self) -> pd.DataFrame:
        from .akshare_client import list_default_universe_symbols

        return list_default_universe_symbols()


# ── 基本面数据源 ────────────────────────────────────────────────────────────


class FundamentalDataSource(ABC):
    """基本面数据源抽象（PE/PB/ROE/毛利率等）。"""

    @abstractmethod
    def fetch_symbol(self, symbol: str) -> FetchResult:
        """拉取单标的基本面数据。"""
        ...

    @abstractmethod
    def update_symbols(self, symbols: Iterable[str]) -> int:
        """批量更新标的的基本面数据，返回更新行数。"""
        ...

    @abstractmethod
    def load_point_in_time(
        self, symbols: Iterable[str], asof_date: Union[str, date]
    ) -> pd.DataFrame:
        """按 PIT 口径加载指定日期的基本面截面。"""
        ...


class AkShareFundamentalDataSource(FundamentalDataSource):
    """AkShare/东方财富 基本面数据源实现。"""

    def __init__(self, *, config_path: Optional[Union[str, Path]] = None) -> None:
        self._config_path = Path(config_path) if config_path else None

    def _client(self):
        from .fundamental_client import FundamentalClient

        return FundamentalClient(config_path=self._config_path)

    def fetch_symbol(self, symbol: str) -> FetchResult:
        client = self._client()
        try:
            with client:
                df = client.fetch_symbol_fundamentals(symbol)
            return FetchResult(
                symbol=symbol,
                data=df,
                source="akshare",
                fetch_date=date.today(),
            )
        except Exception as exc:
            return FetchResult(
                symbol=symbol,
                data=pd.DataFrame(),
                source="akshare",
                fetch_date=date.today(),
                error=str(exc),
            )

    def update_symbols(self, symbols: Iterable[str]) -> int:
        client = self._client()
        with client:
            return client.update_symbols(symbols)

    def load_point_in_time(
        self, symbols: Iterable[str], asof_date: Union[str, date]
    ) -> pd.DataFrame:
        client = self._client()
        with client:
            return client.load_point_in_time(symbols, asof_date)


# ── 资金流数据源 ────────────────────────────────────────────────────────────


class FundFlowDataSource(ABC):
    """个股资金流数据源抽象（主力净流入/超大单/大单/中单/小单）。"""

    @abstractmethod
    def fetch_symbol(self, symbol: str, *, start_date: str, end_date: str) -> FetchResult:
        ...

    @abstractmethod
    def update_symbols(self, symbols: Iterable[str]) -> int:
        ...

    @abstractmethod
    def load_by_date_range(
        self, symbols: Iterable[str], start_date: Union[str, date], end_date: Union[str, date]
    ) -> pd.DataFrame:
        ...


class AkShareFundFlowDataSource(FundFlowDataSource):
    """AkShare/东方财富 资金流数据源实现。"""

    def __init__(self, *, config_path: Optional[Union[str, Path]] = None) -> None:
        self._config_path = Path(config_path) if config_path else None

    def _client(self):
        from .fund_flow_client import FundFlowClient

        return FundFlowClient(config_path=self._config_path)

    def fetch_symbol(self, symbol: str, *, start_date: str, end_date: str) -> FetchResult:
        client = self._client()
        try:
            with client:
                df = client.fetch_symbol_fund_flow(symbol, start_date=start_date, end_date=end_date)
            return FetchResult(
                symbol=symbol,
                data=df,
                source="akshare",
                fetch_date=date.today(),
            )
        except Exception as exc:
            return FetchResult(
                symbol=symbol,
                data=pd.DataFrame(),
                source="akshare",
                fetch_date=date.today(),
                error=str(exc),
            )

    def update_symbols(self, symbols: Iterable[str]) -> int:
        client = self._client()
        with client:
            return client.update_symbols(symbols)

    def load_by_date_range(
        self, symbols: Iterable[str], start_date: Union[str, date], end_date: Union[str, date]
    ) -> pd.DataFrame:
        client = self._client()
        with client:
            return client.load_by_date_range(symbols, start_date, end_date)


# ── 数据源注册与工厂 ────────────────────────────────────────────────────────


#: 已注册的日线数据源实现
_DAILY_REGISTRY: dict[str, type[DailyDataSource]] = {
    "akshare": AkShareDailyDataSource,
}

#: 已注册的基本面数据源实现
_FUNDAMENTAL_REGISTRY: dict[str, type[FundamentalDataSource]] = {
    "akshare": AkShareFundamentalDataSource,
}

#: 已注册的资金流数据源实现
_FUND_FLOW_REGISTRY: dict[str, type[FundFlowDataSource]] = {
    "akshare": AkShareFundFlowDataSource,
}


def register_daily_source(name: str, cls: type[DailyDataSource]) -> None:
    """注册自定义日线数据源实现。"""
    _DAILY_REGISTRY[name] = cls


def register_fundamental_source(name: str, cls: type[FundamentalDataSource]) -> None:
    """注册自定义基本面数据源实现。"""
    _FUNDAMENTAL_REGISTRY[name] = cls


def register_fund_flow_source(name: str, cls: type[FundFlowDataSource]) -> None:
    """注册自定义资金流数据源实现。"""
    _FUND_FLOW_REGISTRY[name] = cls


@dataclass
class DataSourceSet:
    """按类型聚合的数据源集合。"""

    daily: DailyDataSource
    fundamental: FundamentalDataSource
    fund_flow: FundFlowDataSource


def resolve_sources(config_path: Optional[Union[str, Path]] = None) -> DataSourceSet:
    """从配置解析并构造数据源集合。

    默认全部使用 AkShare。可通过 ``data_sources`` 配置节切换：

    .. code-block:: yaml

        data_sources:
          daily: akshare
          fundamental: akshare
          fund_flow: akshare
    """
    cfg = load_config(Path(config_path)) if config_path else load_config()
    ds_cfg = cfg.get("data_sources", {})

    daily_name = ds_cfg.get("daily", "akshare")
    fundamental_name = ds_cfg.get("fundamental", "akshare")
    fund_flow_name = ds_cfg.get("fund_flow", "akshare")

    daily_cls = _DAILY_REGISTRY.get(daily_name)
    if daily_cls is None:
        raise ValueError(
            f"未知日线数据源 {daily_name!r}，可用: {list(_DAILY_REGISTRY)}"
        )

    fundamental_cls = _FUNDAMENTAL_REGISTRY.get(fundamental_name)
    if fundamental_cls is None:
        raise ValueError(
            f"未知基本面数据源 {fundamental_name!r}，可用: {list(_FUNDAMENTAL_REGISTRY)}"
        )

    fund_flow_cls = _FUND_FLOW_REGISTRY.get(fund_flow_name)
    if fund_flow_cls is None:
        raise ValueError(
            f"未知资金流数据源 {fund_flow_name!r}，可用: {list(_FUND_FLOW_REGISTRY)}"
        )

    return DataSourceSet(
        daily=daily_cls(),
        fundamental=fundamental_cls(),
        fund_flow=fund_flow_cls(),
    )
