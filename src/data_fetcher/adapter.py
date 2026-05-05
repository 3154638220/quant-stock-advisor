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

from src.settings import load_config


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


# ── 回退数据源（Fallback）────────────────────────────────────────────────────
#
# 当主数据源（如 AkShare）网络异常时自动切换到备选源。
# 配置方式:
#
# .. code-block:: yaml
#
#     data_sources:
#       daily: akshare
#       daily_fallback: tushare
#       fund_flow: akshare
#       fund_flow_fallback: tushare


class FallbackDailyDataSource(DailyDataSource):
    """日线数据源回退包装器：主源失败时自动切换备选源。"""

    def __init__(self, primary: DailyDataSource, fallback: DailyDataSource) -> None:
        self._primary = primary
        self._fallback = fallback

    def fetch(self, symbol: str, *, start_date: str, end_date: str) -> FetchResult:
        result = self._primary.fetch(symbol, start_date=start_date, end_date=end_date)
        if not result.ok:
            fb = self._fallback.fetch(symbol, start_date=start_date, end_date=end_date)
            if fb.ok:
                return fb
        return result

    def fetch_many(
        self, symbols: Iterable[str], *, start_date: str, end_date: str
    ) -> list[FetchResult]:
        results = self._primary.fetch_many(symbols, start_date=start_date, end_date=end_date)
        # 对失败的标的逐个尝试备选源
        for i, r in enumerate(results):
            if not r.ok:
                fb = self._fallback.fetch(r.symbol, start_date=start_date, end_date=end_date)
                if fb.ok:
                    results[i] = fb
        return results

    def list_symbols(self) -> pd.DataFrame:
        try:
            return self._primary.list_symbols()
        except Exception:
            return self._fallback.list_symbols()


class FallbackFundFlowDataSource(FundFlowDataSource):
    """资金流数据源回退包装器。"""

    def __init__(self, primary: FundFlowDataSource, fallback: FundFlowDataSource) -> None:
        self._primary = primary
        self._fallback = fallback

    def fetch_symbol(self, symbol: str, *, start_date: str, end_date: str) -> FetchResult:
        result = self._primary.fetch_symbol(symbol, start_date=start_date, end_date=end_date)
        if not result.ok:
            fb = self._fallback.fetch_symbol(symbol, start_date=start_date, end_date=end_date)
            if fb.ok:
                return fb
        return result

    def update_symbols(self, symbols: Iterable[str]) -> int:
        try:
            return self._primary.update_symbols(symbols)
        except Exception:
            return self._fallback.update_symbols(symbols)

    def load_by_date_range(
        self, symbols: Iterable[str], start_date: Union[str, date], end_date: Union[str, date]
    ) -> pd.DataFrame:
        try:
            return self._primary.load_by_date_range(symbols, start_date, end_date)
        except Exception:
            return self._fallback.load_by_date_range(symbols, start_date, end_date)


# ── Tushare 适配器（stub）────────────────────────────────────────────────────
#
# Tushare 需要有效 token 才能使用。当前实现为接口桩，供回退链验证和单元测试。
# 正式接入时将替换为真实 API 调用。参见: https://tushare.pro


class TushareDailyDataSource(DailyDataSource):
    """Tushare 日线数据源（桩实现）。

    需要设置环境变量 ``TUSHARE_TOKEN`` 或配置文件中指定 token。
    当前版本为接口桩，供回退策略验证。
    """

    def __init__(self, *, token: Optional[str] = None) -> None:
        import os
        self._token = token or os.environ.get("TUSHARE_TOKEN", "")
        self._available = bool(self._token)

    def _require_token(self) -> None:
        if not self._available:
            raise RuntimeError(
                "Tushare token 未配置。请设置环境变量 TUSHARE_TOKEN 或通过构造函数传入。"
            )

    def fetch(self, symbol: str, *, start_date: str, end_date: str) -> FetchResult:
        self._require_token()
        return FetchResult(
            symbol=symbol,
            data=pd.DataFrame(),
            source="tushare",
            fetch_date=date.today(),
            error="Tushare 适配器尚未完整实现（当前为桩）。",
        )

    def fetch_many(
        self, symbols: Iterable[str], *, start_date: str, end_date: str
    ) -> list[FetchResult]:
        self._require_token()
        return [
            FetchResult(
                symbol=s,
                data=pd.DataFrame(),
                source="tushare",
                fetch_date=date.today(),
                error="Tushare 适配器尚未完整实现（当前为桩）。",
            )
            for s in symbols
        ]

    def list_symbols(self) -> pd.DataFrame:
        self._require_token()
        return pd.DataFrame(columns=["symbol", "name", "list_date"])


class TushareFundFlowDataSource(FundFlowDataSource):
    """Tushare 资金流数据源（桩实现）。"""

    def __init__(self, *, token: Optional[str] = None) -> None:
        import os
        self._token = token or os.environ.get("TUSHARE_TOKEN", "")
        self._available = bool(self._token)

    def _require_token(self) -> None:
        if not self._available:
            raise RuntimeError("Tushare token 未配置。")

    def fetch_symbol(self, symbol: str, *, start_date: str, end_date: str) -> FetchResult:
        self._require_token()
        return FetchResult(
            symbol=symbol,
            data=pd.DataFrame(),
            source="tushare",
            fetch_date=date.today(),
            error="Tushare 适配器尚未完整实现（当前为桩）。",
        )

    def update_symbols(self, symbols: Iterable[str]) -> int:
        self._require_token()
        return 0

    def load_by_date_range(
        self, symbols: Iterable[str], start_date: Union[str, date], end_date: Union[str, date]
    ) -> pd.DataFrame:
        self._require_token()
        return pd.DataFrame()


# ── 数据源注册与工厂 ────────────────────────────────────────────────────────


#: 已注册的日线数据源实现
_DAILY_REGISTRY: dict[str, type[DailyDataSource]] = {
    "akshare": AkShareDailyDataSource,
    "tushare": TushareDailyDataSource,
}

#: 已注册的基本面数据源实现
_FUNDAMENTAL_REGISTRY: dict[str, type[FundamentalDataSource]] = {
    "akshare": AkShareFundamentalDataSource,
}

#: 已注册的资金流数据源实现
_FUND_FLOW_REGISTRY: dict[str, type[FundFlowDataSource]] = {
    "akshare": AkShareFundFlowDataSource,
    "tushare": TushareFundFlowDataSource,
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

    支持回退链（主源失败时自动切换备选源）：

    .. code-block:: yaml

        data_sources:
          daily: akshare
          daily_fallback: tushare
          fund_flow: akshare
          fund_flow_fallback: tushare
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

    daily = daily_cls()
    fund_flow = fund_flow_cls()

    # 回退链：若配置了 daily_fallback / fund_flow_fallback，包装为 FallbackDataSource
    daily_fallback_name = ds_cfg.get("daily_fallback")
    if daily_fallback_name:
        fb_cls = _DAILY_REGISTRY.get(daily_fallback_name)
        if fb_cls is not None:
            daily = FallbackDailyDataSource(daily, fb_cls())

    fund_flow_fallback_name = ds_cfg.get("fund_flow_fallback")
    if fund_flow_fallback_name:
        fb_cls = _FUND_FLOW_REGISTRY.get(fund_flow_fallback_name)
        if fb_cls is not None:
            fund_flow = FallbackFundFlowDataSource(fund_flow, fb_cls())

    return DataSourceSet(
        daily=daily,
        fundamental=fundamental_cls(),
        fund_flow=fund_flow,
    )
