"""AkShare 网络可靠性增强：统一 requests 超时/重试与本地快照回退。"""

from __future__ import annotations

import concurrent.futures
import contextlib
import json
import logging
import re
import signal
import threading
import time
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Callable, Optional, Sequence, TypeVar

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.settings import load_config, project_root

_LOG = logging.getLogger(__name__)
_PATCH_LOCK = threading.Lock()
_PATCHED = False
_TRANSPORT_CONFIG: "AkshareResilienceConfig | None" = None
_ORIG_SESSION_INIT = requests.sessions.Session.__init__
_ORIG_SESSION_REQUEST = requests.sessions.Session.request
_T = TypeVar("_T")


@dataclass(frozen=True)
class AkshareResilienceConfig:
    request_timeout_sec: float
    http_connect_timeout_sec: float
    http_read_timeout_sec: float
    http_transport_retries: int
    http_retry_backoff_sec: float
    http_pool_connections: int
    http_pool_maxsize: int
    api_retry_delay_sec: float
    cache_dir: Path
    stale_cache_on_error: bool


def load_akshare_resilience_config(cfg: Optional[dict] = None) -> AkshareResilienceConfig:
    cfg = cfg or load_config()
    ak_cfg = cfg.get("akshare", {}) if isinstance(cfg, dict) else {}
    request_timeout_sec = float(ak_cfg.get("request_timeout_sec", 10.0))
    connect_timeout_sec = float(
        ak_cfg.get("http_connect_timeout_sec", min(5.0, max(1.0, request_timeout_sec)))
    )
    read_timeout_sec = float(ak_cfg.get("http_read_timeout_sec", max(10.0, request_timeout_sec)))
    cache_dir = Path(ak_cfg.get("cache_dir", "data/cache/akshare"))
    if not cache_dir.is_absolute():
        cache_dir = project_root() / cache_dir
    return AkshareResilienceConfig(
        request_timeout_sec=request_timeout_sec,
        http_connect_timeout_sec=connect_timeout_sec,
        http_read_timeout_sec=read_timeout_sec,
        http_transport_retries=max(0, int(ak_cfg.get("http_transport_retries", 2))),
        http_retry_backoff_sec=float(ak_cfg.get("http_retry_backoff_sec", 0.8)),
        http_pool_connections=max(4, int(ak_cfg.get("http_pool_connections", 64))),
        http_pool_maxsize=max(4, int(ak_cfg.get("http_pool_maxsize", 64))),
        api_retry_delay_sec=float(
            ak_cfg.get("api_retry_delay_sec", ak_cfg.get("retry_delay_sec", 2.0))
        ),
        cache_dir=cache_dir,
        stale_cache_on_error=bool(ak_cfg.get("stale_cache_on_error", True)),
    )


def install_akshare_requests_resilience(cfg: Optional[dict] = None) -> None:
    """为 requests 注入默认超时与 HTTP 层重试；AkShare 大多经此路径发请求。"""
    global _PATCHED, _TRANSPORT_CONFIG
    _TRANSPORT_CONFIG = load_akshare_resilience_config(cfg)
    if _PATCHED:
        return

    with _PATCH_LOCK:
        if _PATCHED:
            return

        def _patched_init(self, *args, **kwargs):
            _ORIG_SESSION_INIT(self, *args, **kwargs)
            conf = _TRANSPORT_CONFIG or load_akshare_resilience_config()
            retry = Retry(
                total=conf.http_transport_retries,
                connect=conf.http_transport_retries,
                read=conf.http_transport_retries,
                status=conf.http_transport_retries,
                backoff_factor=conf.http_retry_backoff_sec,
                allowed_methods=None,
                status_forcelist=(408, 425, 429, 500, 502, 503, 504),
                raise_on_status=False,
                respect_retry_after_header=False,
            )
            adapter = HTTPAdapter(
                max_retries=retry,
                pool_connections=conf.http_pool_connections,
                pool_maxsize=conf.http_pool_maxsize,
            )
            self.mount("http://", adapter)
            self.mount("https://", adapter)

        def _patched_request(self, method, url, *args, **kwargs):
            conf = _TRANSPORT_CONFIG or load_akshare_resilience_config()
            timeout = kwargs.get("timeout")
            if timeout is None:
                kwargs["timeout"] = (
                    conf.http_connect_timeout_sec,
                    conf.http_read_timeout_sec,
                )
            elif isinstance(timeout, (int, float)):
                req_timeout = float(timeout)
                kwargs["timeout"] = (
                    min(conf.http_connect_timeout_sec, req_timeout),
                    req_timeout,
                )
            return _ORIG_SESSION_REQUEST(self, method, url, *args, **kwargs)

        requests.sessions.Session.__init__ = _patched_init  # type: ignore[method-assign]
        requests.sessions.Session.request = _patched_request  # type: ignore[method-assign]
        _PATCHED = True


@contextlib.contextmanager
def time_limit(seconds: float, *, label: str = "AkShare 调用"):
    if (
        seconds <= 0
        or not hasattr(signal, "SIGALRM")
        or threading.current_thread() is not threading.main_thread()
    ):
        yield
        return

    def _handle_timeout(signum, frame):  # type: ignore[unused-arg]
        raise TimeoutError(f"{label} 超时（>{seconds:.1f}s）")

    previous = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _handle_timeout)
    signal.setitimer(signal.ITIMER_REAL, max(1e-3, seconds))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous)


def call_with_timeout(
    func: Callable[[], _T],
    *,
    timeout_sec: float,
    label: str = "AkShare 调用",
) -> _T:
    """
    以跨线程可用的方式执行超时控制。

    优先使用 ``SIGALRM``（仅主线程可用），否则回退到
    ``ThreadPoolExecutor + Future.result(timeout=...)``。
    """
    sec = float(timeout_sec)
    if sec <= 0:
        return func()

    can_use_signal = (
        hasattr(signal, "SIGALRM")
        and threading.current_thread() is threading.main_thread()
    )
    if can_use_signal:
        with time_limit(sec, label=label):
            return func()

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func)
        try:
            return future.result(timeout=sec)
        except concurrent.futures.TimeoutError as exc:
            future.cancel()
            raise TimeoutError(f"{label} 超时（>{sec:.1f}s）") from exc


def fetch_dataframe_with_cache(
    fetchers: Sequence[tuple[str, Callable[[], pd.DataFrame]]],
    *,
    cache_key: str,
    cache_ttl_sec: Optional[float],
    retries: int,
    timeout_sec: Optional[float] = None,
    retry_delay_sec: Optional[float] = None,
    cfg: Optional[dict] = None,
    accept_empty: bool = False,
) -> pd.DataFrame:
    """顺序尝试多个 DataFrame 拉取源；全部失败时回退本地快照。"""
    conf = load_akshare_resilience_config(cfg)
    install_akshare_requests_resilience(cfg)
    effective_timeout = conf.request_timeout_sec if timeout_sec is None else float(timeout_sec)
    effective_retry_delay = (
        conf.api_retry_delay_sec if retry_delay_sec is None else float(retry_delay_sec)
    )
    retries = max(1, int(retries))

    cache_path = _cache_path(cache_key, conf)
    cached_df, cache_age_sec, cache_source = _load_cached_dataframe(cache_path)
    last_exc: BaseException | None = None

    for source, fetcher in fetchers:
        for attempt in range(retries):
            try:
                with time_limit(effective_timeout, label=f"{source}"):
                    df = fetcher()
                if df is None:
                    df = pd.DataFrame()
                if not isinstance(df, pd.DataFrame):
                    raise TypeError(f"{source} 未返回 DataFrame")
                if not df.empty or accept_empty:
                    _save_cached_dataframe(cache_path, df, source=source)
                    return df
                _LOG.warning(
                    "AkShare 来源 %s 第 %s/%s 次返回空表。",
                    source,
                    attempt + 1,
                    retries,
                )
            except Exception as exc:
                last_exc = exc
                _LOG.warning(
                    "AkShare 来源 %s 第 %s/%s 次失败: %s: %s",
                    source,
                    attempt + 1,
                    retries,
                    type(exc).__name__,
                    exc,
                )
            if attempt < retries - 1:
                time.sleep(max(0.0, effective_retry_delay) * (attempt + 1))

    if (
        conf.stale_cache_on_error
        and cached_df is not None
        and (cache_ttl_sec is None or cache_age_sec is not None and cache_age_sec <= cache_ttl_sec)
    ):
        _LOG.warning(
            "AkShare 实时拉取全部失败，回退本地快照 %s | 来源=%s | 年龄=%.0fs",
            cache_path,
            cache_source or "-",
            cache_age_sec or -1.0,
        )
        return cached_df

    if last_exc is not None:
        raise RuntimeError("AkShare fetch exhausted retries") from last_exc
    return pd.DataFrame()


def resolve_cache_ttl_seconds(kind: str, cfg: Optional[dict] = None) -> float:
    conf = load_akshare_resilience_config(cfg)
    ak_cfg = (cfg or load_config()).get("akshare", {})
    ttl_map = {
        "hot_list": float(ak_cfg.get("hot_list_cache_ttl_sec", 900)),
        "news": float(ak_cfg.get("news_cache_ttl_sec", 1800)),
        "financial": float(ak_cfg.get("financial_cache_ttl_sec", 86400)),
    }
    return ttl_map.get(kind, max(300.0, conf.request_timeout_sec * 30))


def _cache_path(cache_key: str, conf: AkshareResilienceConfig) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", cache_key).strip("._") or "cache"
    return conf.cache_dir / f"{safe}.json"


def _save_cached_dataframe(path: Path, df: pd.DataFrame, *, source: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "cached_at": time.time(),
        "source": source,
        "data": df.to_json(orient="table", date_format="iso", force_ascii=False),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _load_cached_dataframe(path: Path) -> tuple[pd.DataFrame | None, float | None, str | None]:
    if not path.exists():
        return None, None, None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        raw = payload.get("data", "")
        if not raw:
            return None, None, None
        df = pd.read_json(StringIO(raw), orient="table")
        cached_at = payload.get("cached_at")
        age_sec = max(0.0, time.time() - float(cached_at)) if cached_at is not None else None
        return df, age_sec, payload.get("source")
    except Exception as exc:
        _LOG.warning("AkShare 本地快照读取失败 %s: %s", path, exc)
        return None, None, None
