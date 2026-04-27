"""轻量研究产物的统一命名与身份字段。"""

from __future__ import annotations

from copy import deepcopy
import re
from typing import Any


CANONICAL_RESEARCH_CONFIGS: dict[str, dict[str, Any]] = {
    "v3_market_ew_full_backtest": {
        "result_type": "full_backtest",
        "score": "vol_to_turnover",
        "portfolio_method": "equal_weight",
        "top_k": 20,
        "rebalance_rule": "M",
        "max_turnover": 1.0,
        "execution_mode": "tplus1_open",
        "prefilter": False,
        "universe_filter": True,
        "benchmark_symbol": "market_ew_proxy",
    },
    "p1_tree_full_backtest": {
        "result_type": "full_backtest",
        "score": "xgboost",
        "portfolio_method": "equal_weight",
        "top_k": 20,
        "rebalance_rule": "M",
        "max_turnover": 1.0,
        "execution_mode": "tplus1_open",
        "prefilter": False,
        "universe_filter": True,
        "benchmark_symbol": "market_ew_proxy",
    },
}


def slugify_token(value: Any) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "na"


def _compact_list_token(values: list[Any], *, max_items: int = 4) -> str:
    items = [slugify_token(v) for v in values if str(v).strip()]
    if not items:
        return "none"
    if len(items) <= max_items:
        return "-".join(items)
    return f"{'-'.join(items[:max_items])}-plus{len(items) - max_items}"


def build_light_research_identity(
    *,
    topic: str,
    output_prefix: str,
    baseline_factor: str,
    rebalance_rule: str,
    top_k: int,
    benchmark_key_years: list[int],
    selector_parts: dict[str, Any] | None = None,
) -> dict[str, str]:
    years_token = "-".join(str(int(item)) for item in benchmark_key_years) if benchmark_key_years else "none"
    selectors = []
    for key, value in (selector_parts or {}).items():
        if isinstance(value, list):
            selectors.append(f"{slugify_token(key)}_{_compact_list_token(list(value))}")
        else:
            selectors.append(f"{slugify_token(key)}_{slugify_token(value)}")
    selector_token = "_".join(selectors) if selectors else "default"
    research_config_id = (
        f"base_{slugify_token(baseline_factor)}_rb_{slugify_token(rebalance_rule)}"
        f"_top{int(top_k)}_yrs_{years_token}_{selector_token}"
    )
    return {
        "research_topic": slugify_token(topic),
        "research_config_id": research_config_id,
        "output_stem": f"{slugify_token(output_prefix)}_{research_config_id}",
    }


def canonical_research_config(name: str) -> dict[str, Any]:
    key = slugify_token(name)
    if key not in CANONICAL_RESEARCH_CONFIGS:
        raise KeyError(f"unknown canonical research config: {name}")
    return deepcopy(CANONICAL_RESEARCH_CONFIGS[key])


def build_full_backtest_research_identity(
    *,
    topic: str = "full_backtest",
    output_prefix: str = "full_backtest",
    sort_by: str,
    rebalance_rule: str,
    top_k: int,
    max_turnover: float,
    portfolio_method: str,
    execution_mode: str,
    prefilter_enabled: bool,
    universe_filter_enabled: bool,
    benchmark_symbol: str,
    start_date: str,
    end_date: str,
    selector_parts: dict[str, Any] | None = None,
) -> dict[str, str]:
    selectors = {
        "score": sort_by,
        "rb": rebalance_rule,
        "top": int(top_k),
        "to": f"{float(max_turnover):.4g}",
        "pm": portfolio_method,
        "exec": execution_mode,
        "pre": "on" if prefilter_enabled else "off",
        "uni": "on" if universe_filter_enabled else "off",
        "bm": benchmark_symbol,
        "start": start_date,
        "end": end_date,
    }
    selectors.update(selector_parts or {})
    research_config_id = "_".join(f"{slugify_token(k)}_{slugify_token(v)}" for k, v in selectors.items())
    return {
        "result_type": "full_backtest",
        "research_topic": slugify_token(topic),
        "research_config_id": research_config_id,
        "output_stem": f"{slugify_token(output_prefix)}_{research_config_id}",
    }
