"""
统一因子注册中心 (FeatureRegistry)。

每个因子（原始特征列）在此注册其元信息：所属家族、计算函数、
PIT 可用性、覆盖率阈值、IC 衰减阈值等。

Registry 是月度选股管线中特征列名和因子治理的唯一权威来源，
替代散落在各模块中的 *_RAW_FEATURES / *_FEATURES 字符串常量。

用法::

    from src.features.registry import (
        FACTOR_REGISTRY,
        FactorSpec,
        get_active_factors,
        get_factor_cols,
        register_factor,
    )

    # 按家族获取活跃因子
    cols = get_factor_cols("fundamental", use_zscore=True)

    # 按 IC 衰减过滤
    active = get_active_factors("fundamental", ic_monitor=monitor, decay_threshold=0.02)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import pandas as pd

# ═══════════════════════════════════════════════════════════════════════════
# FactorSpec
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class FactorSpec:
    """单个因子的注册元信息。

    Attributes
    ----------
    name : str
        因子短名（如 ``"ret_20d"``），同时也是原始列名的后缀。
        完整原始列名为 ``feature_{name}``。
    family : str
        所属家族：``price_volume`` / ``industry_breadth`` / ``fund_flow`` /
        ``fundamental`` / ``shareholder``。
    feature_col : str
        原始特征列名（如 ``"feature_ret_20d"``）。
    compute_fn : Callable | None
        因子计算函数（可选），用于惰性求值场景。
        签名应为 ``(df: pd.DataFrame, **kwargs) -> pd.Series``。
    requires_pit : bool
        是否需要 PIT（Point-in-Time）对齐。基本面/股东因子为 True。
    min_coverage : float
        在候选池内该因子非缺失值占比的最低阈值（0~1）。
    ic_decay_threshold : float
        |滚动 IC 均值| 低于此值时视为衰减，可由 ICMonitor 自动排除。
        默认 0.02。
    direction : int
        因子方向：+1 表示正向（值越大越好），-1 表示反向。
    description : str
        因子中文简述。
    """

    name: str
    family: str
    feature_col: str
    compute_fn: Optional[Callable] = None
    requires_pit: bool = False
    min_coverage: float = 0.10
    ic_decay_threshold: float = 0.02
    direction: int = 1
    description: str = ""

    # ── 运行时状态（由 IC Monitor 更新）────────────────────────────────
    active: bool = True
    """运行时可写：IC 衰减后标记为 False，管线自动跳过。"""

    @property
    def z_col(self) -> str:
        """截面 z-score 后的列名。"""
        return f"{self.feature_col}_z"

    @property
    def ind_z_col(self) -> str:
        """行业内 z-score 中性化后的列名。"""
        return f"{self.feature_col}_ind_z"

    @property
    def is_missing_col(self) -> str:
        """缺失标志列名。"""
        return f"is_missing_{self.feature_col}"


# ═══════════════════════════════════════════════════════════════════════════
# Global Registry
# ═══════════════════════════════════════════════════════════════════════════

FACTOR_REGISTRY: dict[str, FactorSpec] = {}
"""全局因子注册表：key 为因子 short name，value 为 FactorSpec。"""


def register_factor(spec: FactorSpec) -> FactorSpec:
    """注册一个因子到全局注册表。若同名已存在，覆盖更新。"""
    FACTOR_REGISTRY[spec.name] = spec
    return spec


def get_factor(name: str) -> Optional[FactorSpec]:
    """按 short name 查询因子。"""
    return FACTOR_REGISTRY.get(name)


def unregister_factor(name: str) -> bool:
    """从注册表中移除一个因子。返回是否成功移除。"""
    if name in FACTOR_REGISTRY:
        del FACTOR_REGISTRY[name]
        return True
    return False


# ═══════════════════════════════════════════════════════════════════════════
# Query API
# ═══════════════════════════════════════════════════════════════════════════

def get_factor_cols(
    family: str | Sequence[str] | None = None,
    *,
    use_zscore: bool = True,
    use_ind_zscore: bool = False,
    only_active: bool = True,
) -> list[str]:
    """获取指定家族的所有因子列名。

    Parameters
    ----------
    family
        家族名称或名称列表。None 表示所有家族。
    use_zscore
        返回 z-score 列名（``{col}_z``）。默认 True。
    use_ind_zscore
        返回行业中性化列名（``{col}_ind_z``）。若同时为 True，优先 ind_z。
    only_active
        仅返回 ``active=True`` 的因子列（默认）。

    Returns
    -------
    list[str]
        因子列名列表，保持注册顺序。
    """
    if isinstance(family, str):
        families = {family}
    elif family is None:
        families = None
    else:
        families = set(family)

    cols: list[str] = []
    for spec in FACTOR_REGISTRY.values():
        if families is not None and spec.family not in families:
            continue
        if only_active and not spec.active:
            continue
        if use_ind_zscore:
            cols.append(spec.ind_z_col)
        elif use_zscore:
            cols.append(spec.z_col)
        else:
            cols.append(spec.feature_col)
    return cols


def get_active_factors(
    family: str | Sequence[str] | None = None,
    *,
    ic_monitor=None,
    decay_threshold: float | None = None,
    decay_window: int = 20,
) -> list[FactorSpec]:
    """获取活跃因子列表，可选按 IC 衰减过滤。

    Parameters
    ----------
    family
        家族名称或列表。None = 所有。
    ic_monitor : ICMonitor | None
        若传入，查询 IC 衰减状态并自动标记 inactive。
    decay_threshold : float | None
        IC 衰减阈值。若为 None，使用各因子的 ``ic_decay_threshold``。
    decay_window : int
        IC 滚动窗口大小（交易日数）。

    Returns
    -------
    list[FactorSpec]
        活跃的 FactorSpec 列表。
    """
    if ic_monitor is not None:
        # 查询 IC Monitor，自动标记衰减因子
        decayed_names: set[str] = set()
        try:
            all_decayed = ic_monitor.get_decayed_factors(
                window=decay_window,
                threshold=decay_threshold or 0.02,
                factors=list(FACTOR_REGISTRY.keys()) if family is None else (
                    [s.name for s in FACTOR_REGISTRY.values()
                     if (isinstance(family, str) and s.family == family)
                     or (isinstance(family, (list, tuple, set)) and s.family in family)
                     or family is None]
                ),
            )
            decayed_names = set(all_decayed)
        except Exception:
            pass  # IC Monitor 不可用时静默跳过

        for spec in FACTOR_REGISTRY.values():
            if spec.name in decayed_names:
                spec.active = False

    if isinstance(family, str):
        families = {family}
    elif family is None:
        families = None
    else:
        families = set(family)

    result: list[FactorSpec] = []
    for spec in FACTOR_REGISTRY.values():
        if families is not None and spec.family not in families:
            continue
        if not spec.active:
            continue
        result.append(spec)
    return result


def get_families_by_factor_names(names: list[str]) -> list[str]:
    """根据因子列名（_z 或原始列名）反查所属家族列表（去重）。"""
    families: dict[str, None] = {}
    for spec in FACTOR_REGISTRY.values():
        if spec.feature_col in names or spec.z_col in names or spec.ind_z_col in names:
            families[spec.family] = None
    return list(families.keys())


def reset_all_active() -> None:
    """将所有因子恢复为 active 状态。"""
    for spec in FACTOR_REGISTRY.values():
        spec.active = True


# ═══════════════════════════════════════════════════════════════════════════
# Initial Registration — 量价因子
# ═══════════════════════════════════════════════════════════════════════════

_PRICE_VOLUME_FACTORS: list[dict] = [
    {"name": "ret_5d",         "feature_col": "feature_ret_5d",            "direction": 1,  "description": "5 日动量",         "min_coverage": 0.50},
    {"name": "ret_20d",        "feature_col": "feature_ret_20d",           "direction": 1,  "description": "20 日动量",        "min_coverage": 0.50},
    {"name": "ret_60d",        "feature_col": "feature_ret_60d",           "direction": 1,  "description": "60 日动量",        "min_coverage": 0.50},
    {"name": "realized_vol_20d",  "feature_col": "feature_realized_vol_20d",  "direction": -1, "description": "20 日已实现波动率",  "min_coverage": 0.50},
    {"name": "amount_20d_log",    "feature_col": "feature_amount_20d_log",    "direction": 1,  "description": "20 日成交额对数",   "min_coverage": 0.50},
    {"name": "turnover_20d",      "feature_col": "feature_turnover_20d",      "direction": 1,  "description": "20 日换手率",       "min_coverage": 0.50},
    {"name": "price_position_250d", "feature_col": "feature_price_position_250d", "direction": 1, "description": "250 日价格位置",   "min_coverage": 0.50},
    {"name": "limit_move_hits_20d", "feature_col": "feature_limit_move_hits_20d", "direction": -1, "description": "20 日涨跌停命中", "min_coverage": 0.50},
]

for _spec in _PRICE_VOLUME_FACTORS:
    register_factor(FactorSpec(
        name=_spec["name"],
        family="price_volume",
        feature_col=_spec["feature_col"],
        requires_pit=False,
        min_coverage=_spec["min_coverage"],
        direction=_spec["direction"],
        description=_spec["description"],
    ))

# ═══════════════════════════════════════════════════════════════════════════
# Initial Registration — 行业宽度因子
# ═══════════════════════════════════════════════════════════════════════════

_INDUSTRY_BREADTH_FACTORS: list[dict] = [
    {"name": "industry_ret20_mean",            "feature_col": "feature_industry_ret20_mean",            "direction": 1,  "description": "行业内 20 日收益均值"},
    {"name": "industry_ret60_mean",            "feature_col": "feature_industry_ret60_mean",            "direction": 1,  "description": "行业内 60 日收益均值"},
    {"name": "industry_positive_ret20_ratio",  "feature_col": "feature_industry_positive_ret20_ratio",  "direction": 1,  "description": "行业内正收益占比"},
    {"name": "industry_amount20_mean",         "feature_col": "feature_industry_amount20_mean",         "direction": 1,  "description": "行业内成交额均值"},
    {"name": "industry_low_vol20_mean",        "feature_col": "feature_industry_low_vol20_mean",        "direction": -1, "description": "行业内低波均值"},
]

for _spec in _INDUSTRY_BREADTH_FACTORS:
    register_factor(FactorSpec(
        name=_spec["name"],
        family="industry_breadth",
        feature_col=_spec["feature_col"],
        requires_pit=False,
        min_coverage=0.30,
        direction=_spec["direction"],
        description=_spec["description"],
    ))

# ═══════════════════════════════════════════════════════════════════════════
# Initial Registration — 资金流因子
# ═══════════════════════════════════════════════════════════════════════════

_FUND_FLOW_FACTORS: list[dict] = [
    {"name": "fund_flow_main_inflow_5d",     "feature_col": "feature_fund_flow_main_inflow_5d",     "direction": 1,  "description": "主力 5 日净流入"},
    {"name": "fund_flow_main_inflow_10d",    "feature_col": "feature_fund_flow_main_inflow_10d",    "direction": 1,  "description": "主力 10 日净流入"},
    {"name": "fund_flow_main_inflow_20d",    "feature_col": "feature_fund_flow_main_inflow_20d",    "direction": 1,  "description": "主力 20 日净流入"},
    {"name": "fund_flow_super_inflow_10d",   "feature_col": "feature_fund_flow_super_inflow_10d",   "direction": 1,  "description": "超大单 10 日净流入"},
    {"name": "fund_flow_divergence_20d",     "feature_col": "feature_fund_flow_divergence_20d",     "direction": 1,  "description": "主力-小单 20 日分化"},
    {"name": "fund_flow_main_inflow_streak", "feature_col": "feature_fund_flow_main_inflow_streak", "direction": 1,  "description": "主力连续净流入天数"},
]

for _spec in _FUND_FLOW_FACTORS:
    register_factor(FactorSpec(
        name=_spec["name"],
        family="fund_flow",
        feature_col=_spec["feature_col"],
        requires_pit=False,
        min_coverage=0.20,
        ic_decay_threshold=0.015,  # 资金流因子 alpha 衰减更快，设置更严阈值
        direction=_spec["direction"],
        description=_spec["description"],
    ))

# ═══════════════════════════════════════════════════════════════════════════
# Initial Registration — 基本面因子
# ═══════════════════════════════════════════════════════════════════════════

_FUNDAMENTAL_FACTOR_NAMES = [
    "pe_ttm", "pb",
    "ev_ebitda", "roe_ttm",
    "net_profit_yoy", "gross_margin_change", "gross_margin_delta",
    "debt_to_assets_change", "ocf_to_net_profit", "ocf_to_asset",
    "asset_turnover", "net_margin_stability",
]

# 基本面因子方向标记（正值小好为 -1，增长好为 1）
_FUNDAMENTAL_DIRECTION: dict[str, int] = {
    "pe_ttm": -1, "pb": -1, "ev_ebitda": -1,  # 估值：越低越好
    "roe_ttm": 1,
    "net_profit_yoy": 1, "gross_margin_change": 1, "gross_margin_delta": 1,
    "debt_to_assets_change": -1,
    "ocf_to_net_profit": 1, "ocf_to_asset": 1,
    "asset_turnover": 1, "net_margin_stability": 1,
}

for _name in _FUNDAMENTAL_FACTOR_NAMES:
    register_factor(FactorSpec(
        name=_name,
        family="fundamental",
        feature_col=f"feature_fundamental_{_name}",
        requires_pit=True,
        min_coverage=0.10  if _name == "ev_ebitda" else 0.30,
        ic_decay_threshold=0.02,
        direction=_FUNDAMENTAL_DIRECTION.get(_name, 1),
        description=f"基本面: {_name}",
    ))

# ═══════════════════════════════════════════════════════════════════════════
# Initial Registration — 股东因子
# ═══════════════════════════════════════════════════════════════════════════

_SHAREHOLDER_FACTORS: list[dict] = [
    {"name": "shareholder_holder_count_log",    "feature_col": "feature_shareholder_holder_count_log",    "direction": -1, "description": "股东户数对数（少好）"},
    {"name": "shareholder_holder_change_rate",  "feature_col": "feature_shareholder_holder_change_rate",  "direction": -1, "description": "股东户数变化率（减少好）"},
    {"name": "shareholder_concentration_proxy", "feature_col": "feature_shareholder_concentration_proxy", "direction": 1,  "description": "股东集中度代理变量"},
]

for _spec in _SHAREHOLDER_FACTORS:
    register_factor(FactorSpec(
        name=_spec["name"],
        family="shareholder",
        feature_col=_spec["feature_col"],
        requires_pit=True,
        min_coverage=0.30,
        ic_decay_threshold=0.02,
        direction=_spec["direction"],
        description=_spec["description"],
    ))

# ═══════════════════════════════════════════════════════════════════════════
# 便捷常量：向后兼容旧的元组定义
# ═══════════════════════════════════════════════════════════════════════════

# 这些常量保持与原有 *_RAW_FEATURES 元组相同的内容，
# 但从注册表动态生成，确保单一权威来源。

def _build_family_raw_tuple(family: str) -> tuple[str, ...]:
    return tuple(
        spec.feature_col
        for spec in FACTOR_REGISTRY.values()
        if spec.family == family
    )


PRICE_VOLUME_FEATURES_REGISTRY: tuple[str, ...] = _build_family_raw_tuple("price_volume")
INDUSTRY_BREADTH_FEATURES_REGISTRY: tuple[str, ...] = _build_family_raw_tuple("industry_breadth")
FUND_FLOW_FEATURES_REGISTRY: tuple[str, ...] = _build_family_raw_tuple("fund_flow")
FUNDAMENTAL_FEATURES_REGISTRY: tuple[str, ...] = _build_family_raw_tuple("fundamental")
SHAREHOLDER_FEATURES_REGISTRY: tuple[str, ...] = _build_family_raw_tuple("shareholder")
