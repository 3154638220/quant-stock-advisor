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

import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

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

_REGISTRY_LOCK = threading.RLock()
"""全局注册表写操作的锁，保护并发注册/注销场景。"""

FACTOR_REGISTRY: dict[str, FactorSpec] = {}
"""全局因子注册表：key 为因子 short name，value 为 FactorSpec。"""

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FACTOR_GOVERNANCE_LOG = (
    PROJECT_ROOT / "data" / "results" / "factor_audit_2026_05_09_full" / "factor_governance_log.jsonl"
)
"""默认 W5 因子治理日志路径。不存在时治理加载会静默跳过。"""


def register_factor(spec: FactorSpec) -> FactorSpec:
    """注册一个因子到全局注册表。若同名已存在，覆盖更新。"""
    with _REGISTRY_LOCK:
        FACTOR_REGISTRY[spec.name] = spec
    return spec


def get_factor(name: str) -> Optional[FactorSpec]:
    """按 short name 查询因子。"""
    return FACTOR_REGISTRY.get(name)


def unregister_factor(name: str) -> bool:
    """从注册表中移除一个因子。返回是否成功移除。"""
    with _REGISTRY_LOCK:
        if name in FACTOR_REGISTRY:
            del FACTOR_REGISTRY[name]
            return True
    return False


def apply_factor_governance_log(path: str | Path | None = None) -> list[dict]:
    """Apply persisted factor governance actions to ``FACTOR_REGISTRY``.

    The governance log is append-only JSONL as written by
    ``scripts/apply_factor_audit_results.py``.  Each row may address a factor by
    ``registry_name`` or by ``feature_col``.  Missing logs are allowed so clean
    test/dev checkouts keep the default all-active registry behavior.
    """

    log_path = Path(path) if path is not None else DEFAULT_FACTOR_GOVERNANCE_LOG
    if not log_path.exists():
        return []

    feature_index = {spec.feature_col: spec for spec in FACTOR_REGISTRY.values()}
    applied: list[dict] = []
    with log_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            spec = FACTOR_REGISTRY.get(str(row.get("registry_name") or ""))
            if spec is None:
                spec = feature_index.get(str(row.get("feature_col") or ""))
            if spec is None:
                continue

            now_active = row.get("now_active")
            if isinstance(now_active, bool):
                spec.active = now_active
            elif str(now_active).strip().lower() in {"true", "false"}:
                spec.active = str(now_active).strip().lower() == "true"
            else:
                spec.active = False
            applied.append(row)
    return applied


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
    with _REGISTRY_LOCK:
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

# Northbound factors (M11-A: 北向资金)
_NORTHBOUND_FACTORS = [
    {"name": "northbound_hold_ratio", "feature_col": "feature_northbound_hold_ratio",
     "direction": 1, "description": "北向持股占比（最新）"},
    {"name": "northbound_net_buy_1m", "feature_col": "feature_northbound_net_buy_1m",
     "direction": 1, "description": "近 1 月北向净买入合计额"},
    {"name": "northbound_hold_change_1m", "feature_col": "feature_northbound_hold_change_1m",
     "direction": 1, "description": "近 1 月北向持股占比变化"},
    {"name": "northbound_inflow_stability_1m", "feature_col": "feature_northbound_inflow_stability_1m",
     "direction": 1, "description": "近 1 月北向净流入稳定性（正流入天数占比）"},
]

for _spec in _NORTHBOUND_FACTORS:
    register_factor(FactorSpec(
        name=_spec["name"],
        family="northbound",
        feature_col=_spec["feature_col"],
        min_coverage=0.02,
        ic_decay_threshold=0.02,
        direction=_spec["direction"],
        description=_spec["description"],
    ))

# ── M13-C: 北向资金 regime 因子（市场级，无个股维度）────────────────────
_NORTHBOUND_REGIME_FACTORS = [
    {"name": "north_net_inflow_1m", "feature_col": "feature_north_net_inflow_1m",
     "direction": 1, "description": "近 1 月北向净流入总额（市场级 regime）"},
    {"name": "north_net_inflow_3m", "feature_col": "feature_north_net_inflow_3m",
     "direction": 1, "description": "近 3 月北向净流入总额（市场级 regime）"},
    {"name": "north_inflow_zscore_6m", "feature_col": "feature_north_inflow_zscore_6m",
     "direction": 1, "description": "近 1 月净流入相对 6 月均值的 z-score"},
    {"name": "north_consecutive_outflow_days", "feature_col": "feature_north_consecutive_outflow_days",
     "direction": -1, "description": "连续净流出天数（负向风险 regime 信号）"},
]

for _spec in _NORTHBOUND_REGIME_FACTORS:
    register_factor(FactorSpec(
        name=_spec["name"],
        family="northbound_regime",
        feature_col=_spec["feature_col"],
        min_coverage=0.0,
        ic_decay_threshold=0.01,
        direction=_spec["direction"],
        description=_spec["description"],
    ))

# Margin trading factors (M11-B: 融资融券)
_MARGIN_TRADING_FACTORS = [
    {"name": "margin_fin_balance_ratio", "feature_col": "feature_margin_fin_balance_ratio",
     "direction": -1, "description": "融资余额占融资买入额比例（近似杠杆倾向，逆向信号）"},
    {"name": "margin_net_fin_buy_1m", "feature_col": "feature_margin_net_fin_buy_1m",
     "direction": -1, "description": "近 1 月融资净买入额（散户杠杆拥挤，逆向信号）"},
    {"name": "margin_short_pressure_1m", "feature_col": "feature_margin_short_pressure_1m",
     "direction": -1, "description": "近 1 月融券余量变化率（机构做空压力）"},
    {"name": "margin_fin_balance_momentum_1m", "feature_col": "feature_margin_fin_balance_momentum_1m",
     "direction": -1, "description": "近 1 月融资余额增长率（杠杆加速，逆向信号）"},
]

for _spec in _MARGIN_TRADING_FACTORS:
    register_factor(FactorSpec(
        name=_spec["name"],
        family="margin_trading",
        feature_col=_spec["feature_col"],
        min_coverage=0.05,
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
NORTHBOUND_FEATURES_REGISTRY: tuple[str, ...] = _build_family_raw_tuple("northbound")
NORTHBOUND_REGIME_FEATURES_REGISTRY: tuple[str, ...] = _build_family_raw_tuple("northbound_regime")
MARGIN_TRADING_FEATURES_REGISTRY: tuple[str, ...] = _build_family_raw_tuple("margin_trading")

# ═══════════════════════════════════════════════════════════════════════════
# 概念板块（P5: M11-C concept/theme breadth）
# ═══════════════════════════════════════════════════════════════════════════

# V1 市场级 breadth 因子来自板块日线；M13-B 个股绑定因子来自
# a_share_concept_membership 快照，仅按 snapshot_date <= signal_date 且
# rolling 60 天有效期使用，避免把当前成分回填到历史。

_CONCEPT_FACTORS: list[dict] = [
    {"name": "concept_breadth_1m", "feature_col": "feature_concept_breadth_1m",
     "direction": 1, "description": "概念板块近月上涨占比（市场级 breadth）"},
    {"name": "concept_dispersion_1m", "feature_col": "feature_concept_dispersion_1m",
     "direction": 1, "description": "概念板块近月收益率截面标准差（市场 dispersion）"},
    {"name": "concept_momentum_1m", "feature_col": "feature_concept_momentum_1m",
     "direction": 1, "description": "概念板块近月平均收益率（市场 momentum）"},
    {"name": "concept_member_count", "feature_col": "feature_concept_member_count",
     "direction": 1, "description": "个股所属概念数量（M13-B 个股绑定）"},
    {"name": "hot_concept_membership", "feature_col": "feature_hot_concept_membership",
     "direction": 1, "description": "是否属于近月热门概念 Top-10（M13-B）"},
    {"name": "concept_ew_return_1m", "feature_col": "feature_concept_ew_return_1m",
     "direction": 1, "description": "所属概念近月收益均值（M13-B）"},
    {"name": "concept_max_return_1m", "feature_col": "feature_concept_max_return_1m",
     "direction": 1, "description": "所属概念近月最高收益（M13-B）"},
    {"name": "concept_inflow_breadth", "feature_col": "feature_concept_inflow_breadth",
     "direction": 1, "description": "所属概念中资金净流入成员占比均值（M13-B）"},
    {"name": "concept_return_dispersion", "feature_col": "feature_concept_return_dispersion",
     "direction": 1, "description": "所属概念近月收益离散度（M13-B）"},
]

for _spec in _CONCEPT_FACTORS:
    register_factor(FactorSpec(
        name=_spec["name"],
        family="concept",
        feature_col=_spec["feature_col"],
        min_coverage=0.05,
        ic_decay_threshold=0.02,
        direction=_spec["direction"],
        description=_spec["description"],
    ))

CONCEPT_FEATURES_REGISTRY: tuple[str, ...] = _build_family_raw_tuple("concept")
SHAREHOLDER_FEATURES_REGISTRY: tuple[str, ...] = _build_family_raw_tuple("shareholder")

# ═══════════════════════════════════════════════════════════════════════════
# 龙虎榜（M14: 月度选股因子）
# ═══════════════════════════════════════════════════════════════════════════

_LHB_FACTORS: list[dict] = [
    {"name": "lhb_appearance_count_1m", "feature_col": "feature_lhb_appearance_count_1m",
     "direction": 1, "description": "近月上榜次数（活跃度）"},
    {"name": "lhb_appearance_count_3m", "feature_col": "feature_lhb_appearance_count_3m",
     "direction": 1, "description": "近季上榜次数"},
    {"name": "lhb_recent_5d", "feature_col": "feature_lhb_recent_5d",
     "direction": 1, "description": "近 5 日是否上榜"},
    {"name": "lhb_avg_change_1m", "feature_col": "feature_lhb_avg_change_1m",
     "direction": 1, "description": "上榜日平均涨跌幅"},
    {"name": "lhb_avg_amount_1m", "feature_col": "feature_lhb_avg_amount_1m",
     "direction": 1, "description": "上榜日平均成交额(万元)"},
    {"name": "lhb_is_bullish_1m", "feature_col": "feature_lhb_is_bullish_1m",
     "direction": 1, "description": "近月上榜原因含看涨信号"},
    {"name": "lhb_is_bearish_1m", "feature_col": "feature_lhb_is_bearish_1m",
     "direction": -1, "description": "近月上榜原因含看跌信号"},
    {"name": "lhb_is_high_turnover_1m", "feature_col": "feature_lhb_is_high_turnover_1m",
     "direction": 1, "description": "近月因高换手上榜"},
]

for _spec in _LHB_FACTORS:
    register_factor(FactorSpec(
        name=_spec["name"],
        family="lhb",
        feature_col=_spec["feature_col"],
        min_coverage=0.01,  # LHB 天然稀疏，覆盖率要求极低
        ic_decay_threshold=0.02,
        direction=_spec["direction"],
        description=_spec["description"],
    ))

LHB_FEATURES_REGISTRY: tuple[str, ...] = _build_family_raw_tuple("lhb")

# ═══════════════════════════════════════════════════════════════════════════
# W1: 质量因子（Alpha158 对齐）
# ═══════════════════════════════════════════════════════════════════════════

_QUALITY_FACTORS: list[dict] = [
    {"name": "roe_stability", "feature_col": "feature_quality_roe_stability",
     "direction": 1, "description": "ROE 过去 8 季度标准差取负（高稳定性=高质量）"},
    {"name": "accruals_ratio", "feature_col": "feature_quality_accruals_ratio",
     "direction": -1, "description": "应计项目比率代理（高应计=盈利质量差）"},
    {"name": "asset_growth_rate", "feature_col": "feature_quality_asset_growth_rate",
     "direction": -1, "description": "资产增长率代理（过度扩张=负向）"},
    {"name": "earnings_surprise", "feature_col": "feature_quality_earnings_surprise",
     "direction": 1, "description": "盈利惊喜：最新 net_profit_yoy 相对历史均值的偏离"},
]

for _spec in _QUALITY_FACTORS:
    register_factor(FactorSpec(
        name=_spec["name"],
        family="quality",
        feature_col=_spec["feature_col"],
        requires_pit=True,
        min_coverage=0.30,
        ic_decay_threshold=0.02,
        direction=_spec["direction"],
        description=_spec["description"],
    ))

QUALITY_FEATURES_REGISTRY: tuple[str, ...] = _build_family_raw_tuple("quality")

# ═══════════════════════════════════════════════════════════════════════════
# W1 Phase 2: 短期反转 & 量能异常因子（Alpha158 对齐）
# ═══════════════════════════════════════════════════════════════════════════

_REVERSAL_VOLUME_FACTORS: list[dict] = [
    {"name": "st_reversal_1m", "feature_col": "feature_reversal_st_reversal_1m",
     "direction": -1, "description": "近月超额收益（短期反转=做空信号）"},
    {"name": "st_reversal_1w", "feature_col": "feature_reversal_st_reversal_1w",
     "direction": -1, "description": "近周超额收益（短期反转=做空信号）"},
    {"name": "volume_spike", "feature_col": "feature_reversal_volume_spike",
     "direction": -1, "description": "近5日/近60日成交额比（异常放量=情绪过热）"},
    {"name": "turnover_anomaly", "feature_col": "feature_reversal_turnover_anomaly",
     "direction": -1, "description": "换手率相对60日均值的z-score（异常换手=散户拥挤）"},
    {"name": "pv_divergence", "feature_col": "feature_reversal_pv_divergence",
     "direction": -1, "description": "近20日量价相关性取负（价涨量缩/价跌量增=背离）"},
]

for _spec in _REVERSAL_VOLUME_FACTORS:
    register_factor(FactorSpec(
        name=_spec["name"],
        family="reversal_volume",
        feature_col=_spec["feature_col"],
        requires_pit=False,
        min_coverage=0.30,
        ic_decay_threshold=0.02,
        direction=_spec["direction"],
        description=_spec["description"],
    ))

REVERSAL_VOLUME_FEATURES_REGISTRY: tuple[str, ...] = _build_family_raw_tuple("reversal_volume")

# ═══════════════════════════════════════════════════════════════════════════
# W1 Phase 3: 流动性 & 价格位置扩展因子（Alpha158 对齐）
# ═══════════════════════════════════════════════════════════════════════════

_LIQUIDITY_POSITION_FACTORS: list[dict] = [
    {"name": "amihud", "feature_col": "feature_liquidity_amihud",
     "direction": -1, "description": "Amihud非流动性指标（日均|收益|/成交额×10^6，高=流动性差）"},
    {"name": "high52w_ratio", "feature_col": "feature_liquidity_high52w_ratio",
     "direction": 1, "description": "收盘价/52周最高价（接近高点=动量延续）"},
    {"name": "low52w_ratio", "feature_col": "feature_liquidity_low52w_ratio",
     "direction": -1, "description": "收盘价/52周最低价（接近低点=负向信号）"},
    {"name": "price_range_width", "feature_col": "feature_liquidity_price_range_width",
     "direction": -1, "description": "(52周高-低)/52周均价（宽幅震荡=不确定性高）"},
]

for _spec in _LIQUIDITY_POSITION_FACTORS:
    register_factor(FactorSpec(
        name=_spec["name"],
        family="liquidity_position",
        feature_col=_spec["feature_col"],
        requires_pit=False,
        min_coverage=0.30,
        ic_decay_threshold=0.02,
        direction=_spec["direction"],
        description=_spec["description"],
    ))

LIQUIDITY_POSITION_FEATURES_REGISTRY: tuple[str, ...] = _build_family_raw_tuple("liquidity_position")

# ═══════════════════════════════════════════════════════════════════════════
# W4: 结构化事件因子（M16 前置探索）
# ═══════════════════════════════════════════════════════════════════════════

_EVENT_FACTORS: list[dict] = [
    {"name": "earnings_guidance_direction", "feature_col": "feature_event_earnings_guidance_direction",
     "direction": 1, "description": "业绩预告方向（上调/下调）"},
    {"name": "earnings_guidance_magnitude", "feature_col": "feature_event_earnings_guidance_magnitude",
     "direction": 1, "description": "业绩预告幅度（预告净利润相对去年变化）"},
    {"name": "earnings_surprise_ttm", "feature_col": "feature_event_earnings_surprise_ttm",
     "direction": 1, "description": "业绩预告惊喜（相对历史分布的标准化偏离）"},
    {"name": "buyback_amount_ratio", "feature_col": "feature_event_buyback_amount_ratio",
     "direction": 1, "description": "回购金额/市值"},
    {"name": "buyback_recent_30d", "feature_col": "feature_event_buyback_recent_30d",
     "direction": 1, "description": "近30日是否存在回购公告"},
    {"name": "buyback_amount_ratio_180d", "feature_col": "feature_event_buyback_amount_ratio_180d",
     "direction": 1, "description": "近180日回购金额/市值"},
    {"name": "buyback_recent_180d", "feature_col": "feature_event_buyback_recent_180d",
     "direction": 1, "description": "近180日是否存在回购公告"},
    {"name": "reduction_plan_flag", "feature_col": "feature_event_reduction_plan_flag",
     "direction": -1, "description": "近30日是否存在减持计划"},
    {"name": "reduction_plan_flag_180d", "feature_col": "feature_event_reduction_plan_flag_180d",
     "direction": -1, "description": "近180日是否存在减持计划"},
    {"name": "reduction_ratio_180d", "feature_col": "feature_event_reduction_ratio_180d",
     "direction": -1, "description": "近180日减持比例合计"},
    {"name": "unlock_ratio_30d", "feature_col": "feature_event_unlock_ratio_30d",
     "direction": -1, "description": "未来30日解禁市值/市值"},
    {"name": "unlock_ratio_90d", "feature_col": "feature_event_unlock_ratio_90d",
     "direction": -1, "description": "未来90日解禁市值/市值"},
]

for _spec in _EVENT_FACTORS:
    register_factor(FactorSpec(
        name=_spec["name"],
        family="event",
        feature_col=_spec["feature_col"],
        requires_pit=True,
        min_coverage=0.10,
        ic_decay_threshold=0.02,
        direction=_spec["direction"],
        description=_spec["description"],
    ))

EVENT_FEATURES_REGISTRY: tuple[str, ...] = _build_family_raw_tuple("event")
