from .factor_eval import (
    ic_summary,
    information_coefficient,
    long_table_from_wide,
    quantile_returns,
    rank_ic,
    rolling_ic_stability,
)
from .fundamental_factors import DEFAULT_FUNDAMENTAL_COLS, preprocess_fundamental_cross_section
from .neutralize import (
    attach_neutralized_pair,
    neutralize_cross_section,
    neutralize_industry,
    neutralize_size_industry_regression,
)
from .panel import (
    pivot_close_wide,
    pivot_field_aligned_to_close,
    pivot_field_wide,
    wide_close_to_numpy,
)
from .standardize import factor_standardize_pipeline, winsorize_by_date, zscore_by_date
from .tensor_alpha import compute_momentum_rsi_torch, weekly_kdj_from_daily
from .tensor_base_factors import (
    atr_wilder,
    bias_from_close,
    compute_base_factor_bundle,
    daily_returns_from_close,
    forward_returns_from_close,
    forward_returns_tplus1_open,
    limit_move_count,
    log_float_market_cap_proxy,
    max_single_day_drop,
    price_position_in_range,
    rolling_realized_volatility,
    rolling_turnover_mean,
    rolling_volume_return_corr,
    short_term_reversal,
    true_range,
)

try:
    from .fund_flow_factors import DEFAULT_FUND_FLOW_WINDOWS, attach_fund_flow
except ModuleNotFoundError:  # pragma: no cover - 轻量测试环境可不装 duckdb
    DEFAULT_FUND_FLOW_WINDOWS = (5, 10, 20)

    def attach_fund_flow(*args, **kwargs):
        raise ModuleNotFoundError("attach_fund_flow 需要安装 duckdb")


try:
    from .shareholder_factors import (
        DEFAULT_SHAREHOLDER_AVAILABILITY_LAG_DAYS,
        attach_shareholder_factors,
    )
except ModuleNotFoundError:  # pragma: no cover - 轻量测试环境可不装 duckdb
    DEFAULT_SHAREHOLDER_AVAILABILITY_LAG_DAYS = 30

    def attach_shareholder_factors(*args, **kwargs):
        raise ModuleNotFoundError("attach_shareholder_factors 需要安装 duckdb")

__all__ = [
    "atr_wilder",
    "attach_neutralized_pair",
    "bias_from_close",
    "compute_base_factor_bundle",
    "compute_momentum_rsi_torch",
    "daily_returns_from_close",
    "factor_standardize_pipeline",
    "DEFAULT_FUNDAMENTAL_COLS",
    "DEFAULT_FUND_FLOW_WINDOWS",
    "DEFAULT_SHAREHOLDER_AVAILABILITY_LAG_DAYS",
    "forward_returns_from_close",
    "forward_returns_tplus1_open",
    "ic_summary",
    "information_coefficient",
    "limit_move_count",
    "log_float_market_cap_proxy",
    "long_table_from_wide",
    "max_single_day_drop",
    "neutralize_cross_section",
    "neutralize_industry",
    "neutralize_size_industry_regression",
    "pivot_close_wide",
    "pivot_field_aligned_to_close",
    "pivot_field_wide",
    "price_position_in_range",
    "preprocess_fundamental_cross_section",
    "attach_fund_flow",
    "attach_shareholder_factors",
    "quantile_returns",
    "rank_ic",
    "rolling_ic_stability",
    "rolling_realized_volatility",
    "rolling_turnover_mean",
    "rolling_volume_return_corr",
    "short_term_reversal",
    "true_range",
    "weekly_kdj_from_daily",
    "winsorize_by_date",
    "wide_close_to_numpy",
    "zscore_by_date",
]
