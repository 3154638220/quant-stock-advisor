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
from .standardize import factor_standardize_pipeline, winsorize_by_date, zscore_by_date

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
    "attach_neutralized_pair",
    "factor_standardize_pipeline",
    "DEFAULT_FUNDAMENTAL_COLS",
    "DEFAULT_FUND_FLOW_WINDOWS",
    "DEFAULT_SHAREHOLDER_AVAILABILITY_LAG_DAYS",
    "ic_summary",
    "information_coefficient",
    "long_table_from_wide",
    "neutralize_cross_section",
    "neutralize_industry",
    "neutralize_size_industry_regression",
    "preprocess_fundamental_cross_section",
    "attach_fund_flow",
    "attach_shareholder_factors",
    "quantile_returns",
    "rank_ic",
    "rolling_ic_stability",
    "winsorize_by_date",
    "zscore_by_date",
]
