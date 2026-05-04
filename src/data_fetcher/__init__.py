from .akshare_client import fetch_a_share_daily, list_default_universe_symbols

__all__ = [
    "fetch_a_share_daily",
    "list_default_universe_symbols",
    "DuckDBManager",
    "SymbolUpdateResult",
    "QualityConfig",
    "QualityReport",
    "run_quality_checks",
    "validate_daily_frame",
    "FundamentalClient",
    "FundFlowClient",
    "ShareholderClient",
    "DailyDataSource",
    "FundamentalDataSource",
    "FundFlowDataSource",
    "AkShareDailyDataSource",
    "AkShareFundamentalDataSource",
    "AkShareFundFlowDataSource",
    "DataSourceSet",
    "FetchResult",
    "resolve_sources",
    "register_daily_source",
    "register_fundamental_source",
    "register_fund_flow_source",
    "IndustryMapQuality",
    "FALLBACK_SOURCE",
    "align_to_universe",
    "deduplicate_mapping",
    "fetch_industry_mapping",
    "fetch_mapping_by_source",
    "load_current_universe",
    "normalize_symbol",
    "quality_summary",
]


def __getattr__(name: str):
    if name in {"QualityConfig", "QualityReport", "run_quality_checks", "validate_daily_frame"}:
        from .data_quality import QualityConfig, QualityReport, run_quality_checks, validate_daily_frame

        return {
            "QualityConfig": QualityConfig,
            "QualityReport": QualityReport,
            "run_quality_checks": run_quality_checks,
            "validate_daily_frame": validate_daily_frame,
        }[name]
    if name in {"DuckDBManager", "SymbolUpdateResult"}:
        from .db_manager import DuckDBManager, SymbolUpdateResult

        return {"DuckDBManager": DuckDBManager, "SymbolUpdateResult": SymbolUpdateResult}[name]
    if name == "FundamentalClient":
        from .fundamental_client import FundamentalClient

        return FundamentalClient
    if name == "FundFlowClient":
        from .fund_flow_client import FundFlowClient

        return FundFlowClient
    if name == "ShareholderClient":
        from .shareholder_client import ShareholderClient

        return ShareholderClient
    if name in {
        "DailyDataSource",
        "FundamentalDataSource",
        "FundFlowDataSource",
        "AkShareDailyDataSource",
        "AkShareFundamentalDataSource",
        "AkShareFundFlowDataSource",
        "DataSourceSet",
        "FetchResult",
        "resolve_sources",
        "register_daily_source",
        "register_fundamental_source",
        "register_fund_flow_source",
    }:
        from . import adapter

        return getattr(adapter, name)
    if name in {
        "IndustryMapQuality",
        "FALLBACK_SOURCE",
        "align_to_universe",
        "deduplicate_mapping",
        "fetch_industry_mapping",
        "fetch_mapping_by_source",
        "load_current_universe",
        "normalize_symbol",
        "quality_summary",
    }:
        from . import industry_map

        return getattr(industry_map, name)
    raise AttributeError(name)
