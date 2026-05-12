from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd
import pytest

from src.pipeline.shared_loaders import DataLoader, DataLoaderConfig, attach_feature_families


def _dataset() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "signal_date": pd.to_datetime(["2026-02-28", "2026-02-28"]),
            "symbol": ["000001", "000002"],
            "candidate_pool_version": ["U1_liquid_tradable", "U1_liquid_tradable"],
            "industry_level1": ["电子", "计算机"],
            "feature_ret_20d": [0.10, -0.02],
            "feature_ret_60d": [0.20, 0.01],
            "feature_realized_vol_20d": [0.30, 0.15],
            "feature_amount_20d_log": [12.0, 11.0],
        }
    )


def test_data_loader_applies_families_in_canonical_order(tmp_path: Path):
    calls: list[str] = []

    def make_attach(name: str):
        def _attach(frame: pd.DataFrame) -> pd.DataFrame:
            calls.append(name)
            out = frame.copy()
            out[f"loaded_{name}"] = True
            return out

        return _attach

    loader = DataLoader(
        tmp_path / "empty.duckdb",
        registry={
            "fund_flow": make_attach("fund_flow"),
            "fundamental": make_attach("fundamental"),
            "shareholder": make_attach("shareholder"),
        },
        family_order=("fund_flow", "fundamental", "shareholder"),
    )

    out = loader.attach(_dataset(), ["shareholder", "fund_flow"])

    assert calls == ["fund_flow", "shareholder"]
    assert {"loaded_fund_flow", "loaded_shareholder"}.issubset(out.columns)
    assert "loaded_fundamental" not in out.columns


def test_data_loader_rejects_unknown_family_by_default(tmp_path: Path):
    loader = DataLoader(tmp_path / "empty.duckdb", registry={})

    with pytest.raises(ValueError, match="unknown_family"):
        loader.attach(_dataset(), ["unknown_family"])


def test_data_loader_can_ignore_unknown_family_when_configured(tmp_path: Path):
    loader = DataLoader(
        tmp_path / "empty.duckdb",
        config=DataLoaderConfig(strict_unknown_families=False),
        registry={},
    )

    out = loader.attach(_dataset(), ["unknown_family"])

    pd.testing.assert_frame_equal(out.reset_index(drop=True), _dataset().reset_index(drop=True))


def test_data_loader_treats_price_volume_as_builtin_family(tmp_path: Path):
    loader = DataLoader(tmp_path / "empty.duckdb", registry={})

    out = loader.attach(_dataset(), ["price_volume"])

    pd.testing.assert_frame_equal(out.reset_index(drop=True), _dataset().reset_index(drop=True))


def test_attach_feature_families_uses_shared_industry_loader(tmp_path: Path):
    out = attach_feature_families(_dataset(), tmp_path / "empty.duckdb", ["industry_breadth"])

    assert "feature_industry_ret20_mean" in out.columns
    assert "feature_industry_ret20_mean_z" in out.columns


def test_shared_fundamental_loader_keeps_pit_filtering(tmp_path: Path):
    db_path = tmp_path / "fundamental.duckdb"
    with duckdb.connect(str(db_path)) as con:
        con.execute(
            """
            CREATE TABLE a_share_fundamental (
                symbol VARCHAR,
                report_period DATE,
                announcement_date DATE,
                pe_ttm DOUBLE,
                pb DOUBLE,
                roe_ttm DOUBLE,
                source VARCHAR
            )
            """
        )
        con.execute(
            """
            INSERT INTO a_share_fundamental VALUES
            ('000001', DATE '2025-09-30', DATE '2025-10-31', 10, 1, 1, 'stock_value_em'),
            ('000001', DATE '2025-12-31', DATE '2025-12-31', 999, 1, 99, 'stock_financial_analysis_indicator'),
            ('000002', DATE '2025-09-30', DATE '2025-11-05', 20, 2, 2, 'stock_value_em')
            """
        )

    out = attach_feature_families(_dataset(), db_path, ["fundamental"]).sort_values("symbol").reset_index(drop=True)

    assert out.loc[0, "feature_fundamental_pe_ttm"] == 10
    assert out.loc[1, "feature_fundamental_pe_ttm"] == 20
