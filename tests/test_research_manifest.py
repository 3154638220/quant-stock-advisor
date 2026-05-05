"""Tests for src/research/manifest.py - research manifest utilities."""

from __future__ import annotations

from src.models.research_contract import (
    DataSlice,
    ResearchIdentity,
)
from src.research.manifest import (
    make_research_identity,
    record_backtest_result,
    record_experiment_result,
    slugify_token,
)

# ── slugify_token ──────────────────────────────────────────────────────────


def test_slugify_token_basic():
    assert slugify_token("Hello World") == "hello_world"


def test_slugify_token_special():
    assert slugify_token("a/b@c#d!e") == "a_b_c_d_e"


def test_slugify_token_empty():
    assert slugify_token("") == "na"


def test_slugify_token_numeric():
    assert slugify_token(42) == "42"


# ── make_research_identity ─────────────────────────────────────────────────


def test_make_research_identity_minimal():
    ident = make_research_identity(
        result_type="test_type",
        research_topic="test_topic",
        research_config_id="cfg_001",
        output_stem="test_output",
    )
    assert isinstance(ident, ResearchIdentity)
    assert ident.result_type == "test_type"
    assert ident.research_topic == "test_topic"
    assert ident.research_config_id == "cfg_001"
    assert ident.output_stem == "test_output"


def test_make_research_identity_with_optional():
    ident = make_research_identity(
        result_type="test_type",
        research_topic="test_topic",
        research_config_id="cfg_001",
        output_stem="test_output",
        canonical_config_name="canonical_v1",
        parent_result_id="parent_001",
    )
    assert ident.canonical_config_name == "canonical_v1"
    assert ident.parent_result_id == "parent_001"


# ── record_backtest_result ─────────────────────────────────────────────────


def test_record_backtest_result_writes_files(tmp_path):
    from src.models.research_contract import ArtifactRef

    identity = make_research_identity(
        result_type="test_backtest",
        research_topic="test_topic",
        research_config_id="cfg_backtest",
        output_stem="test_backtest_2023",
    )
    data_slice = DataSlice(
        dataset_name="test_ds",
        source_tables=("test.parquet",),
        date_start="2020-01-01",
        date_end="2023-12-31",
        asof_trade_date="2023-12-31",
        signal_date_col="signal_date",
        symbol_col="symbol",
        candidate_pool_version="U1",
        rebalance_rule="M",
        execution_mode="tplus1_open",
        label_return_mode="open_to_open",
        feature_set_id="test_fs",
        pit_policy="test",
        config_path="test_config",
    )
    artifacts = [ArtifactRef("leaderboard_csv", "test.csv", "csv")]

    result = record_backtest_result(
        experiments_dir=tmp_path,
        identity=identity,
        data_slice=data_slice,
        params={"alpha": 0.05},
        metrics={"sharpe": 1.2},
        duration_sec=1.5,
        artifacts=artifacts,
        config_source="test_config.yaml",
        notes="test run",
    )

    assert "manifest" in result
    assert result["manifest"].exists()
    manifest_content = result["manifest"].read_text()
    assert "test_backtest" in manifest_content


def test_record_backtest_result_with_dict_identity(tmp_path):
    identity_dict = {
        "result_type": "test_type",
        "research_topic": "test_topic",
        "research_config_id": "cfg_dict",
        "output_stem": "test_dict",
    }
    data_slice = DataSlice(
        dataset_name="test_ds", source_tables=("test.parquet",),
        date_start="2020-01-01", date_end="2023-12-31",
        asof_trade_date="2023-12-31",
        signal_date_col="signal_date", symbol_col="symbol",
        candidate_pool_version="U1", rebalance_rule="M",
        execution_mode="tplus1_open", label_return_mode="open_to_open",
        feature_set_id="test_fs",
        pit_policy="test", config_path="test",
    )
    result = record_backtest_result(
        experiments_dir=tmp_path,
        identity=identity_dict,
        data_slice=data_slice,
        params={},
        metrics={},
        duration_sec=0.5,
        config_source="test_config.yaml",
    )
    assert "manifest" in result
    assert result["manifest"].exists()


# ── record_experiment_result ───────────────────────────────────────────────


def test_record_experiment_result(tmp_path):
    identity = make_research_identity(
        result_type="test_experiment",
        research_topic="test_topic",
        research_config_id="cfg_experiment",
        output_stem="test_experiment_2023",
    )
    data_slice = DataSlice(
        dataset_name="test_ds", source_tables=("test.parquet",),
        date_start="2020-01-01", date_end="2023-12-31",
        asof_trade_date="2023-12-31",
        signal_date_col="signal_date", symbol_col="symbol",
        candidate_pool_version="U1", rebalance_rule="M",
        execution_mode="tplus1_open", label_return_mode="open_to_open",
        feature_set_id="test_fs",
        pit_policy="test", config_path="test",
    )
    result = record_experiment_result(
        experiments_dir=tmp_path,
        identity=identity,
        data_slice=data_slice,
        params={"lr": 0.01},
        metrics={"ic": 0.05},
        duration_sec=0.8,
        notes="experiment run",
    )
    assert "manifest" in result
    assert result["manifest"].exists()
