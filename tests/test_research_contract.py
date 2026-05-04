from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.cli.research_identity import make_research_identity
try:
    from scripts.validate_research_contracts import validate_manifest
except ImportError:
    validate_manifest = None  # type: ignore[assignment]
from src.models.experiment import append_experiment_result
from src.models.research_contract import (
    ArtifactRef,
    DataSlice,
    ExperimentResult,
    build_result_id,
    write_research_manifest,
)


def _slice(**overrides) -> DataSlice:
    values = {
        "dataset_name": "monthly_selection_features",
        "source_tables": ("a_share_daily",),
        "date_start": "2021-01-01",
        "date_end": "2026-04-30",
        "asof_trade_date": "2026-04-30",
        "signal_date_col": "signal_date",
        "symbol_col": "symbol",
        "candidate_pool_version": "U1_liquid_tradable",
        "rebalance_rule": "M",
        "execution_mode": "tplus1_open",
        "label_return_mode": "open_to_open",
        "feature_set_id": "price_volume_v1",
        "feature_columns": ("feature_ret_20d",),
        "label_columns": ("label_forward_1m_o2o_return",),
        "pit_policy": "signal_date_close_visible_only",
        "config_path": "config.yaml.backtest",
        "extra": {},
    }
    values.update(overrides)
    return DataSlice(**values)


def _result(tmp_path: Path) -> tuple[ExperimentResult, Path]:
    identity = make_research_identity(
        result_type="monthly_selection_dataset",
        research_topic="monthly_selection_dataset",
        research_config_id="Dataset Contract V1",
        output_stem="Monthly Dataset Contract V1",
    )
    data_slice = _slice()
    summary = tmp_path / "summary.json"
    summary.write_text("{}", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    artifacts = (
        ArtifactRef("summary_json", str(summary), "json"),
        ArtifactRef("manifest_json", str(manifest), "json"),
    )
    metrics = {"rows": 10, "label_valid_rows": 8}
    result = ExperimentResult(
        result_id=build_result_id(identity, [data_slice], metrics),
        identity=identity,
        script_name="scripts/run_monthly_selection_dataset.py",
        command="python scripts/run_monthly_selection_dataset.py",
        created_at="2026-05-02T00:00:00Z",
        duration_sec=1.25,
        seed=None,
        data_slices=(data_slice,),
        config={"config_path": "config.yaml.backtest", "config_hash": "abc"},
        params={"top_k": 20},
        metrics=metrics,
        gates={"data_gate": {"passed": True}},
        artifacts=artifacts,
        promotion={"production_eligible": False, "registry_status": "not_registered"},
    )
    return result, manifest


def test_data_slice_hash_is_stable_and_sensitive() -> None:
    first = _slice()
    same = _slice()
    changed = _slice(date_end="2026-05-01")

    assert first.slice_hash() == same.slice_hash()
    assert first.slice_hash() != changed.slice_hash()


def test_research_identity_slugifies_contract_fields() -> None:
    identity = make_research_identity(
        result_type="Monthly Selection Dataset",
        research_topic="Monthly Selection",
        research_config_id="Top 20 / U1",
        output_stem="Monthly Selection Top 20",
    )

    assert identity.result_type == "monthly_selection_dataset"
    assert identity.research_topic == "monthly_selection"
    assert identity.research_config_id == "top_20_u1"
    assert identity.output_stem == "monthly_selection_top_20"


def test_experiment_result_is_json_serializable(tmp_path: Path) -> None:
    result, _ = _result(tmp_path)

    payload = result.to_json_dict()

    assert payload["schema_version"] == "research_result_v1"
    assert payload["identity"]["research_config_id"] == "dataset_contract_v1"
    json.dumps(payload, ensure_ascii=False)


def test_append_experiment_result_writes_research_index(tmp_path: Path) -> None:
    result, _ = _result(tmp_path)

    path = append_experiment_result(tmp_path, result)

    assert path == tmp_path / "research_results.jsonl"
    line = path.read_text(encoding="utf-8").strip()
    assert json.loads(line)["result_id"] == result.result_id


def test_validate_manifest_accepts_standard_contract(tmp_path: Path) -> None:
    if validate_manifest is None:
        pytest.skip("scripts.validate_research_contracts 不可用")
    result, manifest = _result(tmp_path)
    write_research_manifest(manifest, result)

    errors = validate_manifest(manifest, root=Path.cwd())

    assert errors == []
