import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "configs" / "promoted" / "promoted_registry.json"


def load_registry() -> dict:
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))


def test_promoted_registry_records_no_active_research_candidate() -> None:
    registry = load_registry()

    assert registry["current_state"]["has_promoted_research_candidates"] is False
    assert registry["current_state"]["active_promoted_config_id"] is None
    assert registry["promoted_configs"] == []

    candidate_status = registry["research_candidate_status"]
    assert candidate_status
    assert all(item["production_eligible"] is False for item in candidate_status)


def test_promoted_registry_required_entry_fields_cover_r5_gate() -> None:
    registry = load_registry()

    required = set(registry["required_entry_fields"])
    assert {
        "config_id",
        "config_path",
        "promotion_date",
        "full_backtest_report",
        "daily_proxy_report",
        "oos_report",
        "state_slice_report",
        "boundary_report",
        "owner_decision",
    }.issubset(required)


def test_production_template_has_no_r2b_or_r3_research_candidate_ids() -> None:
    text = (ROOT / "config.yaml.example").read_text(encoding="utf-8")

    forbidden_markers = [
        "U2_",
        "U3_",
        "EDGE_GATED",
        "DUAL_V1",
        "UPSIDE_C",
        "R2B",
        "BASELINE_S2_FIXED",
    ]
    for marker in forbidden_markers:
        assert marker not in text


def test_production_template_does_not_promote_weekly_kdj_interaction_weights() -> None:
    cfg = yaml.safe_load((ROOT / "config.yaml.example").read_text(encoding="utf-8"))
    weights = cfg["signals"]["composite_extended"]

    assert all(not str(key).startswith("weekly_kdj") for key in weights)
