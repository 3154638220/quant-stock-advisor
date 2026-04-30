import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "configs" / "promoted" / "promoted_registry.json"
ACTIVE_CONFIG_ID = "monthly_selection_u1_top20_indcap3_hardcap_baseline"


def load_registry() -> dict:
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))


def test_promoted_registry_records_active_monthly_selection_default() -> None:
    registry = load_registry()

    assert registry["current_state"]["has_promoted_research_candidates"] is True
    assert registry["current_state"]["active_promoted_config_id"] == ACTIVE_CONFIG_ID
    assert len(registry["promoted_configs"]) == 1
    promoted = registry["promoted_configs"][0]
    assert promoted["config_id"] == ACTIVE_CONFIG_ID
    assert promoted["production_method"]["candidate_pool_version"] == "U1_liquid_tradable"
    assert promoted["production_method"]["top_k"] == 20
    assert promoted["production_method"]["selection_policy"] == "industry_names_cap"
    assert promoted["production_method"]["max_industry_names"] == 3

    candidate_status = registry["research_candidate_status"]
    assert candidate_status
    active = [item for item in candidate_status if item["candidate_id"] == ACTIVE_CONFIG_ID]
    assert active and active[0]["production_eligible"] is True


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


def test_production_template_points_to_active_monthly_selection_config() -> None:
    registry = load_registry()
    cfg = yaml.safe_load((ROOT / "config.yaml.example").read_text(encoding="utf-8"))

    default_method = cfg["monthly_selection"]["default_method"]

    assert default_method["config_id"] == registry["current_state"]["active_promoted_config_id"]
    assert default_method["candidate_pool_version"] == "U1_liquid_tradable"
    assert default_method["top_k"] == 20
    assert default_method["selection_policy"] == "industry_names_cap"
    assert default_method["max_industry_names"] == 3
    assert cfg["signals"]["top_k"] == 20
    assert cfg["portfolio"]["industry_cap_count"] == 3


def test_active_promoted_config_snapshot_matches_registry_method() -> None:
    registry = load_registry()
    promoted = registry["promoted_configs"][0]
    snapshot_path = ROOT / promoted["config_path"]
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))

    assert snapshot["config_id"] == ACTIVE_CONFIG_ID
    assert snapshot["status"] == "active_promoted"
    assert snapshot["method"]["candidate_pool_version"] == promoted["production_method"]["candidate_pool_version"]
    assert snapshot["method"]["top_k"] == promoted["production_method"]["top_k"]
    assert snapshot["method"]["model"] == promoted["production_method"]["model"]
    assert snapshot["method"]["max_industry_names"] == promoted["production_method"]["max_industry_names"]
    assert snapshot["promotion_metrics"]["m8_gate_pass"] is True
