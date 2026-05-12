from __future__ import annotations

import json
from pathlib import Path

from scripts.check_lightgbm_retest_trigger import (
    build_trigger_report,
    load_active_promoted_method,
    promoted_feature_families,
)


ROOT = Path(__file__).resolve().parents[1]


def test_load_active_promoted_method_uses_registry_default() -> None:
    method = load_active_promoted_method(ROOT / "configs" / "promoted" / "promoted_registry.json")

    assert method["config_id"] == "monthly_selection_m8_indcap3_plus_quality"
    assert method["feature_families"] == [
        "industry_breadth",
        "fund_flow",
        "fundamental",
        "quality",
    ]
    assert promoted_feature_families(method) == [
        "price_volume",
        "industry_breadth",
        "fund_flow",
        "fundamental",
        "quality",
    ]


def test_build_trigger_report_counts_governed_active_features() -> None:
    report = build_trigger_report(
        registry_path=ROOT / "configs" / "promoted" / "promoted_registry.json",
        threshold=130,
    )

    assert report["ready_for_lightgbm_retest"] is False
    assert report["trigger_scope"] == "active_registry"
    assert report["active_registry_feature_count"] < 130
    assert report["features_to_threshold"] == 130 - report["active_registry_feature_count"]
    assert report["governance_actions_applied"] >= 1

    promoted_families = {row["family"]: row for row in report["active_promoted_family_summary"]}
    assert promoted_families["price_volume"]["active"] == 8
    assert promoted_families["quality"]["active"] <= promoted_families["quality"]["registered"]


def test_build_trigger_report_can_use_promoted_scope(tmp_path: Path) -> None:
    registry = {
        "current_state": {"active_promoted_config_id": "active"},
        "promoted_configs": [
            {
                "config_id": "active",
                "production_method": {
                    "feature_families": ["industry_breadth"],
                },
            }
        ],
    }
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(json.dumps(registry), encoding="utf-8")

    report = build_trigger_report(
        registry_path=registry_path,
        threshold=1,
        trigger_scope="active_promoted",
    )

    assert report["ready_for_lightgbm_retest"] is True
    assert report["active_promoted_families"] == ["price_volume", "industry_breadth"]
    assert report["trigger_feature_count"] == report["active_promoted_feature_count"]

