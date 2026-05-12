#!/usr/bin/env python3
"""Check whether the active factor library is large enough to rerun LightGBM.

I6 uses a 130+ active-feature threshold before spending time on the W3
LightGBM regression re-evaluation.  This script makes that gate explicit and
keeps the count aligned with the factor governance log.
"""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.registry import (  # noqa: E402
    FACTOR_REGISTRY,
    apply_factor_governance_log,
    reset_all_active,
)
from src.reporting.markdown_report import json_sanitize  # noqa: E402

DEFAULT_REGISTRY = PROJECT_ROOT / "configs" / "promoted" / "promoted_registry.json"
DEFAULT_THRESHOLD = 130
DEFAULT_TRIGGER_SCOPE = "active_registry"


@contextmanager
def preserved_registry_active_state() -> Iterator[None]:
    state = {name: spec.active for name, spec in FACTOR_REGISTRY.items()}
    try:
        yield
    finally:
        for name, active in state.items():
            if name in FACTOR_REGISTRY:
                FACTOR_REGISTRY[name].active = active


def load_active_promoted_method(registry_path: Path = DEFAULT_REGISTRY) -> dict:
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    active_id = registry.get("current_state", {}).get("active_promoted_config_id")
    for item in registry.get("promoted_configs", []):
        if item.get("config_id") == active_id:
            method = dict(item.get("production_method", {}) or {})
            method["config_id"] = active_id
            return method
    raise ValueError(f"active promoted config not found: {active_id}")


def promoted_feature_families(method: dict) -> list[str]:
    families = ["price_volume"]
    for family in method.get("feature_families", []) or []:
        family = str(family).strip()
        if family and family not in families:
            families.append(family)
    return families


def apply_governance(governance_log: Path | None = None) -> int:
    reset_all_active()
    return len(apply_factor_governance_log(governance_log))


def family_summary(families: Sequence[str] | None = None) -> list[dict]:
    family_filter = set(families) if families is not None else None
    rows: list[dict] = []
    for family in sorted({spec.family for spec in FACTOR_REGISTRY.values()}):
        if family_filter is not None and family not in family_filter:
            continue
        specs = [spec for spec in FACTOR_REGISTRY.values() if spec.family == family]
        rows.append(
            {
                "family": family,
                "registered": len(specs),
                "active": sum(1 for spec in specs if spec.active),
                "inactive": sum(1 for spec in specs if not spec.active),
            }
        )
    return rows


def build_trigger_report(
    *,
    registry_path: Path = DEFAULT_REGISTRY,
    governance_log: Path | None = None,
    threshold: int = DEFAULT_THRESHOLD,
    trigger_scope: str = DEFAULT_TRIGGER_SCOPE,
) -> dict:
    if trigger_scope not in {"active_registry", "active_promoted"}:
        raise ValueError(f"unsupported trigger_scope: {trigger_scope}")

    method = load_active_promoted_method(registry_path)
    families = promoted_feature_families(method)

    with preserved_registry_active_state():
        governance_actions = apply_governance(governance_log)
        registry_active_count = sum(1 for spec in FACTOR_REGISTRY.values() if spec.active)
        promoted_active_count = sum(
            1 for spec in FACTOR_REGISTRY.values() if spec.active and spec.family in set(families)
        )
        registry_registered_count = len(FACTOR_REGISTRY)
        trigger_count = registry_active_count if trigger_scope == "active_registry" else promoted_active_count
        ready = trigger_count >= threshold

        return {
            "threshold": threshold,
            "trigger_scope": trigger_scope,
            "ready_for_lightgbm_retest": ready,
            "features_to_threshold": max(threshold - trigger_count, 0),
            "trigger_feature_count": trigger_count,
            "active_registry_feature_count": registry_active_count,
            "registered_registry_feature_count": registry_registered_count,
            "active_promoted_feature_count": promoted_active_count,
            "active_promoted_config_id": method.get("config_id", ""),
            "active_promoted_families": families,
            "governance_actions_applied": governance_actions,
            "registry_family_summary": family_summary(),
            "active_promoted_family_summary": family_summary(families),
            "recommendation": (
                "Run LightGBM regression re-evaluation."
                if ready
                else "Do not rerun LightGBM yet; active feature count is below threshold."
            ),
        }


def render_markdown(report: dict) -> str:
    def pct_ready() -> str:
        threshold = int(report["threshold"])
        count = int(report["trigger_feature_count"])
        return f"{count}/{threshold}"

    rows = "\n".join(
        "| {family} | {active} | {inactive} | {registered} |".format(**row)
        for row in report["registry_family_summary"]
    )
    promoted_rows = "\n".join(
        "| {family} | {active} | {inactive} | {registered} |".format(**row)
        for row in report["active_promoted_family_summary"]
    )
    return f"""# LightGBM Retest Trigger Check

- Active promoted config: `{report["active_promoted_config_id"]}`
- Trigger scope: `{report["trigger_scope"]}`
- Trigger count / threshold: **{pct_ready()}**
- Ready: **{report["ready_for_lightgbm_retest"]}**
- Features to threshold: {report["features_to_threshold"]}
- Governance actions applied: {report["governance_actions_applied"]}

## Recommendation

{report["recommendation"]}

## Active Promoted Families

{", ".join(f"`{x}`" for x in report["active_promoted_families"])}

## Active Promoted Feature Count

| Family | Active | Inactive | Registered |
|--------|-------:|---------:|-----------:|
{promoted_rows}

## Full Registry Feature Count

| Family | Active | Inactive | Registered |
|--------|-------:|---------:|-----------:|
{rows}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check I6 LightGBM retest feature-count trigger")
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY), help="Promoted registry JSON path")
    parser.add_argument("--governance-log", default="", help="Optional factor governance log path")
    parser.add_argument("--threshold", type=int, default=DEFAULT_THRESHOLD)
    parser.add_argument(
        "--trigger-scope",
        choices=["active_registry", "active_promoted"],
        default=DEFAULT_TRIGGER_SCOPE,
        help="Feature count used for the threshold decision",
    )
    parser.add_argument("--format", choices=["json", "markdown"], default="markdown")
    parser.add_argument("--output", default="", help="Optional output file")
    parser.add_argument("--fail-on-ready", action="store_true", help="Return exit code 2 when threshold is reached")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    governance_log = Path(args.governance_log) if args.governance_log.strip() else None
    report = build_trigger_report(
        registry_path=Path(args.registry),
        governance_log=governance_log,
        threshold=args.threshold,
        trigger_scope=args.trigger_scope,
    )
    text = (
        json.dumps(json_sanitize(report), ensure_ascii=False, indent=2)
        if args.format == "json"
        else render_markdown(report)
    )
    if args.output.strip():
        Path(args.output).write_text(text + ("\n" if not text.endswith("\n") else ""), encoding="utf-8")
    else:
        print(text)
    if args.fail_on_ready and report["ready_for_lightgbm_retest"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

