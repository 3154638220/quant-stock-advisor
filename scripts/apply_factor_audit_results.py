"""Apply factor audit results: mark IC IR < 0.2 factors as active=False in FACTOR_REGISTRY.

Reads weak_factors.csv from a W5 audit run and sets the ``active`` flag on the
corresponding FactorSpec entries.  The registry change is in-memory only by
default; pass ``--persist`` to write a governance log alongside.

Usage::

    python scripts/apply_factor_audit_results.py \
        --audit-dir data/results/factor_audit_2026_05_09_full \
        --persist
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.registry import FACTOR_REGISTRY, FactorSpec

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

WEAK_CSV = "weak_factors.csv"
GOVERNANCE_LOG = "factor_governance_log.jsonl"


def _feature_col_from_audit_name(audit_name: str) -> str:
    """Map an audit CSV 'factor' column value back to FactorSpec.feature_col."""
    col = audit_name
    if col.endswith("_z"):
        col = col[:-2]
    if col.startswith("is_missing_"):
        col = col[len("is_missing_"):]
    return col


def _build_feature_col_index() -> dict[str, FactorSpec]:
    """Index FACTOR_REGISTRY by feature_col for O(1) lookup."""
    idx: dict[str, FactorSpec] = {}
    for spec in FACTOR_REGISTRY.values():
        idx[spec.feature_col] = spec
    return idx


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Apply W5 factor audit results to FACTOR_REGISTRY")
    p.add_argument(
        "--audit-dir",
        required=True,
        help="Path to the audit output directory containing weak_factors.csv",
    )
    p.add_argument(
        "--persist",
        action="store_true",
        help="Write governance log to the audit directory",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions without modifying registry",
    )
    p.add_argument(
        "--skip-missing-flags",
        action="store_true",
        default=True,
        help="Skip is_missing_* flags (already excluded at pipeline level). Default: True.",
    )
    p.add_argument(
        "--apply-missing-flags",
        action="store_true",
        help="Also apply to is_missing_* flags (overrides --skip-missing-flags)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    audit_dir = Path(args.audit_dir)
    weak_path = audit_dir / WEAK_CSV
    if not weak_path.exists():
        logger.error("weak_factors.csv not found at %s", weak_path)
        return 1

    col_index = _build_feature_col_index()

    applied: list[dict] = []
    seen_specs: set[str] = set()
    skipped_missing: list[str] = []
    not_found: list[str] = []

    with open(weak_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("is_weak", "").strip().upper() != "TRUE":
                continue

            audit_name = row["factor"].strip()
            feature_col = _feature_col_from_audit_name(audit_name)

            # Skip is_missing_ flags unless explicitly requested
            if args.skip_missing_flags and not args.apply_missing_flags:
                if audit_name.startswith("is_missing_"):
                    skipped_missing.append(audit_name)
                    continue

            spec = col_index.get(feature_col)
            if spec is None:
                not_found.append(audit_name)
                continue

            # Deduplicate: raw + _z rows map to the same spec
            if spec.name in seen_specs:
                continue
            seen_specs.add(spec.name)

            ic_ir = float(row.get("ic_ir", 0))
            action = {
                "factor": audit_name,
                "feature_col": feature_col,
                "registry_name": spec.name,
                "family": spec.family,
                "ic_ir": ic_ir,
                "ic_mean": float(row.get("ic_mean", 0)),
                "recommendation": row.get("recommendation", ""),
                "was_active": spec.active,
            }

            if not args.dry_run:
                spec.active = False
            action["now_active"] = spec.active
            applied.append(action)

    # ── Report ──
    logger.info("Audit dir      : %s", audit_dir)
    logger.info("Weak factors   : %d (skipped is_missing_: %d)", len(applied) + len(skipped_missing), len(skipped_missing))
    logger.info("Applied        : %d%s", len(applied), " (dry-run)" if args.dry_run else "")
    if not_found:
        logger.warning("Not in registry: %d", len(not_found))
        for nf in not_found:
            logger.warning("  - %s", nf)

    # ── Per-family summary ──
    by_family: dict[str, int] = {}
    for a in applied:
        by_family[a["family"]] = by_family.get(a["family"], 0) + 1
    logger.info("By family:")
    for fam, cnt in sorted(by_family.items()):
        logger.info("  %-25s %d", fam, cnt)

    # ── Persist governance log ──
    if args.persist and not args.dry_run:
        log_path = audit_dir / GOVERNANCE_LOG
        timestamp = datetime.now(timezone.utc).isoformat()
        with open(log_path, "a", encoding="utf-8") as f:
            for a in applied:
                a["applied_at"] = timestamp
                f.write(json.dumps(a, ensure_ascii=False) + "\n")
        logger.info("Governance log : %s (%d entries)", log_path, len(applied))

        # Also write a human-readable summary
        summary_path = audit_dir / "governance_summary.md"
        lines = [
            f"# Factor Governance — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
            "",
            f"**Source audit**: `{weak_path}`",
            f"**Action**: Set `active=False` for {len(applied)} weak factors (IC IR < 0.2)",
            f"**Skipped** `is_missing_*` flags: {len(skipped_missing)} (handled at pipeline level)",
            "",
            "## Summary by family",
            "",
            "| Family | Demoted |",
            "|--------|---------|",
        ]
        for fam, cnt in sorted(by_family.items()):
            lines.append(f"| {fam} | {cnt} |")
        lines.extend([
            "",
            "## Demoted factors",
            "",
            "| Factor | Family | IC IR | IC Mean |",
            "|--------|--------|-------|---------|",
        ])
        for a in applied:
            lines.append(
                f"| {a['registry_name']} | {a['family']} "
                f"| {a['ic_ir']:.4f} | {a['ic_mean']:.4f} |"
            )
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        logger.info("Summary        : %s", summary_path)

    if not_found:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
