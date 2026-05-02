#!/usr/bin/env python3
"""Validate standard research-result manifests and promoted evidence links."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.research_contract import SCHEMA_VERSION, ArtifactRef, DataSlice

ARTIFACT_KINDS = {"csv", "json", "jsonl", "md", "parquet", "model", "directory", "other"}
SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
REQUIRED_MANIFEST_KEYS = {
    "schema_version",
    "result_id",
    "identity",
    "script",
    "data_slices",
    "config",
    "params",
    "metrics",
    "gates",
    "artifacts",
    "promotion",
}
REQUIRED_IDENTITY_KEYS = {"result_type", "research_topic", "research_config_id", "output_stem"}
PROMOTED_EVIDENCE_FIELDS = ("config_path", "full_backtest_report", "oos_report", "state_slice_report", "boundary_report")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate research-result contracts")
    p.add_argument("--manifest", action="append", default=[], help="Manifest JSON path. May be repeated.")
    p.add_argument("--manifest-glob", default="", help="Optional glob such as data/results/*_manifest.json")
    p.add_argument("--root", type=Path, default=PROJECT_ROOT)
    p.add_argument("--promoted-registry", type=Path, default=PROJECT_ROOT / "configs/promoted/promoted_registry.json")
    p.add_argument("--skip-promoted-registry", action="store_true")
    return p.parse_args()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_path(raw: str, *, root: Path) -> Path:
    p = Path(raw)
    return p if p.is_absolute() else root / p


def _is_marked_external(entry: dict[str, Any], field: str, value: str) -> bool:
    markers = set(str(x) for x in entry.get("historical_external_evidence") or ())
    return field in markers or value.startswith(("external:", "http://", "https://"))


def _validate_slug(label: str, value: Any, errors: list[str]) -> None:
    text = str(value or "")
    if not text:
        errors.append(f"{label} is empty")
    elif not SLUG_RE.match(text):
        errors.append(f"{label} is not slug-like: {text!r}")


def validate_manifest(path: Path, *, root: Path) -> list[str]:
    errors: list[str] = []
    if not path.exists():
        return [f"{path}: manifest file does not exist"]
    try:
        payload = _load_json(path)
    except json.JSONDecodeError as exc:
        return [f"{path}: invalid JSON: {exc}"]
    if not isinstance(payload, dict):
        return [f"{path}: manifest must be a JSON object"]

    missing = sorted(REQUIRED_MANIFEST_KEYS - set(payload))
    if missing:
        errors.append(f"{path}: missing top-level keys: {missing}")
    if payload.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"{path}: schema_version must be {SCHEMA_VERSION!r}")
    if not str(payload.get("result_id") or "").strip():
        errors.append(f"{path}: result_id is empty")

    identity = payload.get("identity")
    if not isinstance(identity, dict):
        errors.append(f"{path}: identity must be an object")
    else:
        missing_identity = sorted(REQUIRED_IDENTITY_KEYS - set(identity))
        if missing_identity:
            errors.append(f"{path}: identity missing keys: {missing_identity}")
        for key in ("result_type", "research_topic", "research_config_id", "output_stem"):
            _validate_slug(f"{path}: identity.{key}", identity.get(key), errors)

    script = payload.get("script")
    if not isinstance(script, dict) or not script.get("name") or not script.get("command"):
        errors.append(f"{path}: script.name and script.command are required")

    data_slices = payload.get("data_slices")
    if not isinstance(data_slices, list) or not data_slices:
        errors.append(f"{path}: data_slices must be a non-empty list")
    else:
        for idx, item in enumerate(data_slices):
            if not isinstance(item, dict):
                errors.append(f"{path}: data_slices[{idx}] must be an object")
                continue
            if not item.get("slice_hash"):
                errors.append(f"{path}: data_slices[{idx}].slice_hash is required")
                continue
            try:
                expected = DataSlice.from_json_dict(item).slice_hash()
            except (TypeError, ValueError) as exc:
                errors.append(f"{path}: data_slices[{idx}] cannot be parsed: {exc}")
                continue
            if str(item.get("slice_hash")) != expected:
                errors.append(f"{path}: data_slices[{idx}].slice_hash does not match contract fields")

    for key in ("config", "params", "metrics", "gates", "promotion"):
        if not isinstance(payload.get(key), dict):
            errors.append(f"{path}: {key} must be an object")

    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        errors.append(f"{path}: artifacts must be a non-empty list")
    else:
        seen_names: set[str] = set()
        for idx, item in enumerate(artifacts):
            if not isinstance(item, dict):
                errors.append(f"{path}: artifacts[{idx}] must be an object")
                continue
            artifact = ArtifactRef.from_json_dict(item)
            if not artifact.name:
                errors.append(f"{path}: artifacts[{idx}].name is empty")
            if artifact.name in seen_names:
                errors.append(f"{path}: duplicate artifact name: {artifact.name}")
            seen_names.add(artifact.name)
            if artifact.kind not in ARTIFACT_KINDS:
                errors.append(f"{path}: artifacts[{idx}].kind is invalid: {artifact.kind!r}")
            if not artifact.path:
                errors.append(f"{path}: artifacts[{idx}].path is empty")
                continue
            artifact_path = _resolve_path(artifact.path, root=root)
            if not artifact_path.exists():
                errors.append(f"{path}: artifact does not exist: {artifact.path}")

    return errors


def validate_promoted_registry(path: Path, *, root: Path) -> list[str]:
    errors: list[str] = []
    if not path.exists():
        return [f"{path}: promoted registry does not exist"]
    try:
        registry = _load_json(path)
    except json.JSONDecodeError as exc:
        return [f"{path}: invalid JSON: {exc}"]
    if not isinstance(registry, dict):
        return [f"{path}: promoted registry must be a JSON object"]

    required_fields = set(registry.get("required_entry_fields") or ())
    promoted_configs = registry.get("promoted_configs") or []
    if not isinstance(promoted_configs, list):
        return [f"{path}: promoted_configs must be a list"]
    for idx, entry in enumerate(promoted_configs):
        if not isinstance(entry, dict):
            errors.append(f"{path}: promoted_configs[{idx}] must be an object")
            continue
        config_id = entry.get("config_id") or f"#{idx}"
        missing = sorted(required_fields - set(entry))
        if missing:
            errors.append(f"{path}: promoted config {config_id} missing fields: {missing}")
        for field in PROMOTED_EVIDENCE_FIELDS:
            value = str(entry.get(field) or "").strip()
            if not value:
                errors.append(f"{path}: promoted config {config_id} has empty {field}")
                continue
            if _is_marked_external(entry, field, value):
                continue
            if not _resolve_path(value, root=root).exists():
                errors.append(f"{path}: promoted config {config_id} {field} does not exist: {value}")
    return errors


def _expand_manifests(raw_paths: Iterable[str], manifest_glob: str, *, root: Path) -> list[Path]:
    paths = [Path(p) for p in raw_paths]
    if manifest_glob:
        paths.extend(sorted(root.glob(manifest_glob)))
    out: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        p = path if path.is_absolute() else root / path
        if p not in seen:
            out.append(p)
            seen.add(p)
    return out


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    errors: list[str] = []
    for manifest_path in _expand_manifests(args.manifest, args.manifest_glob, root=root):
        errors.extend(validate_manifest(manifest_path, root=root))
    if not args.skip_promoted_registry:
        errors.extend(validate_promoted_registry(args.promoted_registry, root=root))

    if errors:
        for error in errors:
            print(f"[contract] ERROR {error}", file=sys.stderr)
        return 1
    print("[contract] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
