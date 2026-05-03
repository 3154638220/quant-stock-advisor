"""Lightweight research-run contracts for local experiment governance."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

from src.models.data_slice import hash_slice_spec

ArtifactKind = Literal["csv", "json", "jsonl", "md", "parquet", "model", "directory", "other"]
SCHEMA_VERSION = "research_result_v1"


def utc_now_iso() -> str:
    """Return a second-resolution UTC timestamp for manifests and indexes."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def json_sanitize(value: Any) -> Any:
    """Convert common pandas/numpy/path objects into JSON-serializable values."""
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): json_sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_sanitize(v) for v in value]
    if hasattr(value, "item"):
        try:
            return json_sanitize(value.item())
        except (TypeError, ValueError):
            pass
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except (TypeError, ValueError):
            pass
    return str(value)


def stable_hash(value: Any, *, prefix_len: int | None = None) -> str:
    payload = json.dumps(json_sanitize(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return digest[:prefix_len] if prefix_len else digest


def file_sha256(path: str | Path) -> str | None:
    p = Path(path)
    if not p.exists() or not p.is_file():
        return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def config_snapshot(
    *,
    config_path: str | Path | None,
    resolved_config: Mapping[str, Any] | None = None,
    sections: Sequence[str] = (),
) -> dict[str, Any]:
    """Build a compact config snapshot: path, file hash, and selected resolved sections."""
    out: dict[str, Any] = {
        "config_path": str(config_path) if config_path is not None else "",
        "config_hash": file_sha256(config_path) if config_path is not None else None,
    }
    if resolved_config is not None and sections:
        out["resolved_sections"] = {
            key: json_sanitize(resolved_config.get(key))
            for key in sections
            if key in resolved_config
        }
    return out


@dataclass(frozen=True)
class ResearchIdentity:
    result_type: str
    research_topic: str
    research_config_id: str
    output_stem: str
    canonical_config_name: str | None = None
    parent_result_id: str | None = None

    def to_json_dict(self) -> dict[str, Any]:
        return json_sanitize(asdict(self))

    @classmethod
    def from_json_dict(cls, payload: Mapping[str, Any]) -> "ResearchIdentity":
        return cls(
            result_type=str(payload.get("result_type") or ""),
            research_topic=str(payload.get("research_topic") or ""),
            research_config_id=str(payload.get("research_config_id") or ""),
            output_stem=str(payload.get("output_stem") or ""),
            canonical_config_name=(
                str(payload["canonical_config_name"]) if payload.get("canonical_config_name") is not None else None
            ),
            parent_result_id=str(payload["parent_result_id"]) if payload.get("parent_result_id") is not None else None,
        )


@dataclass(frozen=True)
class DataSlice:
    dataset_name: str
    source_tables: tuple[str, ...]
    date_start: str
    date_end: str
    asof_trade_date: str | None
    signal_date_col: str
    symbol_col: str
    candidate_pool_version: str | None
    rebalance_rule: str
    execution_mode: str
    label_return_mode: str | None
    feature_set_id: str | None
    feature_columns: tuple[str, ...] = ()
    label_columns: tuple[str, ...] = ()
    pit_policy: str = "unknown"
    config_path: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_spec(self) -> dict[str, Any]:
        return json_sanitize(asdict(self))

    def slice_hash(self) -> str:
        return hash_slice_spec(self.to_spec())

    def to_json_dict(self) -> dict[str, Any]:
        out = self.to_spec()
        out["slice_hash"] = self.slice_hash()
        return out

    @classmethod
    def from_json_dict(cls, payload: Mapping[str, Any]) -> "DataSlice":
        return cls(
            dataset_name=str(payload.get("dataset_name") or ""),
            source_tables=tuple(str(x) for x in payload.get("source_tables") or ()),
            date_start=str(payload.get("date_start") or ""),
            date_end=str(payload.get("date_end") or ""),
            asof_trade_date=str(payload["asof_trade_date"]) if payload.get("asof_trade_date") is not None else None,
            signal_date_col=str(payload.get("signal_date_col") or ""),
            symbol_col=str(payload.get("symbol_col") or ""),
            candidate_pool_version=(
                str(payload["candidate_pool_version"]) if payload.get("candidate_pool_version") is not None else None
            ),
            rebalance_rule=str(payload.get("rebalance_rule") or ""),
            execution_mode=str(payload.get("execution_mode") or ""),
            label_return_mode=str(payload["label_return_mode"]) if payload.get("label_return_mode") is not None else None,
            feature_set_id=str(payload["feature_set_id"]) if payload.get("feature_set_id") is not None else None,
            feature_columns=tuple(str(x) for x in payload.get("feature_columns") or ()),
            label_columns=tuple(str(x) for x in payload.get("label_columns") or ()),
            pit_policy=str(payload.get("pit_policy") or "unknown"),
            config_path=str(payload["config_path"]) if payload.get("config_path") is not None else None,
            extra=dict(payload.get("extra") or {}),
        )


@dataclass(frozen=True)
class ArtifactRef:
    name: str
    path: str
    kind: ArtifactKind
    required_for_promotion: bool = False
    description: str = ""

    def to_json_dict(self) -> dict[str, Any]:
        return json_sanitize(asdict(self))

    @classmethod
    def from_json_dict(cls, payload: Mapping[str, Any]) -> "ArtifactRef":
        return cls(
            name=str(payload.get("name") or ""),
            path=str(payload.get("path") or ""),
            kind=str(payload.get("kind") or "other"),  # type: ignore[arg-type]
            required_for_promotion=bool(payload.get("required_for_promotion", False)),
            description=str(payload.get("description") or ""),
        )


@dataclass(frozen=True)
class ExperimentResult:
    result_id: str
    identity: ResearchIdentity
    script_name: str
    command: str
    created_at: str
    duration_sec: float | None
    seed: int | None
    data_slices: tuple[DataSlice, ...]
    config: dict[str, Any]
    params: dict[str, Any]
    metrics: dict[str, Any]
    gates: dict[str, Any]
    artifacts: tuple[ArtifactRef, ...]
    promotion: dict[str, Any]
    notes: str = ""

    def to_json_dict(self) -> dict[str, Any]:
        return json_sanitize(
            {
                "schema_version": SCHEMA_VERSION,
                "result_id": self.result_id,
                "identity": self.identity.to_json_dict(),
                "script": {
                    "name": self.script_name,
                    "command": self.command,
                },
                "created_at": self.created_at,
                "duration_sec": self.duration_sec,
                "seed": self.seed,
                "data_slices": [x.to_json_dict() for x in self.data_slices],
                "config": self.config,
                "params": self.params,
                "metrics": self.metrics,
                "gates": self.gates,
                "artifacts": [x.to_json_dict() for x in self.artifacts],
                "promotion": self.promotion,
                "notes": self.notes,
            }
        )

    @classmethod
    def from_json_dict(cls, payload: Mapping[str, Any]) -> "ExperimentResult":
        script = payload.get("script") or {}
        return cls(
            result_id=str(payload.get("result_id") or ""),
            identity=ResearchIdentity.from_json_dict(payload.get("identity") or {}),
            script_name=str(script.get("name") or payload.get("script_name") or ""),
            command=str(script.get("command") or payload.get("command") or ""),
            created_at=str(payload.get("created_at") or ""),
            duration_sec=float(payload["duration_sec"]) if payload.get("duration_sec") is not None else None,
            seed=int(payload["seed"]) if payload.get("seed") is not None else None,
            data_slices=tuple(DataSlice.from_json_dict(x) for x in payload.get("data_slices") or ()),
            config=dict(payload.get("config") or {}),
            params=dict(payload.get("params") or {}),
            metrics=dict(payload.get("metrics") or {}),
            gates=dict(payload.get("gates") or {}),
            artifacts=tuple(ArtifactRef.from_json_dict(x) for x in payload.get("artifacts") or ()),
            promotion=dict(payload.get("promotion") or {}),
            notes=str(payload.get("notes") or ""),
        )


def build_result_id(
    identity: ResearchIdentity,
    data_slices: Sequence[DataSlice],
    metrics: Mapping[str, Any],
    *,
    created_at: str | None = None,
) -> str:
    """构建唯一结果 ID，含时间戳片段防止重复运行碰撞。"""
    primary_slice = data_slices[0].slice_hash()[:10] if data_slices else "noslice"
    metrics_hash = stable_hash(metrics, prefix_len=10)
    ts_fragment = (created_at or utc_now_iso())[:16].replace(":", "").replace("-", "")  # 如 "20260503T120000"
    return f"{identity.research_topic}:{identity.research_config_id}:{primary_slice}:{metrics_hash}:{ts_fragment}"


def write_research_manifest(path: str | Path, result: ExperimentResult, *, extra: Mapping[str, Any] | None = None) -> Path:
    payload = result.to_json_dict()
    if extra:
        payload.update(json_sanitize(dict(extra)))
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return p
