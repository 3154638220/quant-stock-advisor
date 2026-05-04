"""研究清单：统一的研究身份、结果记录与清单写入。

对 src/models/research_contract.py 和 scripts/research_identity.py
的封装，提供一体化的研究产物治理入口。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from src.models.experiment import append_backtest_result, append_experiment_result
from src.models.research_contract import (
    ArtifactRef,
    DataSlice,
    ExperimentResult,
    ResearchIdentity,
    build_result_id,
    config_snapshot,
    utc_now_iso,
    write_research_manifest,
)
from src.reporting.markdown_report import json_sanitize

# ── 重新导出常用类型 ──────────────────────────────────────────────────────

__all__ = [
    "ArtifactRef",
    "DataSlice",
    "ExperimentResult",
    "ResearchIdentity",
    "build_result_id",
    "config_snapshot",
    "utc_now_iso",
    "write_research_manifest",
    "append_backtest_result",
    "append_experiment_result",
    "json_sanitize",
    "slugify_token",
    "make_research_identity",
    "record_backtest_result",
    "record_experiment_result",
]

# ── Slug 化 ──────────────────────────────────────────────────────────────

import re


def slugify_token(value: Any) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "na"


# ── 研究身份工厂 ─────────────────────────────────────────────────────────

def make_research_identity(
    *,
    result_type: str,
    research_topic: str,
    research_config_id: str,
    output_stem: str,
    canonical_config_name: str | None = None,
    parent_result_id: str | None = None,
) -> ResearchIdentity:
    """统一构造 ResearchIdentity，避免各脚本重复 dataclass 构造。"""
    return ResearchIdentity(
        result_type=str(result_type),
        research_topic=str(research_topic),
        research_config_id=str(research_config_id),
        output_stem=str(output_stem),
        canonical_config_name=canonical_config_name,
        parent_result_id=parent_result_id,
    )


# ── 回测结果记录 ─────────────────────────────────────────────────────────

def record_backtest_result(
    *,
    experiments_dir: str | Path,
    identity: ResearchIdentity | dict[str, Any],
    data_slice: DataSlice,
    params: dict[str, Any],
    metrics: dict[str, Any],
    duration_sec: float,
    artifacts: list[ArtifactRef] | None = None,
    gates: dict[str, Any] | None = None,
    config_source: str = "",
    resolved_config: dict[str, Any] | None = None,
    command: str | None = None,
    notes: str = "",
) -> dict[str, Path]:
    """一站式回测结果记录：写入 JSONL + CSV 实验索引 + 研究清单。"""
    base_dir = Path(experiments_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(identity, dict):
        identity_obj = ResearchIdentity.from_json_dict(identity)
    else:
        identity_obj = identity

    rec_paths = append_backtest_result(
        base_dir=str(base_dir),
        params=json_sanitize(params),
        metrics=json_sanitize(metrics),
        duration_sec=duration_sec,
        bundle_dir=str(base_dir / identity_obj.output_stem),
        extra={"config_source": config_source},
    )

    # 写入研究清单
    gates_final = gates or {
        "governance_gate": {
            "passed": True,
            "manifest_schema": "research_result_v1",
        },
    }

    result = ExperimentResult(
        result_id=build_result_id(identity_obj, [data_slice], metrics),
        identity=identity_obj,
        script_name=Path(sys.argv[0]).name if command is None else command,
        command=command or " ".join(sys.argv),
        created_at=utc_now_iso(),
        duration_sec=round(duration_sec, 6),
        seed=None,
        data_slices=(data_slice,),
        config=config_snapshot(
            config_path=None,
            resolved_config=resolved_config,
            sections=("paths", "signals", "portfolio", "backtest", "transaction_costs", "prefilter"),
        ),
        params=params,
        metrics=metrics,
        gates=gates_final,
        artifacts=tuple(artifacts or ()),
        promotion={
            "production_eligible": False,
            "registry_status": "not_registered",
            "blocking_reasons": ["research_only"],
        },
        notes=notes,
    )

    manifest_path = base_dir / f"{identity_obj.output_stem}_manifest.json"
    write_research_manifest(manifest_path, result)
    append_experiment_result(str(base_dir), result)

    rec_paths["manifest"] = manifest_path
    return rec_paths


# ── 通用实验记录 ─────────────────────────────────────────────────────────

def record_experiment_result(
    *,
    experiments_dir: str | Path,
    identity: ResearchIdentity,
    data_slice: DataSlice,
    params: dict[str, Any],
    metrics: dict[str, Any],
    duration_sec: float,
    artifacts: list[ArtifactRef] | None = None,
    config_source: str = "",
    notes: str = "",
) -> dict[str, Path]:
    """一站式实验记录（非回测场景）。"""
    return record_backtest_result(
        experiments_dir=experiments_dir,
        identity=identity,
        data_slice=data_slice,
        params=params,
        metrics=metrics,
        duration_sec=duration_sec,
        artifacts=artifacts,
        config_source=config_source,
        notes=notes,
    )
