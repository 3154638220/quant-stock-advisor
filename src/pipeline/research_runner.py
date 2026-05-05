"""Shared research contract finalization: DataSlice, ExperimentResult, manifest.

Extracted from duplicated boilerplate across scripts/run_monthly_selection_*.py.
"""

from __future__ import annotations

import shlex
import sys
import time
from pathlib import Path
from typing import Any

from src.models.experiment import append_experiment_result
from src.models.research_contract import (
    ArtifactRef,
    DataSlice,
    ExperimentResult,
    build_result_id,
    config_snapshot,
    utc_now_iso,
    write_research_manifest,
)
from src.pipeline.cli_helpers import project_relative


def finalize_research_contract(
    *,
    identity: Any,
    script_path: str,
    started_at: float,
    config_source: str,
    config_raw: dict[str, Any],
    loaded_config_path: Path | None,
    experiments_dir: Path,
    paths_out: dict[str, Path],
    dataset_path: Path,
    data_slice_kwargs: dict[str, Any],
    metrics: dict[str, Any],
    gates: dict[str, Any] | None = None,
    params_extra: dict[str, Any] | None = None,
    seed: int | None = None,
    promotion_blocking: list[str] | None = None,
    notes: str = "",
    manifest_extra: dict[str, Any] | None = None,
    cli_args: dict[str, Any] | None = None,
    overrides: dict[str, Any] | None = None,
    artifact_paths_raw: list[str] | None = None,
) -> dict[str, Any]:
    """Build research contract (DataSlice, ArtifactRefs, ExperimentResult, manifest) and write outputs.

    Returns the ExperimentResult as a dict for printing/logging.
    """
    data_slice = DataSlice(**data_slice_kwargs)

    artifact_refs: list[ArtifactRef] = []
    for key, p in paths_out.items():
        rel = project_relative(p)
        if key == "manifest":
            artifact_refs.append(ArtifactRef("manifest_json", rel, "json", required_for_promotion=True))
        elif key == "doc":
            artifact_refs.append(ArtifactRef("report_md", rel, "md", required_for_promotion=True))
        else:
            artifact_refs.append(ArtifactRef(f"{key}_csv", rel, "csv"))

    gates_final = gates or {"governance_gate": {"passed": True, "manifest_schema": "research_result_v1"}}
    config_info = config_snapshot(
        config_path=loaded_config_path, resolved_config=config_raw,
        sections=("paths", "database", "signals", "portfolio", "backtest", "transaction_costs", "prefilter", "monthly_selection"),
    )
    config_info["config_path"] = config_source

    overrides_final = overrides or {}
    if cli_args:
        overrides_final = {
            key: value for key, value in {
                **{k: v for k, v in cli_args.items() if v},
            }.items() if value
        }

    result = ExperimentResult(
        result_id=build_result_id(identity, [data_slice], metrics),
        identity=identity,
        script_name=script_path,
        command=shlex.join([sys.executable, *sys.argv]),
        created_at=utc_now_iso(),
        duration_sec=round(time.perf_counter() - started_at, 6),
        seed=seed,
        data_slices=(data_slice,),
        config=config_info,
        params={"cli": cli_args or {}, "run_config": params_extra or {}, "overrides": overrides_final},
        metrics=metrics,
        gates=gates_final,
        artifacts=tuple(artifact_refs),
        promotion={
            "production_eligible": False,
            "registry_status": "not_registered",
            "blocking_reasons": promotion_blocking or ["research_only"],
        },
        notes=notes,
    )
    write_research_manifest(paths_out["manifest"], result, extra=manifest_extra or {})
    append_experiment_result(experiments_dir, result)

    artifact_paths = artifact_paths_raw or [
        project_relative(p) for key, p in paths_out.items()
        if key not in {"manifest", "doc"}
    ]
    return {
        "result": result,
        "artifact_paths": artifact_paths,
    }
