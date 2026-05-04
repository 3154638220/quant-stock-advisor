"""研究契约验证：manifest 结构校验与产物完整性检查。

从 scripts/validate_research_contracts.py 迁入 src/research/，
作为研究治理基础设施的一部分（per docs/plan-05-04.md A3）。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, List, Mapping

logger = logging.getLogger(__name__)

# ── Manifest 顶层必需键 ─────────────────────────────────────────────────
REQUIRED_TOP_KEYS = frozenset({
    "schema_version",
    "result_id",
    "identity",
    "data_slices",
    "metrics",
    "artifacts",
})

# Identity 子对象必需键
REQUIRED_IDENTITY_KEYS = frozenset({
    "result_type",
    "research_topic",
    "research_config_id",
    "output_stem",
})


def validate_manifest(manifest_path: str | Path, *, root: str | Path | None = None) -> List[str]:
    """验证研究清单文件的结构完整性与产物可达性。

    Parameters
    ----------
    manifest_path : Path
        清单 JSON 文件路径。
    root : Path or None
        产物路径的基准目录。若为 None 则仅做结构校验，不检查文件存在性。

    Returns
    -------
    List[str]
        错误信息列表；空列表表示校验通过。
    """
    errors: List[str] = []
    mp = Path(manifest_path)

    # 1. 文件存在且可读
    if not mp.exists():
        return [f"清单文件不存在: {mp}"]
    if not mp.is_file():
        return [f"清单路径不是文件: {mp}"]

    try:
        raw = mp.read_text(encoding="utf-8")
    except Exception as e:
        return [f"无法读取清单文件 {mp}: {e}"]

    # 2. 合法 JSON
    try:
        payload: dict[str, Any] = json.loads(raw)
    except json.JSONDecodeError as e:
        return [f"清单 JSON 解析失败 {mp}: {e}"]

    if not isinstance(payload, dict):
        return [f"清单顶层应为 JSON 对象，实际类型: {type(payload).__name__}"]

    # 3. 顶层必需键
    missing_top = REQUIRED_TOP_KEYS - set(payload.keys())
    if missing_top:
        errors.append(f"清单缺少顶层必需键: {sorted(missing_top)}")

    # 4. schema_version 校验
    sv = payload.get("schema_version")
    if sv != "research_result_v1":
        errors.append(f"schema_version 期望 'research_result_v1'，实际: {sv!r}")

    # 5. result_id 非空
    rid = payload.get("result_id")
    if not rid or not isinstance(rid, str) or not rid.strip():
        errors.append("result_id 缺失或为空")

    # 6. identity 子对象必需键
    identity = payload.get("identity")
    if isinstance(identity, dict):
        missing_id = REQUIRED_IDENTITY_KEYS - set(identity.keys())
        if missing_id:
            errors.append(f"identity 缺少必需键: {sorted(missing_id)}")
    else:
        errors.append(f"identity 应为 JSON 对象，实际类型: {type(identity).__name__}")

    # 7. data_slices 非空数组
    slices = payload.get("data_slices")
    if not isinstance(slices, list) or len(slices) == 0:
        errors.append("data_slices 缺失或为空数组")

    # 8. metrics 非空对象
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict) or len(metrics) == 0:
        errors.append("metrics 缺失或为空对象")

    # 9. artifacts 数组（可为空）
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, list):
        errors.append(f"artifacts 应为数组，实际类型: {type(artifacts).__name__}")

    # 10. 产物文件可达性检查（需要 root）
    if root is not None and isinstance(artifacts, list):
        base = Path(root)
        for i, art in enumerate(artifacts):
            if not isinstance(art, dict):
                errors.append(f"artifacts[{i}] 不是 JSON 对象")
                continue
            art_path = art.get("path", "")
            if art_path:
                full = base / art_path if not Path(art_path).is_absolute() else Path(art_path)
                if not full.exists():
                    errors.append(f"产物文件不存在: {full} (artifact '{art.get('name', f'#{i}')}')")

    return errors
