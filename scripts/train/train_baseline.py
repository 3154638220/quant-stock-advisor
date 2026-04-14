#!/usr/bin/env python3
"""
基线模型训练入口：线性（Ridge / ElasticNet）与树（RandomForest）。

用法（项目根目录）::

    python scripts/train/train_baseline.py --csv data/train.csv --kind ridge \\
      --features f1,f2 --target y --seed 42

输出：``data/models/baseline_<kind>_<run_id>/``（``bundle.json``、``inference_config.json``、``model.joblib``），
并追加 ``data/experiments/experiments.csv`` 与 ``experiments.jsonl``。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.baseline.train import train_baseline
from src.models.data_slice import apply_time_symbol_filter, normalize_slice_spec


def _parse_features(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def main() -> int:
    p = argparse.ArgumentParser(description="训练基线模型（sklearn）")
    p.add_argument("--csv", required=True, help="训练 CSV 路径")
    p.add_argument("--kind", choices=("ridge", "elasticnet", "random_forest"), default="ridge")
    p.add_argument("--features", required=True, help="逗号分隔特征列名")
    p.add_argument("--target", required=True, help="目标列名")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--normalize", choices=("none", "standard"), default="none")
    p.add_argument("--model-version", default="1.0.0")
    p.add_argument("--feature-version", default="v1")
    p.add_argument("--out-root", default="data/models")
    p.add_argument("--experiments-dir", default="data/experiments")
    p.add_argument("--symbols", default="", help="逗号分隔标的，空表示不过滤")
    p.add_argument("--date-start", default="", help="YYYY-MM-DD，空表示不过滤")
    p.add_argument("--date-end", default="", help="YYYY-MM-DD，空表示不过滤")
    p.add_argument("--extra-params", default="{}", help="JSON，如 {\"alpha\":1.0}")
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    feats = _parse_features(args.features)
    syms = [s.strip() for s in args.symbols.split(",") if s.strip()] or None
    d0 = args.date_start.strip() or None
    d1 = args.date_end.strip() or None
    df = apply_time_symbol_filter(df, symbols=syms, date_start=d0, date_end=d1)

    spec = normalize_slice_spec(
        symbols=syms,
        date_start=d0,
        date_end=d1,
        extra={"csv": str(Path(args.csv).resolve())},
    )
    extra_params = json.loads(args.extra_params)

    res = train_baseline(
        df,
        kind=args.kind,
        feature_columns=feats,
        target_column=args.target,
        model_version=args.model_version,
        feature_version=args.feature_version,
        training_seed=args.seed,
        test_size=args.test_size,
        normalize=args.normalize,
        slice_spec=spec,
        extra_params=extra_params,
        out_root=args.out_root,
        experiments_dir=args.experiments_dir,
    )
    print(json.dumps({"bundle_dir": str(res.bundle_dir), "metrics": res.metrics}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
