"""模型训练可复现性：同一数据切片得到稳定内容哈希与（在 CPU 上）稳定指标。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.models.baseline.train import train_baseline
from src.models.data_slice import combined_data_fingerprint, hash_dataframe_content, normalize_slice_spec
from src.models.timeseries.train import train_timeseries


def _synth_panel(n_sym: int = 4, n_days: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for d in range(n_days):
        day = pd.Timestamp("2024-01-01") + pd.Timedelta(days=d)
        for s in range(n_sym):
            sym = str(600000 + s).zfill(6)
            f1 = float(rng.normal())
            f2 = float(rng.normal())
            fwd = float(0.3 * f1 + 0.2 * f2 + rng.normal() * 0.01)
            rows.append(
                {
                    "symbol": sym,
                    "trade_date": day,
                    "f1": f1,
                    "f2": f2,
                    "fwd": fwd,
                }
            )
    return pd.DataFrame(rows)


def test_content_hash_stable():
    df = _synth_panel()
    h1 = hash_dataframe_content(df, columns=["f1", "f2", "fwd"])
    h2 = hash_dataframe_content(df.copy(), columns=["f1", "f2", "fwd"])
    assert h1 == h2


def test_combined_fingerprint_with_slice_spec():
    df = _synth_panel()
    spec = normalize_slice_spec(symbols=["600000"], date_start="2024-01-05", date_end="2024-01-20")
    fp1 = combined_data_fingerprint(df, slice_spec=spec, content_columns=["f1", "f2", "fwd"])
    fp2 = combined_data_fingerprint(df, slice_spec=spec, content_columns=["f1", "f2", "fwd"])
    assert fp1["slice_hash"] == fp2["slice_hash"]
    assert fp1["content_hash"] == fp2["content_hash"]


def test_baseline_repro_metrics(tmp_path: Path):
    df = _synth_panel()
    spec = normalize_slice_spec(extra={"test": "repro"})
    out1 = tmp_path / "m1"
    out2 = tmp_path / "m2"
    exp = tmp_path / "exp"
    r1 = train_baseline(
        df,
        kind="ridge",
        feature_columns=["f1", "f2"],
        target_column="fwd",
        training_seed=7,
        test_size=0.25,
        slice_spec=spec,
        out_root=out1,
        experiments_dir=exp,
        log_experiments=False,
    )
    r2 = train_baseline(
        df,
        kind="ridge",
        feature_columns=["f1", "f2"],
        target_column="fwd",
        training_seed=7,
        test_size=0.25,
        slice_spec=spec,
        out_root=out2,
        experiments_dir=exp,
        log_experiments=False,
    )
    assert r1.data_slice_hash == r2.data_slice_hash
    assert r1.content_hash == r2.content_hash
    for k in r1.metrics:
        assert r1.metrics[k] == pytest.approx(r2.metrics[k], rel=0, abs=1e-9)


def test_timeseries_repro_metrics_cpu(tmp_path: Path):
    df = _synth_panel(n_sym=5, n_days=50)
    spec = normalize_slice_spec(extra={"ts": "repro"})
    out1 = tmp_path / "t1"
    out2 = tmp_path / "t2"
    kwargs = dict(
        kind="lstm",
        feature_columns=["f1", "f2"],
        target_column="fwd",
        seq_len=8,
        training_seed=11,
        test_size=0.2,
        epochs=5,
        batch_size=32,
        lr=1e-2,
        device="cpu",
        hidden=16,
        num_layers=1,
        slice_spec=spec,
        log_experiments=False,
    )
    r1 = train_timeseries(df, out_root=out1, experiments_dir=tmp_path / "e1", **kwargs)
    r2 = train_timeseries(df, out_root=out2, experiments_dir=tmp_path / "e2", **kwargs)
    assert r1.data_slice_hash == r2.data_slice_hash
    assert r1.content_hash == r2.content_hash
    for k in r1.metrics:
        assert r1.metrics[k] == pytest.approx(r2.metrics[k], rel=0, abs=1e-5)


def test_experiment_log_json_roundtrip(tmp_path: Path):
    from src.models.experiment import append_experiment_jsonl, build_experiment_record

    rec = build_experiment_record(
        run_id="abc",
        model_type="baseline_ridge",
        duration_sec=1.0,
        seed=1,
        data_slice_hash="s",
        content_hash="c",
        params={"a": 1},
        metrics={"m": 2.0},
        bundle_dir=tmp_path / "b",
    )
    fp = append_experiment_jsonl(tmp_path, rec)
    line = fp.read_text(encoding="utf-8").strip()
    assert json.loads(line)["run_id"] == "abc"
