"""阶段二：截面 z 与 XGBoost 训练管线（无 DuckDB 时用合成面板）。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.artifacts import (
    BundleMetadata,
    InferenceConfig,
    load_bundle_metadata,
    publish_bundle_with_history,
    save_bundle_metadata,
    save_inference_config,
    save_sklearn_model,
    should_promote_by_quantile,
)
from src.models.inference import predict_xgboost_tree
from src.models.rank_score import apply_cross_section_z_by_date, cross_section_z_columns
from src.models.xtree.train import train_xgboost_panel


class _DummyTreeModel:
    def predict(self, X):
        return np.asarray(X[:, 0], dtype=np.float64)


def test_cross_section_z_single_day():
    df = pd.DataFrame(
        {
            "momentum": [0.1, 0.2, np.nan],
            "rsi": [30.0, 70.0, 50.0],
        }
    )
    z = cross_section_z_columns(df, ["momentum", "rsi"], rsi_mode="level")
    assert "z_momentum" in z.columns and "z_rsi" in z.columns
    assert z["z_momentum"].notna().sum() == 2


def test_apply_cross_section_z_by_date():
    rng = np.random.default_rng(0)
    rows = []
    for d in range(3):
        day = pd.Timestamp("2024-01-02") + pd.Timedelta(days=d)
        for s in range(4):
            rows.append(
                {
                    "trade_date": day,
                    "symbol": str(600000 + s).zfill(6),
                    "momentum": float(rng.normal()),
                    "rsi": float(rng.uniform(20, 80)),
                }
            )
    df = pd.DataFrame(rows)
    zdf = apply_cross_section_z_by_date(
        df,
        date_col="trade_date",
        raw_names=["momentum", "rsi"],
        rsi_mode="level",
    )
    assert len(zdf) == len(df)
    assert np.isfinite(zdf["z_momentum"].to_numpy()).all()


def test_train_xgboost_panel_smoke(tmp_path):
    pytest.importorskip("xgboost")
    rng = np.random.default_rng(1)
    rows = []
    for d in range(30):
        day = pd.Timestamp("2024-01-02") + pd.Timedelta(days=d)
        for s in range(8):
            m = float(rng.normal())
            r = float(rng.uniform(20, 80))
            y = float(0.01 * m + 0.001 * (r - 50) + rng.normal() * 0.02)
            rows.append(
                {
                    "trade_date": day,
                    "symbol": str(600000 + s).zfill(6),
                    "momentum": m,
                    "rsi": r,
                    "forward_ret_5d": y,
                }
            )
    panel = pd.DataFrame(rows)
    res = train_xgboost_panel(
        panel,
        raw_feature_names=["momentum", "rsi"],
        target_column="forward_ret_5d",
        rsi_mode="level",
        val_frac=0.25,
        xgb_params={"n_estimators": 20, "max_depth": 3},
        label_spec={"horizons": [5], "weights": [1.0], "scope": "cross_section_relative"},
        out_root=tmp_path,
        log_experiments=False,
    )
    assert res.bundle_dir.is_dir()
    assert res.metrics["val_mse"] >= 0.0
    assert "val_rank_ic" in res.metrics
    meta = load_bundle_metadata(res.bundle_dir)
    assert "label_spec" in meta.params


def test_train_xgboost_panel_regression_smoke(tmp_path):
    pytest.importorskip("xgboost")
    rng = np.random.default_rng(2)
    rows = []
    for d in range(30):
        day = pd.Timestamp("2024-01-02") + pd.Timedelta(days=d)
        for s in range(8):
            m = float(rng.normal())
            r = float(rng.uniform(20, 80))
            y = float(0.01 * m + 0.001 * (r - 50) + rng.normal() * 0.02)
            rows.append(
                {
                    "trade_date": day,
                    "symbol": str(600000 + s).zfill(6),
                    "momentum": m,
                    "rsi": r,
                    "forward_ret_5d": y,
                }
            )
    panel = pd.DataFrame(rows)
    res = train_xgboost_panel(
        panel,
        raw_feature_names=["momentum", "rsi"],
        target_column="forward_ret_5d",
        rsi_mode="level",
        xgboost_objective="regression",
        val_frac=0.25,
        xgb_params={"n_estimators": 20, "max_depth": 3},
        out_root=tmp_path,
        log_experiments=False,
    )
    assert res.bundle_dir.is_dir()
    assert res.metrics["val_mse"] >= 0.0


def test_predict_xgboost_tree_auto_flip_when_negative_rank_ic(tmp_path):
    bundle_dir = tmp_path / "xgb_bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    save_sklearn_model(_DummyTreeModel(), bundle_dir)
    save_inference_config(
        bundle_dir,
        InferenceConfig(
            feature_columns=["z_momentum"],
            target_column="forward_ret_5d",
            normalize="none",
            extra={"backend": "xgboost"},
        ),
    )
    save_bundle_metadata(
        bundle_dir,
        BundleMetadata(
            model_version="1.0.0",
            feature_version="tree_z_v1",
            model_type="xgboost_panel",
            backend="sklearn",
            training_seed=42,
            data_slice_hash="dummy_slice",
            content_hash="dummy_content",
            created_at="2026-01-01T00:00:00Z",
            metrics={"val_rank_ic": -0.25},
            params={},
        ),
    )
    df = pd.DataFrame({"z_momentum": [1.0, 2.0, 3.0]})
    pred = predict_xgboost_tree(bundle_dir, df)
    assert np.allclose(pred, [-1.0, -2.0, -3.0])


def test_should_promote_by_quantile_guard():
    ok, qv = should_promote_by_quantile(
        candidate_metric=0.04,
        history_metrics=[0.01, 0.02, 0.03, 0.05, 0.08],
        quantile=0.25,
        min_history=4,
    )
    assert ok is True
    assert qv > 0.0

    ok2, qv2 = should_promote_by_quantile(
        candidate_metric=-0.1,
        history_metrics=[0.01, 0.02, 0.03, 0.05, 0.08],
        quantile=0.25,
        min_history=4,
    )
    assert ok2 is False
    assert qv2 > -0.1


def test_publish_bundle_with_history(tmp_path):
    src = tmp_path / "src_bundle"
    src.mkdir(parents=True, exist_ok=True)
    save_sklearn_model(_DummyTreeModel(), src)
    save_inference_config(
        src,
        InferenceConfig(
            feature_columns=["z_momentum"],
            target_column="forward_ret_5d",
            normalize="none",
            extra={"backend": "xgboost"},
        ),
    )
    save_bundle_metadata(
        src,
        BundleMetadata(
            model_version="1.0.0",
            feature_version="tree_z_v1",
            model_type="xgboost_panel",
            backend="sklearn",
            training_seed=1,
            data_slice_hash="s",
            content_hash="c",
            created_at="2026-01-01T00:00:00Z",
            metrics={"val_rank_ic": 0.1},
            params={},
        ),
    )
    pub = tmp_path / "xgboost_panel_latest"
    hist = tmp_path / "xgboost_panel_history"
    meta1 = publish_bundle_with_history(
        source_bundle_dir=src,
        publish_dir=pub,
        history_root=hist,
        keep_recent=2,
    )
    assert pub.is_dir()
    assert (hist / "active_bundle.json").is_file()
    assert "active_version" in meta1

    src2 = tmp_path / "src_bundle2"
    src2.mkdir(parents=True, exist_ok=True)
    save_sklearn_model(_DummyTreeModel(), src2)
    save_inference_config(
        src2,
        InferenceConfig(
            feature_columns=["z_momentum"],
            target_column="forward_ret_5d",
            normalize="none",
            extra={"backend": "xgboost"},
        ),
    )
    save_bundle_metadata(
        src2,
        BundleMetadata(
            model_version="1.0.1",
            feature_version="tree_z_v1",
            model_type="xgboost_panel",
            backend="sklearn",
            training_seed=2,
            data_slice_hash="s2",
            content_hash="c2",
            created_at="2026-01-02T00:00:00Z",
            metrics={"val_rank_ic": 0.2},
            params={},
        ),
    )
    meta2 = publish_bundle_with_history(
        source_bundle_dir=src2,
        publish_dir=pub,
        history_root=hist,
        keep_recent=2,
    )
    assert meta2.get("previous_version") == meta1.get("active_version")
    loaded = load_bundle_metadata(pub)
    assert loaded.model_version == "1.0.1"
