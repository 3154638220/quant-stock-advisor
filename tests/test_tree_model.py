"""阶段二：截面 z 与 XGBoost 训练管线（无 DuckDB 时用合成面板）。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.rank_score import apply_cross_section_z_by_date, cross_section_z_columns
from src.models.xtree.train import train_xgboost_panel


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
        out_root=tmp_path,
        log_experiments=False,
    )
    assert res.bundle_dir.is_dir()
    assert res.metrics["val_mse"] >= 0.0
    assert "val_rank_ic" in res.metrics


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
