"""Tests for src/pipeline/hpo_utils.py - hyperparameter optimization utilities."""

from __future__ import annotations

import pandas as pd

from src.pipeline.hpo_utils import (
    XGBOOST_CLASSIFIER_SEARCH_SPACE,
    XGBOOST_RANKER_SEARCH_SPACE,
    XGBOOST_REGRESSOR_SEARCH_SPACE,
    _default_ranker_params,
    _default_regressor_params,
    _optuna_available,
    _suggest_params,
    _time_series_cv_folds,
)

# ── _optuna_available ──────────────────────────────────────────────────────


def test_optuna_available_returns_bool():
    result = _optuna_available()
    assert isinstance(result, bool)


# ── _suggest_params ────────────────────────────────────────────────────────


class _FakeTrial:
    def __init__(self):
        self._ints = {}
        self._floats = {}

    def suggest_int(self, name, low, high, step=1):
        self._ints[name] = (low, high, step)
        return (low + high) // 2

    def suggest_float(self, name, low, high, step=None):
        self._floats[name] = (low, high, step)
        return (low + high) / 2


def test_suggest_params_integer_step():
    trial = _FakeTrial()
    space = {"max_depth": {"low": 3, "high": 6, "step": 1}}
    params = _suggest_params(trial, space)
    assert "max_depth" in params
    assert isinstance(params["max_depth"], (int, float))
    assert 3 <= params["max_depth"] <= 6


def test_suggest_params_float_no_step():
    trial = _FakeTrial()
    space = {"learning_rate": {"low": 0.01, "high": 0.10}}
    params = _suggest_params(trial, space)
    assert "learning_rate" in params
    assert 0.01 <= params["learning_rate"] <= 0.10


def test_suggest_params_float_step():
    trial = _FakeTrial()
    space = {"reg_alpha": {"low": 0.0, "high": 10.0, "step": 0.5}}
    params = _suggest_params(trial, space)
    assert "reg_alpha" in params


def test_suggest_params_multi_param():
    trial = _FakeTrial()
    params = _suggest_params(trial, XGBOOST_RANKER_SEARCH_SPACE)
    assert len(params) == len(XGBOOST_RANKER_SEARCH_SPACE)
    assert "max_depth" in params
    assert "learning_rate" in params
    assert "n_estimators" in params


# ── _time_series_cv_folds ──────────────────────────────────────────────────


def _make_train_df(months: list[str]) -> pd.DataFrame:
    rows = []
    for i, m in enumerate(months):
        for s in range(5):
            rows.append({"signal_date": str(pd.Timestamp(m).date()), "symbol": f"{s:06d}", "val": float(i * 5 + s)})
    df = pd.DataFrame(rows)
    df["signal_date"] = pd.to_datetime(df["signal_date"])
    return df


def test_cv_folds_basic():
    months = [f"2023-{m:02d}-01" for m in range(1, 13)]
    df = _make_train_df(months)
    folds = _time_series_cv_folds(df, n_folds=3, gap=1)
    assert len(folds) >= 1
    for train_part, val_part in folds:
        assert not train_part.empty
        assert not val_part.empty


def test_cv_folds_too_few_months_falls_back():
    df = _make_train_df(["2023-01-01", "2023-02-01", "2023-03-01"])
    folds = _time_series_cv_folds(df, n_folds=5, gap=2)
    assert len(folds) == 1


def test_cv_folds_few_months():
    df = _make_train_df([f"2023-{m:02d}-01" for m in range(1, 7)])
    folds = _time_series_cv_folds(df, n_folds=2, gap=0)
    assert len(folds) >= 1


# ── Default params ─────────────────────────────────────────────────────────


def test_default_ranker_params():
    p = _default_ranker_params()
    assert p["n_estimators"] == 120
    assert p["max_depth"] == 3
    assert "learning_rate" in p


def test_default_regressor_params():
    p = _default_regressor_params()
    assert p["n_estimators"] == 80
    assert "learning_rate" in p


# ── Search space definitions ──────────────────────────────────────────────


def test_ranker_search_space_keys():
    expected = {"max_depth", "learning_rate", "min_child_weight", "subsample",
                "colsample_bytree", "n_estimators", "reg_alpha", "reg_lambda"}
    assert set(XGBOOST_RANKER_SEARCH_SPACE.keys()) == expected


def test_classifier_search_space_has_required_keys():
    for key in ("max_depth", "learning_rate", "n_estimators"):
        assert key in XGBOOST_CLASSIFIER_SEARCH_SPACE


def test_regressor_search_space_has_required_keys():
    for key in ("max_depth", "learning_rate", "n_estimators"):
        assert key in XGBOOST_REGRESSOR_SEARCH_SPACE


def test_search_spaces_consistent():
    """All three search spaces should define the same parameter keys."""
    r_keys = set(XGBOOST_RANKER_SEARCH_SPACE.keys())
    c_keys = set(XGBOOST_CLASSIFIER_SEARCH_SPACE.keys())
    reg_keys = set(XGBOOST_REGRESSOR_SEARCH_SPACE.keys())
    assert r_keys == c_keys == reg_keys
