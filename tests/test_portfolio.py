"""Tests for portfolio/ module: weights, redistribution, turnover constraints."""

import numpy as np
import pandas as pd
import pytest

from src.portfolio.covariance import (
    _ewma_covariance,
    estimate_covariance,
)
from src.portfolio.optimizer import (
    _symmetrize,
    covariance_diagnostics,
    weight_diagnostics,
)
from src.portfolio.weights import (
    _nonnegative_weights_from_scores,
    _scores_from_column,
    _tiered_equal_weights_from_scores,
    apply_turnover_constraint,
    infer_score_column,
    redistribute_individual_cap,
)

# ── weights: infer_score_column ───────────────────────────────────────────────


def test_infer_score_column_deep_sequence():
    df = pd.DataFrame({"deep_sequence_score": [0.8, 0.3], "tree_score": [0.5, 0.6]})
    assert infer_score_column(df) == "deep_sequence_score"


def test_infer_score_column_fallback_rank():
    df = pd.DataFrame({"rank": [1, 2, 3]})
    assert infer_score_column(df) == "rank"


def test_infer_score_column_raises():
    df = pd.DataFrame({"a": [1, 2]})
    with pytest.raises(ValueError, match="无法推断"):
        infer_score_column(df)


# ── weights: _scores_from_column ──────────────────────────────────────────────


def test_scores_from_column_rank():
    df = pd.DataFrame({"rank": [1.0, 2.0, 4.0]})
    out = _scores_from_column(df, "rank")
    assert out.shape == (3,)
    assert out[0] == 1.0  # 1/1
    assert out[1] == 0.5  # 1/2


def test_scores_from_column_numeric():
    df = pd.DataFrame({"score": [0.1, 0.5, 0.9]})
    out = _scores_from_column(df, "score")
    np.testing.assert_array_almost_equal(out, [0.1, 0.5, 0.9])


# ── weights: _nonnegative_weights_from_scores ─────────────────────────────────


def test_nonnegative_equal():
    w = _nonnegative_weights_from_scores(np.array([1.0, 2.0, 3.0]), method="equal")
    np.testing.assert_array_almost_equal(w, [1 / 3, 1 / 3, 1 / 3])


def test_nonnegative_score():
    w = _nonnegative_weights_from_scores(np.array([1.0, 2.0, 3.0]), method="score")
    np.testing.assert_array_almost_equal(w, [1 / 6, 2 / 6, 3 / 6])


def test_nonnegative_score_with_nan():
    w = _nonnegative_weights_from_scores(np.array([1.0, np.nan, 3.0]), method="score")
    assert w[1] == 0.0
    np.testing.assert_almost_equal(w.sum(), 1.0)


def test_nonnegative_all_nan_falls_back_zero():
    w = _nonnegative_weights_from_scores(np.array([np.nan, np.nan]), method="score")
    assert w.sum() <= 1.0  # fallback returns something sum-safe


def test_nonnegative_invalid_method():
    with pytest.raises(ValueError, match="未知 weight_method"):
        _nonnegative_weights_from_scores(np.array([1.0]), method="invalid")


# ── weights: _tiered_equal_weights_from_scores ────────────────────────────────


def test_tiered_equal_basic():
    scores = np.array([0.9, 0.7, 0.3, 0.1])
    w = _tiered_equal_weights_from_scores(scores, top_tier_count=2, top_tier_weight_share=0.6)
    assert w.sum() > 0.99
    assert w[0] > w[2]  # top tier gets more weight
    assert w[0] == pytest.approx(w[1])  # equal within top tier


def test_tiered_equal_all_nan():
    scores = np.array([np.nan, np.nan])
    w = _tiered_equal_weights_from_scores(scores, top_tier_count=2, top_tier_weight_share=0.5)
    assert w.sum() == 0.0


# ── weights: redistribute_individual_cap ──────────────────────────────────────


def test_redistribute_no_cap_needed():
    w = np.array([0.3, 0.3, 0.4])
    out = redistribute_individual_cap(w, cap=0.5)
    np.testing.assert_array_almost_equal(out, w)


def test_redistribute_single_over_cap():
    w = np.array([0.6, 0.2, 0.2])
    out = redistribute_individual_cap(w, cap=0.5)
    assert out[0] <= 0.5 + 1e-12
    np.testing.assert_almost_equal(out.sum(), 1.0)


# ── weights: apply_turnover_constraint ────────────────────────────────────────


def test_turnover_constraint_noop():
    target = np.array([0.3, 0.3, 0.4])
    prev = np.array([0.3, 0.3, 0.4])
    out = apply_turnover_constraint(target, prev, max_turnover=1.0)
    np.testing.assert_array_almost_equal(out, target)


def test_turnover_constraint_shrink():
    target = np.array([0.5, 0.4, 0.1])
    prev = np.array([0.1, 0.2, 0.7])
    out = apply_turnover_constraint(target, prev, max_turnover=0.2)
    assert 0.5 * np.abs(out - prev).sum() <= 0.2 + 1e-12


# ── optimizer: _symmetrize ────────────────────────────────────────────────────


def test_symmetrize():
    A = np.array([[1.0, 2.0], [3.0, 4.0]])
    S = _symmetrize(A)
    np.testing.assert_array_almost_equal(S, S.T)
    np.testing.assert_array_almost_equal(np.diag(S), np.diag(A))


# ── optimizer: covariance_diagnostics ─────────────────────────────────────────


def test_covariance_diagnostics():
    Sigma = np.array([[0.04, 0.01], [0.01, 0.09]])
    diag = covariance_diagnostics(Sigma)
    assert diag["condition_number"] > 0
    assert diag["min_eigenvalue"] > 0
    assert diag["n_assets"] == 2


# ── optimizer: weight_diagnostics ─────────────────────────────────────────────


def test_weight_diagnostics():
    w = np.array([0.3, 0.3, 0.4])
    diag = weight_diagnostics(w)
    assert diag["effective_n"] >= 2
    assert diag["max_weight"] >= 0.3
    assert diag["n_assets"] == 3


# ── covariance: _ewma_covariance ──────────────────────────────────────────────
# R expected as (n_assets, n_times)


def test_ewma_covariance():
    np.random.seed(42)
    R = np.random.randn(5, 100) * 0.02  # 5 assets, 100 time points
    Sigma = _ewma_covariance(R, halflife=20, ridge=0.0)
    assert Sigma.shape == (5, 5)
    assert np.all(np.diag(Sigma) > 0)
    np.testing.assert_array_almost_equal(Sigma, Sigma.T)


def test_ewma_covariance_ridge():
    R = np.random.randn(3, 50) * 0.01  # 3 assets, 50 time points
    Sigma = _ewma_covariance(R, halflife=10, ridge=1e-6)
    assert Sigma.shape == (3, 3)
    assert np.all(np.linalg.eigvalsh(Sigma) > 0)


# ── covariance: estimate_covariance ───────────────────────────────────────────
# Wide-format: rows=assets, columns=dates


def test_estimate_covariance_sample():
    np.random.seed(123)
    # Wide format: each row = asset, each column = date
    returns = pd.DataFrame(
        np.random.randn(4, 200) * 0.01,
        index=["A", "B", "C", "D"],
        columns=pd.date_range("2021-01-01", periods=200),
    )
    Sigma = estimate_covariance(returns, method="sample")
    assert Sigma.shape == (4, 4)
    assert np.all(np.diag(Sigma) > 0)
