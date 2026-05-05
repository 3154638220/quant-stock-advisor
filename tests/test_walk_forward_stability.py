"""Tests for feature importance CV and Bonferroni correction in walk_forward.py."""

import numpy as np
import pytest

from src.backtest.walk_forward import (
    feature_importance_cv,
    feature_importance_cv_markdown,
    bonferroni_correction,
    bonferroni_markdown,
)


class TestFeatureImportanceCV:
    def test_all_stable_features(self):
        """所有特征 CV < 1.0，无不稳定特征。"""
        fold_importances = {
            "momentum": [0.25, 0.23, 0.24, 0.26, 0.22],
            "reversal": [0.12, 0.11, 0.13, 0.12, 0.11],
            "lowvol": [0.10, 0.09, 0.10, 0.11, 0.10],
        }
        result = feature_importance_cv(fold_importances)
        assert result["n_features"] == 3
        assert result["n_unstable"] == 0
        assert result["unstable_ratio"] == 0.0
        for feat in fold_importances:
            assert result["per_feature"][feat]["cv"] < 1.0

    def test_unstable_feature_detected(self):
        """CV > 1.0 的特征被正确标注。"""
        fold_importances = {
            "stable_a": [0.30, 0.31, 0.29, 0.30, 0.32],
            "unstable_b": [0.30, 0.01, 0.28, 0.80, 0.02],  # 高方差，CV > 1
        }
        result = feature_importance_cv(fold_importances)
        assert result["n_unstable"] == 1
        assert result["unstable_features"] == ["unstable_b"]

    def test_single_fold_treated_as_unstable(self):
        """单折数据无法计算 CV，视为不稳定。"""
        fold_importances = {"momentum": [0.25]}
        result = feature_importance_cv(fold_importances)
        assert result["n_unstable"] == 1
        assert "momentum" in result["unstable_features"]

    def test_empty_input(self):
        result = feature_importance_cv({})
        assert result["n_features"] == 0
        assert result["n_unstable"] == 0

    def test_zero_mean_feature_cv_inf(self):
        """均值为 0 的特征 CV = inf。"""
        fold_importances = {"flat": [0.0, 0.0, 0.0, 0.0, 0.0]}
        result = feature_importance_cv(fold_importances)
        assert result["per_feature"]["flat"]["unstable"] is True
        assert "flat" in result["unstable_features"]

    def test_custom_threshold(self):
        """自定义 unstable_threshold。"""
        fold_importances = {
            "feat_a": [0.10, 0.11, 0.09, 0.12, 0.10],  # CV ≈ 0.1
        }
        result_default = feature_importance_cv(fold_importances)
        assert result_default["n_unstable"] == 0

        result_strict = feature_importance_cv(fold_importances, unstable_threshold=0.05)
        assert result_strict["n_unstable"] == 1

    def test_nan_values_ignored(self):
        """NaN 值被过滤。"""
        fold_importances = {
            "feat": [0.20, float("nan"), 0.22, 0.19, 0.21],
        }
        result = feature_importance_cv(fold_importances)
        assert result["n_features"] == 1
        assert not np.isnan(result["per_feature"]["feat"]["mean"])


class TestFeatureImportanceCVMarkdown:
    def test_produces_markdown(self):
        fold_importances = {
            "momentum": [0.25, 0.23, 0.24, 0.26, 0.22],
            "unstable_x": [0.30, 0.01, 0.28, 0.80, 0.02],  # CV > 1
        }
        result = feature_importance_cv(fold_importances)
        md = feature_importance_cv_markdown(result)
        assert "## 特征重要性跨折叠稳定性" in md
        assert "momentum" in md
        assert "unstable_x" in md
        assert "M12 Promotion Package" in md

    def test_no_unstable_no_warning_section(self):
        fold_importances = {"momentum": [0.25, 0.23, 0.24]}
        result = feature_importance_cv(fold_importances)
        md = feature_importance_cv_markdown(result)
        # 无不稳定特征时不显示 "需说明" 段落
        assert "M12 Promotion Package" not in md


class TestBonferroniCorrection:
    def test_no_significance_after_correction(self):
        """多个边际显著的 p 值在校正后不显著。"""
        p_values = [0.01, 0.02, 0.03, 0.04, 0.049]
        result = bonferroni_correction(p_values, alpha=0.05)
        assert result["n_tests"] == 5
        assert result["alpha_corrected"] == 0.01
        assert len(result["significant_original"]) == 5
        assert result["any_significant_after_correction"] is False

    def test_single_highly_significant(self):
        """单个极显著的检验在校正后仍显著。"""
        p_values = [0.0001]
        result = bonferroni_correction(p_values, alpha=0.05)
        assert result["significant_corrected"] == [0]
        assert result["any_significant_after_correction"] is True

    def test_all_nonsignificant(self):
        p_values = [0.5, 0.6, 0.7]
        result = bonferroni_correction(p_values, alpha=0.05)
        assert len(result["significant_original"]) == 0
        assert len(result["significant_corrected"]) == 0

    def test_empty_input(self):
        result = bonferroni_correction([], alpha=0.05)
        assert result["n_tests"] == 0
        assert result["any_significant_after_correction"] is False

    def test_nan_filtered(self):
        p_values = [0.01, float("nan"), 0.001]
        result = bonferroni_correction(p_values, alpha=0.05)
        assert result["n_tests"] == 2

    def test_gamma_grid_scenario(self):
        """M8 gamma 网格: 5 个 gamma 值，仅最优的 p 极低。"""
        p_values = [0.001, 0.08, 0.15, 0.30, 0.50]  # 5 gamma values
        result = bonferroni_correction(p_values, alpha=0.05)
        assert result["alpha_corrected"] == 0.01
        assert result["any_significant_after_correction"] is True
        assert result["min_p_corrected"] == 0.005  # 0.001 * 5

    def test_marginal_gamma_grid(self):
        """M8 gamma 网格: 所有 p 值都在边际区域，校正后全不显著。"""
        p_values = [0.01, 0.02, 0.015, 0.025, 0.03]
        result = bonferroni_correction(p_values, alpha=0.05)
        assert result["alpha_corrected"] == 0.01
        # 0.01 is not < 0.01
        assert result["any_significant_after_correction"] is False


class TestBonferroniMarkdown:
    def test_produces_markdown(self):
        p_values = [0.001, 0.08, 0.15]
        result = bonferroni_correction(p_values, alpha=0.05)
        md = bonferroni_markdown(result)
        assert "## Bonferroni 多重检验校正" in md
        assert "M8" in md

    def test_significant_message(self):
        result = bonferroni_correction([0.0001, 0.3, 0.4], alpha=0.05)
        md = bonferroni_markdown(result)
        assert "不是单纯由多重比较伪发现驱动" in md

    def test_nonsignificant_message(self):
        result = bonferroni_correction([0.02, 0.03, 0.04, 0.05], alpha=0.05)
        md = bonferroni_markdown(result)
        assert "可能部分来自多重比较伪发现" in md
