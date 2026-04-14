"""张量因子单元测试（CPU 即可）。"""

from __future__ import annotations

import numpy as np
import torch

from src.features.tensor_alpha import compute_momentum_rsi_torch, momentum_n, rsi_wilder


def test_momentum_n_simple():
    # 两标的、价格线性涨：动量应为常数
    close = torch.tensor(
        [[100.0, 101.0, 102.0], [50.0, 50.5, 51.0]],
        dtype=torch.float32,
    )
    m = momentum_n(close, window=2)
    assert torch.isnan(m[:, 0]).all() and torch.isnan(m[:, 1]).all()
    torch.testing.assert_close(m[:, 2], torch.tensor([102 / 100 - 1, 51 / 50 - 1]))


def test_rsi_wilder_flat():
    close = torch.ones(2, 20, dtype=torch.float32) * 10.0
    r = rsi_wilder(close, period=14)
    assert torch.isfinite(r[:, -1]).all()
    assert (r[:, -1] >= 0).all() and (r[:, -1] <= 100).all()


def test_compute_momentum_rsi_torch_numpy():
    rng = np.random.default_rng(0)
    x = 100.0 * np.cumprod(1.0 + rng.normal(0, 0.01, size=(4, 80)), axis=1)
    mom, rsi = compute_momentum_rsi_torch(
        x,
        momentum_window=5,
        rsi_period=14,
        device="cpu",
        dtype=torch.float64,
    )
    assert mom.shape == x.shape
    assert rsi.shape == x.shape
    assert np.isfinite(mom[:, -1].numpy()).any()
