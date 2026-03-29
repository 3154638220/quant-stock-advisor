"""时序回归头：LSTM 与简化 TCN（左填充因果卷积 + 残差）。"""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

ArchKind = Literal["lstm", "gru", "tcn", "transformer"]


class LSTMRegressor(nn.Module):
    def __init__(
        self,
        n_features: int,
        hidden: int = 64,
        num_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.drop_head = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last = self.drop_head(out[:, -1, :])
        return self.head(last).squeeze(-1)


class GRURegressor(nn.Module):
    """与 LSTM 结构对称，参数量略少。"""

    def __init__(
        self,
        n_features: int,
        hidden: int = 64,
        num_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.drop_head = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        last = self.drop_head(out[:, -1, :])
        return self.head(last).squeeze(-1)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / float(d_model))
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = x.size(1)
        return x + self.pe[:, :n, :]


class TransformerEncoderRegressor(nn.Module):
    """
    简化 Informer/编码器风格：线性投影 + 位置编码 + TransformerEncoder + 末 token 回归头。
    """

    def __init__(
        self,
        n_features: int,
        d_model: int = 64,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        max_len: int = 128,
    ) -> None:
        super().__init__()
        self.in_proj = nn.Linear(n_features, d_model)
        self.pos = PositionalEncoding(d_model, max_len=max_len)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=num_encoder_layers)
        self.drop_head = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x)
        h = self.pos(h)
        h = self.enc(h)
        last = self.drop_head(h[:, -1, :])
        return self.head(last).squeeze(-1)


class CausalConv1d(nn.Module):
    """左填充因果卷积，保持时间长度与输入一致。"""

    def __init__(self, in_ch: int, out_ch: int, kernel: int, dilation: int) -> None:
        super().__init__()
        self.kernel = kernel
        self.dilation = dilation
        self.pad = (kernel - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=kernel, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L)
        x = F.pad(x, (self.pad, 0))
        return self.conv(x)


class TCNResidualBlock(nn.Module):
    def __init__(self, ch: int, kernel: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.conv1 = CausalConv1d(ch, ch, kernel, dilation)
        self.conv2 = CausalConv1d(ch, ch, kernel, dilation)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.relu(self.conv1(x))
        y = self.drop(y)
        y = self.conv2(y)
        y = self.drop(torch.relu(y))
        return x + y


class TCNRegressor(nn.Module):
    """输入 ``(batch, seq_len, n_features)``。"""

    def __init__(
        self,
        n_features: int,
        hidden: int = 64,
        kernel: int = 3,
        num_blocks: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.in_proj = nn.Conv1d(n_features, hidden, kernel_size=1)
        blocks = []
        for i in range(num_blocks):
            blocks.append(TCNResidualBlock(hidden, kernel, 2**i, dropout))
        self.blocks = nn.ModuleList(blocks)
        self.drop_head = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.transpose(1, 2)
        h = self.in_proj(h)
        for blk in self.blocks:
            h = blk(h)
        last = self.drop_head(h[:, :, -1])
        return self.head(last).squeeze(-1)


def build_timeseries_model(
    kind: ArchKind,
    n_features: int,
    *,
    hidden: int = 64,
    num_layers: int = 2,
    kernel: int = 3,
    num_blocks: int = 3,
    dropout: float = 0.1,
    d_model: int = 64,
    nhead: int = 4,
    num_encoder_layers: int = 2,
    dim_feedforward: int = 128,
    max_seq_len: int = 128,
) -> nn.Module:
    if kind == "lstm":
        return LSTMRegressor(n_features, hidden=hidden, num_layers=num_layers, dropout=dropout)
    if kind == "gru":
        return GRURegressor(n_features, hidden=hidden, num_layers=num_layers, dropout=dropout)
    if kind == "tcn":
        return TCNRegressor(n_features, hidden=hidden, kernel=kernel, num_blocks=num_blocks, dropout=dropout)
    if kind == "transformer":
        dm = d_model if d_model > 0 else hidden
        nh = max(1, min(nhead if nhead > 0 else 4, dm))
        while nh > 1 and dm % nh != 0:
            nh -= 1
        return TransformerEncoderRegressor(
            n_features,
            d_model=dm,
            nhead=nh,
            num_encoder_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward if dim_feedforward > 0 else max(128, dm * 2),
            dropout=dropout,
            max_len=max(max_seq_len, 64),
        )
    raise ValueError(f"未知 kind: {kind}")
