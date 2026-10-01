"""Nine comparable historical-input deep-learning forecast baselines.

All accept [batch, lookback, features] and output [batch, horizon_steps].
No model receives realised future weather in the operational track.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, features: int, lookback: int, horizon: int,
                 hidden: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(features * lookback, hidden),
                                 nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, horizon))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CNN1D(nn.Module):
    def __init__(self, features: int, horizon: int, hidden: int = 64,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv1d(features, hidden, 3, padding=1), nn.ReLU(),
                                     nn.Conv1d(hidden, hidden, 3, padding=1), nn.ReLU(),
                                     nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Dropout(dropout))
        self.head = nn.Linear(hidden, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x.transpose(1, 2)))


class Recurrent(nn.Module):
    def __init__(self, cell: str, features: int, horizon: int, hidden: int = 64,
                 layers: int = 1, dropout: float = 0.1,
                 bidirectional: bool = False) -> None:
        super().__init__()
        cls = {"lstm": nn.LSTM, "gru": nn.GRU}[cell]
        self.encoder = cls(features, hidden, num_layers=layers, batch_first=True,
                           bidirectional=bidirectional,
                           dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Dropout(dropout),
                                  nn.Linear(hidden * (2 if bidirectional else 1), horizon))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.encoder(x)
        if self.encoder.bidirectional:
            representation = torch.cat([output[:, -1, :self.encoder.hidden_size],
                                        output[:, 0, self.encoder.hidden_size:]], dim=-1)
        else:
            representation = output[:, -1]
        return self.head(representation)


class CNNLSTM(nn.Module):
    def __init__(self, features: int, horizon: int, hidden: int = 64,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(features, hidden, 3, padding=1), nn.ReLU())
        self.lstm = nn.LSTM(hidden, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, horizon))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local = self.conv(x.transpose(1, 2)).transpose(1, 2)
        output, _ = self.lstm(local)
        return self.head(output[:, -1])


class CausalResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.padding = 2 * dilation
        self.conv = nn.Conv1d(channels, channels, 3, dilation=dilation)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.dropout(self.activation(self.conv(nn.functional.pad(x, (self.padding, 0)))))


class TCN(nn.Module):
    def __init__(self, features: int, horizon: int, hidden: int = 64,
                 layers: int = 3, dropout: float = 0.1) -> None:
        super().__init__()
        self.entry = nn.Conv1d(features, hidden, 1)
        self.blocks = nn.Sequential(*[CausalResidualBlock(hidden, 2 ** i, dropout)
                                      for i in range(layers)])
        self.head = nn.Linear(hidden, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        state = self.blocks(self.entry(x.transpose(1, 2)))
        return self.head(state[:, :, -1])


class TransformerForecast(nn.Module):
    def __init__(self, features: int, horizon: int, hidden: int = 64,
                 layers: int = 2, heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        if hidden % heads:
            raise ValueError("Hidden width must be divisible by attention heads")
        self.project = nn.Linear(features, hidden)
        layer = nn.TransformerEncoderLayer(hidden, heads, hidden * 4,
                                           dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.head = nn.Linear(hidden, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(x.shape[1], device=x.device, dtype=x.dtype).unsqueeze(1)
        div = torch.exp(torch.arange(0, self.project.out_features, 2,
                                    device=x.device, dtype=x.dtype) *
                        (-math.log(10000.0) / self.project.out_features))
        encoding = torch.zeros(x.shape[1], self.project.out_features, device=x.device,
                               dtype=x.dtype)
        encoding[:, 0::2] = torch.sin(positions * div)
        encoding[:, 1::2] = torch.cos(positions * div)
        state = self.encoder(self.project(x) + encoding.unsqueeze(0))
        return self.head(state[:, -1])


class PatchTST(nn.Module):
    """Channel-independent shared patch encoder with a joint forecast head."""

    def __init__(self, features: int, lookback: int, horizon: int,
                 hidden: int = 64, layers: int = 2, heads: int = 4,
                 patch_length: int = 8, patch_stride: int = 4,
                 dropout: float = 0.1) -> None:
        super().__init__()
        if hidden % heads:
            raise ValueError("Hidden width must be divisible by attention heads")
        if patch_length > lookback:
            raise ValueError("Patch length exceeds lookback")
        self.features = features
        self.patch_length = patch_length
        self.patch_stride = patch_stride
        self.embed = nn.Linear(patch_length, hidden)
        layer = nn.TransformerEncoderLayer(hidden, heads, hidden * 4,
                                           dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.head = nn.Linear(features * hidden, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, features = x.shape
        patches = x.unfold(1, self.patch_length, self.patch_stride)
        patches = patches.permute(0, 2, 1, 3).reshape(batch * features, -1, self.patch_length)
        encoded = self.encoder(self.embed(patches))
        return self.head(encoded[:, -1].reshape(batch, features * encoded.shape[-1]))


BASELINE_NAMES = ("mlp", "cnn1d", "lstm", "gru", "bilstm", "cnn_lstm",
                  "tcn", "transformer", "patchtst")


def build_baseline(name: str, *, features: int, lookback: int,
                   horizon: int, hidden: int = 64, layers: int = 2,
                   dropout: float = 0.1, heads: int = 4) -> nn.Module:
    if name == "mlp":
        return MLP(features, lookback, horizon, hidden, dropout)
    if name == "cnn1d":
        return CNN1D(features, horizon, hidden, dropout)
    if name in {"lstm", "gru", "bilstm"}:
        return Recurrent("gru" if name == "gru" else "lstm", features,
                         horizon, hidden, layers, dropout, name == "bilstm")
    if name == "cnn_lstm":
        return CNNLSTM(features, horizon, hidden, dropout)
    if name == "tcn":
        return TCN(features, horizon, hidden, layers, dropout)
    if name == "transformer":
        return TransformerForecast(features, horizon, hidden, layers, heads, dropout)
    if name == "patchtst":
        return PatchTST(features, lookback, horizon, hidden, layers, heads,
                        patch_length=min(8, lookback), dropout=dropout)
    raise ValueError(f"Unknown baseline: {name}")
