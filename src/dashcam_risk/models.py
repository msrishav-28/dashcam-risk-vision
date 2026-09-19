from __future__ import annotations

import torch
from torch import nn


class LastFrameClassifier(nn.Module):
    """Baseline 2: single-frame encoder. Tiny on purpose (RTX 3050)."""

    def __init__(self, in_dim: int = 512, n_classes: int = 2):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Identity(),
        )
        self.probe_dim = in_dim
        self.head = nn.Linear(in_dim, n_classes)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        # frames: B, T, C, H, W  — use last frame only
        x = frames[:, -1]
        b, c, h, w = x.shape
        pooled = torch.nn.functional.adaptive_avg_pool2d(x, 1).view(b, c)
        if pooled.shape[1] != self.head.in_features:
            pooled = torch.nn.functional.adaptive_avg_pool1d(
                pooled.unsqueeze(1), self.head.in_features
            ).squeeze(1)
        return self.head(pooled)


class FrameGRUClassifier(nn.Module):
    """Baseline 3: per-frame embed + GRU. Default T=8, 224^2."""

    def __init__(self, in_channels: int = 3, embed_dim: int = 128, hidden: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, embed_dim, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.gru = nn.GRU(embed_dim, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 2)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = frames.shape
        x = frames.reshape(b * t, c, h, w)
        z = self.encoder(x).view(b, t, -1)
        out, _ = self.gru(z)
        return self.head(out[:, -1])


def build_model(name: str) -> nn.Module:
    name = name.lower()
    if name in {"last_frame", "last-frame"}:
        return LastFrameClassifier()
    if name in {"gru", "frame_gru"}:
        return FrameGRUClassifier()
    raise ValueError(f"unknown model {name}; use last_frame or gru")
