#!/usr/bin/env python3
"""1-D ResNet classifier for 12-lead ECG."""

from __future__ import annotations

import torch
import torch.nn as nn


class BasicBlock1D(nn.Module):
    """Single ResNet basic block for 1-D convolutions."""

    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 7,
        stride: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)
        return out


class ResNet1D(nn.Module):
    """Configurable 1-D ResNet for ECG classification.

    Input:  (batch, in_channels, seq_len)
    Output: (batch, num_classes)
    """

    def __init__(
        self,
        in_channels: int = 12,
        base_channels: int = 64,
        layers: tuple[int, ...] = (2, 2, 2, 2),
        kernel_size: int = 7,
        embedding_dim: int = 512,
        num_classes: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        self.stem = nn.Sequential(
            nn.Conv1d(
                in_channels,
                base_channels,
                kernel_size=kernel_size,
                stride=2,
                padding=kernel_size // 2,
                bias=False,
            ),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )

        channels = base_channels
        self.layer_blocks = nn.ModuleList()
        for i, num_blocks in enumerate(layers):
            out_channels = base_channels * (2 ** i)
            stride = 1 if i == 0 else 2
            blocks = []
            for b in range(num_blocks):
                blocks.append(
                    BasicBlock1D(
                        channels,
                        out_channels,
                        kernel_size=kernel_size,
                        stride=stride if b == 0 else 1,
                        dropout=dropout,
                    )
                )
                channels = out_channels
            self.layer_blocks.append(nn.Sequential(*blocks))

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for layer in self.layer_blocks:
            x = layer(x)
        x = self.global_pool(x)
        x = self.embedding(x)
        x = self.classifier(x)
        return x

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Return the fixed-size embedding before the classifier head."""
        x = self.stem(x)
        for layer in self.layer_blocks:
            x = layer(x)
        x = self.global_pool(x)
        x = self.embedding(x)
        return x


def build_model(cfg: dict) -> nn.Module:
    model_cfg = cfg["model"]
    return ResNet1D(
        in_channels=model_cfg["in_channels"],
        base_channels=model_cfg["base_channels"],
        layers=tuple(model_cfg["layers"]),
        kernel_size=model_cfg["kernel_size"],
        embedding_dim=model_cfg["embedding_dim"],
        num_classes=model_cfg["num_classes"],
        dropout=model_cfg["dropout"],
    )


if __name__ == "__main__":
    # Quick smoke test
    model = ResNet1D(in_channels=12, num_classes=3)
    x = torch.randn(2, 12, 1000)
    y = model(x)
    emb = model.get_embedding(x)
    print(f"logits shape: {y.shape}")
    print(f"embedding shape: {emb.shape}")
    assert y.shape == (2, 3)
    print("Smoke test passed.")
