"""
Arhitectura CNN 1D pentru clasificarea secventelor ADN (promotor vs.
non-promotor).

Input: tensor (batch, 4, L) -- codificare one-hot pe canale A/C/G/T.
Output: logit scalar per secventa (batch, 1) -- folosit cu BCEWithLogitsLoss.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from . import config


class ConvBlock(nn.Module):
    """Un bloc Conv1d -> BatchNorm1d -> ReLU -> MaxPool1d."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        pool_size: int,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2  # padding "same" pentru a pastra lungimea inainte de pool
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size=kernel_size, padding=padding
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool1d(kernel_size=pool_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x


class DNA_CNN(nn.Module):
    """CNN 1D pentru clasificare binara a secventelor ADN.

    Structura: N blocuri Conv1d+BN+ReLU+MaxPool (N = len(conv_channels)),
    urmate de global average pooling si un cap fully-connected cu dropout,
    care produce un singur logit (clasificare binara).
    """

    def __init__(self, cfg: config.CNNConfig = config.CNN_CONFIG) -> None:
        super().__init__()

        assert len(cfg.conv_channels) == len(cfg.kernel_sizes), (
            "conv_channels si kernel_sizes trebuie sa aiba aceeasi lungime"
        )

        blocks = []
        in_ch = cfg.in_channels
        for out_ch, kernel_size in zip(cfg.conv_channels, cfg.kernel_sizes):
            blocks.append(ConvBlock(in_ch, out_ch, kernel_size, cfg.pool_size))
            in_ch = out_ch
        self.conv_blocks = nn.Sequential(*blocks)

        self.global_pool = nn.AdaptiveAvgPool1d(1)

        last_channels = cfg.conv_channels[-1]
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(last_channels, cfg.fc_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.fc_hidden_dim, cfg.num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, 4, L) -> logits: (batch, num_classes)."""
        x = self.conv_blocks(x)
        x = self.global_pool(x)
        logits = self.classifier(x)
        return logits


def build_model(cfg: config.CNNConfig = config.CNN_CONFIG) -> DNA_CNN:
    return DNA_CNN(cfg)
