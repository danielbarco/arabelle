"""
Energy network definition.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import torch
from torch import nn

from . import common


class EnergyMLP(nn.Module):
    """
    Simple SiLU-activated MLP producing scalar energy for encoded configurations.
    """

    def __init__(self, hidden_sizes: Sequence[int] = (256, 256, 256), dropout: float = 0.0):
        super().__init__()
        sizes = [common.FEATURE_DIM, *hidden_sizes, 1]
        layers = []
        for in_dim, out_dim in zip(sizes[:-1], sizes[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            if out_dim != 1:
                layers.append(nn.SiLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def build_energy_model(hidden_sizes: Iterable[int] = (256, 256, 256), dropout: float = 0.0) -> EnergyMLP:
    return EnergyMLP(tuple(hidden_sizes), dropout)

