"""
Shared constants and feature encoding helpers for Energy Matching training.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F


# Problem constants (mirrored from minlp_smr_battery_storage.py)
HORIZON = 24
REACTOR_MODELS = [80.0, 160.0, 300.0, 350.0, 470.0]
MAX_NUM_REACTORS = 3
MAX_STORAGE_MODULES = 39
MODULE_CAPACITY_MWH = 50.0

NUM_REACTOR_MODELS = len(REACTOR_MODELS)
NUM_REACTOR_COUNT_OPTIONS = MAX_NUM_REACTORS
NUM_STORAGE_OPTIONS = MAX_STORAGE_MODULES + 1  # include zero-module option

# Normalization scales for continuous trajectories
MAX_PRODUCTION_CAPACITY = max(REACTOR_MODELS) * MAX_NUM_REACTORS  # MW
MAX_SOC_CAPACITY = MODULE_CAPACITY_MWH * MAX_STORAGE_MODULES  # MWh

DISCRETE_DIM = NUM_REACTOR_MODELS + NUM_REACTOR_COUNT_OPTIONS + NUM_STORAGE_OPTIONS
CONTINUOUS_DIM = 2 * HORIZON
FEATURE_DIM = DISCRETE_DIM + CONTINUOUS_DIM

REACTOR_MODEL_TO_INDEX = {float(p): idx for idx, p in enumerate(REACTOR_MODELS)}


def encode_config(
    reactor_index: torch.Tensor,
    n_reactors: torch.Tensor,
    n_storage: torch.Tensor,
    prod: torch.Tensor,
    soc: torch.Tensor,
) -> torch.Tensor:
    """
    Encode discrete + continuous SMR configuration into a single feature vector.
    Continuous trajectories are stored as fractions in [0, 1] of the physical
    capacities implied by the discrete selections so that downstream solvers can
    remain agnostic of the specific unit scaling.
    """
    reactor_index = reactor_index.long()
    n_reactors = n_reactors.long()
    n_storage = n_storage.long()
    prod = prod.float()
    soc = soc.float()

    device = prod.device
    dtype = prod.dtype

    reactor_caps = prod.new_tensor(REACTOR_MODELS, dtype=dtype)[reactor_index]
    prod_capacity = reactor_caps * n_reactors.float()
    prod_capacity_safe = torch.where(prod_capacity > 0, prod_capacity, torch.ones_like(prod_capacity))
    prod_fraction = torch.clamp(prod / prod_capacity_safe.unsqueeze(1), 0.0, 1.0)
    prod_fraction = torch.where(prod_capacity.unsqueeze(1) > 0, prod_fraction, torch.zeros_like(prod_fraction))

    soc_capacity = n_storage.float() * MODULE_CAPACITY_MWH
    soc_capacity_safe = torch.where(soc_capacity > 0, soc_capacity, torch.ones_like(soc_capacity))
    soc_fraction = torch.clamp(soc / soc_capacity_safe.unsqueeze(1), 0.0, 1.0)
    soc_fraction = torch.where(soc_capacity.unsqueeze(1) > 0, soc_fraction, torch.zeros_like(soc_fraction))

    prod_flat = prod_fraction.view(prod_fraction.size(0), -1)
    soc_flat = soc_fraction.view(soc_fraction.size(0), -1)

    one_hot_model = F.one_hot(reactor_index, NUM_REACTOR_MODELS).float().to(device)
    one_hot_n_reactors = F.one_hot(torch.clamp(n_reactors - 1, min=0), NUM_REACTOR_COUNT_OPTIONS).float().to(device)
    one_hot_storage = F.one_hot(n_storage.clamp(min=0), NUM_STORAGE_OPTIONS).float().to(device)

    encoded = torch.cat([one_hot_model, one_hot_n_reactors, one_hot_storage, prod_flat, soc_flat], dim=1)

    return encoded


@dataclass
class DecodedConfig:
    reactor_index: torch.Tensor
    n_reactors: torch.Tensor
    n_storage: torch.Tensor
    prod: torch.Tensor
    soc: torch.Tensor
    prod_fraction: torch.Tensor
    soc_fraction: torch.Tensor


def decode_config(encoded: torch.Tensor) -> DecodedConfig:
    """
    Decode an encoded feature vector back to discrete selections and physical trajectories.
    The stored continuous values are interpreted as fractions of the implied
    production / storage capacities and expanded to MW/MWh only when needed.
    """
    if encoded.dim() == 1:
        encoded = encoded.unsqueeze(0)

    start = 0
    model_block = encoded[:, start : start + NUM_REACTOR_MODELS]
    start += NUM_REACTOR_MODELS
    n_reactors_block = encoded[:, start : start + NUM_REACTOR_COUNT_OPTIONS]
    start += NUM_REACTOR_COUNT_OPTIONS
    storage_block = encoded[:, start : start + NUM_STORAGE_OPTIONS]
    start += NUM_STORAGE_OPTIONS

    prod_block = encoded[:, start : start + HORIZON]
    soc_block = encoded[:, start + HORIZON :]

    reactor_index = model_block.argmax(dim=1)
    n_reactors = n_reactors_block.argmax(dim=1) + 1
    n_storage = storage_block.argmax(dim=1)

    prod_fraction = torch.clamp(prod_block, 0.0, 1.0)
    soc_fraction = torch.clamp(soc_block, 0.0, 1.0)

    reactor_caps = prod_block.new_tensor(REACTOR_MODELS)[reactor_index]
    prod_capacity = (reactor_caps * n_reactors.float()).unsqueeze(1)
    prod = prod_fraction * prod_capacity

    soc_capacity = (n_storage.float() * MODULE_CAPACITY_MWH).unsqueeze(1)
    soc = soc_fraction * soc_capacity

    return DecodedConfig(reactor_index, n_reactors, n_storage, prod, soc, prod_fraction, soc_fraction)


def map_reactor_model_to_index(model_mw: float) -> int:
    try:
        return REACTOR_MODEL_TO_INDEX[float(model_mw)]
    except KeyError as exc:
        raise ValueError(f"Unknown reactor model {model_mw}") from exc


def encode_from_dict(sample: dict) -> torch.Tensor:
    """
    Convenience wrapper to encode a JSON sample entry.
    """
    reactor_index = torch.tensor([map_reactor_model_to_index(sample["m_mw"])], dtype=torch.long)
    n_reactors = torch.tensor([sample["n_r"]], dtype=torch.long)
    n_storage = torch.tensor([sample["n_storage_fixed"]], dtype=torch.long)
    prod = torch.tensor(sample["prod"], dtype=torch.float32).unsqueeze(0)
    soc = torch.tensor(sample["soc"], dtype=torch.float32).unsqueeze(0)
    return encode_config(reactor_index, n_reactors, n_storage, prod, soc).squeeze(0)


def stack_samples(samples: Sequence[dict]) -> torch.Tensor:
    return torch.stack([encode_from_dict(s) for s in samples], dim=0)
