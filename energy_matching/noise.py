"""
Noise sampling utilities for Energy Matching training.
"""

from __future__ import annotations

import torch

from . import common


def sample_uniform_noise(batch_size: int, device: torch.device | None = None) -> torch.Tensor:
    """
    Sample random SMR configurations from simple independent uniforms.
    """
    if device is None:
        device = torch.device("cpu")

    reactor_index = torch.randint(0, common.NUM_REACTOR_MODELS, (batch_size,), device=device)
    n_reactors = torch.randint(1, common.MAX_NUM_REACTORS + 1, (batch_size,), device=device)
    n_storage = torch.randint(0, common.MAX_STORAGE_MODULES + 1, (batch_size,), device=device)

    reactor_caps = torch.tensor(common.REACTOR_MODELS, device=device)[reactor_index]
    prod_capacity = reactor_caps * n_reactors.float()
    prod_frac = torch.rand(batch_size, common.HORIZON, device=device)
    prod = prod_frac * prod_capacity.unsqueeze(1)

    soc_capacity = n_storage.float() * common.MODULE_CAPACITY_MWH
    soc_frac = torch.rand(batch_size, common.HORIZON, device=device)
    soc = soc_frac * soc_capacity.unsqueeze(1)

    return common.encode_config(reactor_index, n_reactors, n_storage, prod, soc)
