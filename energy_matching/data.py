"""
Dataset utilities for Energy Matching training.
"""

from __future__ import annotations

import json
from pathlib import Path
import torch
from torch.utils.data import Dataset

from . import common


class PositiveSamplesDataset(Dataset):
    """
    Wraps positive_samples.json and returns encoded feature vectors.
    """

    def __init__(self, json_path: str | Path = "positive_samples.json"):
        json_path = Path(json_path)
        with json_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        samples = raw["samples"]
        encoded_list = [common.encode_from_dict(s) for s in samples]
        self._encoded = torch.stack(encoded_list, dim=0)
        self.metadata = samples

    def __len__(self) -> int:
        return self._encoded.size(0)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self._encoded[idx]

    @property
    def tensor(self) -> torch.Tensor:
        return self._encoded


def create_dataloader(batch_size: int, shuffle: bool = True, num_workers: int = 0):
    dataset = PositiveSamplesDataset()
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
