from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from energy_matching import data as data_mod, model as model_mod, noise


def pair_with_noise(x_data: torch.Tensor, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = x_data.size(0)
    x_noise = noise.sample_uniform_noise(batch_size, device=device)
    t = torch.rand(batch_size, 1, device=device)
    x_path = (1.0 - t) * x_noise + t * x_data
    target_vel = x_data - x_noise
    return x_path.requires_grad_(True), target_vel, x_noise


def train(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() and not cfg.cpu else "cpu")
    dataset = data_mod.PositiveSamplesDataset(cfg.data_path)
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)

    model = model_mod.build_energy_model(cfg.hidden_sizes, cfg.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    steps_per_epoch = len(dataloader)
    log = {"iter": [], "loss": []}

    global_step = 0
    model.train()
    for epoch in range(cfg.epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{cfg.epochs}", leave=False)
        for batch in pbar:
            x_data = batch.to(device)
            x_path, target_vel, _ = pair_with_noise(x_data, device)
            energies = model(x_path)
            grad = torch.autograd.grad(energies.sum(), x_path, create_graph=True)[0]
            loss = torch.mean((grad + target_vel) ** 2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_step += 1
            log["iter"].append(global_step)
            log["loss"].append(loss.item())
            pbar.set_postfix({"loss": loss.item()})

        if cfg.checkpoint_dir:
            ckpt_path = Path(cfg.checkpoint_dir)
            ckpt_path.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "epoch": epoch + 1}, ckpt_path / f"energy_model_epoch_{epoch+1}.pt")

    if cfg.log_path:
        with open(cfg.log_path, "w", encoding="utf-8") as f:
            json.dump(log, f)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Energy Matching scalar network on SMR dataset.")
    parser.add_argument("--data-path", type=str, default="positive_samples.json")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[512, 512, 512])
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--checkpoint-dir", type=str, default="energy_checkpoints")
    parser.add_argument("--log-path", type=str, default="energy_training_log.json")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
