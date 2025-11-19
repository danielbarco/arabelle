from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from energy_matching import data as data_mod, model as model_mod, noise, solver


def pair_with_noise(x_data: torch.Tensor, device: torch.device):
    batch_size = x_data.size(0)
    x_noise = noise.sample_uniform_noise(batch_size, device=device)
    t = torch.rand(batch_size, 1, device=device)
    x_path = (1.0 - t) * x_noise + t * x_data
    target_vel = x_data - x_noise
    return x_path.requires_grad_(True), target_vel


def fm_loss(model, x_data, device):
    x_path, target_vel = pair_with_noise(x_data, device)
    energies = model(x_path)
    grad = torch.autograd.grad(energies.sum(), x_path, create_graph=True)[0]
    return torch.mean((grad + target_vel) ** 2)


def generate_cd_negatives(
    model,
    dataset_tensor: torch.Tensor,
    batch_size: int,
    device: torch.device,
    *,
    delta_t: float,
    total_time: float,
    beta: float,
    lambda_penalty: float,
    steps: int | None = None,
):
    num_data = batch_size // 2
    num_noise = batch_size - num_data
    with torch.no_grad():
        result = solver.simulate(
            model,
            dataset_tensor,
            num_data_chains=num_data,
            num_noise_chains=num_noise,
            delta_t=delta_t,
            steps=steps,
            total_time=total_time,
            beta=beta,
            lambda_penalty=lambda_penalty,
            device=device,
            record_states=False,
        )
        negatives = result.states.to(device)
    return negatives


def contrastive_divergence_loss(model, x_pos, x_neg):
    energy_pos = model(x_pos).mean()
    energy_neg = model(x_neg).mean()
    return energy_pos - energy_neg


def plot_losses(iters, fm_losses, cd_losses, out_path: str):
    plt.figure(figsize=(8, 5))
    plt.plot(iters, fm_losses, label="Flow-matching loss", color="tab:blue")
    plt.plot(iters, cd_losses, label="Contrastive divergence loss", color="tab:orange")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Training losses per batch")
    plt.grid(True, linestyle=":", alpha=0.4)
    plt.legend(loc="best")
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def train(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() and not cfg.cpu else "cpu")
    hp_summary = {
        "lambda_fm": cfg.lambda_fm,
        "lambda_cd": cfg.lambda_cd,
        "cd_delta_t": cfg.cd_delta_t,
        "cd_total_time": cfg.cd_total_time,
        "cd_beta": cfg.cd_beta,
        "cd_lambda": cfg.cd_lambda,
        "cd_steps": cfg.cd_steps,
        "batch_size": cfg.batch_size,
        "lr": cfg.lr,
        "weight_decay": cfg.weight_decay,
    }
    print("[info] Hyperparameters:", hp_summary)

    dataset = data_mod.PositiveSamplesDataset(cfg.data_path)
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)

    model = model_mod.build_energy_model(cfg.hidden_sizes, cfg.dropout).to(device)
    if cfg.resume:
        state = torch.load(cfg.resume, map_location=device)
        model.load_state_dict(state.get("model", state))
        print(f"[info] Loaded warmup checkpoint from {cfg.resume}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    fm_history = []
    cd_history = []
    iter_history = []
    global_step = 0

    model.train()
    dataset_tensor = dataset.tensor

    for epoch in range(cfg.epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{cfg.epochs}", leave=False)
        for batch in pbar:
            x_data = batch.to(device)

            loss_fm = fm_loss(model, x_data, device)

            x_neg = generate_cd_negatives(
                model,
                dataset_tensor,
                cfg.batch_size,
                device,
                delta_t=cfg.cd_delta_t,
                total_time=cfg.cd_total_time,
                beta=cfg.cd_beta,
                lambda_penalty=cfg.cd_lambda,
                steps=cfg.cd_steps if cfg.cd_steps > 0 else None,
            )
            loss_cd = contrastive_divergence_loss(model, x_data, x_neg)

            total_loss = cfg.lambda_fm * loss_fm + cfg.lambda_cd * loss_cd

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            global_step += 1
            fm_history.append(loss_fm.item())
            cd_history.append(loss_cd.item())
            iter_history.append(global_step)

            pbar.set_postfix({"FM loss": loss_fm.item(), "CD loss": loss_cd.item()})

    if cfg.checkpoint_path:
        Path(cfg.checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "iter": global_step}, cfg.checkpoint_path)
        print(f"[info] Saved checkpoint to {cfg.checkpoint_path}")

    if cfg.loss_plot:
        plot_losses(iter_history, fm_history, cd_history, cfg.loss_plot)
        print(f"[info] Saved loss plot to {cfg.loss_plot}")

    if cfg.log_path:
        log = {"iter": iter_history, "loss_fm": fm_history, "loss_cd": cd_history}
        log["hyperparameters"] = hp_summary
        with open(cfg.log_path, "w", encoding="utf-8") as f:
            json.dump(log, f, indent=2)
        print(f"[info] Saved loss history to {cfg.log_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Main Energy Matching training with FM + CD objectives.")
    parser.add_argument("--data-path", type=str, default="positive_samples.json")
    parser.add_argument("--resume", type=str, default="energy_checkpoints/energy_model_epoch_20.pt")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[512, 512, 512])
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")

    parser.add_argument("--lambda-fm", type=float, default=1.0)
    parser.add_argument("--lambda-cd", type=float, default=1.0)

    parser.add_argument("--cd-delta-t", type=float, default=0.01)
    parser.add_argument("--cd-total-time", type=float, default=2.0)
    parser.add_argument("--cd-beta", type=float, default=5.0)
    parser.add_argument("--cd-lambda", type=float, default=2.0)
    parser.add_argument("--cd-steps", type=int, default=0)

    parser.add_argument("--checkpoint-path", type=str, default="energy_checkpoints/main_training_final.pt")
    parser.add_argument("--loss-plot", type=str, default="figures/main_training_losses.png")
    parser.add_argument("--log-path", type=str, default="energy_training_main_log.json")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
