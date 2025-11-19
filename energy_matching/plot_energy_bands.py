from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from energy_matching import data as data_mod, model as model_mod, noise, common
from minlp_smr_battery_storage import objective, x_from_var


def compute_energy_stats(model, loader, device: torch.device) -> tuple[float, float]:
    values = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            energies = model(batch)
            values.append(energies.detach().cpu())
    energies = torch.cat(values, dim=0)
    return energies.mean().item(), energies.std(unbiased=False).item()


def compute_noise_stats(model, num_samples: int, batch_size: int, device: torch.device) -> tuple[float, float]:
    values = []
    total = 0
    with torch.no_grad():
        while total < num_samples:
            current = min(batch_size, num_samples - total)
            batch = noise.sample_uniform_noise(current, device=device)
            energies = model(batch)
            values.append(energies.detach().cpu())
            total += current
    energies = torch.cat(values, dim=0)
    return energies.mean().item(), energies.std(unbiased=False).item()


def evaluate_paths(model, x_data: torch.Tensor, num_paths: int, num_grid: int, device: torch.device):
    idx = torch.randperm(x_data.size(0))[:num_paths]
    chosen = x_data[idx].to(device)
    t_grid = torch.linspace(0.0, 1.0, num_grid, device=device)
    path_energies = []
    path_states = []
    with torch.no_grad():
        for data_vec in chosen:
            data_vec = data_vec.unsqueeze(0)
            noise_vec = noise.sample_uniform_noise(1, device=device)
            interp = (1.0 - t_grid[:, None]) * noise_vec + t_grid[:, None] * data_vec
            energies = model(interp).detach().cpu().squeeze(-1)
            path_energies.append(energies)
            path_states.append(interp.detach().cpu())
    energies = torch.stack(path_energies, dim=0)
    states = torch.stack(path_states, dim=0)
    return t_grid.cpu(), energies, states


def compute_profit_series(states: torch.Tensor) -> torch.Tensor:
    num_paths, num_grid = states.shape[:2]
    profits = torch.empty((num_paths, num_grid), dtype=torch.float32)
    for i in range(num_paths):
        decoded = common.decode_config(states[i])
        for j in range(num_grid):
            rm = int(decoded.reactor_index[j].item())
            n_r = int(decoded.n_reactors[j].item())
            n_storage = int(decoded.n_storage[j].item())
            prod = decoded.prod[j].numpy()
            soc = decoded.soc[j].numpy()
            x = x_from_var(rm, n_r, n_storage, prod, soc)
            profits[i, j] = objective(x)
    return profits


def plot_bands(
    t_grid,
    interp_mean,
    interp_std,
    noise_mean,
    noise_std,
    data_mean,
    data_std,
    corr_times,
    corr_values,
    output_path: Path,
):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    t_np = t_grid.numpy()
    mean_np = interp_mean.numpy()
    std_np = interp_std.numpy()

    ax.fill_between([0, 1], noise_mean - 3 * noise_std, noise_mean + 3 * noise_std, color="#fddede", alpha=0.5, label="Noise ±3σ")
    ax.axhline(noise_mean, color="#e57373", linestyle="--")

    ax.fill_between([0, 1], data_mean - 3 * data_std, data_mean + 3 * data_std, color="#d0f0d0", alpha=0.5, label="Data ±3σ")
    ax.axhline(data_mean, color="#66bb6a", linestyle="--")

    ax.fill_between(t_np, mean_np - std_np, mean_np + std_np, color="#90caf9", alpha=0.5, label="Interpolated energy ±1σ")
    ax.plot(t_np, mean_np, color="#1976d2", linewidth=2, label="Interpolated mean")

    ax.set_xlabel("Interpolation time")
    ax.set_ylabel("Energy")
    ax.set_title("Energy bands and interpolation trajectories")
    ax.set_xlim(0, 1)
    ax.legend(loc="best")
    ax.grid(True, linestyle=":", alpha=0.4)

    ax_corr = axes[1]
    corr_std = np.std(corr_values)
    std_band = np.clip(corr_std, 0.0, 1.0)
    ax_corr.plot(corr_times, corr_values, color="#5d4037", linewidth=2, label="Corr(ΔEnergy, ΔProfit)")
    ax_corr.fill_between(corr_times, corr_values - std_band, corr_values + std_band, color="#bcaaa4", alpha=0.3, label="±1σ band")
    ax_corr.axhline(0.0, color="black", linestyle="--", linewidth=1)
    ax_corr.set_xlabel("Interpolation time")
    ax_corr.set_ylabel("Correlation coefficient")
    ax_corr.set_title("Local correlation between energy and profit")
    ax_corr.set_xlim(0, 1)
    ax_corr.set_ylim(-1, 1)
    ax_corr.grid(True, linestyle=":", alpha=0.4)
    ax_corr.legend(loc="best")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot energy bands and interpolation trajectories.")
    parser.add_argument("--checkpoint", type=str, default="energy_checkpoints/energy_model_epoch_20.pt")
    parser.add_argument("--data-path", type=str, default="positive_samples.json")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-paths", type=int, default=10)
    parser.add_argument("--num-grid", type=int, default=25)
    parser.add_argument("--output", type=str, default="figures/energy_bands.png")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[512, 512, 512])
    args = parser.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    dataset = data_mod.PositiveSamplesDataset(args.data_path)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = model_mod.build_energy_model(args.hidden_sizes)
    state = torch.load(args.checkpoint, map_location=device)
    state_dict = state.get("model", state)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    data_mean, data_std = compute_energy_stats(model, dataloader, device)
    noise_mean, noise_std = compute_noise_stats(model, len(dataset), args.batch_size, device)

    x_data_all = dataset.tensor
    t_grid, path_energies, path_states = evaluate_paths(model, x_data_all, args.num_paths, args.num_grid, device)
    interp_mean = path_energies.mean(dim=0)
    interp_std = path_energies.std(dim=0, unbiased=False)

    path_profits = compute_profit_series(path_states)
    delta_energy = (path_energies[:, 1:] - path_energies[:, :-1]).numpy()
    delta_profit = (path_profits[:, 1:] - path_profits[:, :-1]).numpy()
    corr_values = []
    for j in range(delta_energy.shape[1]):
        e = delta_energy[:, j]
        p = delta_profit[:, j]
        if np.std(e) < 1e-8 or np.std(p) < 1e-8:
            corr = 0.0
        else:
            corr = float(np.corrcoef(e, p)[0, 1])
        corr_values.append(np.clip(corr, -1.0, 1.0))
    corr_values = np.array(corr_values)
    corr_times = 0.5 * (t_grid[:-1].numpy() + t_grid[1:].numpy())

    plot_bands(
        t_grid,
        interp_mean,
        interp_std,
        noise_mean,
        noise_std,
        data_mean,
        data_std,
        corr_times,
        corr_values,
        Path(args.output),
    )

    print(f"Saved plot to {args.output}")
    print(f"Data energy mean={data_mean:.4f}, std={data_std:.4f}")
    print(f"Noise energy mean={noise_mean:.4f}, std={noise_std:.4f}")


if __name__ == "__main__":
    main()
