from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from energy_matching import data as data_mod, model as model_mod
from energy_matching.plot_energy_bands import (
    compute_energy_stats,
    compute_noise_stats,
    compute_profit_series,
    evaluate_paths,
    plot_bands,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot energy bands and interpolated paths for a trained checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-path", type=str, default="positive_samples.json")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-paths", type=int, default=10)
    parser.add_argument("--num-grid", type=int, default=25)
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[64, 64, 64])
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--output", type=str, default="figures/energy_bands.png")
    args = parser.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    dataset = data_mod.PositiveSamplesDataset(args.data_path)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = model_mod.build_energy_model(args.hidden_sizes, dropout=args.dropout)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state.get("model", state))
    model.to(device)
    model.eval()

    data_mean, data_std = compute_energy_stats(model, dataloader, device)
    noise_mean, noise_std = compute_noise_stats(model, len(dataset), args.batch_size, device)

    t_grid, path_energies, path_states = evaluate_paths(model, dataset.tensor, args.num_paths, args.num_grid, device)
    path_profits = compute_profit_series(path_states)

    delta_energy = (path_energies[:, 1:] - path_energies[:, :-1]).numpy()
    delta_profit = (path_profits[:, 1:] - path_profits[:, :-1]).numpy()
    corr_values = []
    for j in range(delta_energy.shape[1]):
        e = delta_energy[:, j]
        p = delta_profit[:, j]
        if np.std(e) < 1e-8 or np.std(p) < 1e-8:
            corr_values.append(0.0)
        else:
            corr = float(np.corrcoef(e, p)[0, 1])
            corr_values.append(float(np.clip(corr, -1.0, 1.0)))
    corr_times = 0.5 * (t_grid[:-1].numpy() + t_grid[1:].numpy())
    corr_array = np.array(corr_values, dtype=np.float32)

    plot_bands(
        t_grid,
        path_energies.mean(dim=0),
        path_energies.std(dim=0, unbiased=False),
        noise_mean,
        noise_std,
        data_mean,
        data_std,
        corr_times,
        corr_array,
        Path(args.output),
    )

    print(f"Saved energy band plot to {args.output}")
    print(f"Data energy mean={data_mean:.4f}, std={data_std:.4f}")
    print(f"Noise energy mean={noise_mean:.4f}, std={noise_std:.4f}")


if __name__ == "__main__":
    main()
