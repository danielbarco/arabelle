from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from energy_matching import data as data_mod, model as model_mod
import energy_matching.simulated_annealing as simulated_annealing


def plot_trajectories(result: simulated_annealing.SATrajectory, output_path: Path) -> None:
    times = result.times.numpy()
    denom = times[-1] if times[-1] > 0 else 1.0
    t_norm = times / denom
    energies = result.energies.numpy()
    profits = result.profits.numpy()
    mask = result.init_mask.numpy().astype(bool)

    fig, (ax_energy, ax_profit) = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
    label_energy = [False, False]
    label_profit = [False, False]

    for idx, from_data in enumerate(mask):
        color = "#1976d2" if from_data else "#d32f2f"
        label_idx = 0 if from_data else 1
        if not label_energy[label_idx]:
            ax_energy.plot(t_norm, energies[:, idx], color=color, linewidth=1.5, alpha=0.8, label="Init=data" if from_data else "Init=noise")
            label_energy[label_idx] = True
        else:
            ax_energy.plot(t_norm, energies[:, idx], color=color, linewidth=1.0, alpha=0.5)

        if not label_profit[label_idx]:
            ax_profit.plot(t_norm, profits[:, idx], color=color, linewidth=1.5, alpha=0.8, label="Init=data" if from_data else "Init=noise")
            label_profit[label_idx] = True
        else:
            ax_profit.plot(t_norm, profits[:, idx], color=color, linewidth=1.0, alpha=0.5)

    ax_energy.set_xlabel("Normalized time")
    ax_energy.set_ylabel("Energy")
    ax_energy.set_title("Simulated annealing - energy")
    ax_energy.grid(True, linestyle=":", alpha=0.4)
    ax_energy.set_xlim(0.0, 1.0)
    ax_energy.legend(loc="best")

    ax_profit.set_xlabel("Normalized time")
    ax_profit.set_ylabel("Annual profit (EUR)")
    ax_profit.set_title("Simulated annealing - profit")
    ax_profit.grid(True, linestyle=":", alpha=0.4)
    ax_profit.legend(loc="best")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Run simulated annealing benchmark.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-path", type=str, default="positive_samples.json")
    parser.add_argument("--num-data", type=int, default=8)
    parser.add_argument("--num-noise", type=int, default=8)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--total-time", type=float, default=1.0)
    parser.add_argument("--cont-temp", type=float, default=0.05)
    parser.add_argument("--profit-temp", type=float, default=1e7)
    parser.add_argument("--output", type=str, default="figures/solver_simulated_annealing.png")
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[512, 512, 512])
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    dataset = data_mod.PositiveSamplesDataset(args.data_path)
    model = model_mod.build_energy_model(args.hidden_sizes)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state.get("model", state))

    result = simulated_annealing.simulate(
        model,
        dataset.tensor,
        num_data_chains=args.num_data,
        num_noise_chains=args.num_noise,
        steps=args.steps,
        total_time=args.total_time,
        cont_temp=args.cont_temp,
        profit_temp=args.profit_temp,
        device=device,
        seed=args.seed,
    )

    plot_trajectories(result, Path(args.output))
    print(f"Saved simulated annealing plot to {args.output}")
    if result.acceptance_rate is not None:
        print(f"[info] Acceptance rate: {result.acceptance_rate:.4f}")


if __name__ == "__main__":
    main()
