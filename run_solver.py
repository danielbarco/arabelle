from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from energy_matching import common, data as data_mod, model as model_mod, solver
from minlp_smr_battery_storage import objective, x_from_var


def _compute_profits_for_states(states: torch.Tensor) -> torch.Tensor:
    decoded = common.decode_config(states)
    profits = []
    for i in range(states.size(0)):
        rm = int(decoded.reactor_index[i].item())
        n_r = int(decoded.n_reactors[i].item())
        n_storage = int(decoded.n_storage[i].item())
        prod = decoded.prod[i].cpu().numpy()
        soc = decoded.soc[i].cpu().numpy()
        x = x_from_var(rm, n_r, n_storage, prod, soc)
        profits.append(objective(x))
    return torch.tensor(profits, dtype=torch.float32)


def compute_profit_history(state_history: torch.Tensor | None) -> torch.Tensor | None:
    if state_history is None:
        return None
    history = []
    for t in range(state_history.size(0)):
        history.append(_compute_profits_for_states(state_history[t]))
    return torch.stack(history, dim=0)


def plot_trajectories(
    result: solver.TrajectoryResult, profit_history: torch.Tensor | None, output_path: Path
) -> None:
    times = result.times.numpy()
    denom = times[-1] if times[-1] > 0 else 1.0
    time_axis = times / denom
    energies = result.energies.numpy()
    init_mask = result.init_mask.numpy().astype(bool)

    fig, ax_energy = plt.subplots(figsize=(8, 5))
    label_data = False
    label_noise = False
    colors = []
    for idx, from_data in enumerate(init_mask):
        color = "#1976d2" if from_data else "#d32f2f"
        colors.append(color)
        if from_data:
            label = "Energy (init=data)" if not label_data else None
            label_data = True
        else:
            label = "Energy (init=noise)" if not label_noise else None
            label_noise = True
        ax_energy.plot(time_axis, energies[:, idx], color=color, alpha=0.8, linewidth=1.5, label=label)

    ax_profit = None
    if profit_history is not None:
        profit_np = profit_history.numpy()
        ax_profit = ax_energy.twinx()
        label_data_profit = False
        label_noise_profit = False
        for idx, from_data in enumerate(init_mask):
            label = None
            if from_data:
                if not label_data_profit:
                    label = "Annual profit (init=data)"
                    label_data_profit = True
            else:
                if not label_noise_profit:
                    label = "Annual profit (init=noise)"
                    label_noise_profit = True
            ax_profit.plot(
                time_axis,
                profit_np[:, idx],
                color=colors[idx],
                linestyle="--",
                linewidth=1.2,
                alpha=0.7,
                label=label,
            )
        ax_profit.set_ylabel("Annual profit (EUR)")

    ax_energy.set_xlabel("Normalized time")
    ax_energy.set_ylabel("Energy")
    ax_energy.set_title("Sampler trajectories with annual profit overlay")
    ax_energy.set_xlim(0.0, 1.0)
    ax_energy.grid(True, linestyle=":", alpha=0.4)

    handles, labels = ax_energy.get_legend_handles_labels()
    if ax_profit is not None:
        handles2, labels2 = ax_profit.get_legend_handles_labels()
        for h, l in zip(handles2, labels2):
            if l:
                handles.append(h)
                labels.append(l)
    if handles:
        ax_energy.legend(handles, labels, loc="best")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Run Energy Matching solver and plot trajectories.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained energy model checkpoint.")
    parser.add_argument("--data-path", type=str, default="positive_samples.json")
    parser.add_argument("--num-data", type=int, default=8)
    parser.add_argument("--num-noise", type=int, default=8)
    parser.add_argument("--delta-t", type=float, default=0.01)
    parser.add_argument("--steps", type=int, default=0, help="override number of steps (optional)")
    parser.add_argument("--total-time", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=5.0)
    parser.add_argument("--lambda-penalty", type=float, default=2.0)
    parser.add_argument("--output", type=str, default="figures/solver_trajectories.png")
    parser.add_argument("--save-trajectories", type=str, default="")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[512, 512, 512])
    args = parser.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    dataset = data_mod.PositiveSamplesDataset(args.data_path)
    model = model_mod.build_energy_model(args.hidden_sizes)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state.get("model", state))

    result = solver.simulate(
        model,
        dataset.tensor,
        num_data_chains=args.num_data,
        num_noise_chains=args.num_noise,
        delta_t=args.delta_t,
        steps=args.steps if args.steps > 0 else None,
        total_time=args.total_time,
        beta=args.beta,
        lambda_penalty=args.lambda_penalty,
        device=device,
        record_states=True,
    )

    profit_history = compute_profit_history(result.state_history)
    plot_path = Path(args.output)
    plot_trajectories(result, profit_history, plot_path)
    print(f"Saved energy trajectory plot to {plot_path}")

    if args.save_trajectories:
        save_path = Path(args.save_trajectories)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "times": result.times,
                "energies": result.energies,
                "states": result.states,
                "init_mask": result.init_mask,
                "state_history": result.state_history,
                "profit_history": profit_history,
            },
            save_path,
        )
        print(f"Saved trajectory tensor to {save_path}")


if __name__ == "__main__":
    main()
