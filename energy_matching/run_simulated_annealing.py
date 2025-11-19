from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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

    fig, (ax_energy, ax_profit, ax_sched) = plt.subplots(1, 3, figsize=(18, 5))
    energy_labeled = False
    profit_labeled = False

    for idx, from_data in enumerate(mask):
        color = "#1976d2" if from_data else "#d32f2f"
        ax_energy.plot(
            t_norm,
            energies[:, idx],
            color=color,
            linewidth=1.3,
            alpha=0.7,
            label="Energy" if not energy_labeled else None,
        )
        energy_labeled = True

        ax_profit.plot(
            t_norm,
            profits[:, idx],
            color=color,
            linewidth=1.3,
            alpha=0.7,
            label="Profit" if not profit_labeled else None,
        )
        profit_labeled = True

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
    ax_profit.set_xlim(0.0, 1.0)
    ax_profit.legend(loc="best")

    sched_handles = []
    sched_labels = []
    ax_sched_right = None
    if result.profit_temps is not None and result.profit_temps.numel() > 0:
        sched_t = np.linspace(0.0, 1.0, result.profit_temps.shape[0])
        h_profit = ax_sched.plot(
            sched_t,
            result.profit_temps.numpy(),
            color="#1976d2",
            linewidth=2.0,
            label="Profit temperature",
        )[0]
        sched_handles.append(h_profit)
        sched_labels.append("Profit temperature")
    if result.guide_strengths is not None and result.guide_strengths.numel() > 0:
        guide_vals = result.guide_strengths.numpy()
        if np.any(np.abs(guide_vals) > 1e-8):
            sched_t = np.linspace(0.0, 1.0, guide_vals.shape[0])
            if ax_sched_right is None:
                ax_sched_right = ax_sched.twinx()
            h_guide = ax_sched_right.plot(
                sched_t,
                guide_vals,
                color="#ffb300",
                linewidth=2.0,
                label="Guide strength",
            )[0]
            sched_handles.append(h_guide)
            sched_labels.append("Guide strength")
    ax_sched.set_xlabel("Normalized time")
    ax_sched.set_ylabel("Profit temperature (EUR)")
    if ax_sched_right is not None:
        ax_sched_right.set_ylabel("Guide strength")
    ax_sched.set_title("Schedules")
    ax_sched.grid(True, linestyle=":", alpha=0.4)
    ax_sched.set_xlim(0.0, 1.0)
    if sched_handles:
        ax_sched.legend(sched_handles, sched_labels, loc="best")

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
    parser.add_argument(
        "--proposal-mode",
        type=str,
        choices=["guided", "random"],
        default="guided",
        help="Use gradient-guided proposals or the random baseline (matches simple SA).",
    )
    parser.add_argument(
        "--guide-strength",
        type=float,
        default=0.02,
        help="Initial step size for energy-guided continuous proposals (0 disables guidance).",
    )
    parser.add_argument(
        "--guide-prob",
        type=float,
        default=0.7,
        help="Probability of using energy-guided discrete moves instead of random flips.",
    )
    parser.add_argument(
        "--guide-beta",
        type=float,
        default=10.0,
        help="Inverse temperature for discrete guidance logits.",
    )
    parser.add_argument(
        "--guide-lambda",
        type=float,
        default=1.0,
        help="Penalty weight for changing discrete selections under guidance.",
    )
    parser.add_argument(
        "--profit-schedule",
        type=str,
        choices=["constant", "linear"],
        default="linear",
        help="Schedule type applied to the profit temperature across steps.",
    )
    parser.add_argument(
        "--profit-final-ratio",
        type=float,
        default=0.01,
        help="Final/initial ratio for the profit temperature when using a linear schedule.",
    )
    parser.add_argument(
        "--guide-schedule",
        type=str,
        choices=["constant", "linear"],
        default="linear",
        help="Schedule type applied to the guide strength.",
    )
    parser.add_argument(
        "--guide-final-ratio",
        type=float,
        default=0.2,
        help="Final/initial ratio for the guide strength when using a linear schedule.",
    )
    parser.add_argument("--output", type=str, default="figures/solver_simulated_annealing.png")
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[512, 512, 512])
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.0,
        help="Dropout probability for the energy network (mirrors training settings).",
    )
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    dataset = data_mod.PositiveSamplesDataset(args.data_path)
    model = model_mod.build_energy_model(args.hidden_sizes, dropout=args.dropout)
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
        profit_schedule=args.profit_schedule,
        profit_final_ratio=args.profit_final_ratio,
        proposal_mode=args.proposal_mode,
        guide_strength=args.guide_strength,
        guide_prob=args.guide_prob,
        guide_beta=args.guide_beta,
        guide_lambda=args.guide_lambda,
        guide_schedule=args.guide_schedule,
        guide_final_ratio=args.guide_final_ratio,
        device=device,
        seed=args.seed,
    )

    plot_trajectories(result, Path(args.output))
    print(f"Saved simulated annealing plot to {args.output}")
    if result.acceptance_rate is not None:
        print(f"[info] Acceptance rate: {result.acceptance_rate:.4f}")


if __name__ == "__main__":
    main()
