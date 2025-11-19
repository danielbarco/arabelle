from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

import energy_matching.simulated_annealing as simulated_annealing
from energy_matching import data as data_mod, model as model_mod


def _plot_series(ax, times, values, label, color, style, linewidth=1.5):
    t_norm = times / (times[-1] if times[-1] > 0 else 1.0)
    values_np = values.numpy()
    label_used = False
    for idx in range(values_np.shape[1]):
        ax.plot(
            t_norm,
            values_np[:, idx],
            color=color,
            linestyle=style,
            linewidth=linewidth,
            alpha=0.7,
            label=label if not label_used else None,
        )
        label_used = True


def _mark_best(ax, result, label, highlight_color, y_frac: float):
    profits_np = result.profits.numpy()
    flat_idx = np.argmax(profits_np)
    t_idx, chain_idx = np.unravel_index(flat_idx, profits_np.shape)
    times = result.times.numpy()
    t_norm = times / (times[-1] if times[-1] > 0 else 1.0)
    best_time = t_norm[t_idx]
    best_value = profits_np[t_idx, chain_idx]
    ax.scatter(
        [best_time],
        [best_value],
        color=highlight_color,
        edgecolor="black",
        linewidth=0.6,
        marker="o",
        s=70,
        zorder=6,
        label=f"Best {label}",
    )
    ax.annotate(
        f"Best {label}\n{best_value:,.0f} EUR",
        xy=(best_time, best_value),
        xytext=(0.97, y_frac),
        textcoords="axes fraction",
        fontsize=11,
        fontweight="bold",
        color=highlight_color,
        bbox=dict(
            boxstyle="round,pad=0.2", facecolor="white", edgecolor=highlight_color, alpha=0.9
        ),
        arrowprops=dict(arrowstyle="->", color=highlight_color, linewidth=1.2),
        ha="right",
        zorder=7,
    )


def _delta_corr_stats(result) -> tuple[float, float]:
    energies = result.energies.numpy()
    profits = result.profits.numpy()
    delta_e = energies[1:] - energies[:-1]
    delta_p = profits[1:] - profits[:-1]
    corrs = []
    for t in range(delta_e.shape[0]):
        e = delta_e[t]
        p = delta_p[t]
        if np.std(e) < 1e-8 or np.std(p) < 1e-8:
            continue
        corr = np.corrcoef(e, p)[0, 1]
        if np.isnan(corr):
            continue
        corrs.append(float(np.clip(corr, -1.0, 1.0)))
    if corrs:
        return float(np.mean(corrs)), float(np.std(corrs))
    return 0.0, 0.0


def plot_comparison(guided, baseline, output_path: Path) -> None:
    if guided.times.shape != baseline.times.shape:
        raise ValueError(
            "Guided and baseline runs must have the same number of steps to plot together."
        )

    fig, (ax_energy, ax_profit, ax_sched) = plt.subplots(1, 3, figsize=(18, 5))

    _plot_series(ax_energy, guided.times.numpy(), guided.energies, "Guided energy", "#1976d2", "-")
    corr_mean, corr_std = _delta_corr_stats(guided)
    ax_energy.set_xlabel("Normalized time")
    ax_energy.set_ylabel("Energy")
    ax_energy.set_title(
        "Simulated annealing energy trajectories\n"
        f"(ΔProfit-ΔEnergy corr {corr_mean:.2f} ± {corr_std:.2f})"
    )
    ax_energy.grid(True, linestyle=":", alpha=0.4)
    ax_energy.set_xlim(0.0, 1.0)
    ax_energy.legend(loc="best")

    _plot_series(ax_profit, guided.times.numpy(), guided.profits, "Guided profit", "#1976d2", "-")
    _plot_series(
        ax_profit, baseline.times.numpy(), baseline.profits, "Baseline profit", "#6d4c41", "--"
    )
    _mark_best(ax_profit, guided, "guided", "#ffb300", y_frac=0.7)
    _mark_best(ax_profit, baseline, "baseline", "#8e24aa", y_frac=0.5)
    ax_profit.set_xlabel("Normalized time")
    ax_profit.set_ylabel("Annual profit (EUR)")
    ax_profit.set_title("Profit trajectories (guided vs. baseline)")
    ax_profit.grid(True, linestyle=":", alpha=0.4)
    ax_profit.set_xlim(0.0, 1.0)
    ax_profit.legend(loc="best")

    sched_handles = []
    sched_labels = []
    ax_sched_right = None
    if guided.profit_temps is not None and guided.profit_temps.numel() > 0:
        t_sched = np.linspace(0.0, 1.0, guided.profit_temps.shape[0])
        h = ax_sched.plot(
            t_sched,
            guided.profit_temps.numpy(),
            color="#1976d2",
            linewidth=2.0,
            label="Guided profit temp",
        )[0]
        sched_handles.append(h)
        sched_labels.append("Guided profit temp")
    if baseline.profit_temps is not None and baseline.profit_temps.numel() > 0:
        t_sched = np.linspace(0.0, 1.0, baseline.profit_temps.shape[0])
        h = ax_sched.plot(
            t_sched,
            baseline.profit_temps.numpy(),
            color="#6d4c41",
            linestyle="--",
            linewidth=2.0,
            label="Baseline profit temp",
        )[0]
        sched_handles.append(h)
        sched_labels.append("Baseline profit temp")
    if guided.guide_strengths is not None and guided.guide_strengths.numel() > 0:
        if guided.guide_strengths.abs().sum().item() > 0:
            t_sched = np.linspace(0.0, 1.0, guided.guide_strengths.shape[0])
            if ax_sched_right is None:
                ax_sched_right = ax_sched.twinx()
            h = ax_sched_right.plot(
                t_sched,
                guided.guide_strengths.numpy(),
                color="#ffb300",
                linewidth=2.0,
                label="Guide strength",
            )[0]
            sched_handles.append(h)
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
    parser = argparse.ArgumentParser(
        description="Compare guided and random baseline simulated annealing runs."
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-path", type=str, default="energy_matching/positive_samples.json")
    parser.add_argument("--num-chains", type=int, default=16)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--total-time", type=float, default=1.0)
    parser.add_argument("--cont-temp", type=float, default=0.05)
    parser.add_argument("--profit-temp", type=float, default=1e7)
    parser.add_argument("--guide-strength", type=float, default=0.02)
    parser.add_argument("--guide-prob", type=float, default=0.7)
    parser.add_argument("--guide-beta", type=float, default=10.0)
    parser.add_argument("--guide-lambda", type=float, default=1.0)
    parser.add_argument(
        "--profit-schedule", type=str, choices=["constant", "linear"], default="linear"
    )
    parser.add_argument("--profit-final-ratio", type=float, default=0.01)
    parser.add_argument(
        "--guide-schedule", type=str, choices=["constant", "linear"], default="linear"
    )
    parser.add_argument("--guide-final-ratio", type=float, default=0.2)
    parser.add_argument(
        "--output", type=str, default="figures/solver_simulated_annealing_with_baseline.png"
    )
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[64, 64, 64])
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    dataset = data_mod.PositiveSamplesDataset(args.data_path)
    model = model_mod.build_energy_model(args.hidden_sizes, dropout=args.dropout)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state.get("model", state))

    guided = simulated_annealing.simulate(
        model,
        dataset.tensor,
        num_data_chains=0,
        num_noise_chains=args.num_chains,
        steps=args.steps,
        total_time=args.total_time,
        cont_temp=args.cont_temp,
        profit_temp=args.profit_temp,
        profit_schedule=args.profit_schedule,
        profit_final_ratio=args.profit_final_ratio,
        proposal_mode="guided",
        guide_strength=args.guide_strength,
        guide_prob=args.guide_prob,
        guide_beta=args.guide_beta,
        guide_lambda=args.guide_lambda,
        guide_schedule=args.guide_schedule,
        guide_final_ratio=args.guide_final_ratio,
        device=device,
        seed=args.seed,
    )

    baseline = simulated_annealing.simulate(
        model,
        dataset.tensor,
        num_data_chains=0,
        num_noise_chains=args.num_chains,
        steps=args.steps,
        total_time=args.total_time,
        cont_temp=args.cont_temp,
        profit_temp=args.profit_temp,
        profit_schedule=args.profit_schedule,
        profit_final_ratio=args.profit_final_ratio,
        proposal_mode="random",
        guide_strength=0.0,
        guide_prob=0.0,
        guide_schedule=args.guide_schedule,
        guide_final_ratio=args.guide_final_ratio,
        device=device,
        seed=args.seed + 1 if args.seed is not None else None,
    )

    plot_comparison(guided, baseline, Path(args.output))
    print(f"Saved guided vs. baseline SA plot to {args.output}")
    if guided.acceptance_rate is not None:
        print(f"[guided] Acceptance rate: {guided.acceptance_rate:.4f}")
    if baseline.acceptance_rate is not None:
        print(f"[baseline] Acceptance rate: {baseline.acceptance_rate:.4f}")


if __name__ == "__main__":
    main()
