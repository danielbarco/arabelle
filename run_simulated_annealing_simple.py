from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from energy_matching import common, noise
from minlp_smr_battery_storage import objective, x_from_var


def decode_profit(encoded: torch.Tensor) -> torch.Tensor:
    decoded = common.decode_config(encoded.cpu())
    profits = []
    for i in range(encoded.size(0)):
        rm = int(decoded.reactor_index[i].item())
        n_r = int(decoded.n_reactors[i].item())
        n_storage = int(decoded.n_storage[i].item())
        prod = decoded.prod[i].numpy()
        soc = decoded.soc[i].numpy()
        x = x_from_var(rm, n_r, n_storage, prod, soc)
        profits.append(objective(x))
    return torch.tensor(profits, dtype=torch.float32)


def random_discrete_step(block: torch.Tensor) -> torch.Tensor:
    num_classes = block.size(1)
    current_idx = block.argmax(dim=1)
    direction = torch.randint(-2, 3, current_idx.shape, device=block.device).float()
    new_idx = (current_idx.float() + direction).clamp(0, num_classes - 1).long()
    return torch.nn.functional.one_hot(new_idx, num_classes=num_classes).float()


def materialize_state(fraction_state: torch.Tensor) -> torch.Tensor:
    device = fraction_state.device
    state = fraction_state.clone()
    start = 0
    model_block = state[:, start : start + common.NUM_REACTOR_MODELS]
    model_idx = model_block.argmax(dim=1)
    start += common.NUM_REACTOR_MODELS

    n_reactors_block = state[:, start : start + common.NUM_REACTOR_COUNT_OPTIONS]
    n_reactors = n_reactors_block.argmax(dim=1) + 1
    start += common.NUM_REACTOR_COUNT_OPTIONS

    storage_block = state[:, start : start + common.NUM_STORAGE_OPTIONS]
    n_storage = storage_block.argmax(dim=1)
    start += common.NUM_STORAGE_OPTIONS

    prod_slice = slice(start, start + common.HORIZON)
    soc_slice = slice(start + common.HORIZON, start + 2 * common.HORIZON)

    reactor_caps = torch.tensor(common.REACTOR_MODELS, device=device)[model_idx]
    cap_ratio = (
        reactor_caps * n_reactors.float() / common.MAX_PRODUCTION_CAPACITY
    ).clamp(min=0.0)
    state[:, prod_slice] = state[:, prod_slice].clamp(0.0, 1.0) * cap_ratio.unsqueeze(1)

    storage_energy = (
        n_storage.float() * common.MODULE_CAPACITY_MWH / common.MAX_SOC_CAPACITY
    ).clamp(min=0.0)
    state[:, soc_slice] = state[:, soc_slice].clamp(0.0, 1.0) * storage_energy.unsqueeze(1)
    return state


def enforce_physical_limits(state: torch.Tensor) -> torch.Tensor:
    device = state.device
    start = 0
    model_block = state[:, start : start + common.NUM_REACTOR_MODELS]
    model_idx = model_block.argmax(dim=1)
    start += common.NUM_REACTOR_MODELS

    n_reactors_block = state[:, start : start + common.NUM_REACTOR_COUNT_OPTIONS]
    n_reactors = n_reactors_block.argmax(dim=1) + 1
    start += common.NUM_REACTOR_COUNT_OPTIONS

    storage_block = state[:, start : start + common.NUM_STORAGE_OPTIONS]
    n_storage = storage_block.argmax(dim=1)
    start += common.NUM_STORAGE_OPTIONS

    prod_slice = slice(start, start + common.HORIZON)
    soc_slice = slice(start + common.HORIZON, start + 2 * common.HORIZON)

    reactor_caps = torch.tensor(common.REACTOR_MODELS, device=device)[model_idx]
    cap_ratio = (
        reactor_caps * n_reactors.float() / common.MAX_PRODUCTION_CAPACITY
    ).clamp(min=0.0, max=1.0)
    state[:, prod_slice] = torch.minimum(state[:, prod_slice], cap_ratio.unsqueeze(1))

    storage_energy = (
        n_storage.float() * common.MODULE_CAPACITY_MWH / common.MAX_SOC_CAPACITY
    ).clamp(min=0.0, max=1.0)
    state[:, soc_slice] = torch.minimum(state[:, soc_slice], storage_energy.unsqueeze(1))
    return state


def simulated_annealing(
    num_chains: int,
    steps: int,
    total_time: float,
    cont_temp: float,
    profit_temp: float,
    temp_schedule: str,
    device: torch.device,
    seed: int | None = None,
):
    if seed is not None:
        torch.manual_seed(seed)

    states = enforce_physical_limits(noise.sample_uniform_noise(num_chains, device=device))
    init_mask = torch.zeros(num_chains, dtype=torch.bool, device=device)

    times = torch.linspace(0.0, total_time, steps + 1)
    if temp_schedule == "linear":
        temp_series = torch.linspace(profit_temp, max(profit_temp * 0.01, 1e-3), steps + 1, device=device)
    else:
        temp_series = torch.full((steps + 1,), profit_temp, device=device)
    profit_history = []
    acceptance = 0

    for step in range(steps + 1):
        profits = decode_profit(states).to(device)
        profit_history.append(profits.detach().cpu())
        if step == steps:
            break

        proposal = states.clone()
        cont = proposal[:, common.DISCRETE_DIM :]
        cont = cont + cont_temp * torch.randn_like(cont)
        cont = torch.clamp(cont, 0.0, 1.0)
        proposal[:, common.DISCRETE_DIM :] = cont

        offset = 0
        blocks = [common.NUM_REACTOR_MODELS, common.NUM_REACTOR_COUNT_OPTIONS, common.NUM_STORAGE_OPTIONS]
        for block_size in blocks:
            block_slice = slice(offset, offset + block_size)
            block = proposal[:, block_slice]
            new_block = random_discrete_step(block)
            flip_mask = (torch.rand(block.size(0), device=device) > 0.5).unsqueeze(1)
            proposal[:, block_slice] = torch.where(flip_mask, new_block, block)
            offset += block_size

        proposal = enforce_physical_limits(proposal)
        profit_cur = decode_profit(states).to(device)
        profit_prop = decode_profit(proposal).to(device)
        delta = profit_prop - profit_cur
        log_accept = torch.clamp(delta / temp_series[step], max=50.0)
        log_u = torch.log(torch.rand(states.size(0), device=device))
        accept = log_u < log_accept
        acceptance += accept.sum().item()
        states = torch.where(accept[:, None], proposal, states)

    acceptance_rate = acceptance / (steps * states.size(0)) if steps > 0 else None
    return times, temp_series.cpu(), torch.stack(profit_history, dim=0), acceptance_rate


def plot_profit(times: torch.Tensor, profits: torch.Tensor, temp_series: torch.Tensor, output: Path):
    t_norm = times.numpy()
    t_norm = t_norm / (t_norm[-1] if t_norm[-1] > 0 else 1.0)
    profit_np = profits.numpy()

    plt.figure(figsize=(7, 4))
    mean_profit = profit_np.mean(axis=1)
    std_profit = profit_np.std(axis=1)
    plt.plot(t_norm, mean_profit, color="#d95f02", linewidth=2, label="Mean profit")
    plt.fill_between(t_norm, mean_profit - std_profit, mean_profit + std_profit, color="#f6d2b8", alpha=0.4, label="Mean ± 1σ")
    label_drawn = False
    for idx in range(profit_np.shape[1]):
        label = "Chains (init: uniform noise)" if not label_drawn else None
        plt.plot(t_norm, profit_np[:, idx], color="#5c6bc0", alpha=0.15, label=label)
        label_drawn = True

    best_profit_idx = np.unravel_index(np.argmax(profit_np, axis=None), profit_np.shape)
    best_time = t_norm[best_profit_idx[0]]
    best_value = profit_np[best_profit_idx]
    plt.plot([best_time], [best_value], marker="o", color="#2e7d32", label="Best performer")
    plt.text(0.5, best_value * 1.05, f"Best: {best_value:,.0f} EUR", color="#2e7d32", ha="center", va="bottom")
    plt.xlabel("Normalized time")
    plt.ylabel("Annual profit (EUR)")
    plt.title("Simulated annealing")
    plt.axhline(0.0, color="#424242", linestyle=":", linewidth=1.2, label="Profitability threshold")
    plt.grid(True, linestyle=":", alpha=0.4)
    ax = plt.gca()
    ax_temp = ax.twinx()
    ax_temp.plot(t_norm, temp_series.numpy(), color="#8e24aa", linestyle="--", linewidth=1.2, label="Profit temperature")
    ax_temp.set_ylabel("Profit temperature (EUR)")
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax_temp.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="best")
    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Profit-only simulated annealing benchmark")
    parser.add_argument("--num-chains", type=int, default=16)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--total-time", type=float, default=1.0)
    parser.add_argument("--cont-temp", type=float, default=0.05)
    parser.add_argument("--profit-temp", type=float, default=1e7)
    parser.add_argument("--temp-schedule", type=str, choices=["constant", "linear"], default="constant")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    if not args.output:
        args.output = f"figures/solver_simulated_annealing_temp_{args.temp_schedule}.png"

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    times, temp_series, profits, acc = simulated_annealing(
        num_chains=args.num_chains,
        steps=args.steps,
        total_time=args.total_time,
        cont_temp=args.cont_temp,
        profit_temp=args.profit_temp,
        temp_schedule=args.temp_schedule,
        device=device,
        seed=args.seed,
    )

    plot_profit(times, profits, temp_series, Path(args.output))
    print(f"Saved profit-only simulated annealing plot to {args.output}")
    if acc is not None:
        print(f"[info] Acceptance rate: {acc:.4f}")


if __name__ == "__main__":
    main()
