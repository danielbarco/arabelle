"""
Solver variant that uses energy gradients for proposals but Metropolis based on annual profit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from . import common, noise
from minlp_smr_battery_storage import objective, x_from_var


@dataclass
class ProfitTrajectory:
    times: torch.Tensor
    energies: torch.Tensor
    profits: torch.Tensor
    states: torch.Tensor
    init_mask: torch.Tensor
    acceptance_rate: float | None = None


def _dlangevin_update(
    block: torch.Tensor, grad_block: torch.Tensor, beta: float, lambda_penalty: float
) -> torch.Tensor:
    current_idx = block.argmax(dim=1, keepdim=True)
    grad_cur = torch.gather(grad_block, 1, current_idx)
    base = -beta * (grad_block - grad_cur)
    logits = base - lambda_penalty
    logits.scatter_(1, current_idx, torch.zeros_like(current_idx, dtype=logits.dtype))
    dist = Categorical(logits=logits)
    samples = dist.sample()
    stay_mask = samples.eq(current_idx.squeeze(1))
    new_block = F.one_hot(samples, block.size(1)).float()
    if stay_mask.any():
        new_block[stay_mask] = block[stay_mask]
    return new_block


def _compute_profit(encoded: torch.Tensor) -> torch.Tensor:
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


def simulate(
    model: torch.nn.Module,
    dataset_tensor: torch.Tensor,
    *,
    num_data_chains: int = 8,
    num_noise_chains: int = 8,
    delta_t: float = 0.01,
    total_time: float = 2.0,
    beta: float = 5.0,
    lambda_penalty: float = 2.0,
    profit_temp: float = 1e7,
    device: torch.device | None = None,
    seed: int | None = None,
    initial_states: torch.Tensor | None = None,
    initial_mask: torch.Tensor | None = None,
    steps_override: int | None = None,
) -> ProfitTrajectory:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if seed is not None:
        torch.manual_seed(seed)

    prev_mode = model.training
    model = model.to(device)
    model.eval()

    dataset_tensor = dataset_tensor.to(device)

    if initial_states is not None:
        states = initial_states.to(device).clone()
        if initial_mask is None:
            raise ValueError("initial_mask must be provided when initial_states is set")
        init_mask = initial_mask.to(device).clone()
    else:
        profit_dataset = _compute_profit(dataset_tensor).to(device)
        positive_idx = torch.nonzero(profit_dataset > 0, as_tuple=False).flatten()
        if positive_idx.numel() >= num_data_chains:
            perm = positive_idx[torch.randperm(positive_idx.numel())[:num_data_chains]]
        else:
            perm = torch.randperm(dataset_tensor.size(0))[:num_data_chains]
        data_states = dataset_tensor[perm].clone()
        noise_states = noise.sample_uniform_noise(num_noise_chains, device=device)
        states = torch.cat([data_states, noise_states], dim=0)
        init_mask = torch.cat(
            [torch.ones(num_data_chains, dtype=torch.bool), torch.zeros(num_noise_chains, dtype=torch.bool)],
            dim=0,
        ).to(device)
    num_chains = states.size(0)

    steps = steps_override if steps_override is not None else max(1, int(math.ceil(total_time / max(delta_t, 1e-6))))
    dt = total_time / steps
    sqrt_coeff = math.sqrt(2.0 * dt)

    energy_history = []
    profit_history = []
    times = []
    total_accepts = 0
    total_proposals = 0

    for step in range(steps + 1):
        times.append(step * dt)
        with torch.no_grad():
            energies = model(states)
        profits = _compute_profit(states)
        energy_history.append(energies.detach().cpu())
        profit_history.append(profits.detach().cpu())
        if step == steps:
            break

        base_state = states.detach()
        base_state.requires_grad_(True)
        with torch.enable_grad():
            energies = model(base_state)
            grads = torch.autograd.grad(energies.sum(), base_state)[0]
        prev_state = base_state.detach()
        proposal = prev_state.clone()

        cont = proposal[:, common.DISCRETE_DIM :]
        grad_cont = grads[:, common.DISCRETE_DIM :]
        noise_term = torch.randn_like(cont)
        cont = cont - dt * grad_cont + sqrt_coeff * noise_term
        cont = torch.clamp(cont, 0.0, 1.0)
        proposal[:, common.DISCRETE_DIM :] = cont

        offset = 0
        blocks = [
            common.NUM_REACTOR_MODELS,
            common.NUM_REACTOR_COUNT_OPTIONS,
            common.NUM_STORAGE_OPTIONS,
        ]
        for block_size in blocks:
            block_slice = slice(offset, offset + block_size)
            block = prev_state[:, block_slice]
            grad_block = grads[:, block_slice]
            new_block = _dlangevin_update(block, grad_block, beta, lambda_penalty)
            proposal[:, block_slice] = new_block
            offset += block_size

        profit_current = _compute_profit(prev_state).to(device)
        profit_proposal = _compute_profit(proposal).to(device)
        delta_profit = profit_proposal - profit_current
        log_accept = torch.clamp(delta_profit / profit_temp, max=50.0)
        log_u = torch.log(torch.rand(num_chains, device=device))
        accept = log_u < log_accept.to(device)
        total_accepts += accept.sum().item()
        total_proposals += accept.numel()

        states = torch.where(accept[:, None], proposal.detach(), prev_state)

    energy_tensor = torch.stack(energy_history, dim=0)
    profit_tensor = torch.stack(profit_history, dim=0)
    time_tensor = torch.tensor(times, dtype=torch.float32)

    if prev_mode:
        model.train()

    acceptance_rate = total_accepts / total_proposals if total_proposals else None
    return ProfitTrajectory(time_tensor, energy_tensor, profit_tensor, states.cpu(), init_mask.cpu(), acceptance_rate)
