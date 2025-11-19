"""
Simple simulated annealing sampler using random proposals and profit-based acceptance.
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
class SATrajectory:
    times: torch.Tensor
    energies: torch.Tensor
    profits: torch.Tensor
    states: torch.Tensor
    init_mask: torch.Tensor
    acceptance_rate: float | None = None
    profit_temps: torch.Tensor | None = None
    guide_strengths: torch.Tensor | None = None


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


def _random_discrete_step(block: torch.Tensor) -> torch.Tensor:
    num_classes = block.size(1)
    current_idx = block.argmax(dim=1)
    new_idx = torch.randint(0, num_classes, (block.size(0),), device=block.device)
    same = new_idx.eq(current_idx)
    if same.any():
        new_idx[same] = (new_idx[same] + 1) % num_classes
    return F.one_hot(new_idx, num_classes=num_classes).float()


def _local_discrete_step(block: torch.Tensor, window: int = 2) -> torch.Tensor:
    num_classes = block.size(1)
    current_idx = block.argmax(dim=1)
    direction = torch.randint(-window, window + 1, current_idx.shape, device=block.device)
    new_idx = (current_idx + direction).clamp(0, num_classes - 1)
    return F.one_hot(new_idx, num_classes=num_classes).float()


def _dlangevin_update(block: torch.Tensor, grad_block: torch.Tensor, beta: float, lambda_penalty: float) -> torch.Tensor:
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

def _build_schedule(
    initial: float,
    final_ratio: float,
    steps: int,
    device: torch.device,
    mode: str,
    *,
    drop_after_half: bool = False,
) -> torch.Tensor:
    if steps <= 0:
        return torch.empty(0, device=device)
    mode = mode.lower()
    if mode not in {"constant", "linear"}:
        raise ValueError(f"Unknown schedule mode '{mode}'.")
    final_value = float(initial) * float(final_ratio)
    if mode == "constant" or abs(final_value - initial) < 1e-12:
        schedule = torch.full((steps,), float(initial), device=device, dtype=torch.float32)
    else:
        schedule = torch.linspace(float(initial), final_value, steps, device=device, dtype=torch.float32)
    if drop_after_half and steps > 0:
        half_count = max(1, int(math.ceil(0.5 * steps)))
        first_half = torch.linspace(float(initial), 0.0, half_count, device=device, dtype=torch.float32)
        zero_tail = torch.zeros(max(0, steps - half_count), device=device, dtype=torch.float32)
        schedule = torch.cat([first_half, zero_tail], dim=0)
    return schedule


def simulate(
    model: torch.nn.Module,
    dataset_tensor: torch.Tensor,
    *,
    num_data_chains: int = 8,
    num_noise_chains: int = 8,
    steps: int = 500,
    total_time: float = 1.0,
    cont_temp: float = 0.05,
    profit_temp: float = 1e7,
    profit_schedule: str = "constant",
    profit_final_ratio: float = 0.01,
    proposal_mode: str = "guided",
    guide_strength: float = 0.02,
    guide_prob: float = 0.7,
    guide_beta: float = 10.0,
    guide_lambda: float = 1.0,
    guide_schedule: str = "constant",
    guide_final_ratio: float = 0.1,
    device: torch.device | None = None,
    seed: int | None = None,
    initial_states: torch.Tensor | None = None,
    initial_mask: torch.Tensor | None = None,
) -> SATrajectory:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if seed is not None:
        torch.manual_seed(seed)

    proposal_mode = proposal_mode.lower()
    if proposal_mode not in {"guided", "random"}:
        raise ValueError(f"Unknown proposal_mode '{proposal_mode}' (expected 'guided' or 'random')")

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
        profits_dataset = _compute_profit(dataset_tensor).to(device)
        positive_idx = torch.nonzero(profits_dataset > 0, as_tuple=False).flatten()
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

    times = torch.linspace(0.0, total_time, steps + 1)
    profit_temps = _build_schedule(profit_temp, profit_final_ratio, steps, device, profit_schedule)
    guide_strengths = _build_schedule(
        guide_strength,
        guide_final_ratio,
        steps,
        device,
        guide_schedule,
        drop_after_half=True,
    )
    energy_history = []
    profit_history = []
    total_accepts = 0
    total_proposals = 0

    guide_enabled = proposal_mode == "guided" and ((guide_strength > 0.0) or (guide_prob > 0.0))
    initial_guide_strength = guide_strength
    random_override = False
    guide_switch_step = max(0, int(math.ceil(0.5 * max(steps, 1))))

    for step in range(steps + 1):
        if proposal_mode == "guided" and not random_override and step >= guide_switch_step:
            random_override = True
        current_mode = "random" if (random_override or proposal_mode == "random") else "guided"
        compute_grad = (current_mode == "guided") and guide_enabled and step < steps
        if compute_grad:
            base_state = states.detach().clone().requires_grad_(True)
            energy_vals = model(base_state)
            grads = torch.autograd.grad(energy_vals.sum(), base_state)[0]
            energies = energy_vals.detach()
            curr_state = base_state.detach()
        else:
            with torch.no_grad():
                energies = model(states)
            grads = None
            curr_state = states.detach()

        profits = _compute_profit(curr_state).to(device)
        energy_history.append(energies.detach().cpu())
        profit_history.append(profits.detach().cpu())
        if step == steps:
            break

        proposal = curr_state.clone()
        cont = proposal[:, common.DISCRETE_DIM :]
        current_guide = guide_strengths[step] if guide_strengths.numel() > 0 else torch.tensor(guide_strength, device=device)
        current_guide_val = current_guide.item() if current_guide.numel() > 0 else float(guide_strength)
        if grads is not None and current_mode == "guided" and current_guide_val > 0.0:
            cont = cont - current_guide * grads[:, common.DISCRETE_DIM :]
        cont = cont + cont_temp * torch.randn_like(cont)
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
            block = curr_state[:, block_slice]
            if current_mode == "random":
                candidate = _local_discrete_step(block)
            else:
                random_block = _random_discrete_step(block)
                if grads is not None and guide_prob > 0.0:
                    grad_block = grads[:, block_slice]
                    guided_block = _dlangevin_update(block, grad_block, guide_beta, guide_lambda)
                    use_guided = (torch.rand(block.size(0), device=device) < guide_prob).unsqueeze(1)
                    candidate = torch.where(use_guided, guided_block, random_block)
                else:
                    candidate = random_block
            update_mask = (torch.rand(block.size(0), device=device) > 0.5).unsqueeze(1)
            proposal[:, block_slice] = torch.where(update_mask, candidate, block)
            offset += block_size

        profit_cur = profits
        profit_prop = _compute_profit(proposal).to(device)
        delta = profit_prop - profit_cur
        temp_value = profit_temps[step] if profit_temps.numel() > 0 else torch.tensor(profit_temp, device=device)
        log_accept = torch.clamp(delta / temp_value, max=50.0)
        log_u = torch.log(torch.rand(states.size(0), device=device))
        accept = log_u < log_accept
        total_accepts += accept.sum().item()
        total_proposals += accept.numel()
        states = torch.where(accept[:, None], proposal.detach(), curr_state)
        states = states.detach()

    if prev_mode:
        model.train()

    acceptance_rate = total_accepts / total_proposals if total_proposals else None
    return SATrajectory(
        times=times,
        energies=torch.stack(energy_history, dim=0),
        profits=torch.stack(profit_history, dim=0),
        states=states.cpu(),
        init_mask=init_mask.cpu(),
        acceptance_rate=acceptance_rate,
        profit_temps=profit_temps.cpu() if profit_temps.numel() > 0 else None,
        guide_strengths=guide_strengths.cpu() if guide_strengths.numel() > 0 else None,
    )
