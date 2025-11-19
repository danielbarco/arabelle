"""
Hybrid Langevin + Gibbs-with-gradients sampler for SMR configurations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from . import common, noise


@dataclass
class TrajectoryResult:
    times: torch.Tensor  # (steps + 1,)
    energies: torch.Tensor  # (steps + 1, num_chains)
    states: torch.Tensor  # (num_chains, feature_dim)
    init_mask: torch.Tensor  # (num_chains,) bool (True=data, False=noise)
    state_history: torch.Tensor | None = None


def _dlangevin_update(
    block: torch.Tensor, grad_block: torch.Tensor, beta: float, lambda_penalty: float
) -> torch.Tensor:
    """
    Sample new one-hot block using discrete Langevin logits with change penalty.
    """
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


def simulate(
    model: torch.nn.Module,
    dataset_tensor: torch.Tensor,
    *,
    num_data_chains: int = 8,
    num_noise_chains: int = 8,
    delta_t: float = 0.01,
    steps: int | None = None,
    total_time: float = 2.0,
    beta: float = 5.0,
    lambda_penalty: float = 2.0,
    device: torch.device | None = None,
    seed: int | None = None,
    record_states: bool = False,
) -> TrajectoryResult:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if seed is not None:
        torch.manual_seed(seed)

    model = model.to(device)
    prev_mode = model.training
    model.eval()

    dataset_tensor = dataset_tensor.to(device)
    data_idx = torch.randperm(dataset_tensor.size(0))[:num_data_chains]
    data_states = dataset_tensor[data_idx].clone()
    noise_states = noise.sample_uniform_noise(num_noise_chains, device=device)

    states = torch.cat([data_states, noise_states], dim=0)
    init_mask = torch.cat(
        [torch.ones(num_data_chains, dtype=torch.bool), torch.zeros(num_noise_chains, dtype=torch.bool)],
        dim=0,
    ).to(device)
    num_chains = states.size(0)

    if steps is None:
        steps = max(1, int(math.ceil(total_time / max(delta_t, 1e-6))))
    dt = total_time / steps
    sqrt_coeff = math.sqrt(2.0 * dt)

    energy_history = []
    times = []
    state_history = [] if record_states else None

    total_proposed_discrete = 0
    total_changed_discrete = 0
    total_accepts = 0
    total_proposals = 0
    proposal_magnitude_sum = 0.0
    accepted_magnitude_sum = 0.0

    for step in range(steps + 1):
        times.append(step * dt)
        with torch.no_grad():
            energy_vals = model(states)
        energy_history.append(energy_vals.detach().cpu())
        if state_history is not None:
            state_history.append(states.detach().cpu())
        if step == steps:
            break

        base_state = states.detach()
        base_state.requires_grad_(True)
        with torch.enable_grad():
            energies = model(base_state)
            grads = torch.autograd.grad(energies.sum(), base_state)[0]
        current_energy = energies.detach()
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

        diff = proposal - prev_state
        proposal_magnitude_sum += diff.norm(dim=1).sum().item()
        total_proposals += diff.size(0)

        with torch.no_grad():
            proposal_energy = model(proposal)
        delta_energy = proposal_energy - current_energy
        log_accept = torch.clamp(-beta * delta_energy, max=0.0)
        log_u = torch.log(torch.rand(delta_energy.shape, device=device))
        accept = log_u < log_accept

        if accept.any():
            accepted_magnitude_sum += diff[accept].norm(dim=1).sum().item()
        total_accepts += accept.sum().item()

        new_state = torch.where(accept[:, None], proposal.detach(), prev_state)
        states = new_state

        offset = 0
        for block_size in blocks:
            block_slice = slice(offset, offset + block_size)
            before = prev_state[:, block_slice]
            after = states[:, block_slice]
            changed = (after.argmax(dim=1) != before.argmax(dim=1)).sum().item()
            total_changed_discrete += changed
            total_proposed_discrete += after.size(0)
            offset += block_size

    energy_tensor = torch.stack(energy_history, dim=0)
    time_tensor = torch.tensor(times, dtype=torch.float32)
    if total_proposals > 0:
        acceptance_rate = total_accepts / (total_proposals)
        avg_prop_mag = proposal_magnitude_sum / total_proposals
        avg_acc_mag = accepted_magnitude_sum / max(total_accepts, 1)
        print(
            "[solver] Acceptance rate {:.4f} | avg proposal step {:.4f} | avg accepted step {:.4f}".format(
                acceptance_rate, avg_prop_mag, avg_acc_mag
            )
        )
    if total_proposed_discrete > 0:
        avg_discrete = total_changed_discrete / total_proposed_discrete
        avg_moves_per_step = total_changed_discrete / max(steps, 1)
        print(f"[solver] Avg discrete change rate: {avg_discrete:.4f} ; Avg moves/step: {avg_moves_per_step:.2f}")
    history_tensor = torch.stack(state_history, dim=0) if state_history is not None else None
    if prev_mode:
        model.train()
    return TrajectoryResult(time_tensor, energy_tensor, states.cpu(), init_mask.cpu(), history_tensor)
