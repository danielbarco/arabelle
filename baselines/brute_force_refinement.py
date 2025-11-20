#!/usr/bin/env python

"""
© 2025, Arabelle Solutions and/or its affiliates. All rights reserved.

MINLP sample: Nuclear SMR with battery storage modules.
--- HIGH COMPUTE / MULTI-VARIABLE SEARCH VERSION ---

Changes from original:
1. Expanded hardware search ranges (Storage -> 1000, Reactors -> 30).
2. Added 'base_load_bias' variable: Determines if we target 90%, 100%, or 110% of avg demand.
3. Added 'initial_soc' variable: Optimizes the starting battery charge.
4. 3D Heuristic Grid Search (Smoothness x Bias x SOC).
"""

import matplotlib.pyplot as plt
import numpy as np
import time
import itertools
import random

random.seed(2025)
np.random.seed(2025)

# ------------------------- Problem data -------------------------------------
horizon = 24  # hourly horizon

# Hourly electric demand (MW) - daily profile
electric_demand = np.array(
    [
        160,
        152,
        144,
        140,
        144,
        160,
        200,
        240,
        280,
        300,
        320,
        340,
        360,
        368,
        352,
        340,
        320,
        312,
        300,
        280,
        260,
        240,
        220,
        192,
    ],
    dtype=np.float64,
)

# SMR models power offerings
reactor_models = [80.0, 160.0, 300.0, 350.0, 470.0]

# Economic parameters
interest_rate = 0.04

# Reactor capital cost (power-law, economies of scale)
reactor_cap_a = 2.0e7
reactor_cap_b = 0.8
reactor_years = 60
reactor_annuity_factor = interest_rate / (1 - (1 + interest_rate) ** (-reactor_years))

# Storage module
module_capacity_mwh = 50.0
module_power_mw = 10.0
module_cost = 1.0e7
storage_leakage_per_hour = 0.0008
storage_years = 20
storage_annuity_factor = interest_rate / (1 - (1 + interest_rate) ** (-storage_years))

# Operational costs
reactor_fixed_om_frac = 0.03
storage_fixed_om_frac = 0.02
fuel_price = 5.0

# Market pricing
price_base = 70.0
price_sensitivity = 60.0
price_surplus = 10.0


# Storage efficiencies
def charge_efficiency(charge_mw: np.ndarray, max_charge_mw: float) -> np.ndarray:
    if max_charge_mw < 1.0e-6:
        return np.ones(horizon)
    else:
        safe_max_charge = np.maximum(max_charge_mw, 1e-6)
        r = 0.95 - 0.15 * (charge_mw / safe_max_charge) ** 2
        return np.maximum(r, 0.6)


def discharge_efficiency(dis_mw: np.ndarray, max_dis_mw: float) -> np.ndarray:
    if max_dis_mw < 1.0e-6:
        return np.ones(horizon)
    else:
        safe_max_dis = np.maximum(max_dis_mw, 1e-6)
        r = 0.96 - 0.2 * (dis_mw / safe_max_dis) ** 2
    return np.maximum(r, 0.55)


# ------------------------- Model evaluation functions -----------------------
def capital_cost_reactor(reactor_capacity: float, n_reactor: int) -> float:
    return n_reactor * (reactor_cap_a * reactor_capacity ** reactor_cap_b)


def capital_cost_storage(n_storage: int) -> float:
    return module_cost * n_storage


def market_price(demand: np.ndarray) -> np.ndarray:
    peak = np.max(demand)
    return price_base + price_sensitivity * demand / peak


def electricity_supplied(reactor_production: np.ndarray, n_storage: int, dis: np.ndarray, ch: np.ndarray) -> np.ndarray:
    max_storage_power = n_storage * module_power_mw
    eff_d = discharge_efficiency(dis, max_storage_power)
    eff_c = charge_efficiency(ch, max_storage_power)
    return reactor_production + dis * eff_d - ch / eff_c


def compute_charge_discharge(soc: np.ndarray) -> tuple[np.ndarray]:
    soc_prev = np.roll(soc, 1)
    charge = soc - soc_prev * (1 - storage_leakage_per_hour)
    return np.maximum(0.0, charge), np.maximum(0.0, -charge)


def var_from_x(x: np.ndarray) -> tuple:
    reactor_production = x[:horizon]
    soc = x[horizon: 2 * horizon]
    reactor_model = int(round(x[-3]))
    reactor_model = np.clip(reactor_model, 0, len(reactor_models) - 1)
    n_reactor = int(round(x[-2]))
    n_storage = int(round(x[-1]))
    return reactor_model, n_reactor, n_storage, reactor_production, soc


def x_from_var(reactor_model: int, n_reactor: int, n_storage: int, reactor_production: np.ndarray,
               soc: np.ndarray) -> np.ndarray:
    x = np.empty(2 * horizon + 3)
    x[:horizon] = reactor_production
    x[horizon: 2 * horizon] = soc
    x[-3] = reactor_model
    x[-2] = n_reactor
    x[-1] = n_storage
    return x


def objective(x: np.ndarray) -> float:
    reactor_model, n_reactor, n_storage, reactor_production, soc = var_from_x(x)
    if n_reactor == 0 and n_storage == 0: return 0.0

    if n_reactor == 0:
        ann_om_capex = capital_cost_storage(n_storage) * (storage_annuity_factor + storage_fixed_om_frac)
    else:
        reactor_capacity = reactor_models[reactor_model]
        cap_reactor = capital_cost_reactor(reactor_capacity, n_reactor)
        cap_storage = capital_cost_storage(n_storage)
        ann_capex = (cap_reactor * reactor_annuity_factor + cap_storage * storage_annuity_factor)
        annual_fixed_om = (cap_reactor * reactor_fixed_om_frac + cap_storage * storage_fixed_om_frac)
        ann_om_capex = ann_capex + annual_fixed_om

    daily_fuel_cost = np.sum(reactor_production) * fuel_price
    ch, dis = compute_charge_discharge(soc)
    supplied = electricity_supplied(reactor_production, n_storage, dis, ch)
    local_supply = np.minimum(supplied, electric_demand)
    unmet = np.maximum(0.0, electric_demand - supplied)
    surplus = np.maximum(0.0, supplied - electric_demand)
    price = market_price(electric_demand)
    net_market = price * local_supply - price * unmet - price_surplus * surplus
    daily_profit = np.sum(net_market) - daily_fuel_cost
    return daily_profit * 365.0 - ann_om_capex


def constraints_residuals(x: np.ndarray) -> list[float]:
    reactor_model, n_reactor, n_storage, reactor_production, soc = var_from_x(x)
    res = []
    plant_capacity = reactor_models[reactor_model] * n_reactor
    for t in range(horizon):
        res.append(plant_capacity - reactor_production[t])
        res.append(reactor_production[t])
    max_storage_energy = n_storage * module_capacity_mwh
    max_storage_power = n_storage * module_power_mw
    if n_storage == 0:
        max_storage_energy = 0.0
        max_storage_power = 0.0
    ch, dis = compute_charge_discharge(soc)
    for t in range(horizon):
        res.append(max_storage_power - ch[t])
        res.append(max_storage_power - dis[t])
        res.append(max_storage_energy - soc[t])
        res.append(soc[t])
    res.append(n_reactor)
    res.append(n_storage)
    return res


# ----------------------------------------------------------------------------
# UPDATED BUILD CANDIDATE: Accepts more variables
# ----------------------------------------------------------------------------
def build_candidate_with_storage(reactor_model: int,
                                 n_reactor: int,
                                 n_storage: int,
                                 variability_smoothness: float,
                                 base_load_bias: float,
                                 initial_soc_frac: float) -> np.ndarray:
    demand = electric_demand.copy()

    # --- Variable 1: Base Load Bias ---
    # Allows the solver to intentionally target over-production (e.g. 1.1x) or under-production (0.9x)
    avg = np.mean(demand) * base_load_bias

    reactor_production = (variability_smoothness * np.full(horizon, avg) + (1 - variability_smoothness) * demand)
    plant_capacity = reactor_models[reactor_model] * n_reactor
    reactor_production = np.clip(reactor_production, 0, plant_capacity)

    soc = np.zeros(horizon)
    max_power = n_storage * module_power_mw
    max_energy = n_storage * module_capacity_mwh

    if max_energy > 0:
        # --- Variable 2: Initial SOC ---
        # Allows the solver to optimize the boundary condition
        soc_prev = initial_soc_frac * max_energy
    else:
        soc_prev = 0.0

    for t in range(horizon):
        mismatch = reactor_production[t] - demand[t]
        ch_t, dis_t = 0.0, 0.0
        if mismatch > 0:
            available_power = min(mismatch, max_power)
            remaining_energy_cap = max(0, max_energy - soc_prev)
            ch_t = min(available_power, remaining_energy_cap)
        else:
            need = -mismatch
            discharge_power = min(need, max_power)
            dis_t = min(discharge_power, soc_prev)
        soc[t] = soc_prev * (1 - storage_leakage_per_hour) + ch_t - dis_t
        soc[t] = np.clip(soc[t], 0, max_energy)
        soc_prev = soc[t]

    return x_from_var(reactor_model, n_reactor, n_storage, reactor_production, soc)


def build_candidate_wo_storage(reactor_model: int, n_reactor: int) -> np.ndarray:
    plant_capacity = reactor_models[reactor_model] * n_reactor
    reactor_production = np.minimum(plant_capacity, electric_demand)
    soc = np.zeros(horizon)
    return x_from_var(reactor_model, n_reactor, 0, reactor_production, soc)


def evaluate_candidate(cand, title):
    print(f"\n--- Evaluating candidate '{title}' ---")
    obj = objective(cand)
    res = constraints_residuals(cand)
    min_res = np.min(res)
    print(f"Annual Profit (EUR): {obj:,.2f}")
    print(f"Min Inequality Residual (>=0 for feasibility): {min_res:.6f}")

    hours = np.arange(horizon)
    reactor_model, n_reactor, n_storage, reactor_production, soc = var_from_x(cand)
    ch, dis = compute_charge_discharge(soc)
    net_supply = electricity_supplied(reactor_production, n_storage, dis, ch)

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(hours, electric_demand, "k--", label="Electric demand")
    ax.plot(hours, reactor_production, "b-", label="Reactor output")
    ax.plot(hours, dis, "r-", label="Storage discharge")
    ax.plot(hours, ch, "g-", label="Storage charge")
    ax.plot(hours, net_supply, "m-", label="Net supply to grid")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Power (MW)")
    fig.suptitle(title, fontsize=16)

    config_title = f"Model={reactor_model} ({reactor_models[reactor_model]} MW), N_Reactor={n_reactor}, N_Storage={n_storage}"
    ax.set_title(config_title, fontsize="medium")

    ax2 = ax.twinx()
    ax2.plot(hours, soc, "c.", label="SOC")
    ax2.set_ylabel("State of Charge (MWh)", color="c")
    ax2.tick_params(axis="y", labelcolor="c")
    if n_storage > 0:
        max_energy = n_storage * module_capacity_mwh
        ax2.set_ylim(bottom=0, top=max_energy * 1.1)
    else:
        ax2.set_ylim(bottom=0, top=1)
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="upper left", fontsize="small")
    ax.grid(True, ls=":", color="black")
    plt.tight_layout()


# --- MODIFIED: Function now handles the 3D operational parameter grid ---
def run_search_grid(
        model_range,
        reactor_range,
        storage_range,
        smoothness_params,
        bias_params,
        init_soc_params,
        search_name="Search",
        return_all_results=False
):
    start_time = time.time()
    local_best_profit = -np.inf
    local_best_x = None
    local_best_title = ""
    results = []

    model_list = list(model_range)
    reactor_list = list(reactor_range)
    storage_list = list(storage_range)

    # Calculate total combinations for progress bar
    n_hardware = len(model_list) * len(reactor_list) * len(storage_list)
    n_heuristics = len(smoothness_params) * len(bias_params) * len(init_soc_params)

    print(f"\n--- Starting {search_name} ---")
    print(f"Hardware Grid: {len(model_list)} models x {len(reactor_list)} reactors x {len(storage_list)} storage.")
    print(f"Operational Grid: {len(smoothness_params)} smooth x {len(bias_params)} bias x {len(init_soc_params)} soc.")
    print(f"Total Combinations: {n_hardware * n_heuristics:,}")

    count = 0

    # Create operational parameter grid (Cartesian product)
    # This generates all combinations of (smoothness, bias, init_soc)
    op_grid = list(itertools.product(smoothness_params, bias_params, init_soc_params))

    for model_idx in model_list:
        for n_r in reactor_list:
            for n_s in storage_list:

                count += 1
                if count % 500 == 0:
                    elapsed = time.time() - start_time
                    print(f"\r[{search_name}] HW Config {count}/{n_hardware} (Best: {local_best_profit:,.0f})", end="")

                if n_r == 0 and n_s == 0: continue

                current_config_best_profit = -np.inf
                current_config_best_x = None

                # If no storage, we only run once (heuristics don't apply/are simplified)
                if n_s == 0:
                    cand_x = build_candidate_wo_storage(model_idx, n_r)
                    if np.min(constraints_residuals(cand_x)) >= -1e-6:
                        profit = objective(cand_x)
                        current_config_best_profit = profit
                        current_config_best_x = cand_x
                else:
                    # Check all operational combinations for this hardware setup
                    for (sm, bi, iso) in op_grid:
                        cand_x = build_candidate_with_storage(model_idx, n_r, n_s, sm, bi, iso)

                        if np.min(constraints_residuals(cand_x)) < -1e-6: continue
                        profit = objective(cand_x)

                        if profit > current_config_best_profit:
                            current_config_best_profit = profit
                            current_config_best_x = cand_x

                if current_config_best_profit > -np.inf:
                    if return_all_results:
                        results.append((current_config_best_profit, current_config_best_x, model_idx, n_r, n_s))

                    if current_config_best_profit > local_best_profit:
                        local_best_profit = current_config_best_profit
                        local_best_x = current_config_best_x
                        local_best_title = f"M={model_idx}, R={n_r}, S={n_s}"
                        print(f"\n--- New Best ({search_name}): {local_best_profit:,.2f} ({local_best_title}) ---")

    elapsed = time.time() - start_time
    print(f"\n--- {search_name} Complete ({elapsed:.2f}s) ---")

    if return_all_results:
        return results, local_best_profit, local_best_x, local_best_title
    else:
        return local_best_profit, local_best_x, local_best_title


if __name__ == "__main__":
    plt.rcParams["font.family"] = "serif"
    overall_start_time = time.time()

    # =========================================================================
    # 1. DENSE COARSE SEARCH (EXPANDED VARIABLES)
    # =========================================================================

    # EXPANDED HARDWARE RANGES
    N_REACTOR_MAX = 30  # Was 20
    N_STORAGE_MAX = 1000  # Was 400

    stage1_reactor = range(0, N_REACTOR_MAX + 1, 1)
    stage1_storage = range(0, N_STORAGE_MAX + 1, 10)  # Step 10 to keep Stage 1 tractable

    # NEW OPERATIONAL VARIABLES (Coarse grid)
    # 1. Smoothness: 0.0 (Follow demand) to 1.0 (Flat)
    s1_smooth = np.linspace(0.01, 1.0, 10)
    # 2. Bias: 0.9 (Underproduce) to 1.1 (Overproduce)
    s1_bias = [0.9, 1.0, 1.05, 1.1]
    # 3. Init SOC: Start empty or start with buffer
    s1_soc = [0.0, 0.2, 0.5]

    print(">>> STAGE 1: MULTI-VARIABLE GRID SEARCH <<<")
    results_list, _, _, _ = run_search_grid(
        range(len(reactor_models)),
        stage1_reactor,
        stage1_storage,
        s1_smooth,
        s1_bias,
        s1_soc,
        search_name="Stage 1",
        return_all_results=True
    )

    # =========================================================================
    # 2. SEED SELECTION
    # =========================================================================
    results_list.sort(key=lambda x: x[0], reverse=True)
    seeds_to_refine = set()

    # Standard manual heuristics
    seeds_to_refine.add((2, 1, 12))
    seeds_to_refine.add((3, 1, 0))

    print(f"\n>>> SEED SELECTION (Top distinct hardware configs) <<<")
    found_count = 0
    for profit, x, m, nr, ns in results_list:
        signature = (m, nr)
        is_duplicate_basin = False
        for existing_m, existing_nr, _ in seeds_to_refine:
            if existing_m == m and existing_nr == nr:
                is_duplicate_basin = True
                break

        if not is_duplicate_basin:
            print(f"Selected Seed #{found_count + 1}: Model={m}, Nr={nr}, Ns={ns} (Profit={profit:,.0f})")
            seeds_to_refine.add((m, nr, ns))
            found_count += 1

        if found_count >= 5: break

    # =========================================================================
    # 3. HYPER-REFINEMENT (MAXIMUM VARIABLES & COMPUTE)
    # =========================================================================

    global_best_profit = -np.inf
    global_best_x = None
    global_best_title = ""

    # Fine-grained operational variables
    fine_smooth = np.linspace(0.01, 1.0, 50)
    fine_bias = np.linspace(0.85, 1.15, 10)  # Finely tune over/under production
    fine_soc = np.linspace(0.0, 0.5, 5)  # Finely tune start condition

    print(f"\n>>> STAGE 2: HYPER-REFINEMENT ({len(seeds_to_refine)} seeds) <<<")

    for i, (seed_m, seed_nr, seed_ns) in enumerate(seeds_to_refine):
        local_reactor_range = range(max(0, seed_nr - 1), min(N_REACTOR_MAX + 1, seed_nr + 2))
        # Check +/- 20 storage units around seed
        local_storage_range = range(max(0, seed_ns - 20), min(N_STORAGE_MAX + 1, seed_ns + 21))

        s_title = f"Seed {i + 1} (M={seed_m}, R={seed_nr}, S~{seed_ns})"

        p, x, t = run_search_grid(
            [seed_m],
            local_reactor_range,
            local_storage_range,
            fine_smooth,
            fine_bias,
            fine_soc,
            search_name=s_title,
            return_all_results=False
        )

        if p > global_best_profit:
            global_best_profit = p
            global_best_x = x
            global_best_title = t

    # =========================================================================
    # FINAL OUTPUT
    # =========================================================================
    total_time = time.time() - overall_start_time
    print(f"\n\n>>> OPTIMIZATION COMPLETE in {total_time:.1f}s <<<")

    if global_best_x is not None:
        evaluate_candidate(global_best_x, f"GLOBAL OPTIMUM\n{global_best_title}")

    plt.show()