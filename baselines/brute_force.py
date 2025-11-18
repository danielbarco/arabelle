#!/usr/bin/env python

"""
© 2025, Arabelle Solutions and/or its affiliates. All rights reserved.

This Python file is provided for experimentation only in the context of
the AIM Week 2025 challenge.
NO REPRESENTATION OR WARRANTY IS MADE OR IMPLIED AS TO ITS COMPLETENESS,
ACCURACY, OR FITNESS FOR ANY PARTICULAR PURPOSE.
MINLP sample: Nuclear SMR with battery storage modules.

--- MODIFIED FOR BRUTE FORCE SEARCH ---
This script will iterate through a defined space of all integer
combinations (model, n_reactor, n_storage) and use the built-in
heuristics to find a feasible, high-profit configuration.
"""

import matplotlib.pyplot as plt
import numpy as np
import time

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
reactor_years = 60  # Life expectancy of reactor assumed at 60 years
reactor_annuity_factor = interest_rate / (1 - (1 + interest_rate) ** (-reactor_years))

# Storage module
module_capacity_mwh = 50.0  # MWh per module
module_power_mw = 10.0  # MW per module
module_cost = 1.0e7  # EUR per module
storage_leakage_per_hour = 0.0008
storage_years = 20  # Battery modules need to be replaced after 20 years
storage_annuity_factor = interest_rate / (1 - (1 + interest_rate) ** (-storage_years))

# Operational costs
reactor_fixed_om_frac = 0.03  # fraction of capex per year
storage_fixed_om_frac = 0.02
fuel_price = 5.0  # EUR/MWh

# Market pricing
price_base = 70.0  # EUR/MWh base
price_sensitivity = 60.0  # Scales how high price goes with demand
price_surplus = 10.0  # EUR/MWh for surplus sold to market (negative price)


# Storage efficiencies
def charge_efficiency(charge_mw: np.ndarray, max_charge_mw: float) -> np.ndarray:
    if max_charge_mw < 1.0e-6:
        return np.ones(horizon)
    else:
        # Prevent division by zero if max_charge_mw is tiny but non-zero
        safe_max_charge = np.maximum(max_charge_mw, 1e-6)
        r = 0.95 - 0.15 * (charge_mw / safe_max_charge) ** 2
        return np.maximum(r, 0.6)


def discharge_efficiency(dis_mw: np.ndarray, max_dis_mw: float) -> np.ndarray:
    if max_dis_mw < 1.0e-6:
        return np.ones(horizon)
    else:
        # Prevent division by zero if max_dis_mw is tiny but non-zero
        safe_max_dis = np.maximum(max_dis_mw, 1e-6)
        r = 0.96 - 0.2 * (dis_mw / safe_max_dis) ** 2
    return np.maximum(r, 0.55)


# ------------------------- Model evaluation functions -----------------------


def capital_cost_reactor(reactor_capacity: float, n_reactor: int) -> float:
    return n_reactor * (reactor_cap_a * reactor_capacity ** reactor_cap_b)


def capital_cost_storage(n_storage: int) -> float:
    return module_cost * n_storage


def market_price(demand: np.ndarray) -> np.ndarray:
    """Compute the electricity market price according to demand."""
    peak = np.max(demand)
    return price_base + price_sensitivity * demand / peak


def electricity_supplied(
        reactor_production: np.ndarray, n_storage: int, dis: np.ndarray, ch: np.ndarray
) -> np.ndarray:
    """Compute the actual energy supplied to the grid."""
    max_storage_power = n_storage * module_power_mw
    eff_d = discharge_efficiency(dis, max_storage_power)
    eff_c = charge_efficiency(ch, max_storage_power)
    return reactor_production + dis * eff_d - ch / eff_c


def compute_charge_discharge(soc: np.ndarray) -> tuple[np.ndarray]:
    """
    Compute the charge and discharge powers according to state of charge.
    (MODIFIED to use np.roll for correct periodic boundary)
    """
    # Use np.roll for periodic boundary (soc[-1] -> soc[23])
    soc_prev = np.roll(soc, 1)
    charge = soc - soc_prev * (1 - storage_leakage_per_hour)
    return np.maximum(0.0, charge), np.maximum(0.0, -charge)


def var_from_x(x: np.ndarray) -> tuple:
    """Get the problem variables from the optimization vector x."""
    reactor_production = x[:horizon]
    soc = x[horizon: 2 * horizon]
    # Use clip to ensure model index is valid
    reactor_model = int(round(x[-3]))
    reactor_model = np.clip(reactor_model, 0, len(reactor_models) - 1)
    n_reactor = int(round(x[-2]))
    n_storage = int(round(x[-1]))
    return reactor_model, n_reactor, n_storage, reactor_production, soc


def x_from_var(
        reactor_model: int,
        n_reactor: int,
        n_storage: int,
        reactor_production: np.ndarray,
        soc: np.ndarray,
) -> np.ndarray:
    """Get the optimization vector x from the problem variables."""
    x = np.empty(2 * horizon + 3)
    x[:horizon] = reactor_production
    x[horizon: 2 * horizon] = soc
    x[-3] = reactor_model
    x[-2] = n_reactor
    x[-1] = n_storage
    return x


def objective(x: np.ndarray) -> float:
    """
    (CORRECTED OBJECTIVE FUNCTION)
    The objective function is defined as the annualized profit (EUR/year):
    market revenue - capex - fixed O&M - fuel cost.
    """
    reactor_model, n_reactor, n_storage, reactor_production, soc = var_from_x(x)

    if n_reactor == 0 and n_storage == 0:
        return 0.0  # No plant, no profit

    # Calculate costs and CAPEX
    if n_reactor == 0:
        # Storage only
        ann_om_capex = capital_cost_storage(n_storage) * (storage_annuity_factor + storage_fixed_om_frac)
    else:
        # Reactor + Storage
        reactor_capacity = reactor_models[reactor_model]
        cap_reactor = capital_cost_reactor(reactor_capacity, n_reactor)
        cap_storage = capital_cost_storage(n_storage)

        ann_capex = cap_reactor * reactor_annuity_factor + cap_storage * storage_annuity_factor
        annual_fixed_om = cap_reactor * reactor_fixed_om_frac + cap_storage * storage_fixed_om_frac
        ann_om_capex = ann_capex + annual_fixed_om

    # Fuel cost (daily cost)
    daily_fuel_cost = np.sum(reactor_production) * fuel_price

    # Market interactions (daily)
    ch, dis = compute_charge_discharge(soc)
    supplied = electricity_supplied(reactor_production, n_storage, dis, ch)

    local_supply = np.minimum(supplied, electric_demand)
    unmet = np.maximum(0.0, electric_demand - supplied)
    surplus = np.maximum(0.0, supplied - electric_demand)

    price = market_price(electric_demand)

    net_market = price * local_supply
    net_market -= price * unmet
    net_market -= price_surplus * surplus

    daily_market = np.sum(net_market)

    daily_profit = daily_market - daily_fuel_cost
    annual_profit = daily_profit * 365.0 - ann_om_capex

    return annual_profit


def constraints_residuals(x: np.ndarray) -> list[float]:
    """
    Returns list of inequality constraints residuals which should be >= 0.
    """
    reactor_model, n_reactor, n_storage, reactor_production, soc = var_from_x(x)
    res = []

    # Reactor bounds
    plant_capacity = reactor_models[reactor_model] * n_reactor
    for t in range(horizon):
        # reactor_production <= reactor_capacity
        res.append(plant_capacity - reactor_production[t])
        # reactor_production >= 0
        res.append(reactor_production[t])

    # Storage bounds
    max_storage_energy = n_storage * module_capacity_mwh
    max_storage_power = n_storage * module_power_mw

    # Check for division by zero if n_storage is 0
    if n_storage == 0:
        max_storage_energy = 0.0
        max_storage_power = 0.0

    ch, dis = compute_charge_discharge(soc)
    for t in range(horizon):
        # charge and discharge <= max_storage_power
        res.append(max_storage_power - ch[t])
        res.append(max_storage_power - dis[t])
        # soc <= max_storage_energy
        res.append(max_storage_energy - soc[t])
        # soc >= 0
        res.append(soc[t])

    # Reactors count >=0
    res.append(n_reactor)
    # Storage modules count >=0
    res.append(n_storage)

    return res


# ------------------------- Example candidates -------------------------------------


def build_candidate_with_storage(
        reactor_model: int, n_reactor: int, n_storage: int, variability_smoothness: float
) -> np.ndarray:
    """
    Heuristic to build a candidate where:
      - reactor follows a smoothed version of demand
      - storage charges when reactor > demand and discharges when reactor < demand
    (Note: This is an iterative, non-periodic heuristic. It may not be
     perfectly feasible with the periodic `compute_charge_discharge` function,
     which is why we must check residuals.)
    """
    demand = electric_demand.copy()
    avg = np.mean(demand)

    # Smooth reactor profile: weighted average between demand and a flat profile at avg
    reactor_production = (
            variability_smoothness * np.full(horizon, avg)
            + (1 - variability_smoothness) * demand
    )
    # Clip to capacity
    plant_capacity = reactor_models[reactor_model] * n_reactor
    reactor_production = np.clip(reactor_production, 0, plant_capacity)

    # Now compute storage actions to correct net supply toward demand
    soc = np.zeros(horizon)
    max_power = n_storage * module_power_mw
    max_energy = n_storage * module_capacity_mwh

    # Initialize SOC to 30% of capacity
    if max_energy > 0:
        soc_prev = 0.30 * max_energy
    else:
        soc_prev = 0.0

    for t in range(horizon):
        # Compute mismatch
        mismatch = reactor_production[t] - demand[t]
        ch_t, dis_t = 0.0, 0.0

        if mismatch > 0:
            # surplus energy available -> try to charge storage
            available_power = min(mismatch, max_power)
            remaining_energy_cap = max(0, max_energy - soc_prev)
            ch_t = min(available_power, remaining_energy_cap)
        else:
            # deficit -> discharge storage
            need = -mismatch
            discharge_power = min(need, max_power)
            dis_t = min(discharge_power, soc_prev)

        soc[t] = soc_prev * (1 - storage_leakage_per_hour) + ch_t - dis_t
        soc[t] = np.clip(soc[t], 0, max_energy)  # Enforce bounds
        soc_prev = soc[t]

    x = x_from_var(
        reactor_model=reactor_model,
        n_reactor=n_reactor,
        n_storage=n_storage,
        reactor_production=reactor_production,
        soc=soc,
    )
    return x


def build_candidate_wo_storage(reactor_model: int, n_reactor: int) -> np.ndarray:
    """
    Builds a candidate where:
      - reactor follows exactly demand
      - storage is not used
    """
    plant_capacity = reactor_models[reactor_model] * n_reactor
    reactor_production = np.minimum(plant_capacity, electric_demand)
    soc = np.zeros(horizon)
    x = x_from_var(
        reactor_model=reactor_model,
        n_reactor=n_reactor,
        n_storage=0,
        reactor_production=reactor_production,
        soc=soc,
    )
    return x


# ------------------------- Run example and plot -----------------------------
def evaluate_candidate(cand, title):
    print(f"\n--- Evaluating candidate '{title}' ---")

    # Use corrected objective function
    obj = objective(cand)
    res = constraints_residuals(cand)
    min_res = np.min(res)

    print(f"Annual Profit (EUR): {obj:,.2f}")
    print(f"Min Inequality Residual (>=0 for feasibility): {min_res:.6f}")
    if min_res < -1e-6:
        print("WARNING: Candidate is INFEASIBLE.")

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

    # Add second axis for SOC
    ax2 = ax.twinx()
    ax2.plot(hours, soc, 'c.', label='SOC')
    ax2.set_ylabel('State of Charge (MWh)', color='c')
    ax2.tick_params(axis='y', labelcolor='c')
    if n_storage > 0:
        max_energy = n_storage * module_capacity_mwh
        ax2.set_ylim(bottom=0, top=max_energy * 1.1)
    else:
        ax2.set_ylim(bottom=0, top=1)

    # Combine legends
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(
        lines + lines2,
        labels + labels2,
        loc="upper left",
        fontsize="small",
        fancybox=False,
        framealpha=1.0,
        borderaxespad=0,
        edgecolor="black",
    )
    ax.grid(True, ls=":", color="black")
    ax.set_xlim((hours[0], hours[-1]))
    plt.tight_layout()


if __name__ == "__main__":
    plt.rcParams["font.family"] = "serif"

    # --- BRUTE FORCE IMPLEMENTATION ---
    print("--- Starting Brute Force Optimization (Expanded Search) ---")
    start_time = time.time()

    # --- EXPANDED SEARCH SPACE ---
    N_REACTOR_MAX = 20  # Max reactors to check (0 to 20)
    N_STORAGE_MAX = 400  # Max storage modules to check (0 to 400)
    # Heuristic smoothness parameters to test (100 steps)
    SMOOTHNESS_PARAMS = np.linspace(0.01, 1.0, 100)
    # --- END EXPANDED SEARCH ---

    best_profit = -np.inf
    best_candidate_x = None
    best_title = "No feasible solution found"

    n_models = len(reactor_models)
    n_reactors_range = N_REACTOR_MAX + 1
    n_storage_range = N_STORAGE_MAX + 1
    n_combinations = n_models * n_reactors_range * n_storage_range

    print(f"Testing {n_models} models, {n_reactors_range} reactor counts, {n_storage_range} storage counts.")
    print(f"Testing {len(SMOOTHNESS_PARAMS)} heuristic parameters per storage config.")
    print(f"Total integer configurations: {n_combinations:,}")

    count = 0
    for model_idx in range(n_models):
        for n_r in range(n_reactors_range):
            for n_s in range(n_storage_range):

                count += 1
                # Update progress every 500 configs
                if count % 500 == 0:
                    elapsed = time.time() - start_time
                    print(f"Progress: {count:,} / {n_combinations:,} ({elapsed:.1f}s)")

                # Don't test (0, 0) config
                if n_r == 0 and n_s == 0:
                    continue

                # We must test multiple heuristics for this (n_r, n_s) combo
                candidates_to_test = []

                if n_s == 0:
                    # Only one candidate: no storage
                    cand_x = build_candidate_wo_storage(model_idx, n_r)
                    candidates_to_test.append((cand_x, "No Storage"))
                else:
                    # Test all smoothness heuristics
                    for s in SMOOTHNESS_PARAMS:
                        cand_x = build_candidate_with_storage(model_idx, n_r, n_s, s)
                        candidates_to_test.append((cand_x, f"Smooth={s * 100:.0f}%"))

                # Now, evaluate all heuristics for this (model, n_r, n_s)
                for cand_x, heuristic_name in candidates_to_test:
                    # 1. Check feasibility
                    residuals = constraints_residuals(cand_x)
                    min_residual = np.min(residuals)

                    # Use a small tolerance for floating point errors
                    if min_residual < -1e-6:
                        # This candidate is infeasible, skip it
                        continue

                    # 2. If feasible, get profit
                    profit = objective(cand_x)

                    # 3. Check if it's the new best
                    if profit > best_profit:
                        best_profit = profit
                        best_candidate_x = cand_x
                        title = (f"Model={model_idx}, N_R={n_r}, N_S={n_s} "
                                 f"({heuristic_name})")
                        best_title = title
                        print(f"--- New Best Found! ---")
                        print(f"Profit: {profit:,.2f} EUR/year")
                        print(f"Config: {title}")

    # --- End of Brute Force ---
    end_time = time.time()
    print(f"\n--- Brute Force Complete ({end_time - start_time:.2f}s) ---")

    if best_candidate_x is not None:
        evaluate_candidate(best_candidate_x, f"Brute Force Optimum\n{best_title}")
    else:
        print("No feasible candidate was found by any heuristic.")

    # Also show the original heuristics for comparison
    print("\n--- Evaluating Original Heuristics for Comparison ---")
    cand1 = build_candidate_with_storage(
        reactor_model=2, n_reactor=1, n_storage=12, variability_smoothness=0.6
    )
    evaluate_candidate(cand1, "Heuristic 1: (Model 2, N=1, S=12, Smooth=60%)")

    cand2 = build_candidate_wo_storage(reactor_model=3, n_reactor=1)
    evaluate_candidate(cand2, "Heuristic 2: (Model 3, N=1, S=0)")

    plt.show()