"""
SMR and Battery Storage Optimization Script

This script implements a Mixed-Integer Non-Linear Programming (MINLP) problem
to optimize the configuration (reactor model, counts) and operational schedule
(production, charge/discharge) of a Small Modular Reactor (SMR) system with
battery storage modules. The problem is solved using the
scipy.optimize.differential_evolution solver.
"""

import matplotlib.pyplot as plt
import numpy as np
import argparse
from scipy.optimize import differential_evolution, NonlinearConstraint

# ------------------------- Problem Data -------------------------------------
horizon = 24  # Hourly horizon

# Hourly electric demand (MW) - daily profile
electric_demand = np.array(
    [
        160, 152, 144, 140, 144, 160, 200, 240, 280, 300, 320, 340, 360, 368, 352,
        340, 320, 312, 300, 280, 260, 240, 220, 192,
    ],
    dtype=np.float64,
)

# SMR models power offerings (MW per unit)
reactor_models = [80.0, 160.0, 300.0, 350.0, 470.0]

# Economic parameters
interest_rate = 0.04

# Reactor parameters
reactor_cap_a = 2.0e7
reactor_cap_b = 0.8
reactor_years = 60
reactor_annuity_factor = interest_rate / (1 - (1 + interest_rate) ** (-reactor_years))

# Storage module parameters
module_capacity_mwh = 50.0  # MWh per module
module_power_mw = 10.0  # MW per module
module_cost = 1.0e7  # EUR per module
storage_leakage_per_hour = 0.0008
storage_years = 20
storage_annuity_factor = interest_rate / (1 - (1 + interest_rate) ** (-storage_years))

# Operational costs (annual)
reactor_fixed_om_frac = 0.03  # fraction of capex per year
storage_fixed_om_frac = 0.02
fuel_price = 5.0  # EUR/MWh

# Market pricing
price_base = 70.0  # EUR/MWh base
price_sensitivity = 60.0
price_surplus = 10.0  # EUR/MWh for surplus sold to market


# --- State of Charge (SOC) Pre-calculation ---

def get_soc_solver_matrix(h: int, leakage: float) -> np.ndarray:
    """
    Builds the inverse matrix A_inv for solving the periodic SOC.
    The system is A * soc = net_charge.
    """
    k = 1.0 - leakage
    A = np.identity(h)
    # Set the sub-diagonal
    sub_diag = -k * np.ones(h - 1)
    np.fill_diagonal(A[1:], sub_diag)
    # Set the corner element for periodicity
    A[0, -1] = -k
    A_inv = np.linalg.inv(A)
    return A_inv


SOC_SOLVER_MATRIX_INV = get_soc_solver_matrix(horizon, storage_leakage_per_hour)


# Storage efficiencies
def charge_efficiency(charge_mw: np.ndarray, max_charge_mw: float) -> np.ndarray:
    """Calculates charging efficiency based on instantaneous charging power."""
    if max_charge_mw < 1.0e-6:
        return np.ones(horizon)
    safe_max_charge = np.maximum(max_charge_mw, 1e-6)
    r = 0.95 - 0.15 * (charge_mw / safe_max_charge) ** 2
    return np.maximum(r, 0.6)


def discharge_efficiency(dis_mw: np.ndarray, max_dis_mw: float) -> np.ndarray:
    """Calculates discharging efficiency based on instantaneous discharging power."""
    if max_dis_mw < 1.0e-6:
        return np.ones(horizon)
    safe_max_dis = np.maximum(max_dis_mw, 1e-6)
    r = 0.96 - 0.2 * (dis_mw / safe_max_dis) ** 2
    return np.maximum(r, 0.55)


# ------------------------- Model Evaluation Functions -----------------------


def capital_cost_reactor(reactor_capacity: float, n_reactor: int) -> float:
    """Calculates total capital cost for reactors."""
    return n_reactor * (reactor_cap_a * reactor_capacity ** reactor_cap_b)


def capital_cost_storage(n_storage: int) -> float:
    """Calculates total capital cost for storage modules."""
    return module_cost * n_storage


def market_price(demand: np.ndarray) -> np.ndarray:
    """Computes the electricity market price according to demand."""
    peak = np.max(demand)
    return price_base + price_sensitivity * demand / peak


def electricity_supplied(
        reactor_production: np.ndarray, n_storage: int, dis: np.ndarray, ch: np.ndarray
) -> np.ndarray:
    """Computes the actual energy supplied to the grid after storage conversion."""
    max_storage_power = n_storage * module_power_mw
    eff_d = discharge_efficiency(dis, max_storage_power)
    eff_c = charge_efficiency(ch, max_storage_power)
    return reactor_production + dis * eff_d - ch / eff_c


def compute_soc_from_actions(net_charge: np.ndarray) -> np.ndarray:
    """
    Calculates the periodic State of Charge (SOC) array
    by solving the linear system using the pre-computed inverse matrix.
    """
    soc = SOC_SOLVER_MATRIX_INV @ net_charge
    return soc


def get_ch_dis_from_net(net_charge: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Splits net_charge into positive (charge) and negative (discharge) components."""
    ch = np.maximum(0.0, net_charge)
    dis = np.maximum(0.0, -net_charge)
    return ch, dis


def var_from_x(x: np.ndarray) -> tuple:
    """Extracts problem variables from the optimization vector x."""
    reactor_production = x[:horizon]
    net_charge = x[horizon: 2 * horizon]

    # Use floor and clip to ensure valid integer indices/counts
    reactor_model = int(np.floor(x[-3]))
    n_reactor = int(np.floor(x[-2]))
    n_storage = int(np.floor(x[-1]))

    reactor_model = np.clip(reactor_model, 0, len(reactor_models) - 1)

    return reactor_model, n_reactor, n_storage, reactor_production, net_charge


def x_from_var(
        reactor_model: int,
        n_reactor: int,
        n_storage: int,
        reactor_production: np.ndarray,
        net_charge: np.ndarray,
) -> np.ndarray:
    """Creates the optimization vector x from the problem variables."""
    x = np.empty(2 * horizon + 3)
    x[:horizon] = reactor_production
    x[horizon: 2 * horizon] = net_charge
    x[-3] = reactor_model
    x[-2] = n_reactor
    x[-1] = n_storage
    return x


def objective(x: np.ndarray) -> float:
    """
    The objective function is the negative annualized profit (EUR/year).
    The solver minimizes this value to maximize profit.
    """
    reactor_model, n_reactor, n_storage, reactor_production, net_charge = var_from_x(x)

    # Calculate costs and CAPEX
    if n_reactor == 0:
        ann_om_capex = capital_cost_storage(n_storage) * (storage_annuity_factor + storage_fixed_om_frac)
    else:
        reactor_capacity = reactor_models[reactor_model]
        cap_reactor = capital_cost_reactor(reactor_capacity, n_reactor)
        cap_storage = capital_cost_storage(n_storage)

        ann_capex = cap_reactor * reactor_annuity_factor + cap_storage * storage_annuity_factor
        annual_fixed_om = cap_reactor * reactor_fixed_om_frac + cap_storage * storage_fixed_om_frac
        ann_om_capex = ann_capex + annual_fixed_om

    # Fuel cost (daily cost)
    daily_fuel_cost = np.sum(reactor_production) * fuel_price

    # Market interactions (daily)
    ch, dis = get_ch_dis_from_net(net_charge)
    supplied = electricity_supplied(reactor_production, n_storage, dis, ch)

    local_supply = np.minimum(supplied, electric_demand)
    unmet = np.maximum(0.0, electric_demand - supplied)
    surplus = np.maximum(0.0, supplied - electric_demand)

    price = market_price(electric_demand)

    net_market = price * local_supply
    net_market -= price * unmet  # Penalty for unmet demand
    net_market -= price_surplus * surplus  # Revenue from surplus

    daily_market = np.sum(net_market)

    daily_profit = daily_market - daily_fuel_cost
    annual_profit = daily_profit * 365.0 - ann_om_capex

    return -annual_profit


def constraints_residuals(x: np.ndarray) -> list[float]:
    """
    Returns a list of inequality constraints residuals which must be >= 0 for feasibility.
    Total length is 146: 48 (reactor) + 96 (storage) + 2 (counts).
    """
    reactor_model, n_reactor, n_storage, reactor_production, net_charge = var_from_x(x)
    res = []

    # 1. Reactor bounds (48 residuals: 24 upper, 24 lower)
    plant_capacity = reactor_models[reactor_model] * n_reactor
    for t in range(horizon):
        res.append(plant_capacity - reactor_production[t])  # Prod <= Capacity
        res.append(reactor_production[t])  # Prod >= 0

    # 2. Storage bounds (96 residuals: 24 charge, 24 discharge, 48 SOC)
    max_storage_energy = n_storage * module_capacity_mwh
    max_storage_power = n_storage * module_power_mw

    soc = compute_soc_from_actions(net_charge)
    ch, dis = get_ch_dis_from_net(net_charge)

    for t in range(horizon):
        # Power limits
        res.append(max_storage_power - ch[t])  # Charge <= Max Power
        res.append(max_storage_power - dis[t])  # Discharge <= Max Power

        # SOC limits
        res.append(max_storage_energy - soc[t])  # SOC <= Max Energy
        res.append(soc[t])  # SOC >= 0

    # 3. Integer counts (2 residuals)
    res.append(n_reactor)
    res.append(n_storage)

    if len(res) != 146:
        raise RuntimeError(f"Internal Error: Constraint residual length mismatch: Expected 146, got {len(res)}")

    return res


# ------------------------- Solver Function -----------------------------

def solve_optimization(maxiter: int, popsize: int, seed: int):
    """
    Solves the MINLP problem using Differential Evolution.
    """
    print("--- Starting Optimization (Differential Evolution) ---")
    print(f"Max Iterations: {maxiter} | Population Size: {popsize} | Seed: {seed}")

    rng = np.random.default_rng(seed)

    # 1. Define Bounds
    max_prod_bound = 1500.0
    bounds = [(0, max_prod_bound)] * horizon  # 24 for reactor_production

    max_storage_modules_bound = 50
    max_power_possible = max_storage_modules_bound * module_power_mw

    bounds += [(-max_power_possible, max_power_possible)] * horizon  # 24 for net_charge

    # Bounds for integer variables
    reactor_model_bounds = (0, len(reactor_models) - 1)
    reactor_count_bounds = (0, 5)
    storage_count_bounds = (0, max_storage_modules_bound)

    bounds.append(reactor_model_bounds)
    bounds.append(reactor_count_bounds)
    bounds.append(storage_count_bounds)

    # 2. Define Integrality constraints
    integrality = [False] * (2 * horizon) + [True] * 3

    # 3. Define Non-linear constraints (all residuals >= 0)
    nlc = NonlinearConstraint(constraints_residuals, 0, np.inf)

    # 4. Define a progress callback
    def callback_fn(xk, convergence):
        profit = -objective(xk)
        print(f"Current Best Profit: {profit:,.2f} EUR/year | Convergence: {convergence:.4f}")

    # 5. Build the feasible 'warm start' initial population
    print("Building feasible 'warm start' initial population...")
    n_pop_actual = popsize * len(bounds)
    initial_population = []
    for _ in range(n_pop_actual):
        # Generate random integer configuration
        model = rng.integers(reactor_model_bounds[0], reactor_model_bounds[1] + 1)
        n_r = rng.integers(reactor_count_bounds[0], reactor_count_bounds[1] + 1)
        n_s = rng.integers(storage_count_bounds[0], storage_count_bounds[1] + 1)

        smoothness = rng.random() * 0.4 + 0.3

        # Build a feasible candidate vector 'x' using heuristics
        if n_s == 0:
            x = build_candidate_wo_storage(model, n_r)
        else:
            x = build_candidate_with_storage(model, n_r, n_s, smoothness)

        initial_population.append(x)

    initial_population = np.array(initial_population)
    print(f"Initial population of size {n_pop_actual} built.")

    # 6. Run the solver
    result = differential_evolution(
        func=objective,
        bounds=bounds,
        constraints=nlc,
        integrality=integrality,
        init=initial_population,
        maxiter=maxiter,
        popsize=popsize,
        tol=0.01,
        seed=seed,
        polish=True,
        updating='deferred',
        workers=-1,
        disp=False,
        callback=callback_fn
    )

    return result


# ------------------------- Helper/Heuristic Functions ---------------------

def _heuristic_compute_ch_dis(soc: np.ndarray) -> tuple[np.ndarray]:
    """Helper to get charge/discharge from the SOC profile for the heuristic."""
    soc_prev = np.roll(soc, 1)
    charge = soc - soc_prev * (1 - storage_leakage_per_hour)
    return np.maximum(0.0, charge), np.maximum(0.0, -charge)


def build_candidate_with_storage(
        reactor_model: int, n_reactor: int, n_storage: int, variability_smoothness: float
) -> np.ndarray:
    """
    Heuristic to build a candidate vector where the reactor follows a smoothed
    demand profile and storage compensates the difference.
    """
    demand = electric_demand.copy()
    avg = np.mean(demand)

    # Smooth reactor profile
    reactor_production = (
            variability_smoothness * np.full(horizon, avg)
            + (1 - variability_smoothness) * demand
    )
    plant_capacity = reactor_models[reactor_model] * n_reactor
    reactor_production = np.clip(reactor_production, 0, plant_capacity)

    # Compute storage actions using a simple dispatch loop
    soc = np.zeros(horizon)
    max_power = n_storage * module_power_mw
    max_energy = n_storage * module_capacity_mwh

    if max_energy > 0:
        soc_prev = 0.30 * max_energy
    else:
        soc_prev = 0.0

    for t in range(horizon):
        mismatch = reactor_production[t] - demand[t]
        ch_t, dis_t = 0.0, 0.0

        if mismatch > 0:
            available_power = min(mismatch, max_power)
            remaining_energy_cap = max_energy - soc_prev
            ch_t = min(available_power, remaining_energy_cap)
        else:
            need = -mismatch
            discharge_power = min(need, max_power)
            dis_t = min(discharge_power, soc_prev)

        soc[t] = soc_prev * (1 - storage_leakage_per_hour) + ch_t - dis_t
        soc[t] = np.clip(soc[t], 0, max_energy)
        soc_prev = soc[t]

    # Convert the resulting SOC profile into final net_charge actions
    ch_final, dis_final = _heuristic_compute_ch_dis(soc)
    net_charge_final = ch_final - dis_final

    x = x_from_var(
        reactor_model=reactor_model,
        n_reactor=n_reactor,
        n_storage=n_storage,
        reactor_production=reactor_production,
        net_charge=net_charge_final,
    )
    return x


def build_candidate_wo_storage(reactor_model: int, n_reactor: int) -> np.ndarray:
    """
    Builds a candidate vector with no storage used, where the reactor
    production matches demand up to capacity.
    """
    plant_capacity = reactor_models[reactor_model] * n_reactor
    reactor_production = np.minimum(plant_capacity, electric_demand)

    net_charge = np.zeros(horizon)

    x = x_from_var(
        reactor_model=reactor_model,
        n_reactor=n_reactor,
        n_storage=0,
        reactor_production=reactor_production,
        net_charge=net_charge,
    )
    return x


# ------------------------- Evaluation and Plotting --------------------------

def evaluate_candidate(cand, title):
    """Evaluates a candidate solution and plots the operational profile."""
    print(f"\n--- Evaluating candidate '{title}' ---")

    obj_val = objective(cand)
    profit = -obj_val

    res = constraints_residuals(cand)
    min_residual = np.min(res) if len(res) > 0 else 0.0

    hours = np.arange(horizon)

    reactor_model, n_reactor, n_storage, reactor_production, net_charge = var_from_x(cand)
    soc = compute_soc_from_actions(net_charge)
    ch, dis = get_ch_dis_from_net(net_charge)

    net_supply = electricity_supplied(reactor_production, n_storage, dis, ch)

    print(f"Annual Profit (EUR): {profit:,.2f}")
    print(f"Min Inequality Residual (>=0 for feasibility): {min_residual:.4f}")
    if min_residual < -1e-6:
        print("WARNING: Solution is INFEASIBLE.")

    print(
        f"Configuration: Model={reactor_model} ({reactor_models[reactor_model]} MW), N_Reactor={n_reactor}, N_Storage={n_storage}")

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(hours, electric_demand, "k--", label="Electric demand")
    ax.plot(hours, reactor_production, "b-", label="Reactor output")
    ax.plot(hours, dis, "r-", label="Storage discharge")
    ax.plot(hours, ch, "g-", label="Storage charge")
    ax.plot(hours, net_supply, "m-", label="Net supply to grid")

    plant_capacity = reactor_models[reactor_model] * n_reactor
    ax.axhline(y=plant_capacity, color='b', linestyle=':', label=f'Reactor Capacity ({plant_capacity:.0f} MW)')

    ax.set_xlabel("Hour")
    ax.set_ylabel("Power (MW)")
    fig.suptitle(title, fontsize=16)

    # Add second axis for SOC
    ax2 = ax.twinx()
    ax2.plot(hours, soc, 'c.', label='SOC')
    ax2.set_ylabel('State of Charge (MWh)', color='c')
    ax2.tick_params(axis='y', labelcolor='c')
    if n_storage > 0:
        max_energy = n_storage * module_capacity_mwh
        ax2.set_ylim(bottom=-max_energy * 0.05, top=max_energy * 1.1)
    else:
        ax2.set_ylim(bottom=-0.05, top=1)

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


# ------------------------- Main Execution -----------------------------------

if __name__ == "__main__":
    plt.rcParams["font.family"] = "serif"

    parser = argparse.ArgumentParser(
        description="Solve the SMR+Storage optimization problem using Differential Evolution."
    )
    parser.add_argument(
        "--maxiter",
        type=int,
        default=500,
        help="Maximum number of iterations for the optimizer (default: 500).",
    )
    parser.add_argument(
        "--popsize",
        type=int,
        default=30,
        help="Population size multiplier (default: 30).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)."
    )
    parser.add_argument(
        "--plot_only_optimized",
        action="store_true",
        help="If set, only the optimized solution will be plotted, skipping heuristics."
    )
    args = parser.parse_args()

    result = solve_optimization(
        maxiter=args.maxiter,
        popsize=args.popsize,
        seed=args.seed
    )

    # Evaluate and plot the best solution found
    best_profit = -result.fun
    if result.success:
        title = f"Optimized Solution (Profit: {best_profit:,.2f} EUR/year)"
    else:
        title = f"Best Solution Found (Optimization Failed, Profit: {best_profit:,.2f} EUR/year)"

    evaluate_candidate(result.x, title)

    # Evaluate heuristic candidates (for comparison)
    if not args.plot_only_optimized:
        print("\n--- Evaluating Heuristic Candidates for Comparison ---")
        cand1 = build_candidate_with_storage(
            reactor_model=2, n_reactor=1, n_storage=12, variability_smoothness=0.6
        )
        evaluate_candidate(cand1, "Heuristic 1: Smoothed Reactor (Model 2, N=1, Storage=12)")

        cand2 = build_candidate_wo_storage(reactor_model=3, n_reactor=1)
        evaluate_candidate(cand2, "Heuristic 2: No Storage (Model 3, N=1)")

    plt.show()