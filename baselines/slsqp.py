"""
SMR and Battery Storage Optimization - Solved via Decomposition

Methodology:
1. Decomposition: The problem is split into an Outer Loop (Integers) and an Inner Loop (Continuous).
2. Outer Loop: Brute-force Grid Search over all valid combinations of Reactor Model,
   Reactor Count, and Storage Count. (Approx. 1,500 combinations).
3. Inner Loop: For a fixed configuration, use SLSQP (Sequential Least Squares Programming).
   SLSQP is a gradient-based solver that natively handles the equality constraints
   of the SOC dynamics (SOC[t+1] = SOC[t] + Flow) without "breaking" the timeline.

Author: Gemini
Date: 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import concurrent.futures
import time

# ------------------------- Problem Data -------------------------------------
horizon = 24

electric_demand = np.array(
    [160, 152, 144, 140, 144, 160, 200, 240, 280, 300, 320, 340, 360, 368, 352,
     340, 320, 312, 300, 280, 260, 240, 220, 192],
    dtype=np.float64,
)

reactor_models = [80.0, 160.0, 300.0, 350.0, 470.0]
interest_rate = 0.04

reactor_cap_a = 2.0e7
reactor_cap_b = 0.8
reactor_years = 60
reactor_annuity_factor = interest_rate / (1 - (1 + interest_rate) ** (-reactor_years))

module_capacity_mwh = 50.0
module_power_mw = 10.0
module_cost = 1.0e7
storage_leakage_per_hour = 0.0008
storage_years = 20
storage_annuity_factor = interest_rate / (1 - (1 + interest_rate) ** (-storage_years))

reactor_fixed_om_frac = 0.03
storage_fixed_om_frac = 0.02
fuel_price = 5.0

price_base = 70.0
price_sensitivity = 60.0
price_surplus = 10.0


# ------------------------- Helper Functions ---------------------------------

def get_market_prices(demand):
    peak = np.max(demand)
    return price_base + price_sensitivity * demand / peak


MARKET_PRICES = get_market_prices(electric_demand)


def charge_efficiency(charge_mw, max_charge_mw):
    if max_charge_mw < 1e-6: return 1.0
    # Vectorized calculation
    r = 0.95 - 0.15 * (charge_mw / max_charge_mw) ** 2
    return np.maximum(r, 0.6)


def discharge_efficiency(dis_mw, max_dis_mw):
    if max_dis_mw < 1e-6: return 1.0
    r = 0.96 - 0.2 * (dis_mw / max_dis_mw) ** 2
    return np.maximum(r, 0.55)


def calculate_capex_annual(reactor_idx, n_reactor, n_storage):
    """Calculates the fixed annual cost (CAPEX + Fixed O&M) for a configuration."""
    if n_reactor > 0:
        cap_mw = reactor_models[reactor_idx]
        # Capital Costs
        c_reac = n_reactor * (reactor_cap_a * cap_mw ** reactor_cap_b)
        c_stor = n_storage * module_cost

        # Annuity
        ann_capex = c_reac * reactor_annuity_factor + c_stor * storage_annuity_factor

        # Fixed O&M
        ann_om = c_reac * reactor_fixed_om_frac + c_stor * storage_fixed_om_frac
        return ann_capex + ann_om
    elif n_storage > 0:
        # Storage only case
        c_stor = n_storage * module_cost
        ann_capex = c_stor * storage_annuity_factor
        ann_om = c_stor * storage_fixed_om_frac
        return ann_capex + ann_om
    return 0.0


# ------------------------- Inner Loop: SLSQP Solver -------------------------

def solve_operational_schedule(reactor_capacity, n_storage):
    """
    Solves the continuous operational problem for a FIXED hardware configuration.
    Uses SLSQP to handle the non-linear efficiency and SOC dynamics.

    Returns:
        daily_profit (float), x_sol (array), success (bool)
    """

    max_store_p = n_storage * module_power_mw
    max_store_e = n_storage * module_capacity_mwh

    # Optimization Vector X structure (size 3 * 24 = 72):
    # 0-23: Reactor Production
    # 24-47: Storage Charge (positive)
    # 48-71: Storage Discharge (positive)
    # Note: SOC is calculated dynamically to reduce variable count and ensure consistency

    # Bounds
    bounds = []
    # Reactor bounds
    for _ in range(horizon):
        bounds.append((0.0, reactor_capacity))
    # Charge bounds
    for _ in range(horizon):
        bounds.append((0.0, max_store_p))
    # Discharge bounds
    for _ in range(horizon):
        bounds.append((0.0, max_store_p))

    # Initial Guess: Reactor follows demand, storage idle
    x0 = np.zeros(3 * horizon)
    x0[:horizon] = np.minimum(reactor_capacity, electric_demand)

    # --- Constraints ---

    def constraint_soc_dynamics(x):
        """
        Ensures SOC stays within bounds [0, Max] and is periodic (SOC_end == SOC_start).
        This function returns an array of residuals.
        SLSQP expects inequalities to be >= 0.

        However, handling SOC bounds inside a function call is tricky for gradients.
        Strategy: We calculate the full SOC profile based on X.
        We return a concatenated array of checks.
        """
        r_prod = x[:horizon]
        ch = x[horizon:2 * horizon]
        dis = x[2 * horizon:]

        # Efficiency
        eff_c = charge_efficiency(ch, max_store_p)
        eff_d = discharge_efficiency(dis, max_store_p)

        # Calculate SOC profile
        # We assume SOC_start (t=0) is a variable or we enforce periodicity.
        # Let's enforce periodicity: SOC[0] is result of previous day.
        # To solve this strictly, we iterate.

        soc = np.zeros(horizon + 1)
        # Arbitrary start to compute relative change
        soc[0] = 0.0

        for t in range(horizon):
            net_flow = ch[t] * eff_c[t] - dis[t] / eff_d[t]
            soc[t + 1] = soc[t] * (1 - storage_leakage_per_hour) + net_flow

        # The "drift" over 24 hours
        total_drift = soc[-1]

        # If drift is positive, we are gaining energy (impossible if strictly periodic without source?)
        # Actually, we just need SOC_final == SOC_initial.
        # But SOC_initial is unknown.
        # In a linear periodic system: SOC[t] = A^t * SOC[0] + convolution(inputs).
        # For simplification in SLSQP:
        # We add an explicit slack variable for Initial SOC?
        # Or easier: We just penalize violation of SOC limits relative to a "floating" baseline.

        return 0.0  # Placeholder, handled below differently

    # RE-STRATEGY for SOC:
    # To make it robust for SLSQP, we include SOC as explicit variables.
    # New X size: 4 * 24 = 96 (Prod, Ch, Dis, SOC)
    # Constraints:
    # 1. Equality: SOC[t+1] dynamics matches SOC[t] + flows
    # 2. Equality: SOC[0] == SOC[24] (Periodicity)

    bounds = []
    for _ in range(horizon): bounds.append((0.0, reactor_capacity))  # Prod
    for _ in range(horizon): bounds.append((0.0, max_store_p))  # Ch
    for _ in range(horizon): bounds.append((0.0, max_store_p))  # Dis
    for _ in range(horizon): bounds.append((0.0, max_store_e))  # SOC

    x0 = np.zeros(4 * horizon)
    x0[:horizon] = np.minimum(reactor_capacity, electric_demand)
    if max_store_e > 0:
        x0[3 * horizon:] = 0.5 * max_store_e  # Start SOC at 50%

    def obj_func(x):
        r_prod = x[:horizon]
        ch = x[horizon:2 * horizon]
        dis = x[2 * horizon:3 * horizon]

        # Fuel Cost
        cost_fuel = np.sum(r_prod) * fuel_price

        # Market Revenue
        eff_d = discharge_efficiency(dis, max_store_p)
        eff_c = charge_efficiency(ch, max_store_p)

        supplied = r_prod + dis * eff_d - ch / eff_c

        local_supply = np.minimum(supplied, electric_demand)
        unmet = np.maximum(0.0, electric_demand - supplied)
        surplus = np.maximum(0.0, supplied - electric_demand)

        revenue = np.sum(MARKET_PRICES * local_supply)
        penalty = np.sum(MARKET_PRICES * unmet)
        surplus_rev = np.sum(price_surplus * surplus)

        daily_profit = revenue + surplus_rev - penalty - cost_fuel
        return -daily_profit  # Minimize negative profit

    # Constraints definition
    cons = []

    # Dynamics Constraint (Equality)
    def constraint_dynamics(x):
        soc = x[3 * horizon:]
        ch = x[horizon:2 * horizon]
        dis = x[2 * horizon:3 * horizon]

        residuals = []

        # Efficiency calculation
        eff_c = charge_efficiency(ch, max_store_p)
        eff_d = discharge_efficiency(dis, max_store_p)

        # SOC[t] depends on SOC[t-1].
        # For t=0, previous is t=23 (Periodicity)
        prev_soc = soc[-1]  # Wrap around

        # Calculate expected SOC based on physics
        net_flow = ch[0] * eff_c[0] - dis[0] / eff_d[0]
        expected_current = prev_soc * (1 - storage_leakage_per_hour) + net_flow
        residuals.append(soc[0] - expected_current)

        for t in range(1, horizon):
            prev_soc = soc[t - 1]
            net_flow = ch[t] * eff_c[t] - dis[t] / eff_d[t]
            expected_current = prev_soc * (1 - storage_leakage_per_hour) + net_flow
            residuals.append(soc[t] - expected_current)

        return np.array(residuals)

    # Only add dynamics constraint if storage exists
    if n_storage > 0:
        cons.append({'type': 'eq', 'fun': constraint_dynamics})

    # Run Solver
    # 'SLSQP' is excellent for bound-constrained problems with equality constraints
    res = minimize(
        obj_func,
        x0,
        method='SLSQP',
        bounds=bounds,
        constraints=cons,
        tol=1e-4,
        options={'maxiter': 200, 'disp': False}
    )

    return -res.fun, res.x, res.success


# ------------------------- Outer Loop: Grid Search --------------------------

def evaluate_configuration(params):
    """Wrapper for the worker pool."""
    r_mod_idx, n_r, n_s = params

    # 1. Calculate Fixed Costs (Annual)
    annual_fixed = calculate_capex_annual(r_mod_idx, n_r, n_s)
    daily_fixed = annual_fixed / 365.0

    # 2. Check Trivial Cases
    reactor_mw = reactor_models[r_mod_idx] * n_r
    if reactor_mw == 0 and n_s == 0:
        return (0.0, params, None)  # Nothing installed

    # 3. Run Continuous Optimization
    daily_op_profit, x_sol, success = solve_operational_schedule(reactor_mw, n_s)

    # 4. Total Annual Profit
    total_annual_profit = (daily_op_profit - daily_fixed) * 365.0

    if not success:
        # If the inner solver failed, penalize heavily
        return (-1e12, params, None)

    return (total_annual_profit, params, x_sol)


def run_optimization():
    print("--- Starting Decomposition Optimization ---")
    print("Phase 1: Generating Grid of Integer Configurations...")

    # Define Search Space
    # Reactor counts: 0 to 4
    # Storage counts: 0 to 50 (step 5 to save time, or step 1 for precision)
    # Models: All 5

    candidates = []

    # Iterate Combinations
    for r_idx in range(len(reactor_models)):
        for n_r in range(4):  # 0, 1, 2, 3 reactors
            # Heuristic: Don't build massive storage if no reactor (grid arb only? allows it)
            # But for this challenge, let's allow generic grid arb
            storage_steps = [0, 1] + list(range(5, 61, 5))
            for n_s in storage_steps:
                candidates.append((r_idx, n_r, n_s))

    print(f"Phase 2: Evaluating {len(candidates)} configurations using parallel processing...")

    best_profit = -np.inf
    best_config = None
    best_x = None

    start_time = time.time()

    # Parallel Execution
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(evaluate_configuration, candidates))

    # Find Winner
    for profit, params, x_sol in results:
        if profit > best_profit:
            best_profit = profit
            best_config = params
            best_x = x_sol

    duration = time.time() - start_time
    print(f" Optimization Complete in {duration:.2f} seconds.")
    print(f" Best Annual Profit: EUR {best_profit:,.2f}")
    print(f" Best Config: Model index {best_config[0]}, {best_config[1]} Reactors, {best_config[2]} Storage Modules")

    return best_profit, best_config, best_x


# ------------------------- Plotting / Result Parsing ------------------------

def plot_results(profit, config, x):
    r_idx, n_r, n_s = config

    # Reconstruct variables from X
    r_prod = x[:horizon]
    ch = x[horizon:2 * horizon]
    dis = x[2 * horizon:3 * horizon]
    soc = x[3 * horizon:]

    max_store_p = n_s * module_power_mw
    eff_d = discharge_efficiency(dis, max_store_p)
    eff_c = charge_efficiency(ch, max_store_p)

    net_supply = r_prod + dis * eff_d - ch / eff_c

    # Metrics
    hours = np.arange(horizon)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.plot(hours, electric_demand, 'k--', label='Demand', linewidth=2)
    ax1.plot(hours, r_prod, 'b-', label='Reactor Prod', linewidth=2, alpha=0.8)
    ax1.plot(hours, net_supply, 'm-', label='Net Supply', linewidth=2)

    # Stack charge/discharge for visibility
    ax1.fill_between(hours, 0, dis, color='red', alpha=0.3, label='Discharge')
    ax1.fill_between(hours, 0, -ch, color='green', alpha=0.3, label='Charge')

    ax1.set_xlabel("Hour of Day")
    ax1.set_ylabel("Power (MW)")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()
    ax2.plot(hours, soc, 'c.-', label='SOC (MWh)', linewidth=1)
    ax2.set_ylabel("State of Charge (MWh)", color='c')
    ax2.tick_params(axis='y', labelcolor='c')

    plt.title(f"Optimal Result: Profit EUR {profit:,.0f}/yr\n"
              f"Model: {reactor_models[r_idx]}MW | Count: {n_r} | Storage: {n_s} Mods")
    plt.tight_layout()
    plt.show()


# ------------------------- Main ---------------------------------------------

if __name__ == "__main__":
    # Windows multiprocessing safe-guard
    profit, config, x_sol = run_optimization()
    plot_results(profit, config, x_sol)