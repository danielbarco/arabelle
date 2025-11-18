import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from minlp_smr_battery_storage import (
    reactor_models,
    horizon,
    module_capacity_mwh,
    module_power_mw,
    objective,
    constraints_residuals,
    compute_charge_discharge,
    electricity_supplied,
    electric_demand,
    var_from_x,
    x_from_var,
    storage_leakage_per_hour,
)

# ==============================================================================
# CYCLICAL HEURISTIC INITIALIZATION
# ==============================================================================


def compute_cyclical_initial_guess(demand, plant_cap, n_storage):
    """
    Compute initial guess for production and SOC that respects daily cyclical nature.

    Key insight: For a sustainable daily cycle, the battery must return to its
    starting state after 24 hours. This requires the total energy charged equals
    total energy discharged (accounting for losses).
    """
    max_storage_energy = n_storage * module_capacity_mwh
    max_storage_power = n_storage * module_power_mw

    # Step 1: Compute base production profile using demand smoothing
    # Use exponential weighted moving average for smoother transitions
    alpha = 0.3  # Smoothing factor
    demand_smoothed = np.copy(demand)
    for _ in range(3):  # Multiple passes for better smoothing
        demand_smoothed_new = np.zeros(horizon)
        for t in range(horizon):
            t_prev = (t - 1) % horizon
            t_next = (t + 1) % horizon
            demand_smoothed_new[t] = (
                alpha * demand_smoothed[t_prev]
                + (1 - 2 * alpha) * demand_smoothed[t]
                + alpha * demand_smoothed[t_next]
            )
        demand_smoothed = demand_smoothed_new

    # Clip to reactor capacity
    prod_guess = np.clip(demand_smoothed, 0, plant_cap)

    # Step 2: Iteratively adjust production and SOC to achieve periodicity
    # We need: SOC[0] ≈ SOC[horizon] for daily cycling
    if n_storage == 0:
        return prod_guess, np.zeros(horizon)

    # Target SOC level (middle of range for flexibility)
    target_soc = 0.5 * max_storage_energy

    # Iterative refinement (5 iterations usually enough)
    for iteration in range(5):
        soc = np.zeros(horizon)
        soc[0] = target_soc

        # Forward simulate the battery dynamics
        total_charge = 0
        total_discharge = 0

        for t in range(horizon):
            t_next = (t + 1) % horizon

            # Compute energy mismatch
            mismatch = prod_guess[t] - demand[t]

            if mismatch > 0:
                # Surplus -> charge battery
                charge_power = min(mismatch, max_storage_power)
                charge_power = min(charge_power, max_storage_energy - soc[t])
                discharge_power = 0
            else:
                # Deficit -> discharge battery
                discharge_power = min(-mismatch, max_storage_power)
                discharge_power = min(discharge_power, soc[t])
                charge_power = 0

            total_charge += charge_power
            total_discharge += discharge_power

            # Update SOC for next timestep
            if t < horizon - 1:
                soc[t + 1] = (
                    soc[t] * (1 - storage_leakage_per_hour) + charge_power - discharge_power
                )
                soc[t + 1] = np.clip(soc[t + 1], 0, max_storage_energy)

        # Check periodicity: what would SOC be if we wrapped around?
        final_mismatch = prod_guess[-1] - demand[-1]
        if final_mismatch > 0:
            final_charge = min(final_mismatch, max_storage_power, max_storage_energy - soc[-1])
            final_discharge = 0
        else:
            final_discharge = min(-final_mismatch, max_storage_power, soc[-1])
            final_charge = 0

        soc_would_be = soc[-1] * (1 - storage_leakage_per_hour) + final_charge - final_discharge

        # Adjust target SOC to improve periodicity
        soc_error = soc_would_be - target_soc

        # If we're accumulating energy, we need to produce less or discharge more
        # Adjust production slightly to balance the cycle
        if abs(soc_error) > 0.01 * max_storage_energy:
            # Adjustment factor: spread the error across the day
            energy_imbalance = soc_error
            avg_adjustment = energy_imbalance / horizon

            # Reduce production during high-production hours to fix accumulation
            prod_guess = prod_guess - avg_adjustment * 0.5
            prod_guess = np.clip(prod_guess, 0, plant_cap)

            # Update target SOC for next iteration
            target_soc = target_soc - soc_error * 0.3
            target_soc = np.clip(target_soc, 0.2 * max_storage_energy, 0.8 * max_storage_energy)
        else:
            # Converged!
            break

    return prod_guess, soc


# ==============================================================================
# 5. INTELLIGENT GRID SEARCH
# ==============================================================================


def optimize():
    print("Running Optimization with User Constraints...")
    print("-" * 80)
    print(f"{'Config':<12} {'Storage':<10} {'Profit (M€)':<15} {'Status'}")
    print("-" * 80)

    best_profit = -np.inf
    best_result = None

    # --- Wrappers to handle Fixed + Continuous variables ---
    def pack_x(continuous_x, m_idx, n_r, n_s):
        """Reconstructs the full X vector expected by the user's functions."""
        prod = continuous_x[:horizon]
        soc = continuous_x[horizon : 2 * horizon]
        # Use x_from_var to properly construct the vector
        return x_from_var(m_idx, n_r, int(n_s), prod, soc)

    def obj_wrapper(continuous_x, m_idx, n_r, n_s):
        # NEGATED for minimization (we want to maximize profit)
        return -objective(pack_x(continuous_x, m_idx, n_r, n_s))

    def cons_wrapper(continuous_x, m_idx, n_r, n_s):
        # Constraints must return an array for scipy, user function returns list
        return np.array(constraints_residuals(pack_x(continuous_x, m_idx, n_r, n_s)))

    # --- Search Loop ---
    for m_idx, m_mw in enumerate(reactor_models):
        for n_r in range(1, 4):
            plant_cap = m_mw * n_r

            # Heuristic Skip: Don't solve for massive overproduction (e.g. > 1.5x Peak)
            if plant_cap > np.max(electric_demand) * 1.5:
                continue

            # Grid search over different storage amounts
            demand_range = np.max(electric_demand) - np.min(electric_demand)
            storage_base = (demand_range * 6.0) / module_capacity_mwh

            # Try multiple storage configurations
            storage_multipliers = [0.0, 0.1, 0.2, 0.3,0.4, 0.5, 1.0]

            for multiplier in storage_multipliers:
                n_storage_fixed = max(0, int(storage_base * multiplier))

                max_storage_energy = n_storage_fixed * module_capacity_mwh
                max_storage_power = n_storage_fixed * module_power_mw

                # Use cyclical heuristic for initial guess
                prod_guess, soc_guess = compute_cyclical_initial_guess(
                    electric_demand, plant_cap, n_storage_fixed
                )

                x0_cont = np.concatenate([prod_guess, soc_guess])

                # Optimization with proper bounds
                bnds = [(0, plant_cap)] * horizon  # Production bounds
                if n_storage_fixed > 0:
                    bnds += [(0, max_storage_energy)] * horizon  # SOC bounds
                else:
                    bnds += [(0, 0)] * horizon  # No storage case

                res = minimize(
                    obj_wrapper,
                    x0_cont,
                    args=(m_idx, n_r, n_storage_fixed),
                    method="SLSQP",
                    bounds=bnds,
                    constraints={
                        "type": "ineq",
                        "fun": cons_wrapper,
                        "args": (m_idx, n_r, n_storage_fixed),
                    },
                    options={"ftol": 1e-5, "maxiter": 500, "disp": False},
                )

                # Check Feasibility using the user's function
                residuals = cons_wrapper(res.x, m_idx, n_r, n_storage_fixed)
                is_feasible = np.min(residuals) >= -1e-3

                if is_feasible:
                    profit = -res.fun  # Un-negate to get actual profit
                    print(
                        f"{m_mw}x{n_r:<6} {n_storage_fixed:<10} {profit/1e6:<15.2f} {'OK' if profit > 0 else 'Loss'}"
                    )

                    if profit > best_profit:
                        best_profit = profit
                        # Store full x including fixed vars
                        best_result = pack_x(res.x, m_idx, n_r, n_storage_fixed)
                else:
                    min_res = np.min(residuals)
                    print(
                        f"{m_mw}x{n_r:<6} {n_storage_fixed:<10} {'---':<15} {'Infeasible (min={:.2e})'.format(min_res)}"
                    )

    return best_result, best_profit


# Run
best_x, max_profit = optimize()

# ==============================================================================
# 6. RESULTS
# ==============================================================================

if best_x is not None:
    rm, nr, ns, prod, soc = var_from_x(best_x)
    rm_idx = int(round(rm))
    m_mw = reactor_models[rm_idx]

    ch, dis = compute_charge_discharge(soc)
    supplied = electricity_supplied(prod, ns, dis, ch)

    print("\n" + "=" * 40)
    print("OPTIMAL SOLUTION")
    print(f"Reactor: {int(nr)} x {m_mw} MW")
    print(f"Storage: {ns:.2f} Modules")
    print(f"Profit:  {max_profit:,.2f} EUR")
    print("=" * 40)

    # Plot
    t = np.arange(horizon)
    plt.figure(figsize=(10, 6))
    plt.plot(t, electric_demand, "k--", label="Demand")
    plt.step(t, prod, where="mid", label="Nuclear", color="blue")
    plt.plot(t, supplied, "g:", label="Grid Supply", linewidth=2)
    plt.bar(t, dis, color="red", alpha=0.3, label="Discharge", width=1.0)
    plt.bar(t, -ch, color="green", alpha=0.3, label="Charge", width=1.0)
    plt.title(f"Optimal Dispatch: {int(nr)}x{m_mw} MW")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
else:
    print("\n" + "=" * 40)
    print("NO FEASIBLE SOLUTION FOUND")
    print("=" * 40)
