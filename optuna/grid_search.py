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

            # Try multiple storage configurations: 0%, 50%, 100%, 150%, 200% of base estimate
            storage_multipliers = [0.0, 0.5, 1.0, 1.5, 2.0]

            for multiplier in storage_multipliers:
                n_storage_fixed = max(0, int(storage_base * multiplier))

                max_storage_energy = n_storage_fixed * module_capacity_mwh
                max_storage_power = n_storage_fixed * module_power_mw

                # Initial Guess: smooth production with battery cycling
                avg_demand = np.mean(electric_demand)
                prod_guess = np.full(horizon, min(avg_demand, plant_cap))

                # Initialize SOC with a realistic cycling pattern
                soc_guess = np.zeros(horizon)
                if n_storage_fixed > 0:
                    soc_guess[0] = 0.5 * max_storage_energy  # Start at 50%

                    # Simple forward simulation for initial SOC guess
                    for t in range(horizon - 1):
                        mismatch = prod_guess[t] - electric_demand[t]
                        charge_power = np.clip(mismatch, -max_storage_power, max_storage_power)
                        soc_next = soc_guess[t] * (1 - storage_leakage_per_hour) + charge_power
                        soc_guess[t + 1] = np.clip(soc_next, 0, max_storage_energy)

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
                    constraints={"type": "ineq", "fun": cons_wrapper, "args": (m_idx, n_r, n_storage_fixed)},
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
