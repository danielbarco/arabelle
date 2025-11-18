import json
import numpy as np
from scipy.optimize import minimize
import warnings
import multiprocessing
import time
import os

# Import the module. In a multiprocessing environment (especially 'fork'),
# the module state is copied or re-imported. We will modify 'electric_demand'
# inside the worker function to ensure isolation.
import minlp_smr_battery_storage as msbs

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# ==============================================================================
# HELPER: CYCLICAL INITIALIZATION
# ==============================================================================


def compute_cyclical_initial_guess(demand, plant_cap, n_storage):
    """
    Compute initial guess for production and SOC that respects daily cyclical nature.
    """
    max_storage_energy = n_storage * msbs.module_capacity_mwh
    max_storage_power = n_storage * msbs.module_power_mw

    # Smoothing
    alpha = 0.3
    demand_smoothed = np.copy(demand)
    for _ in range(3):
        demand_smoothed_new = np.zeros(msbs.horizon)
        for t in range(msbs.horizon):
            t_prev = (t - 1) % msbs.horizon
            t_next = (t + 1) % msbs.horizon
            demand_smoothed_new[t] = (
                alpha * demand_smoothed[t_prev]
                + (1 - 2 * alpha) * demand_smoothed[t]
                + alpha * demand_smoothed[t_next]
            )
        demand_smoothed = demand_smoothed_new

    prod_guess = np.clip(demand_smoothed, 0, plant_cap)

    if n_storage == 0:
        return prod_guess, np.zeros(msbs.horizon)

    target_soc = 0.5 * max_storage_energy

    for iteration in range(5):
        soc = np.zeros(msbs.horizon)
        soc[0] = target_soc
        total_charge = 0
        total_discharge = 0

        for t in range(msbs.horizon):
            mismatch = prod_guess[t] - demand[t]

            if mismatch > 0:
                charge_power = min(mismatch, max_storage_power)
                charge_power = min(charge_power, max_storage_energy - soc[t])
                discharge_power = 0
            else:
                discharge_power = min(-mismatch, max_storage_power)
                discharge_power = min(discharge_power, soc[t])
                charge_power = 0

            total_charge += charge_power
            total_discharge += discharge_power

            if t < msbs.horizon - 1:
                soc[t + 1] = (
                    soc[t] * (1 - msbs.storage_leakage_per_hour) + charge_power - discharge_power
                )
                soc[t + 1] = np.clip(soc[t + 1], 0, max_storage_energy)

        # Check periodicity
        final_mismatch = prod_guess[-1] - demand[-1]
        if final_mismatch > 0:
            final_charge = min(final_mismatch, max_storage_power, max_storage_energy - soc[-1])
            final_discharge = 0
        else:
            final_discharge = min(-final_mismatch, max_storage_power, soc[-1])
            final_charge = 0

        soc_would_be = (
            soc[-1] * (1 - msbs.storage_leakage_per_hour) + final_charge - final_discharge
        )
        soc_error = soc_would_be - target_soc

        if abs(soc_error) > 0.01 * max_storage_energy:
            energy_imbalance = soc_error
            avg_adjustment = energy_imbalance / msbs.horizon
            prod_guess = prod_guess - avg_adjustment * 0.5
            prod_guess = np.clip(prod_guess, 0, plant_cap)
            target_soc = target_soc - soc_error * 0.3
            target_soc = np.clip(target_soc, 0.2 * max_storage_energy, 0.8 * max_storage_energy)
        else:
            break

    return prod_guess, soc


# ==============================================================================
# WORKER FUNCTION
# ==============================================================================


def solve_profile_task(args):
    """
    Worker function to process a single demand profile.
    Returns a dictionary with all feasible scenarios found.
    """
    idx, profile = args

    # CRITICAL: Set global demand for this worker process instance of the module
    # This ensures the objective function uses the correct demand curve.
    msbs.electric_demand = profile

    feasible_scenarios = []

    # --- Wrappers ---
    def pack_x(continuous_x, m_idx, n_r, n_s):
        prod = continuous_x[: msbs.horizon]
        soc = continuous_x[msbs.horizon : 2 * msbs.horizon]
        return msbs.x_from_var(m_idx, n_r, int(n_s), prod, soc)

    def obj_wrapper(continuous_x, m_idx, n_r, n_s):
        return -msbs.objective(pack_x(continuous_x, m_idx, n_r, n_s))

    def cons_wrapper(continuous_x, m_idx, n_r, n_s):
        return np.array(msbs.constraints_residuals(pack_x(continuous_x, m_idx, n_r, n_s)))

    # --- Search Loop ---
    for m_idx, m_mw in enumerate(msbs.reactor_models):
        for n_r in range(1, 4):
            plant_cap = m_mw * n_r

            # Heuristic Skip: Don't optimize if capacity is vastly larger than peak demand
            if plant_cap > np.max(profile) * 1.5:
                continue

            # Grid search over different storage amounts
            demand_range = np.max(profile) - np.min(profile)
            storage_base = (demand_range * 6.0) / msbs.module_capacity_mwh

            # Storage multipliers
            storage_multipliers = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0]

            for multiplier in storage_multipliers:
                n_storage_fixed = max(0, int(storage_base * multiplier))
                max_storage_energy = n_storage_fixed * msbs.module_capacity_mwh

                # Initial Guess
                prod_guess, soc_guess = compute_cyclical_initial_guess(
                    profile, plant_cap, n_storage_fixed
                )
                x0_cont = np.concatenate([prod_guess, soc_guess])

                # Bounds
                bnds = [(0, plant_cap)] * msbs.horizon
                if n_storage_fixed > 0:
                    bnds += [(0, max_storage_energy)] * msbs.horizon
                else:
                    bnds += [(0, 0)] * msbs.horizon

                try:
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

                    # Check Feasibility
                    residuals = cons_wrapper(res.x, m_idx, n_r, n_storage_fixed)
                    is_feasible = np.min(residuals) >= -1e-3

                    if is_feasible:
                        profit = -res.fun

                        # Reconstruct full solution
                        full_x = pack_x(res.x, m_idx, n_r, n_storage_fixed)
                        _, _, _, prod, soc = msbs.var_from_x(full_x)
                        ch, dis = msbs.compute_charge_discharge(soc)

                        # Store this scenario
                        feasible_scenarios.append(
                            {
                                "m_mw": float(m_mw),
                                "n_r": int(n_r),
                                "n_storage_fixed": int(n_storage_fixed),
                                "profit_eur": float(profit),
                                "prod": prod.tolist(),
                                "soc": soc.tolist(),
                                "charge": ch.tolist(),
                                "discharge": dis.tolist(),
                            }
                        )

                except Exception:
                    continue

    return {
        "index": idx,
        "demand_profile": profile.tolist(),
        "count_feasible": len(feasible_scenarios),
        "scenarios": feasible_scenarios,
    }


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================


def main():
    print("Loading energy_demands.json...")
    try:
        with open("data_gen/energy_demands.json", "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print("Error: energy_demands.json not found.")
        return

    profiles = data.get("energy_demands", [])
    tasks = [(p["index"], np.array(p["profile"])) for p in profiles]

    cpu_count = multiprocessing.cpu_count()
    print(f"Starting parallel processing of {len(tasks)} profiles using {cpu_count} cores...")

    start_time = time.time()

    # Run parallel processing
    with multiprocessing.Pool(processes=cpu_count) as pool:
        results = pool.map(solve_profile_task, tasks)

    elapsed_time = time.time() - start_time
    print(f"Processing complete in {elapsed_time:.2f} seconds.")

    # Prepare Final JSON
    output_filename = "data_gen/results.json"
    final_output = {
        "description": "Grid search results containing all feasible SMR+Battery configurations.",
        "count": len(results),
        "results": results,
    }

    with open(output_filename, "w") as f:
        json.dump(final_output, f, indent=2)

    print("-" * 70)
    print(f"Results for {len(results)} profiles saved to {output_filename}")


if __name__ == "__main__":
    main()
