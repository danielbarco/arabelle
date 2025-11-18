"""
Optuna optimization script for Nuclear SMR with battery storage.

This script uses Optuna to find the optimal configuration of:
- Reactor model selection
- Number of reactors
- Number of storage modules
- Hourly reactor production schedule
- Battery state of charge schedule

Supports multi-CPU parallel optimization.
"""

import sys
from pathlib import Path
import multiprocessing as mp

# Add parent directory to path to import the main module
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import optuna
from optuna.samplers import TPESampler
from sklearn.mixture import GaussianMixture

from minlp_smr_battery_storage import (
    constraints_residuals,
    electric_demand,
    horizon,
    module_capacity_mwh,
    module_power_mw,
    objective,
    reactor_models,
    storage_leakage_per_hour,
    x_from_var,
)


def analyze_demand_with_gmm(demand, n_components=3):
    """
    Analyze demand patterns using Gaussian Mixture Model.

    Args:
        demand: Array of hourly electricity demand
        n_components: Number of demand components to identify

    Returns:
        Dictionary with GMM model and demand classifications
    """
    # Reshape for GMM: treat each hour as a sample with demand as feature
    X = demand.reshape(-1, 1)

    gmm = GaussianMixture(n_components=n_components, random_state=42, n_init=10)
    gmm.fit(X)
    labels = gmm.predict(X)

    # Sort components by mean demand
    sorted_idx = np.argsort(gmm.means_.flatten())

    return {
        "model": gmm,
        "labels": labels,
        "means": gmm.means_.flatten()[sorted_idx],
        "weights": gmm.weights_[sorted_idx],
        "sorted_idx": sorted_idx,
    }


def create_objective_function(penalty_weight=1e8, demand_gmm=None):
    """
    Create an objective function for Optuna that includes constraint penalties.

    Args:
        penalty_weight: Weight for constraint violations
        demand_gmm: Pre-computed GMM analysis of demand patterns

    Returns:
        Objective function for Optuna to maximize
    """
    if demand_gmm is None:
        demand_gmm = analyze_demand_with_gmm(electric_demand)

    def optuna_objective(trial: optuna.Trial) -> float:
        # Suggest discrete variables
        reactor_model = trial.suggest_int("reactor_model", 0, len(reactor_models) - 1)
        n_reactor = trial.suggest_int("n_reactor", 1, 5)
        n_storage = trial.suggest_int("n_storage", 0, 30)

        # Get reactor capacity
        plant_capacity = reactor_models[reactor_model] * n_reactor
        max_storage_energy = n_storage * module_capacity_mwh
        max_storage_power = n_storage * module_power_mw

        # Suggest demand-following parameter
        demand_follow = trial.suggest_float("demand_follow", 0.0, 1.0)

        # Build GMM-informed heuristic profile
        demand = electric_demand.copy()
        avg_demand = np.mean(demand)

        # Use GMM means as anchors: blend between low, medium, and high demand tracking
        gmm_low = demand_gmm["means"][0]
        gmm_high = demand_gmm["means"][-1]

        # Production strategy: follow demand more during high-demand hours
        reactor_base = np.zeros(horizon)
        for t in range(horizon):
            if demand[t] > (gmm_low + gmm_high) / 2:
                # High-demand period: follow demand more
                reactor_base[t] = demand_follow * demand[t] + (1 - demand_follow) * avg_demand
            else:
                # Low-demand period: smoother production
                reactor_base[t] = (1 - demand_follow) * demand[t] + demand_follow * avg_demand

        reactor_base = np.clip(reactor_base, 0, plant_capacity)

        # Allow Optuna to refine around the heuristic with GMM-informed bounds
        reactor_production = np.array(
            [
                trial.suggest_float(
                    f"prod_{t}",
                    max(0.0, reactor_base[t] - 0.15 * plant_capacity),
                    min(plant_capacity, reactor_base[t] + 0.15 * plant_capacity),
                )
                for t in range(horizon)
            ]
        )

        # Initialize SOC using storage logic
        soc = np.zeros(horizon)
        soc[0] = 0.30 * max_storage_energy  # Start at 30% capacity

        for t in range(horizon):
            # Compute mismatch
            mismatch = reactor_production[t] - demand[t]
            if mismatch > 0:
                # Surplus -> charge storage
                available_power = min(mismatch, max_storage_power)
                remaining_energy_cap = max_storage_energy - soc[t]
                charge_power = min(available_power, remaining_energy_cap)
                ch = charge_power
                dis = 0.0
            else:
                # Deficit -> discharge storage
                need = -mismatch
                discharge_power = min(need, max_storage_power)
                discharge_power = min(discharge_power, soc[t])
                dis = discharge_power
                ch = 0.0

            # Compute next SOC
            if t < horizon - 1:
                soc_next = soc[t] * (1 - storage_leakage_per_hour) + ch - dis
                soc[t + 1] = soc_next

        # Allow Optuna to refine SOC around the heuristic
        soc_refined = np.array(
            [
                trial.suggest_float(
                    f"soc_{t}",
                    max(0.0, soc[t] - 0.25 * max_storage_energy),
                    min(max_storage_energy, soc[t] + 0.25 * max_storage_energy),
                )
                for t in range(horizon)
            ]
        )

        # Create optimization vector
        x = x_from_var(reactor_model, n_reactor, n_storage, reactor_production, soc_refined)

        # Compute objective (annual profit)
        profit = objective(x)

        # Compute constraint violations with adaptive penalty
        residuals = constraints_residuals(x)
        min_residual = min(residuals)

        # Apply adaptive penalty based on violation severity
        if min_residual < 0:
            # Scale penalty based on how severe the violation is
            violation_severity = abs(min_residual) / (1.0 + abs(min_residual))
            adaptive_penalty = penalty_weight * violation_severity
            return profit - adaptive_penalty

        return profit

    return optuna_objective


def run_optimization_worker(study_name, storage, n_trials, worker_id):
    """
    Worker function for parallel optimization.

    Args:
        study_name: Name of the Optuna study
        storage: SQLite storage URL
        n_trials: Number of trials for this worker
        worker_id: ID of this worker process
    """
    print(f"Worker {worker_id} starting with {n_trials} trials...")

    # Load existing study from storage
    study = optuna.load_study(
        study_name=study_name,
        storage=storage,
        sampler=TPESampler(seed=42 + worker_id),
    )

    # Create objective function
    obj_func = create_objective_function()

    # Optimize
    study.optimize(obj_func, n_trials=n_trials, show_progress_bar=False)

    print(f"Worker {worker_id} completed.")


def optimize_smr(
    n_trials=2000,
    study_name="smr_optimization",
    storage_path="optuna/optuna_study.db",
    n_jobs=-1,
):
    """
    Run Optuna optimization for the SMR battery storage problem with multi-CPU support.

    Args:
        n_trials: Total number of optimization trials
        study_name: Name of the Optuna study
        storage_path: Path to SQLite database for study persistence
        n_jobs: Number of parallel jobs (-1 for all CPUs, -2 for all but one, etc.)

    Returns:
        Optimized study object
    """
    # Pre-compute demand GMM analysis
    demand_gmm = analyze_demand_with_gmm(electric_demand, n_components=3)
    print(f"Demand GMM analysis: {len(demand_gmm['means'])} components")
    print(f"  Low demand: {demand_gmm['means'][0]:.2f} MW")
    print(f"  Mid demand: {demand_gmm['means'][1]:.2f} MW")
    print(f"  High demand: {demand_gmm['means'][-1]:.2f} MW")

    # Determine number of workers
    if n_jobs == -1:
        n_workers = mp.cpu_count()
    elif n_jobs < -1:
        n_workers = max(1, mp.cpu_count() + n_jobs + 1)
    else:
        n_workers = max(1, n_jobs)

    print(f"\nRunning optimization with {n_workers} parallel workers on {mp.cpu_count()} CPUs")
    print(f"Total trials: {n_trials}")

    # Create storage
    storage = f"sqlite:///{storage_path}"

    # Create or load study
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=TPESampler(seed=42),
        storage=storage,
        load_if_exists=True,
    )

    # Distribute trials among workers
    trials_per_worker = n_trials // n_workers
    remaining_trials = n_trials % n_workers

    if n_workers == 1:
        # Single process optimization
        obj_func = create_objective_function(demand_gmm=demand_gmm)
        print(f"Starting optimization with {n_trials} trials...")
        study.optimize(obj_func, n_trials=n_trials, show_progress_bar=True)
    else:
        # Multi-process optimization
        # Note: demand_gmm cannot be pickled directly, recompute in workers
        processes = []
        for i in range(n_workers):
            worker_trials = trials_per_worker + (1 if i < remaining_trials else 0)
            p = mp.Process(
                target=run_optimization_worker, args=(study_name, storage, worker_trials, i)
            )
            processes.append(p)
            p.start()

        # Wait for all workers to complete
        for p in processes:
            p.join()

        # Reload study to get all results
        study = optuna.load_study(study_name=study_name, storage=storage)

    # Print results
    print("\n" + "=" * 80)
    print("Optimization Results")
    print("=" * 80)
    print(f"Total trials completed: {len(study.trials)}")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best value (annual profit): €{study.best_value:,.2f}")
    print("\nBest parameters:")
    for key, value in study.best_params.items():
        if key.startswith("prod_") or key.startswith("soc_"):
            continue  # Skip hourly values for brevity
        print(f"  {key}: {value}")

    # Extract and validate best solution
    best_params = study.best_params
    reactor_model = best_params["reactor_model"]
    n_reactor = best_params["n_reactor"]
    n_storage = best_params["n_storage"]

    reactor_production = np.array([best_params[f"prod_{t}"] for t in range(horizon)])
    soc = np.array([best_params[f"soc_{t}"] for t in range(horizon)])

    x_best = x_from_var(reactor_model, n_reactor, n_storage, reactor_production, soc)
    residuals = constraints_residuals(x_best)

    print(f"\nReactor model: {reactor_model} ({reactor_models[reactor_model]} MW)")
    print(f"Number of reactors: {n_reactor}")
    print(f"Number of storage modules: {n_storage}")
    print(f"Constraint satisfaction (min residual): {min(residuals):.6f}")
    print(f"Feasible: {min(residuals) >= -1e-6}")

    return study


if __name__ == "__main__":
    # Run optimization with all available CPUs
    study = optimize_smr(
        n_trials=2000,
        study_name="smr_battery_optimization",
        storage_path="optuna/optuna_study.db",
        n_jobs=-1,  # Use all CPUs
    )

    # Optional: Generate optimization history plot
    try:
        import matplotlib.pyplot as plt
        from optuna.visualization.matplotlib import (
            plot_optimization_history,
            plot_param_importances,
        )

        fig1 = plot_optimization_history(study)
        plt.tight_layout()
        plt.savefig("optuna/optimization_history.png", dpi=150)
        print("\nSaved optimization history to optuna/optimization_history.png")

        fig2 = plot_param_importances(study, params=["reactor_model", "n_reactor", "n_storage"])
        plt.tight_layout()
        plt.savefig("optuna/param_importances.png", dpi=150)
        print("Saved parameter importances to optuna/param_importances.png")

        plt.show()
    except ImportError:
        print("\nMatplotlib not available for visualization")
