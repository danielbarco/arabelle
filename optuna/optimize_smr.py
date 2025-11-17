"""
Optuna optimization script for Nuclear SMR with battery storage.

This script uses Optuna to find the optimal configuration of:
- Reactor model selection
- Number of reactors
- Number of storage modules
- Hourly reactor production schedule
- Battery state of charge schedule
"""

import sys
from pathlib import Path

# Add parent directory to path to import the main module
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import optuna
from optuna.samplers import TPESampler

from minlp_smr_battery_storage import (
    constraints_residuals,
    horizon,
    module_capacity_mwh,
    module_power_mw,
    objective,
    reactor_models,
    storage_leakage_per_hour,
    x_from_var,
)


def create_objective_function(penalty_weight=1e8):
    """
    Create an objective function for Optuna that includes constraint penalties.

    Args:
        penalty_weight: Weight for constraint violations

    Returns:
        Objective function for Optuna to maximize
    """

    def optuna_objective(trial: optuna.Trial) -> float:
        # Suggest discrete variables
        reactor_model = trial.suggest_int("reactor_model", 0, len(reactor_models) - 1)
        n_reactor = trial.suggest_int("n_reactor", 1, 5)
        n_storage = trial.suggest_int("n_storage", 0, 30)

        # Get reactor capacity
        plant_capacity = reactor_models[reactor_model] * n_reactor
        max_storage_energy = n_storage * module_capacity_mwh

        # Suggest reactor production for each hour
        reactor_production = np.array(
            [trial.suggest_float(f"prod_{t}", 0.0, plant_capacity) for t in range(horizon)]
        )

        # Suggest state of charge for each hour
        soc = np.array(
            [trial.suggest_float(f"soc_{t}", 0.0, max_storage_energy) for t in range(horizon)]
        )

        # Create optimization vector
        x = x_from_var(reactor_model, n_reactor, n_storage, reactor_production, soc)

        # Compute objective (annual profit)
        profit = objective(x)

        # Compute constraint violations
        residuals = constraints_residuals(x)
        min_residual = min(residuals)

        # Apply penalty for constraint violations
        if min_residual < 0:
            penalty = penalty_weight * abs(min_residual)
            return profit - penalty

        return profit

    return optuna_objective


def optimize_smr(n_trials=100, study_name="smr_optimization", storage_path=optuna):
    """
    Run Optuna optimization for the SMR battery storage problem.

    Args:
        n_trials: Number of optimization trials
        study_name: Name of the Optuna study
        storage_path: Optional path to SQLite database for study persistence

    Returns:
        Optimized study object
    """
    # Create storage if path provided
    storage = None
    if storage_path:
        storage = f"sqlite:///{storage_path}"

    # Create study (maximize profit)
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=TPESampler(seed=42),
        storage=storage,
        load_if_exists=True,
    )

    # Create objective function
    obj_func = create_objective_function()

    # Optimize
    print(f"Starting optimization with {n_trials} trials...")
    study.optimize(obj_func, n_trials=n_trials, show_progress_bar=True)

    # Print results
    print("\n" + "=" * 80)
    print("Optimization Results")
    print("=" * 80)
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
    # Run optimization
    study = optimize_smr(
        n_trials=2000, study_name="smr_battery_optimization", storage_path="optuna/optuna_study.db"
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
        print("\nSaved optimization history to optimization_history.png")

        fig2 = plot_param_importances(study, params=["reactor_model", "n_reactor", "n_storage"])
        plt.tight_layout()
        plt.savefig("param_importances.png", dpi=150)
        print("Saved parameter importances to param_importances.png")

        plt.show()
    except ImportError:
        print("\nMatplotlib not available for visualization")
