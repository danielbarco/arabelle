"""
main_launcher.py
Orchestrates the benchmark with unified configuration management.
"""

import time
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import random
import dataclasses # Needed to modify settings safely
import common

# Import Solvers
import solver_rough_grid
import solver_refined
import solver_ga
import solver_slsqp

# --- GLOBAL SETUP ---
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'legend.fontsize': 10
})
random.seed(2025)
np.random.seed(2025)

# --- HELPER: Wrapper for Refined SLSQP ---
def run_refined_slsqp_wrapper(settings):
    """
    Intercepts the settings object and forces storage_step=1
    before calling the standard SLSQP solver.
    """
    # Create a new settings object based on the incoming one, but override the step
    refined_settings = dataclasses.replace(settings, storage_step=1)
    return solver_slsqp.run_solver(refined_settings)


def run_analysis_plots(df):
    """
    Generates comparative analytics plots:
    1. Pareto Frontier (Time vs Profit)
    2. Combined Profile Overlay
    """

    # --- DEFINE COLOR PALETTE ---
    # Group similar approaches with similar colors (shades)
    algo_colors = {
        "Rough Grid": "#5DADE2",       # Light Blue
        "Refined Grid": "#1B4F72",     # Dark Blue
        "SLSQP (Rough)": "#82E0AA",    # Light Green
        "SLSQP (Refined)": "#196F3D",  # Dark Green
        "Genetic Alg": "#E74C3C"       # Red
    }

    # --- PLOT 1: Pareto Frontier (Scatter) ---
    fig1, ax1 = plt.subplots(figsize=(12, 8))

    # Jitter for visibility in scatter plot
    jitter_x = df['Duration (s)'] * np.random.uniform(0.95, 1.05, len(df))

    # Map markers to Scenario
    markers = {'Fixed 30% SOC': 'X', 'Optimized SOC': 'o'}

    # Get unique algorithms present
    unique_algos = sorted(df['Algorithm'].unique())

    for idx, row in df.iterrows():
        alg = row['Algorithm']
        c = algo_colors.get(alg, 'black') # Fallback to black
        m = markers.get(row['Assumption_Set'], 'o')

        ax1.scatter(jitter_x.iloc[idx], row['Annual Profit'],
                    s=200, color=c, marker=m, edgecolors='k', alpha=0.8, zorder=3)

        # Annotation
        label = f"{alg}\n({row['Assumption_Set']})"
        # Alternate text position to avoid overlap
        xy_offset = (0, 12) if idx % 2 == 0 else (0, -18)

        ax1.annotate(label, (jitter_x.iloc[idx], row['Annual Profit']),
                     xytext=xy_offset, textcoords='offset points', fontsize=9, ha='center')

    ax1.set_xlabel("Execution Time (s)")
    ax1.set_ylabel("Annual Profit (EUR)")
    ax1.set_title("Benchmark: Efficiency Frontier\n(Circle=Optimized, Cross=Fixed 30%)")
    ax1.grid(True, alpha=0.3)

    # Custom Legend for Plot 1
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', color='w', label='Optimized SOC', markerfacecolor='gray', markersize=10),
                       Line2D([0], [0], marker='X', color='w', label='Fixed 30% SOC', markerfacecolor='gray', markersize=10)]
    for alg in unique_algos:
        col = algo_colors.get(alg, 'black')
        legend_elements.append(Line2D([0], [0], marker='s', color='w', label=alg, markerfacecolor=col, markersize=10))

    ax1.legend(handles=legend_elements, loc='lower right')
    plt.tight_layout()
    plt.show()


    # --- PLOT 2: Combined Operational Profiles ---
    fig2, (ax_r, ax_s, ax_net) = plt.subplots(3, 1, figsize=(14, 14), sharex=True)

    hours = np.arange(common.HORIZON)

    # Plot Demand (Base layer)
    ax_net.plot(hours, common.ELECTRIC_DEMAND, 'k--', label='Grid Demand', lw=2.5, zorder=10)

    # Plot every result in the dataframe
    for idx, row in df.iterrows():
        raw = row['Raw_Result']
        x_vec = raw['x']

        # Reconstruct physics
        n_s = row['Raw_Config'][2] # Access from stored config
        max_p = n_s * common.MODULE_POWER_MW
        H = common.HORIZON

        r_prod = x_vec[:H]
        ch = x_vec[H:2*H]
        dis = x_vec[2*H:3*H]
        soc = x_vec[3*H:]

        eff_c = common.charge_efficiency(ch, max_p)
        eff_d = common.discharge_efficiency(dis, max_p)
        net_supply = r_prod + dis * eff_d - ch / eff_c

        # Styling based on Algorithm Type and Scenario
        alg = row['Algorithm']
        color = algo_colors.get(alg, 'black')

        # Dash style distinguishes the scenario
        style = '-' if row['Assumption_Set'] == 'Optimized SOC' else ':'
        width = 2.0
        alpha = 0.8

        label_str = f"{alg} ({row['Assumption_Set']})"

        ax_r.plot(hours, r_prod, color=color, ls=style, lw=width, alpha=alpha, label=label_str)
        ax_s.plot(hours, soc, color=color, ls=style, lw=width, alpha=alpha)
        ax_net.plot(hours, net_supply, color=color, ls=style, lw=width, alpha=alpha)

    ax_r.set_title("Reactor Production Strategy")
    ax_r.set_ylabel("Power (MW)")
    ax_r.grid(True, alpha=0.3)

    # LEGEND: Placed at the top to cover all lines
    ax_r.legend(loc='upper center', bbox_to_anchor=(0.5, 1.4), ncol=3, fontsize=9, title="Solver Strategies")

    ax_s.set_title("Battery State of Charge (SOC)")
    ax_s.set_ylabel("Energy (MWh)")
    ax_s.grid(True, alpha=0.3)

    ax_net.set_title("Net Supply to Grid (vs Demand)")
    ax_net.set_ylabel("Power (MW)")
    ax_net.set_xlabel("Hour of Day")
    ax_net.grid(True, alpha=0.3)

    # Specific Legend for Demand on the bottom plot
    ax_net.legend(loc='upper right', fontsize=9)

    plt.tight_layout()
    plt.show()

def main():
    print("=======================================================")
    print("   UNIFIED SMR BENCHMARK: MULTI-SCENARIO")
    print("=======================================================")

    results_store = []

    # --- DEFINE SCENARIOS ---
    # Note: storage_step=5 is the default "Rough" setting for the main loop
    config_strict = common.SimSettings(fixed_initial_soc=0.3, storage_step=5)
    config_opt = common.SimSettings(fixed_initial_soc=None, storage_step=5)

    scenarios = [
        ("Fixed 30% SOC", config_strict),
        ("Optimized SOC", config_opt)
    ]

    solvers = [
        ("Rough Grid", solver_rough_grid.run_solver),
        ("Refined Grid", solver_refined.run_solver), # Checks every 1 unit (internal logic)
        ("SLSQP (Rough)", solver_slsqp.run_solver), # Checks every 5 units (via settings)
        ("SLSQP (Refined)", run_refined_slsqp_wrapper), # Checks every 1 unit (via wrapper)
        ("Genetic Alg", solver_ga.run_solver),
    ]

    for scenario_name, settings in scenarios:
        print(f"\n>>> SCENARIO: {scenario_name}")

        for alg_name, solver_func in solvers:
            print(f"    Running {alg_name}...", end="", flush=True)
            start_time = time.time()

            result = solver_func(settings)

            duration = time.time() - start_time
            print(f" Done ({duration:.2f}s). Profit: {result['profit']/1e6:.2f}M")

            results_store.append({
                "Algorithm": alg_name,
                "Assumption_Set": scenario_name,
                "Duration (s)": duration,
                "Annual Profit": result['profit'],
                "Configuration": result['config'], # Store raw tuple
                "Raw_Config": result['config'],
                "Raw_Result": result
            })

            # Note: common.plot_result is skipped here in favor of the final consolidated plot
            # common.plot_result(result, alg_name, duration, scenario_name)

    # --- SUMMARY ---
    df = pd.DataFrame(results_store)
    print("\n=======================================================")
    print("                 FINAL RESULTS")
    print("=======================================================")

    df['Gap (%)'] = 0.0
    for scen in df['Assumption_Set'].unique():
        mask = df['Assumption_Set'] == scen
        best_in_scen = df.loc[mask, 'Annual Profit'].max()
        df.loc[mask, 'Gap (%)'] = (best_in_scen - df.loc[mask, 'Annual Profit']) / best_in_scen * 100

    print(df[["Algorithm", "Assumption_Set", "Annual Profit", "Duration (s)", "Gap (%)"]]
          .sort_values(by=["Assumption_Set", "Annual Profit"], ascending=[True, False])
          .to_string(formatters={'Annual Profit': '{:,.0f}'.format, 'Duration (s)': '{:.2f}'.format, 'Gap (%)': '{:.2f}%'.format}))

    print("\nGenerating Comparative Analysis Plots...")
    run_analysis_plots(df)

if __name__ == "__main__":
    main()