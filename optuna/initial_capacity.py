import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from minlp_smr_battery_storage import *

# ==============================================================================
# 2. Grid Search Analysis for Feasibility
# ==============================================================================


def analyze_feasibility():
    print("Analyzing Feasibility Across Configurations...")
    print("-" * 60)
    print(
        f"{'Model (MW)':<12} {'Count':<8} {'Plant Cap':<12} {'Avg Demand':<12} {'Feasibility Check'}"
    )
    print("-" * 60)

    avg_demand = np.mean(electric_demand)
    peak_demand = np.max(electric_demand)
    min_demand = np.min(electric_demand)

    results = []

    for m_idx, m_mw in enumerate(reactor_models):
        for n_r in range(1, 4):
            plant_cap = m_mw * n_r

            status = "Likely Feasible"
            reason = ""

            # Heuristic Checks
            if plant_cap < min_demand * 0.5:
                status = "Difficult"
                reason = "Very Low Cap"
            elif plant_cap > peak_demand * 2.0:
                status = "Difficult"
                reason = "Massive Excess"

            print(
                f"{m_mw:<12.0f} {n_r:<8} {plant_cap:<12.0f} {avg_demand:<12.0f} {status} {reason}"
            )

            # Store for plotting
            results.append({"model": m_mw, "count": n_r, "cap": plant_cap, "status": status})

    return results


# Run Analysis
feasibility_data = analyze_feasibility()

# ==============================================================================
# 3. Plotting Feasibility Map
# ==============================================================================

plt.figure(figsize=(10, 6))

caps = [d["cap"] for d in feasibility_data]
model_labels = [f"{d['model']}MW x {d['count']}" for d in feasibility_data]
y_pos = np.arange(len(caps))

plt.barh(y_pos, caps, align="center", alpha=0.7, color="skyblue")
plt.yticks(y_pos, model_labels)
plt.xlabel("Plant Capacity (MW)")
plt.title("Grid Search Configurations vs Demand Profile")

# Add Demand Lines
plt.axvline(x=np.min(electric_demand), color="green", linestyle="--", label="Min Demand (140 MW)")
plt.axvline(x=np.mean(electric_demand), color="orange", linestyle="--", label="Avg Demand")
plt.axvline(x=np.max(electric_demand), color="red", linestyle="--", label="Peak Demand (368 MW)")

plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("optuna/feasibility_map.png")


# ==============================================================================
# ECONOMIC ANALYSIS
# ==============================================================================
def get_market_price(demand):
    peak = np.max(demand)
    # price_base + price_sensitivity * (demand / peak)
    return price_base + price_sensitivity * (demand / peak)

# --- Analysis ---
def analyze_profitability():
    print(f"{'Config':<12} {'Gen (MW)':<10} {'Revenue (M€)':<14} {'Cost (M€)':<12} {'Profit (M€)':<12} {'Verdict'}")
    print("-" * 75)
    
    results = []
    
    # Calculate hourly prices once
    hourly_prices = get_market_price(electric_demand)
    avg_market_price = np.mean(hourly_prices)
    print(f"Calculated Avg Market Price: {avg_market_price:.2f} EUR/MWh")
    print("-" * 75)

    for m_mw in reactor_models:
        for n_r in range(1, 4):
            # 1. Generation
            plant_cap = m_mw * n_r
            
            # 2. Operational Profile (Baseload Assumption)
            hourly_gen = np.full_like(electric_demand, plant_cap)
            
            # Energy Flows
            sold_to_grid = np.minimum(hourly_gen, electric_demand)
            surplus = np.maximum(0.0, hourly_gen - electric_demand)
            
            # 3. Revenue
            # Sold energy gets market price
            revenue_market = np.sum(sold_to_grid * hourly_prices) * 365
            # Surplus gets surplus price
            revenue_surplus = np.sum(surplus * price_surplus) * 365
            
            total_revenue = revenue_market + revenue_surplus
            
            # 4. Costs
            # CAPEX
            capex_total = n_r * (reactor_cap_a * m_mw**reactor_cap_b)
            ann_capex = capex_total * reactor_annuity_factor
            # O&M
            ann_om = capex_total * reactor_fixed_om_frac
            # Fuel (Full power)
            ann_fuel = np.sum(hourly_gen) * fuel_price * 365
            
            total_cost = ann_capex + ann_om + ann_fuel
            
            # 5. Profit
            profit = total_revenue - total_cost
            
            # Verdict
            if profit > 0:
                verdict = "PROFITABLE"
                color = 'green'
            elif profit > -0.1 * total_cost: # Close call
                verdict = "MARGINAL"
                color = 'orange'
            else:
                verdict = "LOSS"
                color = 'red'

            print(f"{m_mw}x{n_r:<5} {plant_cap:<10.0f} {total_revenue/1e6:<14.1f} {total_cost/1e6:<12.1f} {profit/1e6:<12.1f} {verdict}")
            
            results.append({
                'label': f"{m_mw}x{n_r}",
                'cap': plant_cap,
                'profit': profit,
                'color': color
            })
            
    return results, avg_market_price

data, avg_price = analyze_profitability()

# --- Plotting ---
labels = [d['label'] for d in data]
profits = [d['profit']/1e6 for d in data]
colors = [d['color'] for d in data]

plt.figure(figsize=(12, 6))
plt.bar(labels, profits, color=colors)
plt.axhline(0, color='black', linewidth=0.8)
plt.ylabel('Annual Profit (M EUR)')
plt.title(f'Profitability Analysis (Avg Price: {avg_price:.1f} €/MWh)')
plt.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('optuna/profitability_analysis.png')