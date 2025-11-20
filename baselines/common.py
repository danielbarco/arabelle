"""
common.py
Shared constants, physics, visualization, and Configuration Objects.
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional, List

# --- 1. Problem Constants ---
HORIZON = 24
ELECTRIC_DEMAND = np.array(
    [160, 152, 144, 140, 144, 160, 200, 240, 280, 300, 320, 340, 360, 368, 352,
     340, 320, 312, 300, 280, 260, 240, 220, 192], dtype=np.float64
)

# Hardware Catalog
REACTOR_MODELS = [80.0, 160.0, 300.0, 350.0, 470.0]

# Economic Parameters
INTEREST_RATE = 0.04
REACTOR_YEARS = 60
STORAGE_YEARS = 20
REACTOR_CAP_A = 2.0e7
REACTOR_CAP_B = 0.8
MODULE_COST = 1.0e7

# Operational
REACTOR_FIXED_OM = 0.03
STORAGE_FIXED_OM = 0.02
FUEL_PRICE = 5.0
STORAGE_LEAKAGE = 0.0008

# Technical
MODULE_CAP_MWH = 50.0
MODULE_POWER_MW = 10.0

# Market
PRICE_BASE = 70.0
PRICE_SENSITIVITY = 60.0
PRICE_SURPLUS = 10.0

# --- 2. Computed Factors ---
R_ANNUITY = INTEREST_RATE / (1 - (1 + INTEREST_RATE) ** (-REACTOR_YEARS))
S_ANNUITY = INTEREST_RATE / (1 - (1 + INTEREST_RATE) ** (-STORAGE_YEARS))

def get_market_prices(demand):
    peak = np.max(demand)
    return PRICE_BASE + PRICE_SENSITIVITY * demand / peak

MARKET_PRICES = get_market_prices(ELECTRIC_DEMAND)

# --- 3. Configuration Object ---
@dataclass
class SimSettings:
    """
    Controls the assumptions and search space for a benchmark run.
    """
    # Constraints
    fixed_initial_soc: Optional[float] = None  # If None, optimizer chooses. If 0.3, forced to 30%.

    # Search Space Restrictions (for fair comparison)
    max_reactors: int = 5
    max_storage: int = 20

    # Granularity (Brute vs Refined distinction)
    storage_step: int = 1  # 1 for refined, 5 for brute/coarse

# --- 4. Physics Functions ---

def charge_efficiency(charge_mw, max_charge_mw):
    if np.isscalar(max_charge_mw) and max_charge_mw < 1e-6:
        return 1.0 if np.isscalar(charge_mw) else np.ones_like(charge_mw)
    safe_max = np.maximum(max_charge_mw, 1e-6)
    r = 0.95 - 0.15 * ((charge_mw / safe_max) ** 2)
    return np.maximum(r, 0.6)

def discharge_efficiency(dis_mw, max_dis_mw):
    if np.isscalar(max_dis_mw) and max_dis_mw < 1e-6:
        return 1.0 if np.isscalar(dis_mw) else np.ones_like(dis_mw)
    safe_max = np.maximum(max_dis_mw, 1e-6)
    r = 0.96 - 0.2 * ((dis_mw / safe_max) ** 2)
    return np.maximum(r, 0.55)

def calculate_annual_fixed_cost(reactor_idx, n_reactor, n_storage):
    if n_reactor == 0 and n_storage == 0: return 0.0
    c_reac = 0.0
    if n_reactor > 0:
        cap_mw = REACTOR_MODELS[reactor_idx]
        c_reac = n_reactor * (REACTOR_CAP_A * cap_mw ** REACTOR_CAP_B)
    c_stor = n_storage * MODULE_COST
    ann_capex = c_reac * R_ANNUITY + c_stor * S_ANNUITY
    ann_om = c_reac * REACTOR_FIXED_OM + c_stor * STORAGE_FIXED_OM
    return ann_capex + ann_om

def calculate_daily_profit(reactor_prod, ch, dis, n_storage):
    max_p = n_storage * MODULE_POWER_MW
    eff_d = discharge_efficiency(dis, max_p)
    eff_c = charge_efficiency(ch, max_p)
    supplied = reactor_prod + dis * eff_d - ch / eff_c
    local_supply = np.minimum(supplied, ELECTRIC_DEMAND)
    unmet = np.maximum(0.0, ELECTRIC_DEMAND - supplied)
    surplus = np.maximum(0.0, supplied - ELECTRIC_DEMAND)
    revenue = np.sum(MARKET_PRICES * local_supply)
    penalty = np.sum(MARKET_PRICES * unmet)
    surplus_rev = np.sum(PRICE_SURPLUS * surplus)
    fuel_cost = np.sum(reactor_prod) * FUEL_PRICE
    return revenue + surplus_rev - penalty - fuel_cost

# --- 5. Visualization ---
def plot_result(result_dict, algorithm_name, duration, settings_desc):
    x = result_dict['x']
    r_idx, n_r, n_s = result_dict['config']
    profit = result_dict['profit']

    r_prod = x[:HORIZON]
    ch = x[HORIZON:2 * HORIZON]
    dis = x[2 * HORIZON:3 * HORIZON]
    soc = x[3 * HORIZON:]

    max_p = n_s * MODULE_POWER_MW
    eff_d = discharge_efficiency(dis, max_p)
    eff_c = charge_efficiency(ch, max_p)
    net_supply = r_prod + dis * eff_d - ch / eff_c

    hours = np.arange(HORIZON)
    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.plot(hours, ELECTRIC_DEMAND, 'k--', label='Demand', lw=2, zorder=5)
    ax1.plot(hours, r_prod, 'b-', label='Reactor', lw=2, alpha=0.8)
    ax1.plot(hours, net_supply, 'm-', label='Net Supply', lw=1.5)
    ax1.fill_between(hours, 0, dis, color='red', alpha=0.3, label='Discharge')
    ax1.fill_between(hours, 0, -ch, color='green', alpha=0.3, label='Charge')

    ax1.set_ylabel("Power (MW)")
    ax1.set_title(f"Algo: {algorithm_name} ({settings_desc})\n"
                  f"Profit: EUR {profit:,.0f}/yr | Time: {duration:.2f}s")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(hours, soc, 'c.-', label='SOC', lw=1.5)
    ax2.set_ylabel("SOC (MWh)", color='c')

    plt.tight_layout()
    plt.show()