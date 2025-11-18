"""
© 2025, Arabelle Solutions and/or its affiliates. All rights reserved.

This Python file is provided for experimentation only in the context of
the AIM Week 2025 challenge.
NO REPRESENTATION OR WARRANTY IS MADE OR IMPLIED AS TO ITS COMPLETENESS,
ACCURACY, OR FITNESS FOR ANY PARTICULAR PURPOSE.
MINLP sample: Nuclear SMR with battery storage modules.

This script implements an optimization problem with the following optimization
variables:
- reactor_model: Reactor model to install.
- n_reactor: Number of reactors to install.
- n_storage: Number of storage modules (integer).
- reactor_production: Reactor hourly electric power output (MW, array).
- soc: Battery hourly state of charge (MWh, array).

The objective function is the annual profit, defined as the market net revenue
from which are subtracted annual operating costs, annualized CAPEX and fuel costs.

The inequality constraints allow to fulfill:
- Reactors power output bounding.
- Charge, discharge and SOC bounding.

Two example candidates are implemented and evaluated:
1. Smoothed nuclear production compensated with storage to meet demand.
2. No storage, reactor follows demand (limited by capacity).
"""

import matplotlib.pyplot as plt
import numpy as np

# ------------------------- Problem data -------------------------------------
horizon = 24  # hourly horizon

# Hourly electric demand (MW) - daily profile
electric_demand = np.array(
    [
        160,
        152,
        144,
        140,
        144,
        160,
        200,
        240,
        280,
        300,
        320,
        340,
        360,
        368,
        352,
        340,
        320,
        312,
        300,
        280,
        260,
        240,
        220,
        192,
    ],
    dtype=np.float64,
)

# SMR models power offerings
reactor_models = [80.0, 160.0, 300.0, 350.0, 470.0]

# Economic parameters
interest_rate = 0.04

# Reactor capital cost (power-law, economies of scale)
reactor_cap_a = 2.0e7
reactor_cap_b = 0.8
reactor_years = 60  # Life expectancy of reactor assumed at 60 years
reactor_annuity_factor = interest_rate / (1 - (1 + interest_rate) ** (-reactor_years))

# Storage module
module_capacity_mwh = 50.0  # MWh per module
module_power_mw = 10.0  # MW per module
module_cost = 1.0e7  # EUR per module
storage_leakage_per_hour = 0.0008
storage_years = 20  # Battery modules need to be replaced after 20 years
storage_annuity_factor = interest_rate / (1 - (1 + interest_rate) ** (-storage_years))

# Operational costs
reactor_fixed_om_frac = 0.03  # fraction of capex per year
storage_fixed_om_frac = 0.02
fuel_price = 5.0  # EUR/MWh

# Market pricing
price_base = 70.0  # EUR/MWh base
price_sensitivity = 60.0  # Scales how high price goes with demand
price_surplus = 10.0  # EUR/MWh for surplus sold to market (negative price)


# Storage efficiencies
def charge_efficiency(charge_mw: np.ndarray, max_charge_mw: float) -> np.ndarray:
    if max_charge_mw < 1.0e-6:
        return np.ones(horizon)
    else:
        r = 0.95 - 0.15 * (charge_mw / max_charge_mw) ** 2
        return np.maximum(r, 0.6)


def discharge_efficiency(dis_mw: np.ndarray, max_dis_mw: float) -> np.ndarray:
    if max_dis_mw < 1.0e-6:
        return np.ones(horizon)
    else:
        r = 0.96 - 0.2 * (dis_mw / max_dis_mw) ** 2
    return np.maximum(r, 0.55)


# ------------------------- Model evaluation functions -----------------------


def capital_cost_reactor(reactor_capacity: float, n_reactor: int) -> float:
    return n_reactor * (reactor_cap_a * reactor_capacity**reactor_cap_b)


def capital_cost_storage(n_storage: int) -> float:
    return module_cost * n_storage


def market_price(demand: np.ndarray) -> np.ndarray:
    """Compute the electricity market price according to demand."""
    peak = np.max(demand)
    return price_base + price_sensitivity * demand / peak


def electricity_supplied(
    reactor_production: np.ndarray, n_storage: int, dis: np.ndarray, ch: np.ndarray
) -> np.ndarray:
    """Compute the actual energy supplied to the grid."""
    max_storage_power = n_storage * module_power_mw
    eff_d = discharge_efficiency(dis, max_storage_power)
    eff_c = charge_efficiency(ch, max_storage_power)
    return reactor_production + dis * eff_d - ch / eff_c


def compute_charge_discharge(soc: np.ndarray) -> tuple[np.ndarray]:
    """Compute the charge and discharge powers according to state of charge."""
    charge = np.empty(soc.shape)
    for t in range(-1, horizon - 1):
        charge[t] = soc[t + 1] - soc[t] * (1 - storage_leakage_per_hour)
    return np.maximum(0.0, charge), np.maximum(0.0, -charge)


def var_from_x(x: np.ndarray) -> tuple:
    """Get the problem variables from the optimization vector x."""
    reactor_production = x[:horizon]
    soc = x[horizon : 2 * horizon]
    reactor_model = int(round(x[-3]))
    n_reactor = int(round(x[-2]))
    n_storage = int(round(x[-1]))
    return reactor_model, n_reactor, n_storage, reactor_production, soc


def x_from_var(
    reactor_model: int,
    n_reactor: int,
    n_storage: int,
    reactor_production: np.ndarray,
    soc: np.ndarray,
) -> np.ndarray:
    """Get the optimization vector x from the problem variables."""
    x = np.empty(2 * horizon + 3)
    x[:horizon] = reactor_production
    x[horizon : 2 * horizon] = soc
    x[-3] = reactor_model
    x[-2] = n_reactor
    x[-1] = n_storage
    return x


def objective(x: np.ndarray) -> float:
    """
    The objective function is defined as the annualized profit (EUR/year):
    market revenue - capex - fixed O&M - fuel cost.
    """
    reactor_model, n_reactor, n_storage, reactor_production, soc = var_from_x(x)
    reactor_capacity = reactor_models[reactor_model]

    # Capital
    cap_reactor = capital_cost_reactor(reactor_capacity, n_reactor)
    cap_storage = capital_cost_storage(n_storage)
    ann_capex = cap_reactor * reactor_annuity_factor
    ann_capex += cap_storage * storage_annuity_factor

    # Fixed O&M
    annual_fixed_om = cap_reactor * reactor_fixed_om_frac
    annual_fixed_om += cap_storage * storage_fixed_om_frac

    # Fuel cost
    fuel_cost = np.sum(reactor_production) * fuel_price * 365

    # Market interactions (daily)
    ch, dis = compute_charge_discharge(soc)
    supplied = electricity_supplied(reactor_production, n_storage, dis, ch)

    local_supply = np.minimum(supplied, electric_demand)
    unmet = np.maximum(0.0, electric_demand - supplied)
    surplus = np.maximum(0.0, supplied - electric_demand)

    price = market_price(electric_demand)

    # Price per MWh multiplied by average power (MW) in 1 hour
    net_market = price * local_supply
    # The unmet demand has to be bought on the market
    net_market -= price * unmet
    # The surplus has to be sold at a negative price
    net_market -= price_surplus * surplus

    annual_market = np.sum(net_market) * 365

    annual_profits = annual_market - ann_capex - annual_fixed_om - fuel_cost
    return annual_profits


def constraints_residuals(x: np.ndarray) -> list[float]:
    """
    Returns list of inequality constraints residuals which should be >= 0.
    """
    reactor_model, n_reactor, n_storage, reactor_production, soc = var_from_x(x)
    res = []

    # Reactor bounds
    plant_capacity = reactor_models[reactor_model] * n_reactor
    for t in range(horizon):
        # reactor_production <= reactor_capacity
        res.append(plant_capacity - reactor_production[t])
        # reactor_production >= 0
        res.append(reactor_production[t])

    # Storage bounds
    max_storage_energy = n_storage * module_capacity_mwh
    max_storage_power = n_storage * module_power_mw
    ch, dis = compute_charge_discharge(soc)
    for t in range(horizon):
        # charge and discharge <= max_storage_power
        res.append(max_storage_power - ch[t])
        res.append(max_storage_power - dis[t])
        # soc <= max_storage_energy
        res.append(max_storage_energy - soc[t])
        # soc >= 0
        res.append(soc[t])

    # Reactors count >=0
    res.append(n_reactor)
    # Storage modules count >=0
    res.append(n_storage)

    return res


# ------------------------- Example candidates -------------------------------------


def build_candidate_with_storage(
    reactor_model: int, n_reactor: int, n_storage: int, variability_smoothness: float
) -> np.ndarray:
    """
    Heuristic to build a candidate where:
      - reactor follows a smoothed version of demand
      - storage charges when reactor > demand and discharges when reactor < demand
    """
    demand = electric_demand.copy()
    avg = np.mean(demand)

    # Smooth reactor profile: weighted average between demand and a flat profile at avg
    reactor_production = (
        variability_smoothness * np.full(horizon, avg)
        + (1 - variability_smoothness) * demand
    )
    # Clip to capacity
    plant_capacity = reactor_models[reactor_model] * n_reactor
    reactor_production = np.clip(reactor_production, 0, plant_capacity)

    # Now compute storage actions to correct net supply toward demand
    soc = np.zeros(horizon)
    max_power = n_storage * module_power_mw
    max_energy = n_storage * module_capacity_mwh

    # Initialize SOC to 30% of capacity
    soc[0] = 0.30 * max_energy
    for t in range(horizon):
        # Compute mismatch
        mismatch = reactor_production[t] - demand[t]
        if mismatch > 0:
            # surplus energy available -> try to charge storage
            available_power = min(mismatch, max_power)
            remaining_energy_cap = max_energy - soc[t]
            charge_power = min(available_power, remaining_energy_cap)
            ch = charge_power
            dis = 0.0
        else:
            # deficit -> discharge storage
            need = -mismatch
            discharge_power = min(need, max_power)
            discharge_power = min(discharge_power, soc[t])
            dis = discharge_power
            ch = 0.0
        # Compute next SOC if not last step
        if t < horizon - 1:
            soc[t + 1] = soc[t] * (1 - storage_leakage_per_hour) + ch - dis

    x = x_from_var(
        reactor_model=reactor_model,
        n_reactor=n_reactor,
        n_storage=n_storage,
        reactor_production=reactor_production,
        soc=soc,
    )
    return x


def build_candidate_wo_storage(reactor_model: int, n_reactor: int) -> np.ndarray:
    """
    Builds a candidate where:
      - reactor follows exactly demand
      - storage is not used
    """
    plant_capacity = reactor_models[reactor_model] * n_reactor
    reactor_production = np.minimum(plant_capacity, electric_demand)
    soc = np.zeros(horizon)
    x = x_from_var(
        reactor_model=reactor_model,
        n_reactor=n_reactor,
        n_storage=0,
        reactor_production=reactor_production,
        soc=soc,
    )
    return x


# ------------------------- Run example and plot -----------------------------
def evaluate_candidate(cand, title):
    print(f"Evaluating candidate '{title}'")
    obj = objective(cand)
    res = constraints_residuals(cand)

    print(f"Objective (annual profit EUR): {obj:,}")
    print("Min inequality residual (>=0 for feasibility):", np.min(res))

    hours = np.arange(horizon)
    reactor_model, n_reactor, n_storage, reactor_production, soc = var_from_x(cand)
    ch, dis = compute_charge_discharge(soc)
    net_supply = electricity_supplied(reactor_production, n_storage, dis, ch)

    fig, ax = plt.subplots()
    ax.plot(hours, electric_demand, "k--", label="Electric demand")
    ax.plot(hours, reactor_production, "b-", label="Reactor output")
    ax.plot(hours, dis, "r-", label="Storage discharge")
    ax.plot(hours, ch, "g-", label="Storage charge")
    ax.plot(hours, net_supply, "m-", label="Net supply to grid")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Power (MW)")
    fig.suptitle(title)
    ax.set_title(
        "reactor_model = {} ; n_reactor = {} ; n_storage = {}".format(
            reactor_model, n_reactor, n_storage
        ),
        fontsize="medium",
    )
    ax.legend(
        loc="upper left",
        fontsize="small",
        fancybox=False,
        framealpha=1.0,
        borderaxespad=0,
        edgecolor="black",
    )
    ax.grid(True, ls=":", color="black")
    ax.set_xlim((hours[0], hours[-1]))


if __name__ == "__main__":
    plt.rcParams["font.family"] = "serif"
    cand = build_candidate_with_storage(
        reactor_model=2, n_reactor=1, n_storage=12, variability_smoothness=0.6
    )
    evaluate_candidate(cand, "Smoothed reactor, storage used to track demand")
    cand = build_candidate_wo_storage(reactor_model=3, n_reactor=1)
    evaluate_candidate(cand, "No storage, reactor follows demand")
    plt.show()
