"""
SMR and Battery Storage Optimization - "Ultra-Heavy Compute" Edition

Architecture:
1.  Phase 0: Coarse Grid Seeding (Scans the landscape to seed the GA).
2.  Phase 1: Genetic Algorithm with Adaptive Mutation & Elitism.
3.  Inner Loop: Multi-Start SLSQP (Runs solver twice per config to ensure dispatch optimality).
4.  Phase 2: Deep Neighborhood Polish (Wide-radius search on top candidates).
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import concurrent.futures
import time
import random
import itertools

random.seed(2025)
np.random.seed(2025)

# Search Space Limits
MAX_REACTORS = 5       # Increased search range
MAX_STORAGE = 20       # Increased search range
N_MODELS = 5            # Indices 0-4

# "Spend More Compute" Parameters
POPULATION_SIZE = 200       # Larger gene pool
GENERATIONS = 60            # Longer evolution
GRID_SEED_DENSITY = 25      # How many points to scan per dimension in Seeding
MULTI_START_SOLVER = True   # If True, runs SLSQP twice (smart guess + flat guess)

# Economics
horizon = 24
electric_demand = np.array(
    [160, 152, 144, 140, 144, 160, 200, 240, 280, 300, 320, 340, 360, 368, 352,
     340, 320, 312, 300, 280, 260, 240, 220, 192], dtype=np.float64,
)

reactor_models = [80.0, 160.0, 300.0, 350.0, 470.0]
interest_rate = 0.04
reactor_cap_a = 2.0e7
reactor_cap_b = 0.8
reactor_years = 60
# Annuity factors
r_annuity = interest_rate / (1 - (1 + interest_rate) ** (-reactor_years))

module_capacity_mwh = 50.0
module_power_mw = 10.0
module_cost = 1.0e7
storage_leakage = 0.0008
storage_years = 20
s_annuity = interest_rate / (1 - (1 + interest_rate) ** (-storage_years))

# O&M and Fuel
r_om_frac = 0.03
s_om_frac = 0.02
fuel_price = 5.0
price_base = 70.0
price_sensitivity = 60.0
price_surplus = 10.0

# ------------------------- Physics & Economics Helpers ----------------------

def get_market_prices(demand):
    peak = np.max(demand)
    return price_base + price_sensitivity * demand / peak

MARKET_PRICES = get_market_prices(electric_demand)

def charge_efficiency(charge_mw, max_charge_mw):
    # Non-linear efficiency curve: Efficiency drops as power approaches max
    if max_charge_mw < 1e-6: return 1.0
    safe_max = np.maximum(max_charge_mw, 1e-6)
    # Vectorized
    ratio = charge_mw / safe_max
    # Curve: 0.95 base, drops by 0.15 at full load
    r = 0.95 - 0.15 * (ratio ** 2)
    return np.maximum(r, 0.6)

def discharge_efficiency(dis_mw, max_dis_mw):
    if max_dis_mw < 1e-6: return 1.0
    safe_max = np.maximum(max_dis_mw, 1e-6)
    ratio = dis_mw / safe_max
    # Curve: 0.96 base, drops by 0.2 at full load
    r = 0.96 - 0.2 * (ratio ** 2)
    return np.maximum(r, 0.55)

def calculate_capex_annual(r_idx, n_r, n_s):
    if n_r == 0 and n_s == 0: return 0.0

    c_reac = 0.0
    if n_r > 0:
        cap_mw = reactor_models[r_idx]
        # Scale factor economy
        c_reac = n_r * (reactor_cap_a * cap_mw ** reactor_cap_b)

    c_stor = n_s * module_cost

    ann_capex = c_reac * r_annuity + c_stor * s_annuity
    ann_om = c_reac * r_om_frac + c_stor * s_om_frac
    return ann_capex + ann_om

# ------------------------- Inner Loop: The Solver ---------------------------

def solve_schedule_slsqp(reactor_mw, n_storage, initial_guess_mode='heuristic'):
    """
    Solves the dispatch problem.
    initial_guess_mode: 'heuristic' (tries to match demand) or 'flat' (zeros)
    """
    max_store_p = n_storage * module_power_mw
    max_store_e = n_storage * module_capacity_mwh

    # 1. Setup Bounds
    # x = [Prod(0..23), Charge(0..23), Disch(0..23), SOC(0..23)]
    bounds = []
    bounds += [(0.0, reactor_mw)] * horizon       # Prod
    bounds += [(0.0, max_store_p)] * horizon      # Charge
    bounds += [(0.0, max_store_p)] * horizon      # Discharge
    bounds += [(0.0, max_store_e)] * horizon      # SOC

    # 2. Initial Guess (Crucial for avoiding local optima in non-linear problems)
    x0 = np.zeros(4 * horizon)

    if initial_guess_mode == 'heuristic':
        # Guess: Reactor matches demand up to capacity
        x0[:horizon] = np.minimum(reactor_mw, electric_demand)
        # Guess: Battery starts half full
        if max_store_e > 0:
            x0[3*horizon:] = 0.5 * max_store_e
    else:
        # Flat guess: 80% reactor, 50% SOC
        x0[:horizon] = reactor_mw * 0.8
        if max_store_e > 0:
            x0[3*horizon:] = 0.5 * max_store_e

    # 3. Objective Function
    def obj_func(x):
        r_prod = x[:horizon]
        ch = x[horizon:2*horizon]
        dis = x[2*horizon:3*horizon]

        eff_d = discharge_efficiency(dis, max_store_p)
        eff_c = charge_efficiency(ch, max_store_p)

        # Net Supply to Grid
        supplied = r_prod + dis * eff_d - ch / eff_c

        # Market Calcs
        local_supply = np.minimum(supplied, electric_demand)
        unmet = np.maximum(0.0, electric_demand - supplied)
        surplus = np.maximum(0.0, supplied - electric_demand)

        revenue = np.sum(MARKET_PRICES * local_supply)
        penalty = np.sum(MARKET_PRICES * unmet * 1.5) # 1.5x penalty for reliability
        surplus_rev = np.sum(price_surplus * surplus)
        fuel_cost = np.sum(r_prod) * fuel_price

        daily_profit = revenue + surplus_rev - penalty - fuel_cost
        return -daily_profit # Minimize negative profit

    # 4. Constraints
    cons = []
    if n_storage > 0:
        def soc_dynamics(x):
            soc = x[3*horizon:]
            ch = x[horizon:2*horizon]
            dis = x[2*horizon:3*horizon]

            eff_c = charge_efficiency(ch, max_store_p)
            eff_d = discharge_efficiency(dis, max_store_p)

            residuals = []
            # Start of day assumption: SOC[0] depends on SOC[-1] (Wrap around 24h cycle)
            prev_soc = soc[-1]

            for t in range(horizon):
                net_flow = ch[t]*eff_c[t] - dis[t]/eff_d[t]
                # Leakage happens on the stored amount
                expected = prev_soc * (1 - storage_leakage) + net_flow
                residuals.append(soc[t] - expected)
                prev_soc = soc[t]
            return np.array(residuals)

        cons.append({'type': 'eq', 'fun': soc_dynamics})

    # 5. Run Solver
    res = minimize(obj_func, x0, method='SLSQP', bounds=bounds, constraints=cons,
                   tol=1e-5, options={'maxiter': 200})

    return -res.fun, res.x, res.success

# ------------------------- Master Evaluation --------------------------------

def evaluate_configuration(params):
    """
    Evaluates a specific hardware configuration.
    Includes Multi-Start logic: runs solver twice if needed to find best dispatch.
    """
    r_mod_idx, n_r, n_s = params

    # Basic Sanity
    if n_r < 0 or n_s < 0: return (-1e15, params, None)
    if n_r == 0 and n_s == 0: return (0.0, params, None)

    # 1. Calculate Fixed Costs
    annual_fixed = calculate_capex_annual(r_mod_idx, n_r, n_s)
    daily_fixed = annual_fixed / 365.0

    reactor_mw = reactor_models[r_mod_idx] * n_r

    # 2. Operational Optimization (Multi-Start)
    # Run Heuristic
    profit_h, x_h, suc_h = solve_schedule_slsqp(reactor_mw, n_s, 'heuristic')

    best_op_profit = profit_h
    best_x = x_h
    best_suc = suc_h

    # Run Flat (only if heavy compute enabled and we have storage to manage)
    if MULTI_START_SOLVER and n_s > 0:
        profit_f, x_f, suc_f = solve_schedule_slsqp(reactor_mw, n_s, 'flat')
        if suc_f and profit_f > best_op_profit:
            best_op_profit = profit_f
            best_x = x_f
            best_suc = True

    if not best_suc:
        return (-1e12, params, None) # Penalize failure heavily

    total_annual_profit = (best_op_profit - daily_fixed) * 365.0
    return (total_annual_profit, params, best_x)

# ------------------------- Genetic Algorithm Components ---------------------

def create_random_ind():
    return [
        random.randint(0, N_MODELS - 1),
        random.randint(0, MAX_REACTORS),
        random.randint(0, MAX_STORAGE)
    ]

def mutate(ind, rate):
    if random.random() < rate:
        mut_type = random.randint(0, 3)
        if mut_type == 0: # Model Swap
            ind[0] = random.randint(0, N_MODELS - 1)
        elif mut_type == 1: # Reactor Small Adjust
            ind[1] = max(0, min(MAX_REACTORS, ind[1] + random.choice([-1, 1])))
        elif mut_type == 2: # Storage Slide
            change = int(random.gauss(0, 20))
            ind[2] = max(0, min(MAX_STORAGE, ind[2] + change))
        elif mut_type == 3: # Storage Jump (explore new areas)
            ind[2] = random.randint(0, MAX_STORAGE)
    return ind

def crossover(p1, p2):
    # Uniform Crossover
    c1, c2 = list(p1), list(p2)
    for i in range(3):
        if random.random() < 0.5:
            c1[i], c2[i] = c2[i], c1[i]
    return c1, c2

# ------------------------- Phase 0: Coarse Grid Seeding ---------------------

def perform_grid_seeding():
    print(f"--- Phase 0: Coarse Grid Seeding ---")
    print("Scanning landscape to identify high-potential valleys...")

    # Create a coarse grid
    seeds = []
    # Check every model
    for m in range(N_MODELS):
        # Check reactors in steps of 2 or 3
        for r in range(1, MAX_REACTORS + 1, max(1, MAX_REACTORS // 5)):
            # Check storage in coarse steps
            for s in range(0, MAX_STORAGE + 1, 50):
                seeds.append([m, r, s])

    print(f"Evaluating {len(seeds)} seed points...")

    results = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Map returns generator, convert to list
        mapped = executor.map(evaluate_configuration, seeds)
        for res in mapped:
            results.append(res)

    # Sort by profit descending
    results.sort(key=lambda x: x[0], reverse=True)

    # Return top N configurations to inject into GA
    top_seeds = [list(x[1]) for x in results[:POPULATION_SIZE // 2]]
    print(f"Seeding complete. Top seed profit: EUR {results[0][0]:,.0f}")
    return top_seeds, results # Return results for plotting history

# ------------------------- Phase 2: Deep Polish -----------------------------

def perform_deep_polish(candidates):
    print(f"\n--- Phase 2: Deep Neighborhood Polish ---")
    best_global = -np.inf
    best_cfg = None
    best_sol = None

    # Deduplicate candidates based on signature (Model, N_Reactors)
    # We want to polish distinct hardware setups
    unique_sigs = set()
    to_polish = []

    for profit, cfg, sol in candidates:
        sig = (cfg[0], cfg[1])
        if sig not in unique_sigs:
            unique_sigs.add(sig)
            to_polish.append(cfg)
        if len(to_polish) >= 6: break # Polish top 6 distinct hardware setups

    search_space = set()

    for (r_idx, n_r, n_s) in to_polish:
        # Wide radius for storage, narrow for reactors
        for dr in [-1, 0, 1]:
            for ds in range(-15, 16, 5): # Check -15, -10, -5, 0, 5, 10, 15
                nr_new = max(0, min(MAX_REACTORS, n_r + dr))
                ns_new = max(0, min(MAX_STORAGE, n_s + ds))
                search_space.add((r_idx, nr_new, ns_new))

    print(f"Polishing {len(search_space)} specific configurations...")

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(evaluate_configuration, list(search_space)))

    for p, c, s in results:
        if p > best_global:
            best_global = p
            best_cfg = c
            best_sol = s

    return best_global, best_cfg, best_sol

# ------------------------- Main Optimization Flow ---------------------------

def run_optimization():
    start_time = time.time()

    all_history_points = [] # For visualization (profit, r, s)

    # 1. Grid Seeding
    seed_configs, seed_results = perform_grid_seeding()
    for p, c, _ in seed_results:
        # Store normalized reactor capacity (approx) for plotting
        mw = reactor_models[c[0]] * c[1]
        all_history_points.append((p, mw, c[2]))

    # 2. Initialize Population
    population = seed_configs # Start with half population from seeds
    while len(population) < POPULATION_SIZE:
        population.append(create_random_ind())

    memoization = {} # Cache results to avoid re-solving same configs

    # Pre-fill memoization with seed results
    for p, c, s in seed_results:
        memoization[tuple(c)] = (p, s)

    print(f"\n--- Phase 1: Genetic Algorithm ({GENERATIONS} Gens) ---")

    for gen in range(GENERATIONS):
        # Evaluation
        to_eval = []
        for ind in population:
            if tuple(ind) not in memoization:
                to_eval.append(ind)

        if to_eval:
            with concurrent.futures.ProcessPoolExecutor() as executor:
                results = list(executor.map(evaluate_configuration, to_eval))
            for p, c, s in results:
                memoization[tuple(c)] = (p, s)
                mw = reactor_models[c[0]] * c[1]
                all_history_points.append((p, mw, c[2]))

        # Ranking
        pop_fitness = []
        for ind in population:
            fit, sol = memoization[tuple(ind)]
            pop_fitness.append((fit, ind, sol))

        pop_fitness.sort(key=lambda x: x[0], reverse=True)

        best_gen_fit = pop_fitness[0][0]
        best_gen_cfg = pop_fitness[0][1]

        # Dynamic Mutation Rate
        mut_rate = 0.15 if gen < GENERATIONS * 0.7 else 0.05 # Cool down later

        if gen % 5 == 0 or gen == GENERATIONS - 1:
            print(f"Gen {gen+1}/{GENERATIONS} | Best: EUR {best_gen_fit:,.0f} | "
                  f"Cfg: {best_gen_cfg}")

        # Selection (Elitism + Tournament)
        next_gen = [x[1] for x in pop_fitness[:5]] # Elitism

        # Breed
        while len(next_gen) < POPULATION_SIZE:
            parents = random.sample(pop_fitness[:50], 2) # Selection from top 50
            c1, c2 = crossover(parents[0][1], parents[1][1])
            next_gen.append(mutate(c1, mut_rate))
            if len(next_gen) < POPULATION_SIZE:
                next_gen.append(mutate(c2, mut_rate))

        population = next_gen

    # 3. Deep Polish
    final_candidates = pop_fitness[:20] # Pass top 20 raw results to polish
    best_profit, best_config, best_sol = perform_deep_polish(final_candidates)

    duration = time.time() - start_time
    print("==========================================================")
    print(f" OPTIMIZATION COMPLETE in {duration:.1f} seconds")
    print(f" Total Simulations: {len(memoization)} unique configs")
    print(f" BEST PROFIT: EUR {best_profit:,.2f}")
    print(f" Config: Model {reactor_models[best_config[0]]}MW")
    print(f"         {best_config[1]} Reactors")
    print(f"         {best_config[2]} Storage Modules")
    print("==========================================================")

    return best_profit, best_config, best_sol, all_history_points

# ------------------------- Visualization ------------------------------------

def plot_dashboard(profit, config, x, history):
    r_idx, n_r, n_s = config
    r_prod = x[:horizon]
    ch = x[horizon:2*horizon]
    dis = x[2*horizon:3*horizon]
    soc = x[3*horizon:]

    mw_capacity = reactor_models[r_idx] * n_r

    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 2)

    # Plot 1: Operational Schedule
    ax1 = fig.add_subplot(gs[0, :])
    hours = np.arange(horizon)
    eff_d = discharge_efficiency(dis, n_s * module_power_mw)
    eff_c = charge_efficiency(ch, n_s * module_power_mw)
    net = r_prod + dis*eff_d - ch/eff_c

    ax1.plot(hours, electric_demand, 'k--', label='Demand', lw=2)
    ax1.plot(hours, r_prod, 'b-', label='Reactor Gen', lw=2, alpha=0.8)
    ax1.plot(hours, net, 'g.-', label='Net Supply', lw=1.5)
    ax1.fill_between(hours, 0, r_prod, color='blue', alpha=0.1)

    # Stacked bars for storage
    ax1.bar(hours, dis, color='red', alpha=0.3, label='Discharge')
    ax1.bar(hours, -ch, color='green', alpha=0.3, label='Charge')

    ax1.set_title("Best Operational Schedule (24h)", fontsize=12, fontweight='bold')
    ax1.set_ylabel("Power (MW)")
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # Twin axis for SOC
    ax1b = ax1.twinx()
    ax1b.plot(hours, soc, 'c:', label='SOC', lw=2)
    ax1b.set_ylabel("Energy (MWh)", color='c')

    # Plot 2: Search Landscape (Heatmap style scatter)
    ax2 = fig.add_subplot(gs[1, 0])

    # Unpack history
    profs = [h[0] for h in history if h[0] > 0] # Filter out failures
    reacts = [h[1] for h in history if h[0] > 0]
    stors = [h[2] for h in history if h[0] > 0]

    sc = ax2.scatter(reacts, stors, c=profs, cmap='viridis', s=20, alpha=0.6)
    ax2.set_xlabel("Total Reactor Capacity (MW)")
    ax2.set_ylabel("Storage Modules")
    ax2.set_title("Search Landscape Exploration")
    plt.colorbar(sc, ax=ax2, label='Annual Profit')

    # Highlight Best
    ax2.scatter([mw_capacity], [n_s], color='red', s=150, marker='*', edgecolors='black', label='Best')
    ax2.legend()

    # Plot 3: Economics Breakdown
    ax3 = fig.add_subplot(gs[1, 1])
    capex_ann = calculate_capex_annual(r_idx, n_r, n_s)
    rev = profit + capex_ann # Approx gross revenue

    labels = ['Net Profit', 'Annualized CAPEX + O&M']
    sizes = [profit, capex_ann]
    colors = ['#4CAF50', '#FF9800']
    ax3.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
    ax3.set_title("Annual Financial Breakdown")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Run
    best_profit, best_config, best_sol, history = run_optimization()
    # Visualize
    plot_dashboard(best_profit, best_config, best_sol, history)