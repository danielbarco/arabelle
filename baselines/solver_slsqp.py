import numpy as np
import concurrent.futures
from scipy.optimize import minimize
import common


def solve_operational_slsqp(args):
    # UNPACK 4 ITEMS NOW (includes settings)
    r_idx, n_r, n_s, settings = args

    if n_r == 0 and n_s == 0: return (-1e9, None)

    cap_mw = common.REACTOR_MODELS[r_idx] * n_r
    max_p = n_s * common.MODULE_POWER_MW
    max_e = n_s * common.MODULE_CAP_MWH

    H = common.HORIZON
    x0 = np.zeros(4 * H)
    x0[:H] = np.minimum(cap_mw, common.ELECTRIC_DEMAND)

    start_seed_soc = 0.5
    if settings.fixed_initial_soc is not None:
        start_seed_soc = settings.fixed_initial_soc
    if max_e > 0: x0[3 * H:] = start_seed_soc * max_e

    bounds = []
    bounds += [(0, cap_mw)] * H
    bounds += [(0, max_p)] * H
    bounds += [(0, max_p)] * H
    bounds += [(0, max_e)] * H

    def obj_func(x):
        r_prod = x[:H]
        ch = x[H:2 * H]
        dis = x[2 * H:3 * H]
        profit = common.calculate_daily_profit(r_prod, ch, dis, n_s)
        return -profit

    cons = []
    if n_s > 0:
        def dynamics(x):
            soc = x[3 * H:]
            ch = x[H:2 * H]
            dis = x[2 * H:3 * H]
            eff_c = common.charge_efficiency(ch, max_p)
            eff_d = common.discharge_efficiency(dis, max_p)

            res = []
            prev = soc[-1]

            for t in range(H):
                flow = ch[t] * eff_c[t] - dis[t] / eff_d[t]
                expected = prev * (1 - common.STORAGE_LEAKAGE) + flow
                res.append(soc[t] - expected)
                prev = soc[t]
            return np.array(res)

        cons.append({'type': 'eq', 'fun': dynamics})

        if settings.fixed_initial_soc is not None:
            def fixed_start(x):
                current_soc_0 = x[3 * H]
                target = settings.fixed_initial_soc * max_e
                return current_soc_0 - target

            cons.append({'type': 'eq', 'fun': fixed_start})

    res = minimize(obj_func, x0, method='SLSQP', bounds=bounds, constraints=cons, tol=1e-2)
    fixed_cost = common.calculate_annual_fixed_cost(r_idx, n_r, n_s)
    annual_profit = (-res.fun * 365.0) - fixed_cost

    return (annual_profit, res.x)


def run_solver(settings: common.SimSettings):
    # --- 1. ESTABLISH SEARCH SPACE ---
    storage_range = list(range(0, settings.max_storage + 1, settings.storage_step))
    r_range = list(range(settings.max_reactors + 1))

    # --- 2. ESTIMATE & PRINT SPACE ---
    n_hardware = len(common.REACTOR_MODELS) * len(r_range) * len(storage_range)

    print(f"    [Space Analysis] Hardware: {n_hardware} combos | Ops: Gradient Descent (Continuous)")
    print(f"    [Space Analysis] Total Optimizations: {n_hardware}")

    # --- 3. EXECUTION ---
    candidates = []
    for m in range(len(common.REACTOR_MODELS)):
        for r in r_range:
            for s in storage_range:
                candidates.append((m, r, s, settings))

    best_profit = -np.inf
    best_res = {'profit': -np.inf, 'config': (0, 0, 0), 'x': np.zeros(common.HORIZON * 4)}

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(solve_operational_slsqp, candidates))

    for i, (prof, x) in enumerate(results):
        if prof > best_profit:
            clean_config = candidates[i][:3]
            best_profit = prof
            best_res = {'profit': prof, 'config': clean_config, 'x': x}

    return best_res