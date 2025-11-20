import numpy as np
import itertools
import common


def run_solver(settings: common.SimSettings):
    # 1. Hardware Space based on settings
    N_MODELS = len(common.REACTOR_MODELS)

    # 2. Operational Space
    # Coarse grid for Brute Force (Speed over precision)
    SMOOTHNESS_PARAMS = np.linspace(0.0, 1.0, 10)  # Reduced for brute demo
    BIAS_PARAMS = [0.9, 1.0, 1.1]

    # Handling Assumptions: Fixed vs Optimized SOC
    if settings.fixed_initial_soc is not None:
        INIT_SOC_PARAMS = [settings.fixed_initial_soc]
    else:
        INIT_SOC_PARAMS = [0.0, 0.3, 0.5, 0.8]  # Search optimized start

    best_profit = -np.inf
    best_res = {'profit': -np.inf, 'config': (0, 0, 0), 'x': np.zeros(common.HORIZON * 4)}

    def build_candidate(r_idx, n_r, n_s, smooth, bias, init_s):
        demand = common.ELECTRIC_DEMAND
        avg = np.mean(demand) * bias
        r_prod = (smooth * np.full(common.HORIZON, avg) + (1 - smooth) * demand)
        cap = common.REACTOR_MODELS[r_idx] * n_r
        r_prod = np.clip(r_prod, 0, cap)

        soc = np.zeros(common.HORIZON)
        ch = np.zeros(common.HORIZON)
        dis = np.zeros(common.HORIZON)
        max_e = n_s * common.MODULE_CAP_MWH
        max_p = n_s * common.MODULE_POWER_MW

        curr_soc = init_s * max_e  # Start assumption

        for t in range(common.HORIZON):
            mismatch = r_prod[t] - demand[t]
            c_flow = 0.0
            d_flow = 0.0
            if mismatch > 0:
                c_flow = min(mismatch, max_p)
                c_flow = min(c_flow, max(0, max_e - curr_soc))
                ch[t] = c_flow
            else:
                d_flow = min(-mismatch, max_p)
                d_flow = min(d_flow, curr_soc)
                dis[t] = d_flow
            curr_soc = curr_soc * (1 - common.STORAGE_LEAKAGE) + c_flow - d_flow
            soc[t] = curr_soc
        return r_prod, ch, dis, soc

    op_combinations = list(itertools.product(SMOOTHNESS_PARAMS, BIAS_PARAMS, INIT_SOC_PARAMS))

    # Use the step from settings (e.g., step=5 for coarse brute force)
    storage_range = range(0, settings.max_storage + 1, settings.storage_step)

    for m in range(N_MODELS):
        for r in range(settings.max_reactors + 1):
            for s in storage_range:
                if r == 0 and s == 0: continue

                fixed_cost = common.calculate_annual_fixed_cost(m, r, s) / 365.0
                local_best_daily = -np.inf
                local_best_vecs = None

                if s == 0:
                    vecs = build_candidate(m, r, s, 0.0, 1.0, 0.0)
                    prof = common.calculate_daily_profit(vecs[0], vecs[1], vecs[2], s)
                    local_best_daily = prof
                    local_best_vecs = vecs
                else:
                    for smooth, bias, init_soc in op_combinations:
                        vecs = build_candidate(m, r, s, smooth, bias, init_soc)
                        prof = common.calculate_daily_profit(vecs[0], vecs[1], vecs[2], s)
                        if prof > local_best_daily:
                            local_best_daily = prof
                            local_best_vecs = vecs

                annual = (local_best_daily - fixed_cost) * 365.0
                if annual > best_profit:
                    best_profit = annual
                    x_final = np.concatenate(local_best_vecs)
                    best_res = {'profit': annual, 'config': (m, r, s), 'x': x_final}

    return best_res