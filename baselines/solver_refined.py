import numpy as np
import itertools
import common

def run_solver(settings: common.SimSettings):
    # --- 1. ESTABLISH SEARCH SPACE ---
    # Refined Grid uses denser operational parameters
    SMOOTHNESS = np.linspace(0.0, 1.0, 25)
    BIAS = [0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2]

    if settings.fixed_initial_soc is not None:
        INIT_SOC = [settings.fixed_initial_soc]
    else:
        INIT_SOC = [0.0, 0.3, 0.5, 0.8]

    # Refined ignores settings.storage_step and uses step=1
    r_range = list(range(settings.max_reactors + 1))
    s_range = list(range(0, settings.max_storage + 1, 1))

    # --- 2. ESTIMATE & PRINT SPACE ---
    n_hardware = len(common.REACTOR_MODELS) * len(r_range) * len(s_range)
    n_ops = len(SMOOTHNESS) * len(BIAS) * len(INIT_SOC)
    total_evals = n_hardware * n_ops

    print(f"    [Space Analysis] Hardware: {n_hardware} combos (Step=1) | Ops per Config: {n_ops}")
    print(f"    [Space Analysis] Total Grid Points: {total_evals:,.0f}")

    # --- 3. EXECUTION ---
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

        curr_soc = init_s * max_e
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

    op_grid = list(itertools.product(SMOOTHNESS, BIAS, INIT_SOC))

    for m in range(len(common.REACTOR_MODELS)):
        for r in r_range:
            for s in s_range:
                if r == 0 and s == 0: continue
                fixed_cost = common.calculate_annual_fixed_cost(m, r, s) / 365.0
                local_best = -np.inf
                local_vecs = None

                if s == 0:
                    vecs = build_candidate(m, r, s, 0.0, 1.0, 0.0)
                    prof = common.calculate_daily_profit(vecs[0], vecs[1], vecs[2], s)
                    local_best = prof
                    local_vecs = vecs
                else:
                    for (sm, bi, iso) in op_grid:
                        vecs = build_candidate(m, r, s, sm, bi, iso)
                        prof = common.calculate_daily_profit(vecs[0], vecs[1], vecs[2], s)
                        if prof > local_best:
                            local_best = prof
                            local_vecs = vecs

                annual = (local_best - fixed_cost) * 365.0
                if annual > best_profit:
                    best_profit = annual
                    x_final = np.concatenate(local_vecs)
                    best_res = {'profit': annual, 'config': (m, r, s), 'x': x_final}

    return best_res