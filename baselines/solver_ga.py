import numpy as np
import random
import concurrent.futures
import common
from solver_slsqp import solve_operational_slsqp

random.seed(2025)
np.random.seed(2025)


def run_solver(settings: common.SimSettings):
    POP_SIZE = 28
    GENS = 5

    def create_ind():
        return (random.randint(0, len(common.REACTOR_MODELS) - 1),
                random.randint(0, settings.max_reactors),
                random.randint(0, settings.max_storage))

    # Population stores just (m, r, s)
    population = [create_ind() for _ in range(POP_SIZE)]
    best_profit = -np.inf
    best_res = {'profit': -np.inf, 'config': (0, 0, 0), 'x': np.zeros(common.HORIZON * 4)}

    for g in range(GENS):
        # Prepare arguments for worker: (m, r, s, settings)
        eval_args = [(ind[0], ind[1], ind[2], settings) for ind in population]

        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = list(executor.map(solve_operational_slsqp, eval_args))

        pop_fit = []
        for i, (prof, x) in enumerate(results):
            pop_fit.append((prof, population[i], x))
            if prof > best_profit:
                best_profit = prof
                best_res = {'profit': prof, 'config': population[i], 'x': x}

        # Sort by profit
        pop_fit.sort(key=lambda x: x[0], reverse=True)

        # Next Gen
        next_gen = [x[1] for x in pop_fit[:5]]  # Elitism

        while len(next_gen) < POP_SIZE:
            parent = random.choice(pop_fit[:10])[1]
            m, r, s = parent
            # Mutate
            if random.random() < 0.3: m = random.randint(0, len(common.REACTOR_MODELS) - 1)
            if random.random() < 0.3: r = max(0, min(settings.max_reactors, r + random.choice([-1, 1])))
            if random.random() < 0.3: s = max(0, min(settings.max_storage, s + int(random.gauss(0, 5))))
            next_gen.append((m, r, s))

        population = next_gen

    return best_res

if __name__ == "__main__":
    settings = common.SimSettings(fixed_initial_soc=None, storage_step=5)
    result = run_solver(settings)
    print(f"Best Profit: {result['profit']}, Config: {result['config']}")