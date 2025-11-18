# Adding `differential_evolution`

Here, we try the module `scipy.optimize.differential_evolution`. We added the solver, defined `bounds` for all variables, and marked the integer variables.

**The Problem: The "Cold Start"**

It failed immediately (`Convergence: 0.0000`). The solver's random initial population was **massively infeasible**. A random `n_reactor=1` (80 MW capacity) was paired with a random `reactor_production=[500, 600, ...]` (violating the bound). The entire population was infeasible, giving the solver no path to a solution.

---

## 2. The Second Attempt: The "Warm Start"

**The Idea:** Use the script's *own heuristics* (`build_candidate...`) to create a feasible initial population.

**The Implementation:** We generated hundreds of random *integer* configurations, used the heuristics to build complete, feasible solution vectors, and passed this to the solver's `init` parameter.

**The Problem: Stuck on the Heuristic**

This worked... partially. The solver immediately found a good solution (`Profit: 86,626,999.27`), but then it got stuck. For 500 iterations, the profit never improved, and convergence remained `0.0000`.

**The Diagnosis:** We hit a "razor-thin manifold" problem. Our decision variables were `x = [production, soc, ints]`. The `soc` (State of Charge) variables are not independent; they are rigidly linked in time (`soc[t]` depends on `soc[t-1]`). The optimizer's random mutations (e.g., changing `soc[5]` and `soc[6]` independently) would *instantly* break this physical link, making every new candidate infeasible.

---

## 3. Refactor 1: Optimizing Decisions, Not State

**The Idea:** The `soc` is a *result*, not a *decision*. The real decisions are `charge` and `discharge`. We changed the vector to `x = [production, charge, discharge, ints]`.

**The Implementation:** This was a major refactor.
1.  We created a `get_soc_solver_matrix` to solve the 24x24 linear system for the periodic `soc`.
2.  We wrote a new `compute_soc_from_actions` function: `soc = A_inv @ (ch - dis)`.
3.  All functions (`var_from_x`, `constraints_residuals`, heuristics) were updated.

**The Problem: Still Stuck**

The result was identical. Still stuck at `86,626,999.27`. The "razor-thin manifold" was still there. A tiny mutation to `ch[5]` would propagate through the `A_inv` matrix, minutely changing *every* `soc[t]`. If *any* `soc[t]` was already at its boundary (e.g., `soc[10] = 0.0`), this mutation would make it `soc[10] = -0.002`, again making the candidate infeasible.

---

## 4. Refactor 2: A More Natural Decision

**The Idea:** `ch` and `dis` are unnatural. A single variable, `net_charge`, is better.
* `net_charge > 0` = charging
* `net_charge < 0` = discharging

This *guarantees* you can't charge and discharge at the same time.

**The Implementation:** We refactored *again*. The vector became `x = [production, net_charge, ints]`. The search space was now a simple `[-P_max, +P_max]`.

**The Final Problem: Still Stuck**

...And it *still* failed. The exact same behavior. Stuck on the heuristic, `Convergence: 0.0000`.

---

## Conclusion: The Unbeatable Manifold

The problem isn't the script; it's the fundamental conflict between this **problem's *shape* and the solver's *strategy***.

Our "warm start" population is on the *surface* of a complex, high-dimensional feasible region (a polytope). `differential_evolution` works by adding *random vectors* to mutate solutions. When you're on the surface of a shape, *any* random vector will almost certainly point *outside* the shape, creating an infeasible solution.

The solver is unable to "slide" along the surface to find a better optimum. It is fundamentally the wrong tool for this kind of tightly-constrained problem.