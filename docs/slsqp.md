# Decomposition: Grid Search + Gradient-Based NLP (SLSQP)

Standard stochastic solvers (like Genetic Algorithms) failed because the problem contains a "Thin Manifold": the strict equality constraint linking one hour’s State of Charge (SOC) to the next ($SOC_{t+1} = SOC_t + \dots$) means 99.9% of random mutations are mathematically infeasible.

Our strategy overcomes this by decomposing the problem into two nested layers:
1.  **Outer Loop (Hardware - Integer):** We perform a brute-force Grid Search over the discrete variables (`reactor_model`, `n_reactor`, `n_storage`). Since the search space is manageable (~1,500 combinations), we can iterate through them deterministically.
2.  **Inner Loop (Operations - Continuous):** For each fixed hardware configuration, the problem becomes a continuous Non-Linear Programming (NLP) task. We solve this using **SLSQP (Sequential Least Squares Programming)**.
3.  **Gradient-Based Feasibility:** Unlike random solvers, SLSQP calculates gradients. It "sees" the constraints and mathematically slides the variables along the edge of the feasible region to maximize profit without breaking the physics.
4.  **Parallel Execution:** Since every hardware configuration is independent, we utilize `concurrent.futures` to map the grid search across all available CPU cores.

## Log of Decisions

1.  **Abandon Stochastic Solvers:** We recognized that `differential_evolution` could not handle the time-coupled SOC constraints. Random guesses invariably broke the link between $Time_t$ and $Time_{t+1}$, resulting in zero convergence.

2.  **Decouple the Variables:** We split the problem. The integer variables (counts) are solved via iteration, while the continuous variables (power flows) are solved via optimization. This removes the "mixed-integer" complexity, leaving two simpler problems.

3.  **Switch to SLSQP:** We selected the SLSQP algorithm for the inner loop. It is designed specifically for bounded problems with equality constraints. It allows the solver to temporarily violate constraints while calculating the "direction" of improvement, eventually converging on a mathematically perfect solution.

4.  **Redefine Optimization Variables:** Initially, we tried to calculate SOC as a *result* of power flow. This was unstable for gradients. We decided to include `SOC` as an **explicit decision variable** in the vector ($x = [Prod, Charge, Discharge, SOC]$). We then used equality constraints to force the solver to align the `SOC` variables with the physics. This gave the solver the necessary mathematical "slack" to find the optimum.

5.  **Optimization Scope (Daily vs. Annual):** To keep the inner solver numerically stable, we optimized for **Daily Operational Profit**. We effectively ignored the "365" multiplier inside the gradient calculations and only applied the annualized CAPEX and scaling factors in the final aggregation step.