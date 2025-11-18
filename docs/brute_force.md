# Brute-Force Heuristic Grid Search

A pure brute-force is impossible due to the 48 continuous variables. Our strategy is to:
1.  **Brute-Force Integers:** Iterate through every possible combination of the integer variables (`reactor_model`, `n_reactor`, `n_storage`) within a large range.
2.  **Grid-Search Heuristics:** For each integer combination, we use the script's built-in heuristics (`build_candidate_...`) to generate a feasible schedule.
3.  **Search Heuristic Parameters:** To find the *best* schedule for a given set of integers, we *also* iterate over the heuristic's `variability_smoothness` parameter in fine steps.
4.  **Check and Keep:** We check every single generated candidate for feasibility (`constraints_residuals`) and, if it's feasible, we calculate its profit (`objective`). We then store the configuration that yields the highest profit.

## Log of Decisions

1. **Adopt "Heuristic Grid Search":** The new plan was to loop over all integer combinations and use the (now-fixed) heuristics to generate candidate schedules.

2. **Create a 4D Search Space:** A simple loop over `(model, n_r, n_s)` is not enough. We added the `variability_smoothness` parameter as a fourth dimension to our grid search to find a better heuristic schedule.

3. **Check Feasibility:** We added a crucial step inside the loop to call `constraints_residuals` and check that `np.min(res) >= -1e-6`. This ensures we only evaluate the profit of *feasible* solutions.

4. **Expand Search for Runtime:** The initial run was too fast (0.3s) and found a sub-optimal solution. To use the requested ~10-minute runtime, the search space was massively expanded:
    * **Max Reactors:** 5 &rarr; **20**
    * **Max Storage:** 30 &rarr; **400**
    * **Smoothness Steps:** 4 &rarr; **100**

This created a much larger and finer-grained search, allowing the script to find a significantly better (and still feasible) configuration.

### Result

--- Evaluating candidate 'Brute Force Optimum
Model=2, N_R=1, N_S=11 (Smooth=55%)' ---
Annual Profit (EUR): 86,686,077.04
Min Inequality Residual (>=0 for feasibility): 0.000000


# Two-Stage Multi-Variable Refinement Strategy

To leverage increased compute power for exploring a higher-dimensional solution space, the updated script implements a "Coarse-to-Fine" search strategy across both hardware configuration and operational parameters.

### 1. Stage 1: Multi-Variable Coarse Grid Search (The "Wide Net")
The script first scans a significantly expanded feasible solution space at a **coarse resolution**. It maps the profit landscape to identify promising hardware regions ("basins of attraction") by testing them against a grid of operational strategies.
* **Expanded Hardware Search:**
    * **Reactors:** Scanned at **Step 1** (checking every count from 0 to 30).
    * **Storage:** Scanned at **Step 10** (checking 0, 10, 20... up to 1000).
* **3D Operational Grid:** Instead of a single heuristic line, the algorithm evaluates a Cartesian product of **3 variables** for every hardware setup:
    1.  **Smoothness (10 steps):** Ranging from "Follow Demand" to "Flat Baseload."
    2.  **Bias (4 steps):** Intentionally under-producing (0.9x) or over-producing (1.1x) relative to average demand.
    3.  **Initial SOC (3 steps):** Optimizing the starting battery charge (Empty vs. Buffered).
* **Goal:** Find the approximate hardware setups that yield high profits when paired with general operational strategies.

### 2. Selection: Diversity Filtering
To prevent the algorithm from getting stuck in a single local maximum, it filters the Stage 1 results to ensure diversity.
* **Sorting:** The script sorts all Stage 1 results by profit.
* **Filtering:** It selects the **Top 5 Distinct Hardware Configs**.
* **Definition:** "Distinct" is defined as a unique combination of `(Reactor_Model, N_Reactors)`.
* **Safety Net:** The original manual heuristics are forcibly added to this list to ensure the algorithm beats the baseline.

### 3. Stage 2: Hyper-Refinement (The "Microscope")
For each of the selected "seeds," the script performs a highly localized, high-precision search:
* **Local Window:** It searches a narrow reactor range ($\pm 1$) and a focused storage range ($\pm 20$) around the seed to fill in the gaps left by the "Step 10" search in Stage 1.
* **Extreme Granularity:** It performs a fine-grained search on the operational variables:
    * **Smoothness:** Increased to **50** steps.
    * **Bias:** Refined to **10** steps (0.85x to 1.15x).
    * **Initial SOC:** Refined to **5** steps (0% to 50%).
* **Outcome:** This creates a massive search density around the most promising candidates to squeeze out the final fraction of profitability.

### Result

M=2, R=1, S=7' ---
Annual Profit (EUR): 91,616,931.58
Min Inequality Residual (>=0 for feasibility): 0.000000
