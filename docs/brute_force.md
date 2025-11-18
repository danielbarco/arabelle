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

# Two-Stage "Top-K" Refinement Strategy

To overcome the trade-off between runtime and precision, the updated script implements a "Coarse-to-Fine" search strategy. Instead of treating every potential configuration equally, it uses a two-stage funnel to locate the Global Optimum efficiently.

### 1. Stage 1: Ultra-Dense Grid Search (The "Wide Net")
The script first scans the entire feasible solution space at a **coarse resolution**. This maps the profit landscape to identify promising regions ("basins of attraction") without wasting time on micro-optimizations.
* **Reactors:** Scanned at **Step 1** (checking every count from 0 to 20).
* **Storage:** Scanned at **Step 5** (checking 0, 5, 10... up to 400).
* **Heuristics:** Uses **40** smoothness parameter steps per configuration.
* **Goal:** Find the approximate hardware setups that yield high profits.

### 2. Selection: Diversity Filtering
Simply taking the top 5 results from Stage 1 is risky; the top 5 results might all be identical hardware with negligible differences (e.g., storage=100 vs storage=105).
* **Sorting:** The script sorts all Stage 1 results by profit.
* **Filtering:** It selects the **Top 5 Distinct Hardware Configs**.
* **Definition:** "Distinct" is defined as a unique combination of `(Reactor_Model, N_Reactors)`.
* **Safety Net:** The original manual heuristics are forcibly added to this list to ensure the algorithm beats the baseline.

### 3. Stage 2: Hyper-Refinement (The "Microscope")
For each of the selected "seeds," the script performs a highly localized, high-precision search:
* **Local Window:** It searches a narrow reactor range ($\pm 1$) and a wide storage range ($\pm 50$) around the seed to fill in the gaps left by the "Step 5" search in Stage 1.
* **Extreme Granularity:** It tests **200** smoothness steps (up from 40) to find the exact operational balance between the reactor and storage.