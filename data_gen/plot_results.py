import json
import matplotlib.pyplot as plt
import numpy as np

# Load data
with open("data_gen/results.json", "r") as f:
    data = json.load(f)

results = data["results"]

# --- PLOT 1: Operational Dispatch (Example) ---
# Find a scenario with storage
target_profile = None
target_scenario = None

for res in results:
    for scen in res["scenarios"]:
        if scen["n_storage_fixed"] > 0:
            target_profile = res
            target_scenario = scen
            break
    if target_profile:
        break

if target_profile:
    t = np.arange(len(target_profile["demand_profile"]))

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Power (Left Axis)
    ax1.set_xlabel("Hour")
    ax1.set_ylabel("Power (MW)", color="black")
    ax1.plot(t, target_profile["demand_profile"], "k--", label="Demand", linewidth=2)
    ax1.step(t, target_scenario["prod"], where="mid", label="Nuclear Prod", color="blue")

    # Stack battery actions
    # Charge is load (negative for grid balance, but usually shown as positive consumption)
    # Discharge is supply
    ax1.bar(t, target_scenario["discharge"], width=0.8, label="Discharge", color="green", alpha=0.5)
    ax1.bar(
        t,
        [-c for c in target_scenario["charge"]],
        width=0.8,
        label="Charge",
        color="red",
        alpha=0.5,
    )

    ax1.tick_params(axis="y", labelcolor="black")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    # SOC (Right Axis)
    ax2 = ax1.twinx()
    ax2.set_ylabel("State of Charge (MWh)", color="purple")
    ax2.plot(t, target_scenario["soc"], "m-o", label="SOC", linewidth=2)
    ax2.tick_params(axis="y", labelcolor="purple")

    plt.title(
        f"Dispatch: Profile {target_profile['index']} | {target_scenario['m_mw']}x{target_scenario['n_r']} MW | {target_scenario['n_storage_fixed']} Storage Units"
    )
    plt.tight_layout()
    plt.savefig("data_gen/dispatch_plot.png")

# --- PLOT 2: Peak Demand vs Max Profit ---
peaks = []
max_profits = []
opt_storage = []

for res in results:
    if not res["scenarios"]:
        continue

    peak = np.max(res["demand_profile"])
    # Get best scenario
    best = max(res["scenarios"], key=lambda x: x["profit_eur"])

    peaks.append(peak)
    max_profits.append(best["profit_eur"] / 1e6)  # Convert to M€
    opt_storage.append(best["n_storage_fixed"])

plt.figure(figsize=(10, 6))
sc = plt.scatter(peaks, max_profits, c=opt_storage, cmap="viridis", s=50, alpha=0.8, edgecolors="w")
cbar = plt.colorbar(sc)
cbar.set_label("Optimal Storage Units")
plt.xlabel("Peak Demand (MW)")
plt.ylabel("Max Profit (M€)")
plt.title("Feasibility Overview: Demand vs Profit")
plt.grid(True, alpha=0.3)
plt.savefig("data_gen/feasibility_scatter.png")

# --- PLOT 3: Storage Sensitivity (Aggregate) ---
# For the first 5 profiles, plot profit curves vs storage
plt.figure(figsize=(10, 6))

count = 0
for res in results:
    if count >= 5:
        break

    # Extract unique storage levels and their best profits
    storage_profits = {}
    for scen in res["scenarios"]:
        s = scen["n_storage_fixed"]
        p = scen["profit_eur"] / 1e6
        if s not in storage_profits or p > storage_profits[s]:
            storage_profits[s] = p

    if len(storage_profits) > 1:
        lists = sorted(storage_profits.items())
        x, y = zip(*lists)
        plt.plot(x, y, marker="o", label=f"Profile {res['index']}")
        count += 1

plt.xlabel("Storage Units")
plt.ylabel("Profit (M€)")
plt.title("Sensitivity: Profit vs Storage Capacity (First 5 Profiles)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("data_gen/storage_sensitivity.png")
