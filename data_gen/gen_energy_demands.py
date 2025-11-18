import numpy as np
import json

# 1. Base Profile
electric_demand = np.array(
    [
        160,
        152,
        144,
        140,
        144,
        160,
        200,
        240,
        280,
        300,
        320,
        340,
        360,
        368,
        352,
        340,
        320,
        312,
        300,
        280,
        260,
        240,
        220,
        192,
    ],
    dtype=float,
)

# 2. Generation Parameters
num_samples = 100
generated_demands = []
np.random.seed(42)  # Fix seed for reproducible results

print(f"Generating {num_samples} profiles...")

for _ in range(num_samples):
    # A. Scaling (Seasonality/Day Type): +/- 30%
    scale = np.random.uniform(0.7, 1.3)

    # B. Shape Warp (Weather): Small sine wave distortion
    t = np.linspace(0, 2 * np.pi, 24)
    warp_amp = np.random.uniform(-15, 15)
    warp_phase = np.random.uniform(0, 2 * np.pi)
    shape_warp = warp_amp * np.sin(t + warp_phase)

    # C. Random Noise: Fluctuations per hour
    noise = np.random.normal(0, 5.0, 24)

    # Combine
    new_profile = (electric_demand * scale) + shape_warp + noise

    # Cleanup: Positive values only, rounded
    new_profile = np.maximum(new_profile, 50.0)
    new_profile = np.round(new_profile, 2)

    generated_demands.append(new_profile.tolist())

# 3. Save to JSON
output_data = {
    "description": "300 realistic energy demand profiles based on base curve.",
    "count": num_samples,
    "energy_demands": [
        {"index": i + 1, "profile": profile}
        for i, profile in enumerate(generated_demands)
    ],
}

filename = "data_gen/energy_demands.json"
with open(filename, "w") as f:
    json.dump(output_data, f, indent=2)

print(f"Success! Data saved to '{filename}'")
