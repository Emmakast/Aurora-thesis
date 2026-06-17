import pandas as pd
import numpy as np

df = pd.read_csv("/home/ekasteleyn/aurora_thesis/neuripspaper/results/time_series_aurora_2022.csv")
df = df[(df["forecast_hour"] >= 6) & (df["forecast_hour"] <= 72)]

drifts = {"dry": [], "water": [], "energy": []}
for date, group in df.groupby("date"):
    if len(group) < 2: continue
    group = group.sort_values("forecast_hour")
    h = group["forecast_hour"].values
    
    for k, col in [("dry", "dry_mass_Eg"), ("water", "water_mass_kg"), ("energy", "total_energy_J")]:
        y = group[col].values
        # Filter out nans
        mask = ~np.isnan(y)
        if mask.sum() < 2: continue
        h_clean = h[mask]
        y_clean = y[mask]
        slope, intercept = np.polyfit(h_clean, y_clean, 1)
        if y_clean[0] != 0:
            drift_pct = (slope * 24 / y_clean[0]) * 100
            drifts[k].append(drift_pct)

print("Baseline Drift Pct Per Day (from time_series_aurora_2022.csv):")
print(f"  dry_mass_drift:     {np.mean(drifts['dry']):.6f}")
print(f"  water_mass_drift:   {np.mean(drifts['water']):.6f}")
print(f"  total_energy_drift: {np.mean(drifts['energy']):.6f}")
