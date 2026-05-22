import pandas as pd
import numpy as np

# Load data
ao_df = pd.read_csv('thesis/scripts/steering/norm.daily.ao.cdas.z1000.19500101_current.csv')

# Filter for the years 2016 to 2022 (inclusive), which matches the 232 days > 2
ao_filtered = ao_df[(ao_df['year'] >= 2016) & (ao_df['year'] <= 2022)].copy()

# 1. The 232 days with AO > 2 (most active positive days you referenced earlier)
active_days = ao_filtered[ao_filtered['ao_index_cdas'] > 2].copy()
active_month_dist = active_days['month'].value_counts().sort_index()

# 2. The 232 days closest to 0 (most neutral days)
ao_filtered['abs_ao'] = ao_filtered['ao_index_cdas'].abs()
neutral_days = ao_filtered.sort_values('abs_ao').head(232).copy()

neutral_min = neutral_days['ao_index_cdas'].min()
neutral_max = neutral_days['ao_index_cdas'].max()
neutral_month_dist = neutral_days['month'].value_counts().sort_index()

print(f"--- NEUTRAL DAYS (232 closest to 0) ---")
print(f"Range of AO index: {neutral_min:.5f} to {neutral_max:.5f}")
print(f"Absolute maximum distance from zero: {neutral_days['abs_ao'].max():.5f}")
print("\nMonth Distribution (Neutral):")
for m, c in neutral_month_dist.items():
    print(f"Month {m:2d}: {c} days")

print(f"\n--- ACTIVE DAYS (232 days with AO > 2) ---")
print("\nMonth Distribution (Active):")
for m, c in active_month_dist.items():
    print(f"Month {m:2d}: {c} days")

