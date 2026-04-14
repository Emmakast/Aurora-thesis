import pandas as pd

# Load AO data
ao_df = pd.read_csv('thesis/scripts/steering/norm.daily.ao.cdas.z1000.19500101_current.csv')

# Filter for the years 2016 to 2022
ao_filtered = ao_df[(ao_df['year'] >= 2016) & (ao_df['year'] <= 2022)].copy()

# 1. Active days (AO > 3)
active_days = ao_filtered[ao_filtered['ao_index_cdas'] > 3].copy()
num_active = len(active_days)
active_days['Phenomenon'] = 'AO'
active_days['Type'] = 'Active'

# 2. Same amount of neutral days (closest to 0)
ao_filtered['abs_ao'] = ao_filtered['ao_index_cdas'].abs()
neutral_days = ao_filtered.sort_values('abs_ao').head(num_active).copy()
neutral_days['Phenomenon'] = 'AO'
neutral_days['Type'] = 'Neutral'

# Combine
combined = pd.concat([active_days, neutral_days])
combined = combined.rename(columns={'year': 'Year', 'month': 'Month', 'day': 'Day'})
combined['Needs_Extraction'] = False # Doesn't matter for steer_aurora.py

# Select columns
final_df = combined[['Year', 'Month', 'Day', 'Phenomenon', 'Type', 'Needs_Extraction']]
final_df.to_csv('target_dates_ao_81.csv', index=False)
print("Created target_dates_ao_81.csv with", len(active_days), "Active and", len(neutral_days), "Neutral days.")

