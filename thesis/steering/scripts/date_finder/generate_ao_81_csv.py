import pandas as pd

# Load AO data
ao_df = pd.read_csv('/home/ekasteleyn/aurora_thesis/thesis/steering/data/norm.daily.ao.cdas.z1000.19500101_current.csv')

# Filter for the years 2016 to 2022
ao_filtered = ao_df[(ao_df['year'] >= 2016) & (ao_df['year'] <= 2022)].copy()

# 1. Active days (AO > 3)
active_days = ao_filtered[ao_filtered['ao_index_cdas'] > 3].copy()
num_active = len(active_days)
active_days['Phenomenon'] = 'AO'
active_days['Type'] = 'Active'

# 2. Same amount of neutral days (closest to 0) with matching MONTH distribution
ao_filtered['abs_ao'] = ao_filtered['ao_index_cdas'].abs()
potential_neutral = ao_filtered[ao_filtered['ao_index_cdas'] <= 3].copy()

# Count how many active days we have per MONTH
month_counts = active_days['month'].value_counts()

neutral_days_list = []
# For each month, pick the 'count' neutral days closest to 0
for month, count in month_counts.items():
    month_neutral = potential_neutral[potential_neutral['month'] == month]
    top_neutral = month_neutral.sort_values('abs_ao').head(count)
    neutral_days_list.append(top_neutral)

# Combine active and neutral
neutral_days = pd.concat(neutral_days_list).copy()
neutral_days['Phenomenon'] = 'AO'
neutral_days['Type'] = 'Neutral'

combined = pd.concat([active_days, neutral_days])
combined = combined.rename(columns={'year': 'Year', 'month': 'Month', 'day': 'Day'})
combined['Needs_Extraction'] = False # Doesn't matter for steer_aurora.py

# Select columns
final_df = combined[['Year', 'Month', 'Day', 'Phenomenon', 'Type', 'Needs_Extraction']]
final_df.to_csv('/home/ekasteleyn/aurora_thesis/thesis/steering/data/target_dates_ao_81.csv', index=False)
print("Created target_dates_ao_81.csv with", len(active_days), "Active and", len(neutral_days), "Neutral days.")
