import pandas as pd

# Load NAO data
nao_df = pd.read_csv('/home/ekasteleyn/aurora_thesis/thesis/steering/data/norm.daily.nao.cdas.z500.19500101_current (1).csv')

# Filter for the years 2016 to 2022
nao_filtered = nao_df[(nao_df['year'] >= 2016) & (nao_df['year'] <= 2022)].copy()

# 1. Active days (NAO > 1.5)
# Using 1.5 because NAO indices are generally lower than AO indices in this period
active_days = nao_filtered[nao_filtered['nao_index_cdas'] > 1.5].copy()
num_active = len(active_days)
active_days['Phenomenon'] = 'NAO'
active_days['Type'] = 'Active'

# 2. Same amount of neutral days (closest to 0) with matching MONTH distribution
nao_filtered['abs_nao'] = nao_filtered['nao_index_cdas'].abs()
potential_neutral = nao_filtered[nao_filtered['nao_index_cdas'] <= 1.5].copy()

# Count how many active days we have per MONTH
month_counts = active_days['month'].value_counts()

neutral_days_list = []
# For each month, pick the 'count' neutral days closest to 0
for month, count in month_counts.items():
    month_neutral = potential_neutral[potential_neutral['month'] == month]
    top_neutral = month_neutral.sort_values('abs_nao').head(count)
    neutral_days_list.append(top_neutral)

# Combine active and neutral
neutral_days = pd.concat(neutral_days_list).copy()
neutral_days['Phenomenon'] = 'NAO'
neutral_days['Type'] = 'Neutral'

combined = pd.concat([active_days, neutral_days])
combined = combined.rename(columns={'year': 'Year', 'month': 'Month', 'day': 'Day'})
combined['Needs_Extraction'] = False # Doesn't matter for steer_aurora.py

# Select columns
final_df = combined[['Year', 'Month', 'Day', 'Phenomenon', 'Type', 'Needs_Extraction']]
final_df.to_csv('/home/ekasteleyn/aurora_thesis/thesis/steering/data/target_dates_nao.csv', index=False)
print("Created target_dates_nao.csv with", len(active_days), "Active and", len(neutral_days), "Neutral days.")
