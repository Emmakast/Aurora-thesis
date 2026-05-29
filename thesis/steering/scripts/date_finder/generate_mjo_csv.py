import pandas as pd

# Load MJO data
# The file has 2 header lines and uses variable whitespace as a separator
mjo_df = pd.read_csv(
    '/home/ekasteleyn/aurora_thesis/thesis/steering/data/rmm.74toRealtime.txt',
    sep=r'\s+',
    skiprows=2,
    names=['year', 'month', 'day', 'RMM1', 'RMM2', 'phase', 'amplitude', 'info']
)

# Filter out missing values (1.E36 or 999)
mjo_df = mjo_df[mjo_df['amplitude'] < 999]

# Filter for the years 2016 to 2022
mjo_filtered = mjo_df[(mjo_df['year'] >= 2016) & (mjo_df['year'] <= 2022)].copy()

# 1. Active days (MJO amplitude > 3)
active_days = mjo_filtered[mjo_filtered['amplitude'] > 3].copy()
num_active = len(active_days)
active_days['Phenomenon'] = 'MJO'
active_days['Type'] = 'Active'

# 2. Same amount of neutral days (closest to 0) with matching MONTH distribution
potential_neutral = mjo_filtered[mjo_filtered['amplitude'] <= 3].copy()

# Count how many active days we have per MONTH
month_counts = active_days['month'].value_counts()

neutral_days_list = []
# For each month, pick the 'count' neutral days closest to 0
for month, count in month_counts.items():
    month_neutral = potential_neutral[potential_neutral['month'] == month]
    # For MJO, amplitude is already >= 0, so closest to 0 is just sorting by amplitude ascending
    top_neutral = month_neutral.sort_values('amplitude').head(count)
    neutral_days_list.append(top_neutral)

# Combine active and neutral
neutral_days = pd.concat(neutral_days_list).copy()
neutral_days['Phenomenon'] = 'MJO'
neutral_days['Type'] = 'Neutral'

combined = pd.concat([active_days, neutral_days])
combined = combined.rename(columns={'year': 'Year', 'month': 'Month', 'day': 'Day'})
combined['Needs_Extraction'] = False # Doesn't matter for steer_aurora.py

# Select columns
final_df = combined[['Year', 'Month', 'Day', 'Phenomenon', 'Type', 'Needs_Extraction']]
final_df.to_csv('/home/ekasteleyn/aurora_thesis/thesis/steering/data/target_dates_mjo.csv', index=False)
print("Created target_dates_mjo.csv with", len(active_days), "Active and", len(neutral_days), "Neutral days.")
