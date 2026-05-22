import pandas as pd
import numpy as np

# Load files
ao_df = pd.read_csv('thesis/scripts/steering/norm.daily.ao.cdas.z1000.19500101_current.csv')
target_dates_df = pd.read_csv('target_dates.csv')

# Format dates
target_dates_df = target_dates_df.rename(columns={'Year': 'year', 'Month': 'month', 'Day': 'day'})
ao_df['date'] = pd.to_datetime(ao_df[['year', 'month', 'day']])
target_dates_df['date'] = pd.to_datetime(target_dates_df[['year', 'month', 'day']])

# 1. Filter for the years 2016 to 2022 (inclusive)
ao_filtered = ao_df[(ao_df['year'] >= 2016) & (ao_df['year'] <= 2022)].copy()

# 2. Get the 232 days closest to 0
ao_filtered['abs_ao'] = ao_filtered['ao_index_cdas'].abs()
neutral_days = ao_filtered.sort_values('abs_ao').head(232).copy()

# 3. Filter out 2022 because they are already stored
neutral_days_no2022 = neutral_days[neutral_days['year'] != 2022].copy()

# 4. Filter out dates already in target_dates.csv
final_neutral_dates = neutral_days_no2022[~neutral_days_no2022['date'].isin(target_dates_df['date'])].copy()

# Save to csv
final_neutral_dates[['year', 'month', 'day', 'ao_index_cdas']].to_csv('dates_to_extract_neutral.csv', index=False)
print(f"Total 232 most neutral days.")
print(f"Removed 2022 dates: {len(neutral_days) - len(neutral_days_no2022)}")
print(f"Removed dates already in target_dates.csv: {len(neutral_days_no2022) - len(final_neutral_dates)}")
print(f"Remaining neutral dates to extract: {len(final_neutral_dates)}")
