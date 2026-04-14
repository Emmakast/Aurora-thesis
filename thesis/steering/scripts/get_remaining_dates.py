import pandas as pd

# Load files
ao_df = pd.read_csv('thesis/scripts/steering/norm.daily.ao.cdas.z1000.19500101_current.csv')
target_dates_df = pd.read_csv('target_dates.csv')

# Format dates
target_dates_df = target_dates_df.rename(columns={'Year': 'year', 'Month': 'month', 'Day': 'day'})
target_dates_df['date'] = pd.to_datetime(target_dates_df[['year', 'month', 'day']])

ao_df['date'] = pd.to_datetime(ao_df[['year', 'month', 'day']])

# Filter AO index > 2 and years 2016 to 2021 (excluding 2022)
ao_filtered = ao_df[(ao_df['year'] >= 2016) & (ao_df['year'] <= 2021) & (ao_df['ao_index_cdas'] > 2)]

# Filter out dates already in target_dates.csv
final_dates = ao_filtered[~ao_filtered['date'].isin(target_dates_df['date'])].copy()

# Save to csv
final_dates[['year', 'month', 'day', 'ao_index_cdas']].to_csv('dates_to_extract.csv', index=False)
print(f"Total dates found: {len(ao_filtered)}")
print(f"Dates already in target_dates.csv: {len(ao_filtered) - len(final_dates)}")
print(f"Remaining dates to extract: {len(final_dates)}")
