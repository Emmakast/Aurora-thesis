import pandas as pd

# Load files
ao_df = pd.read_csv('thesis/scripts/steering/norm.daily.ao.cdas.z1000.19500101_current.csv')
target_dates_df = pd.read_csv('target_dates.csv')

# Format dates
target_dates_df = target_dates_df.rename(columns={'Year': 'year', 'Month': 'month', 'Day': 'day'})
ao_df['date'] = pd.to_datetime(ao_df[['year', 'month', 'day']])
target_dates_df['date'] = pd.to_datetime(target_dates_df[['year', 'month', 'day']])

# Merge
merged = pd.merge(target_dates_df, ao_df[['year', 'month', 'day', 'ao_index_cdas']], on=['year', 'month', 'day'], how='inner')

# Sort by AO index descending
highest_ao = merged.sort_values('ao_index_cdas', ascending=False)
print("Top 10 strongest polar vortex (highest AO index) dates in target_dates.csv:\n")
print(highest_ao[['year', 'month', 'day', 'Phenomenon', 'Needs_Extraction', 'ao_index_cdas']].head(10).to_string(index=False))

# Filter just those that are "already extracted / stored" meaning Needs_Extraction == False
stored = highest_ao[highest_ao['Needs_Extraction'] == False]
print("\nTop 10 strongest polar vortex (highest AO index) dates among ALREADY STORED (Needs_Extraction == False):\n")
print(stored[['year', 'month', 'day', 'Phenomenon', 'ao_index_cdas']].head(10).to_string(index=False))
