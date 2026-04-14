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

# Filter for Phenomenon == 'AO' and Type == 'Active'
ao_active = merged[(merged['Phenomenon'] == 'AO') & (merged['Type'] == 'Active')].sort_values('ao_index_cdas', ascending=True)

print("Top 10 MOST NEGATIVE AO index dates labeled as Phenomenon='AO':\n")
print(ao_active[['year', 'month', 'day', 'Phenomenon', 'Type', 'ao_index_cdas']].head(10).to_string(index=False))

