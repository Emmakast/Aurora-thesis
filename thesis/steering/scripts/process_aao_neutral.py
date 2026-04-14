import pandas as pd

# Files to check for already processed/queued dates
csv_files = [
    'target_dates_ao_232.csv',
    'target_dates_ao_81.csv',
    'target_dates.csv',
    'target_dates_aao_medium.csv',
    'target_dates_aao_high.csv',
    'dates_to_extract.csv'
]

known_dates = set()
for f in csv_files:
    try:
        df = pd.read_csv(f)
        if 'Year' in df.columns and 'Month' in df.columns and 'Day' in df.columns:
            for _, row in df.iterrows():
                known_dates.add((int(row['Year']), int(row['Month']), int(row['Day'])))
    except FileNotFoundError:
        pass

# Load AAO data
df_aao = pd.read_csv('thesis/scripts/steering/norm.daily.aao.cdas.z700.19790101_current.csv')

# Filter years 2016-2022
df_aao_filtered = df_aao[(df_aao['year'] >= 2016) & (df_aao['year'] <= 2022)].copy()

# Find most neutral (closest to 0)
df_aao_filtered['abs_aao'] = df_aao_filtered['aao_index_cdas'].abs()
df_neutral = df_aao_filtered.sort_values('abs_aao').head(284).copy()

out = []
for _, row in df_neutral.iterrows():
    y, m, d = int(row['year']), int(row['month']), int(row['day'])
    
    needs_extraction = True
    if y == 2022:
        needs_extraction = False
    if (y, m, d) in known_dates:
        needs_extraction = False
        
    out.append({
        'Year': y,
        'Month': m,
        'Day': d,
        'Phenomenon': 'AAO',
        'Type': 'Neutral',
        'Needs_Extraction': needs_extraction
    })

out_df = pd.DataFrame(out)
# Sort chronologically for better readability
out_df = out_df.sort_values(['Year', 'Month', 'Day'])
out_df.to_csv('target_dates_aao_neutral.csv', index=False)

print(f"Created target_dates_aao_neutral.csv with {len(out_df)} dates.")
print(f"Of these, {out_df['Needs_Extraction'].sum()} need extraction.")
