import pandas as pd

# Load target dates
df_232 = pd.read_csv('target_dates_ao_232.csv')
df_81 = pd.read_csv('target_dates_ao_81.csv')
df_target = pd.read_csv('target_dates.csv')

known_dates = set()
for df in [df_232, df_81, df_target]:
    for _, row in df.iterrows():
        y, m, d = int(row['Year']), int(row['Month']), int(row['Day'])
        known_dates.add((y, m, d))

# Load AAO data
df_aao = pd.read_csv('thesis/scripts/steering/norm.daily.aao.cdas.z700.19790101_current.csv')

# Filter years 2016-2022
df_aao_filtered = df_aao[(df_aao['year'] >= 2016) & (df_aao['year'] <= 2022)].copy()

# High (aao > 3) and medium (aao > 2, which includes > 3)
df_high = df_aao_filtered[df_aao_filtered['aao_index_cdas'] > 3].copy()
df_medium = df_aao_filtered[df_aao_filtered['aao_index_cdas'] > 2].copy()

def prep_df(df, type_name):
    out = []
    for _, row in df.iterrows():
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
            'Type': type_name,
            'Needs_Extraction': needs_extraction
        })
    return pd.DataFrame(out)

out_high = prep_df(df_high, 'Active')
out_medium = prep_df(df_medium, 'Active')

out_high.to_csv('target_dates_aao_high.csv', index=False)
out_medium.to_csv('target_dates_aao_medium.csv', index=False)

print(f"Updated HIGH dates: {len(out_high)}")
print(f"Updated MEDIUM (>=2) dates: {len(out_medium)}")
