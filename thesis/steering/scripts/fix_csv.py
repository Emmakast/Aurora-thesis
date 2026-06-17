import pandas as pd

all_csv = '/home/ekasteleyn/aurora_thesis/thesis/results/all_indices_evaluated.csv'
ao_csv = '/home/ekasteleyn/aurora_thesis/thesis/steering/vectors/AO_1encoder(2)/ao_indices.csv'

df_all = pd.read_csv(all_csv)
df_ao = pd.read_csv(ao_csv)

# Merge the Corrected and Legacy values from df_ao into df_all based on Filename
df_merged = df_all.merge(df_ao[['Filename', 'AO_Index_Legacy', 'AO_Index_Corrected']], on='Filename', how='left', suffixes=('', '_new'))

# Update where there is a new value
mask = df_merged['AO_Index_Corrected_new'].notna()
df_merged.loc[mask, 'AO_Index_Legacy'] = df_merged.loc[mask, 'AO_Index_Legacy_new']
df_merged.loc[mask, 'AO_Index_Corrected'] = df_merged.loc[mask, 'AO_Index_Corrected_new']

# Drop the _new columns
df_merged = df_merged.drop(columns=['AO_Index_Legacy_new', 'AO_Index_Corrected_new'])

# Save it back
df_merged.to_csv(all_csv, index=False)
print('Updated all_indices_evaluated.csv with correct AO indices from AO_1encoder(2)')
