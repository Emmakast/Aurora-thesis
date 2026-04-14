import pandas as pd
import numpy as np

def main():
    import warnings
    warnings.filterwarnings('ignore')

    # Load MJO
    # Format: year, month, day, RMM1, RMM2, phase, amplitude. Missing= 1.E36 or 999
    # with variable whitespace
    mjo_cols = ['Year', 'Month', 'Day', 'RMM1', 'RMM2', 'phase', 'amplitude', 'origin']
    mjo_df = pd.read_csv('thesis/scripts/steering/rmm.74toRealtime.txt', 
                         skiprows=2, 
                         sep=r'\s+', 
                         names=mjo_cols, 
                         usecols=[0, 1, 2, 3, 4, 5, 6, 7],
                         engine='python')
    mjo_df = mjo_df[['Year', 'Month', 'Day', 'amplitude']]
    mjo_df['amplitude'] = pd.to_numeric(mjo_df['amplitude'], errors='coerce')
    mjo_df = mjo_df[(mjo_df['amplitude'] < 999) & (mjo_df['amplitude'] < 1e35)]
    
    # Load AO
    ao_df = pd.read_csv('thesis/scripts/steering/norm.daily.ao.cdas.z1000.19500101_current.csv')
    ao_df.rename(columns={'year': 'Year', 'month': 'Month', 'day': 'Day', 'ao_index_cdas': 'ao_index'}, inplace=True)
    
    # Load ENSO
    enso_df = pd.read_csv('thesis/scripts/steering/soi.long.csv')
    if 'Date' in enso_df.columns:
        enso_df['Date'] = pd.to_datetime(enso_df['Date'])
        enso_df['Year'] = enso_df['Date'].dt.year
        enso_df['Month'] = enso_df['Date'].dt.month
        enso_df['Day'] = 15 # Set day to 15th as requested
    
    enso_df.rename(columns={'SOI': 'soi_index', ' SOI': 'soi_index'}, inplace=True)
    
    if 'soi_index' not in enso_df.columns:
        # Fallback if the column is something else
        enso_col = [c for c in enso_df.columns if 'SOI' in c.upper()][0]
        enso_df.rename(columns={enso_col: 'soi_index'}, inplace=True)

    enso_df = enso_df[enso_df['soi_index'] != -99.99]
    
    # Global tracking set for mutual exclusivity
    selected_dates = set()
    results = []

    def get_dates(df, phenomenon, metric_col, is_active_ascending, is_active_abs, date_type):
        df = df.copy()
        # Filter years 1974 to 2022
        df = df[(df['Year'] >= 1974) & (df['Year'] <= 2022)]
        
        # Priority for > 2016
        df['is_post_2016'] = df['Year'] > 2016

        if is_active_abs:
            df['sort_metric'] = df[metric_col].abs()
        else:
            df['sort_metric'] = df[metric_col]

        # Ensure we drop already selected dates
        df['date_tuple'] = list(zip(df['Year'], df['Month'], df['Day']))
        df = df[~df['date_tuple'].isin(selected_dates)]

        if date_type == 'Active':
            df = df.sort_values(by=['is_post_2016', 'sort_metric'], ascending=[False, is_active_ascending])
            selected = df.head(50)
            
            for _, row in selected.iterrows():
                selected_dates.add(row['date_tuple'])
                results.append({
                    'Year': int(row['Year']),
                    'Month': int(row['Month']),
                    'Day': int(row['Day']),
                    'Phenomenon': phenomenon,
                    'Type': 'Active',
                    'Needs_Extraction': row['Year'] != 2022
                })
            return selected['Month'].value_counts().to_dict()
            
        elif date_type == 'Neutral':
            # Need to match seasonality of Active dates
            # df contains available candidates
            df['abs_metric'] = df[metric_col].abs() # Neutral is closest to 0
            df = df.sort_values(by=['is_post_2016', 'abs_metric'], ascending=[False, True])
            
            for month, count in is_active_ascending.items(): # Hack: pass the target counts here
                month_cands = df[df['Month'] == month]
                # Filter out ones we pick progressively
                # Note: df is already sorted by closeness to 0
                picked = 0
                for _, row in month_cands.iterrows():
                    if row['date_tuple'] not in selected_dates:
                        selected_dates.add(row['date_tuple'])
                        results.append({
                            'Year': int(row['Year']),
                            'Month': int(row['Month']),
                            'Day': int(row['Day']),
                            'Phenomenon': phenomenon,
                            'Type': 'Neutral',
                            'Needs_Extraction': row['Year'] != 2022
                        })
                        picked += 1
                        if picked == count:
                            break

    # 1. Processing AO
    # Active: "lowest AO index for weak vortex" -> lowest actual value (ascending=True, abs=False)
    ao_month_counts = get_dates(ao_df, 'AO', 'ao_index', is_active_ascending=True, is_active_abs=False, date_type='Active')
    get_dates(ao_df, 'AO', 'ao_index', is_active_ascending=ao_month_counts, is_active_abs=False, date_type='Neutral')

    # 2. Processing MJO
    # Active: "highest MJO amplitude" -> highest value (ascending=False, abs=False as it's already positive)
    mjo_month_counts = get_dates(mjo_df, 'MJO', 'amplitude', is_active_ascending=False, is_active_abs=False, date_type='Active')
    get_dates(mjo_df, 'MJO', 'amplitude', is_active_ascending=mjo_month_counts, is_active_abs=False, date_type='Neutral')

    # 3. Processing ENSO
    # Active: "peak absolute extreme SOI" -> highest absolute value (ascending=False, abs=True)
    enso_month_counts = get_dates(enso_df, 'ENSO', 'soi_index', is_active_ascending=False, is_active_abs=True, date_type='Active')
    get_dates(enso_df, 'ENSO', 'soi_index', is_active_ascending=enso_month_counts, is_active_abs=True, date_type='Neutral')

    # Save to CSV
    res_df = pd.DataFrame(results)
    res_df.to_csv('target_dates.csv', index=False)
    print(f"Successfully generated target_dates.csv with {len(res_df)} rows.")

if __name__ == '__main__':
    main()
