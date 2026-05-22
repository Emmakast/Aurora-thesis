import pandas as pd
import numpy as np
import warnings

def main():
    warnings.filterwarnings('ignore')

    # ==========================================
    # 1. Load Data
    # ==========================================
    
    # Load MJO Data
    # Format: year, month, day, RMM1, RMM2, phase, amplitude. Missing = 1.E36 or 999
    # The txt file uses variable whitespace separators
    mjo_cols = ['Year', 'Month', 'Day', 'RMM1', 'RMM2', 'phase', 'amplitude', 'origin']
    mjo_df = pd.read_csv(
        'thesis/scripts/steering/rmm.74toRealtime.txt', 
        skiprows=2, 
        sep=r'\s+', 
        names=mjo_cols, 
        usecols=[0, 1, 2, 3, 4, 5, 6, 7],
        engine='python'
    )
    mjo_df = mjo_df[['Year', 'Month', 'Day', 'amplitude']]
    mjo_df['amplitude'] = pd.to_numeric(mjo_df['amplitude'], errors='coerce')
    # Filter out missing values (999 or 1.E36)
    mjo_df = mjo_df[(mjo_df['amplitude'] < 999) & (mjo_df['amplitude'] < 1e35)]
    
    # Load AO Data
    ao_df = pd.read_csv('thesis/scripts/steering/norm.daily.ao.cdas.z1000.19500101_current.csv')
    ao_df.rename(columns={'year': 'Year', 'month': 'Month', 'day': 'Day', 'ao_index_cdas': 'ao_index'}, inplace=True)
    
    # Load ENSO Data
    enso_df = pd.read_csv('thesis/scripts/steering/soi.long.csv')
    if 'Date' in enso_df.columns:
        enso_df['Date'] = pd.to_datetime(enso_df['Date'])
        enso_df['Year'] = enso_df['Date'].dt.year
        enso_df['Month'] = enso_df['Date'].dt.month
        # Assign the 15th day of the month for monthly indexes
        enso_df['Day'] = 15 
    
    # Ensure standard names and filter missing values
    enso_df.rename(columns={'SOI': 'soi_index', ' SOI': 'soi_index'}, inplace=True)
    if 'soi_index' not in enso_df.columns:
        enso_col = [c for c in enso_df.columns if 'SOI' in c.upper()][0]
        enso_df.rename(columns={enso_col: 'soi_index'}, inplace=True)
    enso_df = enso_df[enso_df['soi_index'] != -99.99]
    
    # ==========================================
    # 2. Filtering & Selection Logic
    # ==========================================

    # Global tracking set to guarantee mutual exclusivity across all phenomena
    selected_dates = set()
    results = []

    def select_dates(df, phenomenon, metric_col, is_active_ascending, sort_by_abs, date_type, active_month_counts=None):
        """
        Samples the targets given the constraints and modifies `selected_dates` & `results` tracking variable arrays.
        """
        df = df.copy()
        
        # Filter years between 1974 to 2022
        df = df[(df['Year'] >= 1974) & (df['Year'] <= 2022)]
        
        # Priority flag: sort items > 2016 first
        df['is_post_2016'] = df['Year'] > 2016

        # Metric to score against
        df['sort_metric'] = df[metric_col].abs() if sort_by_abs else df[metric_col]

        # Enforce distinct (Mutual Exclusivity)
        df['date_tuple'] = list(zip(df['Year'], df['Month'], df['Day']))
        df = df[~df['date_tuple'].isin(selected_dates)]

        if date_type == 'Active':
            # Priorities: 
            # 1. Post-2016 (True > False), 2. Desired Metric (ascending or descending)
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
                    'Needs_Extraction': row['Year'] != 2022  # Set to False if 2022, else True
                })
                
            # Return season representation to be matched by neutral sampler
            return selected['Month'].value_counts().to_dict()
            
        elif date_type == 'Neutral':
            # Neutral is always closest to 0
            df['abs_metric'] = df[metric_col].abs()
            # Sort by post-2016 availability, then minimum extreme (least deviation from 0)
            df = df.sort_values(by=['is_post_2016', 'abs_metric'], ascending=[False, True])
            
            # Find closest to 0 given the target distributions from Active selection
            for month, target_count in active_month_counts.items():
                month_cands = df[df['Month'] == month]
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
                        if picked == target_count:
                            break

    # ==========================================
    # 3. Execution
    # ==========================================

    # A) AO: "lowest AO index for weak vortex" -> Absolute lowest value (ascending=True)
    ao_counts = select_dates(ao_df, 'AO', 'ao_index', is_active_ascending=True, sort_by_abs=False, date_type='Active')
    select_dates(ao_df, 'AO', 'ao_index', is_active_ascending=False, sort_by_abs=False, date_type='Neutral', active_month_counts=ao_counts)

    # B) MJO: "highest MJO amplitude" -> highest value (ascending=False)
    mjo_counts = select_dates(mjo_df, 'MJO', 'amplitude', is_active_ascending=False, sort_by_abs=False, date_type='Active')
    select_dates(mjo_df, 'MJO', 'amplitude', is_active_ascending=False, sort_by_abs=False, date_type='Neutral', active_month_counts=mjo_counts)

    # C) ENSO: "peak absolute extreme SOI" -> highest absolute value (ascending=False, sort_by_abs=True)
    enso_counts = select_dates(enso_df, 'ENSO', 'soi_index', is_active_ascending=False, sort_by_abs=True, date_type='Active')
    select_dates(enso_df, 'ENSO', 'soi_index', is_active_ascending=False, sort_by_abs=True, date_type='Neutral', active_month_counts=enso_counts)

    # Convert results array to DataFrame
    res_df = pd.DataFrame(results)
    
    # Save the output
    res_df.to_csv('target_dates.csv', index=False)
    print(f"Successfully generated target_dates.csv with {len(res_df)} specific dates.")

if __name__ == '__main__':
    main()