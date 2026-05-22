import pandas as pd
import numpy as np
from datetime import datetime

def load_and_merge_data():
    # Load AO and AAO
    ao = pd.read_csv('/home/ekasteleyn/aurora_thesis/thesis/steering/data/norm.daily.ao.cdas.z1000.19500101_current.csv')
    aao = pd.read_csv('/home/ekasteleyn/aurora_thesis/thesis/steering/data/norm.daily.aao.cdas.z700.19790101_current.csv')
    
    # Load RMM (Assuming standard format: Year, Month, Day, RMM1, RMM2, phase, amp)
    # Adjust `skiprows` and `names` according to the exact text file formatting
    rmm = pd.read_csv('/home/ekasteleyn/aurora_thesis/thesis/steering/data/rmm.74toRealtime.txt', sep='\s+', skiprows=2, 
                      names=['year', 'month', 'day', 'RMM1', 'RMM2', 'phase', 'amplitude', 'source'], usecols=[0,1,2,3,4,5,6])
    
    # Load SOI (Assuming Daily or Monthly interpolation: Year, Month, Day/None, SOI)
    soi = pd.read_csv('/home/ekasteleyn/aurora_thesis/thesis/steering/data/soi.long.csv')
    soi.columns = ['date_str', 'soi_value']
    soi['date_str'] = pd.to_datetime(soi['date_str'])
    soi['year'] = soi['date_str'].dt.year
    soi['month'] = soi['date_str'].dt.month

    # Merge on year, month, day
    df = ao.merge(aao, on=['year', 'month', 'day'], how='inner')
    df = df.merge(rmm, on=['year', 'month', 'day'], how='inner')
    df = df.merge(soi[['year', 'month', 'soi_value']], on=['year', 'month'], how='inner') # Monthly join for SOI
    
    # Form Datetime
    df['date'] = pd.to_datetime(df[['year', 'month', 'day']])
    df['day_of_year'] = df['date'].dt.dayofyear
    
    # Optional: Filter for a specific period like 2016-2022
    df = df[(df['year'] >= 2016) & (df['year'] <= 2022)].dropna().reset_index(drop=True)
    return df

def find_contrastive_pairs(df, top_n=10):
    # Definition of regimes
    active_ao = df[df['ao_index_cdas'] > 3.0].copy()
    neutral_ao = df[(df['ao_index_cdas'] >= -0.5) & (df['ao_index_cdas'] <= 0.5)].copy()

    pairs = []
    
    # Normalize features for distance calculation
    aao_std = df['aao_index_cdas'].std()
    
    soi_col = [c for c in df.columns if 'soi' in c.lower()][-1]
    soi_std = df[soi_col].std()

    for _, a_row in active_ao.iterrows():
        # Day of year difference (circular)
        doy_diff = np.abs(neutral_ao['day_of_year'] - a_row['day_of_year'])
        doy_diff = np.minimum(doy_diff, 365 - doy_diff)
        doy_penalty = doy_diff / 30.0 # roughly 1 unit per month difference
        
        # AAO difference
        aao_dist = np.abs(neutral_ao['aao_index_cdas'] - a_row['aao_index_cdas']) / aao_std
        
        # SOI difference
        soi_dist = np.abs(neutral_ao[soi_col] - a_row[soi_col]) / soi_std
        
        # RMM matching: prefer same phase, and similar amplitude
        phase_penalty = (neutral_ao['phase'] != a_row['phase']).astype(float) * 2.0
        amp_dist = np.abs(neutral_ao['amplitude'] - a_row['amplitude'])
        
        # Total distance
        total_dist = doy_penalty + aao_dist + soi_dist + phase_penalty + amp_dist
        
        best_match_idx = total_dist.idxmin()
        best_match = neutral_ao.loc[best_match_idx]
        
        pairs.append({
            'active_date': a_row['date'],
            'neutral_date': best_match['date'],
            'active_ao': a_row['ao_index_cdas'],
            'neutral_ao': best_match['ao_index_cdas'],
            'distance': total_dist.min()
        })
        
    pairs_df = pd.DataFrame(pairs).sort_values('distance').head(top_n)
    return pairs_df

if __name__ == '__main__':
    df = load_and_merge_data()
    
    top_10_pairs = find_contrastive_pairs(df, top_n=10)
    top_10_pairs.to_csv('contrastive_pairs_10.csv', index=False)
    
    top_1_pair = top_10_pairs.head(1)
    top_1_pair.to_csv('contrastive_pair_1.csv', index=False)
    
    print("Top 1 contrastive pair:")
    print(top_1_pair)
    print("\nTop 10 saved to 'contrastive_pairs_10.csv'")
