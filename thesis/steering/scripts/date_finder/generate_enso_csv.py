import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# File paths
SOI_PATH = '/home/ekasteleyn/aurora_thesis/thesis/steering/data/soi.long.csv'
AO_PATH = '/home/ekasteleyn/aurora_thesis/thesis/steering/data/norm.daily.ao.cdas.z1000.19500101_current.csv'
MJO_PATH = '/home/ekasteleyn/aurora_thesis/thesis/steering/data/rmm.74toRealtime.txt'
OUT_PATH = '/home/ekasteleyn/aurora_thesis/thesis/steering/data/target_dates_enso.csv'

# Thresholds
EL_NINO_SOI_THRESH = -0.8  # Using -0.8 to capture weak/moderate El Nino months
NEUTRAL_SOI_MAX = 0.5      # Neutral is between -0.5 and 0.5
MAX_AO = 2.0               # Discard if AO is highly active
MAX_MJO = 2.0             # Discard if MJO is highly active
YEAR_START = 2016
YEAR_END = 2022

def load_data():
    # 1. Load SOI
    # Skip the first row (header), then parse
    soi_df = pd.read_csv(SOI_PATH, skiprows=1, names=['Date', 'SOI'])
    soi_df['Date'] = pd.to_datetime(soi_df['Date'], errors='coerce')
    soi_df = soi_df.dropna(subset=['Date'])
    soi_df['Year'] = soi_df['Date'].dt.year
    soi_df['Month'] = soi_df['Date'].dt.month
    
    # 2. Load AO
    ao_df = pd.read_csv(AO_PATH)
    ao_df['Date'] = pd.to_datetime(dict(year=ao_df.year, month=ao_df.month, day=ao_df.day))
    
    # 3. Load MJO
    mjo_df = pd.read_csv(
        MJO_PATH,
        sep=r'\s+',
        skiprows=2,
        names=['year', 'month', 'day', 'RMM1', 'RMM2', 'phase', 'amplitude', 'info']
    )
    # Filter out missing 999 values
    mjo_df = mjo_df[mjo_df['amplitude'] < 100]
    mjo_df['Date'] = pd.to_datetime(dict(year=mjo_df.year, month=mjo_df.month, day=mjo_df.day), errors='coerce')
    mjo_df = mjo_df.dropna(subset=['Date'])
    
    return soi_df, ao_df, mjo_df

def is_date_clean(date, ao_df, mjo_df):
    """Check if AO and MJO are within acceptable limits on this date."""
    # AO check
    ao_row = ao_df[ao_df['Date'] == date]
    if not ao_row.empty:
        if abs(ao_row.iloc[0]['ao_index_cdas']) > MAX_AO:
            return False
            
    # MJO check
    mjo_row = mjo_df[mjo_df['Date'] == date]
    if not mjo_row.empty:
        if mjo_row.iloc[0]['amplitude'] > MAX_MJO:
            return False
            
    return True

def get_clean_days_for_month(year, month, ao_df, mjo_df, target_days=[3, 8, 13, 18, 23, 28]):
    clean_dates = []
    for day in target_days:
        found_clean = False
        # Try the target day, and if not clean, try up to 3 days after
        for offset in range(4):
            try:
                candidate_date = datetime(year, month, day) + timedelta(days=offset)
            except ValueError:
                continue # Skip invalid dates (e.g. Feb 30)
                
            # Only allow looking within the same month
            if candidate_date.month != month:
                continue
                
            if is_date_clean(candidate_date, ao_df, mjo_df):
                clean_dates.append(candidate_date)
                found_clean = True
                break
                
    return clean_dates

def main():
    print("Loading data...")
    soi_df, ao_df, mjo_df = load_data()
    
    # Filter SOI for our target years
    soi_target_years = soi_df[(soi_df['Year'] >= YEAR_START) & (soi_df['Year'] <= YEAR_END)]
    
    # Find El Nino months
    el_nino_months = soi_target_years[soi_target_years['SOI'] <= EL_NINO_SOI_THRESH]
    print(f"Found {len(el_nino_months)} El Nino months between {YEAR_START}-{YEAR_END}.")
    
    active_dates = []
    active_month_distribution = {} # Keep track of which months we picked
    
    print("Sampling El Nino dates...")
    for _, row in el_nino_months.iterrows():
        year, month = int(row['Year']), int(row['Month'])
        clean_dates = get_clean_days_for_month(year, month, ao_df, mjo_df)
        active_dates.extend(clean_dates)
        
        # Track distribution
        for date in clean_dates:
            active_month_distribution[month] = active_month_distribution.get(month, 0) + 1
            
    print(f"Sampled {len(active_dates)} clean El Nino days.")
    
    # Now sample Neutral days matching the month distribution
    print("\nSampling Neutral dates to match the seasonal distribution...")
    neutral_dates = []
    
    # Candidate Neutral months
    neutral_months = soi_target_years[abs(soi_target_years['SOI']) <= NEUTRAL_SOI_MAX]
    
    for month, count_needed in active_month_distribution.items():
        # Get all neutral months that occurred in this specific month
        candidate_months = neutral_months[neutral_months['Month'] == month]
        
        month_neutral_dates = []
        for _, row in candidate_months.iterrows():
            year = int(row['Year'])
            clean_dates = get_clean_days_for_month(year, month, ao_df, mjo_df)
            month_neutral_dates.extend(clean_dates)
            
        if len(month_neutral_dates) < count_needed:
            print(f"  Warning: Not enough clean Neutral dates for month {month}. Needed {count_needed}, found {len(month_neutral_dates)}.")
            neutral_dates.extend(month_neutral_dates)
        else:
            neutral_dates.extend(month_neutral_dates[:count_needed])
            
    print(f"Sampled {len(neutral_dates)} clean Neutral days.")
    
    # Format for output
    active_df = pd.DataFrame({'Date': active_dates})
    active_df['Type'] = 'Active'
    
    neutral_df = pd.DataFrame({'Date': neutral_dates})
    neutral_df['Type'] = 'Neutral'
    
    final_df = pd.concat([active_df, neutral_df])
    final_df['Year'] = final_df['Date'].dt.year
    final_df['Month'] = final_df['Date'].dt.month
    final_df['Day'] = final_df['Date'].dt.day
    final_df['Phenomenon'] = 'ENSO'
    final_df['Needs_Extraction'] = False
    
    final_df = final_df[['Year', 'Month', 'Day', 'Phenomenon', 'Type', 'Needs_Extraction']]
    
    final_df.to_csv(OUT_PATH, index=False)
    print(f"\nSuccess! Saved {len(final_df)} total dates to {OUT_PATH}")

if __name__ == "__main__":
    main()
