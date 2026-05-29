"""
prefetch_enso_data.py

Goal: Pre-download all required HRES T0 data (current day and prev day for init_hour=0)
for all dates in target_dates_enso.csv. 
This saves expensive GPU idle time during the causal rollout phase.
"""
import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import pandas as pd
from pathlib import Path
import sys

# Add data_loader directory to path
sys.path.append(str(Path("/home/ekasteleyn/aurora_thesis/thesis/steering/scripts/data_loader")))
try:
    from extract_latents_hres import download_data, download_static
except ImportError:
    print("Warning: Could not import dataloader functions.")

def load_all_dates(csv_path):
    """Load all target dates from the CSV."""
    df = pd.read_csv(csv_path)
    dates = []
    for _, row in df.iterrows():
        dates.append(f"{int(row['Year'])}-{int(row['Month']):02d}-{int(row['Day']):02d}")
    # Return unique dates to avoid redundant checks
    return list(set(dates))

def main():
    dates_csv = "/home/ekasteleyn/aurora_thesis/thesis/steering/data/target_dates_enso.csv"
    download_path = Path("/scratch-shared/ekasteleyn/aurora_data")
    download_path.mkdir(parents=True, exist_ok=True)
    
    print("Downloading static variables...")
    download_static(download_path)
    
    all_dates = load_all_dates(dates_csv)
    print(f"Found {len(all_dates)} unique dates in the CSV.")
    
    for i, day in enumerate(all_dates):
        print(f"\n[{i+1}/{len(all_dates)}] Processing {day}...")
        
        # We need the current day's data
        download_data(day, download_path)
        
        # We also need the previous day's data because init_hour=0 requires T-6h (18:00 of prev day)
        prev_day = (pd.to_datetime(day) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        download_data(prev_day, download_path)
        
    print("\nAll data prefetched successfully!")

if __name__ == "__main__":
    main()
