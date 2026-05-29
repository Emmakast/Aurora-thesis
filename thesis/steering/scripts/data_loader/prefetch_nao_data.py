"""
prefetch_nao_data.py
"""
import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path("/home/ekasteleyn/aurora_thesis/thesis/steering/scripts/data_loader")))
try:
    from extract_latents_hres import download_data, download_static
except ImportError:
    pass

def load_all_dates(csv_path):
    df = pd.read_csv(csv_path)
    dates = []
    for _, row in df.iterrows():
        dates.append(f"{int(row['Year'])}-{int(row['Month']):02d}-{int(row['Day']):02d}")
    return list(set(dates))

def main():
    dates_csv = "/home/ekasteleyn/aurora_thesis/thesis/steering/data/target_dates_nao.csv"
    download_path = Path("/scratch-shared/ekasteleyn/aurora_data")
    download_path.mkdir(parents=True, exist_ok=True)
    
    print("Downloading static variables...")
    download_static(download_path)
    
    all_dates = load_all_dates(dates_csv)
    print(f"Found {len(all_dates)} unique dates in the CSV.")
    
    for i, day in enumerate(all_dates):
        print(f"\n[{i+1}/{len(all_dates)}] Processing {day}...")
        download_data(day, download_path)
        prev_day = (pd.to_datetime(day) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        download_data(prev_day, download_path)
        
    print("\nAll data prefetched successfully!")

if __name__ == "__main__":
    main()
