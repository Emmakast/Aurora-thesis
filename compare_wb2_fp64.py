#!/home/ekasteleyn/aurora_thesis/aurora_env/bin/python
"""
Compare H100 FP64 predictions against WeatherBench2 Aurora reference.
Compares temperature, u-wind, v-wind at 500, 700, 850 hPa levels.
"""

import fsspec
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
import sys

def log(msg):
    """Print with flush for immediate output."""
    print(msg, flush=True)

# Paths
LOCAL_DIR = Path("/scratch-shared/ekasteleyn/aurora_h100_fp64")
WB2_URL = "gs://weatherbench2/datasets/aurora/2022-1440x721.zarr"

# Pressure levels to compare
LEVELS = [500, 700, 850]

# Variables to compare (atmospheric)
ATMOS_VARS = ["t", "u", "v"]  # temperature, u-wind, v-wind
# Map from local var names to WB2 var names
VAR_MAP = {
    "t": "temperature",
    "u": "u_component_of_wind",
    "v": "v_component_of_wind",
}


def compute_rmse(a, b):
    """Compute RMSE between two arrays."""
    return float(np.sqrt(np.nanmean((a - b) ** 2)))


def compute_mae(a, b):
    """Compute MAE between two arrays."""
    return float(np.nanmean(np.abs(a - b)))


def compute_max_diff(a, b):
    """Compute max absolute difference."""
    return float(np.nanmax(np.abs(a - b)))


def main():
    log("=" * 70)
    log("  H100 FP64 vs WeatherBench2 Aurora Comparison")
    log("=" * 70)
    
    # Open WB2 Aurora reference
    log("\nOpening WB2 Aurora reference...")
    wb2 = xr.open_zarr(fsspec.get_mapper(WB2_URL), chunks=None)
    log(f"  WB2 init time range: {wb2.time.values[0]} to {wb2.time.values[-1]}")
    log(f"  WB2 levels: {wb2.level.values}")
    log(f"  WB2 prediction_timedelta (hours): {wb2.prediction_timedelta.values.astype('timedelta64[h]').astype(int)}")
    log(f"  WB2 dtype (temperature): {wb2['temperature'].dtype}")
    
    # Find matching local files
    local_files = sorted(LOCAL_DIR.glob("aurora_pred_*.nc"))
    log(f"\nFound {len(local_files)} local prediction files")
    
    # Pre-fetch WB2 data for init times we need (batch load for speed)
    # Our files are from 2022-01-15 and 2022-01-16, 00:00 and 12:00
    init_times_needed = [
        pd.Timestamp("2022-01-15 00:00"),
        pd.Timestamp("2022-01-15 12:00"),
        pd.Timestamp("2022-01-16 00:00"),
        pd.Timestamp("2022-01-16 12:00"),
    ]
    init_times_available = [t for t in init_times_needed if t in wb2.time.values]
    
    if not init_times_available:
        log("⚠ No matching init times in WB2!")
        return
    
    log(f"\nPre-fetching WB2 data for {len(init_times_available)} init times...")
    wb2_vars_needed = list(VAR_MAP.values())
    wb2_subset = wb2[wb2_vars_needed].sel(time=init_times_available, level=LEVELS)
    log("  Loading into memory (this may take a minute)...")
    wb2_subset = wb2_subset.compute()
    log("  ✓ WB2 data loaded")
    
    results = []
    
    total_files = len(local_files)
    for file_idx, local_path in enumerate(local_files):
        log(f"\n[{file_idx+1}/{total_files}] Processing: {local_path.name}")
        
        # Skip incomplete files (normal file should be ~570MB)
        if local_path.stat().st_size < 500_000_000:
            log(f"  Skipping incomplete: {local_path.name}")
            continue
            
        # Parse filename: aurora_pred_YYYYMMDD_HHMM_stepNN_XXh.nc
        parts = local_path.stem.split("_")
        date_str = parts[2]  # YYYYMMDD
        init_str = parts[3]  # HHMM
        step = int(parts[4].replace("step", ""))
        lead_hours = int(parts[5].replace("h", ""))
        
        # Compute init time
        init_time = pd.Timestamp(f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]} {init_str[:2]}:{init_str[2:]}")
        lead_td = np.timedelta64(lead_hours, 'h')
        
        # Check if init time is in WB2
        if init_time not in wb2_subset.time.values:
            continue
        
        # Check if lead time is in WB2 prediction_timedelta
        if lead_td not in wb2_subset.prediction_timedelta.values:
            continue
        
        try:
            # Load local file
            local = xr.open_dataset(local_path)
            
            for var in ATMOS_VARS:
                wb2_var = VAR_MAP[var]
                
                for level in LEVELS:
                    # Get local data
                    local_data = local[var].sel(level=level).values
                    
                    # Get WB2 data: select by init time and lead time (already in memory)
                    wb2_data = wb2_subset[wb2_var].sel(time=init_time, prediction_timedelta=lead_td, level=level).values
                    
                    # WB2 might have different lat ordering - check and align
                    if wb2_subset.latitude.values[0] < wb2_subset.latitude.values[-1]:
                        # WB2 is S->N, local is N->S, flip WB2
                        wb2_data = wb2_data[::-1, :]
                    
                    # Compute metrics
                    rmse = compute_rmse(local_data, wb2_data)
                    mae = compute_mae(local_data, wb2_data)
                    max_diff = compute_max_diff(local_data, wb2_data)
                    
                    results.append({
                        "init_time": init_time,
                        "lead_hours": lead_hours,
                        "variable": var,
                        "level": level,
                        "rmse": rmse,
                        "mae": mae,
                        "max_diff": max_diff,
                    })
            
            local.close()
            log(f"  ✓ Done ({len(ATMOS_VARS) * len(LEVELS)} comparisons)")
        except Exception as e:
            log(f"  ✗ Skipping corrupt: {local_path.name} ({e})")
    
    # Create results DataFrame
    df = pd.DataFrame(results)
    
    if len(df) == 0:
        log("\n⚠ No matching times found between local files and WB2!")
        return
    
    log(f"\nCompared {len(df)} variable/level/time combinations")
    
    # Summary by variable and level
    log("\n" + "=" * 70)
    log("  SUMMARY: Mean metrics by variable and level")
    log("=" * 70)
    
    summary = df.groupby(["variable", "level"]).agg({
        "rmse": ["mean", "std", "max"],
        "mae": ["mean", "max"],
        "max_diff": ["mean", "max"],
    }).round(6)
    
    log(summary.to_string())
    
    # Summary by lead time
    log("\n" + "=" * 70)
    log("  RMSE by lead time (averaged over variables/levels)")
    log("=" * 70)
    
    lead_summary = df.groupby("lead_hours")["rmse"].agg(["mean", "std", "max"]).round(6)
    log(lead_summary.to_string())
    
    # Check for any significant differences
    log("\n" + "=" * 70)
    log("  ASSESSMENT")
    log("=" * 70)
    
    max_rmse = df["rmse"].max()
    mean_rmse = df["rmse"].mean()
    max_max_diff = df["max_diff"].max()
    
    log(f"  Overall mean RMSE: {mean_rmse:.6f}")
    log(f"  Overall max RMSE:  {max_rmse:.6f}")
    log(f"  Overall max diff:  {max_max_diff:.6f}")
    
    if max_rmse < 1e-5:
        log("\n  ✓ EXCELLENT: Differences are within numerical precision (< 1e-5)")
    elif max_rmse < 1e-3:
        log("\n  ✓ GOOD: Differences are very small (< 1e-3)")
    elif max_rmse < 0.1:
        log("\n  ⚠ WARNING: Some differences detected (< 0.1)")
    else:
        log("\n  ⚠ SIGNIFICANT: Notable differences detected (>= 0.1)")
    
    # Save results
    output_path = LOCAL_DIR / "comparison_results.csv"
    df.to_csv(output_path, index=False)
    log(f"\n  Results saved to: {output_path}")
    
    wb2.close()


if __name__ == "__main__":
    main()
