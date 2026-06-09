#!/usr/bin/env python3
import os
import re
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr

def standardize_coords(ds: xr.Dataset) -> xr.Dataset:
    """Standardize latitude and longitude coordinate names."""
    rename_dict = {}
    if 'latitude' in ds.coords:
        rename_dict['latitude'] = 'lat'
    if 'longitude' in ds.coords:
        rename_dict['longitude'] = 'lon'
    if 'valid_time' in ds.coords:
        rename_dict['valid_time'] = 'time'
    if 'geopotential' in ds.variables:
        rename_dict['geopotential'] = 'z'
    return ds.rename(rename_dict) if rename_dict else ds

def get_anomaly(ds: xr.Dataset, clim: xr.Dataset, var: str, level: int, doy: int, hour: int) -> xr.DataArray:
    """Calculate the anomaly from climatology."""
    data = ds[var].sel(level=level)
    
    if 'dayofyear' in clim.coords and 'hour' in clim.coords:
        clim_slice = clim[var].sel(level=level, dayofyear=doy, hour=hour, method='nearest')
    elif 'time' in clim.coords:
        clim_slice = clim[var].sel(level=level).groupby('time.dayofyear')[doy].mean('time')
    else:
        raise ValueError("Unknown climatology time coordinate structure.")
        
    clim_interp = clim_slice.interp(lat=data.lat, lon=data.lon, method='linear')
    return data - clim_interp

def calculate_indices(filepath: Path, ds_clim: xr.Dataset, eof_pattern: xr.DataArray, pc1_std: float) -> tuple[float, float, pd.Timestamp]:
    """Calculate legacy and corrected AO indices for a single file."""
    ds = standardize_coords(xr.open_dataset(filepath))
    
    # Determine target time (last rollout step)
    if 'time' in ds.coords:
        valid_times = ds['time'].values
        target_time = pd.to_datetime(valid_times[-1])
    else:
        # Fallback: parse from filename
        match = re.search(r'(\d{8})_(\d{4})', filepath.name)
        if not match:
            raise ValueError(f"Could not extract date and time from filename '{filepath.name}'.")
        date_str, time_str = match.groups()
        from datetime import datetime
        init_time = datetime.strptime(f"{date_str}{time_str}", "%Y%m%d%H%M")
        target_time = pd.to_datetime(init_time) + pd.Timedelta(hours=72)
        
    doy = target_time.dayofyear
    hour = target_time.hour
    
    # Calculate anomaly at 1000 hPa
    anom_1000 = get_anomaly(ds, ds_clim, 'z', 1000, doy, hour).squeeze()
    
    # Restrict to Northern Hemisphere (lat >= 20)
    anom_nh = anom_1000.where(anom_1000.lat >= 20, drop=True)
    eof_nh = eof_pattern.interp(lat=anom_nh.lat, lon=anom_nh.lon, method='nearest').where(anom_nh.lat >= 20, drop=True)
    
    # 1. Legacy Projection: cos(lat) weighting (effectively cos^1.5(lat) when combined with weighted EOF)
    lat_weights_legacy = np.cos(np.deg2rad(anom_nh.lat))
    anom_weighted_legacy = anom_nh * lat_weights_legacy
    raw_proj_legacy = (anom_weighted_legacy * eof_nh).sum(dim=['lat', 'lon']).values
    ao_index_legacy = float(raw_proj_legacy / pc1_std)
    
    # 2. Corrected Projection: sqrt(cos(lat)) weighting (mathematically consistent with weighted EOF)
    # Prevent tiny negative floating point numbers at lat=90 from breaking the sqrt using np.clip
    lat_weights_corrected = np.sqrt(np.clip(np.cos(np.deg2rad(anom_nh.lat)), 0, None))
    anom_weighted_corrected = anom_nh * lat_weights_corrected
    raw_proj_corrected = (anom_weighted_corrected * eof_nh).sum(dim=['lat', 'lon']).values
    ao_index_corrected = float(raw_proj_corrected / pc1_std)
    
    return ao_index_legacy, ao_index_corrected, target_time

def main():
    parser = argparse.ArgumentParser(description="Batch calculate AO indices for steered and base files.")
    parser.add_argument("--vectors-dir", type=str, default="/home/ekasteleyn/aurora_thesis/thesis/steering/vectors", help="Path to vectors directory")
    parser.add_argument("--eof", type=str, default="/home/ekasteleyn/aurora_thesis/thesis/steering/scripts/oscillation_calculator/indices/ao_loading_pattern.nc", help="Path to EOF pattern file")
    parser.add_argument("--output", type=str, default="/home/ekasteleyn/aurora_thesis/thesis/steering/vectors/all_ao_indices.csv", help="Global output CSV path")
    parser.add_argument("--climatology", type=str, default="gs://weatherbench2/datasets/era5-hourly-climatology/1990-2017_6h_1440x721.zarr", help="Path to climatology Zarr store or local NC file")
    args = parser.parse_args()
    
    vectors_dir = Path(args.vectors_dir)
    eof_path = Path(args.eof)
    
    if not eof_path.exists():
        print(f"Error: EOF file not found at {eof_path}")
        return
        
    print(f"Loading EOF pattern from {eof_path}...")
    ds_eof = standardize_coords(xr.open_dataset(eof_path))
    eof_pattern = ds_eof['eof'].squeeze()
    pc1_std = float(ds_eof['pc_std'].values)
    
    print(f"Loading climatology from: {args.climatology}...")
    if args.climatology.startswith('gs://') or args.climatology.endswith('.zarr'):
        ds_clim = standardize_coords(xr.open_zarr(args.climatology, consolidated=True))
    else:
        ds_clim = standardize_coords(xr.open_dataset(args.climatology))
    
    all_results = []
    
    # Loop through subdirectories
    subdirs = sorted([d for d in vectors_dir.iterdir() if d.is_dir()])
    
    for subdir in subdirs:
        if subdir.name.startswith("AAO"):
            print(f"\n[Warning] Skipping AAO directory: {subdir.name} (AAO loading pattern not available)")
            continue
            
        print(f"\nProcessing directory: {subdir.name}")
        
        # Find NetCDF files
        nc_files = list(subdir.glob("*.nc"))
        if not nc_files:
            print(f"  No NetCDF files found in {subdir.name}")
            continue
            
        local_results = []
        
        # Parse alpha value and run calculations
        for nc_file in nc_files:
            # Detect if it's base or steered
            is_base = "base_" in nc_file.name
            is_steered = "steered_" in nc_file.name
            
            if not is_base and not is_steered:
                continue
                
            # Parse alpha from filename
            alpha_match = re.search(r'alpha_(-?\d+\.?\d*)', nc_file.name)
            if alpha_match:
                alpha = float(alpha_match.group(1))
            else:
                alpha = 0.0 if is_base else None
                
            try:
                idx_legacy, idx_corrected, target_time = calculate_indices(nc_file, ds_clim, eof_pattern, pc1_std)
                local_results.append({
                    "Filename": nc_file.name,
                    "Alpha": alpha,
                    "Type": "Base" if is_base else "Steered",
                    "AO_Index_Legacy": idx_legacy,
                    "AO_Index_Corrected": idx_corrected,
                    "Target_Time": str(target_time)
                })
            except Exception as e:
                print(f"  Error processing {nc_file.name}: {e}")
                
        if local_results:
            df_local = pd.DataFrame(local_results).sort_values(by="Alpha")
            # Write local CSV
            local_csv_path = subdir / "ao_indices.csv"
            df_local.to_csv(local_csv_path, index=False)
            print(f"  Saved indices to {local_csv_path}")
            
            # Add to global results
            for row in local_results:
                row["Folder"] = subdir.name
                all_results.append(row)
                
    if all_results:
        df_global = pd.DataFrame(all_results)
        df_global.to_csv(args.output, index=False)
        print(f"\nSaved all results to global CSV: {args.output}")
        
        # Print a summary table
        print("\n=== Summary of AO Indices ===")
        # Group and pivot for readable output
        summary_df = df_global.pivot_table(
            index=["Folder", "Alpha"],
            values=["AO_Index_Legacy", "AO_Index_Corrected"]
        ).round(4)
        print(summary_df.to_markdown())

if __name__ == "__main__":
    main()
