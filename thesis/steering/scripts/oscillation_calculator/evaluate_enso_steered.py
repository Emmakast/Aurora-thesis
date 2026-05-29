import os
import re
from pathlib import Path
import pandas as pd
import xarray as xr
import numpy as np
import logging

logging.getLogger().setLevel(logging.ERROR)

def standardize_coords(ds: xr.Dataset) -> xr.Dataset:
    rename_dict = {}
    if 'latitude' in ds.coords: rename_dict['latitude'] = 'lat'
    if 'longitude' in ds.coords: rename_dict['longitude'] = 'lon'
    if 'valid_time' in ds.coords: rename_dict['valid_time'] = 'time'
    return ds.rename(rename_dict) if rename_dict else ds

def slice_nino34(ds):
    lat_min, lat_max = -5, 5
    if ds.lat.values[0] > ds.lat.values[-1]: ds = ds.sel(lat=slice(lat_max, lat_min))
    else: ds = ds.sel(lat=slice(lat_min, lat_max))
    ds = ds.assign_coords(lon=(ds.lon % 360)).sortby('lon')
    return ds.sel(lon=slice(190, 240))

def evaluate_enso_runs(directory):
    print(f"\n======================================")
    print(f"Evaluating ENSO runs for ENSO index")
    print(f"======================================")
    
    print("Loading climatology once...")
    clim_ds = standardize_coords(xr.open_zarr("gs://weatherbench2/datasets/era5-hourly-climatology/1990-2017_6h_1440x721.zarr", consolidated=True))
    clim_sliced = slice_nino34(clim_ds)['2m_temperature']
    
    files = list(Path(directory).glob("*_enso_*.nc"))
    results = []
    
    for f in sorted(files):
        filename = f.name
        is_base = "base_" in filename
        is_steered = "steered_" in filename
        if not is_base and not is_steered:
            continue
            
        alpha_match = re.search(r'alpha_(-?\d+\.?\d*)', filename)
        alpha = float(alpha_match.group(1)) if alpha_match else (0.0 if is_base else None)
        
        try:
            target_ds = standardize_coords(xr.open_dataset(f))
            target_var = slice_nino34(target_ds)['2t']
            
            anom_list = []
            target_var_expanded = target_var if 'time' in target_var.dims else target_var.expand_dims('time')
            
            for t in target_var_expanded.time:
                t_val = t.values
                t_dt = pd.to_datetime(t_val)
                c_var = clim_sliced
                if 'dayofyear' in c_var.coords:
                    c_var = c_var.sel(dayofyear=t_dt.dayofyear, hour=t_dt.hour)
                anom_list.append(target_var_expanded.sel(time=t) - c_var)
                
            anom = xr.concat(anom_list, dim='time')
            weights = np.cos(np.deg2rad(anom.lat))
            weights.name = 'weights'
            anom_weighted_mean = anom.weighted(weights).mean(dim=['lat', 'lon'])
            
            if len(anom_weighted_mean.time) >= 3:
                oni = anom_weighted_mean.rolling(time=3, center=True).mean()
            else:
                oni = anom_weighted_mean
                
            arr = np.array(oni.values).flatten()
            val = float(arr[-1]) # Or arr[~np.isnan(arr)][-1]
            # Handle NaN from rolling mean:
            valid_vals = arr[~np.isnan(arr)]
            val = float(valid_vals[-1]) if len(valid_vals) > 0 else float(arr[-1])
            
            results.append({
                "Filename": filename,
                "Type": "Base" if is_base else "Steered",
                "Alpha": alpha,
                "ENSO_Index (t=last)": val
            })
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            
    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values(by="Alpha")
        print(df.to_markdown(index=False))
        
        output_csv = Path(directory) / "evaluation_results_enso.csv"
        df.to_csv(output_csv, index=False)
        print(f"Saved results to {output_csv}")

if __name__ == "__main__":
    evaluate_enso_runs("/home/ekasteleyn/aurora_thesis/thesis/results/")
