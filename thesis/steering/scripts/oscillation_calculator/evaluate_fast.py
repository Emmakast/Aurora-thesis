#!/usr/bin/env python3
import os
import re
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr

CLIMATOLOGY_ZARR = "gs://weatherbench2/datasets/era5-hourly-climatology/1990-2017_6h_1440x721.zarr"
EOFS_DIR = "/home/ekasteleyn/aurora_thesis/thesis/steering/scripts/oscillation_calculator/indices"

def standardize_coords(ds: xr.Dataset) -> xr.Dataset:
    rename_dict = {}
    if 'latitude' in ds.coords: rename_dict['latitude'] = 'lat'
    if 'longitude' in ds.coords: rename_dict['longitude'] = 'lon'
    if 'valid_time' in ds.coords: rename_dict['valid_time'] = 'time'
    if 'geopotential' in ds.variables: rename_dict['geopotential'] = 'z'
    return ds.rename(rename_dict) if rename_dict else ds

def get_anomaly(ds: xr.Dataset, clim: xr.Dataset, var: str, level: int, target_time: pd.Timestamp) -> xr.DataArray:
    data = ds[var]
    if 'level' in data.coords:
        data = data.sel(level=level)
        
    doy = target_time.dayofyear
    hour = target_time.hour
    
    if 'dayofyear' in clim.coords and 'hour' in clim.coords:
        clim_slice = clim[var].sel(level=level, dayofyear=doy, hour=hour, method='nearest')
    elif 'time' in clim.coords:
        clim_slice = clim[var].sel(level=level).groupby('time.dayofyear')[doy].mean('time')
    else:
        raise ValueError("Unknown climatology format.")
        
    clim_interp = clim_slice.interp(lat=data.lat, lon=data.lon, method='linear')
    return data - clim_interp

def evaluate_fast(name, target_index, directory, ds_clim):
    print(f"\n======================================")
    print(f"Evaluating {name} runs for {target_index} index")
    print(f"======================================")
    
    eof_path = Path(EOFS_DIR) / f"{target_index.lower()}_loading_pattern.nc"
    if not eof_path.exists():
        print(f"EOF file not found: {eof_path}")
        return
        
    ds_eof = standardize_coords(xr.open_dataset(eof_path))
    eof_pattern = ds_eof['eof'].squeeze()
    if 'mode' in eof_pattern.dims:
        eof_pattern = eof_pattern.sel(mode=0)
    pc1_std = float(ds_eof['pc_std'].values)
    
    lat_min, lat_max = float(eof_pattern.lat.min()), float(eof_pattern.lat.max())
    lon_min, lon_max = float(eof_pattern.lon.min()), float(eof_pattern.lon.max())
    
    files = list(Path(directory).glob("*.nc"))
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
            ds = standardize_coords(xr.open_dataset(f))
            if 'time' in ds.coords:
                target_time = pd.to_datetime(ds['time'].values[-1])
                ds = ds.sel(time=ds['time'].values[-1])
            else:
                match = re.search(r'(\d{8})_(\d{4})', filename)
                date_str, time_str = match.groups()
                from datetime import datetime
                init_time = datetime.strptime(f"{date_str}{time_str}", "%Y%m%d%H%M")
                target_time = pd.to_datetime(init_time) + pd.Timedelta(hours=72)
            
            level = 500 if target_index in ['NAO', 'PNA'] else 700
            
            # Standardize lon
            lon_format = '180' if eof_pattern.lon.min() < 0 else '360'
            if lon_format == '180':
                ds = ds.assign_coords(lon=(((ds.lon + 180) % 360) - 180)).sortby('lon')
            else:
                ds = ds.assign_coords(lon=(ds.lon % 360)).sortby('lon')
                
            anom = get_anomaly(ds, ds_clim, 'z', level, target_time).squeeze()
            
            # Slice domain
            lat_slice = slice(lat_max, lat_min) if anom.lat.values[0] > anom.lat.values[-1] else slice(lat_min, lat_max)
            anom_sliced = anom.sel(lat=lat_slice, lon=slice(lon_min, lon_max))
            
            # Interpolate EOF to match anomaly grid exactly in domain
            eof_sliced = eof_pattern.interp(lat=anom_sliced.lat, lon=anom_sliced.lon, method='nearest')
            
            # Apply weighting
            lat_weights = np.sqrt(np.clip(np.cos(np.deg2rad(anom_sliced.lat)), 0, None))
            anom_weighted = anom_sliced * lat_weights
            
            raw_proj = (anom_weighted * eof_sliced).sum(dim=['lat', 'lon']).values
            index_val = float(raw_proj / pc1_std)
            
            results.append({
                "Filename": filename,
                "Type": "Base" if is_base else "Steered",
                "Alpha": alpha,
                f"{target_index}_Index (t=72h)": index_val
            })
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            
    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values(by="Alpha")
        print(df.to_markdown(index=False))
        
        # Save to CSV
        output_csv = Path(EOFS_DIR).parent / f"evaluation_results_{target_index.lower()}.csv"
        df.to_csv(output_csv, index=False)
        print(f"Saved results to {output_csv}")

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    print("Loading climatology...")
    ds_clim = standardize_coords(xr.open_zarr(CLIMATOLOGY_ZARR, consolidated=True))
    evaluate_fast("NAO", "NAO", "/scratch-shared/ekasteleyn/nao_steered/", ds_clim)
    evaluate_fast("PNA", "PNA", "/scratch-shared/ekasteleyn/pna_neutral_steered/", ds_clim)
    evaluate_fast("AAO", "AAO", "/home/ekasteleyn/aurora_thesis/thesis/steering/vectors/AAO_1encoder(2)", ds_clim)
