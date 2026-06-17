import pandas as pd
import xarray as xr
from pathlib import Path
import re
import numpy as np
import sys
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

def standardize_coords(ds: xr.Dataset) -> xr.Dataset:
    rename_dict = {}
    if 'latitude' in ds.coords: rename_dict['latitude'] = 'lat'
    if 'longitude' in ds.coords: rename_dict['longitude'] = 'lon'
    if 'valid_time' in ds.coords: rename_dict['valid_time'] = 'time'
    if 'geopotential' in ds.variables: rename_dict['geopotential'] = 'z'
    return ds.rename(rename_dict) if rename_dict else ds

def get_anomaly(ds: xr.Dataset, clim: xr.Dataset, var: str, level: int, doy: int, hour: int) -> xr.DataArray:
    data = ds[var].sel(level=level)
    if 'dayofyear' in clim.coords and 'hour' in clim.coords:
        clim_slice = clim[var].sel(level=level, dayofyear=doy, hour=hour, method='nearest')
    else:
        clim_slice = clim[var].sel(level=level).groupby('time.dayofyear')[doy].mean('time')
    clim_interp = clim_slice.interp(lat=data.lat, lon=data.lon, method='linear')
    return data - clim_interp

def calculate_ao(filepath: Path, ds_clim: xr.Dataset, eof_pattern: xr.DataArray, pc1_std: float):
    ds = standardize_coords(xr.open_dataset(filepath))
    if 'time' in ds.coords:
        valid_times = ds['time'].values
        target_time = pd.to_datetime(valid_times[-1])
    else:
        match = re.search(r'(\d{8})_(\d{4})', filepath.name)
        date_str, time_str = match.groups()
        from datetime import datetime
        init_time = datetime.strptime(f"{date_str}{time_str}", "%Y%m%d%H%M")
        target_time = pd.to_datetime(init_time) + pd.Timedelta(hours=72)
        
    doy = target_time.dayofyear
    hour = target_time.hour
    
    anom_1000 = get_anomaly(ds, ds_clim, 'z', 1000, doy, hour).squeeze()
    anom_nh = anom_1000.where(anom_1000.lat >= 20, drop=True)
    eof_nh = eof_pattern.interp(lat=anom_nh.lat, lon=anom_nh.lon, method='nearest').where(anom_nh.lat >= 20, drop=True)
    
    lat_weights_legacy = np.cos(np.deg2rad(anom_nh.lat))
    anom_weighted_legacy = anom_nh * lat_weights_legacy
    ao_index_legacy = float((anom_weighted_legacy * eof_nh).sum(dim=['lat', 'lon']).values / pc1_std)
    
    lat_weights_corrected = np.sqrt(np.clip(np.cos(np.deg2rad(anom_nh.lat)), 0, None))
    anom_weighted_corrected = anom_nh * lat_weights_corrected
    ao_index_corrected = float((anom_weighted_corrected * eof_nh).sum(dim=['lat', 'lon']).values / pc1_std)
    
    ds.close()
    return ao_index_legacy, ao_index_corrected, target_time

csv_path = "/home/ekasteleyn/aurora_thesis/thesis/results/all_indices_evaluated.csv"
eof_path = "/home/ekasteleyn/aurora_thesis/thesis/steering/scripts/oscillation_calculator/indices/ao_loading_pattern.nc"
clim_path = "gs://weatherbench2/datasets/era5-hourly-climatology/1990-2017_6h_1440x721.zarr"

print("Loading EOF...")
ds_eof = standardize_coords(xr.open_dataset(eof_path))
eof_pattern = ds_eof['eof'].squeeze()
pc1_std = float(ds_eof['daily_pc_std'].values) if 'daily_pc_std' in ds_eof else float(ds_eof['pc_std'].values)

print("Loading Climatology...")
ds_clim = standardize_coords(xr.open_zarr(clim_path, consolidated=True))

df = pd.read_csv(csv_path)

search_dirs = [
    Path("/scratch-shared/ekasteleyn/nao_steered"),
    Path("/scratch-shared/ekasteleyn/pna_neutral_steered"),
    Path("/scratch-shared/ekasteleyn/mjo_steered"),
    Path("/home/ekasteleyn/aurora_thesis/thesis/steering/vectors/AAO_1encoder(2)"),
    Path("/home/ekasteleyn/aurora_thesis/thesis/steering/vectors/AO_1encoder(2)"),
    Path("/home/ekasteleyn/aurora_thesis/thesis/results")
]

for idx, row in df.iterrows():
    fname = row['Filename']
    fpath = None
    for d in search_dirs:
        if (d / fname).exists():
            fpath = d / fname
            break
            
    if fpath:
        print(f"Processing {fname}...")
        try:
            leg, corr, ttime = calculate_ao(fpath, ds_clim, eof_pattern, pc1_std)
            df.at[idx, 'AO_Index_Legacy'] = leg
            df.at[idx, 'AO_Index_Corrected'] = corr
            df.at[idx, 'Target_Time'] = str(ttime)
        except Exception as e:
            print(f"Failed to calculate for {fname}: {e}")
    else:
        print(f"Could not find file {fname}")

df.to_csv(csv_path, index=False)
print("Saved updated CSV.")
