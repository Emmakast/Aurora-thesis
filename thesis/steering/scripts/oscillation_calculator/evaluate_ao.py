import argparse
import xarray as xr
import numpy as np
import pandas as pd
import re
from datetime import datetime

def standardize_coords(ds: xr.Dataset) -> xr.Dataset:
    """Standardize latitude and longitude coordinate names."""
    rename_dict = {}
    if 'latitude' in ds.coords:
        rename_dict['latitude'] = 'lat'
    if 'longitude' in ds.coords:
        rename_dict['longitude'] = 'lon'
    if 'valid_time' in ds.coords:
        rename_dict['valid_time'] = 'time'
        
    # Standardize variable name using ds.variables to work with both Dataset and Zarr structures
    if 'geopotential' in ds.variables:
        rename_dict['geopotential'] = 'z'
        
    return ds.rename(rename_dict) if rename_dict else ds

def get_anomaly(ds: xr.Dataset, clim: xr.Dataset, var: str, level: int, doy: int, hour: int) -> xr.DataArray:
    """Calculate the anomaly from climatology."""
    data = ds[var].sel(level=level)
    
    # Try to match climatology format (either directly by time, or dayofyear/hour)
    if 'dayofyear' in clim.coords and 'hour' in clim.coords:
        clim_slice = clim[var].sel(level=level, dayofyear=doy, hour=hour, method='nearest')
    elif 'time' in clim.coords:
        # Fallback if climatology uses a generic time calendar
        clim_slice = clim[var].sel(level=level).groupby('time.dayofyear')[doy].mean('time')
    else:
        raise ValueError("Unknown climatology time coordinate structure.")
        
    # Interpolate climatology to match model grid if necessary
    clim_interp = clim_slice.interp(lat=data.lat, lon=data.lon, method='linear')
    return data - clim_interp

def main():
    parser = argparse.ArgumentParser(description="Evaluate AO Steering")
    parser.add_argument("--base", type=str, required=True, help="Base NetCDF file")
    parser.add_argument("--steered", type=str, required=True, help="Steered NetCDF file")
    parser.add_argument("--eof", type=str, default="ao_loading_pattern.nc", help="EOF loading pattern NC file")
    parser.add_argument("--climatology", type=str, default="gs://weatherbench2/datasets/era5-hourly-climatology/1990-2017_6h_1440x721.zarr", help="Path to climatology Zarr store or local NC file")
    args = parser.parse_args()

    print("Loading data...")
    ds_base = standardize_coords(xr.open_dataset(args.base))
    ds_steered = standardize_coords(xr.open_dataset(args.steered))
    
    print(f"Loading climatology from: {args.climatology}...")
    if args.climatology.startswith('gs://') or args.climatology.endswith('.zarr'):
        ds_clim = standardize_coords(xr.open_zarr(args.climatology, consolidated=True))
    else:
        ds_clim = standardize_coords(xr.open_dataset(args.climatology))

    print("\n--- Part 1 & 2: Quick Sanity Check (500 hPa Zonal Mean at 60N) ---")
    base_500 = ds_base['z'].sel(level=500).squeeze()
    steered_500 = ds_steered['z'].sel(level=500).squeeze()
    
    # Select nearest to 60N and take zonal mean
    base_60n_zm = base_500.sel(lat=60, method='nearest').mean(dim='lon').values
    steered_60n_zm = steered_500.sel(lat=60, method='nearest').mean(dim='lon').values
    diff = steered_60n_zm - base_60n_zm
    
    print(f"Base Z500 ZM @ 60N:    {float(base_60n_zm):.2f}")
    print(f"Steered Z500 ZM @ 60N: {float(steered_60n_zm):.2f}")
    print(f"Difference:            {float(diff):.2f} (Negative = tighter polar vortex)")

    print("\n--- Part 3: Formal AO Index (1000 hPa spatial projection) ---")
    print("Calculating 1000 hPa anomalies...")
    
    # Accurately determine target time based on step in filename
    if 'time' in ds_base.coords:
        valid_times = ds_base['time'].values
        target_time = pd.to_datetime(valid_times[-1]) 
    else:
        match = re.search(r'(\d{8})_(\d{4})', args.base)
        match_hyphen = re.search(r'(\d{4})-(\d{2})-(\d{2})(?:_step(\d+))?', args.base)
        if match:
            date_str, time_str = match.groups()
            init_time = datetime.strptime(f"{date_str}{time_str}", "%Y%m%d%H%M")
            target_time = pd.to_datetime(init_time) + pd.Timedelta(hours=72)
        elif match_hyphen:
            groups = match_hyphen.groups()
            y, m, d = groups[:3]
            step_str = groups[3]
            init_time = datetime.strptime(f"{y}{m}{d}0000", "%Y%m%d%H%M")
            if step_str:
                target_time = pd.to_datetime(init_time) + pd.Timedelta(hours=int(step_str)*6)
            else:
                target_time = pd.to_datetime(init_time) + pd.Timedelta(hours=72)
        else:
            raise ValueError(f"Could not extract date and time from filename '{args.base}'.")
        
    doy = target_time.dayofyear
    hour = target_time.hour
    print(f"Target Verification Timestamp: {target_time} (DOY: {doy}, Hour: {hour})")

    # Pass the accurate DOY and hour so we subtract the correct climatology slice
    base_anom_1000 = get_anomaly(ds_base, ds_clim, 'z', 1000, doy, hour).squeeze()
    steered_anom_1000 = get_anomaly(ds_steered, ds_clim, 'z', 1000, doy, hour).squeeze()

    try:
        ds_eof = standardize_coords(xr.open_dataset(args.eof))
        eof_pattern = ds_eof['eof'].squeeze()
        pc1_std = ds_eof['pc_std'].values
    except FileNotFoundError:
        print(f"Warning: EOF file '{args.eof}' not found.")
        print("Creating a dummy uniform EOF pattern so the pipeline can complete.")
        ds_eof = xr.Dataset(
            data_vars={'eof': (['lat', 'lon'], np.ones((base_anom_1000.lat.size, base_anom_1000.lon.size)))},
            coords={'lat': base_anom_1000.lat, 'lon': base_anom_1000.lon}
        )
        eof_pattern = ds_eof['eof']
        pc1_std = 1.0

    # Restrict to NH (lat >= 20)
    base_nh = base_anom_1000.where(base_anom_1000.lat >= 20, drop=True)
    steered_nh = steered_anom_1000.where(steered_anom_1000.lat >= 20, drop=True)
    eof_nh = eof_pattern.interp(lat=base_nh.lat, lon=base_nh.lon, method='nearest').where(base_nh.lat >= 20, drop=True)

    # Apply latitude weighting (sqrt(cos(lat)) is standard for EOFs, but using cos(lat) as requested)
    lat_weights = np.cos(np.deg2rad(base_nh.lat))
    
    base_weighted = base_nh * lat_weights
    steered_weighted = steered_nh * lat_weights

    # Spatial dot product (sum across lat/lon)
    base_raw_proj = (base_weighted * eof_nh).sum(dim=['lat', 'lon']).values
    steered_raw_proj = (steered_weighted * eof_nh).sum(dim=['lat', 'lon']).values

    # STANDARDIZE to get the -4 to +4 index
    base_ao_idx = base_raw_proj / pc1_std
    steered_ao_idx = steered_raw_proj / pc1_std
    
    print(f"Base AO Index:    {float(base_ao_idx):.4f}")
    print(f"Steered AO Index: {float(steered_ao_idx):.4f}")
    print(f"Delta:            {float(steered_ao_idx - base_ao_idx):.4f}")

if __name__ == "__main__":
    main()
