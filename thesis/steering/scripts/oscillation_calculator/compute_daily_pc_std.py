#!/usr/bin/env python3
"""
Compute daily Principal Component standard deviations for all oscillation indices.

This script projects 5 years (2015-2019) of daily ERA5 anomalies onto the existing
static EOF loading patterns and computes the standard deviation of the resulting
1D PC time series. The daily_pc_std is then saved into the existing .nc files as
a new variable, fixing the inflated index values caused by dividing daily-frequency
projections by monthly-frequency pc_std.

Usage:
    python compute_daily_pc_std.py [--index NAO|PNA|AAO|AO|MJO|ALL]
"""

import argparse
import logging
import os
import numpy as np
import xarray as xr

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ─── Configuration ────────────────────────────────────────────────────────────

HISTORICAL_ZARR = 'gs://weatherbench2/datasets/era5/1959-2022-6h-1440x721.zarr'
CLIMATOLOGY_ZARR = 'gs://weatherbench2/datasets/era5-hourly-climatology/1990-2017_6h_1440x721.zarr'

INDICES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'indices')

TIME_SLICE = ('2015-01-01', '2019-12-31')

# Index definitions matching generate_eof.py exactly
INDICES = {
    'NAO': {
        'level': 500,
        'lat_range': (20, 90),
        'lon_range': (-90, 40),
        'lon_format': '180',
    },
    'PNA': {
        'level': 500,
        'lat_range': (20, 85),
        'lon_range': (160, 300),
        'lon_format': '360',
    },
    'AAO': {
        'level': 700,
        'lat_range': (-90, -20),
        'lon_range': (0, 360),
        'lon_format': '360',
    },
}


def standardize_coords(ds):
    """Standardize coordinate names to lat/lon/time/z."""
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


def slice_domain(ds, lat_range, lon_range, lon_format='180'):
    """Slice dataset to geographic domain, matching generate_eof.py logic."""
    if lon_format == '180':
        ds = ds.assign_coords(lon=(((ds.lon + 180) % 360) - 180)).sortby('lon')
    elif lon_format == '360':
        ds = ds.assign_coords(lon=(ds.lon % 360)).sortby('lon')

    lat_min, lat_max = sorted(lat_range)
    if ds.lat.values[0] > ds.lat.values[-1]:
        ds = ds.sel(lat=slice(lat_max, lat_min))
    else:
        ds = ds.sel(lat=slice(lat_min, lat_max))

    lon_min, lon_max = sorted(lon_range)
    ds = ds.sel(lon=slice(lon_min, lon_max))
    return ds


# ─── NAO / PNA / AAO ─────────────────────────────────────────────────────────

def compute_daily_pc_std_geopotential(index_name):
    """
    Compute daily_pc_std for NAO, PNA, or AAO by projecting daily ERA5
    anomalies onto the static EOF loading pattern.
    """
    params = INDICES[index_name]
    nc_path = os.path.join(INDICES_DIR, f"{index_name.lower()}_loading_pattern.nc")

    logging.info(f"[{index_name}] Loading EOF from {nc_path}")
    eof_ds = standardize_coords(xr.open_dataset(nc_path)).load()
    eof_pattern = eof_ds['eof']

    logging.info(f"[{index_name}] Opening historical ERA5 Zarr...")
    ds = standardize_coords(xr.open_zarr(HISTORICAL_ZARR, consolidated=True))

    # Select variable, level, time, domain — minimize data volume before loading
    da = ds['z'].sel(level=params['level'], method='nearest')
    da = da.sel(time=slice(*TIME_SLICE))
    # Subsample to 1 timestep per day (00z from 6-hourly data)
    da = da.isel(time=slice(0, None, 4))

    da_ds = da.to_dataset(name='z')
    da_sliced = slice_domain(da_ds, params['lat_range'], params['lon_range'], params['lon_format'])['z']

    logging.info(f"[{index_name}] Opening climatology Zarr...")
    clim_ds = standardize_coords(xr.open_zarr(CLIMATOLOGY_ZARR, consolidated=True))
    clim_da = clim_ds['z'].sel(level=params['level'], method='nearest')
    clim_da_ds = clim_da.to_dataset(name='z')
    clim_sliced = slice_domain(clim_da_ds, params['lat_range'], params['lon_range'], params['lon_format'])['z']

    logging.info(f"[{index_name}] Loading daily data into memory...")
    da_sliced = da_sliced.compute()

    logging.info(f"[{index_name}] Computing daily anomalies...")
    # Match each timestep to climatology by dayofyear and hour
    clim_grouped = clim_sliced.sel(dayofyear=da_sliced.time.dt.dayofyear,
                                    hour=da_sliced.time.dt.hour)
    anomalies = da_sliced - clim_grouped

    # Apply area weighting: sqrt(cos(lat)), identical to generate_eof.py L114
    logging.info(f"[{index_name}] Applying area weighting...")
    weights = np.sqrt(np.clip(np.cos(np.deg2rad(anomalies.lat)), 0, None))
    anomalies_weighted = anomalies * weights

    # Interpolate EOF to match anomaly grid if needed, then spatial dot product
    eof_interp = eof_pattern.interp(lat=anomalies_weighted.lat,
                                     lon=anomalies_weighted.lon,
                                     method='nearest')

    logging.info(f"[{index_name}] Computing spatial dot product...")
    pc_timeseries = (anomalies_weighted * eof_interp).sum(dim=['lat', 'lon'])

    daily_std = float(pc_timeseries.std(dim='time').values)
    monthly_std = float(eof_ds['pc_std'].values) if 'pc_std' in eof_ds else None

    logging.info(f"[{index_name}] daily_pc_std = {daily_std:.4f}")
    if monthly_std:
        logging.info(f"[{index_name}] monthly pc_std = {monthly_std:.4f} (ratio: {daily_std / monthly_std:.2f}x)")

    # Save back into the .nc file (write to temp file, then rename to avoid lock conflicts)
    logging.info(f"[{index_name}] Saving daily_pc_std to {nc_path}")
    eof_ds['daily_pc_std'] = xr.DataArray(daily_std)
    tmp_path = nc_path + '.tmp'
    eof_ds.to_netcdf(tmp_path)
    os.replace(tmp_path, nc_path)
    logging.info(f"[{index_name}] Done.")

    return daily_std


# ─── AO ───────────────────────────────────────────────────────────────────────

def compute_daily_pc_std_ao():
    """
    Compute daily_pc_std for AO by projecting daily ERA5 Z1000 NH anomalies
    onto the static AO EOF loading pattern.
    """
    nc_path = os.path.join(INDICES_DIR, 'ao_loading_pattern.nc')

    logging.info("[AO] Loading EOF from %s", nc_path)
    eof_ds = standardize_coords(xr.open_dataset(nc_path)).load()
    eof_pattern = eof_ds['eof']

    logging.info("[AO] Opening historical ERA5 Zarr...")
    ds = standardize_coords(xr.open_zarr(HISTORICAL_ZARR, consolidated=True))

    # AO domain: Z1000, 20°N to 90°N
    da = ds['z'].sel(level=1000, method='nearest')
    da = da.sel(time=slice(*TIME_SLICE))
    da = da.isel(time=slice(0, None, 4))  # Daily subsampling

    # Slice to NH (≥20°N)
    if da.lat.values[0] > da.lat.values[-1]:
        da = da.sel(lat=slice(90, 20))
    else:
        da = da.sel(lat=slice(20, 90))

    logging.info("[AO] Opening climatology Zarr...")
    clim_ds = standardize_coords(xr.open_zarr(CLIMATOLOGY_ZARR, consolidated=True))
    clim_da = clim_ds['z'].sel(level=1000, method='nearest')

    if clim_da.lat.values[0] > clim_da.lat.values[-1]:
        clim_da = clim_da.sel(lat=slice(90, 20))
    else:
        clim_da = clim_da.sel(lat=slice(20, 90))

    logging.info("[AO] Loading daily data into memory...")
    da = da.compute()

    logging.info("[AO] Computing daily anomalies...")
    clim_grouped = clim_da.sel(dayofyear=da.time.dt.dayofyear,
                                hour=da.time.dt.hour)
    anomalies = da - clim_grouped

    # Area weighting: sqrt(cos(lat)), matching generate_ao_eof.py L36-38
    logging.info("[AO] Applying area weighting...")
    weights = np.sqrt(np.clip(np.cos(np.deg2rad(anomalies.lat)), 0, None))
    anomalies_weighted = anomalies * weights

    # Interpolate EOF to match anomaly grid
    eof_interp = eof_pattern.interp(lat=anomalies_weighted.lat,
                                     lon=anomalies_weighted.lon,
                                     method='nearest')

    logging.info("[AO] Computing spatial dot product...")
    pc_timeseries = (anomalies_weighted * eof_interp).sum(dim=['lat', 'lon'])

    daily_std = float(pc_timeseries.std(dim='time').values)
    monthly_std = float(eof_ds['pc_std'].values) if 'pc_std' in eof_ds else None

    logging.info(f"[AO] daily_pc_std = {daily_std:.4f}")
    if monthly_std:
        logging.info(f"[AO] monthly pc_std = {monthly_std:.4f} (ratio: {daily_std / monthly_std:.2f}x)")

    logging.info("[AO] Saving daily_pc_std to %s", nc_path)
    eof_ds['daily_pc_std'] = xr.DataArray(daily_std)
    tmp_path = nc_path + '.tmp'
    eof_ds.to_netcdf(tmp_path)
    os.replace(tmp_path, nc_path)
    logging.info("[AO] Done.")

    return daily_std


# ─── MJO ──────────────────────────────────────────────────────────────────────

def compute_daily_pc_std_mjo():
    """
    Compute daily_pc1_std and daily_pc2_std for MJO by projecting daily ERA5
    anomalies onto the static combined EOF vectors and normalizing as in
    the generation script.
    """
    from windspharm.xarray import VectorWind

    nc_path = os.path.join(INDICES_DIR, 'mjo_loading_pattern.nc')

    logging.info("[MJO] Loading EOF from %s", nc_path)
    eof_ds = xr.open_dataset(nc_path).load()

    logging.info("[MJO] Opening historical ERA5 Zarr...")
    ds = standardize_coords(xr.open_zarr(HISTORICAL_ZARR, consolidated=True))
    ds = ds.sel(time=slice(*TIME_SLICE))
    ds = ds.isel(time=slice(0, None, 4))  # Daily subsampling

    # Extract wind variables
    if 'u' in ds.data_vars and 'level' in ds.coords:
        u200 = ds['u'].sel(level=200, method='nearest')
        u850 = ds['u'].sel(level=850, method='nearest')
    elif 'u_component_of_wind' in ds.data_vars:
        u200 = ds['u_component_of_wind'].sel(level=200, method='nearest')
        u850 = ds['u_component_of_wind'].sel(level=850, method='nearest')
    else:
        raise ValueError("Cannot find u-wind variable in dataset")

    if 'v' in ds.data_vars and 'level' in ds.coords:
        v200 = ds['v'].sel(level=200, method='nearest')
    elif 'v_component_of_wind' in ds.data_vars:
        v200 = ds['v_component_of_wind'].sel(level=200, method='nearest')
    else:
        raise ValueError("Cannot find v-wind variable in dataset")

    def slice_tropics(da):
        da = da.assign_coords(lon=(da.lon % 360)).sortby('lon')
        if da.lat.values[0] > da.lat.values[-1]:
            return da.sel(lat=slice(15, -15))
        else:
            return da.sel(lat=slice(-15, 15))

    # Process in chunks to avoid windspharm overflow
    chunk_size = 30
    n_time = len(u200.time)
    vp200_1d_list = []
    u850_1d_list = []
    u200_1d_list = []

    logging.info("[MJO] Processing %d timesteps in chunks of %d...", n_time, chunk_size)

    for start in range(0, n_time, chunk_size):
        end = min(start + chunk_size, n_time)
        u_chunk = u200.isel(time=slice(start, end)).load()
        v_chunk = v200.isel(time=slice(start, end)).load()
        u850_chunk = u850.isel(time=slice(start, end)).load()

        w = VectorWind(u_chunk, v_chunk)
        vp_chunk = w.velocitypotential()

        vp_tropics = slice_tropics(vp_chunk).mean(dim='lat')
        u850_tropics = slice_tropics(u850_chunk).mean(dim='lat')
        u200_tropics = slice_tropics(u_chunk).mean(dim='lat')

        # Standardize lon to 0-360
        vp_tropics = vp_tropics.assign_coords(lon=(vp_tropics.lon % 360)).sortby('lon')
        u850_tropics = u850_tropics.assign_coords(lon=(u850_tropics.lon % 360)).sortby('lon')
        u200_tropics = u200_tropics.assign_coords(lon=(u200_tropics.lon % 360)).sortby('lon')

        vp200_1d_list.append(vp_tropics)
        u850_1d_list.append(u850_tropics)
        u200_1d_list.append(u200_tropics)

        if (start // chunk_size) % 10 == 0:
            logging.info(f"[MJO] Processed chunk {start}/{n_time}")

    vp200_1d = xr.concat(vp200_1d_list, dim='time')
    u850_1d = xr.concat(u850_1d_list, dim='time')
    u200_1d = xr.concat(u200_1d_list, dim='time')

    # Compute anomalies using day-of-year climatology
    logging.info("[MJO] Computing anomalies...")

    def get_anomalies(da):
        clim = da.groupby('time.dayofyear').mean('time')
        return da.groupby('time.dayofyear') - clim

    vp200_anom = get_anomalies(vp200_1d)
    u850_anom = get_anomalies(u850_1d)
    u200_anom = get_anomalies(u200_1d)

    # Normalize by saved standard deviations from the EOF file
    logging.info("[MJO] Normalizing by saved standard deviations...")
    vp200_norm = vp200_anom / eof_ds['vp200_std']
    u850_norm = u850_anom / eof_ds['u850_std']
    u200_norm = u200_anom / eof_ds['u200_std']

    # Concatenate into combined vector
    n_lon = len(vp200_norm.lon)
    vp200_c = vp200_norm.rename({'lon': 'combined_lon'}).assign_coords(
        combined_lon=np.arange(n_lon)).drop_vars('level', errors='ignore')
    u850_c = u850_norm.rename({'lon': 'combined_lon'}).assign_coords(
        combined_lon=np.arange(n_lon) + n_lon).drop_vars('level', errors='ignore')
    u200_c = u200_norm.rename({'lon': 'combined_lon'}).assign_coords(
        combined_lon=np.arange(n_lon) + 2 * n_lon).drop_vars('level', errors='ignore')

    combined = xr.concat([vp200_c, u850_c, u200_c], dim='combined_lon')

    # Project onto EOFs
    logging.info("[MJO] Projecting onto EOF vectors...")
    rmm1 = (combined * eof_ds['eof1']).sum(dim='combined_lon')
    rmm2 = (combined * eof_ds['eof2']).sum(dim='combined_lon')

    daily_pc1_std = float(rmm1.std(dim='time').values)
    daily_pc2_std = float(rmm2.std(dim='time').values)

    monthly_pc1 = float(eof_ds['pc1_std'].values) if 'pc1_std' in eof_ds else None
    monthly_pc2 = float(eof_ds['pc2_std'].values) if 'pc2_std' in eof_ds else None

    logging.info(f"[MJO] daily_pc1_std = {daily_pc1_std:.4f}, daily_pc2_std = {daily_pc2_std:.4f}")
    if monthly_pc1 and monthly_pc2:
        logging.info(f"[MJO] monthly pc1_std = {monthly_pc1:.4f} (ratio: {daily_pc1_std / monthly_pc1:.2f}x)")
        logging.info(f"[MJO] monthly pc2_std = {monthly_pc2:.4f} (ratio: {daily_pc2_std / monthly_pc2:.2f}x)")

    logging.info("[MJO] Saving daily_pc1_std and daily_pc2_std to %s", nc_path)
    eof_ds['daily_pc1_std'] = xr.DataArray(daily_pc1_std)
    eof_ds['daily_pc2_std'] = xr.DataArray(daily_pc2_std)
    tmp_path = nc_path + '.tmp'
    eof_ds.to_netcdf(tmp_path)
    os.replace(tmp_path, nc_path)
    logging.info("[MJO] Done.")

    return daily_pc1_std, daily_pc2_std


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Compute daily PC standard deviations for oscillation indices.')
    parser.add_argument('--index', type=str, default='ALL',
                        choices=['NAO', 'PNA', 'AAO', 'AO', 'MJO', 'ALL'],
                        help='Which index to compute (default: ALL)')
    args = parser.parse_args()

    if args.index == 'ALL':
        targets = ['NAO', 'PNA', 'AAO', 'AO', 'MJO']
    else:
        targets = [args.index]

    results = {}

    for target in targets:
        try:
            if target in INDICES:
                results[target] = compute_daily_pc_std_geopotential(target)
            elif target == 'AO':
                results[target] = compute_daily_pc_std_ao()
            elif target == 'MJO':
                results[target] = compute_daily_pc_std_mjo()
        except Exception as e:
            logging.error(f"Failed to compute daily_pc_std for {target}: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    logging.info("\n=== Summary ===")
    for name, val in results.items():
        if isinstance(val, tuple):
            logging.info(f"  {name}: daily_pc1_std={val[0]:.4f}, daily_pc2_std={val[1]:.4f}")
        else:
            logging.info(f"  {name}: daily_pc_std={val:.4f}")


if __name__ == '__main__':
    main()
