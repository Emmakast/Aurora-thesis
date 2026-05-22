import argparse
import logging
import os
import xarray as xr
import numpy as np
from eofs.xarray import Eof

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def standardize_coords(ds: xr.Dataset) -> xr.Dataset:
    """Standardize latitude and longitude coordinate names."""
    rename_dict = {}
    if 'latitude' in ds.coords:
        rename_dict['latitude'] = 'lat'
    if 'longitude' in ds.coords:
        rename_dict['longitude'] = 'lon'
    if 'valid_time' in ds.coords:
        rename_dict['valid_time'] = 'time'
    return ds.rename(rename_dict) if rename_dict else ds

def slice_tropics(ds):
    """Slice to 15°N-15°S and ensure longitudes are 0-360 continuously."""
    # Standardize longitude to 0-360 for consistency in MJO EOF vector
    ds = ds.assign_coords(lon=(ds.lon % 360)).sortby('lon')
    
    # Domain: 15°N to 15°S
    lat_min, lat_max = -15, 15
    if ds.lat.values[0] > ds.lat.values[-1]:
        ds = ds.sel(lat=slice(lat_max, lat_min))
    else:
        ds = ds.sel(lat=slice(lat_min, lat_max))
    return ds

def extract_variables(ds):
    """Attempt to extract OLR, U850, and U200 from dataset."""
    vars_dict = {}
    
    # OLR
    for v in ['olr', 'OLR', 'ttr', 'rlut', 'top_net_thermal_radiation', 'toa_outgoing_longwave_flux', 'mean_top_net_long_wave_radiation_flux']:
        if v in ds.data_vars:
            vars_dict['olr'] = ds[v]
            break
            
    # U850 and U200
    if 'u850' in ds.data_vars:
        vars_dict['u850'] = ds['u850']
    elif 'u' in ds.data_vars and 'level' in ds.coords:
        vars_dict['u850'] = ds['u'].sel(level=850, method='nearest')
    elif 'u_component_of_wind' in ds.data_vars and 'level' in ds.coords:
        vars_dict['u850'] = ds['u_component_of_wind'].sel(level=850, method='nearest')
        
    if 'u200' in ds.data_vars:
        vars_dict['u200'] = ds['u200']
    elif 'u' in ds.data_vars and 'level' in ds.coords:
        vars_dict['u200'] = ds['u'].sel(level=200, method='nearest')
    elif 'u_component_of_wind' in ds.data_vars and 'level' in ds.coords:
        vars_dict['u200'] = ds['u_component_of_wind'].sel(level=200, method='nearest')
        
    if len(vars_dict) < 3:
        logging.warning(f"Could not find all required variables. Found: {list(vars_dict.keys())}")
        
    return vars_dict

def generate_mjo_eofs(input_zarr, output_dir):
    logging.info(f"Loading Zarr store from {input_zarr}")
    ds = xr.open_zarr(input_zarr)
    ds = standardize_coords(ds)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Slice data to 15°N-15°S.
    logging.info("Slicing tropical domain...")
    ds_tropics = slice_tropics(ds)
    vars_dict = extract_variables(ds_tropics)
    
    olr = vars_dict['olr']
    u850 = vars_dict['u850']
    u200 = vars_dict['u200']
    
    # Calculate anomalies against daily climatology
    logging.info("Calculating anomalies...")
    def get_anomalies(da):
        clim = da.groupby('time.dayofyear').mean('time')
        return da.groupby('time.dayofyear') - clim
        
    olr_anom = get_anomalies(olr)
    u850_anom = get_anomalies(u850)
    u200_anom = get_anomalies(u200)
    
    # 2. Average each variable meridionally (across latitude)
    logging.info("Meridional averaging...")
    olr_1d = olr_anom.mean(dim='lat')
    u850_1d = u850_anom.mean(dim='lat')
    u200_1d = u200_anom.mean(dim='lat')
    
    # 3. CRITICAL: Normalize each of the three 1D arrays by dividing by their global longitudinal standard deviations.
    logging.info("Normalizing by global standard deviations...")
    olr_std = olr_1d.std(dim=['time', 'lon'])
    u850_std = u850_1d.std(dim=['time', 'lon'])
    u200_std = u200_1d.std(dim=['time', 'lon'])
    
    olr_norm = olr_1d / olr_std
    u850_norm = u850_1d / u850_std
    u200_norm = u200_1d / u200_std
    
    # 4. Concatenate the three normalized 1D arrays along the longitude axis
    logging.info("Concatenating into combined vector...")
    n_lon = len(olr_norm.lon)
    olr_c = olr_norm.rename({'lon': 'combined_lon'}).assign_coords(combined_lon=np.arange(n_lon)).drop_vars('level', errors='ignore')
    u850_c = u850_norm.rename({'lon': 'combined_lon'}).assign_coords(combined_lon=np.arange(n_lon) + n_lon).drop_vars('level', errors='ignore')
    u200_c = u200_norm.rename({'lon': 'combined_lon'}).assign_coords(combined_lon=np.arange(n_lon) + 2*n_lon).drop_vars('level', errors='ignore')
    
    combined = xr.concat([olr_c, u850_c, u200_c], dim='combined_lon')
    
    # Call compute to load into memory for Eof solver
    combined = combined.compute()
    
    # 5. Run the EOF solver on this combined vector.
    logging.info("Running EOF solver...")
    solver = Eof(combined)
    
    eofs = solver.eofs(neofs=2).squeeze()
    pcs = solver.pcs(npcs=2).squeeze()
    
    eof1 = eofs.sel(mode=0).drop_vars('mode')
    eof2 = eofs.sel(mode=1).drop_vars('mode')
    
    # 6. Save the EOF patterns and the normalization factors
    logging.info("Saving results...")
    out_ds = xr.Dataset({
        'eof1': eof1,
        'eof2': eof2,
        'olr_std': olr_std,
        'u850_std': u850_std,
        'u200_std': u200_std
    })
    
    out_path = os.path.join(output_dir, "mjo_loading_pattern.nc")
    out_ds.to_netcdf(out_path)
    logging.info(f"Saved MJO EOFs to {out_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Combined EOF loading patterns for MJO (RMM).')
    parser.add_argument('--input', type=str, required=True, help='Path to input Zarr store (e.g., ERA5)')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the generated NetCDF files')
    
    args = parser.parse_args()
    generate_mjo_eofs(args.input, args.output_dir)
