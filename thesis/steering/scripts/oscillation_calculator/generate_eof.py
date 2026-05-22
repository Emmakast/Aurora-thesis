import argparse
import logging
import os
import xarray as xr
import numpy as np
from eofs.xarray import Eof

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def slice_domain(ds, lat_range, lon_range, lon_format='180'):
    """Slice dataset optimally depending on the region to avoid discontinuous longitude arrays."""
    # Standardize longitude
    if lon_format == '180':
        ds = ds.assign_coords(lon=(((ds.lon + 180) % 360) - 180)).sortby('lon')
    elif lon_format == '360':
        ds = ds.assign_coords(lon=(ds.lon % 360)).sortby('lon')
    
    # Slice latitude based on array ordering
    lat_min, lat_max = sorted(lat_range)
    if ds.lat.values[0] > ds.lat.values[-1]:
        ds = ds.sel(lat=slice(lat_max, lat_min))
    else:
        ds = ds.sel(lat=slice(lat_min, lat_max))
        
    # Slice longitude
    lon_min, lon_max = sorted(lon_range)
    ds = ds.sel(lon=slice(lon_min, lon_max))
    
    return ds

def generate_eofs(input_zarr, output_dir):
    logging.info(f"Loading Zarr store from {input_zarr}")
    ds = xr.open_zarr(input_zarr)
    
    var_name = 'z'
    if var_name not in ds.data_vars:
        logging.warning(f"'{var_name}' not found in data vars. Available vars: {list(ds.data_vars.keys())}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    indices = {
        'NAO': {
            'level': 500,
            'lat_range': (20, 90),
            'lon_range': (-90, 40),
            'lon_format': '180',
            'check_lat': 65,
            'check_lon': -20,
            'expected_sign': -1
        },
        'PNA': {
            'level': 500,
            'lat_range': (20, 85),
            'lon_range': (160, 300),  # 160E to 60W (360 - 60 = 300)
            'lon_format': '360',
            'check_lat': 50,
            'check_lon': 200,  # 160W = 200E
            'expected_sign': -1
        },
        'AAO': {
            'level': 700,
            'lat_range': (-90, -20),
            'lon_range': (0, 360),
            'lon_format': '360',
            'check_lat': -90,
            'check_lon': 0,
            'expected_sign': -1
        }
    }
    
    for name, params in indices.items():
        logging.info(f"Processing {name}...")
        
        ds_subset = ds
        if 'level' in ds_subset.coords:
            ds_subset = ds_subset.sel(level=params['level'], method='nearest')
            
        # Slice domain
        ds_sliced = slice_domain(ds_subset, params['lat_range'], params['lon_range'], params['lon_format'])
        
        # Resample to monthly means ('1MS') to isolate low-frequency modes
        logging.info(f"  Resampling {name} to monthly means...")
        ds_monthly = ds_sliced.resample(time='1MS').mean()
        
        # Calculate monthly anomalies
        logging.info(f"  Calculating anomalies for {name}...")
        climatology = ds_monthly.groupby('time.month').mean('time')
        anomalies = ds_monthly.groupby('time.month') - climatology
        anomalies = anomalies[var_name].compute()
        
        # Area Weighting
        # You MUST weight the data before feeding it to the eofs solver
        logging.info(f"  Applying area weighting for {name}...")
        weights = np.sqrt(np.clip(np.cos(np.deg2rad(anomalies.lat)), 0, None))
        anomalies_weighted = anomalies * weights
        
        logging.info(f"  Computing EOF for {name}...")
        solver = Eof(anomalies_weighted)
        
        # Retrieve EOF1 and PC1
        eof1 = solver.eofs(neofs=1).squeeze()
        pc1 = solver.pcs(npcs=1).squeeze()
        
        # Extract the standard deviation of PC1
        pc_std = pc1.std(dim='time')
        
        # Check polarity
        check_point = eof1.sel(lat=params['check_lat'], lon=params['check_lon'], method='nearest')
        if np.sign(check_point.values) != np.sign(params['expected_sign']):
            logging.info(f"  Reversing polarity for {name}...")
            eof1 = eof1 * -1
            
        # Create output dataset
        out_ds = xr.Dataset(
            {
                'eof': eof1,
                'pc_std': pc_std
            }
        )
        
        out_path = os.path.join(output_dir, f"{name.lower()}_loading_pattern.nc")
        out_ds.to_netcdf(out_path)
        logging.info(f"  Saved {name} to {out_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate EOF loading patterns for atmospheric indices.')
    parser.add_argument('--input', type=str, required=True, help='Path to input Zarr store (e.g., ERA5)')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the generated NetCDF files')
    
    args = parser.parse_args()
    generate_eofs(args.input, args.output_dir)
