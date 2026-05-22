import argparse
import logging
import os
import xarray as xr
import numpy as np

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
    ds = ds.assign_coords(lon=(ds.lon % 360)).sortby('lon')
    
    lat_min, lat_max = -15, 15
    if ds.lat.values[0] > ds.lat.values[-1]:
        ds = ds.sel(lat=slice(lat_max, lat_min))
    else:
        ds = ds.sel(lat=slice(lat_min, lat_max))
    return ds

def extract_variables(ds):
    vars_dict = {}
    for v in ['olr', 'OLR', 'ttr', 'rlut', 'top_net_thermal_radiation', 'toa_outgoing_longwave_flux', 'mean_top_net_long_wave_radiation_flux']:
        if v in ds.data_vars:
            vars_dict['olr'] = ds[v]
            break
            
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
        
    return vars_dict

def calculate_mjo(target_file, climatology_file, eof_file):
    logging.info(f"Loading target file: {target_file}")
    target_ds = standardize_coords(xr.open_dataset(target_file))
    
    logging.info(f"Loading climatology file: {climatology_file}")
    clim_ds = standardize_coords(xr.open_dataset(climatology_file))
    
    logging.info(f"Loading EOF pattern file: {eof_file}")
    eof_ds = xr.open_dataset(eof_file)
    
    target_sliced = slice_tropics(target_ds)
    clim_sliced = slice_tropics(clim_ds)
    
    target_vars = extract_variables(target_sliced)
    clim_vars = extract_variables(clim_sliced)
    
    combined_list = []
    
    for var_key in ['olr', 'u850', 'u200']:
        t_var = target_vars[var_key]
        c_var = clim_vars[var_key]
        
        # 1. Calculate anomalies
        if 'time' in t_var.coords:
            anom_list = []
            t_var_expanded = t_var if 'time' in t_var.dims else t_var.expand_dims('time')
            
            for t in t_var_expanded.time:
                t_val = t.values
                t_dt = xr.DataArray(t_val).dt
                
                c_slice = c_var
                if 'dayofyear' in c_slice.coords:
                    c_slice = c_slice.sel(dayofyear=t_dt.dayofyear)
                elif 'month' in c_slice.coords:
                    c_slice = c_slice.sel(month=t_dt.month)
                
                anom_list.append(t_var_expanded.sel(time=t) - c_slice)
                
            anom = xr.concat(anom_list, dim='time')
            if 'time' not in t_var.dims:
                anom = anom.squeeze('time')
        else:
            anom = t_var - c_var
            
        # 2. Meridionally average (15N-15S)
        anom_1d = anom.mean(dim='lat')
        
        # 3. Normalize with saved standard deviations
        std_val = eof_ds[f"{var_key}_std"]
        anom_norm = anom_1d / std_val
        
        combined_list.append(anom_norm)
        
    # 4. Concatenate and project
    olr_norm, u850_norm, u200_norm = combined_list
    n_lon = len(olr_norm.lon)
    
    olr_c = olr_norm.rename({'lon': 'combined_lon'}).assign_coords(combined_lon=np.arange(n_lon)).drop_vars('level', errors='ignore')
    u850_c = u850_norm.rename({'lon': 'combined_lon'}).assign_coords(combined_lon=np.arange(n_lon) + n_lon).drop_vars('level', errors='ignore')
    u200_c = u200_norm.rename({'lon': 'combined_lon'}).assign_coords(combined_lon=np.arange(n_lon) + 2*n_lon).drop_vars('level', errors='ignore')
    
    combined = xr.concat([olr_c, u850_c, u200_c], dim='combined_lon')
    
    eof1 = eof_ds['eof1']
    eof2 = eof_ds['eof2']
    
    rmm1 = (combined * eof1).sum(dim='combined_lon')
    rmm2 = (combined * eof2).sum(dim='combined_lon')
    
    # 5. Calculate MJO Phase: arctan2(RMM2, RMM1) (mapped to phases 1-8)
    # Using standard angle mapping starting at 0 deg, with 45 degree phase bins
    angle = np.arctan2(rmm2, rmm1) * 180.0 / np.pi
    angle = (angle + 360) % 360
    
    mjo_phase = np.floor(((angle + 22.5) % 360) / 45.0) + 1
    
    # 6. Calculate MJO Amplitude
    mjo_amp = np.sqrt(rmm1**2 + rmm2**2)
    
    logging.info(f"RMM1: \n{rmm1.values}")
    logging.info(f"RMM2: \n{rmm2.values}")
    logging.info(f"MJO Phase (1-8): \n{mjo_phase.values}")
    logging.info(f"MJO Amplitude: \n{mjo_amp.values}")
    
    return {
        'rmm1': rmm1,
        'rmm2': rmm2,
        'phase': mjo_phase,
        'amplitude': mjo_amp
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate MJO RMM Indices.')
    parser.add_argument('--target', type=str, required=True, help='Path to target NetCDF file (forecast/state)')
    parser.add_argument('--climatology', type=str, required=True, help='Path to climatology NetCDF file')
    parser.add_argument('--eof_file', type=str, required=True, help='Path to static MJO EOF NetCDF file')
    
    args = parser.parse_args()
    calculate_mjo(args.target, args.climatology, args.eof_file)
