import os
import re
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import logging

logging.getLogger().setLevel(logging.ERROR)

from calculate_3_indices import calculate_indices
try:
    from calculate_mjo import calculate_mjo
except ImportError:
    calculate_mjo = None

def standardize_coords(ds: xr.Dataset) -> xr.Dataset:
    rename_dict = {}
    if 'latitude' in ds.coords: rename_dict['latitude'] = 'lat'
    if 'longitude' in ds.coords: rename_dict['longitude'] = 'lon'
    if 'valid_time' in ds.coords: rename_dict['valid_time'] = 'time'
    if 'geopotential' in ds.variables: rename_dict['geopotential'] = 'z'
    return ds.rename(rename_dict) if rename_dict else ds

def slice_nino34(ds):
    lat_min, lat_max = -5, 5
    if ds.lat.values[0] > ds.lat.values[-1]: ds = ds.sel(lat=slice(lat_max, lat_min))
    else: ds = ds.sel(lat=slice(lat_min, lat_max))
    ds = ds.assign_coords(lon=(ds.lon % 360)).sortby('lon')
    return ds.sel(lon=slice(190, 240))

def get_anomaly_ao(ds: xr.Dataset, clim: xr.Dataset, var: str, level: int, doy: int, hour: int) -> xr.DataArray:
    data = ds[var].sel(level=level)
    if 'dayofyear' in clim.coords and 'hour' in clim.coords:
        clim_slice = clim[var].sel(level=level, dayofyear=doy, hour=hour, method='nearest')
    else:
        clim_slice = clim[var].sel(level=level).groupby('time.dayofyear')[doy].mean('time')
    clim_interp = clim_slice.interp(lat=data.lat, lon=data.lon, method='linear')
    return data - clim_interp

def calculate_ao(ds, ds_clim, eof_pattern, pc1_std):
    if 'time' in ds.coords:
        valid_times = ds['time'].values
        target_time = pd.to_datetime(valid_times[-1])
    else:
        target_time = pd.Timestamp("2020-01-01") # Dummy fallback
    
    anom_1000 = get_anomaly_ao(ds, ds_clim, 'z', 1000, target_time.dayofyear, target_time.hour).squeeze()
    anom_nh = anom_1000.where(anom_1000.lat >= 20, drop=True)
    eof_nh = eof_pattern.interp(lat=anom_nh.lat, lon=anom_nh.lon, method='nearest').where(anom_nh.lat >= 20, drop=True)
    
    lat_weights = np.sqrt(np.clip(np.cos(np.deg2rad(anom_nh.lat)), 0, None))
    anom_weighted = anom_nh * lat_weights
    raw_proj = (anom_weighted * eof_nh).sum(dim=['lat', 'lon']).values
    return float(raw_proj / pc1_std)

def calculate_enso_inline(target_ds, clim_sliced):
    target_var = slice_nino34(target_ds)
    if '2t' in target_var: target_var = target_var['2t']
    elif '2m_temperature' in target_var: target_var = target_var['2m_temperature']
    elif 't2m' in target_var: target_var = target_var['t2m']
    
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
    valid_vals = arr[~np.isnan(arr)]
    val = float(valid_vals[-1]) if len(valid_vals) > 0 else float(arr[-1])
    return val

def main():
    parser = argparse.ArgumentParser(description='Evaluate indices')
    parser.add_argument('--AO', action='store_true', help='Evaluate AO')
    parser.add_argument('--NAO', action='store_true', help='Evaluate NAO')
    parser.add_argument('--PNA', action='store_true', help='Evaluate PNA')
    parser.add_argument('--MJO', action='store_true', help='Evaluate MJO')
    parser.add_argument('--AAO', action='store_true', help='Evaluate AAO')
    parser.add_argument('--date', type=str, default=None, help='Filter by specific date (e.g. 20160123_1200)')
    parser.add_argument('--negative_only', action='store_true', help='Only evaluate negative alphas and base state')
    args = parser.parse_args()

    run_all = not any([args.AO, args.NAO, args.PNA, args.MJO, args.AAO])

    print("Loading global climatology (this takes ~1 min)...")
    clim_ds = standardize_coords(xr.open_zarr("gs://weatherbench2/datasets/era5-hourly-climatology/1990-2017_6h_1440x721.zarr", consolidated=True))
    
    # Pre-slice for ENSO
    clim_enso = slice_nino34(clim_ds)['2m_temperature']
    
    # Load AO pattern
    ao_ds = standardize_coords(xr.open_dataset("/home/ekasteleyn/aurora_thesis/thesis/steering/scripts/oscillation_calculator/indices/ao_loading_pattern.nc"))
    ao_pattern = ao_ds['eof'].squeeze()
    ao_std = float(ao_ds['pc_std'].values)
    
    # MJO EOF
    mjo_eof_path = Path("/home/ekasteleyn/aurora_thesis/thesis/steering/scripts/oscillation_calculator/indices/mjo_loading_pattern.nc")
    if mjo_eof_path.exists():
        mjo_eof = xr.open_dataset(mjo_eof_path)
    else:
        print("MJO EOF not found, skipping MJO index calculations.")
        mjo_eof = None
    
    EOFS_DIR = "/home/ekasteleyn/aurora_thesis/thesis/steering/scripts/oscillation_calculator/indices"
    
    directories = []
    if run_all: directories.append("/home/ekasteleyn/aurora_thesis/thesis/results")
    if run_all or args.AAO: directories.append("/home/ekasteleyn/aurora_thesis/thesis/steering/vectors/AAO_1encoder(2)")
    if run_all or args.NAO: directories.append("/scratch-shared/ekasteleyn/nao_steered")
    if run_all or args.PNA: directories.append("/scratch-shared/ekasteleyn/pna_neutral_steered")
    if run_all or args.AO: directories.append("/scratch-shared/ekasteleyn/ao_neutral_steered")
    if run_all or args.MJO: directories.append("/scratch-shared/ekasteleyn/mjo_steered")
    
    for directory in directories:
        dir_path = Path(directory)
        if not dir_path.exists():
            continue
            
        print(f"\n======================================")
        print(f"Evaluating all indices for {dir_path.name}")
        print(f"======================================")
        
        nc_files = list(dir_path.glob("*.nc"))
        if not nc_files:
            continue
            
        results = []
        for nc_file in sorted(nc_files):
            filename = nc_file.name
            
            if args.date and args.date not in filename:
                continue
                
            is_base = "base_" in filename
            is_steered = "steered_" in filename
            if not is_base and not is_steered:
                continue
                
            alpha_match = re.search(r'alpha_(-?\d+\.?\d*)', filename)
            alpha = float(alpha_match.group(1)) if alpha_match else (0.0 if is_base else None)
            
            if args.negative_only and alpha is not None and alpha > 0:
                continue
            
            print(f"  -> {filename}")
            try:
                target_ds = standardize_coords(xr.open_dataset(nc_file))
                if 'time' not in target_ds.coords:
                    match = re.search(r'(\d{8})_(\d{4})', filename)
                    if match:
                        from datetime import datetime
                        init_time = datetime.strptime(f"{match.group(1)}{match.group(2)}", "%Y%m%d%H%M")
                        target_time = pd.to_datetime(init_time) + pd.Timedelta(hours=72)
                        target_ds = target_ds.expand_dims(time=[target_time])
                
                
                def extract_val(arr):
                    if 'mode' in arr.dims:
                        arr = arr.sel(mode=0)
                    if 'time' in arr.dims:
                        arr = arr.isel(time=-1)
                    if arr.ndim > 0:
                        return float(arr.mean().values)
                    return float(arr.values)

                target_ds_last = target_ds.isel(time=[-1]) if 'time' in target_ds.dims else target_ds

                # 1. NAO, PNA, AAO
                res_3 = calculate_indices(target_ds_last, clim_ds, EOFS_DIR)
                val_nao = extract_val(res_3["NAO"])
                val_pna = extract_val(res_3["PNA"])
                val_aao = extract_val(res_3["AAO"])
                
                # 2. AO
                val_ao = calculate_ao(target_ds_last, clim_ds, ao_pattern, ao_std)
                
                # 3. MJO
                if mjo_eof is not None and calculate_mjo is not None:
                    mjo_res = calculate_mjo(target_ds_last, clim_ds, mjo_eof)
                    val_mjo = float(np.array(mjo_res['amplitude'].values).flatten()[-1])
                else:
                    val_mjo = np.nan
                
                # 4. ENSO
                val_enso = calculate_enso_inline(target_ds, clim_enso)
                
                results.append({
                    "Filename": filename,
                    "Type": "Base" if is_base else "Steered",
                    "Alpha": alpha,
                    "NAO": val_nao,
                    "PNA": val_pna,
                    "AAO": val_aao,
                    "AO": val_ao,
                    "MJO": val_mjo,
                    "ENSO": val_enso
                })
            except Exception as e:
                print(f"Error evaluating {filename}: {e}")
                import traceback
                traceback.print_exc()
                
        if results:
            df = pd.DataFrame(results)
            out_csv = dir_path / f"all_indices_evaluated.csv"
            
            if out_csv.exists():
                existing_df = pd.read_csv(out_csv)
                df = pd.concat([existing_df, df]).drop_duplicates(subset=['Filename'], keep='last')
                
            df = df.sort_values(by=["Type", "Alpha", "Filename"])
            print(df.to_markdown(index=False))
            df.to_csv(out_csv, index=False)
            print(f"Saved {out_csv}")
            
            # Move to normal directory
            normal_dir = Path("/home/ekasteleyn/aurora_thesis/thesis/results")
            normal_dir.mkdir(parents=True, exist_ok=True)
            normal_csv = normal_dir / "all_indices_evaluated.csv"
            
            if normal_csv.exists():
                existing_normal_df = pd.read_csv(normal_csv)
                df_normal = pd.concat([existing_normal_df, df]).drop_duplicates(subset=['Filename'], keep='last')
                df_normal = df_normal.sort_values(by=["Type", "Alpha", "Filename"])
                df_normal.to_csv(normal_csv, index=False)
            else:
                df.to_csv(normal_csv, index=False)
                
            print(f"Also appended results to the master CSV at {normal_csv}")

if __name__ == "__main__":
    main()
