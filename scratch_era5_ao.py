import xarray as xr
import pandas as pd
import numpy as np
import gc

eof_ds = xr.open_dataset("/home/ekasteleyn/aurora_thesis/thesis/steering/scripts/oscillation_calculator/indices/ao_loading_pattern.nc")
eof_pattern = eof_ds['eof']
pc_std = float(eof_ds['daily_pc_std'].values) if 'daily_pc_std' in eof_ds else float(eof_ds['pc_std'].values)

era5_ds = xr.open_zarr("gs://weatherbench2/datasets/era5/1959-2022-6h-1440x721.zarr", consolidated=True)
clim_ds = xr.open_zarr("gs://weatherbench2/datasets/era5-hourly-climatology/1990-2017_6h_1440x721.zarr", consolidated=True)

valid_times = pd.date_range("2017-03-08 12:00:00", periods=41, freq='6h')
era5_vals = []

for t in valid_times:
    try:
        t_da = era5_ds['geopotential'].sel(time=t, level=1000, method='nearest')
        if t_da.latitude.values[0] > t_da.latitude.values[-1]:
            t_da = t_da.sel(latitude=slice(90, 20))
        else:
            t_da = t_da.sel(latitude=slice(20, 90))
        t_da = t_da.compute()
            
        c_da = clim_ds['geopotential'].sel(level=1000, dayofyear=t.dayofyear, hour=t.hour, method='nearest')
        if c_da.latitude.values[0] > c_da.latitude.values[-1]:
            c_da = c_da.sel(latitude=slice(90, 20))
        else:
            c_da = c_da.sel(latitude=slice(20, 90))
        c_da = c_da.compute()
            
        anom = t_da - c_da
        weights = np.sqrt(np.clip(np.cos(np.deg2rad(anom.latitude)), 0, None))
        anom_w = anom * weights
        
        if 'lat' in eof_pattern.coords and 'latitude' in anom_w.coords:
            eof_renamed = eof_pattern.rename({'lat': 'latitude', 'lon': 'longitude'})
            proj = (anom_w * eof_renamed).sum()
        else:
            proj = (anom_w * eof_pattern).sum()
            
        era5_vals.append(float(proj / pc_std))
        
        del t_da, c_da, anom, anom_w, weights
        gc.collect()
    except Exception as e:
        print(f"Error on {t}: {e}")
        era5_vals.append(0.0)

df = pd.DataFrame({'time': valid_times, 'era5_ao': era5_vals})
df.to_csv("era5_ao_reference.csv", index=False)
print("Saved era5_ao_reference.csv")
