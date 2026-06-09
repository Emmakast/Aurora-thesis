import xarray as xr
import numpy as np
from windspharm.xarray import VectorWind
import logging
from calculate_mjo import standardize_coords, slice_tropics

logging.basicConfig(level=logging.INFO)

# 1. Load data
ds = xr.open_zarr("gs://weatherbench2/datasets/era5-hourly-climatology/1990-2017_6h_1440x721.zarr", consolidated=True)
ds = standardize_coords(ds)

# Select 1 year but only 1 timestep per day to save memory, or just 1 season
ds = ds.sel(time=slice('1990-01-01', '1995-12-31')).isel(time=slice(0, None, 4)) # Daily data
ds = ds.load()

# 2. Extract variables
u200 = ds['u_component_of_wind'].sel(level=200, method='nearest')
v200 = ds['v_component_of_wind'].sel(level=200, method='nearest')
u850 = ds['u_component_of_wind'].sel(level=850, method='nearest')

# 3. Compute VP
w = VectorWind(u200, v200)
vp200 = w.velocitypotential()

# 4. Slice tropics and average lat
vp200_1d = slice_tropics(vp200).mean('lat')
u850_1d = slice_tropics(u850).mean('lat')
u200_1d = slice_tropics(u200).mean('lat')

# 5. Anomalies
clim_vp = vp200_1d.groupby('time.dayofyear').mean('time')
vp200_anom = vp200_1d.groupby('time.dayofyear') - clim_vp

clim_u850 = u850_1d.groupby('time.dayofyear').mean('time')
u850_anom = u850_1d.groupby('time.dayofyear') - clim_u850

clim_u200 = u200_1d.groupby('time.dayofyear').mean('time')
u200_anom = u200_1d.groupby('time.dayofyear') - clim_u200

# 6. Load EOF
eof_ds = xr.open_dataset('indices/mjo_loading_pattern.nc')

# 7. Normalize
vp200_norm = vp200_anom / eof_ds['vp200_std']
u850_norm = u850_anom / eof_ds['u850_std']
u200_norm = u200_anom / eof_ds['u200_std']

# 8. Combine
n_lon = len(vp200_norm.lon)
vp200_c = vp200_norm.rename({'lon': 'combined_lon'}).assign_coords(combined_lon=np.arange(n_lon))
u850_c = u850_norm.rename({'lon': 'combined_lon'}).assign_coords(combined_lon=np.arange(n_lon) + n_lon)
u200_c = u200_norm.rename({'lon': 'combined_lon'}).assign_coords(combined_lon=np.arange(n_lon) + 2*n_lon)

combined = xr.concat([vp200_c, u850_c, u200_c], dim='combined_lon')

# 9. Project
rmm1 = (combined * eof_ds['eof1']).sum(dim='combined_lon')
rmm2 = (combined * eof_ds['eof2']).sum(dim='combined_lon')

print("RMM1 std:", rmm1.std().values)
print("RMM2 std:", rmm2.std().values)

# Write to file
with open('pc_stds.txt', 'w') as f:
    f.write(f"{rmm1.std().values},{rmm2.std().values}")

