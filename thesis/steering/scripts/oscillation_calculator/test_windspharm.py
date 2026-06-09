import xarray as xr
import numpy as np
from windspharm.xarray import VectorWind

ds = xr.open_dataset("/home/ekasteleyn/aurora_thesis/thesis/results/base_enso_20170103_1200_alpha_0.0.nc")
if 'latitude' in ds.coords:
    ds = ds.rename({'latitude': 'lat'})
if 'longitude' in ds.coords:
    ds = ds.rename({'longitude': 'lon'})
    
print("original lat shape:", ds.lat.shape)
new_lat = np.linspace(90, -90, 721)
ds = ds.interp(lat=new_lat, kwargs={'fill_value': 'extrapolate'})
print("new lat shape:", ds.lat.shape)

if 'u200' in ds.data_vars:
    u200 = ds['u200']
    v200 = ds['v200']
elif 'u' in ds.data_vars:
    u200 = ds['u'].sel(level=200, method='nearest')
    v200 = ds['v'].sel(level=200, method='nearest')
elif 'u_component_of_wind' in ds.data_vars:
    u200 = ds['u_component_of_wind'].sel(level=200, method='nearest')
    v200 = ds['v_component_of_wind'].sel(level=200, method='nearest')
    
if 'time' in u200.dims:
    u200 = u200.isel(time=-1)
    v200 = v200.isel(time=-1)
    
w_t = VectorWind(u200, v200)
vp = w_t.velocitypotential()
print("Success! VP shape:", vp.shape)
