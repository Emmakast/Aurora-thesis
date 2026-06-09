import xarray as xr
import numpy as np

da = xr.DataArray(np.zeros((4, 2)), dims=["hour", "x"], coords={"hour": [0, 6, 12, 18]})
t_val = np.datetime64('2017-01-06T12:00:00')
t_dt = xr.DataArray(t_val).dt

print("1. Indexer dims:", t_dt.hour.dims)
print("1. Indexer name:", t_dt.hour.name)
res = da.sel(hour=t_dt.hour)
print("1. Result dims:", res.dims)

t_dt_named = xr.DataArray(t_val, name="hour").dt
print("2. Indexer dims:", t_dt_named.hour.dims)
print("2. Indexer name:", t_dt_named.hour.name)
res2 = da.sel(hour=t_dt_named.hour)
print("2. Result dims:", res2.dims)

