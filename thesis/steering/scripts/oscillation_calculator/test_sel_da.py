import xarray as xr
import numpy as np
import pandas as pd

da = xr.DataArray(np.zeros((4, 2)), dims=["hour", "x"], coords={"hour": [0, 6, 12, 18]})
t_val = np.datetime64('2017-01-06T12:00:00')
t_dt = xr.DataArray(t_val).dt

print("Indexer dims:", t_dt.hour.dims)
res = da.sel(hour=t_dt.hour)
print("Result dims:", res.dims)
