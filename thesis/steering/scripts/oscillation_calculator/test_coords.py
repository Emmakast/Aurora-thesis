import xarray as xr
import pandas as pd

ds = xr.Dataset(
    {"z": (("lat", "lon"), [[1, 2], [3, 4]])},
    coords={"lat": [10, 20], "lon": [30, 40]}
)

ds = ds.assign_coords(time=[pd.to_datetime("2020-01-01")])
print("ds coords:", list(ds.coords.keys()))
print("ds['z'] coords:", list(ds['z'].coords.keys()))
