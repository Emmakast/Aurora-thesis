
import xarray as xr
import numpy as np

print("Opening Zarr...")
ds = xr.open_zarr('gs://weatherbench2/datasets/aurora/2022-1440x721.zarr', storage_options={'token': 'anon'})
rename = {v: v.strip() for v in ds.data_vars if v != v.strip()}
ds = ds.rename(rename)

print(f"Levels: {ds.level.values}")

print("Loading 500 hPa geopotential at latitude 90.0...")
try:
    z_pole = ds['geopotential'].sel(level=500, latitude=90.0).isel(time=0, prediction_timedelta=0).load()
    
    vals = z_pole.values
    variance = np.var(vals)
    ptp = np.ptp(vals)
    
    print(f"Min: {vals.min()}")
    print(f"Max: {vals.max()}")
    print(f"Variance at 90.0: {variance}")
    print(f"Peak-to-peak at 90.0: {ptp}")
    
    if ptp > 1e-9:
        print("CONFIRMED: Pole has variance -> Gradient non-zero -> Singularity.")
    else:
        print("SURPRISE: Pole is constant.")

except KeyError:
    print("Could not select latitude=90.0. Latitudes available:")
    print(ds.latitude.values)
