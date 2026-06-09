import xarray as xr
ds = xr.open_dataset("/home/ekasteleyn/aurora_thesis/thesis/results/base_enso_20170103_1200_alpha_0.0.nc")
lat = ds.latitude.values if 'latitude' in ds.coords else ds.lat.values
print("lat shape:", lat.shape)
print("lat first 5:", lat[:5])
print("lat last 5:", lat[-5:])
import numpy as np
diff = np.diff(lat)
print("diff min/max:", diff.min(), diff.max())
