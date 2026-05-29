import xarray as xr
from pathlib import Path
import sys
sys.path.append("/home/ekasteleyn/aurora_thesis/thesis/steering/scripts/oscillation_calculator")
from calculate_3_indices import calculate_indices
clim_ds = xr.open_zarr("gs://weatherbench2/datasets/era5-hourly-climatology/1990-2017_6h_1440x721.zarr", consolidated=True)
target = "/home/ekasteleyn/aurora_thesis/thesis/results/base_enso_20170103_1200_alpha_0.0.nc"
eofs_dir = "/home/ekasteleyn/aurora_thesis/thesis/steering/scripts/oscillation_calculator/indices"
res = calculate_indices(target, clim_ds, eofs_dir)
print("NAO dims:", res['NAO'].dims)
print("NAO shape:", res['NAO'].shape)
