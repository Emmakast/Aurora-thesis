import xarray as xr
import pandas as pd
import re
from evaluate_all_modes import standardize_coords
from calculate_3_indices import calculate_indices

filename = "base_enso_20170103_1200_alpha_0.0.nc"
target_ds = standardize_coords(xr.open_dataset("/home/ekasteleyn/aurora_thesis/thesis/results/base_enso_20170103_1200_alpha_0.0.nc"))
if 'time' not in target_ds.coords:
    match = re.search(r'(\d{8})_(\d{4})', filename)
    if match:
        from datetime import datetime
        init_time = datetime.strptime(f"{match.group(1)}{match.group(2)}", "%Y%m%d%H%M")
        target_time = pd.to_datetime(init_time) + pd.Timedelta(hours=72)
        target_ds = target_ds.expand_dims(time=[target_time])

target_ds_last = target_ds.isel(time=[-1]) if 'time' in target_ds.dims else target_ds
print("target_ds_last dims:", target_ds_last.dims)

clim_ds = standardize_coords(xr.open_zarr("gs://weatherbench2/datasets/era5-hourly-climatology/1990-2017_6h_1440x721.zarr", consolidated=True))
EOFS_DIR = "/home/ekasteleyn/aurora_thesis/thesis/steering/scripts/oscillation_calculator/indices"

res = calculate_indices(target_ds_last, clim_ds, EOFS_DIR)
