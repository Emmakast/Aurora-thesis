import xarray as xr
import pandas as pd
import re
from evaluate_all_modes import standardize_coords
filename = "base_enso_20170103_1200_alpha_0.0.nc"
target_ds = standardize_coords(xr.open_dataset("/home/ekasteleyn/aurora_thesis/thesis/results/base_enso_20170103_1200_alpha_0.0.nc"))
if 'time' not in target_ds.coords:
    match = re.search(r'(\d{8})_(\d{4})', filename)
    if match:
        from datetime import datetime
        init_time = datetime.strptime(f"{match.group(1)}{match.group(2)}", "%Y%m%d%H%M")
        target_time = pd.to_datetime(init_time) + pd.Timedelta(hours=72)
        target_ds = target_ds.assign_coords(time=[target_time])
print("target_ds.dims:", target_ds.dims)
print("'time' in target_ds.dims:", 'time' in target_ds.dims)

target_ds_last_1 = target_ds.isel(time=-1) if 'time' in target_ds.dims else target_ds
print("target_ds_last_1 dims:", target_ds_last_1.dims)

target_ds_last_2 = target_ds.isel(time=[-1]) if 'time' in target_ds.dims else target_ds
print("target_ds_last_2 dims:", target_ds_last_2.dims)
