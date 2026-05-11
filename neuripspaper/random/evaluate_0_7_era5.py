import xarray as xr
import gcsfs

fs = gcsfs.GCSFileSystem(project="weatherbench2")
mapper = fs.get_mapper("weatherbench2/datasets/era5/1959-2022-6h-512x256_equiangular_conservative.zarr")
ds = xr.open_zarr(mapper, consolidated=True)[['geopotential', 'temperature', 'u_component_of_wind', 'v_component_of_wind']].isel(time=0)

print(ds.data_vars)

import sys
sys.path.append("/home/ekasteleyn/aurora_thesis/neuripspaper/scripts")
from run_all_metrics import compute_hydrostatic_imbalance, get_grid_cell_area, _detect_level_dim, compute_geostrophic_imbalance

ds = ds.load()
ld = _detect_level_dim(ds)
area = get_grid_cell_area(ds)
print("Hydrostatic:", compute_hydrostatic_imbalance(ds, area, level_dim=ld))
print("Geostrophic:", compute_geostrophic_imbalance(ds, area, level_dim=ld))

