import xarray as xr
from evaluate_all_modes import standardize_coords
clim_ds = standardize_coords(xr.open_zarr("gs://weatherbench2/datasets/era5-hourly-climatology/1990-2017_6h_1440x721.zarr", consolidated=True))
print("clim_ds coords:", list(clim_ds.coords.keys()))
print("clim_ds dims:", list(clim_ds.dims.keys()))
