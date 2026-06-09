import xarray as xr
clim_ds = xr.open_zarr("gs://weatherbench2/datasets/era5-hourly-climatology/1990-2017_6h_1440x721.zarr", consolidated=True)
print(clim_ds.coords)
