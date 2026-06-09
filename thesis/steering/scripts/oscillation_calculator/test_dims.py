import xarray as xr
ds = xr.open_dataset("/home/ekasteleyn/aurora_thesis/thesis/results/base_enso_20170103_1200_alpha_0.0.nc")
print(ds.dims)
print(ds['z'].shape)
