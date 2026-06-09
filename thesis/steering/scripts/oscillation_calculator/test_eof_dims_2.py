import xarray as xr
eof_ds = xr.open_dataset("/home/ekasteleyn/aurora_thesis/thesis/steering/scripts/oscillation_calculator/indices/nao_loading_pattern.nc")
print("pc_std dims:", eof_ds['pc_std'].dims)
print("pc_std shape:", eof_ds['pc_std'].shape)
print("pc_std values:\n", eof_ds['pc_std'].values)
