import xarray as xr
def std_coords(d):
    r = {}
    if 'latitude' in d.coords: r['latitude'] = 'lat'
    if 'longitude' in d.coords: r['longitude'] = 'lon'
    if 'valid_time' in d.coords: r['valid_time'] = 'time'
    if 'geopotential' in d.variables: r['geopotential'] = 'z'
    return d.rename(r) if r else d

eof_ds = std_coords(xr.open_dataset("/home/ekasteleyn/aurora_thesis/thesis/steering/scripts/oscillation_calculator/indices/nao_loading_pattern.nc"))
print("EOF dims:", eof_ds['eof'].dims)

clim_ds = std_coords(xr.open_zarr("gs://weatherbench2/datasets/era5-hourly-climatology/1990-2017_6h_1440x721.zarr", consolidated=True))
print("Clim dims:", clim_ds['z'].dims)
