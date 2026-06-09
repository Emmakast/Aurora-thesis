import xarray as xr
import pandas as pd
from calculate_3_indices import calculate_indices

# Open clim_ds
clim_ds = xr.open_zarr("gs://weatherbench2/datasets/era5-hourly-climatology/1990-2017_6h_1440x721.zarr", consolidated=True)
def std_coords(d):
    r = {}
    if 'latitude' in d.coords: r['latitude'] = 'lat'
    if 'longitude' in d.coords: r['longitude'] = 'lon'
    if 'valid_time' in d.coords: r['valid_time'] = 'time'
    if 'geopotential' in d.variables: r['geopotential'] = 'z'
    return d.rename(r) if r else d

clim_ds = std_coords(clim_ds)

# Open target_ds
target_ds = xr.open_dataset("/home/ekasteleyn/aurora_thesis/thesis/results/base_enso_20170103_1200_alpha_0.0.nc")
target_ds = std_coords(target_ds)
target_time = pd.to_datetime("2017-01-06 12:00:00")
target_ds = target_ds.assign_coords(time=[target_time])
target_ds = target_ds.isel(time=[-1])

lon_format = '180'
def standardize_lon(ds, fmt):
    if fmt == '180':
        return ds.assign_coords(lon=(((ds.lon + 180) % 360) - 180)).sortby('lon')
    else:
        return ds.assign_coords(lon=(ds.lon % 360)).sortby('lon')

clim_subset = standardize_lon(clim_ds, lon_format)
clim_subset = clim_subset.sel(level=500, method='nearest')

def sel_domain(ds, lat_bnds, lon_bnds):
    lat_slice = slice(lat_bnds[1], lat_bnds[0]) if ds.lat.values[0] > ds.lat.values[-1] else slice(lat_bnds[0], lat_bnds[1])
    return ds.sel(lat=lat_slice, lon=slice(lon_bnds[0], lon_bnds[1]))

clim_sliced = sel_domain(clim_subset, (90, 20), (-90, 40))
c_var = clim_sliced['z']

print("Is dayofyear in c_var.coords?", 'dayofyear' in c_var.coords)
print("c_var coords:", list(c_var.coords.keys()))

t = target_ds.time[0]
t_val = t.values
t_dt = xr.DataArray(t_val).dt

print("t_val:", t_val)
print("t_dt.dayofyear:", t_dt.dayofyear.values)

if 'dayofyear' in c_var.coords:
    c_var = c_var.sel(dayofyear=t_dt.dayofyear)
    print("Selected dayofyear! c_var dims:", c_var.dims)
else:
    print("dayofyear NOT in c_var.coords!")

if 'hour' in c_var.coords:
    c_var = c_var.sel(hour=t_dt.hour)
    print("Selected hour! c_var dims:", c_var.dims)
else:
    print("hour NOT in c_var.coords!")

