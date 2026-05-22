import xarray as xr
import numpy as np
import os

# Mock Eof output (DataArray with mode and combined_lon)
eofs = xr.DataArray(
    np.random.rand(2, 144),
    coords={'mode': [0, 1], 'combined_lon': np.arange(144)},
    dims=['mode', 'combined_lon']
)

# Apply the fix we made
eof1 = eofs.sel(mode=0).drop_vars('mode')
eof2 = eofs.sel(mode=1).drop_vars('mode')

# Mock std devs (scalars)
olr_std = xr.DataArray(1.5)
u850_std = xr.DataArray(2.5)
u200_std = xr.DataArray(3.5)

# Try merging and saving
try:
    out_ds = xr.Dataset({
        'eof1': eof1,
        'eof2': eof2,
        'olr_std': olr_std,
        'u850_std': u850_std,
        'u200_std': u200_std
    })
    out_ds.to_netcdf("test_out.nc")
    print("Success! No merge errors and file saved.")
    os.remove("test_out.nc")
except Exception as e:
    print(f"Error: {e}")
