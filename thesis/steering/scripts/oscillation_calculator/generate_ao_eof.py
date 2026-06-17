import xarray as xr
import numpy as np
from eofs.xarray import Eof
import os

def main():
    output_path = '/home/ekasteleyn/aurora_thesis/thesis/steering/scripts/oscillation_calculator/indices/ao_loading_pattern.nc'
    if os.path.exists(output_path):
        print(f"File {output_path} already exists. Removing to regenerate.")
        os.remove(output_path)

    # 1. Lazily connect to the massive WB2 historical bucket
    historical_url = 'gs://weatherbench2/datasets/era5/1959-2022-6h-1440x721.zarr'
    print(f"Opening Zarr store: {historical_url}")
    ds = xr.open_zarr(historical_url, consolidated=True)

    # 2. Slice out ONLY what you need before loading into memory
    print("Slicing data from cloud (1990-2019, 1000hPa, >20N) and downloading year-by-year...")
    
    monthly_data_list = []
    for year in range(1990, 2020):
        print(f"  Downloading and processing year {year}...")
        year_data = ds['geopotential'].sel(
            level=1000, 
            latitude=slice(90, 20),
            time=slice(f'{year}-01-01', f'{year}-12-31')
        ).resample(time='1MS').mean().compute()
        monthly_data_list.append(year_data)
        
    print("Concatenating all years...")
    nh_data = xr.concat(monthly_data_list, dim='time')

    # 4. Calculate anomalies
    print("Calculating anomalies...")
    climatology = nh_data.groupby('time.month').mean('time')
    anomalies = nh_data.groupby('time.month') - climatology

    # 5. Apply Latitude Weighting
    print("Applying latitude weighting...")
    coslat = np.cos(np.deg2rad(anomalies.latitude.values))
    coslat = np.clip(coslat, 0, None) # Prevent sqrt of negative from float imprecision near 90 deg
    wgts = np.sqrt(coslat)[..., np.newaxis]

    # 6. Calculate the EOF
    print("Running PCA/EOF solver...")
    solver = Eof(anomalies, weights=wgts)
    eof1 = solver.eofs(neofs=1).squeeze()
    pc1 = solver.pcs(npcs=1).squeeze()

    # 7. Enforce polarity convention (AO is negative over the pole)
    polar_val = float(eof1.sel(latitude=90, method='nearest').mean())
    if polar_val > 0:
        print("Flipping EOF polarity to match standard convention (negative over pole).")
        eof1 = eof1 * -1
        pc1 = pc1 * -1

    # 8. Save your static ruler
    pc1_std = pc1.std(dim='time')
    ds_out = xr.Dataset({
        'eof': eof1,
        'pc_std': pc1_std
    })
    ds_out.to_netcdf(output_path)
    print(f"Successfully saved loading pattern and pc_std to {output_path}")

if __name__ == "__main__":
    main()
