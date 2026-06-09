import time
import xarray as xr
import pandas as pd

import sys
sys.path.append('.')

from evaluate_all_modes import standardize_coords, calculate_ao, calculate_enso_inline, slice_nino34
from calculate_mjo import calculate_mjo
from pathlib import Path

def test():
    print("Loading climatology...")
    t0 = time.time()
    clim_ds = standardize_coords(xr.open_zarr("gs://weatherbench2/datasets/era5-hourly-climatology/1990-2017_6h_1440x721.zarr", consolidated=True))
    clim_enso = slice_nino34(clim_ds)['2m_temperature']
    print(f"Loaded climatology in {time.time() - t0:.2f}s")
    
    print("Loading AO pattern...")
    ao_ds = standardize_coords(xr.open_dataset("/home/ekasteleyn/aurora_thesis/thesis/steering/scripts/oscillation_calculator/indices/ao_loading_pattern.nc"))
    ao_pattern = ao_ds['eof'].squeeze()
    ao_std = float(ao_ds['pc_std'].values)
    
    print("Loading MJO EOF...")
    mjo_eof = xr.open_dataset("/home/ekasteleyn/aurora_thesis/thesis/steering/scripts/oscillation_calculator/indices/mjo_loading_pattern.nc")
    
    print("Loading target dataset...")
    target_path = "/home/ekasteleyn/aurora_thesis/thesis/results/base_enso_20170103_1200_alpha_0.0.nc"
    target_ds = standardize_coords(xr.open_dataset(target_path))
    
    # ensure time is assigned if needed
    if 'time' not in target_ds.coords:
        target_ds = target_ds.assign_coords(time=[pd.Timestamp("2017-01-06 12:00:00")])
    
    print("Running AO calculation...")
    t0 = time.time()
    val_ao = calculate_ao(target_ds, clim_ds, ao_pattern, ao_std)
    print(f"AO calculation finished in {time.time() - t0:.2f}s, value: {val_ao}")
    
    print("Running MJO calculation...")
    t0 = time.time()
    mjo_res = calculate_mjo(target_ds, clim_ds, mjo_eof)
    print(f"MJO calculation finished in {time.time() - t0:.2f}s, amplitude: {float(mjo_res['amplitude'].values[-1])}")
    
    print("Running ENSO calculation...")
    t0 = time.time()
    val_enso = calculate_enso_inline(target_ds, clim_enso)
    print(f"ENSO calculation finished in {time.time() - t0:.2f}s, value: {val_enso}")

if __name__ == '__main__':
    test()
